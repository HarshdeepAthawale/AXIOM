"""The optional local query LLM, and the wall that keeps it away from code.

Qwen2.5-1.5B-Instruct Q4_K_M GGUF through ``llama-cpp-python``, ``temperature=0``,
a fixed seed and a fixed ``n_ctx``, output capped at ``AXIOM_LLM_MAX_TOKENS``
(TechSpecifications.md section 3.3). ``llama_cpp`` is an optional dependency and
is imported inside :meth:`QueryLLM._load`, never at module scope, so
``import axiom.agent.llm`` succeeds on a bare install (NFR-07).

**ADR-008 -- the query LLM never reads code -- is enforced here by construction,
not by convention.** The class exposes exactly four methods, matching the four
permitted jobs, and none of them has a parameter that can carry chunk text:

===================  ===========================================================
:meth:`QueryLLM.classify`            ``(query: str)``
:meth:`QueryLLM.expand`              ``(query: str, identifiers: Sequence[str])``
:meth:`QueryLLM.decompose`           ``(query: str, max_sub_queries: int)``
:meth:`QueryLLM.judge_sufficiency`   ``(top1: float, above_floor: int, ...)`` -- numbers only
===================  ===========================================================

There is no ``rerank``, no ``summarise``, no ``results`` argument anywhere. The
sufficiency judge -- the one job where a careless implementation would be tempted
to show the model its candidates -- receives *scalars*: it cannot see a chunk
because there is no parameter a chunk fits into. :func:`_guard_query_text` backs
that up at runtime for dynamically-typed callers.

Every method degrades to ``None``/``[]`` rather than raising, so a caller's
heuristic path takes over silently at the call site (Rules.md Rule 3).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axiom.config import Settings
from axiom.core.errors import AxiomContractError
from axiom.core.logging import get_logger, log_degradation
from axiom.schema.enums import QueryType

_LOG = get_logger("agent.llm")

#: Longest string any of the four jobs will send to the model.
#:
#: This is the ADR-008 wall made numeric. A query is at most
#: ``planner.MAX_QUERY_CHARS``; a chunk is 64-512 tokens of source, which is
#: routinely longer. The cap is enforced on every text parameter, so smuggling a
#: function body in through the ``query`` argument fails loudly instead of
#: quietly turning the agent into an LLM reranker.
LLM_MAX_INPUT_CHARS = 2048

#: ``n_ctx`` for llama.cpp.
#:
#: Setup.md section 7.3 documents this as ``AXIOM_LLM_CONTEXT=2048``, but the
#: frozen :class:`~axiom.config.Settings` has no such field and config.py is not
#: ours to extend, so the documented default is pinned here. Fixed, not adaptive:
#: Rules.md section 5 item 4 requires a fixed ``n_ctx`` for determinism.
DEFAULT_LLM_CONTEXT = 2048

#: Where ``scripts/fetch_models.sh`` puts the GGUF (Setup.md section 7.3).
DEFAULT_GGUF_DIR = Path("data/models/gguf")

#: Upper bound on a generated sub-query, in characters. Anything longer is the
#: model narrating instead of answering, and is discarded.
_MAX_SUB_QUERY_CHARS = 200

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{1,31}$")
_LIST_SPLIT_RE = re.compile(r"[,\n;]")
_LEADING_BULLET_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def _guard_query_text(value: object, *, field: str) -> str:
    """Refuse anything that is not short, single-line, user-authored query text.

    The signatures already make chunk text unrepresentable; this catches the
    dynamically-typed caller that gets clever. Passing a schema object is a
    violated internal contract, not bad user input, so it raises
    (:class:`~axiom.core.errors.AxiomContractError`) -- which is the one thing
    Rules.md Rule 3 does permit an exception for. Over-long *text* merely
    truncates, because that can legitimately happen with a pathological query.
    """
    if not isinstance(value, str):
        raise AxiomContractError(
            f"{field} must be query text; got {type(value).__name__}. The query LLM "
            f"never receives chunk text or schema objects (ADR-008)."
        )
    collapsed = " ".join(value.split())
    if len(collapsed) > LLM_MAX_INPUT_CHARS:
        log_degradation(
            _LOG,
            "agent.llm",
            f"{field} is {len(collapsed)} chars, over the {LLM_MAX_INPUT_CHARS}-char wall",
            f"truncated to {LLM_MAX_INPUT_CHARS} chars",
        )
        collapsed = collapsed[:LLM_MAX_INPUT_CHARS]
    return collapsed


def gguf_model_dir() -> Path:
    """The GGUF directory under the resolved model directory (``AXIOM_MODEL_DIR``)."""
    from axiom.config import resolve_model_dir

    return resolve_model_dir() / "gguf"


def resolve_model_path(settings: Settings) -> Path | None:
    """Locate the GGUF weights named by ``settings.llm_model``.

    ``Settings.llm_model`` holds a *filename* (``Qwen2.5-1.5B-Instruct-Q4_K_M.gguf``)
    while Setup.md documents a full path under ``AXIOM_LLM_MODEL_PATH`` -- a field
    the frozen Settings does not have. Both readings are honoured: an existing
    path is used as given, otherwise the name is looked for in the documented
    model directory, case-insensitively (the Hugging Face artifact is lowercase,
    the Settings default is not).

    Returns ``None`` when the weights are absent, which is the normal state on a
    machine that never ran the model fetch.
    """
    named = Path(settings.llm_model)
    if named.is_file():
        return named
    gguf_dir = gguf_model_dir()
    direct = gguf_dir / named.name
    if direct.is_file():
        return direct
    if gguf_dir.is_dir():
        wanted = named.name.lower()
        for candidate in sorted(gguf_dir.glob("*.gguf")):
            if candidate.name.lower() == wanted:
                return candidate
    return None


@dataclass(frozen=True)
class _HandleKey:
    """Everything that can change a completion, and therefore the cache key."""

    model_path: str
    n_ctx: int
    seed: int
    n_threads: int
    max_tokens: int


#: Loaded ``llama_cpp.Llama`` handles, keyed by :class:`_HandleKey`.
#:
#: Not a result cache -- Rules.md Rule 2 forbids those. It caches the *model
#: handle*, which is provably result-neutral at ``temperature=0`` with a fixed
#: seed and ``n_ctx``: reloading the same weights with the same key would produce
#: the same completions, only 20-40 s slower per query.
_HANDLES: dict[_HandleKey, Any] = {}

#: Adapters already constructed for a given key, so a loaded handle is shared
#: across queries instead of being rebuilt per call.
_ADAPTERS: dict[_HandleKey, QueryLLM] = {}

#: Model names whose "weights not found" degradation has already been logged.
#: The degradation is real and must be loud (Rules.md Rule 3), but on a 3,765-
#: query eval run it is the same fact 3,765 times, so it is stated once.
_MISSING_LOGGED: set[str] = set()


class QueryLLM:
    """Adapter over the local GGUF model, exposing exactly four jobs.

    Construct through :func:`get_llm`, which applies the ``llm_enabled`` gate and
    reuses the loaded handle. Every method returns a "no opinion" value on any
    failure -- ``None`` or an empty list -- so callers fall through to their
    heuristic path without a try/except at each call site.
    """

    def __init__(self, settings: Settings, model_path: Path | None = None) -> None:
        self._settings = settings
        self._model_path = model_path if model_path is not None else resolve_model_path(settings)
        self._max_tokens = min(settings.llm_max_tokens, 512)
        self._failed = False

    @property
    def model_path(self) -> Path | None:
        """The resolved GGUF path, or ``None`` when the weights are absent."""
        return self._model_path

    @property
    def available(self) -> bool:
        """Whether a completion could plausibly be produced.

        Cheap: it checks the weights exist and that a previous load did not
        already fail. It deliberately does not import ``llama_cpp`` -- that would
        put a heavy import on the path of every caller that only wants to know
        whether to bother.
        """
        return self._model_path is not None and not self._failed

    # --- The four permitted jobs -----------------------------------------

    def classify(self, query: str) -> QueryType | None:
        """Job 1: classify. Returns ``None`` when the output is not one enum value.

        Ambiguity is reported as "no opinion" rather than guessed at, so the
        caller's heuristic engine decides instead -- the LLM is allowed to
        override the heuristic only when it is unambiguous.
        """
        text = _guard_query_text(query, field="query")
        prompt = (
            "You label developer code-search queries. Reply with exactly one word.\n"
            "semantic = asks how behaviour works, no symbol names known\n"
            "structural = asks about calls, imports, exports, ordering between code units\n"
            "usage = asks where a named symbol or literal is used\n"
            "hybrid = mixed intent or unclear\n\n"
            f"Query: {text}\n"
            "Label:"
        )
        completion = self._complete(prompt, max_tokens=8)
        if completion is None:
            return None
        lowered = completion.lower()
        found = [
            (lowered.find(member.value), member) for member in QueryType if member.value in lowered
        ]
        if len(found) != 1:
            return None
        return found[0][1]

    def expand(self, query: str, identifiers: Sequence[str]) -> list[str]:
        """Job 2: expand. Supplements the curated map; never replaces it.

        ``identifiers`` are the symbols already lifted from the query, passed so
        the model does not waste its budget proposing terms the planner has
        already got. They are query-derived tokens, not corpus content.
        """
        text = _guard_query_text(query, field="query")
        known = ", ".join(_guard_query_text(i, field="identifier") for i in identifiers[:10])
        prompt = (
            "Suggest code identifiers a developer might have used for this concept.\n"
            "Reply with a comma-separated list of single words. No sentences.\n"
            f"Already known, do not repeat: {known or 'none'}\n\n"
            f"Query: {text}\n"
            "Words:"
        )
        completion = self._complete(prompt, max_tokens=64)
        if completion is None:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for raw in _LIST_SPLIT_RE.split(completion):
            token = raw.strip().strip("`'\"").lower()
            if _IDENT_RE.match(token) and token not in seen:
                seen.add(token)
                out.append(token)
        return out

    def decompose(self, query: str, max_sub_queries: int) -> list[str]:
        """Job 3: decompose. Returns ``[]`` when the query is not compound.

        A single returned line is treated as "not decomposable" by the caller --
        one sub-query identical in spirit to the original just buys a redundant
        fan-out.
        """
        text = _guard_query_text(query, field="query")
        limit = max(1, max_sub_queries)
        prompt = (
            f"Split this code-search query into at most {limit} independent search "
            "queries, one per line. If it cannot be usefully split, reply NONE.\n\n"
            f"Query: {text}\n"
            "Queries:"
        )
        completion = self._complete(prompt, max_tokens=128)
        if completion is None:
            return []
        out: list[str] = []
        for line in completion.splitlines():
            cleaned = " ".join(_LEADING_BULLET_RE.sub("", line).split())
            if not cleaned or cleaned.upper().startswith("NONE"):
                continue
            if len(cleaned) > _MAX_SUB_QUERY_CHARS or cleaned in out:
                continue
            out.append(cleaned)
            if len(out) >= limit:
                break
        return out

    def judge_sufficiency(
        self,
        top1_score: float,
        results_above_floor: int,
        top1_threshold: float,
        floor: float,
        min_results: int,
    ) -> bool | None:
        """Job 4: judge sufficiency -- **from scores alone** (ADR-008).

        Note the signature: five numbers. There is no candidate list, no chunk,
        no text. That is the structural guarantee that the "agentic" part of the
        system cannot become an LLM reranker in disguise, which is exactly what
        NG-23 forecloses.

        In practice the arithmetic predicate in ``agent.evaluator`` is the one
        that runs; this exists so the LLM path has all four of its declared jobs
        and can be swept against the heuristic during threshold tuning (T-141).

        Returns ``True``/``False``, or ``None`` when the model has no clear
        answer and the caller should use the arithmetic predicate.
        """
        prompt = (
            "A code search returned candidates. Decide if they are good enough.\n"
            f"Top score: {top1_score:.4f} (needs >= {top1_threshold:.4f})\n"
            f"Results above {floor:.4f}: {results_above_floor} (needs >= {min_results})\n"
            "Reply YES if good enough, NO if the search should be retried.\n"
            "Answer:"
        )
        completion = self._complete(prompt, max_tokens=4)
        if completion is None:
            return None
        lowered = completion.strip().lower()
        if lowered.startswith("yes"):
            return True
        if lowered.startswith("no"):
            return False
        return None

    # --- Inference ---------------------------------------------------------

    def _handle_key(self) -> _HandleKey | None:
        if self._model_path is None:
            return None
        return _HandleKey(
            model_path=str(self._model_path),
            n_ctx=DEFAULT_LLM_CONTEXT,
            seed=self._settings.seed,
            n_threads=self._settings.num_threads,
            max_tokens=self._max_tokens,
        )

    def _load(self) -> Any | None:
        """Import and construct ``llama_cpp.Llama``; ``None`` on any failure.

        The import sits here rather than at module scope because
        ``llama-cpp-python`` is in the optional ``agent`` extra: on an evaluator's
        machine with only the base dependencies installed, importing it at module
        scope would take the whole ``axiom.agent`` package down with it (NFR-07).
        """
        key = self._handle_key()
        if key is None:
            return None
        cached = _HANDLES.get(key)
        if cached is not None:
            return cached
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            self._failed = True
            log_degradation(
                _LOG, "agent.llm", f"llama_cpp not installed ({exc})", "heuristic rule engine"
            )
            return None
        try:
            handle = Llama(
                model_path=key.model_path,
                n_ctx=key.n_ctx,
                seed=key.seed,
                n_threads=key.n_threads or None,
                logits_all=False,
                verbose=False,
            )
        except Exception as exc:
            self._failed = True
            log_degradation(_LOG, "agent.llm", f"GGUF load failed: {exc}", "heuristic rule engine")
            return None
        _HANDLES[key] = handle
        _LOG.info(
            "query llm loaded",
            extra={"axiom_extra": {"stage": "agent", "model": key.model_path, "n_ctx": key.n_ctx}},
        )
        return handle

    def _complete(self, prompt: str, max_tokens: int) -> str | None:
        """One greedy completion, or ``None``.

        ``temperature=0.0`` and the seeded handle make this reproducible, which
        is what lets an eval run claim determinism with the LLM enabled (NFR-08).
        """
        if not self.available:
            return None
        handle = self._load()
        if handle is None:
            return None
        try:
            response = handle(
                prompt,
                max_tokens=min(max_tokens, self._max_tokens),
                temperature=0.0,
                top_k=1,
                seed=self._settings.seed,
                stop=["\n\n"],
                echo=False,
            )
            return str(response["choices"][0]["text"])
        except Exception as exc:
            log_degradation(_LOG, "agent.llm", f"inference failed: {exc}", "heuristic rule engine")
            return None


def get_llm(settings: Settings) -> QueryLLM | None:
    """The adapter for these settings, or ``None`` when the LLM is not in play.

    ``None`` means "use the heuristic path" and covers three cases that callers
    do not need to distinguish: ``llm_enabled=false`` (a selected mode, not a
    failure -- Appflow.md Flow 4), the weights not being on disk, and a previous
    load having failed in this process.
    """
    if not settings.llm_enabled:
        return None
    probe = QueryLLM(settings)
    key = probe._handle_key()
    if key is None:
        if settings.llm_model not in _MISSING_LOGGED:
            _MISSING_LOGGED.add(settings.llm_model)
            log_degradation(
                _LOG,
                "agent.llm",
                f"GGUF weights for {settings.llm_model!r} not found under {gguf_model_dir()}",
                "heuristic rule engine",
            )
        return None
    return _ADAPTERS.setdefault(key, probe)
