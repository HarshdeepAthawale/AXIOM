"""Stage 4: cross-encoder reranking of the fused candidate set.

This is the single most expensive stage in the pipeline -- 620 ms of the 768 ms
measured p50 ([TechSpecifications.md] section 8) -- because it runs one forward
pass per ``(query, candidate)`` pair instead of one pass total. Everything in
this module is therefore organised around two facts: the model must never be
loaded until a query actually needs it (Rules.md AP-06), and the stage must be
able to fail without taking the query down with it (Rules.md Rule 3).

The degradation ladder, quoted from Rules.md section 3::

    cross-encode
      -> truncate to the model window and cross-encode
      -> pass the RRF order through unchanged (rerank_score=None)

with three refinements this module adds:

* the *fallback checkpoint* rung (``cross-encoder/ms-marco-MiniLM-L-6-v2``) sits
  between "primary model" and "passthrough", per TechSpecifications.md section 3.2;
* char-window truncation is applied unconditionally on the way in rather than
  only after a failure, because a 20k-char chunk would otherwise blow the token
  budget on the first attempt and burn the latency budget discovering it
  (TestPlan.md TC-063);
* an opt-in, dependency-free *lexical* rung can be inserted just above
  passthrough so that reranking degrades rather than vanishing on a machine with
  no optional dependencies. It is **off by default** because TestPlan.md TC-062
  (P0) requires a raising cross-encoder to produce pure RRF order with every
  ``rerank_score is None``.

THE CONTRACT FOR THE AGENT LOOP AUTHOR
--------------------------------------
TechSpecifications.md section 5.3 requires ``agent/evaluator.py`` to apply its two
thresholds against ``rerank_score`` when the real model ran and against
``rrf_score`` when reranking degraded. "Did the real model run" is *not*
recoverable from the result list alone once the lexical rung is enabled -- a
lexical score is a populated ``rerank_score`` that the 0.35/0.20 thresholds were
never calibrated for. So:

* call :func:`rerank_detailed`, not :func:`rerank`, when you need to know;
* read :attr:`RerankOutcome.score_field` -- it is literally the attribute name to
  threshold against, ``"rerank_score"`` or ``"rrf_score"``;
* :attr:`RerankOutcome.mode` additionally tells the formatter whether to stamp
  ``match_reason="rerank_passthrough"`` (Appflow.md Flow 8, Rules.md Rule 3).

:func:`rerank` exists because FR-12 and Design.md section 4 describe the stage as
``list[FusedResult] -> list[FusedResult]``; it is a thin wrapper that discards the
report, and is safe only while the lexical rung stays disabled.
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from axiom.core.logging import get_logger, log_degradation
from axiom.core.timing import Deadline, StageTiming, TimingLedger
from axiom.schema.chunk import Chunk
from axiom.schema.retrieval import FusedResult

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from axiom.config import Settings

_LOGGER = get_logger("rerank.cross_encoder")

#: Stage tag for every log record and timing entry from this module (Rules.md section 9.1).
STAGE = "rerank"

#: Component name passed to :func:`log_degradation`, matching the ladder table key.
COMPONENT = "axiom.rerank.cross_encoder:rerank"

#: Checkpoint used when the primary reranker cannot be loaded (TechSpecifications.md section 3.2).
FALLBACK_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Directory ONNX artifacts are exported into by Setup.md section 6.1.
ONNX_MODEL_ROOT = Path("data/models/onnx")

# --- Defaults for tunables the frozen Settings does not (yet) carry ----------
# TechSpecifications.md section 4.7 and TestPlan.md TC-063 both name
# ``AXIOM_RERANK_MAX_CHARS`` / ``settings.rerank_max_chars``, but ``Settings`` has
# no such field and is frozen. :func:`_tunable` resolves these in the documented
# precedence order -- Settings field, then AXIOM_ env var, then the default here --
# so adding the field to Settings later silently starts winning, with no edit
# needed at this callsite (Rules.md AP-07's intent, if not yet its letter).

#: Document-side character window. 4096 chars is ~1000-1300 tokens, which keeps the
#: 512-token upper chunk bound comfortably whole (TechSpecifications.md section 5.4).
DEFAULT_RERANK_MAX_CHARS = 4096

#: Token-side window. Sized above the 4096-char budget's ~1300-token worst case so
#: char truncation, not token truncation, is the binding constraint TC-063 observes.
DEFAULT_RERANK_MAX_TOKENS = 1536

#: Pairs per forward pass. Scoring is independent per pair, so this trades peak RSS
#: against call overhead and never changes results (TestPlan.md TC-061).
DEFAULT_RERANK_BATCH_SIZE = 8

#: Saturation constant for the lexical rung's term-frequency term, BM25's k1 by analogy.
DEFAULT_LEXICAL_K1 = 1.2

#: Fraction of the remaining headroom awarded when the whole query appears verbatim.
DEFAULT_LEXICAL_PHRASE_BONUS = 0.35

#: Half-width of the logit range the lexical rung maps its [0,1] similarity onto, so
#: that the shared sigmoid spreads it across roughly (0.002, 0.998) instead of (0.5, 0.73).
DEFAULT_LEXICAL_LOGIT_SPAN = 6.0

#: Splits identifiers into retrievable words: camelCase, snake_case, digits, punctuation.
_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class RerankMode(StrEnum):
    """Which rung of the degradation ladder actually produced the returned order."""

    CROSS_ENCODER = "cross_encoder"
    """A real cross-encoder scored every pair. ``rerank_score`` is calibrated."""

    LEXICAL = "lexical"
    """The dependency-free scorer ran. ``rerank_score`` is populated but NOT calibrated."""

    PASSTHROUGH = "passthrough"
    """RRF order unchanged, every ``rerank_score is None``. A first-class outcome."""

    DISABLED = "disabled"
    """``reranker_enabled=false``: the FR-12 ablation switch, not a degradation."""


@dataclass(frozen=True)
class RerankOutcome:
    """The reranked list plus everything a caller needs to interpret it.

    Separated from the bare list because the *meaning* of the final score changes
    with the rung taken (Design.md section 6.2), and a caller that cannot see which
    rung ran will silently threshold an RRF score against a rerank threshold.
    """

    results: list[FusedResult]
    mode: RerankMode
    model_name: str | None = None
    pairs_scored: int = 0
    batch_size: int = 0
    truncated_documents: int = 0
    elapsed_ms: float = 0.0
    degraded_reason: str | None = None

    @property
    def model_ran(self) -> bool:
        """True only when a genuine cross-encoder produced the scores.

        The lexical rung deliberately reports ``False``: its scores order results
        usefully but were never calibrated against the 0.35/0.20 sufficiency
        thresholds, so the agent loop must not threshold them.
        """
        return self.mode is RerankMode.CROSS_ENCODER

    @property
    def score_field(self) -> str:
        """Name of the :class:`FusedResult` attribute the sufficiency predicate must read.

        ``"rerank_score"`` when the model ran, ``"rrf_score"`` otherwise -- exactly
        the switch TechSpecifications.md section 5.3 describes, reduced to a string the
        evaluator can ``getattr`` with.
        """
        return "rerank_score" if self.model_ran else "rrf_score"

    @property
    def degraded(self) -> bool:
        """True when the primary path did not run, ablation excluded."""
        return self.mode in (RerankMode.LEXICAL, RerankMode.PASSTHROUGH)

    def top_score(self) -> float | None:
        """Top-1 value of :attr:`score_field`, or ``None`` on an empty result set."""
        if not self.results:
            return None
        value = getattr(self.results[0], self.score_field)
        return None if value is None else float(value)


@runtime_checkable
class PairScorer(Protocol):
    """Anything that can turn ``(query, document)`` pairs into relevance logits.

    Injectable by design: TestPlan.md TC-059..TC-061 drive this stage with a
    ``FakeCrossEncoder`` and count the pair scorings, which is only possible if the
    model is a parameter rather than a module-level global.

    Implementations return **raw logits**, not probabilities -- normalisation to
    ``[0, 1]`` is this module's job, so every rung shares one calibration path and
    a fake that returns a known score table still produces a monotone ordering.
    """

    name: str
    kind: str

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Score one batch of pairs, returning one logit per pair in input order."""
        ...


def _tunable(settings: Settings | None, field_name: str, default: Any) -> Any:
    """Resolve a tunable that may not exist as a :class:`Settings` field yet.

    Precedence matches Rules.md section 7 minus the CLI rung (which reaches us as an
    explicit argument): ``Settings`` field, then ``AXIOM_<FIELD>``, then ``default``.
    Needed because TechSpecifications.md section 4.7, Setup.md section 7.2 and
    TestPlan.md TC-063 all reference ``AXIOM_RERANK_MAX_CHARS`` and friends while
    the frozen ``Settings`` defines none of them.
    """
    if settings is not None:
        value = getattr(settings, field_name, None)
        if value is not None:
            return value
    raw = os.environ.get(f"AXIOM_{field_name.upper()}")
    if raw is None:
        return default
    try:
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, Path):
            return Path(raw)
    except ValueError:
        _LOGGER.warning(
            "ignoring malformed AXIOM_%s=%r, using %r",
            field_name.upper(),
            raw,
            default,
            extra={"axiom_extra": {"stage": STAGE}},
        )
        return default
    return raw


def _sigmoid(logit: float) -> float:
    """Logistic squash, clamped into ``[0, 1]``.

    ``FusedResult.rerank_score`` carries ``ge=0.0, le=1.0``, and a float that lands
    at 1.0000000000000002 through rounding would raise a validation error in the
    middle of a successful query. The clamp is cheap insurance, not sloppiness.
    """
    if logit >= 0.0:
        value = 1.0 / (1.0 + math.exp(-min(logit, 60.0)))
    else:
        exp_l = math.exp(max(logit, -60.0))
        value = exp_l / (1.0 + exp_l)
    return min(1.0, max(0.0, value))


def truncate_document(text: str, max_chars: int) -> tuple[str, bool]:
    """Clip the *document* side of a pair to the model's character window.

    Returns the possibly-clipped text and whether clipping happened.

    Head-truncation is deliberate: a code chunk leads with its signature, its
    JSDoc and its first statements, which is where the retrieval signal lives. The
    query is **never** truncated (TestPlan.md TC-063) -- queries are short, and
    silently dropping half a query would corrupt every score in the batch rather
    than degrade one of them.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _tokenise(text: str) -> list[str]:
    """Lowercased code-aware word split, shared by the lexical rung's two sides."""
    return [token.lower() for token in _TOKEN_SPLIT.split(text) if token]


class LexicalPairScorer:
    """Deterministic, dependency-free pair scorer: the rung above pure passthrough.

    Not in the spec's ladder. It exists so a demo on a machine with no optional
    dependencies still *reorders* rather than handing back raw RRF, while staying
    honest about it -- :attr:`kind` is ``"lexical"``, which drives
    :attr:`RerankOutcome.model_ran` to ``False`` and switches the sufficiency
    predicate onto ``rrf_score``.

    The score is a saturating term-coverage measure over code-aware tokens, plus a
    verbatim-phrase bonus. No IDF: this rung sees 25 documents, which is far too
    few for a corpus statistic to mean anything, and inventing one would make the
    score depend on the candidate set rather than on the pair.
    """

    kind = "lexical"

    def __init__(
        self,
        k1: float = DEFAULT_LEXICAL_K1,
        phrase_bonus: float = DEFAULT_LEXICAL_PHRASE_BONUS,
        logit_span: float = DEFAULT_LEXICAL_LOGIT_SPAN,
    ) -> None:
        self.name = "lexical-overlap"
        self._k1 = k1
        self._phrase_bonus = phrase_bonus
        self._logit_span = logit_span

    def similarity(self, query: str, document: str) -> float:
        """Coverage of the query's distinct tokens by the document, in ``[0, 1]``."""
        query_tokens = list(dict.fromkeys(_tokenise(query)))
        if not query_tokens:
            return 0.0
        counts = Counter(_tokenise(document))
        total = 0.0
        for token in query_tokens:
            freq = counts.get(token, 0)
            if freq:
                total += freq / (freq + self._k1)
        similarity = total / len(query_tokens)
        needle = " ".join(_tokenise(query))
        if needle and needle in " ".join(_tokenise(document)):
            similarity += (1.0 - similarity) * self._phrase_bonus
        return min(1.0, max(0.0, similarity))

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Map each pair's similarity onto a logit the shared sigmoid can spread out."""
        return [
            self._logit_span * (2.0 * self.similarity(query, document) - 1.0)
            for query, document in pairs
        ]


@dataclass
class OnnxCrossEncoder:
    """Lazy ONNX Runtime cross-encoder session (Rules.md AP-06's sanctioned shape).

    Neither ``onnxruntime`` nor ``transformers`` is imported until :meth:`warm` or
    :meth:`score_pairs` runs, so ``import axiom.rerank`` stays free on a bare
    install (NFR-07). Construction of this object is likewise free -- it is the
    *session*, not the class, that costs 900 ms and 1.2 GB.
    """

    model_name: str
    onnx_path: Path
    max_tokens: int = DEFAULT_RERANK_MAX_TOKENS
    num_threads: int = 0
    offline: bool = False
    kind: str = field(default="cross_encoder", init=False)
    _session: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _input_names: frozenset[str] = field(default=frozenset(), init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _load_error: Exception | None = field(default=None, init=False, repr=False)
    _failure_reported: bool = field(default=False, init=False, repr=False)

    @property
    def name(self) -> str:
        return self.model_name

    def warm(self) -> None:
        """Construct the session and tokenizer eagerly.

        The ``/warm`` endpoint (API.md section on warm-up) and the loader both call
        this so that a model-load failure surfaces *before* the batch loop, where it
        can still be logged once and degraded cleanly rather than mid-batch.

        A load failure is remembered and re-raised on every later call. Rediscovering
        a missing ``onnxruntime`` on all 8,765 eval queries would spend real time
        proving the same negative, and the answer cannot change inside one process.
        """
        if self._load_error is not None:
            raise self._load_error
        try:
            self._ensure_session()
            self._ensure_tokenizer()
        except Exception as exc:
            self._load_error = exc
            raise

    def claim_failure_report(self) -> bool:
        """True the first time a caller asks to log this encoder's load failure.

        Rules.md section 3 wants every degradation loud, and Rules.md section 9.1
        forbids flooding INFO/WARNING per query. Both hold: the *ladder walk* is
        reported once here, while :func:`rerank_detailed` still emits a per-query
        degradation WARNING naming the rung it actually took.
        """
        if self._failure_reported:
            return False
        self._failure_reported = True
        return True

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:
                return self._session
            import onnxruntime as ort

            if not self.onnx_path.is_file():
                raise FileNotFoundError(f"reranker ONNX artifact missing at {self.onnx_path}")
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if self.num_threads > 0:
                options.intra_op_num_threads = self.num_threads
                options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = frozenset(i.name for i in session.get_inputs())
            self._session = session
        return self._session

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._lock:
            if self._tokenizer is not None:
                return self._tokenizer
            from transformers import AutoTokenizer

            # The exported artifact directory carries the matching tokenizer; prefer it
            # over the hub id so an offline box never reaches for the network (RISK-11).
            sources: list[str] = []
            if self.onnx_path.parent.is_dir():
                sources.append(str(self.onnx_path.parent))
            if not self.offline:
                sources.append(self.model_name)
            last_error: Exception | None = None
            for source in sources:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(source)
                    break
                except Exception as exc:
                    last_error = exc
            if self._tokenizer is None:
                where = sources or ["<offline, no local artifact dir>"]
                raise RuntimeError(f"no tokenizer for {self.model_name} in {where}") from last_error
        return self._tokenizer

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """One batched forward pass over the pairs, returning one logit each."""
        if not pairs:
            return []
        import numpy as np

        session = self._ensure_session()
        tokenizer = self._ensure_tokenizer()
        encoded = tokenizer(
            [query for query, _ in pairs],
            [document for _, document in pairs],
            padding=True,
            truncation="only_second",
            max_length=self.max_tokens,
            return_tensors="np",
        )
        feed = {name: encoded[name] for name in self._input_names if name in encoded}
        if not feed:
            raise RuntimeError(f"tokenizer produced none of the session inputs {self._input_names}")
        logits = np.asarray(session.run(None, feed)[0], dtype="float64")
        return _logits_to_scores(logits, len(pairs))


def _logits_to_scores(logits: Any, expected: int) -> list[float]:
    """Collapse a classifier head of unknown arity into one logit per pair.

    ``bge-reranker-v2-m3`` and ``ms-marco-MiniLM-L-6-v2`` both export a single
    regression logit, but a two-label export is common enough in the wild that
    guessing wrong silently inverts the ranking -- so the arity is inspected rather
    than assumed (Rules.md AP-10's failure mode, one level up).
    """
    if logits.ndim == 1:
        flat = logits
    elif logits.shape[-1] == 1:
        flat = logits.reshape(-1)
    elif logits.shape[-1] == 2:
        flat = logits[..., 1] - logits[..., 0]
    else:
        flat = logits[..., 0]
    scores = [float(value) for value in flat.reshape(-1)[:expected]]
    if len(scores) != expected:
        raise RuntimeError(f"cross-encoder returned {len(scores)} scores for {expected} pairs")
    return scores


#: Process-wide session cache. Caching the *session* is required (AP-06, API.md's
#: ``warm``); caching rerank *scores* is forbidden (Rules.md Rule 2). Keyed by the
#: artifact identity so a model swap cannot serve a stale session.
_ENCODER_CACHE: dict[tuple[str, str, int, int], OnnxCrossEncoder] = {}
_CACHE_LOCK = threading.Lock()


def _derive_onnx_path(model_name: str) -> Path:
    """Map a hub id to its exported INT8 artifact, per Setup.md section 6.1's layout.

    ``BAAI/bge-reranker-v2-m3`` -> ``data/models/onnx/bge-reranker-v2-m3-int8/model.onnx``,
    which is exactly ``AXIOM_RERANKER_ONNX_PATH``'s documented default. Derived rather
    than hardcoded so the fallback checkpoint needs no second constant.
    """
    slug = model_name.rsplit("/", 1)[-1].lower()
    return ONNX_MODEL_ROOT / f"{slug}-int8" / "model.onnx"


def _build_encoder(
    settings: Settings | None, model_name: str, is_primary: bool
) -> OnnxCrossEncoder:
    """Construct (but do not load) a cached ONNX encoder for one checkpoint."""
    override = _tunable(settings, "reranker_onnx_path", None) if is_primary else None
    onnx_path = Path(override) if override else _derive_onnx_path(model_name)
    max_tokens = int(_tunable(settings, "rerank_max_tokens", DEFAULT_RERANK_MAX_TOKENS))
    num_threads = int(getattr(settings, "num_threads", 0) or 0)
    offline = bool(getattr(settings, "offline", False))
    key = (model_name, str(onnx_path), max_tokens, num_threads)
    with _CACHE_LOCK:
        cached = _ENCODER_CACHE.get(key)
        if cached is None:
            cached = OnnxCrossEncoder(
                model_name=model_name,
                onnx_path=onnx_path,
                max_tokens=max_tokens,
                num_threads=num_threads,
                offline=offline,
            )
            _ENCODER_CACHE[key] = cached
    return cached


def load_cross_encoder(settings: Settings | None = None) -> PairScorer | None:
    """Walk the model rungs of the ladder and return the first checkpoint that loads.

    Primary (``Settings.reranker_model``) -> fallback checkpoint -> the lexical rung
    when ``AXIOM_RERANK_LEXICAL_FALLBACK`` is on -> ``None``, which the caller reads
    as "passthrough". Each failed rung logs once via :func:`log_degradation`; none
    of them raises, because an unavailable model is bad *environment*, not a
    violated contract (Rules.md Rule 3).
    """
    primary = str(getattr(settings, "reranker_model", None) or "BAAI/bge-reranker-v2-m3")
    candidates: list[tuple[str, bool]] = [(primary, True)]
    if primary != FALLBACK_RERANKER_MODEL:
        candidates.append((FALLBACK_RERANKER_MODEL, False))

    for index, (model_name, is_primary) in enumerate(candidates):
        encoder = _build_encoder(settings, model_name, is_primary)
        try:
            encoder.warm()
        except Exception as exc:
            has_next = index + 1 < len(candidates)
            next_rung = candidates[index + 1][0] if has_next else "lexical-or-passthrough"
            reason = f"{model_name} unavailable: {type(exc).__name__}: {exc}"
            if encoder.claim_failure_report():
                log_degradation(_LOGGER, COMPONENT, reason, next_rung)
            else:
                _LOGGER.debug("reranker rung still unavailable: %s", reason)
            continue
        _LOGGER.info(
            "reranker ready model=%s path=%s",
            model_name,
            encoder.onnx_path,
            extra={"axiom_extra": {"stage": STAGE, "model": model_name}},
        )
        return encoder

    if bool(_tunable(settings, "rerank_lexical_fallback", False)):
        _LOGGER.debug("falling back to the lexical scorer")
        return LexicalPairScorer(
            k1=float(_tunable(settings, "rerank_lexical_k1", DEFAULT_LEXICAL_K1)),
            phrase_bonus=float(
                _tunable(settings, "rerank_lexical_phrase_bonus", DEFAULT_LEXICAL_PHRASE_BONUS)
            ),
            logit_span=float(
                _tunable(settings, "rerank_lexical_logit_span", DEFAULT_LEXICAL_LOGIT_SPAN)
            ),
        )
    return None


def warm_reranker(settings: Settings | None = None) -> str | None:
    """Eagerly construct the reranker singleton; return the model that loaded.

    Backs ``/warm``'s ``warmed: ["reranker"]`` (API.md). ``None`` means every rung
    failed and queries will run in passthrough.
    """
    encoder = load_cross_encoder(settings)
    return None if encoder is None else encoder.name


def _order(results: Iterable[FusedResult]) -> list[FusedResult]:
    """Canonical descending order with the NFR-08 tie-break.

    Sorts on ``final_score`` rather than ``rerank_score or 0.0`` because a ``None``
    rerank score means "not a rerank candidate", not "irrelevant"
    (Schema.md section 8). Ties break on ascending ``chunk_id`` so the same inputs
    always produce the same list, whatever order the caller happened to build it in.
    """
    return sorted(results, key=lambda r: (-r.final_score, r.chunk_id))


def _passthrough(
    candidates: Sequence[FusedResult],
    top_k: int,
    mode: RerankMode,
    reason: str | None,
    elapsed_ms: float,
    model_name: str | None = None,
) -> RerankOutcome:
    """Build the well-formed no-rerank outcome: RRF order, every score left ``None``.

    Not an error path even though it is usually reached from one -- Rules.md
    section 3 calls passthrough "a first-class outcome", and the caller can always
    detect it from :attr:`RerankOutcome.mode`.
    """
    return RerankOutcome(
        results=_order(candidates)[:top_k],
        mode=mode,
        model_name=model_name,
        elapsed_ms=elapsed_ms,
        degraded_reason=reason,
    )


def rerank_detailed(
    query: str,
    candidates: Sequence[FusedResult],
    chunks: dict[str, Chunk],
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
    encoder: PairScorer | None = None,
    batch_size: int | None = None,
    ledger: TimingLedger | None = None,
) -> RerankOutcome:
    """Rerank the fused candidates and report which rung of the ladder ran.

    Args:
        query: The user's query, verbatim. Never truncated (TestPlan.md TC-063).
        candidates: Fused results, at most ``Settings.fusion_top_n`` of which are
            scored. Treated as read-only -- new frozen models are returned
            (Rules.md AP-08).
        chunks: ``chunk_id -> Chunk`` hydration map from Appflow.md's step 5. A
            candidate missing from it keeps ``rerank_score=None`` and sinks to the
            bottom rather than aborting the batch.
        settings: Resolved configuration. ``None`` loads the active profile.
        top_k: Final result count, overriding ``Settings.top_k_default``. Exists
            because ``POST /search`` carries a per-request ``k`` (API.md) that must
            not require rebuilding the whole ``Settings`` object.
        encoder: Injected pair scorer. Present for TestPlan.md TC-059..TC-061,
            which need to count pair scorings against a known score table.
        batch_size: Overrides the configured batch size. Results are identical at
            every batch size (TC-061); only peak RSS and call overhead move.
        ledger: Optional :class:`TimingLedger` for NFR-10's ``timings`` block.

    Returns:
        A :class:`RerankOutcome`. Its ``results`` are sorted descending by
        ``final_score``, tie-broken by ascending ``chunk_id``, and truncated to
        ``Settings.top_k_default``.
    """
    if settings is None:
        from axiom.config import get_settings

        settings = get_settings()

    resolved_top_k = int(top_k if top_k is not None else settings.top_k_default)
    resolved_top_k = max(0, resolved_top_k)
    fusion_top_n = int(settings.fusion_top_n)
    max_chars = int(_tunable(settings, "rerank_max_chars", DEFAULT_RERANK_MAX_CHARS))
    resolved_batch = int(
        batch_size
        if batch_size is not None
        else _tunable(settings, "rerank_batch_size", DEFAULT_RERANK_BATCH_SIZE)
    )
    resolved_batch = max(1, resolved_batch)
    deadline = Deadline(float(settings.reranker_timeout_ms))

    if not candidates:
        return RerankOutcome(
            results=[], mode=RerankMode.PASSTHROUGH, degraded_reason="no_candidates"
        )

    if not settings.reranker_enabled:
        # FR-12's ablation switch. Deliberately NOT a degradation: nothing failed,
        # so this logs at INFO and never reaches log_degradation.
        _LOGGER.info(
            "rerank disabled, passing RRF order through n=%d",
            len(candidates),
            extra={"axiom_extra": {"stage": STAGE, "n_candidates": len(candidates)}},
        )
        return _passthrough(
            candidates,
            resolved_top_k,
            RerankMode.DISABLED,
            "reranker_enabled=false",
            deadline.elapsed_ms,
        )

    # Cap to the declared pre-rerank width before paying for a single forward pass:
    # the widths chain (TechSpecifications.md section 5.2) is what bounds this stage's
    # 620 ms, and an over-long caller list would quietly blow NFR-03.
    pool = _order(candidates)[:fusion_top_n]

    scorer = encoder if encoder is not None else load_cross_encoder(settings)
    if scorer is None:
        log_degradation(_LOGGER, COMPONENT, "no reranker available", "rerank passthrough")
        return _passthrough(
            pool, resolved_top_k, RerankMode.PASSTHROUGH, "model_unavailable", deadline.elapsed_ms
        )

    scorable: list[int] = []
    pairs: list[tuple[str, str]] = []
    truncated = 0
    missing = 0
    for index, candidate in enumerate(pool):
        chunk = chunks.get(candidate.chunk_id)
        if chunk is None:
            missing += 1
            continue
        document, was_cut = truncate_document(chunk.text, max_chars)
        truncated += int(was_cut)
        scorable.append(index)
        pairs.append((query, document))

    if missing:
        # Hydration gaps are the indexer's problem, not a reason to lose the query.
        _LOGGER.warning(
            "rerank skipped %d candidate(s) absent from the hydration map",
            missing,
            extra={"axiom_extra": {"stage": STAGE, "n_candidates": len(pool)}},
        )
    if not pairs:
        log_degradation(_LOGGER, COMPONENT, "no hydrated candidate text", "rerank passthrough")
        return _passthrough(
            pool, resolved_top_k, RerankMode.PASSTHROUGH, "no_hydrated_chunks", deadline.elapsed_ms
        )

    scores: list[float] = []
    for start in range(0, len(pairs), resolved_batch):
        if deadline.expired:
            # Checked between batches, never mid-batch: a forward pass already in
            # flight is allowed to finish, mirroring the agent loop's deadline
            # discipline (Rules.md AP-03).
            log_degradation(
                _LOGGER,
                COMPONENT,
                f"RERANKER_TIMEOUT after {deadline.elapsed_ms:.0f}ms "
                f"(budget {settings.reranker_timeout_ms}ms, {len(scores)}/{len(pairs)} scored)",
                "rerank passthrough",
            )
            return _passthrough(
                pool,
                resolved_top_k,
                RerankMode.PASSTHROUGH,
                "RERANKER_TIMEOUT",
                deadline.elapsed_ms,
                getattr(scorer, "name", None),
            )
        batch = pairs[start : start + resolved_batch]
        try:
            raw = list(scorer.score_pairs(batch))
        except Exception as exc:
            # Logged exactly once for the whole batch loop, not once per pair
            # (TestPlan.md TC-062, Rules.md section 9.1 point 4). ERROR carries the
            # exception for TC-062; the WARNING from log_degradation names the rung.
            _LOGGER.error(
                "cross-encoder failed on batch %d: %s: %s",
                start // resolved_batch,
                type(exc).__name__,
                exc,
                extra={"axiom_extra": {"stage": STAGE, "model": getattr(scorer, "name", None)}},
            )
            log_degradation(
                _LOGGER, COMPONENT, f"scoring failed: {type(exc).__name__}", "rerank passthrough"
            )
            return _passthrough(
                pool,
                resolved_top_k,
                RerankMode.PASSTHROUGH,
                "scoring_failed",
                deadline.elapsed_ms,
                getattr(scorer, "name", None),
            )
        if len(raw) != len(batch):
            log_degradation(
                _LOGGER,
                COMPONENT,
                f"scorer returned {len(raw)} scores for {len(batch)} pairs",
                "rerank passthrough",
            )
            return _passthrough(
                pool, resolved_top_k, RerankMode.PASSTHROUGH, "arity_mismatch", deadline.elapsed_ms
            )
        scores.extend(float(value) for value in raw)

    if any(not math.isfinite(value) for value in scores):
        # A NaN would sort unpredictably and poison the sufficiency predicate, so the
        # whole batch is discarded rather than partially trusted.
        log_degradation(_LOGGER, COMPONENT, "non-finite cross-encoder score", "rerank passthrough")
        return _passthrough(
            pool, resolved_top_k, RerankMode.PASSTHROUGH, "non_finite_score", deadline.elapsed_ms
        )

    rescored = list(pool)
    for position, score in zip(scorable, scores, strict=True):
        rescored[position] = pool[position].model_copy(update={"rerank_score": _sigmoid(score)})

    mode = (
        RerankMode.LEXICAL
        if getattr(scorer, "kind", "cross_encoder") == "lexical"
        else RerankMode.CROSS_ENCODER
    )
    elapsed_ms = deadline.elapsed_ms
    if mode is RerankMode.LEXICAL:
        # Loud once per query, not once per process: a caller reading the log must be
        # able to tell that THIS answer's rerank_score is uncalibrated (Rules.md Rule 3).
        log_degradation(
            _LOGGER, COMPONENT, "no cross-encoder loadable", "lexical scorer (rrf_score governs)"
        )
    if ledger is not None:
        ledger.stages.append(
            StageTiming(
                stage=STAGE,
                elapsed_ms=elapsed_ms,
                detail={"n_candidates": len(pool), "pairs": len(pairs), "mode": str(mode)},
            )
        )
    _LOGGER.info(
        "rerank scored %d pair(s) mode=%s batch=%d truncated=%d",
        len(pairs),
        mode,
        resolved_batch,
        truncated,
        extra={
            "axiom_extra": {
                "stage": STAGE,
                "n_candidates": len(pool),
                "elapsed_ms": round(elapsed_ms, 3),
            }
        },
    )
    return RerankOutcome(
        results=_order(rescored)[:resolved_top_k],
        mode=mode,
        model_name=getattr(scorer, "name", None),
        pairs_scored=len(pairs),
        batch_size=resolved_batch,
        truncated_documents=truncated,
        elapsed_ms=elapsed_ms,
        degraded_reason=None if mode is RerankMode.CROSS_ENCODER else "lexical_fallback",
    )


def rerank(
    query: str,
    candidates: Sequence[FusedResult],
    chunks: dict[str, Chunk],
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
    encoder: PairScorer | None = None,
    batch_size: int | None = None,
    ledger: TimingLedger | None = None,
) -> list[FusedResult]:
    """FR-12's stage signature: fused candidates in, top-``k`` reranked results out.

    A thin wrapper over :func:`rerank_detailed` that drops the report. Prefer
    :func:`rerank_detailed` anywhere the *provenance* of the score matters --
    above all in ``agent/evaluator.py``, which must know whether to threshold
    ``rerank_score`` or ``rrf_score`` (TechSpecifications.md section 5.3).
    """
    return rerank_detailed(
        query,
        candidates,
        chunks,
        settings,
        top_k=top_k,
        encoder=encoder,
        batch_size=batch_size,
        ledger=ledger,
    ).results


def sufficiency_score_field(outcome: RerankOutcome) -> str:
    """Name of the score attribute ``agent/evaluator.py`` must threshold.

    Free function as well as a property so the evaluator can import one symbol and
    stay ignorant of this module's types beyond it.
    """
    return outcome.score_field
