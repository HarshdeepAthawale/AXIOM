"""Axiom as an MTEB v2 retrieval model, plus the offline data path it needs.

This module is the seam between a code-retrieval *pipeline* and a benchmark that
expects an *embedding model*. Three facts shape everything below.

1. **MTEB v2, never v1.** Setup.md section 4.4 pins ``mteb>=2.0``: v2 replaced
   ``MTEB(tasks=...).run(model)`` with :func:`mteb.evaluate`, replaced
   ``mteb.encoder_interface.Encoder`` with ``mteb.models.abs_encoder.AbsEncoder``,
   and added ``mteb.abstasks.SearchProtocol`` -- the interface that lets a model
   own both indexing and searching instead of only producing vectors. Axiom is a
   three-signal fused pipeline with a reranker on top; expressing it as a bare
   encoder would throw away everything after the dense stage. So
   :class:`AxiomSearchModel` implements ``SearchProtocol``, and
   :class:`AxiomEncoder` exists only for the ablation runs where a plain dense
   baseline is what we actually want to measure.

2. **MTEB may not be installed.** ``mteb`` and ``datasets`` are optional extras
   (NFR-07), so nothing here imports them at module scope. Every class in this
   file is importable, constructible, and unit-testable on a bare install, and
   the corpus can come from a vendored BEIR-shaped directory under ``data/``
   instead of the Hub -- that is assumption A-1's mitigation in PRD.md section 10,
   and it is also what makes an offline demo survive venue wifi.

3. **Ids are sacred, and here they are load-bearing twice over.** TechSpecifications.md
   section 4.10 makes this the one module where a contract violation must raise
   rather than degrade: every document id handed to the scorer is checked for
   membership in the corpus id set first, because a mangled id does not fail --
   it silently scores 0.0 and looks like a bad model. Corpus documents become
   :class:`~axiom.schema.Chunk` objects whose ``chunk_id`` is a digest of their
   text, so the mapping back to MTEB's document ids lives in an explicit dict
   (Rules.md AP-01, AP-13), never in string surgery on an id.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from axiom.config import Settings, get_settings
from axiom.core.errors import AxiomContractError, IndexNotFoundError
from axiom.core.logging import get_logger, log_degradation
from axiom.schema import Chunk, ChunkKind, ChunkLocation, ChunkMetadata

__all__ = [
    "AxiomEncoder",
    "AxiomSearchModel",
    "CallableBackend",
    "Corpus",
    "EncoderBackend",
    "LexicalBackend",
    "Queries",
    "RetrievalTaskData",
    "SearchBackend",
    "build_search_model",
    "code_tokens",
    "corpus_to_chunks",
    "load_local_task",
    "load_mteb_task",
    "normalise_corpus",
    "normalise_queries",
    "resolve_backend",
    "resolve_task",
]

_LOGGER = get_logger("eval.mteb_adapter")

#: BEIR/MTEB corpus: document id -> {"title": ..., "text": ...}.
Corpus = dict[str, dict[str, str]]

#: BEIR/MTEB queries: query id -> query text.
Queries = dict[str, str]

#: Default vendored dataset root (A-1's mitigation). ``data/`` is gitignored.
DEFAULT_LOCAL_DATA_ROOT = Path("data/datasets")

#: Okapi BM25 saturation and length-normalisation parameters for the *fallback*
#: lexical backend only. These are the literature-standard values, not tuned
#: Axiom constants: the real sparse signal is ``retrieval/sparse.py`` over
#: ``bm25s``, and this backend exists so the harness can be exercised end to end
#: with zero optional dependencies. A run that used it is stamped degraded and is
#: never reportable, so no published number can rest on them (Rules.md AP-07's
#: concern is tunables that influence a reported score).
_FALLBACK_BM25_K1 = 1.5
_FALLBACK_BM25_B = 0.75

#: Splits an identifier into its parts while keeping the original token:
#: ``preprocessInput`` -> ``preprocess``, ``input``, ``preprocessinput``.
_TOKEN_RE = re.compile(r"[A-Za-z][a-z0-9]*|[A-Z]+(?![a-z])|\d+")
_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def code_tokens(text: str) -> list[str]:
    """Lower-cased, code-aware tokens: camelCase and snake_case split, original kept.

    Mirrors the tokenisation policy of FR-07 so the fallback backend and the real
    sparse signal at least agree on what a token *is*. Splitting alone would lose
    the exact-identifier match that makes ``handleDeeplink`` findable by its own
    name, so both the parts and the whole are emitted.
    """
    out: list[str] = []
    for raw in _SPLIT_RE.split(text):
        if not raw:
            continue
        lowered = raw.lower()
        parts = [piece.lower() for piece in _TOKEN_RE.findall(raw)]
        out.extend(parts)
        if len(parts) != 1 or parts[0] != lowered:
            out.append(lowered)
    return out


# --------------------------------------------------------------------------
# Task data: the corpus, the queries, the judgments, and where they came from
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalTaskData:
    """One BEIR-shaped retrieval task, however it was sourced.

    Carrying ``source`` and ``dataset_revision`` alongside the data is not
    decoration: TestPlan.md section 6.4 requires a results file to be reproducible
    from its experiment-log row, and "which copy of AppsRetrieval was this?" is
    part of that. A vendored corpus with an unknown revision is still a valid run
    -- it is just a run that says so out loud.
    """

    task_name: str
    split: str
    corpus: Corpus
    queries: Queries
    qrels: dict[str, dict[str, float]]
    source: str
    dataset_revision: str | None = None
    hf_subset: str = "default"
    languages: tuple[str, ...] = ("eng-Latn", "python-Code")

    def limited(self, limit: int | None) -> RetrievalTaskData:
        """Return a truncated copy for a smoke run, corpus and queries together.

        TestPlan.md section 6.2 is emphatic that a limited run is a *different
        task*, not a cheaper estimate of this one: fewer distractors inflate NDCG
        mechanically. Truncation therefore keeps the qrels consistent (queries
        whose positives fell out of the corpus keep only the judgments that
        survive) so the number is at least internally coherent, and
        ``scripts/run_eval.py`` refuses to mark such a run reportable.
        """
        if limit is None or limit < 1:
            return self
        query_ids = sorted(self.queries)[:limit]
        kept_queries = {qid: self.queries[qid] for qid in query_ids}
        # Keep every judged positive for the surviving queries, then pad the
        # corpus with the first `limit` documents so there are real distractors.
        needed = {doc_id for qid in query_ids for doc_id in self.qrels.get(qid, {})}
        padding = [doc_id for doc_id in sorted(self.corpus) if doc_id not in needed][:limit]
        kept_ids = sorted(needed.union(padding).intersection(self.corpus))
        kept_corpus = {doc_id: self.corpus[doc_id] for doc_id in kept_ids}
        kept_qrels = {
            qid: {d: g for d, g in self.qrels.get(qid, {}).items() if d in kept_corpus}
            for qid in query_ids
        }
        return RetrievalTaskData(
            task_name=self.task_name,
            split=self.split,
            corpus=kept_corpus,
            queries=kept_queries,
            qrels=kept_qrels,
            source=f"{self.source}+limit={limit}",
            dataset_revision=self.dataset_revision,
            hf_subset=self.hf_subset,
            languages=self.languages,
        )


def _doc_text(document: Mapping[str, Any]) -> str:
    """Flatten a BEIR document to the text the retriever actually sees.

    Title and body are joined with a blank line rather than a space: on APPS the
    "title" is usually empty, and where it is not, a hard break stops a title from
    fusing into the first line of code and creating a token that exists in neither.
    """
    title = str(document.get("title") or "").strip()
    text = str(document.get("text") or "")
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def normalise_corpus(raw: Any) -> Corpus:
    """Accept any of MTEB v2's corpus shapes and return ``{doc_id: {title, text}}``.

    MTEB v2 hands a ``datasets.Dataset`` with ``id``/``title``/``text`` columns;
    MTEB v1 and BEIR hand a dict keyed by ``_id``; our vendored JSONL is a list of
    records. All three arrive here and leave in one shape, so nothing downstream
    needs a branch. Records missing an id are dropped with a warning rather than
    aborting the run (Rule 3) -- one malformed line must not cost a two-hour eval.
    """
    corpus: Corpus = {}
    dropped = 0

    def add(record: Mapping[str, Any], fallback_id: str | None = None) -> None:
        nonlocal dropped
        doc_id = record.get("id") or record.get("_id") or record.get("doc_id") or fallback_id
        if doc_id is None:
            dropped += 1
            return
        corpus[str(doc_id)] = {
            "title": str(record.get("title") or ""),
            "text": str(record.get("text") or ""),
        }

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                add(value, fallback_id=str(key))
            else:
                dropped += 1
    else:
        for record in raw:
            if isinstance(record, Mapping):
                add(record)
            else:
                dropped += 1
    if dropped:
        _LOGGER.warning(
            "dropped %d corpus records with no usable id",
            dropped,
            extra={"axiom_extra": {"stage": "eval", "kept": len(corpus)}},
        )
    return corpus


def normalise_queries(raw: Any) -> Queries:
    """Accept any of MTEB v2's query shapes and return ``{query_id: text}``.

    Conversational tasks give a query as a list of turns; AppsRetrieval does not,
    but joining a list defensively costs one line and avoids a ``str(list)`` that
    would embed Python quoting into the query text.
    """
    queries: Queries = {}
    dropped = 0

    def coerce(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence):
            return "\n".join(str(part) for part in value)
        return str(value)

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                queries[str(key)] = coerce(value.get("text", ""))
            else:
                queries[str(key)] = coerce(value)
    else:
        for record in raw:
            if not isinstance(record, Mapping):
                dropped += 1
                continue
            query_id = record.get("id") or record.get("_id") or record.get("query_id")
            if query_id is None:
                dropped += 1
                continue
            queries[str(query_id)] = coerce(record.get("text", ""))
    if dropped:
        _LOGGER.warning(
            "dropped %d query records with no usable id",
            dropped,
            extra={"axiom_extra": {"stage": "eval", "kept": len(queries)}},
        )
    return queries


def _normalise_qrels(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Coerce judgments to ``{query_id: {doc_id: float grade}}``, dropping junk."""
    qrels: dict[str, dict[str, float]] = {}
    for query_id, judgments in raw.items():
        graded: dict[str, float] = {}
        for doc_id, grade in judgments.items():
            try:
                graded[str(doc_id)] = float(grade)
            except (TypeError, ValueError):
                _LOGGER.warning(
                    "non-numeric qrel grade %r for (%s, %s); dropped",
                    grade,
                    query_id,
                    doc_id,
                    extra={"axiom_extra": {"stage": "eval"}},
                )
        qrels[str(query_id)] = graded
    return qrels


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines loudly (Rule 3)."""
    records: list[dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
            else:
                bad += 1
    if bad:
        _LOGGER.warning(
            "skipped %d malformed lines in %s",
            bad,
            path.name,
            extra={"axiom_extra": {"stage": "eval", "path": str(path)}},
        )
    return records


def load_local_task(
    root: Path,
    *,
    task_name: str,
    split: str,
) -> RetrievalTaskData:
    """Load a vendored BEIR-shaped dataset from disk -- the offline path.

    Expected layout under ``root`` (exactly what ``beir``/``mteb`` write to disk,
    so a cached copy can be dropped in unmodified)::

        <root>/corpus.jsonl          {"_id": ..., "title": ..., "text": ...}
        <root>/queries.jsonl         {"_id": ..., "text": ...}
        <root>/qrels/<split>.tsv     query-id \\t corpus-id \\t score   (with header)

    A missing *directory or file* raises :class:`IndexNotFoundError`: the dataset
    not being there is an environment failure the CLI reports as exit 3, not bad
    input to degrade around. A malformed *record* inside a present file is bad
    input and is skipped with a warning (Rule 3).

    Args:
        root: directory holding ``corpus.jsonl``, ``queries.jsonl`` and ``qrels/``.
        task_name: recorded into the results JSON, e.g. ``"AppsRetrieval"``.
        split: which qrels file to read, e.g. ``"test"``.

    Raises:
        IndexNotFoundError: the directory or one of the three required files is
            absent.
    """
    root = Path(root)
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"
    missing = [p for p in (corpus_path, queries_path, qrels_path) if not p.is_file()]
    if missing:
        raise IndexNotFoundError(
            f"local dataset at {root} is incomplete; missing "
            f"{', '.join(str(p.relative_to(root)) for p in missing)}"
        )

    corpus = normalise_corpus(_read_jsonl(corpus_path))
    queries = normalise_queries(_read_jsonl(queries_path))

    raw_qrels: dict[str, dict[str, Any]] = {}
    with qrels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            query_id, doc_id, score = row[0].strip(), row[1].strip(), row[2].strip()
            if query_id in {"query-id", "query_id", "qid"}:  # header row
                continue
            try:
                grade = float(score)
            except ValueError:
                continue
            raw_qrels.setdefault(query_id, {})[doc_id] = grade

    revision_path = root / "revision.txt"
    revision = (
        revision_path.read_text(encoding="utf-8").strip() if revision_path.is_file() else None
    )

    data = RetrievalTaskData(
        task_name=task_name,
        split=split,
        corpus=corpus,
        queries=queries,
        qrels=_normalise_qrels(raw_qrels),
        source=f"local:{root}",
        dataset_revision=revision,
    )
    _LOGGER.info(
        "loaded local task %s/%s: %d docs, %d queries, %d judged queries",
        task_name,
        split,
        len(data.corpus),
        len(data.queries),
        len(data.qrels),
        extra={"axiom_extra": {"stage": "eval", "source": data.source}},
    )
    return data


def load_mteb_task(task_name: str, split: str) -> RetrievalTaskData:
    """Load a task through MTEB v2 -- the online path.

    Uses ``mteb.get_task(...)`` + ``load_data()`` and reads the loaded splits off
    the task object. MTEB v2 exposes them as ``task.dataset[subset][split]`` with
    ``corpus`` / ``queries`` / ``relevant_docs`` keys; v1's ``task.corpus[split]``
    layout is also accepted so a v1 install fails on the *version check* in
    Setup.md section 4.4 rather than on an obscure ``KeyError`` here.

    Raises:
        ImportError: ``mteb`` is not installed. Callers catch this and fall back
            to :func:`load_local_task` -- see :func:`resolve_task`.
        IndexNotFoundError: MTEB is present but the task or split is not.
    """
    import mteb

    version = getattr(mteb, "__version__", "unknown")
    if not hasattr(mteb, "evaluate"):
        _LOGGER.warning(
            "mteb %s has no evaluate(); this is the v1 API. Setup.md 4.4 pins mteb>=2.0",
            version,
            extra={"axiom_extra": {"stage": "eval", "mteb_version": version}},
        )

    task = mteb.get_task(task_name)
    task.load_data()

    subset = "default"
    corpus_raw: Any = None
    queries_raw: Any = None
    qrels_raw: Any = None

    dataset = getattr(task, "dataset", None)
    if isinstance(dataset, Mapping):
        subsets = dataset.get(subset, dataset)
        payload = subsets.get(split) if isinstance(subsets, Mapping) else None
        if isinstance(payload, Mapping):
            corpus_raw = payload.get("corpus")
            queries_raw = payload.get("queries")
            qrels_raw = payload.get("relevant_docs") or payload.get("qrels")
    if corpus_raw is None:  # MTEB v1 layout
        corpus_raw = getattr(task, "corpus", {}).get(split)
        queries_raw = getattr(task, "queries", {}).get(split)
        qrels_raw = getattr(task, "relevant_docs", {}).get(split)
    if corpus_raw is None or queries_raw is None or qrels_raw is None:
        raise IndexNotFoundError(f"mteb task {task_name!r} exposes no split {split!r}")

    metadata = getattr(task, "metadata", None)
    revision = None
    languages: tuple[str, ...] = ("eng-Latn", "python-Code")
    if metadata is not None:
        dataset_meta = getattr(metadata, "dataset", None)
        if isinstance(dataset_meta, Mapping):
            revision = dataset_meta.get("revision")
        langs = getattr(metadata, "eval_langs", None)
        if isinstance(langs, Sequence) and langs and all(isinstance(x, str) for x in langs):
            languages = tuple(langs)

    return RetrievalTaskData(
        task_name=task_name,
        split=split,
        corpus=normalise_corpus(corpus_raw),
        queries=normalise_queries(queries_raw),
        qrels=_normalise_qrels(qrels_raw),
        source=f"mteb:{version}",
        dataset_revision=revision,
        hf_subset=subset,
        languages=languages,
    )


def resolve_task(
    task_name: str,
    split: str,
    *,
    local_path: Path | None = None,
    settings: Settings | None = None,
) -> RetrievalTaskData:
    """Source the task, preferring a local copy, then MTEB, then failing clearly.

    The ladder, and why it is ordered this way:

    1. ``local_path`` if the caller named one -- an explicit flag always wins.
    2. ``data/datasets/<task_name>/`` if it exists -- the vendored copy (A-1).
       Checked *before* the Hub so a pre-downloaded corpus makes the run offline
       by default; venue wifi is a known failure mode (Setup.md section 5.4).
    3. ``mteb``, unless ``settings.offline`` forbids the network.

    Raises:
        IndexNotFoundError: no rung produced data. The message names every path
            that was tried, because "dataset not found" with no paths is the
            least actionable error a judge can hit.
    """
    resolved = settings or get_settings()
    tried: list[str] = []

    candidates: list[Path] = []
    if local_path is not None:
        candidates.append(Path(local_path))
    candidates.append(DEFAULT_LOCAL_DATA_ROOT / task_name)

    for candidate in candidates:
        tried.append(str(candidate))
        if not candidate.is_dir():
            continue
        try:
            return load_local_task(candidate, task_name=task_name, split=split)
        except IndexNotFoundError as exc:
            _LOGGER.warning(
                "local dataset candidate rejected: %s",
                exc,
                extra={"axiom_extra": {"stage": "eval", "path": str(candidate)}},
            )

    if resolved.offline:
        raise IndexNotFoundError(
            f"AXIOM_OFFLINE=true and no vendored copy of {task_name!r} found; tried "
            f"{', '.join(tried)}"
        )

    tried.append("mteb hub")
    try:
        return load_mteb_task(task_name, split)
    except ImportError as exc:
        log_degradation(_LOGGER, "eval.dataset", f"mteb unavailable ({exc})", "local corpus only")
    raise IndexNotFoundError(
        f"cannot source task {task_name!r} split {split!r}; tried {', '.join(tried)}. "
        f"Vendor a BEIR-shaped copy under {DEFAULT_LOCAL_DATA_ROOT / task_name} "
        f"(corpus.jsonl, queries.jsonl, qrels/{split}.tsv) or install the 'eval' extra."
    )


# --------------------------------------------------------------------------
# Corpus -> Chunks
# --------------------------------------------------------------------------


def corpus_to_chunks(
    corpus: Corpus,
    *,
    version_id: str = "mteb-eval",
    language: str = "python",
) -> tuple[list[Chunk], dict[str, str]]:
    """Turn BEIR documents into ``Chunk`` objects the Axiom pipeline can index.

    One document becomes exactly one chunk. That is not a simplification, it is
    what the corpus is: PRD.md section 2.2 records that APPS entries are
    standalone single-file Python solutions with no cross-file structure -- which
    is also why ``configs/eval.yaml`` disables the structural signal rather than
    down-weighting it.

    The returned mapping is ``chunk_id -> doc_id``. It exists because ``chunk_id``
    is a digest of (text, path, line) and can never be parsed back into a document
    id (Rules.md AP-01, AP-13). Every id that leaves :class:`AxiomSearchModel`
    goes through this dict.

    Documents with empty text are skipped: ``Chunk.text`` has ``min_length=1`` and
    an empty document is unretrievable anyway.

    Returns:
        ``(chunks, chunk_id_to_doc_id)``, chunks in ascending document-id order so
        the index build is deterministic (NFR-08).
    """
    chunks: list[Chunk] = []
    id_map: dict[str, str] = {}
    skipped = 0
    for doc_id in sorted(corpus):
        text = _doc_text(corpus[doc_id])
        if not text.strip():
            skipped += 1
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id)
        encoded = text.encode("utf-8")
        location = ChunkLocation(
            file_path=f"corpus/{safe_name}.py",
            start_line=1,
            end_line=max(1, text.count("\n") + 1),
            start_byte=0,
            end_byte=len(encoded),
        )
        metadata = ChunkMetadata(
            symbol=None,
            kind=ChunkKind.MODULE,
            language=language,
            version_id=version_id,
        )
        chunk = Chunk.create(text, location, metadata)
        if chunk.chunk_id in id_map and id_map[chunk.chunk_id] != doc_id:
            # Two corpus documents with byte-identical text at the same synthetic
            # path collide on chunk_id. Rule 1 forbids inventing a new id, so the
            # duplicate is dropped and named -- it is a property of the corpus,
            # not of our hashing, and silently keeping one would make the qrels
            # for the other unsatisfiable without saying so.
            _LOGGER.warning(
                "corpus documents %s and %s are byte-identical; %s dropped",
                id_map[chunk.chunk_id],
                doc_id,
                doc_id,
                extra={"axiom_extra": {"stage": "eval", "chunk_id": chunk.chunk_id}},
            )
            skipped += 1
            continue
        id_map[chunk.chunk_id] = doc_id
        chunks.append(chunk)
    if skipped:
        _LOGGER.warning(
            "skipped %d of %d corpus documents (empty or duplicate)",
            skipped,
            len(corpus),
            extra={"axiom_extra": {"stage": "eval"}},
        )
    return chunks, id_map


# --------------------------------------------------------------------------
# Search backends
# --------------------------------------------------------------------------


@runtime_checkable
class SearchBackend(Protocol):
    """What :class:`AxiomSearchModel` needs from whatever actually retrieves.

    Deliberately narrower than the full Axiom pipeline: ``index`` takes chunks and
    ``search`` returns ``(chunk_id, score)`` pairs, higher first. That is the only
    contract, so the real pipeline, a dense-only ablation, and the zero-dependency
    fallback are interchangeable and every one of them can be swapped in from a
    test without touching MTEB.
    """

    name: str

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build whatever state ``search`` needs over these chunks."""

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(chunk_id, score)`` pairs, best first."""


@dataclass
class LexicalBackend:
    """Pure-Python BM25 over code-aware tokens. The declared degraded rung.

    Exists so the eval harness is runnable, testable, and demonstrable on an
    install with nothing but pydantic and numpy -- which is precisely the state of
    a judge's machine before the model downloads finish. PRD.md section 2 is blunt
    that BM25 scores 4.8 on AppsRetrieval against a dense model's 14.7, so this is
    a *wiring proof*, never a result: ``scripts/run_eval.py`` stamps any run that
    used it as degraded and non-reportable.
    """

    #: Not a dataclass field: a caller must never be able to construct this
    #: backend and claim it is not the fallback.
    degraded: ClassVar[bool] = True

    name: str = "lexical-bm25-fallback"
    k1: float = _FALLBACK_BM25_K1
    b: float = _FALLBACK_BM25_B
    _postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict, repr=False)
    _chunk_ids: list[str] = field(default_factory=list, repr=False)
    _lengths: list[int] = field(default_factory=list, repr=False)
    _avg_length: float = field(default=0.0, repr=False)

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the inverted index. O(total tokens), one pass, no dependencies."""
        self._postings = {}
        self._chunk_ids = []
        self._lengths = []
        for position, chunk in enumerate(chunks):
            tokens = code_tokens(chunk.text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                self._postings.setdefault(token, []).append((position, count))
            self._chunk_ids.append(chunk.chunk_id)
            self._lengths.append(len(tokens))
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        _LOGGER.info(
            "lexical fallback indexed %d chunks, %d distinct tokens",
            len(self._chunk_ids),
            len(self._postings),
            extra={"axiom_extra": {"stage": "eval", "backend": self.name}},
        )

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Score by Okapi BM25, ties broken by ascending ``chunk_id`` (NFR-08)."""
        if not self._chunk_ids or top_k < 1:
            return []
        total_docs = len(self._chunk_ids)
        scores: dict[int, float] = {}
        for token in set(code_tokens(query)):
            postings = self._postings.get(token)
            if not postings:
                continue
            idf = math.log(1.0 + (total_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, frequency in postings:
                length_norm = (
                    1.0
                    - self.b
                    + self.b
                    * (self._lengths[position] / self._avg_length if self._avg_length else 1.0)
                )
                weight = (frequency * (self.k1 + 1.0)) / (frequency + self.k1 * length_norm)
                scores[position] = scores.get(position, 0.0) + idf * weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self._chunk_ids[item[0]]))[
            :top_k
        ]
        return [(self._chunk_ids[position], score) for position, score in ranked]


@dataclass
class EncoderBackend:
    """Exact cosine search over any object exposing ``encode(list[str]) -> array``.

    The ``AbsEncoder`` half of FR-22: it turns a bare embedder -- Axiom's own, or
    a ``sentence-transformers`` model in an ablation -- into a retriever without
    FAISS. Exact search is deliberate here: this path exists to measure a model,
    and an ANN index would fold its own recall loss into the number. FAISS is the
    production read path (FR-05), not the measurement one.

    Uses ``numpy`` only, which is a hard dependency.
    """

    encoder: Any
    name: str = "encoder-exact-cosine"
    batch_size: int = 64

    _matrix: Any = field(default=None, repr=False)
    _chunk_ids: list[str] = field(default_factory=list, repr=False)

    @property
    def degraded(self) -> bool:
        """True when the wrapped embedder says it fell back to a weaker model.

        Read through ``getattr`` rather than declared on the protocol: an embedder
        owned by another workstream is not obliged to expose the flag, and its
        absence must mean "no claim made", never "not degraded" by fiat. Anything
        that does expose it is believed.
        """
        return bool(getattr(self.encoder, "degraded", False))

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Embed and L2-normalise every chunk, so inner product *is* cosine (AP-10)."""
        import numpy as np

        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        if not chunks:
            self._matrix = None
            return
        vectors: list[Any] = []
        texts = [chunk.text for chunk in chunks]
        for start in range(0, len(texts), max(1, self.batch_size)):
            batch = texts[start : start + max(1, self.batch_size)]
            vectors.append(np.asarray(self.encoder.encode(batch), dtype="float32"))
        matrix = np.vstack(vectors).astype("float32", copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._matrix = matrix / norms

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Cosine similarity against every chunk; ties break by ``chunk_id``."""
        import numpy as np

        if self._matrix is None or not self._chunk_ids or top_k < 1:
            return []
        vector = np.asarray(self.encoder.encode([query]), dtype="float32").reshape(-1)
        norm = float(np.linalg.norm(vector)) or 1.0
        similarities = self._matrix @ (vector / norm)
        order = sorted(
            range(len(self._chunk_ids)),
            key=lambda i: (-float(similarities[i]), self._chunk_ids[i]),
        )[:top_k]
        return [(self._chunk_ids[i], float(similarities[i])) for i in order]


@dataclass
class CallableBackend:
    """Adapter for the full Axiom pipeline, injected as two callables.

    ``eval/`` deliberately does not import ``retrieval/``, ``rerank/`` or
    ``agent/``: those modules are owned elsewhere and a hard import here would
    make the eval harness un-runnable whenever one of them is mid-refactor. The
    pipeline is instead handed in -- by a test, or by ``--backend pkg.mod:factory``
    on ``scripts/run_eval.py``, where ``factory(settings)`` returns anything
    satisfying :class:`SearchBackend`.
    """

    #: The injected pipeline reports its own degradations through its own logs;
    #: the adapter has no way to inspect them and does not guess.
    degraded: ClassVar[bool] = False

    index_fn: Callable[[Sequence[Chunk]], None]
    search_fn: Callable[[str, int], Sequence[tuple[str, float]]]
    name: str = "callable"

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.index_fn(chunks)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Re-sort whatever the pipeline returned, enforcing Rule 4 at the seam."""
        results = list(self.search_fn(query, top_k))
        ordered = sorted(results, key=lambda item: (-float(item[1]), str(item[0])))
        return [(str(cid), float(score)) for cid, score in ordered[:top_k]]


def _call_factory(factory: Any, settings: Settings) -> Any:
    """Call a user-supplied factory, passing ``settings`` only when it asks for it.

    The obvious implementation -- try ``factory(settings)``, fall back to
    ``factory()`` on ``TypeError`` -- is wrong in a way that is nearly invisible:
    a dataclass whose first field happens to be ``name`` accepts one positional
    argument quite happily, and silently binds the entire ``Settings`` object to
    it. The object still passes an ``isinstance`` check, and the mistake surfaces
    hours later as a garbled backend name in the results file.

    So the test is the parameter *name*, resolved from the signature, and the
    argument is passed by keyword. A factory that cannot be introspected (a C
    callable, say) is called with no arguments, which is the safe direction: a
    factory that needed settings and did not get them fails loudly, whereas one
    that got settings it did not want corrupts quietly.
    """
    import inspect

    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory()
    accepts_settings = "settings" in parameters and parameters["settings"].kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    if accepts_settings:
        return factory(settings=settings)
    return factory()


def resolve_backend(spec: str | None, settings: Settings) -> SearchBackend:
    """Build the search backend named by ``spec``, degrading to lexical BM25.

    ``spec`` is ``"module.path:attribute"``. The attribute is called with the
    :class:`~axiom.config.Settings` **only if it declares a parameter named
    ``settings``** (see :func:`_call_factory` for why the name, not the arity, is
    the test) and must return a :class:`SearchBackend`. ``None`` or ``"lexical"``
    selects the fallback directly.

    Any failure -- module missing, attribute missing, wrong return type -- logs a
    degradation and returns :class:`LexicalBackend`, because a harness that dies
    on a typo'd ``--backend`` at hour two of a full-split run is worse than one
    that finishes and says loudly that it measured the wrong thing.
    """
    if spec is None or spec.strip().lower() in {"", "lexical", "fallback"}:
        return LexicalBackend()

    import importlib

    module_name, _, attribute = spec.partition(":")
    reason = ""
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute) if attribute else module
        candidate = _call_factory(factory, settings)
        if isinstance(candidate, SearchBackend):
            _LOGGER.info(
                "using injected backend %s",
                getattr(candidate, "name", spec),
                extra={"axiom_extra": {"stage": "eval", "spec": spec}},
            )
            return candidate
        reason = f"{spec} returned {type(candidate).__name__}, not a SearchBackend"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    log_degradation(_LOGGER, "eval.backend", f"{spec} unusable ({reason})", "lexical BM25 fallback")
    return LexicalBackend()


# --------------------------------------------------------------------------
# The MTEB-facing objects
# --------------------------------------------------------------------------


class AxiomEncoder:
    """MTEB v2 ``AbsEncoder``-shaped wrapper around Axiom's dense embedder.

    Only used for the dense-only ablation row; the scored configuration goes
    through :class:`AxiomSearchModel`, which keeps fusion and reranking in the
    loop. The signature accepts MTEB v2's keyword-only ``task_metadata`` /
    ``hf_split`` / ``hf_subset`` / ``prompt_type`` and ignores them: AppsRetrieval
    is single-subset and Axiom uses no task-specific instruction prefix.

    Degradation: with no embedder resolvable (``onnxruntime``/``transformers``
    absent, or ``indexing/dense.py`` not yet importable), it falls back to a
    deterministic hashed bag-of-tokens vector. That is a real, seeded, repeatable
    embedding -- it is simply a weak one, and :attr:`degraded` says so, which is
    what stops ``scripts/run_eval.py`` calling the resulting number reportable.
    """

    def __init__(self, settings: Settings | None = None, embedder: Any = None) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._resolved = embedder is not None

    @property
    def degraded(self) -> bool:
        """True when either this wrapper or the embedder beneath it fell back.

        Two rungs can fail independently: no embedder resolves at all (the hashed
        fallback answers here), or one resolves but has itself degraded to a
        weaker model. Both make the run non-reportable, so both fold into one
        flag rather than leaving the caller to check two.
        """
        embedder = self.embedder
        if embedder is None:
            return True
        return bool(getattr(embedder, "degraded", False))

    @property
    def model_identity(self) -> str:
        """Which embedder is actually answering, resolving it if needed."""
        embedder = self.embedder
        return _identity(embedder) if embedder is not None else "axiom/hashed-tokens-fallback"

    @property
    def embedder(self) -> Any:
        """The underlying embedder, resolved lazily on first use (Rules.md AP-06)."""
        if not self._resolved:
            self._embedder = _resolve_axiom_embedder(self.settings)
            self._resolved = True
        return self._embedder

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Encode text to a ``(n, dim)`` float32 array, L2-normalised.

        ``inputs`` may be a sequence of strings or MTEB v2's ``DataLoader`` of
        batched dicts; both are flattened to strings first, because a benchmark
        harness changing its batching shape must not be able to change our
        vectors.
        """
        import numpy as np

        texts = _flatten_inputs(inputs)
        embedder = self.embedder
        if embedder is not None:
            matrix = np.asarray(embedder.encode(texts), dtype="float32")
        else:
            matrix = _hashed_embeddings(texts, self.settings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms


@dataclass
class AxiomSearchModel:
    """Axiom as an MTEB v2 ``SearchProtocol`` implementation.

    ``index()`` converts the benchmark corpus into chunks and hands them to the
    backend; ``search()`` runs each query and maps chunk ids back to document ids.
    The mapping is checked in both directions: a chunk id the backend invented and
    a document id outside the corpus are both :class:`AxiomContractError`, not
    warnings. TechSpecifications.md section 4.10 singles this module out as the
    one place where a contract violation must raise unconditionally, and the
    reason is that the failure is otherwise invisible -- MTEB scores an unknown id
    as a miss, so a corrupted id map reads as "the model is bad" rather than "the
    harness is broken".
    """

    backend: SearchBackend
    settings: Settings
    version_id: str = "mteb-eval"
    _chunk_to_doc: dict[str, str] = field(default_factory=dict, repr=False)
    _corpus_ids: frozenset[str] = field(default_factory=frozenset, repr=False)
    _indexed: int = field(default=0, repr=False)

    @property
    def name(self) -> str:
        """Backend identity, stamped into the results JSON."""
        return getattr(self.backend, "name", type(self.backend).__name__)

    @property
    def indexed_count(self) -> int:
        """How many chunks the backend actually holds."""
        return self._indexed

    @property
    def mteb_model_meta(self) -> Any:
        """MTEB v2's ``ModelMeta``, built lazily, or ``None`` when MTEB is absent.

        ``mteb.evaluate`` reads this to name the results directory and to record
        what was evaluated. It is built defensively: the constructor's required
        fields have moved between v2 minor releases, and a ``TypeError`` here
        would take down a run whose *scoring* does not depend on MTEB at all
        (``scripts/run_eval.py`` computes its own metrics). Returning ``None``
        lets MTEB fall back to its own defaults instead.
        """
        try:
            import mteb
        except ImportError:
            return None
        meta_cls = getattr(mteb, "ModelMeta", None)
        if meta_cls is None:
            return None
        try:
            return meta_cls(
                name=f"axiom/{self.name}",
                revision=self.settings.profile,
                release_date=None,
                languages=["eng-Latn", "python-Code"],
                n_parameters=None,
                max_tokens=None,
                embed_dim=self.settings.embedding_dim,
                license=None,
                open_weights=True,
                framework=[],
                similarity_fn_name="cosine",
                use_instructions=False,
            )
        except TypeError as exc:
            _LOGGER.warning(
                "mteb.ModelMeta signature mismatch (%s); evaluating without model metadata",
                exc,
                extra={"axiom_extra": {"stage": "eval"}},
            )
            return None

    @property
    def degraded(self) -> bool:
        """Whether any part of the retrieval path admitted to falling back.

        ``scripts/run_eval.py`` turns this into the ``reportable: false`` stamp in
        the results file. A degraded run is a perfectly good wiring proof and a
        perfectly bad number, and the file has to say which one it is (NFR-07).
        """
        return bool(getattr(self.backend, "degraded", False))

    @property
    def backend_detail(self) -> dict[str, Any]:
        """What actually ran, for the provenance block.

        Names the backend and, where one exists, the embedder underneath it --
        because "encoder-exact-cosine" alone does not distinguish a Qwen3 run from
        a run where every model download failed and the hashing fallback answered.
        """
        detail: dict[str, Any] = {"name": self.name, "degraded": self.degraded}
        encoder = getattr(self.backend, "encoder", None)
        if encoder is not None:
            detail["encoder"] = _identity(encoder)
            detail["encoder_degraded"] = bool(getattr(encoder, "degraded", False))
        return detail

    def index(
        self,
        corpus: Any,
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the index over an MTEB corpus. Keyword names match SearchProtocol."""
        normalised = normalise_corpus(corpus)
        chunks, id_map = corpus_to_chunks(normalised, version_id=self.version_id)
        self._chunk_to_doc = id_map
        self._corpus_ids = frozenset(normalised)
        self._indexed = len(chunks)
        self.backend.index(chunks)
        _LOGGER.info(
            "indexed %d chunks from %d corpus documents via %s",
            len(chunks),
            len(normalised),
            self.name,
            extra={"axiom_extra": {"stage": "eval", "hf_split": hf_split, "backend": self.name}},
        )

    def search(
        self,
        queries: Any,
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        top_k: int = 100,
        encode_kwargs: Mapping[str, Any] | None = None,
        top_ranked: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run every query and return a BEIR-shaped run.

        Args:
            queries: MTEB queries in any of the shapes :func:`normalise_queries`
                accepts.
            top_k: candidate depth. Recall@100 is a reported metric (PRD.md
                section 2), so this must be at least 100 on a scored run.
            top_ranked: reranking-task restriction. AppsRetrieval is a retrieval
                task and never sets it; if it arrives, results are filtered to it
                rather than ignored.

        Returns:
            ``{query_id: {doc_id: score}}``.

        Raises:
            AxiomContractError: the backend returned an id that is not in the
                corpus id map.
        """
        normalised = normalise_queries(queries)
        depth = max(1, top_k)
        run: dict[str, dict[str, float]] = {}
        allowed_per_query = top_ranked or {}
        empty_queries = 0

        for query_id in sorted(normalised):
            text = normalised[query_id]
            if not text.strip():
                # Rules.md AP-11: an empty query is bad input mid-run, not a
                # reason to lose the other 3,764 results.
                _LOGGER.warning(
                    "empty query text; returning no results for it",
                    extra={"axiom_extra": {"stage": "eval", "query_id": query_id}},
                )
                run[query_id] = {}
                empty_queries += 1
                continue
            hits = self.backend.search(text, depth)
            scored = self._to_doc_scores(hits, query_id)
            restriction = allowed_per_query.get(query_id)
            if restriction is not None:
                allowed = set(restriction)
                scored = {d: s for d, s in scored.items() if d in allowed}
            run[query_id] = scored
            if not scored:
                empty_queries += 1

        if empty_queries:
            _LOGGER.warning(
                "%d of %d queries returned no documents",
                empty_queries,
                len(normalised),
                extra={"axiom_extra": {"stage": "eval", "backend": self.name}},
            )
        return run

    def _to_doc_scores(self, hits: Iterable[tuple[str, float]], query_id: str) -> dict[str, float]:
        """Map ``chunk_id -> doc_id``, asserting membership before MTEB sees it."""
        scored: dict[str, float] = {}
        for chunk_id, score in hits:
            doc_id = self._chunk_to_doc.get(chunk_id)
            if doc_id is None:
                raise AxiomContractError(
                    f"backend {self.name!r} returned chunk_id {chunk_id!r} for query "
                    f"{query_id!r} which is not in the corpus id map "
                    f"({len(self._chunk_to_doc)} entries). Rules.md Rule 1: ids are "
                    f"never rewritten, re-derived, or invented between stages."
                )
            if doc_id not in self._corpus_ids:
                raise AxiomContractError(
                    f"id map yielded doc_id {doc_id!r}, absent from the "
                    f"{len(self._corpus_ids)}-document corpus"
                )
            # A chunk id maps to exactly one doc id, so no collision is possible;
            # keeping the max is defensive against a backend emitting duplicates.
            previous = scored.get(doc_id)
            scored[doc_id] = score if previous is None else max(previous, score)
        return scored


def build_search_model(
    settings: Settings | None = None,
    *,
    backend: SearchBackend | None = None,
    backend_spec: str | None = None,
) -> AxiomSearchModel:
    """Construct the MTEB-facing model, resolving the backend if none was given.

    Precedence matches the config chain in Rules.md section 7: an explicit object
    beats a dotted-path spec, which beats the declared fallback.
    """
    resolved = settings or get_settings()
    chosen = backend if backend is not None else resolve_backend(backend_spec, resolved)
    return AxiomSearchModel(backend=chosen, settings=resolved)


# --------------------------------------------------------------------------
# Optional-dependency helpers
# --------------------------------------------------------------------------

#: Where an Axiom dense embedder might live, in the order we try. The package
#: layout is locked in _CONTRACT.md section 3, but the factory's *name* is the
#: owning workstream's choice, so several are accepted. Every rung is optional:
#: when none resolves, the hashed fallback runs and says so.
_EMBEDDER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("axiom.indexing.dense", "load_embedder"),
    ("axiom.indexing.dense", "get_embedder"),
    ("axiom.indexing.dense", "DenseEmbedder"),
    ("axiom.retrieval.dense", "load_embedder"),
    ("axiom.retrieval.dense", "DenseEmbedder"),
)


def _resolve_axiom_embedder(settings: Settings) -> Any:
    """Find an object with ``.encode(list[str])``, or ``None`` after logging why.

    Never raises. Import errors here are the normal case on a bare install, and
    the encoder degrades rather than taking the run down with it (NFR-07).
    """
    import importlib

    for module_name, attribute in _EMBEDDER_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        factory = getattr(module, attribute, None)
        if factory is None:
            continue
        try:
            instance = _call_factory(factory, settings)
        except Exception as exc:
            _LOGGER.warning(
                "embedder factory %s:%s failed: %s",
                module_name,
                attribute,
                exc,
                extra={"axiom_extra": {"stage": "eval"}},
            )
            continue
        if hasattr(instance, "encode"):
            _LOGGER.info(
                "resolved dense embedder from %s:%s",
                module_name,
                attribute,
                extra={"axiom_extra": {"stage": "eval", "model": settings.embedding_model}},
            )
            return instance
    log_degradation(
        _LOGGER,
        "eval.embedder",
        "no axiom dense embedder importable",
        "deterministic hashed bag-of-tokens vectors",
    )
    return None


def _identity(obj: Any) -> str:
    """Best available human name for a model object, for the provenance stamp.

    Checked in decreasing order of specificity. The class name is the floor, not
    the answer: two different Qwen revisions share a class and the results file
    has to be able to tell them apart (NFR-09 pins model revisions).
    """
    for attribute in ("model_identity", "model_id", "model_name", "name", "identity"):
        value = getattr(obj, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(obj).__name__


def _flatten_inputs(inputs: Any) -> list[str]:
    """Reduce MTEB's several input shapes to a flat list of strings.

    v2 passes a ``DataLoader`` yielding dicts of batched columns; v1 and every
    test pass a plain sequence of strings. Both land here.
    """
    if isinstance(inputs, str):
        return [inputs]
    texts: list[str] = []
    for item in inputs:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, Mapping):
            column = item.get("text")
            if isinstance(column, str):
                texts.append(column)
            elif isinstance(column, Sequence):
                texts.extend(str(part) for part in column)
            else:
                texts.append(_doc_text(item))
        elif isinstance(item, Sequence):
            texts.extend(str(part) for part in item)
        else:
            texts.append(str(item))
    return texts


def _hashed_embeddings(texts: Sequence[str], settings: Settings) -> Any:
    """Deterministic hashed bag-of-tokens vectors -- the encoder's floor.

    A token's coordinate is a blake2b digest of ``(seed, token)`` reduced modulo
    the embedding dimension, with the sign taken from a second digest bit so
    unrelated tokens cancel instead of always adding. Seeded from
    ``Settings.seed`` and free of any wall-clock or RNG state, so two runs of the
    same text produce byte-identical vectors (NFR-08).
    """
    import numpy as np

    from axiom.core.hashing import blake2b_128

    dim = max(8, settings.embedding_dim)
    matrix = np.zeros((len(texts), dim), dtype="float32")
    seed = str(settings.seed).encode("ascii")
    for row, text in enumerate(texts):
        for token in code_tokens(text):
            digest = blake2b_128(seed, token.encode("utf-8"))
            index = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
            matrix[row, index] += sign
    return matrix
