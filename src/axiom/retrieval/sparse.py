"""Sparse (BM25) retrieval over ``sparse.bm25s/``.

Answers "what code *contains these exact tokens*?" (Design.md section 4.1) and
is the dominant signal for ``USAGE`` queries at weight 0.55
(TechSpecifications.md section 5.1.1). Candidate width K=100.

Read path only. The index is loaded, never rebuilt -- re-chunking or re-embedding
here is Rules.md AP-12 and would turn a 3 ms stage into a minutes-long one.

The query token stream is the raw query **plus** ``plan.extracted_identifiers``
**plus** ``plan.expansion_terms``. That last one is the asymmetry worth naming:
expansion terms go to sparse and *never* to the dense encoder
(TechSpecifications.md section 3.3.3). Dense already models semantic
neighbourhoods, so injecting ``normalize``/``sanitize``/``transform`` into the
embedding input would blur the query vector toward their centroid; BM25 has no
such notion and genuinely cannot find ``normalize`` from ``preprocess`` unless
the term is handed to it.

Every rung of the ladder (Rules.md section 3: *code-aware tokenise -> whitespace
tokenise -> empty list*) returns a valid ``list[ScoredChunk]``. An empty sparse
signal is routine, not exceptional -- fusion renormalises the remaining weights
around it (TestPlan.md TC-055), so nothing here ever raises on a bad query.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from axiom.core.logging import get_logger, log_degradation
from axiom.indexing.sparse import (
    BACKEND_BM25S,
    BACKEND_PURE_PYTHON,
    CHUNK_IDS_FILENAME,
    FALLBACK_FILENAME,
    META_FILENAME,
    TOKENIZER_VERSION,
    sparse_dir,
)
from axiom.retrieval.tokenizer import tokenize, whitespace_tokenize
from axiom.schema import ScoredChunk, SignalKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from axiom.config import Settings
    from axiom.schema import QueryPlan

__all__ = ["SparseRetriever", "search"]

_LOG = get_logger("retrieval.sparse")

_COMPONENT = "retrieval.sparse"


def _query_tokens(query: str, plan: QueryPlan | None) -> list[str]:
    """Build the sparse query stream, descending the tokenisation ladder.

    Repetition is load-bearing and deliberate. An identifier that appears in the
    raw query *and* in ``extracted_identifiers`` contributes its tokens twice,
    which is a term-frequency boost on precisely the terms the planner judged
    highest-precision. Deduplicating here would throw that signal away in the
    name of tidiness.

    Everything -- query, identifiers, expansion terms alike -- goes through the
    same :func:`~axiom.retrieval.tokenizer.tokenize` the index was built with.
    An identifier tokenised by any other route would not match the postings it
    was extracted to hit.
    """
    parts: list[str] = [query]
    if plan is not None:
        parts.extend(plan.extracted_identifiers)
        parts.extend(plan.expansion_terms)

    tokens: list[str] = []
    for part in parts:
        tokens.extend(tokenize(part))
    if tokens:
        return tokens

    for part in parts:
        tokens.extend(whitespace_tokenize(part))
    if tokens:
        log_degradation(
            _LOG, _COMPONENT, "code-aware tokeniser yielded zero tokens", "whitespace tokenise"
        )
        return tokens

    log_degradation(_LOG, _COMPONENT, "query yielded zero tokens", "empty list")
    return []


def _rank(scored: list[tuple[str, float]], k: int) -> list[ScoredChunk]:
    """Sort, truncate, and stamp contiguous 1-indexed ranks.

    Ties break by ascending ``chunk_id`` (NFR-08). BM25 ties are common in a
    code corpus -- two chunks matching one rare identifier once each score
    identically -- so without an explicit tiebreak the ranked order would depend
    on dict iteration order and the repeat-run diff would fail.

    Non-positive scores are dropped rather than ranked. bm25s pads its result
    array out to ``k`` with zero-scoring documents when fewer than ``k`` match,
    and a zero BM25 score is the absence of evidence, not weak evidence -- it
    must not enter fusion as a rank-1..100 contribution (TC-045).
    """
    positive = [(chunk_id, score) for chunk_id, score in scored if score > 0.0]
    positive.sort(key=lambda item: (-item[1], item[0]))
    return [
        ScoredChunk(chunk_id=chunk_id, score=float(score), rank=index + 1, signal=SignalKind.SPARSE)
        for index, (chunk_id, score) in enumerate(positive[:k])
    ]


class SparseRetriever:
    """BM25 search over one version's ``sparse.bm25s/`` directory.

    Construction is cheap and never touches disk; the index loads on the first
    :meth:`search` and is held for the retriever's lifetime. That keeps
    ``axiom --help`` and test collection free of index-load cost (Rules.md
    AP-06) while still amortising the load across a multi-pass agent loop, where
    the same retriever is queried once per refinement pass.
    """

    def __init__(self, index_dir: Path | str, settings: Settings | None = None) -> None:
        self.dir = sparse_dir(Path(index_dir))
        self.settings = settings
        self._loaded = False
        self._available = False
        # Same race as ``DenseRetriever``: the agent's fan-out calls ``search``
        # from several threads at once, and an unsynchronised latch let the
        # losers read ``_available`` before the winner had finished setting it.
        self._load_lock = threading.Lock()
        self._backend = ""
        self._chunk_ids: list[str] = []
        self._fallback: dict[str, Any] | None = None
        self._bm25s: Any = None

    # -- loading ---------------------------------------------------------

    def _load_json(self, name: str) -> dict[str, Any] | None:
        path = self.dir / name
        if not path.is_file():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log_degradation(_LOG, _COMPONENT, f"unreadable {name} ({exc})", "empty list")
            return None
        return loaded if isinstance(loaded, dict) else None

    def _load(self) -> bool:
        """Load the index once. Returns availability; never raises."""
        if self._loaded:
            return self._available
        with self._load_lock:
            if self._loaded:
                return self._available
            # ``_loaded`` is published after ``_available``; see the same note in
            # ``DenseRetriever._load``.
            try:
                self._available = self._load_locked()
            finally:
                self._loaded = True
            return self._available

    def _load_locked(self) -> bool:
        """The body of :meth:`_load`, run by exactly one thread under the lock."""
        meta = self._load_json(META_FILENAME)
        if meta is None:
            log_degradation(
                _LOG, _COMPONENT, f"no sparse index at {self.dir}", "empty list (signal absent)"
            )
            return False

        # A tokeniser mismatch is the one failure mode this signal cannot detect
        # from its own scores -- it just returns nothing and looks weak. Say so.
        on_disk_version = meta.get("tokenizer_version")
        if on_disk_version != TOKENIZER_VERSION:
            log_degradation(
                _LOG,
                _COMPONENT,
                f"tokenizer_version {on_disk_version!r} != {TOKENIZER_VERSION} (reindex required)",
                "empty list (signal absent)",
            )
            return False

        idmap = self._load_json(CHUNK_IDS_FILENAME)
        rows = idmap.get("rows") if idmap else None
        if not isinstance(rows, list):
            log_degradation(
                _LOG, _COMPONENT, f"missing or malformed {CHUNK_IDS_FILENAME}", "empty list"
            )
            return False
        self._chunk_ids = [str(row) for row in rows]

        self._backend = str(meta.get("backend", BACKEND_PURE_PYTHON))
        if self._backend == BACKEND_BM25S:
            self._available = self._load_bm25s()
        else:
            self._available = self._load_fallback()
        return self._available

    def _load_fallback(self) -> bool:
        payload = self._load_json(FALLBACK_FILENAME)
        if payload is None or "postings" not in payload:
            log_degradation(
                _LOG, _COMPONENT, f"missing {FALLBACK_FILENAME}", "empty list (signal absent)"
            )
            return False
        self._fallback = payload
        return True

    def _load_bm25s(self) -> bool:
        """Load the native bm25s index, falling back to the JSON backend if present.

        Lazy import per NFR-07. ``mmap=True`` matches Schema.md section 14.6 and
        keeps the resident set flat across the three concurrent retrievers.

        If bm25s is gone but the index was built with it, there is nothing left
        to read: the native format is not parseable without the library, and
        rebuilding on the read path is forbidden (AP-12). The signal is declared
        absent and the log names the remedy, because a silent shortfall here
        costs roughly a sixth of the fused weight on the ``eval`` profile.
        """
        try:
            import bm25s

            self._bm25s = bm25s.BM25.load(str(self.dir), mmap=True)
        except ImportError as exc:
            if (self.dir / FALLBACK_FILENAME).is_file():
                log_degradation(
                    _LOG, _COMPONENT, f"bm25s unavailable ({exc})", "pure-python BM25 postings"
                )
                self._backend = BACKEND_PURE_PYTHON
                return self._load_fallback()
            log_degradation(
                _LOG,
                _COMPONENT,
                f"index built with bm25s but bm25s is unavailable ({exc}); reindex to "
                f"regenerate a dependency-free sparse index",
                "empty list (signal absent)",
            )
            return False
        except Exception as exc:
            log_degradation(
                _LOG, _COMPONENT, f"bm25s load failed ({exc!r})", "empty list (signal absent)"
            )
            return False
        return True

    # -- scoring ---------------------------------------------------------

    def _score_fallback(self, tokens: list[str]) -> list[tuple[str, float]]:
        """Pure-Python Okapi BM25, Lucene variant.

        ``idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))`` is the Lucene form,
        which is non-negative for every ``df`` -- the classic Robertson form goes
        negative once a term appears in more than half the corpus, and a negative
        contribution would let a common token *penalise* a chunk that contains it.
        In a code corpus where tokens like ``function`` and ``return`` are
        near-universal that is not a corner case, it is most of the vocabulary.

        The ``(k1 + 1)`` numerator factor is omitted, as Lucene and bm25s omit
        it: it is a constant multiplier across all documents and terms, so it
        changes every score by the same factor and no ranking at all. Leaving it
        out keeps this backend's scores on the same scale as the bm25s backend's.

        Query term frequency multiplies the contribution, so a term arriving from
        both the raw query and ``extracted_identifiers`` counts twice -- the
        boost described in :func:`_query_tokens`.
        """
        assert self._fallback is not None  # guarded by _load_fallback
        postings: dict[str, list[int]] = self._fallback["postings"]
        doc_lens: list[int] = self._fallback["doc_lens"]
        doc_count = int(self._fallback["doc_count"])
        avg_doc_len = float(self._fallback["avg_doc_len"])
        k1 = float(self._fallback["k1"])
        b = float(self._fallback["b"])
        if doc_count == 0 or avg_doc_len <= 0.0:
            return []

        query_tf: dict[str, int] = {}
        for token in tokens:
            query_tf[token] = query_tf.get(token, 0) + 1

        accumulator: dict[int, float] = {}
        for term, qtf in query_tf.items():
            flat = postings.get(term)
            if not flat:
                continue
            df = len(flat) // 2
            idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            weight = qtf * idf
            for offset in range(0, len(flat), 2):
                doc_index = flat[offset]
                tf = flat[offset + 1]
                norm = k1 * (1.0 - b + b * doc_lens[doc_index] / avg_doc_len)
                accumulator[doc_index] = accumulator.get(doc_index, 0.0) + weight * tf / (tf + norm)

        return [
            (self._chunk_ids[doc_index], score)
            for doc_index, score in accumulator.items()
            if 0 <= doc_index < len(self._chunk_ids)
        ]

    def _score_bm25s(self, tokens: list[str], k: int) -> list[tuple[str, float]]:
        """Score via the native backend, clamping ``k`` to the corpus size.

        bm25s raises rather than truncating when ``k`` exceeds the document
        count, which on a small fixture corpus is every query. Clamping is not
        optional.
        """
        limit = min(k, len(self._chunk_ids))
        if limit <= 0:
            return []
        try:
            documents, scores = self._bm25s.retrieve([tokens], k=limit)
        except Exception as exc:
            log_degradation(_LOG, _COMPONENT, f"bm25s retrieve failed ({exc!r})", "empty list")
            return []
        out: list[tuple[str, float]] = []
        for doc_index, score in zip(documents[0], scores[0], strict=False):
            index = int(doc_index)
            if 0 <= index < len(self._chunk_ids):
                out.append((self._chunk_ids[index], float(score)))
        return out

    # -- public API ------------------------------------------------------

    def search(
        self,
        query: str,
        plan: QueryPlan | None = None,
        k: int | None = None,
    ) -> list[ScoredChunk]:
        """Return up to ``k`` sparse hits, best first.

        Args:
            query: Raw query text. Never mutated, never required to be well
                formed -- all-punctuation and empty strings are ordinary inputs.
            plan: The pass's :class:`~axiom.schema.QueryPlan`. Its
                ``extracted_identifiers`` and ``expansion_terms`` are appended to
                the token stream; ``None`` means "query terms only".
            k: Candidate width. Defaults to ``Settings.sparse_top_k`` (100,
                ``_CONTRACT.md`` section 5).

        Returns:
            ``list[ScoredChunk]`` with ``signal=SPARSE``, sorted by descending
            score, ranks exactly ``1..len(result)``. Possibly empty.
        """
        settings = self.settings
        width = k if k is not None else (settings.sparse_top_k if settings else 100)
        if settings is not None and not settings.sparse_enabled:
            _LOG.info(
                "sparse signal disabled by configuration",
                extra={"axiom_extra": {"stage": "sparse", "event": "signal_disabled"}},
            )
            return []
        if width <= 0 or not self._load():
            return []

        tokens = _query_tokens(query, plan)
        if not tokens:
            return []

        if self._backend == BACKEND_BM25S:
            scored = self._score_bm25s(tokens, width)
        else:
            scored = self._score_fallback(tokens)
        return _rank(scored, width)


#: Loaded retrievers, keyed by (resolved directory, ``prism_meta.json`` mtime).
#: Keying on mtime rather than path alone means a ``axiom reindex`` that
#: rewrites the directory invalidates the entry instead of serving a stale
#: postings list for the rest of the process's life.
_CACHE: dict[tuple[str, float], SparseRetriever] = {}


def search(
    query: str,
    plan: QueryPlan | None = None,
    k: int | None = None,
    index_dir: Path | str = Path("."),
    settings: Settings | None = None,
) -> list[ScoredChunk]:
    """Module-level entry point named by Rules.md section 3 and Design.md section 3.1.

    Holds one :class:`SparseRetriever` per index directory so the postings are
    parsed once per process rather than once per query.
    """
    directory = sparse_dir(Path(index_dir))
    meta = directory / META_FILENAME
    try:
        stamp = meta.stat().st_mtime
    except OSError:
        stamp = 0.0
    key = (str(directory.resolve()) if directory.exists() else str(directory), stamp)
    retriever = _CACHE.get(key)
    if retriever is None:
        retriever = SparseRetriever(directory, settings)
        _CACHE[key] = retriever
    return retriever.search(query, plan, k)
