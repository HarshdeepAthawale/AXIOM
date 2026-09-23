"""Dense retrieval: the semantic leg of the three-signal first stage.

Implements FR-06. Reads a built index only -- it never re-chunks and never
re-embeds a chunk on the read path (Rules.md AP-12); the only thing encoded here
is the query itself.

Degradation ladder (Rules.md section 3, ``axiom.retrieval.dense:search``):
embed + search -> empty ``list[ScoredChunk]``, **signal marked absent**. Absent
is not the same as zero-filled: fusion drops the missing signal's weight and
renormalises the rest (TechSpecifications.md section 5.1.2), so returning
zero-scored rows here would poison the fused ranking with 100 irrelevant
candidates. Every early return in this module returns ``[]`` for that reason.

The one deliberately fatal case is a dimension mismatch between the runtime
embedder and the index (Schema.md invariant 18, TestPlan.md TC-030): inner
products between vectors of different rank are not a degraded answer, they are
an impossible one, so it raises ``AxiomContractError`` before any search call.
A *model identity* mismatch at equal dim degrades to signal-absent instead --
the arithmetic is well-defined but the geometry is meaningless, and a silently
garbage dense leg is worse for the user than no dense leg at all.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from axiom.core.errors import AxiomContractError
from axiom.core.logging import get_logger, log_degradation
from axiom.indexing.dense import (
    BACKEND_FAISS,
    BACKEND_NUMPY,
    DENSE_IDMAP_FILENAME,
    DENSE_INDEX_FILENAME,
    DENSE_MATRIX_FILENAME,
)
from axiom.indexing.embedder import Embedder, load_embedder
from axiom.schema import ScoredChunk, SignalKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from axiom.config import Settings
    from axiom.schema import QueryPlan

_LOG = get_logger("retrieval.dense")

_COMPONENT = "retrieval.dense:search"


class DenseRetriever:
    """Inner-product search over one version's dense index.

    Construction is cheap and never fails: the index and the embedder are opened
    on first use so that ``axiom --help`` and test collection pay no model-load
    cost (Rules.md AP-06), and so that a missing index degrades at query time
    into an absent signal rather than exploding at wiring time.

    One retriever is bound to one version directory. Version scoping
    (``--version v1``, FR-20 / TestPlan.md TC-040) is therefore structural: a
    query against v1 opens ``.axiom/index/v1/`` and can physically not see
    another version's rows.
    """

    def __init__(
        self,
        index_dir: Path,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.settings = settings
        self._embedder = embedder
        self._owns_embedder = embedder is None
        self._index: Any | None = None
        self._matrix: np.ndarray | None = None
        self._chunk_ids: list[str] = []
        self._backend = ""
        self._loaded = False
        self._usable = False
        # The agent fans a plan's sub-queries out across a thread pool, so the
        # first three searches of a process hit this object concurrently. Without
        # the lock, the thread that set ``_loaded`` first was still reading the
        # idmap while the others returned ``self._usable`` -- still False -- and
        # answered an empty dense list. Measured: two of three cold concurrent
        # searches silently returned nothing, which made the first query of every
        # CLI invocation differ from the second (NFR-08).
        self._load_lock = threading.Lock()

    # -- loading -----------------------------------------------------------

    def _load(self) -> bool:
        """Open the idmap, the vectors, and the embedder. Returns usability.

        Runs at most once; a failed load latches to unusable so that a 3,765-query
        eval run does not pay a filesystem probe and a stack trace per query.
        """
        if self._loaded:
            return self._usable
        with self._load_lock:
            if self._loaded:
                return self._usable
            # ``_loaded`` is published *after* ``_usable``, not before. Setting
            # the latch first is what made the unlocked version wrong even once
            # a lock existed: the fast path above would see ``_loaded`` already
            # true, skip the lock entirely, and read a ``_usable`` the loading
            # thread had not written yet.
            try:
                self._usable = self._load_locked()
            finally:
                self._loaded = True
            return self._usable

    def _load_locked(self) -> bool:
        """The body of :meth:`_load`, run by exactly one thread under the lock."""
        idmap_path = self.index_dir / DENSE_IDMAP_FILENAME
        try:
            payload = json.loads(idmap_path.read_text(encoding="utf-8"))
            rows = [str(row) for row in payload["rows"]]
            index_dim = int(payload["embedding_dim"])
            index_model = str(payload["embedding_model"])
            backend = str(payload.get("backend", BACKEND_FAISS))
        except Exception as exc:
            log_degradation(
                _LOG,
                _COMPONENT,
                f"cannot read {idmap_path}: {type(exc).__name__}: {exc}",
                "empty dense result, signal marked absent",
            )
            return False

        embedder = self._embedder if self._embedder is not None else load_embedder(self.settings)
        self._embedder = embedder
        if embedder.dim != index_dim:
            # Rules.md section 3 category 1: "an embedding of the wrong
            # dimension" is one of exactly three cases where raising is correct.
            raise AxiomContractError(
                f"dense index at {self.index_dir} was built at dim {index_dim} by "
                f"{index_model}; the runtime embedder {embedder.model_id} is dim "
                f"{embedder.dim}. Rebuild the index or restore the matching model."
            )
        if embedder.model_id != index_model:
            log_degradation(
                _LOG,
                _COMPONENT,
                f"index built by {index_model}, runtime embedder is {embedder.model_id}",
                "empty dense result, signal marked absent",
            )
            return False

        if not self._open_vectors(backend, len(rows)):
            return False
        self._chunk_ids = rows
        self._backend = backend
        self._usable = True
        return True

    def _open_vectors(self, backend: str, expected: int) -> bool:
        """Load the FAISS index or the degraded matrix; verify the row count."""
        try:
            if backend == BACKEND_NUMPY:
                matrix = np.load(self.index_dir / DENSE_MATRIX_FILENAME, allow_pickle=False)
                self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)
                actual = int(self._matrix.shape[0])
            else:
                import faiss

                if self.settings.num_threads > 0:
                    faiss.omp_set_num_threads(self.settings.num_threads)
                self._index = faiss.read_index(str(self.index_dir / DENSE_INDEX_FILENAME))
                actual = int(self._index.ntotal)
        except Exception as exc:
            log_degradation(
                _LOG,
                _COMPONENT,
                f"cannot open {backend} dense index in {self.index_dir}: "
                f"{type(exc).__name__}: {exc}",
                "empty dense result, signal marked absent",
            )
            return False
        if actual != expected:
            # Schema.md invariant 12 (row correspondence is triple). Treated as
            # an unusable index rather than a raise: at query time a truncated
            # artefact is indistinguishable from a missing one, and the declared
            # ladder for a missing index is signal-absent.
            log_degradation(
                _LOG,
                _COMPONENT,
                f"row-count mismatch in {self.index_dir}: idmap has {expected}, "
                f"{backend} index has {actual}",
                "empty dense result, signal marked absent",
            )
            return False
        return True

    @property
    def available(self) -> bool:
        """True when a search would actually consult vectors. Forces the lazy load."""
        return self._load()

    @property
    def count(self) -> int:
        """Number of indexed vectors, or 0 when the index is unusable."""
        return len(self._chunk_ids) if self._load() else 0

    def close(self) -> None:
        """Drop native handles. Only closes the embedder if this object opened it."""
        self._index = None
        self._matrix = None
        if self._owns_embedder and self._embedder is not None:
            self._embedder.close()
        self._embedder = None
        self._usable = False

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        plan: QueryPlan | None = None,
        k: int | None = None,
    ) -> list[ScoredChunk]:
        """Return the top-``k`` dense candidates for one query string.

        ``plan`` is accepted for interface symmetry with the other two
        retrievers and used for exactly one thing: recovering the verbatim
        original query when the caller hands over an empty or whitespace-only
        string. Notably it does *not* append ``expansion_terms`` -- Schema.md
        section 10 states those are "appended to the sparse query only", because
        padding a dense query with synonyms moves the embedding away from the
        user's actual intent rather than toward it. Sub-query decomposition is
        the agent loop's job: it calls this once per ``effective_query`` and lets
        RRF merge the lists.

        Args:
            query: The (sub-)query text to encode.
            plan: The active plan, used only as a fallback source for the text.
            k: Candidate width. Defaults to ``Settings.dense_top_k`` (100).

        Returns:
            Up to ``k`` :class:`ScoredChunk` records with ``signal=DENSE``, raw
            inner-product scores (== cosine, vectors are normalised), sorted
            descending, ranks ``1..N`` contiguous, ties broken by ascending
            ``chunk_id``. ``[]`` whenever the signal is unavailable.
        """
        if not self.settings.dense_enabled:
            return []
        text = query.strip() or (plan.original_query.strip() if plan is not None else "")
        if not text:
            log_degradation(
                _LOG, _COMPONENT, "query is empty after stripping", "empty dense result"
            )
            return []
        if not self._load():
            return []

        width = self.settings.dense_top_k if k is None else k
        width = min(max(int(width), 0), len(self._chunk_ids))
        if width == 0:
            return []

        try:
            assert self._embedder is not None
            vector = self._embedder.encode_query(text, self.settings.embedding_query_instruction)
        except Exception as exc:
            log_degradation(
                _LOG,
                _COMPONENT,
                f"query embedding failed: {type(exc).__name__}: {exc}",
                "empty dense result, signal marked absent",
            )
            return []

        try:
            hits = self._probe(vector, width)
        except Exception as exc:
            log_degradation(
                _LOG,
                _COMPONENT,
                f"vector search failed: {type(exc).__name__}: {exc}",
                "empty dense result, signal marked absent",
            )
            return []

        # Sort with an explicit total order. FAISS breaks score ties by internal
        # row order, which is stable within a build but not across rebuilds;
        # chunk_id is stable forever (Rules.md Rule 4, NFR-08).
        hits.sort(key=lambda item: (-item[1], item[0]))
        return [
            ScoredChunk(chunk_id=chunk_id, score=score, rank=rank, signal=SignalKind.DENSE)
            for rank, (chunk_id, score) in enumerate(hits, start=1)
        ]

    def _probe(self, vector: np.ndarray, width: int) -> list[tuple[str, float]]:
        """Return a deterministic top-``width`` by inner product.

        The tie-break has to be part of *selecting* the set, not a sort applied
        afterwards. Both backends pick their ``width`` rows by a rule of their
        own -- FAISS by internal row order, numpy by ``argpartition``'s unstable
        introselect -- so when the score at the cut-off is shared by more rows
        than fit, each backend keeps a different, arbitrary subset. Sorting that
        subset by ``(-score, chunk_id)`` afterwards yields a correctly ordered
        list of the wrong members, which is worse than an obvious failure: it
        looks deterministic, and is, within one build.

        So both paths over-fetch until the tie group straddling the boundary is
        fully in hand, then apply the documented total order and cut. Identical
        output across backends and rebuilds (NFR-08, Rules.md Rule 4).

        It matters most on the degraded path: with the hash embedder, exact
        float32 ties are pervasive, and that is precisely the configuration a
        locked-down evaluator runs.
        """
        query = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)

        if self._index is not None:
            total = len(self._chunk_ids)
            fetch = min(total, max(width, 1))
            pairs: list[tuple[str, float]] = []
            while True:
                scores, labels = self._index.search(query, fetch)
                pairs = []
                for label, score in zip(labels[0], scores[0], strict=True):
                    row = int(label)
                    # FAISS pads with -1 when an IVF probe finds fewer than `fetch`.
                    if row < 0 or row >= total or not np.isfinite(score):
                        continue
                    pairs.append((self._chunk_ids[row], float(score)))
                # Stop once the boundary score is strictly better than the last
                # one fetched: every row that could displace a keeper is present.
                # `< width` (not `<=`): at exactly `width` results the
                # boundary row IS the last fetched row, so `<=` would stop
                # before over-fetching and reinstate the truncation this whole
                # method exists to avoid.
                exhausted = fetch >= total or len(pairs) < width
                boundary_settled = len(pairs) >= width and pairs[width - 1][1] > pairs[-1][1]
                if exhausted or boundary_settled:
                    break
                fetch = min(total, fetch * 2)
            return self._deterministic_top(pairs, width)

        matrix = self._matrix
        if matrix is None or matrix.size == 0:
            return []
        sims = np.nan_to_num(matrix @ query[0], nan=-np.inf)
        if width >= sims.shape[0]:
            rows = np.arange(sims.shape[0])
        else:
            # argpartition is unstable, so widen the cut to the whole tie group
            # sitting on the boundary before the total order is applied.
            rows = np.argpartition(-sims, width - 1)[:width]
            boundary = sims[rows].min()
            rows = np.flatnonzero(sims >= boundary)
        return self._deterministic_top(
            [(self._chunk_ids[int(row)], float(sims[int(row)])) for row in rows], width
        )

    @staticmethod
    def _deterministic_top(pairs: list[tuple[str, float]], width: int) -> list[tuple[str, float]]:
        """Apply the documented total order -- score desc, then chunk_id asc -- and cut."""
        pairs.sort(key=lambda item: (-item[1], item[0]))
        return pairs[:width]


def search(
    index_dir: Path,
    query: str,
    settings: Settings,
    plan: QueryPlan | None = None,
    k: int | None = None,
    *,
    embedder: Embedder | None = None,
) -> list[ScoredChunk]:
    """One-shot convenience wrapper around :class:`DenseRetriever`.

    Appflow.md names ``axiom.retrieval.dense:search`` as the stage entrypoint, so
    it exists as a function too. Prefer the class inside the agent loop: it
    keeps the embedder and the FAISS handle warm across passes, and the loop
    runs up to two passes per query under a 5 s budget.
    """
    retriever = DenseRetriever(index_dir, settings, embedder=embedder)
    try:
        return retriever.search(query, plan, k)
    finally:
        retriever.close()


__all__ = ["DenseRetriever", "search"]
