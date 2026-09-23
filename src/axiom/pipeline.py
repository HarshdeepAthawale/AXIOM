"""The one orchestrator the CLI, the API and the UI all call.

Every surface that can run a query runs *this* code. That is the entire point:
three call sites reimplementing "fan out, fuse, rerank, format" drift within a
day, and then the CLI's answer and the API's answer to the same question differ
for reasons nobody can reconstruct on demo day. So the CLI is an argument
parser, the API is a serialiser, and the pipeline is the system.

Two entry points:

* :func:`build_index` -- the offline path of Appflow.md Flow 1. Chunk, embed,
  build all three indexes, write the manifest, register the version. Stages into
  ``index/<version>.tmp/`` and ``os.replace``s it into position, so an
  interrupted build leaves the previous version intact and nothing half-written
  registered (TC-034).
* :func:`query` -- the online path of Appflow.md Flows 2, 3, 6 and 7. Builds a
  plan, runs the bounded agent loop over a concurrent three-signal fan-out, and
  formats survivors into :class:`~axiom.schema.RetrievalResult`.

Three properties this module is responsible for and nothing else is:

**One failing signal must not sink a query.** Each retriever call runs inside its
own ``try``; an exception becomes an absent signal and a logged degradation, and
fusion renormalises the remaining weights (TechSpecifications.md section 5.1.2).
A query with a corrupt FAISS index, no bm25s and no structural database still
answers -- with whatever is left.

**The three signals run concurrently on threads, not processes** (Design.md
section 5.1). FAISS's search kernel and SQLite both release the GIL, and the
dense retriever holds a warm ONNX session that a process pool would force every
worker to load independently -- exactly the pathology AP-06 exists to prevent.

**Every stage is timed** (NFR-10). The ``timings`` block in a ``--json``
response is what makes a latency claim auditable rather than asserted.

No optional dependency is imported here, at module scope or anywhere else. The
whole pipeline runs -- indexing and querying, end to end -- with none of faiss,
bm25s, onnxruntime, tokenizers, transformers, tree-sitter or llama-cpp
installed. That degraded path is the one an evaluator's machine is most likely
to take, so it is the path this module is written for first.
"""

from __future__ import annotations

import contextlib
import shutil
import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from axiom.agent import loop as agent_loop
from axiom.config import Settings, get_settings
from axiom.core.errors import IndexNotFoundError
from axiom.core.hashing import compute_file_hash
from axiom.core.logging import capture_degradations, get_logger, log_degradation
from axiom.core.timing import TimingLedger
from axiom.indexing import manifest as mf
from axiom.retrieval.fusion import SIGNAL_ORDER, merge_signal_results
from axiom.schema import (
    Chunk,
    FusedResult,
    QueryPlan,
    QueryType,
    RetrievalResult,
    ScoredChunk,
    SignalKind,
    SnippetFamily,
    VersionManifest,
)

_LOG = get_logger("pipeline")

#: Upper bound on the query-time fan-out thread pool.
#:
#: Design.md section 5.1 sizes the pool at 3, one per signal, which is right for
#: a single query against a single version. A decomposed query against several
#: versions multiplies that by ``len(effective_queries) * len(versions)``, and an
#: unbounded pool would spawn a thread per SQLite connection on a 20-version
#: registry. There is no ``Settings`` field for this and it cannot affect a
#: score -- only how much of the wall clock is overlapped -- so it is a named
#: module constant rather than an invented tunable (Rules.md AP-07).
MAX_FANOUT_WORKERS = 12

#: Stage tags used in the timing ledger, matching Rules.md section 9.1's list.
STAGE_CHUNK = "chunk"
STAGE_DENSE = "dense"
STAGE_SPARSE = "sparse"
STAGE_STRUCT = "struct"
STAGE_MANIFEST = "manifest"
STAGE_FORMAT = "format"

#: ``match_reason`` for a query that fused nothing (TestPlan.md TC-056).
NO_CANDIDATES = "no candidates"


# ---------------------------------------------------------------------------
# Offline path: build_index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexReport:
    """What one index build produced, for the CLI's ``--json`` summary."""

    manifest: VersionManifest
    chunk_count: int
    file_count: int
    dense_backend: str
    dense_index_kind: str
    sparse_backend: str
    structural_skipped: bool
    degradations: list[str] = field(default_factory=list)
    ledger: TimingLedger = field(default_factory=TimingLedger)
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """The documented ``axiom index --json`` body (API.md section 8).

        Delegates to :class:`~axiom.api.models.IndexSummary` rather than
        rebuilding the field list here. An earlier hand-written copy had already
        drifted from the documented shape in two ways -- it emitted ``index_kind``
        where the contract says ``dense_index_kind``, and a nested
        ``{total_ms, stages}`` record where the contract types ``timings`` as a
        flat ``dict[str, float]`` -- which is precisely the second copy
        ``IndexSummary``'s own docstring exists to prevent. The import is local
        because ``axiom.api.models`` is pydantic-only but still not something the
        pipeline should depend on at import time.
        """
        from axiom.api.models import IndexSummary

        return IndexSummary.from_report(self).model_dump(mode="json")


def _file_hashes(root: Path, settings: Settings) -> dict[str, str]:
    """Hash every indexable source file, keyed by repo-relative POSIX path.

    These are what the incremental reindexer diffs against when git is
    unavailable (TechSpecifications.md section 5.5), so they must cover the same
    file set the chunker walked -- hence the shared discovery function rather
    than a second, subtly different walk.
    """
    from axiom.chunking import discover_source_files

    hashes: dict[str, str] = {}
    for path in discover_source_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log_degradation(
                _LOG, "pipeline.build_index", f"unreadable {path}: {exc}", "file hash omitted"
            )
            continue
        hashes[path.relative_to(root).as_posix()] = compute_file_hash(text)
    return hashes


def _build_index_detailed(
    repo_path: str | Path,
    version_id: str,
    settings: Settings | None = None,
    *,
    make_active: bool = True,
    commit_sha: str | None = None,
    parent_version: str | None = None,
    ledger: TimingLedger | None = None,
) -> IndexReport:
    """Build every index artefact for one version of a repository (Flow 1).

    The body of :func:`build_index_detailed`; see it for the contract. Split out
    so the public entry point can listen for degradation rungs around the whole
    build without indenting three hundred lines under a ``with``.

    Args:
        repo_path: Repository root. Chunk paths are recorded relative to it.
        version_id: Logical version label, e.g. ``v1`` or a commit sha.
        settings: Active configuration; the ambient profile when omitted.
        make_active: Point ``registry.json``'s ``active_version`` at this build.
        commit_sha: Provenance stamped on every chunk and on the manifest.
        parent_version: The version this one descends from, for the registry's
            parent chain (which is how version ordering is recovered -- parsing
            order out of a version label is forbidden, Rules.md AP-13).
        ledger: Timing ledger to append to; one is created when absent.

    Returns:
        An :class:`IndexReport` carrying the written
        :class:`~axiom.schema.VersionManifest`, the ledger, and every
        degradation taken. Model identity on the manifest comes from the
        embedder that *actually ran*, not from ``Settings`` -- the ladder may
        have degraded to the hash embedder, and a manifest that lies about its
        own vectors makes invariant 18 unenforceable.

    Raises:
        OSError: Only from the final publish. Appflow.md Flow 1 step 6 is one of
            the three places raising is correct: a half-written index tree is a
            state the next reader must refuse, not tolerate.

    A build never aborts because one artefact failed. A dense build that raises
    leaves the version with a sparse and a structural index and a logged
    degradation; the query path then drops the dense signal and renormalises.
    """
    resolved = settings if settings is not None else get_settings()
    ledger = ledger if ledger is not None else TimingLedger()
    started = perf_counter()
    root = Path(repo_path)
    degradations: list[str] = []

    mf.require_valid_version_id(version_id)

    from axiom.chunking import chunk_repo

    with ledger.measure(STAGE_CHUNK) as detail:
        try:
            raw_chunks = chunk_repo(root, version_id, resolved, commit_sha=commit_sha)
        except Exception as exc:  # pragma: no cover - chunker contains its own errors
            log_degradation(
                _LOG,
                "pipeline.build_index",
                f"chunker raised {type(exc).__name__}: {exc}",
                "empty corpus; the version is still registered",
            )
            degradations.append(f"chunking failed: {type(exc).__name__}")
            raw_chunks = []
        chunks = mf.canonical_chunk_order(raw_chunks)
        detail["chunks"] = len(chunks)
        detail["dropped_duplicates"] = len(raw_chunks) - len(chunks)

    final_dir = mf.version_dir(resolved, version_id)
    staging = final_dir.with_name(final_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    mf.write_chunks(chunks, staging / mf.chunks_path(resolved, version_id).name)

    # Dense first and alone: it is the dominant cost, and both other builders
    # only need the chunk list, which is already in memory.
    dense_backend = "none"
    dense_kind = mf.index_kind_for(len(chunks), resolved)
    embedding_model = resolved.embedding_model
    embedding_dim = resolved.embedding_dim
    with ledger.measure(STAGE_DENSE) as detail:
        try:
            from axiom.indexing.dense import build_dense_index

            dense = build_dense_index(chunks, resolved, staging)
            dense_backend = dense.backend
            dense_kind = dense.index_kind
            embedding_model = dense.embedding_model
            embedding_dim = dense.embedding_dim
            detail.update(
                {
                    "vectors": dense.count,
                    "backend": dense.backend,
                    "index_kind": dense.index_kind,
                    "cache_hits": dense.cache_hits,
                    "embedded": dense.embedded,
                }
            )
        except Exception as exc:
            log_degradation(
                _LOG,
                "pipeline.build_index",
                f"dense build failed: {type(exc).__name__}: {exc}",
                "version built without a dense index; the signal will be absent",
            )
            degradations.append(f"dense build failed: {type(exc).__name__}")

    # Appflow.md Flow 1: sparse and structural build concurrently with each
    # other, after chunking. Neither reads the other's output and both are
    # dominated by file I/O, so overlapping them is free.
    sparse_backend = "none"
    structural_skipped = not resolved.structural_enabled

    def _build_sparse() -> tuple[str, dict[str, object]]:
        from axiom.indexing.sparse import build_sparse_index

        result = build_sparse_index(chunks, resolved, staging)
        return result.backend, {
            "documents": result.doc_count,
            "vocab": result.vocab_size,
            "backend": result.backend,
        }

    def _build_structural() -> tuple[bool, dict[str, object]]:
        from axiom.indexing.structural import build_structural_index

        stats = build_structural_index(chunks, resolved, staging)
        return stats.skipped, {
            "skipped": stats.skipped,
            "symbols": stats.symbol_count,
            "calls": stats.call_count,
            "resolved_calls": stats.resolved_call_count,
        }

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="axiom-build") as pool:
        sparse_future = pool.submit(_build_sparse)
        structural_future = pool.submit(_build_structural)

        with ledger.measure(STAGE_SPARSE) as detail:
            try:
                sparse_backend, sparse_detail = sparse_future.result()
                detail.update(sparse_detail)
            except Exception as exc:
                log_degradation(
                    _LOG,
                    "pipeline.build_index",
                    f"sparse build failed: {type(exc).__name__}: {exc}",
                    "version built without a sparse index; the signal will be absent",
                )
                degradations.append(f"sparse build failed: {type(exc).__name__}")

        with ledger.measure(STAGE_STRUCT) as detail:
            try:
                structural_skipped, struct_detail = structural_future.result()
                detail.update(struct_detail)
            except Exception as exc:
                log_degradation(
                    _LOG,
                    "pipeline.build_index",
                    f"structural build failed: {type(exc).__name__}: {exc}",
                    "version built without a structural index; the signal will be absent",
                )
                degradations.append(f"structural build failed: {type(exc).__name__}")
                structural_skipped = True

    with ledger.measure(STAGE_MANIFEST) as detail:
        hashes = _file_hashes(root, resolved)
        manifest = mf.build_manifest(
            version_id,
            len(chunks),
            resolved,
            file_hashes=hashes,
            commit_sha=commit_sha,
            parent_version=parent_version,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            index_kind=dense_kind,
        )
        # Publish the staged tree first, then the manifest, then the registry
        # row: a version directory is only discoverable once every artefact in
        # it is complete (Schema.md invariant 19).
        mf.commit_version_dir(staging, final_dir)
        mf.finalize_version(manifest, resolved, make_active=make_active)
        detail["files"] = len(hashes)
        detail["version_id"] = version_id

    elapsed_ms = (perf_counter() - started) * 1000.0
    _LOG.info(
        "indexed %s: %d chunks from %d files in %.0f ms",
        version_id,
        len(chunks),
        len(hashes),
        elapsed_ms,
        extra={
            "axiom_extra": {
                "stage": STAGE_MANIFEST,
                "event": "index_built",
                "version_id": version_id,
                "chunk_count": len(chunks),
                "dense_backend": dense_backend,
                "sparse_backend": sparse_backend,
                "structural_skipped": structural_skipped,
                "degradations": degradations,
            }
        },
    )
    return IndexReport(
        manifest=manifest,
        chunk_count=len(chunks),
        file_count=len(hashes),
        dense_backend=dense_backend,
        dense_index_kind=dense_kind,
        sparse_backend=sparse_backend,
        structural_skipped=structural_skipped,
        degradations=degradations,
        ledger=ledger,
        elapsed_ms=elapsed_ms,
    )


def build_index_detailed(
    repo_path: str | Path,
    version_id: str,
    settings: Settings | None = None,
    *,
    make_active: bool = True,
    commit_sha: str | None = None,
    parent_version: str | None = None,
    ledger: TimingLedger | None = None,
) -> IndexReport:
    """Build every index artefact for one version of a repository (Flow 1).

    Args:
        repo_path: Repository root. Chunk paths are recorded relative to it.
        version_id: Logical version label, e.g. ``v1`` or a commit sha.
        settings: Active configuration; the ambient profile when omitted.
        make_active: Point ``registry.json``'s ``active_version`` at this build.
        commit_sha: Provenance stamped on every chunk and on the manifest.
        parent_version: The version this one descends from, for the registry's
            parent chain (which is how version ordering is recovered -- parsing
            order out of a version label is forbidden, Rules.md AP-13).
        ledger: Timing ledger to append to; one is created when absent.

    Returns:
        An :class:`IndexReport` carrying the written
        :class:`~axiom.schema.VersionManifest`, the ledger, and every
        degradation taken -- both the whole-step failures this function catches
        itself and the *ladder rungs* taken inside the chunker, the embedder,
        the dense builder and the sparse builder, which are collected by
        listening on the ``axiom`` logger
        (:func:`~axiom.core.logging.capture_degradations`). Without that, a
        build that fell all the way back to the regex chunker, the hash embedder
        and pure-python BM25 reported an *empty* degradation list while ten
        WARNINGs scrolled past, and ``axiom index --json`` could not tell an
        operator which ladder had run -- NFR-07 asks for a report, not a log.

        Model identity on the manifest comes from the embedder that *actually
        ran*, not from ``Settings``: the ladder may have degraded to the hash
        embedder, and a manifest that lies about its own vectors makes invariant
        18 unenforceable.

    Raises:
        OSError: Only from the final publish. Appflow.md Flow 1 step 6 is one of
            the three places raising is correct: a half-written index tree is a
            state the next reader must refuse, not tolerate.

    A build never aborts because one artefact failed. A dense build that raises
    leaves the version with a sparse and a structural index and a logged
    degradation; the query path then drops the dense signal and renormalises.
    """
    with capture_degradations() as rungs:
        report = _build_index_detailed(
            repo_path,
            version_id,
            settings,
            make_active=make_active,
            commit_sha=commit_sha,
            parent_version=parent_version,
            ledger=ledger,
        )
    # The step failures this function recorded come first: they are the reason a
    # signal is missing entirely, which outranks "a cheaper rung answered".
    merged = list(dict.fromkeys([*report.degradations, *rungs]))
    return replace(report, degradations=merged)


def build_index(
    repo_path: str | Path,
    version_id: str,
    settings: Settings | None = None,
    *,
    make_active: bool = True,
    commit_sha: str | None = None,
    parent_version: str | None = None,
    ledger: TimingLedger | None = None,
) -> VersionManifest:
    """:func:`build_index_detailed`, returning only the manifest.

    The signature the CLI and Appflow.md Flow 1 name. Callers that want the
    per-stage timings or the list of degradations taken -- the ``axiom index
    --json`` summary, above all -- should call
    :func:`build_index_detailed` instead of reconstructing them.
    """
    return build_index_detailed(
        repo_path,
        version_id,
        settings,
        make_active=make_active,
        commit_sha=commit_sha,
        parent_version=parent_version,
        ledger=ledger,
    ).manifest


# ---------------------------------------------------------------------------
# Online path: the fan-out backend
# ---------------------------------------------------------------------------


class _VersionIndex:
    """One version's three retrievers and its chunk corpus, all opened lazily.

    Lazy because a query that never reaches the structural signal should not pay
    for a SQLite connection, and because construction must not be able to fail:
    a missing index becomes an absent signal at query time, not an exception at
    wiring time (Rules.md Rule 3).
    """

    def __init__(self, version_id: str, directory: Path, settings: Settings) -> None:
        self.version_id = version_id
        self.directory = directory
        self.settings = settings
        self._dense: Any | None = None
        self._sparse: Any | None = None
        self._structural: Any | None = None
        self._chunks: dict[str, Chunk] | None = None
        # :meth:`IndexBackend.fan_out` submits one task per (version, sub-query,
        # signal), so several threads reach the same accessor at the same instant
        # on the first query. Unguarded, each would build its own retriever and
        # each would then pay its own index load -- and, worse, they would race
        # inside that load. One lock per version, held only across construction.
        self._lock = threading.Lock()

    # -- lazily constructed retrievers -------------------------------------

    def dense(self) -> Any:
        if self._dense is None:
            from axiom.retrieval.dense import DenseRetriever

            with self._lock:
                if self._dense is None:
                    self._dense = DenseRetriever(self.directory, self.settings)
        return self._dense

    def sparse(self) -> Any:
        if self._sparse is None:
            from axiom.retrieval.sparse import SparseRetriever

            with self._lock:
                if self._sparse is None:
                    self._sparse = SparseRetriever(self.directory, self.settings)
        return self._sparse

    def structural(self) -> Any:
        if self._structural is None:
            from axiom.retrieval.structural import StructuralRetriever

            with self._lock:
                if self._structural is None:
                    self._structural = StructuralRetriever(self.directory, self.settings)
        return self._structural

    def chunks(self) -> dict[str, Chunk]:
        """``chunk_id -> Chunk`` for this version, read once per process."""
        if self._chunks is None:
            with self._lock:
                if self._chunks is None:
                    path = self.directory / mf.chunks_path(self.settings, self.version_id).name
                    self._chunks = {chunk.chunk_id: chunk for chunk in mf.read_chunks(path)}
        return self._chunks

    def close(self) -> None:
        """Release handles. Suppressed because a failed close must not fail a query."""
        for retriever in (self._dense, self._structural):
            if retriever is not None:
                with contextlib.suppress(Exception):
                    retriever.close()


class IndexBackend:
    """The :class:`~axiom.agent.loop.RetrievalBackend` the agent loop drives.

    Owns the concurrency. One task per ``(version, sub-query, signal)`` triple
    goes onto a bounded thread pool; each task is individually wrapped, so a
    retriever that raises contributes an empty list and a logged degradation
    rather than propagating out of ``fan_out`` and killing the query.

    Results are recombined in a fixed order -- versions as the registry lists
    them, sub-queries as the planner emitted them, signals in ``SIGNAL_ORDER`` --
    so two runs of the same query produce byte-identical candidate lists even
    though the threads finish in whatever order the OS scheduled (NFR-08).
    """

    def __init__(self, versions: Sequence[_VersionIndex], settings: Settings) -> None:
        self.versions = list(versions)
        self.settings = settings
        self.degradations: list[str] = []
        #: ``chunk_id -> human-readable graph evidence``, harvested from the
        #: structural retriever so the formatter can say *why* (FR-14). It
        #: cannot ride on the frozen ``ScoredChunk``, and the structural module
        #: returns it alongside for exactly this reason.
        self.evidence: dict[str, str] = {}

    # -- one (version, query, signal) task ---------------------------------

    def _run_signal(
        self,
        signal: SignalKind,
        version: _VersionIndex,
        text: str,
        plan: QueryPlan,
    ) -> list[ScoredChunk]:
        try:
            if signal is SignalKind.DENSE:
                if not self.settings.dense_enabled:
                    return []
                return list(version.dense().search(text, plan))
            if signal is SignalKind.SPARSE:
                if not self.settings.sparse_enabled:
                    return []
                return list(version.sparse().search(text, plan))
            if not self.settings.structural_enabled:
                return []
            hits, evidence = version.structural().search_with_evidence(text, plan)
            self.evidence.update(evidence)
            return list(hits)
        except Exception as exc:
            reason = f"{signal.value} retrieval raised {type(exc).__name__}: {exc}"
            log_degradation(
                _LOG,
                f"pipeline.fan_out:{signal.value}",
                reason,
                "signal absent; fusion renormalises the remaining weights",
            )
            self.degradations.append(reason)
            return []

    def fan_out(self, plan: QueryPlan) -> Mapping[SignalKind, Sequence[ScoredChunk]]:
        """Run every signal, for every sub-query, over every selected version."""
        queries = list(plan.effective_queries)
        tasks: list[tuple[tuple[int, int], SignalKind, _VersionIndex, str]] = []
        for version_index, version in enumerate(self.versions):
            for query_index, text in enumerate(queries):
                for signal in SIGNAL_ORDER:
                    tasks.append(((version_index, query_index), signal, version, text))
        if not tasks:
            return {}

        workers = max(1, min(len(tasks), MAX_FANOUT_WORKERS))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="axiom-fanout") as pool:
            futures = [
                pool.submit(self._run_signal, signal, version, text, plan)
                for _, signal, version, text in tasks
            ]
            outputs = [future.result() for future in futures]

        # Recombine deterministically: one mapping per (version, sub-query),
        # then a per-signal RRF merge across all of them. A single version and a
        # single sub-query short-circuits through merge_ranked_lists untouched.
        per_group: dict[tuple[int, int], dict[SignalKind, list[ScoredChunk]]] = {}
        for (key, signal, _version, _text), entries in zip(tasks, outputs, strict=True):
            per_group.setdefault(key, {})[signal] = entries
        grouped = [per_group[key] for key in sorted(per_group)]

        return merge_signal_results(grouped, self.settings)

    def hydrate(self, chunk_ids: Sequence[str]) -> Mapping[str, Chunk]:
        """Resolve ids to chunk bodies, searching versions in registry order."""
        wanted = set(chunk_ids)
        resolved: dict[str, Chunk] = {}
        for version in self.versions:
            if not wanted:
                break
            corpus = version.chunks()
            for chunk_id in list(wanted):
                chunk = corpus.get(chunk_id)
                if chunk is not None:
                    resolved[chunk_id] = chunk
                    wanted.discard(chunk_id)
        if wanted:
            log_degradation(
                _LOG,
                "pipeline.hydrate",
                f"{len(wanted)} fused chunk_id(s) absent from chunks.jsonl",
                "those candidates keep rerank_score=None and are dropped from the output",
            )
        return resolved

    def close(self) -> None:
        for version in self.versions:
            version.close()


# ---------------------------------------------------------------------------
# Online path: formatting
# ---------------------------------------------------------------------------


def _matched_terms(chunk: Chunk, plan: QueryPlan) -> list[str]:
    """Which of the plan's identifiers and expansion terms the chunk contains.

    Substring containment rather than tokenisation: the point is a human-readable
    ``match_reason``, and "the chunk mentions ``handleDeeplink``" is true whether
    the token stream split it or not. Order follows the plan, so the explanation
    is stable across runs.
    """
    haystack = f"{chunk.metadata.symbol or ''} {chunk.text}".lower()
    terms: list[str] = []
    for term in list(plan.extracted_identifiers) + list(plan.expansion_terms):
        lowered = term.lower()
        if lowered and lowered in haystack and term not in terms:
            terms.append(term)
    return terms[:4]


def _signal_phrase(signal: SignalKind, rank: int) -> str:
    """How one signal's contribution reads in a ``match_reason``."""
    if signal is SignalKind.DENSE:
        return f"dense rank {rank}"
    if signal is SignalKind.SPARSE:
        return f"sparse rank {rank}"
    return f"structural rank {rank}"


def match_reason(
    result: FusedResult,
    chunk: Chunk,
    plan: QueryPlan,
    evidence: Mapping[str, str],
) -> str:
    """One sentence naming the dominant signal and the concrete evidence (FR-14).

    "dense: rank 3" is not an explanation, it is a restatement of the ranking.
    What US-6 asks for is the thing a reviewer can check against the file: which
    identifiers matched, or which call-graph edge fired. So the structural branch
    quotes the retriever's own evidence string -- "handleDeeplink calls
    preprocessInput before resolveTool" -- and the lexical branch names the terms
    that were actually present in the chunk.

    Corroboration from the other signals is appended rather than merged: knowing
    that the chunk was also the sparse leg's rank 1 is the difference between one
    retriever's opinion and three agreeing.
    """
    dominant = result.dominant_signal
    rank = result.contributions[dominant]
    symbol = chunk.metadata.symbol
    where = f"{chunk.metadata.kind.value} {symbol}" if symbol else chunk.metadata.kind.value

    if dominant is SignalKind.STRUCTURAL:
        detail = evidence.get(result.chunk_id)
        if not detail:
            identifiers = ", ".join(plan.extracted_identifiers[:3]) or "the query identifiers"
            detail = f"call-graph match for {identifiers}"
        head = f"structural: {detail} (rank {rank} in the call graph)"
    elif dominant is SignalKind.SPARSE:
        terms = _matched_terms(chunk, plan)
        subject = ", ".join(terms) if terms else "the query terms"
        head = f"sparse: {where} matches {subject} lexically (BM25 rank {rank})"
    else:
        head = f"dense: {where} is a nearest neighbour of the query embedding (rank {rank})"

    corroboration = [
        _signal_phrase(signal, result.contributions[signal])
        for signal in SIGNAL_ORDER
        if signal in result.contributions and signal is not dominant
    ]
    if corroboration:
        head = f"{head}; corroborated at {' and '.join(corroboration)}"
    return head


def optimization_hint(chunk: Chunk) -> str | None:
    """A cheap, deterministic note about the surfaced code, or ``None``.

    Schema.md calls this a bonus-feature field and gives the canonical example:
    an ``await`` inside a ``for`` loop, which serialises requests that
    ``Promise.all`` would overlap. Two checks only, both pure text scans with no
    parser behind them, because a wrong hint on stage is worse than no hint --
    and never a rewrite (NG: the system retrieves, it does not generate code).
    """
    text = chunk.text
    lowered = text.lower()
    if "await" in lowered and ("for (" in lowered or "for(" in lowered):
        return "awaits inside a for-loop; consider Promise.all for independent calls"
    if ".foreach(" in lowered and "async" in lowered:
        return "async callback inside forEach; the loop does not await it"
    return None


# ---------------------------------------------------------------------------
# Online path: query
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResponse:
    """Everything one query produced. The body of ``POST /v1/query`` (API.md 3.1)."""

    results: list[RetrievalResult]
    query_plan: QueryPlan
    passes_used: int
    profile: str
    stop_reason: str
    score_field: str
    version_ids: list[str]
    degradations: list[str] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    ledger: TimingLedger = field(default_factory=TimingLedger)
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """Serialise for ``--json`` and for the API layer.

        ``elapsed_ms`` is the single field excluded from the determinism
        contract (TestPlan.md TC-066); every other value here is a pure function
        of the query, the config and the index.
        """
        return {
            "results": [result.model_dump(mode="json") for result in self.results],
            "query_plan": self.query_plan.model_dump(mode="json"),
            "passes_used": self.passes_used,
            "profile": self.profile,
            "stop_reason": self.stop_reason,
            "score_field": self.score_field,
            "version_ids": list(self.version_ids),
            "degradations": list(self.degradations),
            "warnings": [dict(w) for w in self.warnings],
            "timings": self.ledger.as_dict(),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


def _select_versions(
    settings: Settings, version_id: str | None, all_versions: bool
) -> list[_VersionIndex]:
    """Resolve which version directories this query reads (FR-20).

    Version scoping is structural, not a filter: a query against ``v1`` opens
    ``.axiom/index/v1/`` and can physically not see another version's rows.
    """
    if all_versions:
        ids = [
            candidate
            for candidate in mf.load_registry(settings).newest_first()
            if mf.manifest_path(settings, candidate).is_file()
        ]
        if not ids:
            raise IndexNotFoundError(f"no built index under {mf.index_root(settings)}")
    else:
        ids = [mf.resolve_version(settings, version_id)]
    return [_VersionIndex(vid, mf.version_dir(settings, vid), settings) for vid in ids]


def _families_by_chunk(
    results: Sequence[FusedResult],
    chunks: Mapping[str, Chunk],
    versions: Sequence[_VersionIndex],
    settings: Settings,
) -> dict[str, SnippetFamily]:
    """Map each result chunk to its snippet family, for an ``--all-versions`` query (FR-21).

    Only the families the results actually belong to are built: grouping the
    whole corpus would cost a full cross-version pass for a bonus that touches at
    most ten chunks. Candidate members are filtered by the family key --
    ``(symbol, file_path)``, which TC-082 requires to match *in addition* to
    similarity, never similarity alone.

    Embeddings come from the content-addressed blob store, keyed by
    ``content_hash``. When numpy or the blobs are absent every family degrades to
    a single member, :meth:`SnippetFamily.is_multi_version` is false, and the
    bonus is exactly neutral -- which is the right answer, not a silent 1.0
    pretending the comparison happened.

    Families are returned rather than precomputed multipliers because
    ``apply_stability_bonus`` clamps its result into ``[0, 1]``: handing it a
    base of 1.0 to "extract the multiplier" would clamp 1.10 back to 1.00 and
    silently disable the whole feature. The bonus has to be applied to the real
    score, by the one function that owns the formula (Rules.md AP-07).
    """
    from axiom.versioning.evolutionary import build_families

    keys = {
        (chunks[r.chunk_id].metadata.symbol, chunks[r.chunk_id].location.file_path)
        for r in results
        if r.chunk_id in chunks
    }
    if not keys:
        return {}

    by_version: dict[str, list[Chunk]] = {}
    embeddings: dict[str, Any] = {}
    for version in versions:
        members = [
            chunk
            for chunk in version.chunks().values()
            if (chunk.metadata.symbol, chunk.location.file_path) in keys
        ]
        if not members:
            continue
        by_version[version.version_id] = members
        manifest = mf.try_read_manifest(settings, version.version_id)
        dim = manifest.embedding_dim if manifest is not None else None
        for chunk in members:
            if chunk.content_hash in embeddings:
                continue
            try:
                vector = mf.load_blob(settings, chunk.content_hash, expected_dim=dim)
            except Exception as exc:
                log_degradation(
                    _LOG,
                    "pipeline.evolutionary",
                    f"blob {chunk.content_hash} unusable: {type(exc).__name__}: {exc}",
                    "chunk excluded from family grouping",
                )
                continue
            if vector is not None:
                # ``evolutionary.cosine`` takes any ``Sequence[float]``, ndarray
                # included, so the blob goes straight in. It did not always: the
                # function tested emptiness with ``not left``, which an ndarray
                # refuses to answer, and every call site had to convert.
                embeddings[chunk.content_hash] = vector

    families = build_families(
        by_version,
        embeddings or None,
        settings,
        version_order=[v.version_id for v in versions],
        total_versions=len(versions),
        with_diffs=False,
    )
    by_chunk: dict[str, SnippetFamily] = {}
    for family in families:
        for member in family.members:
            by_chunk[member.chunk_id] = family
    return by_chunk


def _format_results(
    outcome: agent_loop.AgentOutcome,
    evidence: Mapping[str, str],
    families: Mapping[str, SnippetFamily],
    settings: Settings,
    top_k: int,
) -> list[RetrievalResult]:
    """Turn the winning pass's fused results into user-facing hits (FR-14).

    A candidate that could not be hydrated is dropped rather than emitted with a
    placeholder body: ``RetrievalResult.chunk`` is the chunk, and there is no
    honest way to fill it in.
    """
    from axiom.versioning.evolutionary import apply_stability_bonus

    formatted: list[RetrievalResult] = []
    for result in outcome.results:
        chunk = outcome.chunks.get(result.chunk_id)
        if chunk is None:
            continue
        score = result.final_score
        family = families.get(result.chunk_id)
        if family is not None:
            score = apply_stability_bonus(score, family, settings)
        formatted.append(
            RetrievalResult(
                chunk=chunk,
                score=min(max(score, 0.0), 1.0),
                match_reason=match_reason(result, chunk, outcome.plan, evidence),
                signals=dict(result.contributions),
                optimization_hint=optimization_hint(chunk),
            )
        )
    if families:
        # The bonus can reorder, so the final list is re-sorted on the score the
        # user is shown. Ties break on ascending chunk_id, as everywhere else.
        formatted.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
    return formatted[:top_k]


def query(
    text: str,
    settings: Settings | None = None,
    version_id: str | None = None,
    top_k: int | None = None,
    all_versions: bool = False,
    *,
    query_type: QueryType | None = None,
    ledger: TimingLedger | None = None,
) -> QueryResponse:
    """Answer one query end to end (Appflow.md Flows 2, 3, 6, 7).

    Args:
        text: The user's query, verbatim.
        settings: Active configuration; the ambient profile when omitted.
        version_id: Query one specific version (FR-20). ``None`` uses the
            registry's active version.
        top_k: Final result count; defaults to ``Settings.top_k_default``.
        all_versions: Search every registered version and, when
            ``Settings.evolutionary_enabled``, apply the stability bonus (FR-21).
        query_type: Force the classification instead of running it, for
            ``axiom query --type structural`` (US-3) and the API's ``query_type``
            field (API.md section 3.1). The forced type reaches the plan, and
            therefore the weight vector, exactly as a classified one would --
            it is the classifier that is skipped, not the planner.
        ledger: Timing ledger to append to; one is created when absent.

    Returns:
        A :class:`QueryResponse`. An index that does not exist is the one thing
        that raises -- :class:`~axiom.core.errors.IndexNotFoundError`, because
        "there is nothing to search" is not a ranking a caller can degrade into.
        Everything else degrades: an empty query, a dead signal, a reranker that
        will not load, an exhausted agent budget.
    """
    resolved = settings if settings is not None else get_settings()
    ledger = ledger if ledger is not None else TimingLedger()
    started = perf_counter()
    width = top_k if top_k is not None else resolved.top_k_default

    forced_plan: QueryPlan | None = None
    if query_type is not None:
        from axiom.agent.planner import build_plan

        forced_plan = build_plan(text, resolved, query_type=query_type)

    versions = _select_versions(resolved, version_id, all_versions)
    backend = IndexBackend(versions, resolved)
    try:
        outcome = agent_loop.run(
            text, backend, resolved, plan=forced_plan, ledger=ledger, top_k=width
        )

        families: dict[str, SnippetFamily] = {}
        if all_versions and resolved.evolutionary_enabled and outcome.results:
            with ledger.measure("evolutionary") as detail:
                families = _families_by_chunk(outcome.results, outcome.chunks, versions, resolved)
                detail["chunks_in_families"] = len(families)
                detail["multi_version_families"] = len(
                    {f.family_id for f in families.values() if f.is_multi_version}
                )

        with ledger.measure(STAGE_FORMAT) as detail:
            results = _format_results(outcome, backend.evidence, families, resolved, width)
            detail["returned"] = len(results)
    finally:
        backend.close()

    warnings: list[dict[str, str]] = []
    if outcome.stop_reason == agent_loop.STOP_BUDGET:
        warnings.append(
            {
                "code": "AGENT_BUDGET_EXCEEDED",
                "detail": (
                    f"agent wall-clock budget of {resolved.agent_wall_clock_ms} ms was "
                    f"reached; returning the best of {outcome.passes_used} pass(es)"
                ),
            }
        )
    if outcome.stop_reason == agent_loop.STOP_EMPTY_QUERY:
        warnings.append({"code": "EMPTY_QUERY", "detail": NO_CANDIDATES})
    if not results:
        warnings.append({"code": "NO_RESULTS", "detail": NO_CANDIDATES})

    degradations = list(dict.fromkeys(outcome.degradations + backend.degradations))
    elapsed_ms = (perf_counter() - started) * 1000.0
    _LOG.info(
        "query answered: %d result(s) in %.0f ms over %d pass(es)",
        len(results),
        elapsed_ms,
        outcome.passes_used,
        extra={
            "axiom_extra": {
                "stage": "query",
                "results": len(results),
                "passes_used": outcome.passes_used,
                "stop_reason": outcome.stop_reason,
                "profile": resolved.profile,
                "versions": [v.version_id for v in versions],
            }
        },
    )
    return QueryResponse(
        results=results,
        query_plan=outcome.plan,
        passes_used=outcome.passes_used,
        profile=resolved.profile,
        stop_reason=outcome.stop_reason,
        score_field=outcome.score_field,
        version_ids=[v.version_id for v in versions],
        degradations=degradations,
        warnings=warnings,
        ledger=ledger,
        elapsed_ms=elapsed_ms,
    )


def format_ranked_lines(response: QueryResponse) -> list[str]:
    """``rank. file:line-line  score  symbol`` lines, for the CLI's text mode."""
    lines: list[str] = []
    for position, result in enumerate(response.results, start=1):
        symbol = result.chunk.metadata.symbol or result.chunk.metadata.kind.value
        lines.append(
            f"{position:2d}. {result.chunk.location.as_ref():<40} {result.score:.4f}  {symbol}"
        )
        lines.append(f"    {result.match_reason}")
    return lines


def list_families(
    settings: Settings,
    *,
    version: str | None = None,
    multi_only: bool = False,
    with_diffs: bool = False,
) -> tuple[list[SnippetFamily], list[str]]:
    """Every snippet family across the indexed versions (FR-21, browse path).

    Lifted out of ``axiom families``'s command body so the HTTP surface can
    answer the same question. It was fifty lines of index loading sitting inside
    a Typer callback, which meant ``GET /v1/families`` could only have existed as
    a second copy of it -- and a second copy of "which versions count, and which
    blobs were loadable" is exactly the kind of divergence that makes the CLI and
    the API disagree about what the corpus contains.

    Grouping needs vectors: two same-named functions in one file are one family
    only if they are also near-identical (TC-082). Those come from the
    content-addressed blob store, and an unreadable blob is logged and skipped
    rather than raised -- every family then degrades to a single member and says
    so, which is the honest answer rather than a merge that never compared
    anything (Rule 3).

    Args:
        settings: Resolved configuration; names the index root.
        version: Restrict to one version id. ``None`` uses every built version,
            newest first.
        multi_only: Drop families that exist in only one version.
        with_diffs: Attach per-transition unified diffs. Costs real work, so it
            is off unless asked for.

    Returns:
        ``(families, version_ids)`` -- the families, and the versions they were
        computed over, in the order that defined ``stability``'s denominator.

    Raises:
        IndexNotFoundError: No built index, or the named version has no manifest.
    """
    from axiom.indexing import manifest as mf
    from axiom.versioning.evolutionary import build_families

    if version is not None:
        version_ids = [mf.resolve_version(settings, version)]
    else:
        version_ids = [
            candidate
            for candidate in mf.load_registry(settings).newest_first()
            if mf.manifest_path(settings, candidate).is_file()
        ]
    if not version_ids:
        raise IndexNotFoundError(f"no built index under {mf.index_root(settings)}")

    chunks_by_version: dict[str, list[Chunk]] = {}
    embeddings: dict[str, list[float]] = {}
    for version_id in version_ids:
        members = mf.read_version_chunks(settings, version_id)
        if not members:
            continue
        chunks_by_version[version_id] = members
        manifest = mf.try_read_manifest(settings, version_id)
        dim = manifest.embedding_dim if manifest is not None else None
        for chunk in members:
            if chunk.content_hash in embeddings:
                continue
            try:
                vector = mf.load_blob(settings, chunk.content_hash, expected_dim=dim)
            except Exception as exc:
                _LOG.warning("blob %s unusable: %s", chunk.content_hash, exc)
                continue
            if vector is not None:
                # evolutionary.cosine takes a Sequence[float]; a numpy array
                # cannot answer its truthiness test, so convert at the call site
                # rather than loosening a frozen module's types.
                embeddings[chunk.content_hash] = [float(value) for value in vector]

    families = build_families(
        chunks_by_version,
        embeddings or None,
        settings,
        version_order=version_ids,
        total_versions=len(version_ids),
        with_diffs=with_diffs,
    )
    if multi_only:
        families = [family for family in families if family.is_multi_version]
    return families, version_ids


def iter_degradations(response: QueryResponse) -> Iterable[str]:
    """Deduplicated degradation notes, for the UI's "what fell back" panel."""
    return dict.fromkeys(response.degradations)


__all__ = [
    "MAX_FANOUT_WORKERS",
    "NO_CANDIDATES",
    "IndexBackend",
    "IndexReport",
    "QueryResponse",
    "build_index",
    "build_index_detailed",
    "format_ranked_lines",
    "iter_degradations",
    "list_families",
    "match_reason",
    "optimization_hint",
    "query",
]
