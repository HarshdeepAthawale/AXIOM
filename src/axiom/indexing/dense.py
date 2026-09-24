"""Dense index builder: chunks -> vectors -> ``dense.faiss`` + ``dense.idmap.json``.

Implements FR-05. The index kind switches on corpus size at
``Settings.faiss_ivf_threshold`` (ADR-004): exact ``IndexFlatIP`` below it,
approximate ``IndexIVFPQ`` at or above. Both the 8,765-vector APPS corpus and
the ~10k-chunk demo repo sit on the flat side; IVF-PQ is scale headroom that
must nonetheless exist and be seeded, because an unseeded k-means makes two
builds of the same corpus rank differently (NFR-08).

When ``faiss`` is not importable the builder degrades to a plain ``dense.npy``
matrix searched by brute-force inner product. On a 10k-chunk corpus that is a
single 10k x 1024 matmul -- a few milliseconds, entirely adequate for the demo
-- so the dense signal survives a machine with no FAISS wheel rather than
vanishing from fusion. The chosen backend is recorded in the idmap so the read
side never has to guess.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from axiom.core.logging import get_logger, log_degradation
from axiom.indexing.embedder import BlobCache, Embedder, embed_chunks, load_embedder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from axiom.config import Settings
    from axiom.schema import Chunk

_LOG = get_logger("indexing.dense")

#: On-disk names, locked by _CONTRACT.md section 6 / Schema.md section 14.
DENSE_INDEX_FILENAME: Final = "dense.faiss"
DENSE_IDMAP_FILENAME: Final = "dense.idmap.json"

#: Degraded-backend artefact. Not in the locked layout because the locked layout
#: assumes FAISS is installed; its presence *is* the degradation marker.
DENSE_MATRIX_FILENAME: Final = "dense.npy"

#: ``dense.idmap.json`` generation (Schema.md section 14.4).
IDMAP_SCHEMA_VERSION: Final = 1

#: Backend tags written into the idmap's ``backend`` key.
BACKEND_FAISS: Final = "faiss"
BACKEND_NUMPY: Final = "numpy"

#: ``VersionManifest.index_kind`` vocabulary (pattern-enforced by the schema).
INDEX_KIND_FLAT: Final = "flat_ip"
INDEX_KIND_IVF_PQ: Final = "ivf_pq"

#: IVF-PQ shape. TechSpecifications.md section 5 and TestPlan.md TC-041 say
#: nlist/m/nbits/nprobe "live in config", but ``Settings`` declares none of them
#: (see the report in the module header of retrieval/dense.py). These are the
#: documented defaults, read through ``getattr`` so a future Settings field
#: wins automatically, and derived from corpus size rather than guessed.
DEFAULT_PQ_NBITS: Final = 8
DEFAULT_MAX_PQ_SUBQUANTIZERS: Final = 64
#: FAISS wants >= 39 training points per centroid before it stops warning.
_MIN_POINTS_PER_CENTROID: Final = 39


@dataclass(frozen=True)
class DenseBuildResult:
    """What the dense build produced, for the manifest writer and the CLI summary.

    ``embedding_model`` and ``embedding_dim`` are the embedder that *actually*
    ran, not the configured one: the ladder may have degraded, and a manifest
    that records the configured model would make Schema.md invariant 18 (model
    identity is enforced, not assumed) a lie.
    """

    index_kind: str
    backend: str
    embedding_model: str
    embedding_dim: int
    count: int
    cache_hits: int
    embedded: int
    index_path: Path
    idmap_path: Path


def choose_index_kind(count: int, settings: Settings) -> str:
    """``flat_ip`` below ``faiss_ivf_threshold`` vectors, ``ivf_pq`` at or above it."""
    return INDEX_KIND_FLAT if count < settings.faiss_ivf_threshold else INDEX_KIND_IVF_PQ


def _ivf_parameters(count: int, dim: int, settings: Settings) -> tuple[int, int, int]:
    """Derive ``(nlist, m, nbits)`` for an IVF-PQ of ``count`` vectors at ``dim``.

    ``nlist`` follows the standard FAISS heuristic of ``4*sqrt(n)`` centroids,
    capped so that every centroid still sees the ~39 training points FAISS wants;
    ``m`` is the largest divisor of ``dim`` at or below
    :data:`DEFAULT_MAX_PQ_SUBQUANTIZERS`, since PQ requires ``dim % m == 0``.
    Both are overridable by ``Settings`` fields the moment they exist.
    """
    nlist = int(getattr(settings, "faiss_nlist", 0)) or max(
        1, min(int(4 * math.sqrt(max(count, 1))), max(1, count // _MIN_POINTS_PER_CENTROID))
    )
    configured_m = int(getattr(settings, "faiss_pq_m", 0))
    if configured_m and dim % configured_m == 0:
        m = configured_m
    else:
        m = next(
            (c for c in range(min(DEFAULT_MAX_PQ_SUBQUANTIZERS, dim), 0, -1) if dim % c == 0),
            1,
        )
    nbits = int(getattr(settings, "faiss_pq_nbits", DEFAULT_PQ_NBITS))
    return nlist, m, nbits


def _build_faiss_index(vectors: np.ndarray, index_kind: str, settings: Settings) -> Any:
    """Construct and populate the FAISS index. Caller has already vetted the import."""
    import faiss

    count, dim = vectors.shape
    if settings.num_threads > 0:
        # Thread count changes float reduction order, which flips ties (NFR-08).
        faiss.omp_set_num_threads(settings.num_threads)

    if index_kind == INDEX_KIND_FLAT or count == 0:
        index = faiss.IndexFlatIP(dim)
    else:
        nlist, m, nbits = _ivf_parameters(count, dim, settings)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        # Seed both clustering stages: the coarse quantizer's k-means and the
        # PQ codebook's. An unseeded training run makes a rebuilt index a
        # different index, which Rules.md section 5.3 forbids reporting against.
        for params in (getattr(index, "cp", None), getattr(getattr(index, "pq", None), "cp", None)):
            if params is not None:
                params.seed = settings.seed
        index.train(vectors)
        index.nprobe = int(getattr(settings, "faiss_nprobe", 16))
    if count:
        index.add(vectors)
    return index


def _write_idmap(
    path: Path,
    chunk_ids: Sequence[str],
    *,
    embedding_model: str,
    embedding_dim: int,
    index_kind: str,
    backend: str,
) -> None:
    """Write ``dense.idmap.json``: FAISS row -> ``chunk_id``, positional (Schema.md 14.4).

    A positional array rather than an object keyed by stringified ints, so the
    row invariant is structural rather than conventional. ``backend`` is Axiom's
    one addition to the locked key set: without it the reader cannot tell a
    FAISS index from the degraded numpy matrix except by probing the filesystem.
    """
    payload = {
        "schema_version": IDMAP_SCHEMA_VERSION,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "index_kind": index_kind,
        "backend": backend,
        "count": len(chunk_ids),
        "rows": list(chunk_ids),
    }
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_dense_index(
    chunks: Sequence[Chunk],
    settings: Settings,
    out_dir: Path,
    *,
    embedder: Embedder | None = None,
    cache: BlobCache | None = None,
) -> DenseBuildResult:
    """Embed ``chunks``, build the vector index, and persist both artefacts.

    Row ``i`` of the index is ``chunks[i]`` and ``idmap.rows[i]`` is
    ``chunks[i].chunk_id``: the caller's chunk order is preserved verbatim
    because it is the same order ``chunks.jsonl`` and the bm25s corpus use
    (Schema.md invariant 12). This function does not reorder, deduplicate, or
    filter its input -- an empty corpus produces a valid empty index with
    ``ntotal == 0`` (TestPlan.md TC-078), not an error.

    Args:
        chunks: The corpus, in canonical ``(file_path, start_line)`` order.
        settings: Active settings; supplies batch size, seed, and the IVF threshold.
        out_dir: ``.axiom/index/<version_id>/``. Created if absent.
        embedder: Injected embedder, for tests and for sharing one loaded model
            across the dense and evolutionary builds. Defaults to the ladder.
        cache: Injected blob store. Defaults to ``.axiom/blobs`` under
            ``Settings.index_root``.

    Returns:
        A :class:`DenseBuildResult` carrying everything the manifest writer
        needs -- crucially the *actual* embedder identity and dim, plus the
        index kind, which ``VersionManifest`` records verbatim.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    active = embedder if embedder is not None else load_embedder(settings)
    run = embed_chunks(chunks, active, settings, cache)
    vectors = np.ascontiguousarray(run.vectors, dtype=np.float32)
    count = int(vectors.shape[0])
    index_kind = choose_index_kind(count, settings)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    index_path = out_dir / DENSE_INDEX_FILENAME
    backend = BACKEND_FAISS
    try:
        import faiss

        index = _build_faiss_index(vectors, index_kind, settings)
        faiss.write_index(index, str(index_path))
    except Exception as exc:
        log_degradation(
            _LOG,
            "indexing.dense:build_dense_index",
            f"faiss unusable: {type(exc).__name__}: {exc}",
            f"{DENSE_MATRIX_FILENAME} + brute-force inner product",
        )
        backend = BACKEND_NUMPY
        index_kind = INDEX_KIND_FLAT  # brute force is exact; it is flat by definition
        index_path = out_dir / DENSE_MATRIX_FILENAME
        tmp = index_path.with_name(f"{index_path.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        os.replace(tmp, index_path)

    idmap_path = out_dir / DENSE_IDMAP_FILENAME
    _write_idmap(
        idmap_path,
        chunk_ids,
        embedding_model=run.model_id,
        embedding_dim=run.dim,
        index_kind=index_kind,
        backend=backend,
    )
    _LOG.info(
        "dense index built: %d vectors, kind=%s, backend=%s, model=%s dim=%d",
        count,
        index_kind,
        backend,
        run.model_id,
        run.dim,
        extra={
            "axiom_extra": {
                "event": "dense_index_built",
                "count": count,
                "index_kind": index_kind,
                "backend": backend,
                "cache_hits": run.cache_hits,
                "embedded": run.embedded,
            }
        },
    )
    return DenseBuildResult(
        index_kind=index_kind,
        backend=backend,
        embedding_model=run.model_id,
        embedding_dim=run.dim,
        count=count,
        cache_hits=run.cache_hits,
        embedded=run.embedded,
        index_path=index_path,
        idmap_path=idmap_path,
    )


__all__ = [
    "BACKEND_FAISS",
    "BACKEND_NUMPY",
    "DENSE_IDMAP_FILENAME",
    "DENSE_INDEX_FILENAME",
    "DENSE_MATRIX_FILENAME",
    "IDMAP_SCHEMA_VERSION",
    "INDEX_KIND_FLAT",
    "INDEX_KIND_IVF_PQ",
    "DenseBuildResult",
    "build_dense_index",
    "choose_index_kind",
]
