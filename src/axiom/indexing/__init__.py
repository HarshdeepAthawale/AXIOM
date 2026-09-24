"""The offline half of the system: chunks in, queryable index artefacts out.

Four builders write into one version directory (Appflow.md Flow 1):

* :mod:`~axiom.indexing.dense` -- embeddings plus ``dense.faiss`` (or
  ``dense.npy`` when faiss is absent) and ``dense.idmap.json``.
* :mod:`~axiom.indexing.sparse` -- ``sparse.bm25s/``, native or pure-Python.
* :mod:`~axiom.indexing.structural` -- ``structural.sqlite``, the call/import graph.
* :mod:`~axiom.indexing.manifest` -- ``manifest.json``, ``chunks.jsonl``,
  ``registry.json``, and the content-addressed blob store. Written last, because
  a version directory without a manifest is by definition an incomplete build
  and the next reader must refuse it (Schema.md invariant 19).

Every name is served lazily through PEP 562. :mod:`axiom.indexing.dense` imports
numpy and the embedder ladder at module scope, and a CLI that only wants to
print ``--help`` -- or a test session that only wants to collect -- must not pay
for that (Rules.md AP-06, NFR-07). ``from axiom.indexing import manifest`` and
``import axiom.indexing.dense`` behave exactly as normal submodule imports; only
the flat re-exports below are deferred.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BlobCache",
    "DenseBuildResult",
    "Embedder",
    "SparseIndexResult",
    "StructuralIndexStats",
    "VersionRecord",
    "build_dense_index",
    "build_manifest",
    "build_sparse_index",
    "build_structural_index",
    "canonical_chunk_order",
    "chunks_path",
    "finalize_version",
    "load_embedder",
    "load_registry",
    "read_chunks",
    "read_manifest",
    "read_version_chunks",
    "register_version",
    "resolve_version",
    "version_dir",
    "write_chunks",
    "write_manifest",
]

#: Flat re-export name -> the module that actually defines it.
_LAZY: dict[str, str] = {
    "BlobCache": "axiom.indexing.embedder",
    "DenseBuildResult": "axiom.indexing.dense",
    "Embedder": "axiom.indexing.embedder",
    "SparseIndexResult": "axiom.indexing.sparse",
    "StructuralIndexStats": "axiom.indexing.structural",
    "VersionRecord": "axiom.indexing.manifest",
    "build_dense_index": "axiom.indexing.dense",
    "build_manifest": "axiom.indexing.manifest",
    "build_sparse_index": "axiom.indexing.sparse",
    "build_structural_index": "axiom.indexing.structural",
    "canonical_chunk_order": "axiom.indexing.manifest",
    "chunks_path": "axiom.indexing.manifest",
    "finalize_version": "axiom.indexing.manifest",
    "load_embedder": "axiom.indexing.embedder",
    "load_registry": "axiom.indexing.manifest",
    "read_chunks": "axiom.indexing.manifest",
    "read_manifest": "axiom.indexing.manifest",
    "read_version_chunks": "axiom.indexing.manifest",
    "register_version": "axiom.indexing.manifest",
    "resolve_version": "axiom.indexing.manifest",
    "version_dir": "axiom.indexing.manifest",
    "write_chunks": "axiom.indexing.manifest",
    "write_manifest": "axiom.indexing.manifest",
}


def __getattr__(name: str) -> Any:
    """Resolve a builder or manifest helper on first access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """Include the lazily-served names in ``dir()`` and tab completion."""
    return sorted(__all__)
