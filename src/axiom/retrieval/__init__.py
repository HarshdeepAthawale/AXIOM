"""Stage 2 and 3: the three first-stage signals and the fusion that combines them.

Layout mirrors the signals themselves -- :mod:`~axiom.retrieval.dense` (FAISS or
a numpy matrix), :mod:`~axiom.retrieval.sparse` (bm25s or a pure-Python postings
list), :mod:`~axiom.retrieval.structural` (SQLite graph traversal) -- plus
:mod:`~axiom.retrieval.fusion`, which is pure arithmetic over their ranks.

Every retriever obeys one contract: **an unavailable signal returns ``[]``, never
an exception and never zero-scored filler.** Fusion then drops that signal's
weight and renormalises the rest (TechSpecifications.md section 5.1.2). That is
what lets the whole pipeline survive a missing FAISS, a missing bm25s, and a
missing structural database simultaneously, which is the configuration NFR-07
requires to keep working.

Fusion and the shared tokenizer are re-exported eagerly -- they pull in nothing
beyond the standard library and the schema. The three retrievers are resolved
lazily through PEP 562, because :mod:`axiom.retrieval.dense` imports numpy and
the embedder ladder at module scope, and ``axiom --help`` has no business paying
for that (Rules.md AP-06). Importing a submodule directly still works exactly as
it always did.
"""

from __future__ import annotations

from typing import Any

from axiom.retrieval.fusion import (
    SIGNAL_ORDER,
    FusionOutcome,
    fuse,
    merge_ranked_lists,
    merge_signal_results,
    plan_with_weights,
    rank_key,
    reciprocal_rank_fusion,
    renormalise_weights,
    signal_width,
)
from axiom.retrieval.tokenizer import DEFAULT_TOKENIZER, CodeTokenizer, tokenize

__all__ = [
    "DEFAULT_TOKENIZER",
    "SIGNAL_ORDER",
    "CodeTokenizer",
    "DenseRetriever",
    "FusionOutcome",
    "SparseRetriever",
    "StructuralRetriever",
    "fuse",
    "merge_ranked_lists",
    "merge_signal_results",
    "plan_with_weights",
    "rank_key",
    "reciprocal_rank_fusion",
    "renormalise_weights",
    "signal_width",
    "tokenize",
]

#: Retriever classes served on first access, keyed to their defining module.
_LAZY: dict[str, str] = {
    "DenseRetriever": "axiom.retrieval.dense",
    "SparseRetriever": "axiom.retrieval.sparse",
    "StructuralRetriever": "axiom.retrieval.structural",
}


def __getattr__(name: str) -> Any:
    """Resolve the retriever classes on first access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """Include the lazily-served names in ``dir()`` and tab completion."""
    return sorted(__all__)
