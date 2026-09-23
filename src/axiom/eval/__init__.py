"""Evaluation harness: the metrics, and the bridge from Axiom to MTEB.

Importing this package pulls in nothing heavier than the standard library and
:mod:`axiom.schema` -- ``mteb``, ``datasets`` and ``numpy``-beyond-basics are
reached for inside the functions that need them (NFR-07). That is what lets
``scripts/run_eval.py`` start, explain itself, and fail with a readable message
on a machine where the optional extras never installed.

The two halves are deliberately separate. :mod:`axiom.eval.metrics` is pure
arithmetic over ``(qrels, run)`` and knows nothing about Axiom;
:mod:`axiom.eval.mteb_adapter` knows about Axiom and about MTEB but computes no
metrics. Neither can quietly corrupt the other, and the reported number can be
re-derived from a saved prediction file without re-running retrieval.
"""

from __future__ import annotations

from axiom.eval.metrics import (
    evaluate_run,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    rank_documents,
    recall_at_k,
    run_coverage,
)

__all__ = [
    "evaluate_run",
    "map_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "rank_documents",
    "recall_at_k",
    "run_coverage",
]
