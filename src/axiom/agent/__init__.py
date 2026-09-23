"""The agent: classify, plan, judge, refine -- and the loop that bounds it.

Four decision components live here, plus the LLM adapter that may optionally
supplement them. All four work with ``AXIOM_LLM_ENABLED=false`` and with no
optional dependency installed; that is not a degraded mode but the default one
(TechSpecifications.md section 3.3, NFR-07).

``loop.py`` is deliberately not imported eagerly. It is the only module here that
pulls in retrieval, fusion, and reranking, so importing it at package scope would
drag the whole pipeline -- and every optional dependency behind it -- into
``import axiom.agent``. The module-level ``__getattr__`` below resolves
``axiom.agent.run`` on first access instead, keeping the bare-install import
path clean (NFR-07).
"""

from __future__ import annotations

from typing import Any

from axiom.agent.classifier import classify, classify_heuristic
from axiom.agent.evaluator import (
    assess_sufficiency,
    is_sufficient,
    next_plan,
    sufficiency_detail,
)
from axiom.agent.planner import (
    build_plan,
    decompose,
    extract_identifiers,
    normalise_query,
    plans_differ,
    refine_plan,
)
from axiom.agent.synonyms import CODE_SYNONYMS

__all__ = [
    "CODE_SYNONYMS",
    "assess_sufficiency",
    "build_plan",
    "classify",
    "classify_heuristic",
    "decompose",
    "extract_identifiers",
    "is_sufficient",
    "next_plan",
    "normalise_query",
    "plans_differ",
    "refine_plan",
    "run",
    "sufficiency_detail",
]

#: Names served lazily from :mod:`axiom.agent.loop`.
_LAZY: dict[str, str] = {"run": "axiom.agent.loop"}


def __getattr__(name: str) -> Any:
    """Resolve loop symbols on first access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
