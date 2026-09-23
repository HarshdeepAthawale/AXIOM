"""The HTTP surface: four endpoints over the one pipeline (FR-24, API.md).

``models`` is re-exported eagerly -- it is pure Pydantic over the shared schema,
it is what ``cli.py`` serialises ``--json`` output with, and it costs nothing.
Everything that can touch FastAPI (:func:`create_app`, :func:`serve`,
``build_router``) is resolved lazily through PEP 562, so ``import axiom.api``
succeeds on a bare install and ``axiom --help`` does not pay for a web framework
it is not about to use (Rules.md AP-06, NFR-07).

The API and the CLI are two renderings of one call into :mod:`axiom.pipeline`
and one set of models. That is the mechanism behind FR-24's "typed by the same
Pydantic models as the CLI": not a convention, but the absence of a second code
path that could disagree.
"""

from __future__ import annotations

from typing import Any

from axiom.api.models import (
    ErrorCode,
    ErrorResponse,
    HealthResponse,
    IndexSummary,
    QueryRequest,
    QueryResponse,
    VersionsResponse,
    VersionSummary,
    WarningItem,
)

__all__ = [
    "SERVE_INSTALL_HINT",
    "ApiError",
    "ErrorCode",
    "ErrorResponse",
    "HealthResponse",
    "IndexSummary",
    "QueryRequest",
    "QueryResponse",
    "ServeDependencyError",
    "VersionSummary",
    "VersionsResponse",
    "WarningItem",
    "create_app",
    "get_chunk",
    "health",
    "list_versions",
    "missing_dependencies",
    "run_query",
    "serve",
    "warm_models",
]

#: Names served on first access, keyed to their defining module. ``routes`` is
#: safe to import eagerly in principle -- it holds no optional import at module
#: scope -- but it pulls in the whole pipeline, and a ``--help`` that walks the
#: retrieval stack is the same mistake AP-06 names, one layer up.
_LAZY: dict[str, str] = {
    "ApiError": "axiom.api.routes",
    "get_chunk": "axiom.api.routes",
    "health": "axiom.api.routes",
    "list_versions": "axiom.api.routes",
    "run_query": "axiom.api.routes",
    "warm_models": "axiom.api.routes",
    "SERVE_INSTALL_HINT": "axiom.api.app",
    "ServeDependencyError": "axiom.api.app",
    "create_app": "axiom.api.app",
    "missing_dependencies": "axiom.api.app",
    "serve": "axiom.api.app",
}


def __getattr__(name: str) -> Any:
    """Resolve the pipeline- and FastAPI-touching names on first access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """Include the lazily-served names in ``dir()`` and tab completion."""
    return sorted(__all__)
