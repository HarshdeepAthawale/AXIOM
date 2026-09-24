"""The three closed vocabularies of the retrieval pipeline."""

from __future__ import annotations

from enum import StrEnum


class QueryType(StrEnum):
    """How the agent classified the incoming query.

    Selects the RRF weight vector (see TechSpecifications.md section 5.1.1).
    """

    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    USAGE = "usage"
    HYBRID = "hybrid"


class SignalKind(StrEnum):
    """One of the three first-stage retrieval signals."""

    DENSE = "dense"
    SPARSE = "sparse"
    STRUCTURAL = "structural"


class ChunkKind(StrEnum):
    """The syntactic category of the AST node a chunk was cut from."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    BLOCK = "block"
