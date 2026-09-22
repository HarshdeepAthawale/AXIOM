"""The shared data contract. All four workstreams code against this package.

Any change here is a breaking change until proven otherwise: see Schema.md
section 16 and RISK-12 (a late schema change breaks every workstream at once).
"""

from __future__ import annotations

from axiom.schema._base import AxiomModel
from axiom.schema.chunk import Chunk, ChunkLocation, ChunkMetadata
from axiom.schema.enums import ChunkKind, QueryType, SignalKind
from axiom.schema.plan import QueryPlan
from axiom.schema.retrieval import FusedResult, RetrievalResult, ScoredChunk
from axiom.schema.version import SnippetFamily, VersionManifest

__all__ = [
    "AxiomModel",
    "Chunk",
    "ChunkKind",
    "ChunkLocation",
    "ChunkMetadata",
    "FusedResult",
    "QueryPlan",
    "QueryType",
    "RetrievalResult",
    "ScoredChunk",
    "SignalKind",
    "SnippetFamily",
    "VersionManifest",
]
