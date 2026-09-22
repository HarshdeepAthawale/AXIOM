"""Per-signal scores, their fusion, and the single user-facing result type."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from axiom.schema._base import AxiomModel
from axiom.schema.chunk import Chunk
from axiom.schema.enums import SignalKind


class ScoredChunk(AxiomModel):
    """A single (chunk, score, rank) triple emitted by one retrieval signal.

    A ``list[ScoredChunk]`` handed to fusion must satisfy: all elements share one
    ``signal``; ``rank`` values are exactly ``1..len(list)`` with no gaps or ties;
    and ``score`` is non-increasing as ``rank`` increases. Ties in raw score are
    broken deterministically by ascending ``chunk_id`` (NFR-08).
    """

    chunk_id: str = Field(
        ..., pattern=r"^[0-9a-f]{32}$", description="Identity of the scored chunk."
    )
    score: float = Field(
        ...,
        description="Raw, signal-native score. Cosine for DENSE, BM25 for SPARSE, "
        "graph score for STRUCTURAL. NOT comparable across signals.",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Position within this signal's ranked list. 1-indexed; rank 1 is best.",
    )
    signal: SignalKind = Field(..., description="Which retriever produced this.")


class FusedResult(AxiomModel):
    """A chunk after Reciprocal Rank Fusion, optionally after cross-encoder reranking.

    ``rerank_score`` being ``None`` is not "unknown relevance", it is "not a rerank
    candidate". Sorting a mixed list must therefore use :attr:`final_score`, never
    ``rerank_score or 0.0``.
    """

    chunk_id: str = Field(
        ..., pattern=r"^[0-9a-f]{32}$", description="Identity of the fused chunk."
    )
    rrf_score: float = Field(
        ...,
        gt=0.0,
        description="Sum over contributing signals of w_i / (60 + rank_i). Strictly "
        "positive: a chunk with no contributions is never constructed.",
    )
    rerank_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sigmoid-normalised cross-encoder relevance in [0,1]. None when "
        "the chunk fell outside top-N=25, or when reranking is disabled.",
    )
    contributions: dict[SignalKind, int] = Field(
        ...,
        min_length=1,
        description="signal -> 1-indexed rank that signal assigned this chunk. "
        "Absent key means that signal did not return this chunk at all.",
    )
    dominant_signal: SignalKind = Field(
        ...,
        description="The signal contributing the largest w_i/(60+rank_i) term. Ties "
        "resolve in the order DENSE, SPARSE, STRUCTURAL.",
    )

    @field_validator("contributions")
    @classmethod
    def _ranks_positive(cls, value: dict[SignalKind, int]) -> dict[SignalKind, int]:
        for signal, rank in value.items():
            if rank < 1:
                raise ValueError(f"contribution rank for {signal} must be >= 1, got {rank}")
        return value

    @model_validator(mode="after")
    def _dominant_is_contributor(self) -> FusedResult:
        if self.dominant_signal not in self.contributions:
            raise ValueError(
                f"dominant_signal {self.dominant_signal} absent from contributions "
                f"{sorted(self.contributions)}"
            )
        return self

    @property
    def final_score(self) -> float:
        """Score used for the final ordering: rerank score when present, else RRF score."""
        return self.rrf_score if self.rerank_score is None else self.rerank_score

    @property
    def signal_count(self) -> int:
        """How many of the three signals independently surfaced this chunk."""
        return len(self.contributions)


class RetrievalResult(AxiomModel):
    """A user-facing search hit: the snippet, its location, its score, and why it matched.

    The only schema object that crosses the API and UI boundary.
    """

    chunk: Chunk = Field(..., description="The full retrieved chunk, text included.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final relevance in [0,1]: the rerank score, multiplied by the "
        "evolutionary stability bonus when the chunk belongs to a multi-version family.",
    )
    match_reason: str = Field(
        ...,
        min_length=1,
        description="One-sentence explanation naming the dominant signal and the "
        'concrete evidence, e.g. "structural: calls preprocessInput before main".',
    )
    signals: dict[SignalKind, int] = Field(
        ...,
        min_length=1,
        description="signal -> 1-indexed first-stage rank. Copied verbatim from "
        "FusedResult.contributions; powers the 'why did this match' UI panel.",
    )
    optimization_hint: str | None = Field(
        default=None,
        description="Optional bonus-feature note about the surfaced code, e.g. "
        '"awaits inside a for-loop; consider Promise.all". Never a code rewrite.',
    )

    @property
    def location_ref(self) -> str:
        """Convenience passthrough: 'src/agents/bluetooth.js:42-67'."""
        return self.chunk.location.as_ref()
