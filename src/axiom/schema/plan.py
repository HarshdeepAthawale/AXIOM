"""The agent's decision record for one retrieval pass."""

from __future__ import annotations

from pydantic import Field, model_validator

from axiom.schema._base import AxiomModel
from axiom.schema.enums import QueryType, SignalKind


class QueryPlan(AxiomModel):
    """Classification, expansion, and weights for one retrieval pass.

    Produced by ``agent/classifier.py`` + ``agent/planner.py``, consumed by fusion
    (for ``strategy_weights``) and by the structural retriever (for
    ``extracted_identifiers``). Rewritten on each refinement pass, so the plan of
    the final pass is what gets logged and shown in the demo UI.
    """

    original_query: str = Field(
        ...,
        min_length=1,
        description="The user's query, verbatim and never mutated across passes.",
    )
    query_type: QueryType = Field(
        ..., description="Classification result. Selects the default weight vector."
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Decomposition of a compound query. Empty means 'run the original "
        "query as-is'. Each sub-query is retrieved independently and RRF-merged.",
    )
    extracted_identifiers: list[str] = Field(
        default_factory=list,
        description="Code identifiers lifted out of the query text, e.g. 'resolveTool'. "
        "Fed to the structural retriever as exact symbol lookups.",
    )
    expansion_terms: list[str] = Field(
        default_factory=list,
        description="Added synonyms/related code vocabulary, e.g. 'preprocess' -> "
        "['normalize', 'sanitize', 'transform']. Appended to the sparse query only.",
    )
    strategy_weights: dict[SignalKind, float] = Field(
        ...,
        min_length=1,
        description="Per-signal RRF weights. Must sum to 1.0 within 1e-6 over the "
        "signals actually available for this query.",
    )

    @model_validator(mode="after")
    def _weights_normalised(self) -> QueryPlan:
        total = sum(self.strategy_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"strategy_weights must sum to 1.0, got {total!r}")
        for signal, weight in self.strategy_weights.items():
            if weight < 0.0:
                raise ValueError(f"weight for {signal} must be >= 0, got {weight}")
        return self

    @property
    def effective_queries(self) -> list[str]:
        """Queries to actually retrieve: the sub-queries, or the original if none."""
        return self.sub_queries or [self.original_query]
