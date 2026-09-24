"""Shared Pydantic configuration for every Axiom schema model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AxiomModel(BaseModel):
    """Base class for every Axiom schema model.

    ``extra="forbid"`` is deliberate: a typo'd field name in a config file or a
    stale ``chunks.jsonl`` written by an older Axiom build must fail loudly at
    load time rather than being silently dropped and producing a subtly wrong
    index. ``frozen=True`` makes every schema object hashable and safe to share
    across the ONNX embedding thread pool without defensive copying.

    ``str_strip_whitespace=False`` is load-bearing: ``Chunk.text`` must preserve
    leading indentation exactly, or line offsets and the reranker's view of the
    code both drift.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        validate_assignment=True,
        use_enum_values=False,
    )
