"""Deterministic test doubles (TestPlan.md section 1.3).

Every double here is a *real implementation of the protocol the production code
declares*, not a mock that records calls. TestPlan.md section 1.2 rule 5 bans
asserting that a mock was called; what these classes give a test instead is a
**counter** and a **table**. A counter makes a claim observable ("a pure rename
costs zero embedding calls", TC-075); a table makes a ranking arithmetically
forced ("top-1 is exactly 0.34", TC-087) without pinning a real model's output.

None of them imports an optional dependency, so the whole suite runs on a bare
install -- which is the configuration NFR-07 promises the evaluator's laptop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from axiom.core.hashing import blake2b_128
from axiom.indexing.embedder import Embedder
from axiom.schema import Chunk, QueryPlan, ScoredChunk, SignalKind
from axiom.schema.enums import QueryType

__all__ = [
    "FakeCrossEncoder",
    "FakeEmbedder",
    "FakeLLM",
    "ScriptedBackend",
    "make_chunk",
    "make_plan",
    "ranked",
]


class FakeEmbedder(Embedder):
    """``blake2b(text)`` -> a deterministic unit vector of the configured dim.

    Subclasses the real :class:`~axiom.indexing.embedder.Embedder` rather than
    duck-typing it, so the base class's batching, L2 normalisation, dtype and --
    critically -- its ``texts_encoded`` counter are the production ones. A test
    that asserts an embedding count is then asserting against the same code path
    the index build uses, not against a parallel re-implementation.

    Deterministic across processes and machines: the vector is derived from a
    hash of the text, never from ``random`` or from object identity.
    """

    def __init__(
        self, dim: int = 384, batch_size: int = 64, model_id: str = "fake-embedder"
    ) -> None:
        super().__init__(model_id=model_id, dim=dim, batch_size=batch_size)

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = blake2b_128(text.encode("utf-8"))
            # Expand the 128-bit digest deterministically to `dim` floats by
            # re-hashing with a counter: a plain digest is 16 bytes and the
            # smallest configured dim is 384.
            stream = bytearray()
            block = 0
            while len(stream) < self.dim:
                stream.extend(bytes.fromhex(blake2b_128(digest.encode(), str(block).encode())))
                block += 1
            for column in range(self.dim):
                rows[row, column] = (stream[column] / 255.0) - 0.5
        return rows


class FakeCrossEncoder:
    """A :class:`~axiom.rerank.cross_encoder.PairScorer` backed by a lookup table.

    Satisfies the protocol structurally: ``name``, ``kind``, ``score_pairs``.
    ``kind`` is ``"cross_encoder"`` because that is what makes
    :attr:`RerankOutcome.model_ran` true, which in turn is what switches the
    sufficiency predicate onto ``rerank_score`` -- the branch TC-087 exercises.

    The table is keyed by *document text*, not by ``chunk_id``: ``score_pairs``
    receives ``(query, document)`` pairs and never sees an id. Anything not in
    the table scores :attr:`default`.
    """

    def __init__(
        self,
        table: Mapping[str, float] | None = None,
        default: float = 0.5,
        name: str = "fake-cross-encoder",
    ) -> None:
        self.table = dict(table or {})
        self.default = default
        self.name = name
        self.kind = "cross_encoder"
        #: Pairs scored so far. TC-059's "how many pairs did rerank actually
        #: score" is a count, not a mock assertion.
        self.pairs_scored = 0
        self.calls = 0

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls += 1
        self.pairs_scored += len(pairs)
        return [self._lookup(document) for _, document in pairs]

    def _lookup(self, document: str) -> float:
        for needle, score in self.table.items():
            if needle in document:
                return score
        return self.default


class FakeLLM:
    """Scripted stand-in for :class:`~axiom.agent.llm.QueryLLM`.

    Raises on any call while ``enabled`` is false. That is deliberate and is the
    assertion TestPlan.md section 1.3 asks for: the heuristic path is the
    *default* path (TechSpecifications.md section 3.3), so a test that runs with
    ``AXIOM_LLM_ENABLED=false`` and still reaches the LLM has found a real bug --
    a reachable network/model dependency on the evaluator's offline laptop.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        classification: QueryType | None = None,
        expansions: Sequence[str] = (),
        sub_queries: Sequence[str] = (),
        sufficiency: bool | None = None,
    ) -> None:
        self.enabled = enabled
        self.classification = classification
        self.expansions = list(expansions)
        self.sub_queries = list(sub_queries)
        self.sufficiency = sufficiency
        self.calls: list[str] = []

    def _guard(self, method: str) -> None:
        self.calls.append(method)
        if not self.enabled:
            raise AssertionError(
                f"FakeLLM.{method} called while AXIOM_LLM_ENABLED=false; the heuristic "
                "path must not reach the LLM"
            )

    def classify(self, query: str) -> QueryType | None:
        self._guard("classify")
        return self.classification

    def expand(self, query: str, identifiers: Sequence[str]) -> list[str]:
        self._guard("expand")
        return list(self.expansions)

    def decompose(self, query: str, max_sub_queries: int) -> list[str]:
        self._guard("decompose")
        return list(self.sub_queries)[:max_sub_queries]

    def judge_sufficiency(self, *args: Any, **kwargs: Any) -> bool | None:
        self._guard("judge_sufficiency")
        return self.sufficiency


class ScriptedBackend:
    """A :class:`~axiom.agent.loop.RetrievalBackend` whose fan-out is a script.

    The agent loop takes its backend as a parameter precisely so the loop's
    *control flow* -- pass count, budget, stop reason -- can be tested without an
    index on disk (TC-086..TC-089). One entry of ``script`` is consumed per
    ``fan_out`` call; the last entry repeats once the script runs dry, so a loop
    that runs more passes than expected still gets well-formed input rather than
    an IndexError that would mask the real assertion.
    """

    def __init__(
        self,
        script: Sequence[Mapping[SignalKind, Sequence[ScoredChunk]]],
        chunks: Mapping[str, Chunk] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self.script = [dict(step) for step in script] or [{}]
        self._chunks = dict(chunks or {})
        self.raises = raises
        #: Every plan the loop handed us, in order. TC-089 reads this to prove
        #: the pass-2 query differs from the pass-1 query.
        self.plans: list[QueryPlan] = []

    def fan_out(self, plan: QueryPlan) -> Mapping[SignalKind, Sequence[ScoredChunk]]:
        self.plans.append(plan)
        if self.raises is not None:
            raise self.raises
        index = min(len(self.plans) - 1, len(self.script) - 1)
        return self.script[index]

    def hydrate(self, chunk_ids: Sequence[str]) -> Mapping[str, Chunk]:
        return {cid: self._chunks[cid] for cid in chunk_ids if cid in self._chunks}


# ---------------------------------------------------------------------------
# Small builders. Tests that need a Chunk or a ranked list should not each
# re-derive the schema's construction rules.
# ---------------------------------------------------------------------------


def make_chunk(
    text: str = "function demo() { return 1; }",
    *,
    file_path: str = "src/demo.js",
    start_line: int = 1,
    version_id: str = "v1",
    symbol: str | None = "demo",
    **metadata: Any,
) -> Chunk:
    """Build a valid :class:`~axiom.schema.Chunk` with both digests derived.

    Byte span is computed from the text so ``ChunkLocation``'s half-open
    ``end_byte > start_byte`` rule holds for multibyte text too.
    """
    from axiom.schema import ChunkKind, ChunkLocation, ChunkMetadata

    encoded = text.encode("utf-8")
    location = ChunkLocation(
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + text.count("\n"),
        start_byte=0,
        end_byte=len(encoded),
    )
    fields: dict[str, Any] = {
        "symbol": symbol,
        "kind": ChunkKind.FUNCTION,
        "version_id": version_id,
    }
    fields.update(metadata)
    return Chunk.create(text, location, ChunkMetadata(**fields))


def make_plan(
    query: str = "how is the input preprocessed",
    *,
    query_type: QueryType = QueryType.HYBRID,
    weights: Mapping[SignalKind, float] | None = None,
    identifiers: Sequence[str] = (),
    sub_queries: Sequence[str] = (),
    expansion_terms: Sequence[str] = (),
) -> QueryPlan:
    """Build a :class:`~axiom.schema.QueryPlan` with contract-default weights."""
    from axiom.config import DEFAULT_STRATEGY_WEIGHTS

    return QueryPlan(
        original_query=query,
        query_type=query_type,
        sub_queries=list(sub_queries),
        extracted_identifiers=list(identifiers),
        expansion_terms=list(expansion_terms),
        strategy_weights=dict(weights or DEFAULT_STRATEGY_WEIGHTS[query_type]),
    )


def ranked(
    signal: SignalKind, chunk_ids: Sequence[str], scores: Sequence[float] | None = None
) -> list[ScoredChunk]:
    """Turn ids into a well-formed ranked list: contiguous 1-indexed ranks.

    Default scores descend from 1.0 in steps of 0.1 so the list satisfies the
    "score is non-increasing as rank increases" clause of
    :class:`~axiom.schema.ScoredChunk`'s contract without every caller inventing
    numbers. Fusion never reads them (TC-051); they exist so the input is legal.
    """
    values = list(scores) if scores is not None else [1.0 - 0.1 * i for i in range(len(chunk_ids))]
    return [
        ScoredChunk(chunk_id=cid, score=value, rank=position, signal=signal)
        for position, (cid, value) in enumerate(zip(chunk_ids, values, strict=True), start=1)
    ]
