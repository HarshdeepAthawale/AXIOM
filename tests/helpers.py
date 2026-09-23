"""Shared invariant assertions.

TestPlan.md names ``assert_ranked_list`` in four different test cases (TC-057,
TC-090, and the retriever rows), which is the signal that it belongs in one
place: a ranked-list invariant asserted four slightly different ways is an
invariant that is not actually enforced.
"""

from __future__ import annotations

from collections.abc import Sequence

from axiom.schema import FusedResult, ScoredChunk

__all__ = ["assert_fused_list", "assert_ranked_list", "hex32"]


def hex32(label: str) -> str:
    """A syntactically valid ``chunk_id`` derived from a readable label.

    Schema.md pins ``chunk_id`` to ``^[0-9a-f]{32}$``, so a test cannot use
    ``"doc_a"`` as an id. Deriving it from the label keeps the test readable
    (``hex32("A")``) while satisfying the pattern, and keeps it stable across
    runs, which matters because ascending ``chunk_id`` is the documented
    tie-break (NFR-08) and a test that engineers a tie must know the order.
    """
    from axiom.core.hashing import blake2b_128

    return blake2b_128(label.encode("utf-8"))


def assert_ranked_list(
    entries: Sequence[ScoredChunk],
    *,
    expected_signal: object | None = None,
    max_length: int | None = None,
) -> None:
    """Assert the global ranked-list contract on a ``list[ScoredChunk]``.

    Schema.md section 11: one signal throughout, ranks exactly ``1..n`` with no
    gaps or ties, scores non-increasing, and no duplicate ``chunk_id``.
    """
    assert [entry.rank for entry in entries] == list(range(1, len(entries) + 1)), (
        "ranks must be contiguous and 1-indexed"
    )
    ids = [entry.chunk_id for entry in entries]
    assert len(set(ids)) == len(ids), "ranked list contains a duplicate chunk_id"
    scores = [entry.score for entry in entries]
    assert scores == sorted(scores, reverse=True), "scores must be non-increasing with rank"
    if expected_signal is not None:
        assert {entry.signal for entry in entries} <= {expected_signal}
    if max_length is not None:
        assert len(entries) <= max_length


def assert_fused_list(results: Sequence[FusedResult]) -> None:
    """Assert the post-fusion ordering contract.

    Descending by ``final_score``, deduplicated, ties broken by ascending
    ``chunk_id``. ``final_score`` rather than ``rrf_score`` because a partially
    reranked list is exactly the case where ``rerank_score or 0.0`` would be a
    bug (Schema.md section 11).
    """
    ids = [result.chunk_id for result in results]
    assert len(set(ids)) == len(ids), "fused list contains a duplicate chunk_id"
    keys = [(-result.final_score, result.chunk_id) for result in results]
    assert keys == sorted(keys), "fused list is not sorted by (-score, chunk_id)"
    for result in results:
        assert result.dominant_signal in result.contributions
        assert result.rrf_score > 0.0
