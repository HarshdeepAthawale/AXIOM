"""Weighted Reciprocal Rank Fusion: deterministic arithmetic with a published reference.

Highest coverage bar in the plan alongside ``schema/`` (95% line, 90% branch,
TestPlan.md section 9), and the only module where TestPlan.md section 1.2 rule 1
permits asserting a document's *position*: the fixtures here are built so the
ordering is arithmetically forced, and every expected value is re-derived in the
test body rather than copied from a golden file (section 8.2 rule 4).

A note on the "unweighted mode" TC-049 and Appendix B describe. ``QueryPlan``
validates that ``strategy_weights`` sums to 1.0, so ``w_i = 1.0`` for all three
signals is literally unrepresentable in the schema. Fusion is linear in the
weights, so the equal-weight vector ``{1/3, 1/3, 1/3}`` produces the reference
scores scaled by exactly 1/3 -- identical ordering, identical ratios. Each test
below states the scale it works in.
"""

from __future__ import annotations

import math

import pytest

from axiom.config import DEFAULT_STRATEGY_WEIGHTS, get_settings
from axiom.retrieval.fusion import (
    SIGNAL_ORDER,
    merge_ranked_lists,
    merge_signal_results,
    plan_with_weights,
    rank_key,
    reciprocal_rank_fusion,
    renormalise_weights,
    signal_width,
)
from axiom.schema import QueryPlan, QueryType, ScoredChunk, SignalKind

from .fakes import make_plan, ranked
from .helpers import assert_fused_list, hex32

DENSE, SPARSE, STRUCT = SignalKind.DENSE, SignalKind.SPARSE, SignalKind.STRUCTURAL

#: Readable labels -> the 32-hex ids the schema requires. Ordering matters for
#: the tie-break tests, so the mapping is computed once and asserted below.
A, B, C = hex32("A"), hex32("B"), hex32("C")

EQUAL_THIRDS = {DENSE: 1 / 3, SPARSE: 1 / 3, STRUCT: 1 / 3}


@pytest.fixture
def settings():
    """Default settings with ``rrf_k=60`` and ``fusion_top_n=25``."""
    from pathlib import Path

    return get_settings("default", configs_dir=Path(__file__).resolve().parent.parent / "configs")


def _by_id(results) -> dict[str, float]:
    return {result.chunk_id: result.rrf_score for result in results}


def _padded(signal: SignalKind, placements: dict[str, int], depth: int) -> list[ScoredChunk]:
    """A ranked list of ``depth`` entries placing each id at its stated rank.

    Filler ids are derived from the signal and rank so they never collide with
    A/B/C or with another signal's filler -- a filler that accidentally matched
    would add a contribution and silently change the arithmetic under test.
    """
    at_rank = {rank: chunk_id for chunk_id, rank in placements.items()}
    ids = [
        at_rank.get(rank, hex32(f"filler-{signal.value}-{rank}")) for rank in range(1, depth + 1)
    ]
    return ranked(signal, ids)


# ---------------------------------------------------------------------------
# TC-049 / TC-050 -- the worked example
# ---------------------------------------------------------------------------


class TestWorkedExample:
    """TC-049's three lists, built once and reused.

    * ``A`` at rank 5 in all three lists.
    * ``B`` at rank 1 in dense and sparse, absent from structural.
    * ``C`` at rank 1 in structural and rank 10 in dense.
    """

    @staticmethod
    def _lists() -> dict[SignalKind, list[ScoredChunk]]:
        return {
            DENSE: _padded(DENSE, {A: 5, B: 1, C: 10}, 10),
            SPARSE: _padded(SPARSE, {A: 5, B: 1}, 10),
            STRUCT: _padded(STRUCT, {A: 5, C: 1}, 10),
        }

    def test_tc049_rrf_arithmetic_matches_the_hand_computed_example(self, settings) -> None:
        """TC-049, in the 1/3-scaled equal-weight basis.

        Unweighted reference from the doc::

            score(A) = 1/65 + 1/65 + 1/65 = 3/65  = 0.046154
            score(B) = 1/61 + 1/61        = 2/61  = 0.032787
            score(C) = 1/61 + 1/70               = 0.030679

        With ``w_i = 1/3`` every term is scaled by 1/3, so the assertion is on
        ``3 * score``. **A beats B** -- one chunk that three signals agree on at
        rank 5 outranks one that two signals put first -- and that property is
        the entire justification for using RRF rather than a max or a mean.
        """
        plan = make_plan(weights=EQUAL_THIRDS)
        outcome = reciprocal_rank_fusion(self._lists(), plan, settings)
        scores = _by_id(outcome.results)

        assert 3 * scores[A] == pytest.approx(3 / 65, abs=1e-6)
        assert 3 * scores[B] == pytest.approx(2 / 61, abs=1e-6)
        assert 3 * scores[C] == pytest.approx(1 / 61 + 1 / 70, abs=1e-6)

        assert 3 * scores[A] == pytest.approx(0.046154, abs=1e-6)
        assert 3 * scores[B] == pytest.approx(0.032787, abs=1e-6)
        assert 3 * scores[C] == pytest.approx(0.030679, abs=1e-6)

        assert [r.chunk_id for r in outcome.results[:3]] == [A, B, C]
        assert scores[A] > scores[B] > scores[C]

    def test_tc050_weighted_rrf_matches_the_contract_formula_term_by_term(self, settings) -> None:
        """TC-050: ``Σ w_i/(60+rank_i)`` with SEMANTIC weights ``{.6,.3,.1}``, to 1e-9.

        Hand-computed here rather than looked up::

            A = .6/65 + .3/65 + .1/65
            B = .6/61 + .3/61
            C = .1/61 + .6/70
        """
        plan = make_plan(query_type=QueryType.SEMANTIC)
        assert plan.strategy_weights == {DENSE: 0.60, SPARSE: 0.30, STRUCT: 0.10}
        outcome = reciprocal_rank_fusion(self._lists(), plan, settings)
        scores = _by_id(outcome.results)

        assert settings.rrf_k == 60
        assert scores[A] == pytest.approx(0.6 / 65 + 0.3 / 65 + 0.1 / 65, abs=1e-9)
        assert scores[B] == pytest.approx(0.6 / 61 + 0.3 / 61, abs=1e-9)
        assert scores[C] == pytest.approx(0.1 / 61 + 0.6 / 70, abs=1e-9)

    def test_rrf_k_is_read_from_settings_not_hard_coded(self, settings) -> None:
        """AP-07: no magic numbers. Changing ``rrf_k`` must change the output."""
        plan = make_plan(weights=EQUAL_THIRDS)
        default = reciprocal_rank_fusion(self._lists(), plan, settings)
        widened = reciprocal_rank_fusion(
            self._lists(), plan, settings.model_copy(update={"rrf_k": 10})
        )
        assert _by_id(default.results)[A] != _by_id(widened.results)[A]
        assert 3 * _by_id(widened.results)[A] == pytest.approx(3 / 15, abs=1e-9)

    def test_tc053_contributions_record_the_true_per_signal_rank(self, settings) -> None:
        """TC-053: a signal the doc did not appear in is *absent*, never 0 or -1."""
        plan = make_plan(weights=EQUAL_THIRDS)
        results = {
            r.chunk_id: r for r in reciprocal_rank_fusion(self._lists(), plan, settings).results
        }

        assert results[A].contributions == {DENSE: 5, SPARSE: 5, STRUCT: 5}
        assert results[B].contributions == {DENSE: 1, SPARSE: 1}
        assert STRUCT not in results[B].contributions
        assert results[C].contributions == {DENSE: 10, STRUCT: 1}
        assert results[A].signal_count == 3 and results[B].signal_count == 2


# ---------------------------------------------------------------------------
# US-2 acceptance -- the multi-signal agreement property
# ---------------------------------------------------------------------------


def test_us2_agreement_beats_a_single_strong_signal(settings) -> None:
    """US-2: a chunk both signals rank 5th outranks one only dense ranks 1st.

    Hand-computed with HYBRID weights ``{dense .34, sparse .33, struct .33}``
    and ``k = 60``::

        agreed  = .34/(60+5) + .33/(60+5) = .67/65 = 0.010308
        dense_1 = .34/(60+1)              = .34/61 = 0.005574

        0.010308 > 0.005574, so `agreed` ranks first.

    The margin is close to 2x, so this is not a knife-edge that a weight retune
    would flip -- the property survives any weight vector where the two signals'
    weights sum to more than ``(61/65) * w_dense``, which every row of the §5
    table satisfies.
    """
    agreed, dense_only = hex32("agreed"), hex32("dense-only")
    lists = {
        DENSE: _padded(DENSE, {dense_only: 1, agreed: 5}, 5),
        SPARSE: _padded(SPARSE, {agreed: 5}, 5),
    }
    plan = make_plan(query_type=QueryType.HYBRID)
    outcome = reciprocal_rank_fusion(lists, plan, settings)
    scores = _by_id(outcome.results)

    expected_agreed = 0.34 / 65 + 0.33 / 65
    expected_dense_only = 0.34 / 61
    # Structural returned nothing, so its .33 is dropped and the survivors are
    # renormalised over .67 -- the same scale factor for both docs, so the
    # comparison the story asserts is unaffected.
    scale = 1.0 / (0.34 + 0.33)
    assert scores[agreed] == pytest.approx(expected_agreed * scale, abs=1e-9)
    assert scores[dense_only] == pytest.approx(expected_dense_only * scale, abs=1e-9)
    assert scores[agreed] > scores[dense_only]
    assert outcome.results[0].chunk_id == agreed


# ---------------------------------------------------------------------------
# TC-051 -- rank-space invariance
# ---------------------------------------------------------------------------


def test_tc051_fusion_is_invariant_to_monotonic_score_rescaling(settings) -> None:
    """TC-051: rescale every raw score by a strictly-increasing map; output is
    byte-identical. Proves fusion never reads :attr:`ScoredChunk.score`.

    Three different maps, one per signal, so a bug that happened to cancel
    across signals would still show up.
    """
    base = {
        DENSE: _padded(DENSE, {A: 5, B: 1, C: 10}, 10),
        SPARSE: _padded(SPARSE, {A: 5, B: 1}, 10),
        STRUCT: _padded(STRUCT, {A: 5, C: 1}, 10),
    }
    maps = {
        DENSE: lambda s: 10.0 * s + 3.0,
        SPARSE: math.exp,
        STRUCT: lambda s: math.log1p(s + 1.0),
    }
    rescaled = {
        signal: [entry.model_copy(update={"score": maps[signal](entry.score)}) for entry in entries]
        for signal, entries in base.items()
    }

    plan = make_plan(weights=EQUAL_THIRDS)
    before = reciprocal_rank_fusion(base, plan, settings)
    after = reciprocal_rank_fusion(rescaled, plan, settings)

    assert [r.chunk_id for r in before.results] == [r.chunk_id for r in after.results]
    assert [r.rrf_score for r in before.results] == [r.rrf_score for r in after.results]
    assert [r.dominant_signal for r in before.results] == [r.dominant_signal for r in after.results]


# ---------------------------------------------------------------------------
# TC-052 -- determinism and tie-breaking
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_rank_key_is_negated_score_then_ascending_id(self) -> None:
        assert rank_key("ff", 0.5) == (-0.5, "ff")
        assert rank_key("aa", 0.5) < rank_key("bb", 0.5)

    def test_tc052_exact_ties_break_by_ascending_chunk_id(self, settings) -> None:
        """TC-052: two docs at identical ranks in identical signals tie exactly.

        Both sit at dense rank 1 and 2 respectively in *mirrored* lists, so the
        float arithmetic is identical, not merely close.
        """
        low, high = sorted([hex32("tie-one"), hex32("tie-two")])
        lists = {
            DENSE: ranked(DENSE, [low, high]),
            SPARSE: ranked(SPARSE, [high, low]),
        }
        plan = make_plan(weights={DENSE: 0.5, SPARSE: 0.5})
        outcome = reciprocal_rank_fusion(lists, plan, settings)
        assert outcome.results[0].rrf_score == outcome.results[1].rrf_score
        assert [r.chunk_id for r in outcome.results] == [low, high]

    def test_tc052_reversed_input_order_gives_the_same_output(self, settings) -> None:
        """The mapping is iterated in ``SIGNAL_ORDER``, not in insertion order."""
        lists = {
            DENSE: _padded(DENSE, {A: 5, B: 1}, 5),
            SPARSE: _padded(SPARSE, {A: 5, B: 1}, 5),
            STRUCT: _padded(STRUCT, {A: 5}, 5),
        }
        plan = make_plan(weights=EQUAL_THIRDS)
        forward = reciprocal_rank_fusion(lists, plan, settings)
        reversed_mapping = dict(reversed(list(lists.items())))
        backward = reciprocal_rank_fusion(reversed_mapping, plan, settings)
        assert [r.chunk_id for r in forward.results] == [r.chunk_id for r in backward.results]
        assert [r.rrf_score for r in forward.results] == [r.rrf_score for r in backward.results]

    def test_repeated_fusion_is_bit_identical(self, settings) -> None:
        """NFR-08: no wall clock, no randomness, no set iteration order."""
        lists = {DENSE: _padded(DENSE, {A: 3, B: 1}, 20), SPARSE: _padded(SPARSE, {B: 7}, 20)}
        plan = make_plan(weights={DENSE: 0.5, SPARSE: 0.5})
        runs = [reciprocal_rank_fusion(lists, plan, settings) for _ in range(5)]
        signatures = {tuple((r.chunk_id, r.rrf_score) for r in run.results) for run in runs}
        assert len(signatures) == 1


# ---------------------------------------------------------------------------
# TC-054 -- dominant_signal
# ---------------------------------------------------------------------------


class TestDominantSignal:
    def test_tc054_dominant_signal_is_the_largest_weighted_term(self, settings) -> None:
        """TC-054: rank 1 in sparse, rank 40 in dense, SEMANTIC weights.

        The row's stated justification is wrong and the row's own instruction
        ("assert the comparison, not a hard-coded signal") is what is
        implemented here::

            sparse term = .3/(60+1)  = 0.0049180
            dense term  = .6/(60+40) = 0.0060000

        ``0.0049180 < 0.0060000``, so the dominant signal is **DENSE**, not
        SPARSE. The crossover is at dense rank ~62: ``.6/(60+r) = .3/61``
        gives ``r = 62``. The TC-054 expected-result cell needs correcting.
        """
        doc = hex32("dom")
        lists = {
            DENSE: _padded(DENSE, {doc: 40}, 40),
            SPARSE: _padded(SPARSE, {doc: 1}, 5),
        }
        plan = make_plan(query_type=QueryType.SEMANTIC)
        outcome = reciprocal_rank_fusion(lists, plan, settings)
        result = next(r for r in outcome.results if r.chunk_id == doc)

        weights = renormalise_weights(plan.strategy_weights, [DENSE, SPARSE])
        dense_term = weights[DENSE] / (settings.rrf_k + 40)
        sparse_term = weights[SPARSE] / (settings.rrf_k + 1)
        expected = DENSE if dense_term > sparse_term else SPARSE
        assert result.dominant_signal is expected
        assert result.dominant_signal is DENSE

    def test_dominant_signal_flips_past_the_crossover_rank(self, settings) -> None:
        """The same setup with dense at rank 90 makes SPARSE dominant."""
        doc = hex32("dom")
        lists = {
            DENSE: _padded(DENSE, {doc: 90}, 90),
            SPARSE: _padded(SPARSE, {doc: 1}, 5),
        }
        plan = make_plan(query_type=QueryType.SEMANTIC)
        outcome = reciprocal_rank_fusion(lists, plan, settings, top_n=200)
        result = next(r for r in outcome.results if r.chunk_id == doc)
        assert result.dominant_signal is SPARSE

    def test_exact_tie_resolves_toward_dense_then_sparse_then_structural(self, settings) -> None:
        """Schema.md section 11: ties resolve in ``SIGNAL_ORDER``."""
        assert SIGNAL_ORDER == (DENSE, SPARSE, STRUCT)
        doc = hex32("tie")
        lists = {signal: ranked(signal, [doc]) for signal in SIGNAL_ORDER}
        plan = make_plan(weights=EQUAL_THIRDS)
        assert reciprocal_rank_fusion(lists, plan, settings).results[0].dominant_signal is DENSE


# ---------------------------------------------------------------------------
# TC-055 / TC-056 -- empty-signal handling
# ---------------------------------------------------------------------------


class TestEmptySignals:
    def test_tc055_structural_empty_renormalises_dense_and_sparse(self, settings) -> None:
        """TC-055: ``{.6,.3,.1}`` with structural empty becomes ``{.6667,.3333}``.

        The scores are then checked against a *directly hand-computed two-signal
        fusion*, which is the assertion that actually matters: zero-filling the
        absent signal would leave the weights at ``{.6,.3,.1}`` and shrink every
        score by a factor of 0.9.
        """
        doc = hex32("doc")
        lists = {
            DENSE: _padded(DENSE, {doc: 1}, 5),
            SPARSE: _padded(SPARSE, {doc: 3}, 5),
            STRUCT: [],
        }
        plan = make_plan(query_type=QueryType.SEMANTIC)
        outcome = reciprocal_rank_fusion(lists, plan, settings)

        assert outcome.effective_weights == pytest.approx({DENSE: 0.6 / 0.9, SPARSE: 0.3 / 0.9})
        assert sum(outcome.effective_weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert outcome.dropped_signals == (STRUCT,)

        by_hand = (0.6 / 0.9) / 61 + (0.3 / 0.9) / 63
        assert outcome.results[0].rrf_score == pytest.approx(by_hand, abs=1e-12)

        zero_filled = 0.6 / 61 + 0.3 / 63
        assert outcome.results[0].rrf_score > zero_filled, "score must not shrink"

    def test_tc055_the_logged_plan_carries_the_renormalised_vector(self, settings) -> None:
        """Section 5.1.2: the number in the log must be the number the arithmetic used."""
        doc = hex32("doc")
        plan = make_plan(query_type=QueryType.SEMANTIC)
        outcome = reciprocal_rank_fusion({DENSE: ranked(DENSE, [doc])}, plan, settings)
        assert outcome.plan.strategy_weights == {DENSE: 1.0}
        assert plan.strategy_weights == DEFAULT_STRATEGY_WEIGHTS[QueryType.SEMANTIC], (
            "the input plan must not be mutated (AP-08)"
        )

    def test_a_missing_key_and_an_empty_list_mean_the_same_thing(self, settings) -> None:
        doc = hex32("doc")
        plan = make_plan(query_type=QueryType.SEMANTIC)
        absent = reciprocal_rank_fusion({DENSE: ranked(DENSE, [doc])}, plan, settings)
        empty = reciprocal_rank_fusion(
            {DENSE: ranked(DENSE, [doc]), SPARSE: [], STRUCT: []}, plan, settings
        )
        assert _by_id(absent.results) == _by_id(empty.results)
        assert absent.effective_weights == empty.effective_weights

    def test_tc056_all_signals_empty_yields_an_empty_list(self, settings) -> None:
        """TC-056: ``[]``, not a crash, and the nominal plan is returned intact."""
        plan = make_plan(query_type=QueryType.SEMANTIC)
        outcome = reciprocal_rank_fusion({DENSE: [], SPARSE: [], STRUCT: []}, plan, settings)
        assert outcome.results == []
        assert outcome.top1 == 0.0
        assert outcome.candidate_count == 0
        assert outcome.effective_weights == {}
        assert outcome.plan is plan
        assert outcome.dropped_signals == (DENSE, SPARSE, STRUCT)

    def test_empty_mapping_is_also_handled(self, settings) -> None:
        assert reciprocal_rank_fusion({}, make_plan(), settings).results == []

    def test_a_signal_with_zero_plan_weight_is_ignored(self, settings) -> None:
        """The eval profile's structural leg: rows present, weight zero."""
        doc, other = hex32("doc"), hex32("other")
        lists = {DENSE: ranked(DENSE, [doc]), STRUCT: ranked(STRUCT, [other])}
        plan = QueryPlan(
            original_query="q",
            query_type=QueryType.SEMANTIC,
            strategy_weights={DENSE: 1.0, STRUCT: 0.0},
        )
        outcome = reciprocal_rank_fusion(lists, plan, settings)
        assert [r.chunk_id for r in outcome.results] == [doc]
        assert any("plan weight is 0" in note for note in outcome.degradations)


class TestRenormaliseWeights:
    def test_worked_example_from_section_5_1_2(self) -> None:
        weights = renormalise_weights(DEFAULT_STRATEGY_WEIGHTS[QueryType.SEMANTIC], [DENSE, SPARSE])
        assert weights == pytest.approx({DENSE: 0.6 / 0.9, SPARSE: 0.3 / 0.9})

    def test_all_present_is_the_identity(self) -> None:
        nominal = DEFAULT_STRATEGY_WEIGHTS[QueryType.STRUCTURAL]
        assert renormalise_weights(nominal, SIGNAL_ORDER) == pytest.approx(dict(nominal))

    def test_no_present_signal_yields_an_empty_map(self) -> None:
        assert renormalise_weights(DEFAULT_STRATEGY_WEIGHTS[QueryType.HYBRID], []) == {}

    def test_a_present_signal_with_zero_weight_is_dropped(self) -> None:
        assert renormalise_weights({DENSE: 1.0, SPARSE: 0.0}, [DENSE, SPARSE]) == {DENSE: 1.0}

    def test_output_always_sums_to_one(self) -> None:
        for query_type in QueryType:
            for present in ([DENSE], [SPARSE], [STRUCT], [DENSE, SPARSE], SIGNAL_ORDER):
                weights = renormalise_weights(DEFAULT_STRATEGY_WEIGHTS[query_type], present)
                assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)


def test_plan_with_weights_rebuilds_rather_than_mutates() -> None:
    """AP-08 plus the validator: the rebuilt plan is re-validated on the way out."""
    plan = make_plan(query_type=QueryType.SEMANTIC)
    rebuilt = plan_with_weights(plan, {DENSE: 0.7, SPARSE: 0.3})
    assert rebuilt is not plan
    assert plan.strategy_weights == DEFAULT_STRATEGY_WEIGHTS[QueryType.SEMANTIC]
    assert rebuilt.strategy_weights == {DENSE: 0.7, SPARSE: 0.3}
    assert rebuilt.original_query == plan.original_query


def test_plan_with_weights_returns_the_same_object_when_unchanged() -> None:
    plan = make_plan(query_type=QueryType.SEMANTIC)
    assert plan_with_weights(plan, dict(plan.strategy_weights)) is plan
    assert plan_with_weights(plan, {}) is plan


# ---------------------------------------------------------------------------
# TC-057 / TC-058 -- invariants, truncation, widths
# ---------------------------------------------------------------------------


class TestInvariantsAndWidths:
    def test_tc057_300_candidates_truncate_to_25_with_no_duplicates(self, settings) -> None:
        """TC-057: ``len(output) == min(fusion_top_n, unique_candidates)``."""
        lists = {
            signal: ranked(signal, [hex32(f"{signal.value}-{i}") for i in range(100)])
            for signal in SIGNAL_ORDER
        }
        plan = make_plan(weights=EQUAL_THIRDS)
        outcome = reciprocal_rank_fusion(lists, plan, settings)

        assert outcome.candidate_count == 300
        assert len(outcome.results) == min(settings.fusion_top_n, 300) == 25
        assert_fused_list(outcome.results)

    def test_output_is_shorter_than_top_n_when_candidates_are_scarce(self, settings) -> None:
        lists = {DENSE: ranked(DENSE, [hex32(str(i)) for i in range(4)])}
        outcome = reciprocal_rank_fusion(lists, make_plan(), settings)
        assert len(outcome.results) == 4

    def test_top_n_override_is_honoured(self, settings) -> None:
        lists = {DENSE: ranked(DENSE, [hex32(str(i)) for i in range(40)])}
        outcome = reciprocal_rank_fusion(lists, make_plan(), settings, top_n=7)
        assert len(outcome.results) == 7

    def test_top_n_of_zero_returns_nothing_but_still_counts_candidates(self, settings) -> None:
        lists = {DENSE: ranked(DENSE, [hex32(str(i)) for i in range(4)])}
        outcome = reciprocal_rank_fusion(lists, make_plan(), settings, top_n=0)
        assert outcome.results == [] and outcome.candidate_count == 4

    def test_tc058_candidate_widths_come_from_the_contract(self, settings) -> None:
        """TC-058: dense 100, sparse 100, structural 50, fusion 25, final 10."""
        assert signal_width(DENSE, settings) == settings.dense_top_k == 100
        assert signal_width(SPARSE, settings) == settings.sparse_top_k == 100
        assert signal_width(STRUCT, settings) == settings.structural_top_k == 50
        assert settings.fusion_top_n == 25
        assert settings.top_k_default == 10

    def test_per_signal_counts_report_what_each_signal_returned(self, settings) -> None:
        lists = {DENSE: ranked(DENSE, [A, B]), SPARSE: ranked(SPARSE, [A])}
        outcome = reciprocal_rank_fusion(lists, make_plan(), settings)
        assert outcome.per_signal_counts == {DENSE: 2, SPARSE: 1, STRUCT: 0}


class TestMalformedInput:
    def test_a_duplicate_chunk_id_keeps_the_best_rank_and_reports_it(self, settings) -> None:
        """Rule 3: a degrade is loud. The duplicate must not double-count."""
        doc = hex32("dup")
        entries = [
            ScoredChunk(chunk_id=doc, score=0.9, rank=1, signal=DENSE),
            ScoredChunk(chunk_id=doc, score=0.5, rank=4, signal=DENSE),
        ]
        outcome = reciprocal_rank_fusion({DENSE: entries}, make_plan(), settings)
        assert len(outcome.results) == 1
        assert outcome.results[0].contributions == {DENSE: 1}
        assert any("duplicate" in note for note in outcome.degradations)

    def test_a_mis_tagged_candidate_is_dropped_and_reported(self, settings) -> None:
        """A sparse row filed under the dense key would otherwise be scored at
        the dense weight -- a silently wrong answer."""
        doc, stray = hex32("doc"), hex32("stray")
        entries = [
            ScoredChunk(chunk_id=doc, score=0.9, rank=1, signal=DENSE),
            ScoredChunk(chunk_id=stray, score=0.8, rank=2, signal=SPARSE),
        ]
        outcome = reciprocal_rank_fusion({DENSE: entries}, make_plan(), settings)
        assert [r.chunk_id for r in outcome.results] == [doc]
        assert any("tagged sparse" in note for note in outcome.degradations)


# ---------------------------------------------------------------------------
# Sub-query and cross-version merging
# ---------------------------------------------------------------------------


class TestMerging:
    def test_a_single_list_passes_through_untouched(self, settings) -> None:
        """The common path must not re-score and perturb the retriever's order."""
        entries = ranked(DENSE, [A, B, C])
        assert merge_ranked_lists([entries], DENSE, settings) == entries

    def test_empty_input_merges_to_empty(self, settings) -> None:
        assert merge_ranked_lists([], DENSE, settings) == []
        assert merge_ranked_lists([[], []], DENSE, settings) == []

    def test_merged_output_is_a_well_formed_ranked_list(self, settings) -> None:
        merged = merge_ranked_lists([ranked(DENSE, [A, B]), ranked(DENSE, [C, A])], DENSE, settings)
        from .helpers import assert_ranked_list

        assert_ranked_list(merged, expected_signal=DENSE)
        assert {entry.chunk_id for entry in merged} == {A, B, C}

    def test_a_doc_found_by_two_sub_queries_outranks_one_found_by_one(self, settings) -> None:
        """Hand-computed: ``A`` = 1/61 + 1/61, ``B`` = 1/61, ``C`` = 1/62."""
        merged = merge_ranked_lists([ranked(DENSE, [A, C]), ranked(DENSE, [A, B])], DENSE, settings)
        scores = {entry.chunk_id: entry.score for entry in merged}
        assert scores[A] == pytest.approx(2 / 61)
        assert merged[0].chunk_id == A

    def test_merge_respects_the_signal_width(self, settings) -> None:
        wide = [
            ranked(DENSE, [hex32(f"a{i}") for i in range(80)]),
            ranked(DENSE, [hex32(f"b{i}") for i in range(80)]),
        ]
        merged = merge_ranked_lists(wide, DENSE, settings, width=12)
        assert len(merged) == 12

    def test_merge_signal_results_looks_like_a_single_fan_out(self, settings) -> None:
        per_query = [
            {DENSE: ranked(DENSE, [A]), SPARSE: ranked(SPARSE, [B])},
            {DENSE: ranked(DENSE, [C])},
        ]
        merged = merge_signal_results(per_query, settings)
        assert set(merged) == {DENSE, SPARSE}
        assert {entry.chunk_id for entry in merged[DENSE]} == {A, C}
        assert STRUCT not in merged, "a signal nobody returned must not appear as an empty key"

    def test_merged_results_fuse_without_a_special_case(self, settings) -> None:
        merged = merge_signal_results(
            [{DENSE: ranked(DENSE, [A, B])}, {DENSE: ranked(DENSE, [B, A])}], settings
        )
        outcome = reciprocal_rank_fusion(merged, make_plan(), settings)
        assert_fused_list(outcome.results)
        assert len(outcome.results) == 2
