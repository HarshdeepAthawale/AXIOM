"""The agent: classification, the sufficiency predicate, and the bounded loop.

TestPlan.md section 9 sets ``src/axiom/agent/`` at 75% line coverage and says
why: the *quality* of a rewrite is not testable -- it shows up in NDCG, not in
an assert. So what is tested here is the **contract**: it terminates, it stays
in budget, it degrades, it never raises, and the three PRD archetypes classify
correctly with the LLM off (which is the default configuration, not a fallback
-- TechSpecifications.md section 3.3).

Every test in this module runs with ``AXIOM_LLM_ENABLED=false``, set in
``conftest.py``. :class:`~tests.fakes.FakeLLM` raises if it is reached at all,
which is how "the heuristic path is genuinely the default" is asserted rather
than assumed.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from axiom.agent.classifier import classify, classify_heuristic
from axiom.agent.evaluator import (
    is_sufficient,
    max_achievable_rrf_score,
    next_plan,
    sufficiency_detail,
)
from axiom.agent.loop import (
    STOP_BUDGET,
    STOP_EMPTY_QUERY,
    STOP_MAX_PASSES,
    STOP_NO_NEW_QUERY,
    STOP_PASS_FAILED,
    STOP_SUFFICIENT,
    run,
)
from axiom.agent.planner import (
    MAX_QUERY_CHARS,
    STOPWORDS,
    build_plan,
    extract_identifiers,
    normalise_query,
    plans_differ,
    refine_plan,
)
from axiom.config import get_settings
from axiom.schema import FusedResult, QueryType, SignalKind

from .fakes import FakeCrossEncoder, FakeLLM, ScriptedBackend, make_chunk, make_plan, ranked
from .helpers import hex32

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
DENSE, SPARSE, STRUCT = SignalKind.DENSE, SignalKind.SPARSE, SignalKind.STRUCTURAL

#: The three PRD archetypes, verbatim from PRD.md section 1.1 / TC-001..TC-003.
Q1 = "How is the input preprocessed before going to the main function?"
Q2 = "Which files call tool XYZ before tool ABC?"
Q3 = "Where is the Bluetooth-settings deeplink used?"


@pytest.fixture
def settings():
    """Default profile with the LLM and reranker off -- the evaluator's laptop."""
    return get_settings("default", configs_dir=CONFIGS, llm_enabled=False, reranker_enabled=False)


# ---------------------------------------------------------------------------
# US-1 / TC-001..TC-004 -- classification with the LLM off
# ---------------------------------------------------------------------------


class TestClassification:
    def test_tc001_semantic_archetype(self, settings) -> None:
        """TC-001 / archetype Q1 -> ``SEMANTIC``.

        The interesting part is the word "before": it is an ordering cue, but
        the classifier scopes ordering words to ``calls|invokes before``. A
        standalone reading would fire both cue sets here and collapse Q1 to
        HYBRID, which is exactly the misclassification the cue table forbids.
        """
        assert build_plan(Q1, settings).query_type is QueryType.SEMANTIC

    def test_tc002_structural_archetype_with_both_identifiers(self, settings) -> None:
        """TC-002 / archetype Q2 -> ``STRUCTURAL``, and both symbols extracted."""
        plan = build_plan(Q2, settings)
        assert plan.query_type is QueryType.STRUCTURAL
        assert {"XYZ", "ABC"} <= set(plan.extracted_identifiers)

    def test_tc003_usage_archetype(self, settings) -> None:
        """TC-003 / archetype Q3 -> ``USAGE``: one identifier, no other cue."""
        assert build_plan(Q3, settings).query_type is QueryType.USAGE

    def test_tc004_multi_intent_query_is_hybrid(self, settings) -> None:
        """TC-004: two cue sets fire, so ambiguity resolves to HYBRID by rule.

        Schema.md section 3.1: a wrong confident classification costs more NDCG
        than a correct-but-flat weighting.
        """
        query = "How does the auth module validate tokens before calling the API?"
        assert build_plan(query, settings).query_type is QueryType.HYBRID

    def test_classification_selects_the_locked_weight_vector(self, settings) -> None:
        """The whole point of classifying: it picks the row of the §5 table."""
        plan = build_plan(Q2, settings)
        assert plan.strategy_weights[STRUCT] == 0.60

    def test_no_cue_at_all_falls_back_to_hybrid(self, settings) -> None:
        assert classify_heuristic("qqqq zzzz", []) is QueryType.HYBRID

    def test_a_quoted_literal_forces_usage(self, settings) -> None:
        """The strongest USAGE tell: the user typed the exact corpus token."""
        assert classify_heuristic("find 'bluetooth-settings' somewhere", []) is QueryType.USAGE

    def test_identifier_spelling_does_not_leak_into_the_cue_sets(self, settings) -> None:
        """ "Which files import utils.normalize?" carries the semantic stem
        ``normali[sz]`` *inside a symbol*. Masking identifiers first is what
        keeps this an unambiguous STRUCTURAL query."""
        plan = build_plan("Which files import utils.normalize?", settings)
        assert plan.query_type is QueryType.STRUCTURAL

    def test_classify_never_raises_on_hostile_input(self, settings) -> None:
        for query in ["", "   ", "🔵🔵🔵", "(((((", "x" * 10_000, "a" * 3 + "\x00"]:
            assert classify(query, settings) in set(QueryType)

    def test_the_llm_is_not_reached_when_disabled(self, settings, monkeypatch) -> None:
        """TestPlan.md section 1.3: ``FakeLLM`` raises if called while disabled.

        Installed as ``agent.llm.get_llm``'s return value, so any path that
        would consult the model on an offline laptop fails the test loudly.
        """
        import axiom.agent.llm as llm_module

        monkeypatch.setattr(llm_module, "get_llm", lambda _s: FakeLLM(enabled=False))
        assert settings.llm_enabled is False
        assert build_plan(Q1, settings).query_type is QueryType.SEMANTIC


# ---------------------------------------------------------------------------
# TC-007..TC-011 -- the planner's handling of query text
# ---------------------------------------------------------------------------


class TestPlanner:
    def test_tc007_identifier_extraction_covers_all_four_shapes(self) -> None:
        """TC-007: camelCase, snake_case, dotted, quoted -- order preserved."""
        found = extract_identifiers(
            "call handleDeeplink, parse_input, utils.normalize and 'MAX_RETRY'"
        )
        assert found == ["handleDeeplink", "parse_input", "utils.normalize", "MAX_RETRY"]

    def test_tc008_no_extracted_identifier_is_a_stopword(self, fixtures_dir: Path) -> None:
        """TC-008: run the extractor over the 50-query fixture.

        Two clauses: no identifier is an English stopword, and none is a single
        character unless it came out of quotes.
        """
        lines = [
            line.strip()
            for line in (fixtures_dir / "queries.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert len(lines) >= 50, "the TC-008 fixture must carry 50 queries"

        for query in lines:
            for identifier in extract_identifiers(query):
                assert identifier.lower() not in STOPWORDS, f"{identifier!r} from {query!r}"
                assert len(identifier) >= 2, f"{identifier!r} from {query!r}"

    def test_tc009_empty_and_whitespace_queries_normalise_to_empty(self) -> None:
        """TC-009 asks for an ``EmptyQueryError``. No such class exists in the
        frozen ``core.errors``, and Rule 3 forbids raising on user input, so the
        implemented contract is: normalise to ``""`` and let the loop return
        ``stop_reason="empty_query"``. Asserted as implemented; flagged in the
        report."""
        assert normalise_query("") == ""
        assert normalise_query("   \t\n ") == ""

    def test_tc009_an_empty_query_still_produces_a_well_formed_plan(self, settings) -> None:
        plan = build_plan("", settings)
        assert plan.original_query  # min_length=1 is satisfied by the placeholder
        assert abs(sum(plan.strategy_weights.values()) - 1.0) < 1e-9

    def test_tc010_a_10000_char_query_is_truncated_not_fatal(self, settings) -> None:
        """TC-010: truncated to ``MAX_QUERY_CHARS`` (2048), plan still valid."""
        plan = build_plan("x" * 10_000, settings)
        assert len(plan.original_query) == MAX_QUERY_CHARS == 2048

    def test_tc011_unicode_and_emoji_survive_planning(self, settings) -> None:
        """TC-011: no ``UnicodeError`` anywhere on the plan path."""
        plan = build_plan("où est le déeplink 🔵 Bluetooth ?", settings)
        assert plan.original_query
        assert plan.query_type in set(QueryType)

    def test_sub_queries_stay_within_the_fr03_cap(self, settings) -> None:
        """FR-03: at most three ordered sub-queries; each costs a full fan-out."""
        plan = build_plan(Q2, settings)
        assert len(plan.sub_queries) <= 3

    def test_expansion_terms_are_appended_for_a_semantic_query(self, settings) -> None:
        plan = build_plan(Q1, settings)
        assert plan.expansion_terms, "a prose query should expand its vocabulary"
        assert len(plan.expansion_terms) <= 12

    def test_planning_is_reproducible(self, settings) -> None:
        """NFR-08: a plan must be reconstructable from the logs."""
        assert build_plan(Q2, settings) == build_plan(Q2, settings)


# ---------------------------------------------------------------------------
# TC-087 -- the sufficiency predicate, both branches
# ---------------------------------------------------------------------------


def _fused(score: float, label: str) -> FusedResult:
    """A rerank-scored candidate at a chosen score."""
    return FusedResult(
        chunk_id=hex32(label),
        rrf_score=0.01,
        rerank_score=score,
        contributions={DENSE: 1},
        dominant_signal=DENSE,
    )


class TestSufficiency:
    def test_tc087_a_top1_below_the_threshold_refines(self, settings) -> None:
        """TC-087(a): top-1 = 0.34 < 0.35 -> refine, even with plenty above the floor."""
        results = [_fused(0.34, "a"), _fused(0.33, "b"), _fused(0.32, "c"), _fused(0.31, "d")]
        assert settings.agent_sufficiency_top1 == 0.35
        assert is_sufficient(results, settings, used_rerank=True) is False

    def test_tc087_a_clear_top1_with_enough_support_stops(self, settings) -> None:
        """TC-087(b): top-1 = 0.36 with five results above 0.20 -> stop."""
        results = [_fused(0.36, "a")] + [_fused(0.25, f"x{i}") for i in range(4)]
        assert is_sufficient(results, settings, used_rerank=True) is True

    def test_tc087_a_strong_top1_with_too_little_support_refines(self, settings) -> None:
        """TC-087(c): top-1 = 0.90 but only two results clear 0.20 -> refine.

        This is the branch that distinguishes "found the answer" from "found one
        lucky chunk", and it is the reason the predicate is a conjunction.
        """
        results = [_fused(0.90, "a"), _fused(0.50, "b"), _fused(0.05, "c")]
        assert settings.agent_sufficiency_min_results == 3
        assert is_sufficient(results, settings, used_rerank=True) is False

    def test_the_floor_is_strict_not_inclusive(self, settings) -> None:
        """A score exactly at 0.20 does not clear a floor of 0.20."""
        at_floor = [_fused(0.90, "a")] + [_fused(0.20, f"x{i}") for i in range(5)]
        assert is_sufficient(at_floor, settings, used_rerank=True) is False

    def test_thresholds_are_read_from_settings(self, settings) -> None:
        """AP-07 again: lowering the threshold must change the verdict."""
        results = [_fused(0.30, "a"), _fused(0.25, "b"), _fused(0.24, "c")]
        assert is_sufficient(results, settings, used_rerank=True) is False
        relaxed = settings.model_copy(update={"agent_sufficiency_top1": 0.25})
        assert is_sufficient(results, relaxed, used_rerank=True) is True

    def test_an_empty_candidate_list_is_insufficient(self, settings) -> None:
        assert is_sufficient([], settings, used_rerank=True) is False
        assert is_sufficient([], settings, used_rerank=False) is False

    def test_the_predicate_reads_scores_only(self, settings) -> None:
        """ADR-008: never chunk text, never a path -- which is what keeps the
        loop's cost independent of corpus size. Asserted structurally: the
        predicate accepts ``FusedResult``, which carries no text at all."""
        assert "text" not in FusedResult.model_fields
        assert is_sufficient([_fused(0.9, "a")], settings, used_rerank=True) is False

    def test_the_verdict_is_order_independent(self, settings) -> None:
        """Rules.md Rule 2: stages are pure; the predicate takes a max and a count."""
        results = [_fused(0.36, "a")] + [_fused(0.25, f"x{i}") for i in range(4)]
        assert is_sufficient(results, settings, True) == is_sufficient(
            list(reversed(results)), settings, True
        )

    def test_the_third_rung_fires_when_the_rrf_predicate_cannot_be_met(self, settings) -> None:
        """Rules.md 3, "Sufficiency check": declare sufficient, never loop blindly.

        The PLACEHOLDER thresholds are on the rerank scale. Max achievable
        ``rrf_score`` is ``sum(w)/(k+1) = 1/61 ~ 0.0164``, so a 0.35 threshold is
        not strict on the no-reranker path -- it is *unsatisfiable*, and the
        second rung can only ever answer "insufficient". Left there, every query
        without a live reranker burns the full pass budget to learn nothing,
        which is the blind loop the third rung exists to prevent.
        """
        rrf_only = [
            FusedResult(
                chunk_id=hex32(f"r{i}"),
                rrf_score=1 / 61,
                contributions={DENSE: 1},
                dominant_signal=DENSE,
            )
            for i in range(5)
        ]
        ceiling = max_achievable_rrf_score(settings)
        assert max(r.rrf_score for r in rrf_only) <= ceiling
        assert settings.agent_sufficiency_top1 > ceiling, "the premise of the third rung"
        assert is_sufficient(rrf_only, settings, used_rerank=False) is True

        detail = sufficiency_detail(rrf_only, settings, used_rerank=False)
        assert detail["rung"] == "declare_sufficient"
        assert detail["sufficient"] is True

    def test_the_second_rung_still_judges_rrf_when_the_thresholds_fit(self, settings) -> None:
        """TechSpec 5.3: the thresholds are shared, only the score field changes.

        The third rung is a guard against an unmeetable threshold, not a blanket
        surrender. Once OQ-10/T-141 replaces the PLACEHOLDER values with ones on
        the RRF scale, the second rung discriminates again -- and must, or the
        agent stops refining altogether.
        """
        on_scale = settings.model_copy(
            update={"agent_sufficiency_top1": 0.015, "agent_sufficiency_floor": 0.010}
        )
        assert on_scale.agent_sufficiency_top1 < max_achievable_rrf_score(on_scale)

        strong = [
            FusedResult(
                chunk_id=hex32(f"s{i}"),
                rrf_score=1 / 61,
                contributions={DENSE: 1},
                dominant_signal=DENSE,
            )
            for i in range(5)
        ]
        assert is_sufficient(strong, on_scale, used_rerank=False) is True
        assert sufficiency_detail(strong, on_scale, used_rerank=False)["rung"] == (
            "rrf_score_predicate"
        )

        weak = [
            FusedResult(
                chunk_id=hex32(f"w{i}"),
                rrf_score=0.002,
                contributions={DENSE: 40},
                dominant_signal=DENSE,
            )
            for i in range(5)
        ]
        assert is_sufficient(weak, on_scale, used_rerank=False) is False

    def test_an_empty_candidate_list_is_never_declared_sufficient(self, settings) -> None:
        """The third rung must not turn "nothing was found" into "good enough"."""
        assert is_sufficient([], settings, used_rerank=False) is False
        assert sufficiency_detail([], settings, used_rerank=False)["sufficient"] is False

    def test_sufficiency_detail_reports_the_numbers_behind_the_verdict(self, settings) -> None:
        """NFR-10 wants the decision auditable, not inferred."""
        results = [_fused(0.36, "a")] + [_fused(0.25, f"x{i}") for i in range(4)]
        detail = sufficiency_detail(results, settings, used_rerank=True)
        assert detail["score_field"] == "rerank_score"
        assert detail["top1"] == pytest.approx(0.36)
        assert detail["above_floor"] == 5
        assert detail["sufficient"] is True
        assert detail["top1_threshold"] == settings.agent_sufficiency_top1

    def test_sufficiency_detail_names_rrf_score_when_no_model_ran(self, settings) -> None:
        detail = sufficiency_detail([], settings, used_rerank=False)
        assert detail["score_field"] == "rrf_score"
        assert detail["sufficient"] is False


# ---------------------------------------------------------------------------
# TC-089 -- refinement must change the retrieval
# ---------------------------------------------------------------------------


class TestRefinement:
    def test_refine_broadens_the_plan(self, settings) -> None:
        plan = build_plan(Q1, settings)
        refined = refine_plan(plan, [], settings)
        assert refined.original_query == plan.original_query, "the query is never mutated"
        assert plans_differ(plan, refined)

    def test_next_plan_returns_none_when_the_rewrite_repeats(self, settings) -> None:
        """TC-089's guard: an identical retrieval is not worth a pass."""
        plan = build_plan(Q1, settings)
        refined = refine_plan(plan, [], settings)
        assert next_plan(refined, [], settings) is None or plans_differ(
            refined, refine_plan(refined, [], settings)
        )

    def test_plans_differ_compares_what_gets_retrieved(self, settings) -> None:
        base = make_plan("q", expansion_terms=["alpha"])
        assert plans_differ(base, make_plan("q", expansion_terms=["alpha", "beta"]))
        assert not plans_differ(base, make_plan("q", expansion_terms=["alpha"]))


# ---------------------------------------------------------------------------
# TC-086 / TC-088 / TC-089 / TC-090 -- the loop
# ---------------------------------------------------------------------------


def _backend(*steps, chunks=None) -> ScriptedBackend:
    return ScriptedBackend(list(steps) or [{}], chunks or {})


class TestLoop:
    def test_tc086_terminates_within_max_passes_and_the_wall_clock(self, settings) -> None:
        """TC-086: 20 runs, adversarial input, never exceeds the caps.

        A 10k-character query, an empty corpus, and a scorer that can never
        satisfy sufficiency -- the configuration designed to make the loop spin.
        """
        encoder = FakeCrossEncoder(default=0.05)
        for _ in range(20):
            started = time.monotonic()
            outcome = run("x" * 10_000, _backend({}), settings, encoder=encoder)
            elapsed = time.monotonic() - started

            assert outcome.passes_used <= settings.agent_max_passes == 2
            assert elapsed < 5.0
            assert outcome.stop_reason in {
                STOP_MAX_PASSES,
                STOP_BUDGET,
                STOP_SUFFICIENT,
                STOP_NO_NEW_QUERY,
            }

    def test_tc086_a_populated_corpus_also_terminates(self, settings) -> None:
        """The same caps with candidates present, so the loop actually fuses."""
        chunk_ids = [hex32(f"c{i}") for i in range(5)]
        chunks = {
            cid: make_chunk(f"function f{i}() {{ return {i}; }}") for i, cid in enumerate(chunk_ids)
        }
        chunks = {cid: chunk for cid, chunk in zip(chunk_ids, chunks.values(), strict=True)}
        step = {DENSE: ranked(DENSE, chunk_ids), SPARSE: ranked(SPARSE, list(reversed(chunk_ids)))}
        outcome = run(
            Q1, _backend(step, chunks=chunks), settings, encoder=FakeCrossEncoder(default=0.05)
        )
        assert 1 <= outcome.passes_used <= 2
        assert outcome.results

    def test_agent_disabled_collapses_to_exactly_one_pass(self, settings) -> None:
        """The ``eval`` profile: refinement is the ablation, not the retrieval."""
        backend = _backend({})
        single = settings.model_copy(update={"agent_enabled": False})
        outcome = run(Q1, backend, single)
        assert outcome.passes_used <= 1
        assert len(backend.plans) == 1

    def test_an_empty_query_returns_empty_query_and_runs_no_pass(self, settings) -> None:
        """TC-009's degraded form. ``passes_used == 0`` contradicts API.md's
        stated ``1..AXIOM_AGENT_MAX_PASSES`` range; ``stop_reason``
        disambiguates it. Flagged in the report."""
        backend = _backend({})
        outcome = run("   ", backend, settings)
        assert outcome.stop_reason == STOP_EMPTY_QUERY
        assert outcome.passes_used == 0
        assert outcome.results == []
        assert backend.plans == [], "an empty query must not touch the index"

    def test_tc088_an_expired_budget_returns_the_earlier_results(self, settings) -> None:
        """TC-088: the check is *before* starting a new pass, and the result set
        is pass 1's, not an empty list and not an error.

        The clock is advanced by giving the loop a **zero** millisecond budget,
        which ``Deadline`` reports as expired from its first tick. That is
        deterministic where a small non-zero budget is a race: no monkeypatched
        clock, no ``sleep``, and no dependence on how fast the box is. Pass 1 is
        exempt by design (AP-03: a query must return something even under a
        budget that was already spent), so exactly one pass runs.
        """
        chunk_ids = [hex32(f"b{i}") for i in range(4)]
        chunks = {
            cid: make_chunk(f"function b{i}() {{ return {i}; }}") for i, cid in enumerate(chunk_ids)
        }
        chunks = dict(zip(chunk_ids, chunks.values(), strict=True))
        step = {DENSE: ranked(DENSE, chunk_ids)}

        # The thresholds are moved onto the RRF scale for this test. With a zero
        # budget the reranker never runs, so the evaluator lands on its
        # passthrough rung -- and with the PLACEHOLDER 0.35 threshold that rung
        # declares sufficient (it is unmeetable on the RRF scale), which would
        # end the loop before the budget branch is ever reached. On-scale
        # thresholds keep the predicate discriminating, so what this test
        # observes is the budget check and nothing else.
        spent = settings.model_copy(
            update={
                "agent_wall_clock_ms": 0,
                "agent_sufficiency_top1": 0.015,
                "agent_sufficiency_floor": 0.010,
                "agent_sufficiency_min_results": len(chunk_ids) + 1,
            }
        )
        backend = _backend(step, chunks=chunks)
        outcome = run(Q1, backend, spent, encoder=FakeCrossEncoder(default=0.05))

        assert outcome.passes_used == 1
        assert len(backend.plans) == 1, "pass 2 must not even reach the index"
        assert outcome.results, "budget exhaustion must not empty the result set"
        assert outcome.stop_reason == STOP_BUDGET
        assert any("budget" in note for note in outcome.degradations)

    def test_tc089_the_second_pass_issues_a_different_query(self, settings) -> None:
        """TC-089: capture the plans the loop handed the backend and compare."""
        backend = _backend({})
        outcome = run(Q1, backend, settings, encoder=FakeCrossEncoder(default=0.05))
        if outcome.passes_used >= 2:
            first, second = backend.plans[0], backend.plans[1]
            assert plans_differ(first, second)
            assert first.original_query == second.original_query, "the query text is verbatim"
        else:
            assert outcome.stop_reason in {STOP_NO_NEW_QUERY, STOP_SUFFICIENT, STOP_BUDGET}

    def test_sufficient_results_stop_after_one_pass(self, settings) -> None:
        """The other branch of the loop: a live reranker that clears both
        thresholds ends the query in one pass."""
        chunk_ids = [hex32(f"s{i}") for i in range(5)]
        chunks = {
            cid: make_chunk(f"function preprocessInput{i}() {{ return {i}; }}")
            for i, cid in enumerate(chunk_ids)
        }
        chunks = dict(zip(chunk_ids, chunks.values(), strict=True))
        backend = _backend({DENSE: ranked(DENSE, chunk_ids)}, chunks=chunks)

        outcome = run(
            Q1,
            backend,
            settings.model_copy(update={"reranker_enabled": True}),
            encoder=FakeCrossEncoder(default=0.8),
        )
        assert outcome.stop_reason == STOP_SUFFICIENT
        assert outcome.passes_used == 1
        assert outcome.used_rerank is True
        assert outcome.score_field == "rerank_score"

    def test_a_backend_that_raises_degrades_rather_than_propagating(self, settings) -> None:
        """Rule 3. A traceback reaching the demo is the failure this prevents."""
        backend = ScriptedBackend([{}], raises=RuntimeError("index on fire"))
        outcome = run(Q1, backend, settings)
        assert outcome.stop_reason == STOP_PASS_FAILED
        assert outcome.results == []
        assert outcome.passes_used == 0
        # The exception detail is logged (one ERROR with a traceback, one
        # log_degradation WARNING) *and* survives into
        # ``AgentOutcome.degradations``. It used not to: ``_empty_outcome``
        # rebuilt the list as ``[stop_reason]`` and dropped everything the loop
        # had accumulated, which discarded the only line naming the exception.
        assert outcome.degradations[0] == STOP_PASS_FAILED
        assert "pass 1 raised RuntimeError" in outcome.degradations
        # The same call used to drop ``elapsed_ms`` on the floor too.
        assert outcome.elapsed_ms > 0.0

    def test_tc090_the_returned_list_has_no_duplicates_and_records_provenance(
        self, settings
    ) -> None:
        """TC-090, as the loop actually specifies itself.

        The row asks for results "merged across passes without duplicates".
        TechSpec section 5.3, Rules.md AP-03 and Design.md section 5.2 all
        specify **best-of replacement** instead -- the winning pass's list is
        returned whole. What is asserted is therefore the testable half of the
        row (no duplicate ``chunk_id``; each pass's provenance is recorded and
        the winner is named) plus the documented behaviour. The row needs
        rewording; see the report.
        """
        chunk_ids = [hex32(f"p{i}") for i in range(6)]
        chunks = {
            cid: make_chunk(f"function p{i}() {{ return {i}; }}") for i, cid in enumerate(chunk_ids)
        }
        chunks = dict(zip(chunk_ids, chunks.values(), strict=True))
        pass1 = {DENSE: ranked(DENSE, chunk_ids[:4])}
        pass2 = {DENSE: ranked(DENSE, chunk_ids[2:])}  # overlapping candidate sets

        outcome = run(
            Q1,
            _backend(pass1, pass2, chunks=chunks),
            settings,
            encoder=FakeCrossEncoder(default=0.05),
        )

        ids = [result.chunk_id for result in outcome.results]
        assert len(set(ids)) == len(ids), "duplicate chunk_id in the returned list"
        assert outcome.winning_pass in {record.pass_no for record in outcome.passes}
        assert len(outcome.passes) == outcome.passes_used
        assert [record.pass_no for record in outcome.passes] == list(
            range(1, outcome.passes_used + 1)
        )

    def test_the_outcome_is_json_serialisable_for_the_timings_block(self, settings) -> None:
        """NFR-10: every ``--json`` response carries this, so it must serialise."""
        import json

        outcome = run(Q1, _backend({}), settings)
        json.dumps(outcome.as_dict(), default=str)

    def test_repeated_runs_are_deterministic(self, settings) -> None:
        """NFR-08 at the loop level: same input, same ranking and stop reason."""
        chunk_ids = [hex32(f"d{i}") for i in range(5)]
        chunks = {
            cid: make_chunk(f"function d{i}() {{ return {i}; }}") for i, cid in enumerate(chunk_ids)
        }
        chunks = dict(zip(chunk_ids, chunks.values(), strict=True))
        step = {DENSE: ranked(DENSE, chunk_ids), SPARSE: ranked(SPARSE, chunk_ids[::-1])}

        signatures = set()
        for _ in range(3):
            outcome = run(
                Q2, _backend(step, chunks=chunks), settings, encoder=FakeCrossEncoder(default=0.05)
            )
            signatures.add(
                (
                    tuple((r.chunk_id, r.rrf_score) for r in outcome.results),
                    outcome.stop_reason,
                    outcome.passes_used,
                )
            )
        assert len(signatures) == 1


# ---------------------------------------------------------------------------
# The real ONNX cross-encoder's token budget
# ---------------------------------------------------------------------------


class TestCrossEncoderTokenBudget:
    """``rerank_max_tokens`` is one number; the reranker ladder has several rungs.

    ``bge-reranker-v2-m3`` accepts 8192 tokens, ``ms-marco-MiniLM-L-6-v2`` stops
    at 512. Feeding the configured 1536 to the smaller model does not truncate,
    it fails the ONNX graph outright -- which ``rerank_detailed`` catches and
    turns into a passthrough, so the reranker reads as configured-and-running
    while scoring nothing. These pin the cap that prevents that.
    """

    @staticmethod
    def _encoder(tmp_path: Path, config: str | None, max_tokens: int = 1536):
        from axiom.rerank.cross_encoder import OnnxCrossEncoder

        artifact = tmp_path / "model-int8"
        artifact.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (artifact / "config.json").write_text(config, encoding="utf-8")
        return OnnxCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            onnx_path=artifact / "model.onnx",
            max_tokens=max_tokens,
        )

    def test_the_models_own_limit_caps_the_configured_budget(self, tmp_path: Path) -> None:
        encoder = self._encoder(tmp_path, '{"max_position_embeddings": 512}')
        assert encoder.max_tokens == 1536
        assert encoder.effective_max_tokens == 512

    def test_a_larger_model_limit_does_not_raise_the_configured_budget(
        self, tmp_path: Path
    ) -> None:
        """The cap is a ceiling, never a floor -- config stays the operator's lever."""
        encoder = self._encoder(tmp_path, '{"max_position_embeddings": 8192}')
        assert encoder.effective_max_tokens == 1536

    @pytest.mark.parametrize(
        "config",
        [None, "not json at all", "{}", '{"max_position_embeddings": null}',
         '{"max_position_embeddings": 0}'],
    )
    def test_an_unreadable_config_leaves_the_budget_alone(
        self, tmp_path: Path, config: str | None
    ) -> None:
        """A missing config is not evidence of a smaller limit."""
        assert self._encoder(tmp_path, config).effective_max_tokens == 1536

    def test_the_lookup_is_cached_and_survives_a_deleted_config(self, tmp_path: Path) -> None:
        """Read once: this sits on the per-query path."""
        encoder = self._encoder(tmp_path, '{"max_position_embeddings": 512}')
        assert encoder.effective_max_tokens == 512
        (tmp_path / "model-int8" / "config.json").unlink()
        assert encoder.effective_max_tokens == 512

    @pytest.mark.slow
    def test_a_query_longer_than_the_budget_scores_instead_of_raising(self) -> None:
        """``only_second`` raises when the *query* alone overflows the budget.

        AppsRetrieval queries are whole problem statements, so a query over 512
        tokens is the ordinary case there, not an edge one -- and the failure is
        not a truncation, it is ``Sequence to truncate too short to respect the
        provided max_length`` propagating out of the tokenizer and surfacing as a
        rerank that fell through to passthrough.

        Runs against the real exported artifact, which is gitignored (Setup.md
        section 6.1), so it skips where the export has not been done rather than
        asserting on a mock that cannot reproduce the bug.

        On macOS/arm64 onnxruntime may print ``recursive_mutex lock failed`` as
        the interpreter tears the session down. That is ORT's own shutdown path
        firing after pytest has already reported, the exit code is still 0, and
        it is not this test failing.
        """
        pytest.importorskip("onnxruntime", reason="needs the retrieval extra")
        pytest.importorskip("transformers", reason="needs the retrieval extra")

        from axiom.rerank.cross_encoder import OnnxCrossEncoder, _derive_onnx_path

        onnx_path = _derive_onnx_path("cross-encoder/ms-marco-MiniLM-L-6-v2")
        if not onnx_path.is_file():
            pytest.skip(f"no exported artifact at {onnx_path} (Setup.md section 6.1)")

        encoder = OnnxCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            onnx_path=onnx_path,
            max_tokens=1536,
        )
        assert encoder.effective_max_tokens == 512, "the 512-token model must cap the 1536 budget"

        long_query = "how is the user utterance normalised before dispatch " * 200
        long_document = "function normalizeInput(value) { return value.trim() }\n" * 200
        scores = encoder.score_pairs([(long_query, long_document)])
        assert len(scores) == 1
        assert isinstance(scores[0], float)


class TestRerankBudget:
    """``reranker_timeout_ms`` bounds scoring, not the one-time model load.

    Loading the ONNX session and tokenizer takes seconds in a fresh process. With
    the clock started before the load, the first query of every CLI run timed out
    with 0 pairs scored and fell through to passthrough, although scoring 25 pairs
    takes about half a second once the model is resident.
    """

    @pytest.mark.smoke
    def test_a_slow_model_load_does_not_spend_the_scoring_budget(self, monkeypatch) -> None:
        from axiom.rerank import RerankMode, cross_encoder

        scorer = FakeCrossEncoder(default=1.0)

        def slow_load(_settings):
            time.sleep(0.3)
            return scorer

        monkeypatch.setattr(cross_encoder, "load_cross_encoder", slow_load)
        settings = get_settings(
            "default", configs_dir=CONFIGS, llm_enabled=False, reranker_timeout_ms=200
        )
        chunks = [make_chunk(f"function f{i}() {{ return {i}; }}", symbol=f"f{i}") for i in range(3)]
        candidates = [
            FusedResult(
                chunk_id=chunk.chunk_id,
                rrf_score=1 / (61 + i),
                contributions={DENSE: i + 1},
                dominant_signal=DENSE,
            )
            for i, chunk in enumerate(chunks)
        ]

        outcome = cross_encoder.rerank_detailed(
            "which function returns one", candidates, {c.chunk_id: c for c in chunks}, settings
        )

        assert outcome.mode is RerankMode.CROSS_ENCODER
        assert scorer.pairs_scored == 3
