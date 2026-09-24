"""The bounded agentic refinement loop (FR-13, ADR-007, NFR-04).

This is the module the hackathon theme is named after, and the one feature that
is never cut. It is also the module where "agentic" most easily turns into "hangs
forever", so the shape below is deliberately boring::

    deadline = Deadline(settings.agent_wall_clock_ms)          # 5000 ms
    for pass_no in 1 .. settings.agent_max_passes:             # 2
        if pass_no > 1 and deadline.expired: break
        candidate = fan_out -> fuse -> hydrate -> rerank
        if candidate beats best: best = candidate
        if sufficient(candidate): break
        plan = refine(plan, candidate)      # None -> stop, nothing new to try
    return best

Four independent things stop it, and the loop is correct if *any one* of them
fires: the pass counter, the monotonic deadline, the sufficiency predicate, and
the planner reporting that its rewrite is identical to the query already run.
TestPlan.md TC-086 drives it with a 10,000-character query, an empty corpus and a
scorer rigged never to satisfy sufficiency, and requires termination anyway --
so no exit condition may depend on retrieval quality improving.

**The deadline is checked before starting a pass, never after finishing one.**
Checking afterwards would let a pass that began at 4.9 s run to completion and
report a 9 s query as within budget. Pass 1 is exempt from the check for the
same reason Rules.md AP-03's reference loop seeds ``best`` from a retrieval taken
outside the loop: a query must return *something*, and a zero-length budget must
degrade to one pass rather than to an empty answer.

**Best-of, not last.** A refinement is a hypothesis, not an improvement. Pass 2
wins only if it measurably beats pass 1 on the score field that actually ran
(Design.md section 5.2, Appflow.md Flow 3).

Nothing here imports an optional dependency. The loop reaches retrieval only
through the :class:`RetrievalBackend` protocol its caller supplies, so it can be
exercised against a dictionary of canned results with nothing installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from axiom.agent.evaluator import is_sufficient, next_plan, sufficiency_detail
from axiom.agent.planner import build_plan, normalise_query
from axiom.config import Settings
from axiom.core.logging import get_logger, log_degradation
from axiom.core.timing import Deadline, TimingLedger
from axiom.rerank import RerankMode, RerankOutcome, rerank_detailed
from axiom.rerank.cross_encoder import PairScorer
from axiom.retrieval.fusion import FusionOutcome, reciprocal_rank_fusion
from axiom.schema import Chunk, FusedResult, QueryPlan, ScoredChunk, SignalKind

_LOG = get_logger("agent.loop")

STAGE = "agent"

#: ``stop_reason`` values. TestPlan.md TC-086 pins the first three; the fourth is
#: Appflow.md Flow 3 / TC-089; the fifth is the empty-query short circuit that
#: TechSpecifications.md section 3.7 assigns to this layer; the sixth exists so
#: that a backend violating its own protocol is visible in the response rather
#: than only in the log.
STOP_SUFFICIENT = "sufficient"
STOP_MAX_PASSES = "max_passes"
STOP_BUDGET = "budget_exhausted"
STOP_NO_NEW_QUERY = "no_new_query"
STOP_EMPTY_QUERY = "empty_query"
STOP_PASS_FAILED = "pass_failed"


class RetrievalBackend(Protocol):
    """What the loop needs from the index layer, and nothing more.

    A protocol rather than a concrete dependency for two reasons. First, it keeps
    every optional dependency behind the caller: ``pipeline.py`` owns the FAISS
    handle, the bm25s postings and the SQLite connection, and this module never
    learns they exist. Second, it makes TC-086's pathological corpus a
    three-line fake instead of a built index.

    Implementations must honour the same contract the retrievers do: an
    unavailable signal is **absent from the returned mapping or maps to an empty
    list**, never zero-scored filler, so that fusion's empty-signal
    renormalisation stays meaningful (TechSpecifications.md section 5.1.2).
    """

    def fan_out(self, plan: QueryPlan) -> Mapping[SignalKind, Sequence[ScoredChunk]]:
        """Run every enabled signal for one plan and return their ranked lists."""
        ...

    def hydrate(self, chunk_ids: Sequence[str]) -> Mapping[str, Chunk]:
        """Resolve fused ids to full chunk bodies for the reranker (Appflow step 5)."""
        ...


@dataclass(frozen=True)
class PassRecord:
    """What one pass did, kept so the demo can show *why* a second pass ran.

    US-5 asks the jury to see the agent reason, and "passes_used: 2" alone does
    not show reasoning -- the rewritten query, the score that failed the
    threshold, and which score field was being thresholded do.
    """

    pass_no: int
    plan: QueryPlan
    queries: list[str]
    candidates: int
    top1: float
    score_field: str
    rerank_mode: str
    sufficient: bool
    elapsed_ms: float
    dropped_signals: list[str] = field(default_factory=list)
    per_signal_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Serialise for the ``timings`` block and the UI's reasoning panel."""
        return {
            "pass": self.pass_no,
            "queries": list(self.queries),
            "query_type": self.plan.query_type.value,
            "candidates": self.candidates,
            "top1": round(self.top1, 6),
            "score_field": self.score_field,
            "rerank_mode": self.rerank_mode,
            "sufficient": self.sufficient,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "dropped_signals": list(self.dropped_signals),
            "per_signal_counts": dict(self.per_signal_counts),
        }


@dataclass(frozen=True)
class AgentOutcome:
    """The winning pass's results plus the loop's own audit trail."""

    results: list[FusedResult]
    plan: QueryPlan
    chunks: dict[str, Chunk]
    passes_used: int
    stop_reason: str
    used_rerank: bool
    score_field: str
    rerank_mode: str
    winning_pass: int
    passes: list[PassRecord] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def top1(self) -> float:
        """Best final score, on whichever field the winning pass was judged by."""
        if not self.results:
            return 0.0
        return max(r.final_score for r in self.results)

    def as_dict(self) -> dict[str, object]:
        """The agent's contribution to a ``--json`` response (NFR-10)."""
        return {
            "passes_used": self.passes_used,
            "winning_pass": self.winning_pass,
            "stop_reason": self.stop_reason,
            "score_field": self.score_field,
            "rerank_mode": self.rerank_mode,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "passes": [record.as_dict() for record in self.passes],
            "degradations": list(self.degradations),
        }


@dataclass(frozen=True)
class _PassResult:
    """Internal: one pass's candidate answer, before it is compared to ``best``."""

    record: PassRecord
    results: list[FusedResult]
    plan: QueryPlan
    chunks: dict[str, Chunk]
    fused: FusionOutcome
    rerank: RerankOutcome

    @property
    def quality(self) -> tuple[int, float]:
        """Cross-pass comparison key: calibration first, then top-1 score.

        Comparing a cross-encoder's 0.71 against a passthrough pass's RRF score
        of 0.016 is comparing two different quantities, and the RRF pass would
        always lose on magnitude regardless of whether it was better. So a pass
        whose cross-encoder actually ran outranks one whose did not, and only
        within the same rung does the raw top-1 decide (Design.md section 5.2:
        pass 2 is "only preferred if it measurably is" better).
        """
        calibrated = 1 if self.rerank.mode is RerankMode.CROSS_ENCODER else 0
        top1 = max((r.final_score for r in self.results), default=0.0)
        return (calibrated, top1)


def _empty_outcome(
    plan: QueryPlan,
    reason: str,
    elapsed_ms: float,
    degradations: Sequence[str] = (),
) -> AgentOutcome:
    """A well-formed zero-result answer. Never an exception (Rules.md Rule 3).

    ``degradations`` is whatever the loop accumulated before it gave up. It is
    carried through rather than replaced by ``reason`` alone: when every pass
    raised, ``"pass 1 raised RuntimeError"`` is the only description of *why*
    there are no results, and rebuilding the list from the stop reason discarded
    exactly the line the caller needed.
    """
    return AgentOutcome(
        results=[],
        plan=plan,
        chunks={},
        passes_used=0,
        stop_reason=reason,
        used_rerank=False,
        score_field="rrf_score",
        rerank_mode=RerankMode.PASSTHROUGH.value,
        winning_pass=0,
        degradations=list(dict.fromkeys([reason, *degradations])),
        elapsed_ms=elapsed_ms,
    )


def _run_pass(
    pass_no: int,
    plan: QueryPlan,
    backend: RetrievalBackend,
    settings: Settings,
    *,
    deadline: Deadline,
    ledger: TimingLedger,
    top_k: int | None,
    encoder: PairScorer | None,
) -> _PassResult | None:
    """Execute one full retrieve -> fuse -> hydrate -> rerank pass.

    Returns ``None`` when the deadline expires after fusion but before reranking
    on a refinement pass: TC-088 requires the budget check "before starting a new
    pass **and** before the rerank of a new pass", and the reranker is ~80% of a
    pass's cost, so abandoning a pass there and keeping the previous best is
    strictly better than blowing the budget for a result nobody promised.
    """
    started = perf_counter()
    queries = list(plan.effective_queries)

    with ledger.measure(f"{STAGE}.fan_out", **{"pass": pass_no}) as detail:
        signal_results = backend.fan_out(plan)
        detail["signals"] = {
            signal.value: len(entries or ()) for signal, entries in signal_results.items()
        }

    with ledger.measure("fuse", **{"pass": pass_no}) as detail:
        fused = reciprocal_rank_fusion(signal_results, plan, settings)
        detail["candidates"] = fused.candidate_count
        detail["returned"] = len(fused.results)
        detail["dropped"] = [signal.value for signal in fused.dropped_signals]

    if pass_no > 1 and deadline.expired:
        _LOG.warning(
            "budget expired before rerank of pass %d; keeping the previous best",
            pass_no,
            extra={"axiom_extra": {"stage": STAGE, "pass": pass_no, "event": "budget_mid_pass"}},
        )
        return None

    with ledger.measure("hydrate", **{"pass": pass_no}) as detail:
        chunks = dict(backend.hydrate([result.chunk_id for result in fused.results]))
        detail["hydrated"] = len(chunks)

    outcome = rerank_detailed(
        plan.original_query,
        fused.results,
        chunks,
        settings,
        top_k=top_k,
        encoder=encoder,
        ledger=ledger,
    )

    sufficient = is_sufficient(outcome.results, settings, outcome.model_ran)
    top1 = max((r.final_score for r in outcome.results), default=0.0)
    record = PassRecord(
        pass_no=pass_no,
        plan=fused.plan,
        queries=queries,
        candidates=fused.candidate_count,
        top1=top1,
        score_field=outcome.score_field,
        rerank_mode=outcome.mode.value,
        sufficient=sufficient,
        elapsed_ms=(perf_counter() - started) * 1000.0,
        dropped_signals=[signal.value for signal in fused.dropped_signals],
        per_signal_counts={s.value: n for s, n in fused.per_signal_counts.items()},
    )
    return _PassResult(
        record=record,
        results=list(outcome.results),
        # The plan that gets logged carries fusion's renormalised weights, not
        # the nominal ones (TechSpecifications.md section 5.1.2).
        plan=fused.plan,
        chunks=chunks,
        fused=fused,
        rerank=outcome,
    )


def run(
    query: str,
    backend: RetrievalBackend,
    settings: Settings,
    *,
    plan: QueryPlan | None = None,
    ledger: TimingLedger | None = None,
    top_k: int | None = None,
    encoder: PairScorer | None = None,
) -> AgentOutcome:
    """Run the bounded refinement loop and return the best pass's results.

    Args:
        query: The user's raw query. Normalised (and truncated) by the planner;
            ``QueryPlan.original_query`` then holds the text every pass and
            every log line agrees on.
        backend: Supplies the concurrent fan-out and the hydration map.
        settings: Caps, thresholds and widths. ``agent_enabled=False`` (the
            ``eval`` profile) collapses this to exactly one pass -- refinement is
            the ablation, not the retrieval.
        plan: A pre-built plan, for the API's ``query_type`` override and for
            tests that want to pin the classification.
        ledger: Timing ledger; one is created when absent, and the caller's is
            reused so a query's stages all land in one block (NFR-10).
        top_k: Final result count, overriding ``Settings.top_k_default``.
        encoder: Injected pair scorer, for tests.

    Returns:
        An :class:`AgentOutcome`. Never raises: an empty query, an empty corpus,
        an exploding retriever and an exhausted budget all produce a well-formed
        outcome carrying a ``stop_reason`` that names what happened.
    """
    started = perf_counter()
    ledger = ledger if ledger is not None else TimingLedger()
    deadline = Deadline(settings.agent_wall_clock_ms)

    with ledger.measure("plan") as detail:
        active_plan = plan if plan is not None else build_plan(query, settings)
        detail["query_type"] = active_plan.query_type.value
        detail["identifiers"] = len(active_plan.extracted_identifiers)
        detail["sub_queries"] = len(active_plan.sub_queries)

    if not normalise_query(query):
        # TechSpecifications.md section 3.7: an empty query returns an empty
        # result list with match_reason="empty_query". TC-009 asks for an
        # EmptyQueryError instead, but that class does not exist in the frozen
        # core and Rule 3 forbids raising on user input -- so it degrades here
        # and the API layer maps the stop_reason to its 422.
        _LOG.warning(
            "empty query after normalisation",
            extra={"axiom_extra": {"stage": STAGE, "stop_reason": STOP_EMPTY_QUERY}},
        )
        return _empty_outcome(active_plan, STOP_EMPTY_QUERY, (perf_counter() - started) * 1000.0)

    max_passes = max(1, settings.agent_max_passes if settings.agent_enabled else 1)
    best: _PassResult | None = None
    records: list[PassRecord] = []
    degradations: list[str] = []
    stop_reason = STOP_MAX_PASSES

    for pass_no in range(1, max_passes + 1):
        # Pass 1 is exempt: a query must return something even under a budget
        # that was already exhausted before the loop started (AP-03).
        if pass_no > 1 and deadline.expired:
            stop_reason = STOP_BUDGET
            degradations.append(f"agent budget exhausted before pass {pass_no}")
            break

        try:
            candidate = _run_pass(
                pass_no,
                active_plan,
                backend,
                settings,
                deadline=deadline,
                ledger=ledger,
                top_k=top_k,
                encoder=encoder,
            )
        except Exception as exc:
            # ``pipeline.IndexBackend`` wraps every retriever individually, so
            # this should be unreachable through the real backend. It is caught
            # anyway because the alternative is a traceback reaching the demo:
            # a pass that explodes is one more absent signal, and pass 1's
            # results (if any) remain a perfectly good answer.
            log_degradation(
                _LOG,
                "agent.loop:run",
                f"pass {pass_no} raised {type(exc).__name__}: {exc}",
                "best result from the passes that did complete",
            )
            _LOG.error(
                "agent pass %d failed",
                pass_no,
                exc_info=True,
                extra={"axiom_extra": {"stage": STAGE, "pass": pass_no}},
            )
            degradations.append(f"pass {pass_no} raised {type(exc).__name__}")
            stop_reason = STOP_PASS_FAILED
            break
        if candidate is None:
            stop_reason = STOP_BUDGET
            degradations.append(f"agent budget exhausted during pass {pass_no}")
            break

        records.append(candidate.record)
        degradations.extend(candidate.fused.degradations)
        if candidate.rerank.degraded_reason:
            degradations.append(f"rerank: {candidate.rerank.degraded_reason}")
        if best is None or candidate.quality > best.quality:
            best = candidate

        _LOG.info(
            "agent pass %d complete",
            pass_no,
            extra={
                "axiom_extra": {
                    "stage": STAGE,
                    "pass": pass_no,
                    **sufficiency_detail(candidate.results, settings, candidate.rerank.model_ran),
                    "elapsed_ms": round(deadline.elapsed_ms, 3),
                }
            },
        )

        if candidate.record.sufficient:
            stop_reason = STOP_SUFFICIENT
            break
        if pass_no == max_passes:
            stop_reason = STOP_MAX_PASSES
            break
        if deadline.expired:
            stop_reason = STOP_BUDGET
            degradations.append(f"agent budget exhausted after pass {pass_no}")
            break

        refined = next_plan(active_plan, candidate.results, settings)
        if refined is None:
            # TC-089: the planner has nothing new to try, so a second pass would
            # buy an identical retrieval at full price.
            stop_reason = STOP_NO_NEW_QUERY
            break
        active_plan = refined

    elapsed_ms = (perf_counter() - started) * 1000.0
    if best is None:
        return _empty_outcome(active_plan, stop_reason, elapsed_ms, degradations)

    _LOG.info(
        "agent loop finished: %d pass(es), stop_reason=%s",
        len(records),
        stop_reason,
        extra={
            "axiom_extra": {
                "stage": STAGE,
                "passes_used": len(records),
                "winning_pass": best.record.pass_no,
                "stop_reason": stop_reason,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        },
    )
    return AgentOutcome(
        results=best.results,
        plan=best.plan,
        chunks=best.chunks,
        passes_used=len(records),
        stop_reason=stop_reason,
        used_rerank=best.rerank.model_ran,
        score_field=best.rerank.score_field,
        rerank_mode=best.rerank.mode.value,
        winning_pass=best.record.pass_no,
        passes=records,
        degradations=degradations,
        elapsed_ms=elapsed_ms,
    )


__all__ = [
    "STAGE",
    "STOP_BUDGET",
    "STOP_EMPTY_QUERY",
    "STOP_MAX_PASSES",
    "STOP_NO_NEW_QUERY",
    "STOP_PASS_FAILED",
    "STOP_SUFFICIENT",
    "AgentOutcome",
    "PassRecord",
    "RetrievalBackend",
    "run",
]
