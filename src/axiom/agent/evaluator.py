"""The sufficiency predicate: is another retrieval pass worth paying for?

This is the whole of the agent's "agency" (FR-13). It reads **scores only** --
never chunk text, never a snippet, never a file path (ADR-008). That constraint
is what keeps the loop's cost independent of corpus size, and it is why the
predicate behaves identically whether the LLM is on or off (Appflow.md Flow 4,
step 3).

Trigger (TechSpecifications.md section 5.3): refine when the top-1 score is below
``agent_sufficiency_top1``, **or** when fewer than ``agent_sufficiency_min_results``
results score above ``agent_sufficiency_floor``. Both thresholds are PLACEHOLDER
(OQ-10) and both are read from :class:`~axiom.config.Settings`, never inlined.

The ladder has three rungs, not two -- Rules.md section 3's "Sufficiency check"
row, TechSpecifications.md sections 3.3.4, 3.7 and 4.8, and Appflow.md Flow 6 all
spell it the same way: *rerank-score predicate -> RRF-score predicate -> declare
sufficient (never loop blindly)*. The third rung is not decoration. The
PLACEHOLDER thresholds are on the **rerank** scale, a calibrated [0, 1]
probability, while ``rrf_score`` is bounded above by
:func:`max_achievable_rrf_score` -- ``1/(rrf_k + 1)``, about 0.0164 at the locked
``rrf_k=60``. Applied to RRF, a 0.35 threshold is not strict, it is
*unsatisfiable*: no ranking whatsoever can clear it, so the second rung can only
ever answer "insufficient" and every query without a live reranker pays for the
full pass budget to learn nothing. That is the blind loop the third rung exists
to prevent, and :func:`is_sufficient` takes it when the arithmetic says the
second rung cannot discriminate.
"""

from __future__ import annotations

from collections.abc import Sequence

# Re-exported so the loop has one import site for "decide, then rewrite". The
# refinement itself lives in the planner, which is where Appflow.md and Rules.md
# name it (``agent.planner:refine``) and where the identifier and expansion
# machinery it needs already lives.
from axiom.agent.planner import plans_differ, refine_plan
from axiom.config import Settings
from axiom.core.logging import get_logger, log_degradation
from axiom.schema.plan import QueryPlan
from axiom.schema.retrieval import FusedResult

_LOG = get_logger("agent.evaluator")

__all__ = [
    "assess_sufficiency",
    "is_sufficient",
    "max_achievable_rrf_score",
    "next_plan",
    "plans_differ",
    "refine_plan",
    "sufficiency_detail",
]


def max_achievable_rrf_score(settings: Settings) -> float:
    """The largest ``rrf_score`` any chunk can hold, given the fusion constants.

    A chunk's RRF score is ``sum(w_i / (rrf_k + rank_i))`` over the signals that
    ranked it. Every rank is at least 1 and ``QueryPlan`` guarantees the weights
    sum to 1.0, so the maximum -- reached only by a chunk ranked first by every
    signal -- is ``1 / (rrf_k + 1)``.

    It exists so the sufficiency ladder can tell "this ranking is weak" apart
    from "this threshold cannot be met by any ranking", which is the difference
    between refining usefully and looping blindly.
    """
    return 1.0 / (settings.rrf_k + 1)


def _rrf_predicate_is_satisfiable(settings: Settings) -> bool:
    """Whether the configured thresholds can be cleared on the ``rrf_score`` scale.

    Checks the top-1 threshold only: the floor gates a *count*, so a list can
    still be judged on it, whereas an unreachable top-1 threshold makes the whole
    conjunction constant-false regardless of the results.
    """
    return settings.agent_sufficiency_top1 <= max_achievable_rrf_score(settings)


def _scores(results: Sequence[FusedResult], used_rerank: bool) -> list[float]:
    """Project the score field the active ladder rung is judging on.

    When reranking ran, a ``rerank_score`` of ``None`` means "not a rerank
    candidate" -- the chunk fell outside the top-N window (Schema.md section 11).
    For an *ordering* decision that distinction matters and ``rerank_score or
    0.0`` would be a bug; for this predicate it does not, because a chunk outside
    the top-N window is by construction below every chunk inside it, and the
    predicate only ever counts how many scores clear a floor.
    """
    if used_rerank:
        return [0.0 if r.rerank_score is None else r.rerank_score for r in results]
    return [r.rrf_score for r in results]


def is_sufficient(
    results: Sequence[FusedResult],
    settings: Settings,
    used_rerank: bool,
) -> bool:
    """Whether ``results`` are good enough to stop refining (FR-13).

    Args:
        results: The current pass's fused candidates, in any order -- the
            predicate takes a max and a count, so it does not depend on the
            caller having sorted them (Rules.md Rule 2, stages are pure).
        settings: Supplies the two thresholds and the minimum count.
        used_rerank: Whether the cross-encoder actually scored this list. ``False``
            selects the ladder's second rung: the *same* two thresholds applied
            to ``rrf_score`` (TechSpecifications.md section 5.3 -- "the thresholds
            are shared, only the score field changes").

    Returns:
        ``True`` to stop, ``False`` to refine. Never raises: an empty list is
        simply insufficient, and the loop's pass and wall-clock caps -- not this
        predicate -- are what guarantee termination (ADR-007).
    """
    if not used_rerank:
        log_degradation(
            _LOG,
            "agent.evaluator:assess_sufficiency",
            "no rerank scores available (reranker disabled, timed out, or passed through)",
            "same thresholds applied to rrf_score",
        )
        if results and not _rrf_predicate_is_satisfiable(settings):
            ceiling = max_achievable_rrf_score(settings)
            log_degradation(
                _LOG,
                "agent.evaluator:assess_sufficiency",
                (
                    f"top1 threshold {settings.agent_sufficiency_top1:.4f} exceeds the "
                    f"maximum achievable rrf_score {ceiling:.6f} "
                    f"(1/(rrf_k+1), rrf_k={settings.rrf_k}), so the RRF-score "
                    "predicate is constant-false"
                ),
                "declared sufficient (Rules.md section 3: never loop blindly)",
            )
            _LOG.info(
                "sufficiency verdict",
                extra={
                    "axiom_extra": {
                        "stage": "agent",
                        "field": "rrf_score",
                        "rung": "declare_sufficient",
                        "rrf_ceiling": round(ceiling, 6),
                        "verdict": "sufficient",
                    }
                },
            )
            return True

    scores = _scores(results, used_rerank)
    if not scores:
        _LOG.info(
            "sufficiency: no candidates to judge",
            extra={"axiom_extra": {"stage": "agent", "verdict": "insufficient"}},
        )
        return False

    top1 = max(scores)
    above_floor = sum(1 for score in scores if score > settings.agent_sufficiency_floor)
    verdict = (
        top1 >= settings.agent_sufficiency_top1
        and above_floor >= settings.agent_sufficiency_min_results
    )
    _LOG.info(
        "sufficiency verdict",
        extra={
            "axiom_extra": {
                "stage": "agent",
                "field": "rerank_score" if used_rerank else "rrf_score",
                "top1": round(top1, 6),
                "above_floor": above_floor,
                "verdict": "sufficient" if verdict else "insufficient",
            }
        },
    )
    return verdict


def assess_sufficiency(
    results: Sequence[FusedResult],
    settings: Settings,
    used_rerank: bool,
) -> bool:
    """Alias for :func:`is_sufficient` under the name the flow docs use.

    Appflow.md and Rules.md both name this stage entrypoint
    ``axiom.agent.evaluator:assess_sufficiency``, while Rules.md AP-03's
    reference loop calls it ``evaluator.is_sufficient``. Both names resolve here
    so a grep for either one finds the real implementation.
    """
    return is_sufficient(results, settings, used_rerank)


def sufficiency_detail(
    results: Sequence[FusedResult],
    settings: Settings,
    used_rerank: bool,
) -> dict[str, object]:
    """The numbers behind the verdict, for the timings block and the UI panel.

    Recomputed rather than returned from :func:`is_sufficient` so the predicate
    keeps a plain ``bool`` signature for the loop, and so the demo UI can show
    "why did the agent run a second pass" without the loop having to thread a
    report object through (NFR-10 wants the decision auditable, not inferred).
    """
    scores = _scores(results, used_rerank)
    top1 = max(scores) if scores else 0.0
    above_floor = sum(1 for score in scores if score > settings.agent_sufficiency_floor)
    unsatisfiable = not used_rerank and not _rrf_predicate_is_satisfiable(settings)
    if unsatisfiable:
        rung = "declare_sufficient"
    elif used_rerank:
        rung = "rerank_score_predicate"
    else:
        rung = "rrf_score_predicate"
    return {
        "score_field": "rerank_score" if used_rerank else "rrf_score",
        "rung": rung,
        "candidates": len(results),
        "top1": top1,
        "above_floor": above_floor,
        "top1_threshold": settings.agent_sufficiency_top1,
        "floor": settings.agent_sufficiency_floor,
        "min_results": settings.agent_sufficiency_min_results,
        # Must agree with ``is_sufficient`` exactly, third rung included -- this
        # is what the UI and the timings block report as the reason a pass ran.
        "sufficient": bool(
            scores
            and (
                unsatisfiable
                or (
                    top1 >= settings.agent_sufficiency_top1
                    and above_floor >= settings.agent_sufficiency_min_results
                )
            )
        ),
    }


def next_plan(
    plan: QueryPlan,
    results: Sequence[FusedResult],
    settings: Settings,
) -> QueryPlan | None:
    """Refine, and return ``None`` when the refinement would repeat the last pass.

    A convenience over :func:`refine_plan` + :func:`plans_differ` for the loop's
    ``stop_reason="no_new_query"`` branch (Appflow.md Flow 3, TestPlan.md
    TC-089): a plan that issues the same retrievals is not worth a pass.
    """
    candidate = refine_plan(plan, results, settings)
    return candidate if plans_differ(plan, candidate) else None
