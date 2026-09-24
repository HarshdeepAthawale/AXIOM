"""Stage 3: weighted Reciprocal Rank Fusion over the three signals (FR-11).

The formula, verbatim from TechSpecifications.md section 5.1::

    score(d) = sum_i  w_i / (k + rank_i(d))          k = Settings.rrf_k (60)

Two properties of that formula are the whole reason this module exists, and both
are easy to destroy by accident:

**It reads ranks, never scores.** A cosine of 0.83, a BM25 score of 11.4 and a
graph score of 0.71 are three incomparable quantities; normalising them onto a
shared scale means inventing a calibration nobody measured (ADR-002). Rank is the
only quantity all three signals agree on the meaning of. So the raw
:attr:`~axiom.schema.ScoredChunk.score` field is never read here -- rescaling any
input list by any strictly-increasing map leaves this module's output
byte-identical (TestPlan.md TC-051).

**A signal that returns nothing must not be counted as a signal that returned
zeros.** Zero-filling would let an absent structural index drag every candidate's
score down by the same factor and, worse, would make a two-signal query score
strictly lower than the same query with one signal genuinely disabled. Instead
the empty signal's weight is *dropped* and the survivors are renormalised over
their new sum (section 5.1.2, TC-055), and the :class:`~axiom.schema.QueryPlan`
that gets logged carries the renormalised vector, not the nominal one -- the
number in the log has to be the number the arithmetic used.

Nothing here imports an optional dependency, or any dependency at all beyond the
standard library and the schema: fusion is pure arithmetic over integers, which
is exactly why it is the most testable stage in the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from axiom.config import Settings
from axiom.core.logging import get_logger, log_degradation
from axiom.schema import FusedResult, QueryPlan, ScoredChunk, SignalKind

_LOG = get_logger("retrieval.fusion")

_COMPONENT = "retrieval.fusion:reciprocal_rank_fusion"

#: Canonical signal order. Drives tie-breaking for ``dominant_signal``
#: (Schema.md section 11: "Ties resolve in the order DENSE, SPARSE, STRUCTURAL")
#: and the iteration order of every loop here, so that two runs over the same
#: inputs visit candidates in the same sequence (NFR-08).
SIGNAL_ORDER: tuple[SignalKind, ...] = (
    SignalKind.DENSE,
    SignalKind.SPARSE,
    SignalKind.STRUCTURAL,
)

SignalResults = Mapping[SignalKind, Sequence[ScoredChunk]]


def rank_key(chunk_id: str, score: float) -> tuple[float, str]:
    """The one sort key every ranker in Axiom uses: ``(-score, chunk_id)``.

    Rules.md Rule 4 asks for this to live in ``core/ranking.py`` and be imported
    everywhere; that module does not exist in the frozen foundation, so the key
    is defined here -- the stage that produces the *final* ordering -- and the
    per-signal retrievers each inline the identical tuple. Ascending
    ``chunk_id`` as the secondary key makes a float tie resolve the same way in
    every process, which is what NFR-08's byte-identical-output claim rests on.
    """
    return (-score, chunk_id)


@dataclass(frozen=True)
class FusionOutcome:
    """The fused list plus the evidence needed to explain and audit it.

    Returned instead of a bare ``list[FusedResult]`` because three of these
    fields are load-bearing downstream and unrecoverable afterwards: ``plan``
    carries the renormalised weights that must be logged (section 5.1.2),
    ``dropped_signals`` is what the ``timings`` block reports, and
    ``candidate_count`` distinguishes "fusion truncated 300 candidates to 25"
    from "only 25 candidates existed".
    """

    results: list[FusedResult]
    plan: QueryPlan
    effective_weights: dict[SignalKind, float]
    dropped_signals: tuple[SignalKind, ...] = ()
    candidate_count: int = 0
    degradations: tuple[str, ...] = ()
    per_signal_counts: dict[SignalKind, int] = field(default_factory=dict)

    @property
    def top1(self) -> float:
        """Best ``rrf_score``, or ``0.0`` when nothing fused."""
        return self.results[0].rrf_score if self.results else 0.0


def renormalise_weights(
    weights: Mapping[SignalKind, float],
    present: Iterable[SignalKind],
) -> dict[SignalKind, float]:
    """Drop absent signals' weights and rescale the survivors to sum to 1.0.

    Section 5.1.2's worked example: ``SEMANTIC`` weights ``{.6, .3, .1}`` with
    structural empty become ``{dense .6/.9, sparse .3/.9}``. A candidate's score
    therefore does not shrink merely because a signal was unavailable, which is
    the failure mode zero-filling produces.

    Args:
        weights: The plan's nominal per-signal weights.
        present: Signals that actually returned at least one candidate.

    Returns:
        A weight map over exactly the present signals summing to 1.0, or ``{}``
        when no present signal carries positive nominal weight (the caller then
        has nothing to fuse).
    """
    kept = {
        signal: weights[signal]
        for signal in SIGNAL_ORDER
        if signal in present and weights.get(signal, 0.0) > 0.0
    }
    total = sum(kept.values())
    if total <= 0.0:
        return {}
    return {signal: weight / total for signal, weight in kept.items()}


def plan_with_weights(plan: QueryPlan, weights: Mapping[SignalKind, float]) -> QueryPlan:
    """Return ``plan`` carrying ``weights``, rebuilt rather than mutated.

    ``QueryPlan`` is frozen (Rules.md AP-08: a stage never mutates its input), so
    the renormalised vector arrives as a new object. It is constructed field by
    field rather than through ``model_copy`` so that the "weights sum to 1.0"
    validator actually runs on the value we are about to log.
    """
    if not weights or dict(plan.strategy_weights) == dict(weights):
        return plan
    return QueryPlan(
        original_query=plan.original_query,
        query_type=plan.query_type,
        sub_queries=list(plan.sub_queries),
        extracted_identifiers=list(plan.extracted_identifiers),
        expansion_terms=list(plan.expansion_terms),
        strategy_weights=dict(weights),
    )


def _best_ranks(
    candidates: Sequence[ScoredChunk],
    signal: SignalKind,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Collapse one signal's ranked list to ``chunk_id -> best (lowest) rank``.

    A well-formed list has contiguous unique ranks, but fusion is downstream of
    three independent retrievers and one duplicated ``chunk_id`` must not be able
    to double-count a candidate into the top slot. Keeping the *lowest* rank is
    the conservative reading -- it credits the signal with its own best opinion
    -- and the duplicate is reported rather than silently absorbed (Rule 3: a
    degrade is loud).
    """
    ranks: dict[str, int] = {}
    notes: list[str] = []
    duplicates = 0
    for scored in candidates:
        if scored.signal is not signal:
            notes.append(f"{signal.value}: dropped a candidate tagged {scored.signal.value}")
            continue
        previous = ranks.get(scored.chunk_id)
        if previous is None:
            ranks[scored.chunk_id] = scored.rank
        else:
            duplicates += 1
            ranks[scored.chunk_id] = min(previous, scored.rank)
    if duplicates:
        notes.append(f"{signal.value}: {duplicates} duplicate chunk_id(s), kept best rank")
    return ranks, tuple(notes)


def reciprocal_rank_fusion(
    signal_results: SignalResults,
    plan: QueryPlan,
    settings: Settings,
    *,
    top_n: int | None = None,
) -> FusionOutcome:
    """Fuse the per-signal ranked lists into one ordered candidate set (FR-11).

    Args:
        signal_results: ``signal -> ranked list``. A signal may be absent from
            the mapping entirely or map to an empty list; both mean the same
            thing -- that signal contributed nothing and its weight is dropped.
        plan: The active plan. Supplies the nominal ``strategy_weights``; a
            signal the plan gives no positive weight (the ``eval`` profile's
            structural leg, for instance) is dropped even if it returned rows.
        settings: Supplies ``rrf_k`` and ``fusion_top_n``. No constant in this
            module is a literal at a callsite (Rules.md AP-07).
        top_n: Override for ``Settings.fusion_top_n``, for an API request
            carrying its own width.

    Returns:
        A :class:`FusionOutcome`. ``results`` is sorted by descending
        ``rrf_score`` with ties broken by ascending ``chunk_id``, contains no
        duplicate ``chunk_id``, and is truncated to ``top_n``. All signals empty
        yields an empty list and a plan with its nominal weights intact
        (TC-056) -- there is nothing to renormalise over.
    """
    k = settings.rrf_k
    width = settings.fusion_top_n if top_n is None else top_n
    notes: list[str] = []

    ranks_by_signal: dict[SignalKind, dict[str, int]] = {}
    counts: dict[SignalKind, int] = {}
    for signal in SIGNAL_ORDER:
        candidates = signal_results.get(signal) or ()
        ranks, signal_notes = _best_ranks(candidates, signal)
        notes.extend(signal_notes)
        counts[signal] = len(ranks)
        if ranks:
            ranks_by_signal[signal] = ranks

    nominal = dict(plan.strategy_weights)
    weights = renormalise_weights(nominal, ranks_by_signal)
    dropped = tuple(
        signal
        for signal in SIGNAL_ORDER
        if nominal.get(signal, 0.0) > 0.0 and signal not in weights
    )
    for signal in dropped:
        log_degradation(
            _LOG,
            _COMPONENT,
            f"{signal.value} returned no candidates (nominal weight {nominal[signal]:.4f})",
            f"weight dropped, remaining renormalised over {[s.value for s in weights]}",
        )
        notes.append(f"{signal.value}_signal_absent")
    for signal, ranks in ranks_by_signal.items():
        if signal not in weights:
            # The signal produced rows but the plan gives it no weight at all --
            # the eval profile's structural leg is the real instance. Fusing it
            # at an invented weight would be a magic number (AP-07).
            notes.append(f"{signal.value}: {len(ranks)} candidates ignored, plan weight is 0")

    if not weights:
        _LOG.info(
            "fusion produced no candidates",
            extra={"axiom_extra": {"stage": "fuse", "candidates": 0}},
        )
        return FusionOutcome(
            results=[],
            plan=plan,
            effective_weights={},
            dropped_signals=dropped,
            candidate_count=0,
            degradations=tuple(notes),
            per_signal_counts=counts,
        )

    # Union of candidates, visited in signal order then in each signal's own rank
    # order, so the pre-sort sequence is fixed and the sort is stable-in-effect.
    seen: dict[str, None] = {}
    for signal in SIGNAL_ORDER:
        if signal not in weights:
            continue
        for chunk_id, _ in sorted(ranks_by_signal[signal].items(), key=lambda kv: (kv[1], kv[0])):
            seen.setdefault(chunk_id, None)

    fused: list[tuple[tuple[float, str], FusedResult]] = []
    for chunk_id in seen:
        contributions: dict[SignalKind, int] = {}
        score = 0.0
        best_term = -1.0
        dominant = SignalKind.DENSE
        for signal in SIGNAL_ORDER:
            if signal not in weights:
                continue
            rank = ranks_by_signal[signal].get(chunk_id)
            if rank is None:
                continue
            term = weights[signal] / (k + rank)
            score += term
            contributions[signal] = rank
            # Strict ``>`` walking SIGNAL_ORDER resolves an exact tie toward
            # DENSE, then SPARSE, then STRUCTURAL, as Schema.md section 11 requires.
            if term > best_term:
                best_term = term
                dominant = signal
        if not contributions:
            continue
        result = FusedResult(
            chunk_id=chunk_id,
            rrf_score=score,
            contributions=contributions,
            dominant_signal=dominant,
        )
        fused.append((rank_key(chunk_id, score), result))

    fused.sort(key=lambda pair: pair[0])
    results = [result for _, result in fused[: max(width, 0)]]

    _LOG.info(
        "fused %d candidates into %d results",
        len(fused),
        len(results),
        extra={
            "axiom_extra": {
                "stage": "fuse",
                "rrf_k": k,
                "candidates": len(fused),
                "returned": len(results),
                "weights": {s.value: round(w, 6) for s, w in weights.items()},
                "dropped": [s.value for s in dropped],
            }
        },
    )
    return FusionOutcome(
        results=results,
        plan=plan_with_weights(plan, weights),
        effective_weights=weights,
        dropped_signals=dropped,
        candidate_count=len(fused),
        degradations=tuple(notes),
        per_signal_counts=counts,
    )


#: Name used by Appflow.md Flow 2 step 4 for this stage.
fuse = reciprocal_rank_fusion


def signal_width(signal: SignalKind, settings: Settings) -> int:
    """Configured candidate width for one signal (TechSpecifications.md section 5.2)."""
    if signal is SignalKind.DENSE:
        return settings.dense_top_k
    if signal is SignalKind.SPARSE:
        return settings.sparse_top_k
    return settings.structural_top_k


def merge_ranked_lists(
    lists: Sequence[Sequence[ScoredChunk]],
    signal: SignalKind,
    settings: Settings,
    *,
    width: int | None = None,
) -> list[ScoredChunk]:
    """RRF-merge several ranked lists *from the same signal* into one.

    This is the multi-sub-query path. ``QueryPlan.sub_queries`` decomposes a
    compound question ("which files call XYZ before ABC?") into independently
    retrievable parts; Schema.md section 10 says each sub-query "is retrieved
    independently and RRF-merged". Merging happens per signal and *before*
    cross-signal fusion, so a chunk that only the third sub-query found still
    enters :func:`reciprocal_rank_fusion` with a single honest rank for its
    signal rather than three competing ones.

    Sub-queries are weighted equally: no document assigns them relative
    importance, and inventing a decay would be a magic number (AP-07). The same
    ``rrf_k`` damps rank differences here as in cross-signal fusion.

    Args:
        lists: One ranked list per sub-query, all carrying ``signal``.
        signal: The signal these lists came from; stamped on the output.
        settings: Supplies ``rrf_k`` and the signal's candidate width.
        width: Override for the signal's configured width.

    Returns:
        One ranked list with contiguous 1-indexed ranks. A single input list is
        returned untouched, so the common single-query path never pays a
        re-scoring that would perturb the retriever's own ordering.

    Note:
        The merged ``score`` is an RRF score, not the signal's native score.
        Nothing downstream reads it -- fusion is rank-space (TC-051) -- but the
        field's docstring promises "signal-native", so this is the one place in
        the pipeline where that promise is relaxed, deliberately and here only.
    """
    populated = [entries for entries in lists if entries]
    if not populated:
        return []
    if len(populated) == 1:
        return list(populated[0])

    k = settings.rrf_k
    cap = signal_width(signal, settings) if width is None else width
    scores: dict[str, float] = {}
    for entries in populated:
        for scored in entries:
            if scored.signal is not signal:
                continue
            scores[scored.chunk_id] = scores.get(scored.chunk_id, 0.0) + 1.0 / (k + scored.rank)

    ordered = sorted(scores.items(), key=lambda kv: rank_key(kv[0], kv[1]))
    merged = [
        ScoredChunk(chunk_id=chunk_id, score=score, rank=position, signal=signal)
        for position, (chunk_id, score) in enumerate(ordered[: max(cap, 0)], start=1)
    ]
    _LOG.info(
        "merged %d sub-query lists for %s into %d candidates",
        len(populated),
        signal.value,
        len(merged),
        extra={
            "axiom_extra": {
                "stage": "fuse",
                "event": "subquery_merge",
                "signal": signal.value,
                "lists": len(populated),
                "candidates": len(merged),
            }
        },
    )
    return merged


def merge_signal_results(
    per_query: Sequence[SignalResults],
    settings: Settings,
    *,
    widths: Mapping[SignalKind, int] | None = None,
) -> dict[SignalKind, list[ScoredChunk]]:
    """Apply :func:`merge_ranked_lists` to every signal of a multi-query fan-out.

    The caller runs the three retrievers once per ``plan.effective_queries``
    entry and hands the per-query mappings here; what comes back looks exactly
    like a single query's fan-out, so :func:`reciprocal_rank_fusion` needs no
    special case for decomposed queries.

    The same function merges *across versions* for an ``--all-versions`` query:
    structurally it is the identical problem -- several ranked lists from one
    signal that have to become one.
    """
    merged: dict[SignalKind, list[ScoredChunk]] = {}
    for signal in SIGNAL_ORDER:
        lists = [mapping.get(signal) or () for mapping in per_query]
        width = None if widths is None else widths.get(signal)
        entries = merge_ranked_lists(lists, signal, settings, width=width)
        if entries:
            merged[signal] = entries
    return merged


__all__ = [
    "SIGNAL_ORDER",
    "FusionOutcome",
    "SignalResults",
    "fuse",
    "merge_ranked_lists",
    "merge_signal_results",
    "plan_with_weights",
    "rank_key",
    "reciprocal_rank_fusion",
    "renormalise_weights",
    "signal_width",
]
