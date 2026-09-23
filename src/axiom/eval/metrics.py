"""Retrieval metrics over BEIR-shaped ``(qrels, run)`` pairs.

Why Axiom carries its own arithmetic instead of importing ``pytrec_eval`` or
letting MTEB do it: the number this module produces is the number the screening
gate reads (PRD.md section 2), and ``mteb``/``datasets`` are optional extras that
may not be installed on the evaluator's box (NFR-07). A metric that only exists
when a 400 MB dependency tree resolves is a metric that can vanish the night
before submission. So the definitions live here, in the standard library, and
:mod:`axiom.eval.mteb_adapter` uses MTEB only to *source the data*, never to
score it.

Conventions, chosen to match ``trec_eval`` / ``pytrec_eval`` / BEIR so our
numbers are comparable to the published CoIR leaderboard rather than merely
internally consistent:

* **Gain is linear.** ``DCG@k = sum_i rel_i / log2(i + 1)``. ``trec_eval``'s
  ``ndcg_cut`` uses the raw relevance grade as the gain, not ``2**rel - 1``.
  AppsRetrieval qrels are binary, so the two agree there, but the choice is
  explicit via ``gain=`` because it silently changes any graded-relevance number
  by several points.
* **The ideal ranking comes from the qrels, not from the run.** ``IDCG@k`` sorts
  every judged grade for the query descending and takes the first ``k``. A
  relevant document the run never retrieved still raises the denominator -- that
  is the whole point of a recall-sensitive metric.
* **Unretrieved and unjudged both mean grade 0.** A document in the run with no
  qrels entry contributes nothing; a document in the qrels with grade ``<= 0`` is
  a judged non-relevant and also contributes nothing.
* **Ties break by ascending document id** (NFR-08). Two documents at the same
  score must not swap places between runs, or NDCG@10 wobbles for reasons that
  have nothing to do with retrieval quality.

Scores are returned as fractions in ``[0, 1]``. The leaderboard in PRD.md
section 2 quotes percentages (``14.7``, not ``0.147``); the conversion happens
once, at the reporting boundary in ``scripts/run_eval.py``, and never here.

Worked example, computed by hand and asserted in the doctests below so a
regression in this file cannot pass silently::

    qrels = {"q1": {"d1": 3, "d2": 2, "d3": 0, "d4": 1},
             "q2": {"dA": 1}}
    run   = {"q1": {"d3": 0.9, "d1": 0.8, "d5": 0.7, "d2": 0.6},
             "q2": {}}

    q1 ranks:  1. d3 (grade 0)   2. d1 (grade 3)   3. d5 (unjudged -> 0)
    DCG@3     = 0/log2(2) + 3/log2(3) + 0/log2(4) = 3/1.5849625 = 1.89278926
    ideal@3   = grades [3, 2, 1]  (d3's 0 and d4's absence from the run are
                                   both irrelevant to the ideal ordering)
    IDCG@3    = 3/1 + 2/1.5849625 + 1/2                        = 4.76185951
    NDCG@3    = 1.89278926 / 4.76185951                        = 0.39748952
    MRR@3     = first relevant is d1 at rank 2 -> 1/2          = 0.5
    Recall@3  = 1 of the 3 positives ({d1, d2, d4}) retrieved  = 0.33333333
    AP@3      = (1/3) * P@2 = (1/3) * (1/2)                    = 0.16666667

    q2 has an empty run: every metric is 0.0, not skipped -- the query is judged,
    we simply returned nothing for it.

    macro NDCG@3 = (0.39748952 + 0.0) / 2                      = 0.19874476
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

from axiom.core.logging import get_logger

__all__ = [
    "GainKind",
    "Qrels",
    "Run",
    "dcg",
    "evaluate_run",
    "map_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "per_query_ndcg_at_k",
    "rank_documents",
    "recall_at_k",
    "run_coverage",
]

_LOGGER = get_logger("eval.metrics")

#: BEIR/MTEB qrels: query id -> document id -> graded relevance (0 means judged
#: non-relevant; an absent key means unjudged, which scores the same).
Qrels = Mapping[str, Mapping[str, float]]

#: BEIR/MTEB run: query id -> document id -> score, higher is better (Rule 4).
Run = Mapping[str, Mapping[str, float]]

#: Which gain function maps a relevance grade to a DCG contribution.
GainKind = Literal["linear", "exponential"]

#: Cut-offs reported by default, matching BEIR's ``[1, 3, 5, 10, 100]``.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10, 100)


def _gain(relevance: float, kind: GainKind) -> float:
    """Map a relevance grade to its DCG contribution before discounting.

    Negative grades clamp to zero: ``trec_eval`` treats a negative judgment as
    "judged, not relevant", never as a penalty that could drive DCG below zero
    and make the ratio meaningless.
    """
    graded = max(0.0, float(relevance))
    if kind == "exponential":
        return float(2.0**graded) - 1.0
    return graded


def _discount(rank: int) -> float:
    """Positional discount ``1 / log2(rank + 1)`` for a 1-indexed rank."""
    return 1.0 / math.log2(rank + 1)


def _as_score(value: float) -> float:
    """Coerce a run score to a sortable float, mapping NaN/inf to a floor.

    Rule 3: a NaN score somewhere in a 3,765-query run is bad *data*, not a bug
    in the metric. It sinks to the bottom of the ranking and the run continues,
    loudly.
    """
    score = float(value)
    if math.isnan(score):
        _LOGGER.warning("NaN score in run; ranked last", extra={"axiom_extra": {"stage": "eval"}})
        return -math.inf
    return score


def rank_documents(scored: Mapping[str, float], k: int | None = None) -> list[str]:
    """Order one query's retrieved documents best-first, deterministically.

    The sort key is ``(-score, doc_id)``: descending score (Rule 4, higher is
    better) with ascending document id as the tie-break, so an identical run
    always produces an identical ranking regardless of dict insertion order or
    ``PYTHONHASHSEED`` (NFR-08, Rules.md AP-05).

    Args:
        scored: document id -> score for a single query.
        k: truncate to this many documents; ``None`` keeps all of them.

    Returns:
        Document ids, best first.
    """
    ordered = sorted(scored.items(), key=lambda item: (-_as_score(item[1]), item[0]))
    if k is not None:
        ordered = ordered[: max(0, k)]
    return [doc_id for doc_id, _ in ordered]


def dcg(grades: Sequence[float], *, gain: GainKind = "linear") -> float:
    """Discounted cumulative gain of an already-ordered grade sequence.

    Exposed because it is the one piece of the NDCG computation worth testing in
    isolation: if this is wrong, both the numerator and the denominator are
    wrong in the same direction and the ratio can still look plausible.

    >>> round(dcg([0, 3, 0]), 8)
    1.89278926
    >>> round(dcg([3, 2, 1]), 8)
    4.76185951
    """
    return sum(_gain(grade, gain) * _discount(rank) for rank, grade in enumerate(grades, start=1))


def _positives(relevance: Mapping[str, float]) -> set[str]:
    """Document ids judged relevant (grade strictly above zero) for one query."""
    return {doc_id for doc_id, grade in relevance.items() if float(grade) > 0.0}


def _scored_query_ids(
    qrels: Qrels,
    run: Run,
    *,
    skip_queries_without_positives: bool,
) -> list[str]:
    """The query ids that enter the macro-average, in deterministic order.

    The denominator is the **qrels**, not the run: a query we failed to answer
    scores 0 and drags the average down, which is the honest accounting. A query
    present in the run but absent from the qrels is ignored -- we have no
    judgments for it, so it cannot contribute either way.
    """
    selected = [
        query_id
        for query_id in sorted(qrels)
        if not skip_queries_without_positives or _positives(qrels[query_id])
    ]
    extra = sorted(set(run) - set(qrels))
    if extra:
        _LOGGER.warning(
            "run contains %d queries absent from qrels; ignored for scoring",
            len(extra),
            extra={"axiom_extra": {"stage": "eval", "example_query_id": extra[0]}},
        )
    return selected


def _check_k(k: int) -> int:
    """Validate a cut-off, degrading rather than raising (Rule 3)."""
    if k < 1:
        _LOGGER.warning(
            "non-positive cut-off k=%d; treated as 0 (every metric will be 0.0)",
            k,
            extra={"axiom_extra": {"stage": "eval"}},
        )
        return 0
    return k


def per_query_ndcg_at_k(
    relevance: Mapping[str, float],
    scored: Mapping[str, float],
    k: int,
    *,
    gain: GainKind = "linear",
) -> float:
    """NDCG@k for a single query.

    Returns ``0.0`` when the query has no relevant document at all -- the ideal
    DCG is then zero and the ratio is undefined. Callers that want such queries
    excluded from the average entirely use ``skip_queries_without_positives``
    on :func:`ndcg_at_k` rather than reinterpreting this ``0.0``.

    >>> qrels_q = {"d1": 3, "d2": 2, "d3": 0, "d4": 1}
    >>> run_q = {"d3": 0.9, "d1": 0.8, "d5": 0.7, "d2": 0.6}
    >>> round(per_query_ndcg_at_k(qrels_q, run_q, 3), 8)
    0.39748952
    >>> round(per_query_ndcg_at_k(qrels_q, run_q, 3, gain="exponential"), 8)
    0.470202
    >>> per_query_ndcg_at_k(qrels_q, {}, 3)
    0.0
    >>> per_query_ndcg_at_k({}, run_q, 3)
    0.0
    """
    cut = _check_k(k)
    if cut == 0:
        return 0.0
    ideal_grades = sorted((float(g) for g in relevance.values()), reverse=True)[:cut]
    ideal = dcg(ideal_grades, gain=gain)
    if ideal <= 0.0:
        return 0.0
    retrieved = rank_documents(scored, cut)
    actual = dcg([float(relevance.get(doc_id, 0.0)) for doc_id in retrieved], gain=gain)
    return actual / ideal


def per_query_mrr_at_k(
    relevance: Mapping[str, float], scored: Mapping[str, float], k: int
) -> float:
    """Reciprocal rank of the first relevant document within the top ``k``.

    ``0.0`` when no relevant document appears in the cut-off window, which is the
    MS MARCO / ``trec_eval recip_rank`` convention.

    >>> per_query_mrr_at_k({"d1": 3, "d2": 2, "d4": 1}, {"d3": 0.9, "d1": 0.8}, 3)
    0.5
    >>> per_query_mrr_at_k({"d1": 3}, {"d9": 0.9}, 3)
    0.0
    """
    cut = _check_k(k)
    if cut == 0:
        return 0.0
    positives = _positives(relevance)
    if not positives:
        return 0.0
    for rank, doc_id in enumerate(rank_documents(scored, cut), start=1):
        if doc_id in positives:
            return 1.0 / rank
    return 0.0


def per_query_recall_at_k(
    relevance: Mapping[str, float], scored: Mapping[str, float], k: int
) -> float:
    """Fraction of a query's relevant documents that appear in the top ``k``.

    The denominator is every positive in the qrels, *not* ``min(k, positives)``:
    that is ``trec_eval``'s ``recall_k`` and it is why Recall@100 is the honest
    ceiling the reranker works under (PRD.md section 2). Capping the denominator
    would let a query with 200 positives score 1.0 at k=100.

    >>> round(per_query_recall_at_k({"d1": 3, "d2": 2, "d4": 1}, {"d3": 0.9, "d1": 0.8}, 3), 8)
    0.33333333
    >>> per_query_recall_at_k({}, {"d1": 0.9}, 3)
    0.0
    """
    cut = _check_k(k)
    if cut == 0:
        return 0.0
    positives = _positives(relevance)
    if not positives:
        return 0.0
    retrieved = set(rank_documents(scored, cut))
    return len(positives & retrieved) / len(positives)


def per_query_ap_at_k(relevance: Mapping[str, float], scored: Mapping[str, float], k: int) -> float:
    """Average precision of the top ``k``, normalised by all known positives.

    ``trec_eval``'s ``map`` with the run truncated at ``k``: the sum of
    precision-at-each-relevant-hit divided by the *total* number of positives, so
    a positive that never surfaces costs recall inside the metric.

    >>> round(per_query_ap_at_k({"d1": 3, "d2": 2, "d4": 1}, {"d3": 0.9, "d1": 0.8}, 3), 8)
    0.16666667
    """
    cut = _check_k(k)
    if cut == 0:
        return 0.0
    positives = _positives(relevance)
    if not positives:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(rank_documents(scored, cut), start=1):
        if doc_id in positives:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(positives)


#: A per-query scorer: (relevance, scored, k) -> score in [0, 1].
PerQueryScorer = Callable[[Mapping[str, float], Mapping[str, float], int], float]


def _macro(
    qrels: Qrels,
    run: Run,
    k: int,
    scorer: PerQueryScorer,
    *,
    skip_queries_without_positives: bool,
) -> float:
    """Average a per-query scorer over the qrels' query set.

    ``scorer`` is one of the ``per_query_*`` functions. An empty query set scores
    ``0.0`` rather than raising :class:`ZeroDivisionError` -- an eval over an
    empty qrels is a wiring mistake that must show up as a visible zero in the
    results JSON, not as a traceback three hours into a full-split run.
    """
    query_ids = _scored_query_ids(
        qrels, run, skip_queries_without_positives=skip_queries_without_positives
    )
    if not query_ids:
        _LOGGER.warning(
            "no scorable queries; metric is 0.0",
            extra={"axiom_extra": {"stage": "eval", "qrels_queries": len(qrels)}},
        )
        return 0.0
    total = sum(scorer(qrels[query_id], run.get(query_id, {}), k) for query_id in query_ids)
    return total / len(query_ids)


def ndcg_at_k(
    qrels: Qrels,
    run: Run,
    k: int = 10,
    *,
    gain: GainKind = "linear",
    skip_queries_without_positives: bool = False,
) -> float:
    """Macro-averaged NDCG@k -- the metric the screening gate reads.

    Args:
        qrels: query id -> document id -> graded relevance.
        run: query id -> document id -> score, higher is better.
        k: rank cut-off. ``10`` is the reported figure (PRD.md section 2).
        gain: ``"linear"`` (``trec_eval``/BEIR, the default) or ``"exponential"``
            (``2**rel - 1``). Identical on binary qrels such as AppsRetrieval.
        skip_queries_without_positives: exclude queries with no relevant document
            from the denominator. ``False`` keeps them and scores them ``0.0``.

    Returns:
        A fraction in ``[0, 1]``. Multiply by 100 to compare against the CoIR
        leaderboard.

    >>> qrels = {"q1": {"d1": 3, "d2": 2, "d3": 0, "d4": 1}, "q2": {"dA": 1}}
    >>> run = {"q1": {"d3": 0.9, "d1": 0.8, "d5": 0.7, "d2": 0.6}, "q2": {}}
    >>> round(ndcg_at_k(qrels, run, 3), 8)
    0.19874476
    >>> ndcg_at_k({}, {}, 10)
    0.0
    """

    def scorer(relevance: Mapping[str, float], scored: Mapping[str, float], cut: int) -> float:
        return per_query_ndcg_at_k(relevance, scored, cut, gain=gain)

    return _macro(
        qrels, run, k, scorer, skip_queries_without_positives=skip_queries_without_positives
    )


def mrr_at_k(
    qrels: Qrels,
    run: Run,
    k: int = 10,
    *,
    skip_queries_without_positives: bool = False,
) -> float:
    """Macro-averaged reciprocal rank of the first relevant hit in the top ``k``.

    Early relevance, PRD.md section 2's second P0 metric: NDCG rewards getting
    the whole top-10 right, MRR rewards getting rank 1 right, and a code-search
    user reads rank 1 first.

    >>> qrels = {"q1": {"d1": 3, "d2": 2, "d4": 1}, "q2": {"dA": 1}}
    >>> run = {"q1": {"d3": 0.9, "d1": 0.8}, "q2": {}}
    >>> mrr_at_k(qrels, run, 3)
    0.25
    """
    return _macro(
        qrels,
        run,
        k,
        per_query_mrr_at_k,
        skip_queries_without_positives=skip_queries_without_positives,
    )


def recall_at_k(
    qrels: Qrels,
    run: Run,
    k: int = 100,
    *,
    skip_queries_without_positives: bool = False,
) -> float:
    """Macro-averaged recall at ``k`` -- the first-stage ceiling (PRD.md section 2).

    Reported at ``k=100`` because that is the candidate width the three signals
    hand to fusion: a document the first stage never surfaces cannot be rescued
    by the reranker, so Recall@100 bounds the achievable NDCG@10 from above.

    >>> qrels = {"q1": {"d1": 3, "d2": 2, "d4": 1}}
    >>> run = {"q1": {"d3": 0.9, "d1": 0.8}}
    >>> round(recall_at_k(qrels, run, 3), 8)
    0.33333333
    """
    return _macro(
        qrels,
        run,
        k,
        per_query_recall_at_k,
        skip_queries_without_positives=skip_queries_without_positives,
    )


def map_at_k(
    qrels: Qrels,
    run: Run,
    k: int = 10,
    *,
    skip_queries_without_positives: bool = False,
) -> float:
    """Macro-averaged average precision at ``k``.

    Not a gating metric, but TestPlan.md section 6.4's experiment log carries a
    ``map`` column, and a row without it is an incomplete experiment record.

    >>> round(map_at_k({"q1": {"d1": 3, "d2": 2, "d4": 1}}, {"q1": {"d3": 0.9, "d1": 0.8}}, 3), 8)
    0.16666667
    """
    return _macro(
        qrels,
        run,
        k,
        per_query_ap_at_k,
        skip_queries_without_positives=skip_queries_without_positives,
    )


def run_coverage(qrels: Qrels, run: Run) -> dict[str, int]:
    """Counts that let a reader audit a metric rather than trust it.

    A high NDCG computed over eleven of 3,765 queries is not a high NDCG, and the
    only way to notice is to publish the denominators next to the number. These
    land in ``appsretrieval_results.json`` for exactly that reason.

    >>> coverage = run_coverage({"q1": {"d1": 1}, "q2": {}}, {"q1": {"d1": 0.9}, "q3": {"d2": 0.1}})
    >>> coverage["qrels_queries"], coverage["queries_with_positives"]
    (2, 1)
    >>> coverage["answered_queries"], coverage["empty_result_queries"]
    (1, 1)
    >>> coverage["run_only_queries"], coverage["retrieved_documents"]
    (1, 1)
    """
    with_positives = [q for q in qrels if _positives(qrels[q])]
    answered = [q for q in qrels if run.get(q)]
    retrieved = sum(len(run.get(q, {})) for q in qrels)
    return {
        "qrels_queries": len(qrels),
        "queries_with_positives": len(with_positives),
        "answered_queries": len(answered),
        "empty_result_queries": len(qrels) - len(answered),
        "run_only_queries": len(set(run) - set(qrels)),
        "retrieved_documents": retrieved,
    }


def evaluate_run(
    qrels: Qrels,
    run: Run,
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    gain: GainKind = "linear",
    skip_queries_without_positives: bool = False,
) -> dict[str, float]:
    """Compute every metric at every cut-off, in MTEB's flat ``<metric>_at_<k>`` naming.

    The key names are chosen to match what MTEB writes into a ``TaskResult``
    score block (``ndcg_at_10``, ``mrr_at_10``, ``recall_at_100``), so the JSON
    ``scripts/run_eval.py`` emits is loadable by MTEB tooling whether or not MTEB
    was installed when it ran.

    >>> scores = evaluate_run({"q1": {"d1": 1}}, {"q1": {"d1": 0.9}}, k_values=(1,))
    >>> scores["ndcg_at_1"], scores["mrr_at_1"], scores["recall_at_1"]
    (1.0, 1.0, 1.0)
    """
    cuts = sorted({_check_k(int(k)) for k in k_values} - {0})
    scores: dict[str, float] = {}
    for cut in cuts:
        scores[f"ndcg_at_{cut}"] = ndcg_at_k(
            qrels,
            run,
            cut,
            gain=gain,
            skip_queries_without_positives=skip_queries_without_positives,
        )
        scores[f"mrr_at_{cut}"] = mrr_at_k(
            qrels, run, cut, skip_queries_without_positives=skip_queries_without_positives
        )
        scores[f"recall_at_{cut}"] = recall_at_k(
            qrels, run, cut, skip_queries_without_positives=skip_queries_without_positives
        )
        scores[f"map_at_{cut}"] = map_at_k(
            qrels, run, cut, skip_queries_without_positives=skip_queries_without_positives
        )
    return scores
