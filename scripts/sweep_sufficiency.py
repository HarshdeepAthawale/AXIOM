"""Resolve `OQ-10`: are the agent sufficiency thresholds 0.35 / 0.20 right?

Run on the **train** split only. `NG-29` fences the test split off from every
tuning activity, and that fence is absolute.

The thresholds gate the agent's one decision (`FR-13`): refine when the top-1
rerank score is below ``agent_sufficiency_top1``, **or** when fewer than
``agent_sufficiency_min_results`` results clear ``agent_sufficiency_floor``.
Both are on the calibrated ``[0, 1]`` rerank scale that
``cross_encoder._sigmoid`` produces.

Two questions, in the order that matters:

1. **Can the thresholds be met at all?** A threshold no ranking ever clears is
   not strict, it is unsatisfiable: every query then pays the full pass budget
   to learn nothing. This is the same failure the evaluator's third ladder rung
   exists to catch on the RRF scale, and it is worth checking on the rerank
   scale too rather than assuming a calibrated score is well-spread.
2. **Does the top-1 score discriminate?** The predicate is only worth having if
   a low top-1 score actually means a bad result list. If top-1 score and
   NDCG@10 are uncorrelated, no threshold works and the honest answer is to say
   so rather than to pick the value that looks tidiest.

Retrieve once, rerank once, then sweep every candidate threshold in memory --
the ranked lists do not depend on the threshold, only the refine/stop decision
does.

    python scripts/sweep_sufficiency.py --limit 1500
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_backends import dense_only
from scripts.sweep_sparse_weight import fuse, load_train_task

from axiom.config import get_settings
from axiom.eval import metrics
from axiom.eval.mteb_adapter import LexicalBackend, build_search_model

#: Candidate top-1 thresholds. Spans four orders of magnitude on purpose: a
#: cross-encoder trained on web passages (`ms-marco-MiniLM`) scores code far
#: from its training distribution, so the interesting region may be nowhere
#: near the placeholder 0.35.
TOP1_CANDIDATES = (0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50)

#: Candidate floors, paired with the ``min_results`` count already in config.
FLOOR_CANDIDATES = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="AppsRetrieval")
    parser.add_argument("--split", default="train", help="Never 'test' (NG-29).")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--profile", default="demo", help="Which reranker chain to measure.")
    parser.add_argument("--out", type=Path, default=Path("data/sweeps/oq10_sufficiency.json"))
    args = parser.parse_args(argv)

    if args.split == "test":
        print("refusing: NG-29 forbids tuning on the test split", file=sys.stderr)
        return 2

    settings = get_settings(args.profile)
    sparse_weight = float(settings.eval_sparse_weight)
    top_n = int(settings.fusion_top_n)
    min_results = int(settings.agent_sufficiency_min_results)

    task = load_train_task(args.task, args.limit)
    print(f"{args.task}/{args.split}: {len(task.corpus)} docs, {len(task.queries)} queries")

    from axiom.rerank.cross_encoder import load_cross_encoder

    scorer = load_cross_encoder(settings)
    if scorer is None:
        print(
            "refusing: no cross-encoder loaded, so there is no rerank scale to tune on.\n"
            "Export the INT8 artifact first (Setup.md section 6.1); a sweep against the\n"
            "passthrough rung would measure the fallback, not the thresholds.",
            file=sys.stderr,
        )
        return 3
    print(f"reranker: {scorer.name}")

    dense_model = build_search_model(settings, backend=dense_only(settings))
    sparse_model = build_search_model(settings, backend=LexicalBackend())

    print("indexing both legs once ...")
    for model in (dense_model, sparse_model):
        model.index(task.corpus, hf_split=task.split)

    print("retrieving once per query ...")
    dense_run = dense_model.search(task.queries, hf_split=task.split, top_k=args.top_k)
    sparse_run = sparse_model.search(task.queries, hf_split=task.split, top_k=args.top_k)

    def ranked(run_for_query):
        return sorted(run_for_query.items(), key=lambda kv: (-kv[1], kv[0]))

    print(f"reranking top-{top_n} per query ({len(task.queries)} queries) ...")
    from axiom.rerank.cross_encoder import _sigmoid

    observations = []
    for done, (qid, qtext) in enumerate(sorted(task.queries.items()), start=1):
        fused = fuse(
            ranked(dense_run.get(qid, {})),
            ranked(sparse_run.get(qid, {})),
            sparse_weight,
            args.top_k,
        )
        pool = list(fused)[:top_n]
        if not pool:
            continue
        pairs = [(qtext, task.corpus[doc_id]["text"]) for doc_id in pool]
        logits = scorer.score_pairs(pairs)
        rerank_scores = [_sigmoid(float(v)) for v in logits]

        reranked = dict(
            sorted(
                zip(pool, rerank_scores, strict=False),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
        relevance = task.qrels.get(qid, {})
        observations.append(
            {
                "qid": qid,
                "top1": max(rerank_scores),
                "scores": rerank_scores,
                "ndcg_fused": metrics.per_query_ndcg_at_k(relevance, fused, 10),
                "ndcg_reranked": metrics.per_query_ndcg_at_k(relevance, reranked, 10),
            }
        )
        if done % 250 == 0:
            print(f"  {done}/{len(task.queries)}")

    if not observations:
        print("no queries produced candidates", file=sys.stderr)
        return 4

    top1s = sorted(o["top1"] for o in observations)

    def pct(p: float) -> float:
        idx = min(len(top1s) - 1, max(0, round(p / 100.0 * (len(top1s) - 1))))
        return top1s[idx]

    distribution = {
        "min": top1s[0],
        "p05": pct(5),
        "p25": pct(25),
        "median": statistics.median(top1s),
        "p75": pct(75),
        "p95": pct(95),
        "max": top1s[-1],
        "mean": statistics.fmean(top1s),
    }
    print("\ntop-1 rerank score distribution (calibrated [0,1]):")
    for key, value in distribution.items():
        print(f"  {key:<7} {value:.6f}")

    # Question 2: does top-1 discriminate a good result list from a bad one?
    # Spearman on ranks, computed here rather than pulled in from scipy.
    def spearman(xs, ys):
        def rank(values):
            order = sorted(range(len(values)), key=lambda i: values[i])
            out = [0.0] * len(values)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                    j += 1
                shared = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    out[order[k]] = shared
                i = j + 1
            return out

        rx, ry = rank(xs), rank(ys)
        mx, my = statistics.fmean(rx), statistics.fmean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=False))
        den = (
            sum((a - mx) ** 2 for a in rx) ** 0.5 * sum((b - my) ** 2 for b in ry) ** 0.5
        )
        return num / den if den else 0.0

    rho = spearman(
        [o["top1"] for o in observations], [o["ndcg_reranked"] for o in observations]
    )
    print(f"\nspearman(top-1 rerank score, NDCG@10) = {rho:.4f}")

    rows = []
    for top1_threshold in TOP1_CANDIDATES:
        refine = [o for o in observations if o["top1"] < top1_threshold]
        stop = [o for o in observations if o["top1"] >= top1_threshold]
        rows.append(
            {
                "agent_sufficiency_top1": top1_threshold,
                "refine_rate": round(len(refine) / len(observations), 4),
                "mean_ndcg_refined": round(
                    statistics.fmean([o["ndcg_reranked"] for o in refine]) * 100, 4
                )
                if refine
                else None,
                "mean_ndcg_stopped": round(
                    statistics.fmean([o["ndcg_reranked"] for o in stop]) * 100, 4
                )
                if stop
                else None,
            }
        )
        gap = (
            None
            if rows[-1]["mean_ndcg_stopped"] is None or rows[-1]["mean_ndcg_refined"] is None
            else round(rows[-1]["mean_ndcg_stopped"] - rows[-1]["mean_ndcg_refined"], 4)
        )
        rows[-1]["separation"] = gap
        print(
            f"  top1={top1_threshold:<7} refine_rate={rows[-1]['refine_rate']:<7} "
            f"ndcg(stop)={rows[-1]['mean_ndcg_stopped']} "
            f"ndcg(refine)={rows[-1]['mean_ndcg_refined']} separation={gap}"
        )

    floor_rows = []
    for floor in FLOOR_CANDIDATES:
        counts = [sum(1 for s in o["scores"] if s >= floor) for o in observations]
        trip = sum(1 for c in counts if c < min_results) / len(counts)
        floor_rows.append(
            {
                "agent_sufficiency_floor": floor,
                "min_results": min_results,
                "refine_rate": round(trip, 4),
                "mean_clearing": round(statistics.fmean(counts), 3),
            }
        )
        print(
            f"  floor={floor:<8} refine_rate={trip:<7.4f} "
            f"mean_results_clearing={floor_rows[-1]['mean_clearing']}"
        )

    payload = {
        "question": "OQ-10",
        "task": args.task,
        "split": args.split,
        "profile": args.profile,
        "queries": len(observations),
        "corpus": len(task.corpus),
        "reranker": scorer.name,
        "fusion_top_n": top_n,
        "sparse_weight": sparse_weight,
        "agent_sufficiency_min_results": min_results,
        "top1_distribution": {k: round(v, 8) for k, v in distribution.items()},
        "spearman_top1_vs_ndcg": round(rho, 6),
        "top1_candidates": rows,
        "floor_candidates": floor_rows,
        "placeholder_values": {"agent_sufficiency_top1": 0.35, "agent_sufficiency_floor": 0.20},
        "note": "Measured on the TRAIN split only; the test split was not consulted (NG-29).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
