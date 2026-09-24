"""Resolve `OQ-10`'s second half: does refining a weak query actually help?

Run on the **train** split only (`NG-29`).

``sweep_sufficiency.py`` measures whether the top-1 rerank score can *tell* a
weak result list from a strong one. It can, modestly. But a predicate that
identifies weak queries is only worth having if the thing it triggers --
a second retrieval pass on a rewritten plan (`FR-13`) -- makes those queries
better. A threshold tuned on separation alone optimises the detector and
ignores the treatment, which is how you end up paying two passes for every
query and calling it agency.

So this runs the actual refinement: build the plan, retrieve, rerank, ask the
planner to rewrite, retrieve the rewritten sub-queries, fuse the union exactly
as ``agent.loop`` does, rerank again, and compare NDCG@10 before and after --
bucketed by the pass-1 top-1 score, which is what the threshold reads.

The answer is a cost/benefit curve, not a single number: for each candidate
threshold, how many queries pay a second pass, and what NDCG those queries
actually gain.

    python scripts/sweep_refinement.py --limit 600
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
from scripts.sweep_sufficiency import TOP1_CANDIDATES

from axiom.config import get_settings
from axiom.core.hashing import blake2b_128
from axiom.eval import metrics
from axiom.eval.mteb_adapter import LexicalBackend, build_search_model
from axiom.schema.enums import SignalKind


def as_fused(doc_ids, rrf_scores, rerank_scores):
    """Minimal :class:`FusedResult` rows for the planner's refinement decision.

    ``refine_plan`` reads only the *shape* of the last pass -- how many
    candidates came back and whether any of them scored -- never chunk text
    (ADR-008). So these carry real scores and synthetic ids: ``chunk_id`` is a
    digest of the doc id purely to satisfy Rules.md Rule 1's format, and nothing
    downstream of here dereferences it.
    """
    from axiom.schema.retrieval import FusedResult

    return [
        FusedResult(
            chunk_id=blake2b_128(doc_id.encode("utf-8")),
            rrf_score=float(rrf),
            rerank_score=float(rer),
            contributions={SignalKind.DENSE: rank},
            dominant_signal=SignalKind.DENSE,
        )
        for rank, (doc_id, rrf, rer) in enumerate(
            zip(doc_ids, rrf_scores, rerank_scores, strict=False), start=1
        )
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="AppsRetrieval")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--profile", default="demo")
    parser.add_argument("--out", type=Path, default=Path("data/sweeps/oq10_refinement.json"))
    args = parser.parse_args(argv)

    settings = get_settings(args.profile)
    sparse_weight = float(settings.eval_sparse_weight)
    top_n = int(settings.fusion_top_n)

    task = load_train_task(args.task, args.limit)
    print(f"{args.task}/train: {len(task.corpus)} docs, {len(task.queries)} queries")

    from axiom.agent.planner import build_plan, plans_differ, refine_plan
    from axiom.rerank.cross_encoder import _sigmoid, load_cross_encoder

    scorer = load_cross_encoder(settings)
    if scorer is None:
        print("refusing: no cross-encoder loaded (Setup.md section 6.1)", file=sys.stderr)
        return 3
    print(f"reranker: {scorer.name}")

    dense_model = build_search_model(settings, backend=dense_only(settings))
    sparse_model = build_search_model(settings, backend=LexicalBackend())
    print("indexing both legs once ...")
    for model in (dense_model, sparse_model):
        model.index(task.corpus, hf_split=task.split)

    def ranked(run_for_query):
        return sorted(run_for_query.items(), key=lambda kv: (-kv[1], kv[0]))

    def retrieve(query_map):
        d = dense_model.search(query_map, hf_split=task.split, top_k=args.top_k)
        s = sparse_model.search(query_map, hf_split=task.split, top_k=args.top_k)
        return d, s

    print("pass 1: retrieving ...")
    dense1, sparse1 = retrieve(task.queries)

    def rerank(qtext, fused):
        pool = list(fused)[:top_n]
        if not pool:
            return {}, []
        pairs = [(qtext, task.corpus[d]["text"]) for d in pool]
        scores = [_sigmoid(float(v)) for v in scorer.score_pairs(pairs)]
        order = dict(sorted(zip(pool, scores, strict=False), key=lambda kv: (-kv[1], kv[0])))
        return order, scores

    print("pass 1: reranking ...")
    pass1 = {}
    for done, (qid, qtext) in enumerate(sorted(task.queries.items()), start=1):
        fused = fuse(
            ranked(dense1.get(qid, {})), ranked(sparse1.get(qid, {})), sparse_weight, args.top_k
        )
        order, scores = rerank(qtext, fused)
        if not order:
            continue
        pass1[qid] = {
            "fused": fused,
            "order": order,
            "scores": scores,
            "top1": max(scores),
            "ndcg": metrics.per_query_ndcg_at_k(task.qrels.get(qid, {}), order, 10),
        }
        if done % 200 == 0:
            print(f"  {done}/{len(task.queries)}")

    print("refining every query, so the benefit can be read at any threshold ...")
    refined_queries: dict[str, str] = {}
    unchanged = 0
    for qid, row in pass1.items():
        qtext = task.queries[qid]
        plan = build_plan(qtext, settings)
        pool = list(row["order"])
        results = as_fused(pool, [row["fused"][d] for d in pool], row["scores"])
        new_plan = refine_plan(plan, results, settings)
        if not plans_differ(plan, new_plan):
            unchanged += 1
            continue
        # The loop retrieves each effective query; joining them is the same
        # union of terms the fan-out would issue, at one retrieval instead of N.
        refined_queries[qid] = " ".join(new_plan.effective_queries)
    print(f"  {len(refined_queries)} rewritten, {unchanged} had nothing new to try")

    print("pass 2: retrieving the rewritten plans ...")
    observations = []
    if refined_queries:
        dense2, sparse2 = retrieve(refined_queries)
        for done, qid in enumerate(sorted(refined_queries), start=1):
            row = pass1[qid]
            fused2 = fuse(
                ranked(dense2.get(qid, {})),
                ranked(sparse2.get(qid, {})),
                sparse_weight,
                args.top_k,
            )
            # agent.loop fuses the union of both passes, not pass 2 alone.
            union = dict(row["fused"])
            for doc_id, score in fused2.items():
                union[doc_id] = max(union.get(doc_id, 0.0), score)
            union = dict(sorted(union.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top_k])
            order2, _ = rerank(task.queries[qid], union)
            observations.append(
                {
                    "qid": qid,
                    "top1": row["top1"],
                    "ndcg_pass1": row["ndcg"],
                    "ndcg_pass2": metrics.per_query_ndcg_at_k(
                        task.qrels.get(qid, {}), order2, 10
                    ),
                }
            )
            if done % 200 == 0:
                print(f"  {done}/{len(refined_queries)}")

    if not observations:
        print("no query was refinable", file=sys.stderr)
        return 4

    deltas = [o["ndcg_pass2"] - o["ndcg_pass1"] for o in observations]
    helped = sum(1 for d in deltas if d > 1e-9)
    hurt = sum(1 for d in deltas if d < -1e-9)
    print(
        f"\nrefinable queries: {len(observations)}  "
        f"helped {helped}  hurt {hurt}  unchanged {len(observations) - helped - hurt}"
    )
    print(f"mean NDCG@10 delta over refinable queries: {statistics.fmean(deltas) * 100:+.4f}")

    rows = []
    for threshold in TOP1_CANDIDATES:
        fired = [o for o in observations if o["top1"] < threshold]
        if not fired:
            rows.append({"agent_sufficiency_top1": threshold, "refined": 0, "mean_delta": None})
            continue
        d = [o["ndcg_pass2"] - o["ndcg_pass1"] for o in fired]
        rows.append(
            {
                "agent_sufficiency_top1": threshold,
                "refined": len(fired),
                "refine_rate": round(len(fired) / len(pass1), 4),
                "mean_delta": round(statistics.fmean(d) * 100, 4),
                "helped": sum(1 for v in d if v > 1e-9),
                "hurt": sum(1 for v in d if v < -1e-9),
                # What the whole corpus gains, not just the queries that paid.
                "corpus_delta": round(sum(d) / len(pass1) * 100, 4),
            }
        )
        print(
            f"  top1={threshold:<7} refined={rows[-1]['refined']:<5} "
            f"mean_delta={rows[-1]['mean_delta']:+8.4f} "
            f"corpus_delta={rows[-1]['corpus_delta']:+8.4f} "
            f"helped={rows[-1]['helped']} hurt={rows[-1]['hurt']}"
        )

    payload = {
        "question": "OQ-10 (refinement benefit)",
        "task": args.task,
        "split": "train",
        "profile": args.profile,
        "queries_pass1": len(pass1),
        "queries_refinable": len(observations),
        "reranker": scorer.name,
        "mean_delta_over_refinable": round(statistics.fmean(deltas) * 100, 4),
        "helped": helped,
        "hurt": hurt,
        "threshold_candidates": rows,
        "note": "Measured on the TRAIN split only; the test split was not consulted (NG-29).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
