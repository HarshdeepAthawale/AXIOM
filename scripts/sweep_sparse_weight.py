"""Resolve `OQ-02`: the sparse weight in ``configs/eval.yaml``.

Run on the **train** split only. `NG-29` fences the test split off from every
tuning activity, and that fence is absolute: a weight chosen by looking at test
makes the reported number self-selected and the claim worthless.

The naive way to sweep is to re-run the whole evaluation once per candidate
weight, which re-embeds the corpus and re-retrieves every query each time. That
is wasted work: the *ranked lists* do not depend on the fusion weight, only the
fusion of them does. So this retrieves once, keeps the per-signal ranks, and
re-fuses in memory at every candidate weight. One retrieval pass, N fusions.

    python scripts/sweep_sparse_weight.py --limit 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_backends import dense_only

from axiom.config import get_settings
from axiom.eval import metrics
from axiom.eval.mteb_adapter import (
    LexicalBackend,
    build_search_model,
    load_mteb_task,
)

#: Candidate weights for the sparse leg. 0.0 is included deliberately: if the
#: best weight is zero, the honest conclusion is that BM25 earns nothing on this
#: task and the eval profile should say so rather than carry a token value.
CANDIDATES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

RRF_K = 60


def load_train_task(task_name: str, limit: int | None):
    """Load the CoIR apps **train** partition straight from the HF parquet files.

    `OQ-05` asked whether the 5,000 train pairs are usable for tuning. They are,
    but not through MTEB's task API: `mteb` declares only ``test`` as an eval
    split for ``AppsRetrieval``, so ``load_mteb_task(..., "train")`` raises. The
    underlying dataset does ship the train qrels, and both the corpus and the
    query files carry a ``partition`` column separating the two.

    Loading it directly is therefore the correct route, and it also gives the
    strictest possible `NG-29` guarantee: this restricts the corpus to the train
    partition, so a weight chosen here has never seen a test *document*, let
    alone a test query.
    """
    from datasets import load_dataset

    from axiom.eval.mteb_adapter import RetrievalTaskData

    repo = "CoIR-Retrieval/apps"
    corpus_rows = load_dataset(repo, "corpus", split="corpus")
    query_rows = load_dataset(repo, "queries", split="queries")
    qrel_rows = load_dataset(repo, split="train")

    corpus = {
        r["_id"]: {"title": r.get("title") or "", "text": r["text"]}
        for r in corpus_rows
        if r.get("partition") == "train"
    }
    queries = {r["_id"]: r["text"] for r in query_rows if r.get("partition") == "train"}
    qrels: dict[str, dict[str, float]] = {}
    for row in qrel_rows:
        qid, did = str(row["query-id"]), str(row["corpus-id"])
        if qid in queries and did in corpus:
            qrels.setdefault(qid, {})[did] = float(row["score"])

    queries = {q: t for q, t in queries.items() if q in qrels}
    if limit:
        keep = sorted(queries)[:limit]
        queries = {q: queries[q] for q in keep}
        qrels = {q: qrels[q] for q in keep}

    return RetrievalTaskData(
        task_name=task_name,
        split="train",
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        source="hf:CoIR-Retrieval/apps (train partition, loaded directly)",
    )


def fuse(dense_hits, sparse_hits, sparse_weight, top_k, rrf_k=RRF_K):
    """Weighted reciprocal-rank fusion of two already-ranked lists."""
    dense_weight = 1.0 - sparse_weight
    fused: dict[str, float] = {}
    for hits, weight in ((dense_hits, dense_weight), (sparse_hits, sparse_weight)):
        if weight <= 0.0:
            continue
        for rank, (doc_id, _score) in enumerate(hits, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (rrf_k + rank)
    # Ties break on doc_id so a sweep is reproducible (NFR-08).
    return dict(sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="AppsRetrieval")
    parser.add_argument("--split", default="train", help="Never 'test' (NG-29).")
    parser.add_argument("--limit", type=int, default=1500, help="Train queries to sweep over.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data/sweeps/oq02_sparse_weight.json"))
    args = parser.parse_args(argv)

    if args.split == "test":
        print("refusing: NG-29 forbids tuning on the test split", file=sys.stderr)
        return 2

    settings = get_settings("eval")
    # mteb exposes only the test split for this task; the train partition is
    # loaded directly from the dataset (see load_train_task).
    if args.split == "train":
        task = load_train_task(args.task, args.limit)
    else:
        task = load_mteb_task(args.task, args.split)
        if args.limit:
            task = task.limited(args.limit)
    print(f"{args.task}/{args.split}: {len(task.corpus)} docs, {len(task.queries)} queries")

    # Wrap each leg in AxiomSearchModel rather than driving the backends
    # directly: it owns corpus -> chunk conversion and the chunk-id -> doc-id
    # mapping back, and both are exactly what the reported runs use. Driving the
    # backends raw would tune against a slightly different pipeline than the one
    # being measured.
    dense_model = build_search_model(settings, backend=dense_only(settings))
    sparse_model = build_search_model(settings, backend=LexicalBackend())

    print("indexing both legs once ...")
    for model in (dense_model, sparse_model):
        model.index(task.corpus, hf_split=task.split)

    print("retrieving once per query ...")
    dense_run = dense_model.search(task.queries, hf_split=task.split, top_k=args.top_k)
    sparse_run = sparse_model.search(task.queries, hf_split=task.split, top_k=args.top_k)

    def ranked(run_for_query: dict[str, float]) -> list[tuple[str, float]]:
        # Score -> rank, ties broken on doc id so the sweep is reproducible.
        return sorted(run_for_query.items(), key=lambda kv: (-kv[1], kv[0]))

    per_query = {
        qid: (ranked(dense_run.get(qid, {})), ranked(sparse_run.get(qid, {})))
        for qid in task.queries
    }

    rows = []
    for weight in CANDIDATES:
        run = {qid: fuse(d, s, weight, args.top_k) for qid, (d, s) in per_query.items()}
        scored = metrics.evaluate_run(task.qrels, run, k_values=(10, 100))
        rows.append(
            {
                "sparse_weight": weight,
                "dense_weight": round(1.0 - weight, 4),
                "ndcg_at_10": round(scored["ndcg_at_10"] * 100, 4),
                "mrr_at_10": round(scored["mrr_at_10"] * 100, 4),
                "recall_at_100": round(scored["recall_at_100"] * 100, 4),
            }
        )
        print(
            f"  sparse={weight:<5} ndcg@10={rows[-1]['ndcg_at_10']:7.4f} "
            f"mrr@10={rows[-1]['mrr_at_10']:7.4f} recall@100={rows[-1]['recall_at_100']:7.4f}"
        )

    best = max(rows, key=lambda r: r["ndcg_at_10"])
    payload = {
        "question": "OQ-02",
        "task": args.task,
        "split": args.split,
        "queries": len(task.queries),
        "corpus": len(task.corpus),
        "encoder": getattr(dense_model.backend.encoder, "model_id", "unknown"),
        "rrf_k": RRF_K,
        "candidates": rows,
        "selected": best,
        "note": "Selected on the TRAIN split only; the test split was not consulted (NG-29).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nbest sparse weight: {best['sparse_weight']} (ndcg@10 {best['ndcg_at_10']})")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
