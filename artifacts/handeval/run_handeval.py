"""Hand-labelled retrieval measurement over tests/fixtures/repo_v1 (36 chunks)."""
import json, subprocess, os, sys, shutil, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qrels import QUERIES
sys.path.insert(0, "/Users/anish/Desktop/prism hack/src")
from axiom.eval.metrics import per_query_ndcg_at_k, per_query_mrr_at_k, per_query_recall_at_k

ROOT="/Users/anish/Desktop/prism hack"
LIVE=f"{ROOT}/data/models/onnx/all-minilm-l6-v2-int8"; PARK="/tmp/minilm_parked_he"

ARMS = {
 # label:                (index_root,   env overrides,                                        extra cli)
 "A dense-only  HASH":   (".axiom_hash", dict(AXIOM_SPARSE_ENABLED="false", AXIOM_STRUCTURAL_ENABLED="false"), ["--no-agent"]),
 "B dense-only  MiniLM": (".axiom_real", dict(AXIOM_SPARSE_ENABLED="false", AXIOM_STRUCTURAL_ENABLED="false"), ["--no-agent"]),
 "C sparse-only (bm25s)":(".axiom_real", dict(AXIOM_DENSE_ENABLED="false", AXIOM_STRUCTURAL_ENABLED="false"), ["--no-agent"]),
 "D fused demo  HASH":   (".axiom_hash", {}, ["--no-agent"]),
 "E fused demo  MiniLM": (".axiom_real", {}, ["--no-agent"]),
 "F fused+agent MiniLM": (".axiom_real", {}, []),
}

def run(root, q, env_over, extra):
    env = dict(os.environ, AXIOM_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
               AXIOM_EMBEDDING_DIM="384", **env_over)
    cmd=[f"{ROOT}/.venv/bin/axiom","--profile","demo","--index-root",root,"--json",
         "query",q,"--top-k","10"]+extra
    p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,env=env)
    if p.returncode!=0: raise SystemExit(f"FAIL {cmd}\n{p.stderr[-2000:]}")
    return json.loads(p.stdout)

def key(r):
    l=r["chunk"]["location"]; return f'{l["file_path"]}:{l["start_line"]}'

results={}
for label,(root,env_over,extra) in ARMS.items():
    park = "HASH" in label
    if park: shutil.move(LIVE,PARK)
    try:
        per={}
        t0=time.time()
        for qid,(text,rel) in QUERIES.items():
            d=run(root,text,env_over,extra)
            ranked=[key(x) for x in d["results"]]
            # dedupe while preserving rank (fused lists can repeat a location across versions)
            seen=set(); order=[]
            for k in ranked:
                if k not in seen: seen.add(k); order.append(k)
            scored={k:1.0/(i+1) for i,k in enumerate(order)}
            qr={k:1.0 for k in rel}
            per[qid]=dict(
                ndcg10=per_query_ndcg_at_k(qr,scored,10),
                mrr10=per_query_mrr_at_k(qr,scored,10),
                recall10=per_query_recall_at_k(qr,scored,10),
                top1=order[0] if order else None, hit1=bool(order and order[0] in rel),
                n=len(order))
        el=time.time()-t0
    finally:
        if park: shutil.move(PARK,LIVE)
    n=len(per)
    results[label]=dict(
        ndcg10=sum(v["ndcg10"] for v in per.values())/n,
        mrr10=sum(v["mrr10"] for v in per.values())/n,
        recall10=sum(v["recall10"] for v in per.values())/n,
        hit1=sum(v["hit1"] for v in per.values())/n,
        per=per, elapsed=el)
    print(f'{label:24s} NDCG@10={results[label]["ndcg10"]*100:5.2f}  MRR@10={results[label]["mrr10"]*100:5.2f}  '
          f'R@10={results[label]["recall10"]*100:5.2f}  P@1={results[label]["hit1"]*100:5.1f}  ({el:.0f}s)', flush=True)

json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"handeval.json"),"w"), indent=1)
print("\n--- per-query NDCG@10, hash-dense vs MiniLM-dense vs fused-MiniLM ---")
A,B,E=results["A dense-only  HASH"]["per"],results["B dense-only  MiniLM"]["per"],results["E fused demo  MiniLM"]["per"]
for qid,(text,rel) in QUERIES.items():
    print(f'{qid} A={A[qid]["ndcg10"]:.3f} B={B[qid]["ndcg10"]:.3f} E={E[qid]["ndcg10"]:.3f}  delta(B-A)={B[qid]["ndcg10"]-A[qid]["ndcg10"]:+.3f}  {text[:54]}')
