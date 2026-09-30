# Submission Kit

The PPT outline, the demo-video script and the demo-day runbook, in the order they are needed.

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-23
**Status:** Active

Related: [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) · [PRD.md](PRD.md) · [Deployment.md](Deployment.md) · [Tracker.md](Tracker.md)

---

## 0. The one rule for everything below

**Every number that reaches a slide, the video, or the jury must be one we ran.** Three runs exist
and all three are logged. No external model's score appears anywhere without a citation a human on
this team has personally opened and read — that rule cost us the `BGE 0.6B = 14.7` row, and it
applies with equal force to numbers that flatter us. See [PRD.md §2](PRD.md).

---

## 1. PPT outline

Twelve slides. The jury weighting each slide serves is in brackets.

| # | Slide | Content | Serves |
|---|---|---|---|
| 1 | Title | Axiom — Agentic Code Intelligence. Team Incognito, Thapar Institute. Theme 01. | — |
| 2 | The question engineers actually ask | Not "write me code" but **"where does this already happen?"**. 10k JS files, ~50M tokens, no context window holds it. Three archetype queries Q1/Q2/Q3 verbatim from the problem statement. | Relevance 15% |
| 3 | Why one signal is not enough | The three failure modes, one row each: "preprocessing" never matches `normalize()` lexically; "calls X before Y" is not in the text of any snippet; a literal deeplink string is diluted by dense search. This slide motivates everything after it. | Depth 25% |
| 4 | Architecture | The one diagram: classify → 3 signals in parallel → weighted RRF → cross-encoder → bounded agent loop → ranked `file:line`. Mark clearly which parts are **measured** and which are **projected**. | Depth 25% |
| 5 | The structural signal | The differentiator. `ChunkMetadata.calls` keeps **source order**, so ordered-pair queries are answerable by SQL over a call graph. Show the real Q2 output with the ordinal evidence. Say plainly: no embedding of any size answers this. | Innovation 20% |
| 6 | The agent loop, bounded | Sufficiency predicate, ≤2 passes, hard 5 s monotonic deadline checked *before* each pass. Agentic in a way that is measurable, not decorative. **The LLM never reads code** — it classifies, expands, decomposes, judges sufficiency. Enforced by module boundary. | Innovation 20% + Relevance 15% |
| 7 | Results — what we measured | The ablation table from §3 below, plus the honest headline: the binding constraint is **Recall@100**, not NDCG. | Prototype 30% + Depth 25% |
| 8 | Results — what we did not reach | The ≥20.0 target and why: a 22M-parameter fallback embedder, because the 0.6B primary has no ONNX export yet. State the gap, own it, show the diagnosis. A jury trusts a team that reports its own miss. | Depth 25% |
| 9 | Version-aware retrieval (P1) | A 50-file diff re-embeds only its **50 changed chunks** and reuses the other 51, in 5.0 s against a 45 s budget. Content already seen (a rename, a revert) costs **0 embedding calls**. Content addressing is why. | Prototype 30% |
| 10 | Evolutionary retrieval (Bonus) | 109 snippet families over 3 versions, 44 carrying real diffs. Dedupe threshold 0.95 **measured**, not guessed: minimum-error over 45,753 pairs. | Innovation 20% |
| 11 | It runs on the evaluator's laptop | The degradation ladder. Four rungs, every one exercised: no models, no faiss, no bm25s, no tree-sitter — still returns ranked results. This is why the demo cannot fail on an unknown machine. | Prototype 30% |
| 12 | Engineering discipline | Locked Pydantic contract, 500+ tests, the eval harness that **refuses to mark its own run reportable** while a placeholder is active or the tree is dirty. Show that refusal on screen. | Depth 25% + Docs 10% |

**Slide 12 is the sleeper.** A harness that blocks its own number is a stronger integrity signal
than any number it could have printed.

---

## 2. Demo video script (≤ 5 minutes)

Record at 1080p, terminal at a large font, no cuts inside a command's execution — the jury needs to
see real wall-clock time elapse.

| Time | Action | Say |
|---|---|---|
| 0:00–0:25 | Title slide, then a terminal | "Axiom answers *where does this already happen* over a codebase too large for any context window. Retrieval only — we never generate code." |
| 0:25–1:00 | `axiom index /path/to/demo --at v3.0.0` | "Three indexes: dense embeddings, BM25, and a SQLite call graph. Watch the degradation lines — it's naming which model rung actually loaded." |
| 1:00–1:45 | **Q1** semantic query | "The query never names `preprocessInput`, and it still comes back in the top five, found by meaning rather than by keyword. Note the per-signal ranks — that panel is the hybrid architecture, visible." Check the ranking before recording: with the MiniLM models it sits at rank 4, behind `main` because the query says "main function". |
| 1:45–2:40 | **Q2** structural query | "This is the one no embedding answers. It's a question about the call graph and about *order*." Point at `calls parseIntent (ordinal 0) before resolveTool`. "That ordinal is read from source order we preserved at chunk time." |
| 2:40–3:10 | **Q3** usage query | "A literal string. BM25 wins outright, and fusion lets it win — that's what per-query-type weights are for." |
| 3:10–3:50 | `axiom reindex --from v1.0.0 --to v2.0.0` | "Fifty files changed, so exactly fifty chunks are re-embedded. The other fifty-one are reused by content hash. Code that is only renamed or moved costs zero embedding calls." Read the real counts off the `embeddings` line on screen. |
| 3:50–4:25 | `axiom families --diffs` | "The same function across three versions, collapsed into one result with diffs, instead of three near-identical hits crowding the list." |
| 4:25–5:00 | `AXIOM_LLM_ENABLED=false`, fast profile, rerun Q1 | "No LLM, fallback models. Still ranked results. This is what runs on your laptop." |

**Do not** show the NDCG number in the video. It belongs on a slide where the caveats fit.

---

## 3. The results table, as it goes on the slide

CoIR `AppsRetrieval`, **full test split** — 3,765 judged queries over 8,765 documents, no
truncation. Embedder `all-MiniLM-L6-v2` (22M params, the declared fallback). Reranker passthrough:
no cross-encoder weights are exported yet.

| Arm | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| Sparse only (BM25) | 0.91 | — | — |
| Dense only | 7.59 | 6.39 | 27.22 |
| Dense + sparse, weighted RRF @ 0.15 | **7.78** | **6.60** | 27.17 |
| PRD target | 20.0 | 22.0 | 65.0 |

**Say this out loud, do not bury it:** fusion adds `+0.19` NDCG and `+0.00` recall. Recall@100 of
27.2 is the ceiling — roughly three-quarters of relevant documents never enter the candidate pool,
and nothing downstream of first-stage retrieval can reach them. Reranking and agentic refinement
reorder; they cannot retrieve. The lever is a stronger embedder.

Caveat to keep in your own head: `+0.19` on a single run with no confidence interval is not a
robust improvement. The defensible claim is "fusion does not hurt, and the recall number explains
why it does not help much."

---

## 4. Demo-day runbook

### The day before

1. `uv sync --frozen`, then run the three archetype queries end to end. Do not skip this.
2. Pre-download every model, primary and fallback, and then **rehearse once with the network off**.
3. Build the demo index and keep it. Do not rebuild live.
4. Record the video. The live demo can fail; the video cannot.

### Phrase Q2 with two code-shaped identifiers

Use **"which files call `parseIntent` before `resolveTool`"**.

Do **not** say "before dispatch". A bare lowercase word matches none of the identifier shapes the
planner extracts, so it is never lifted into `extracted_identifiers`, and the ordering evidence
silently does not fire — you get "calls parseIntent (rank 2)" instead of the ordinal comparison that
is the whole point of the slide. This is a known limitation, not a bug, and it is avoidable by
phrasing.

### Install tree-sitter, even if something else breaks

On the regex fallback the chunker does not merely produce coarser chunks — it can attribute a
function's `calls` to the wrong symbol when it fails to find a declaration boundary. Since `calls`
order is what answers Q2, **the fallback can answer the headline query confidently and wrongly.**
Every other rung degrades gracefully; this one degrades misleadingly.

### If a judge asks "why is the score low?"

The honest answer, in one breath: *"That's a 22-million-parameter fallback embedder. Our primary is
27 times larger and has no ONNX export yet. We measured that first-stage recall is the ceiling — 27%
— so we know exactly which component is responsible, and it isn't the architecture. Every other
component's contribution is in the ablation."*

Do not claim the architecture would hit 20 with a better model. We have not measured that.
