# Build Log

The commit-by-commit engineering history of Axiom: what was built, in what order, why each change
was made, and what was measured rather than assumed.

**Developed by:** Anish Grover
**Owner:** Anish Grover
**Last updated:** 2026-09-24
**Status:** Active — branch `build/axiom-implementation`, 16 commits, **not yet pushed**

Related: [Changelog.md](Changelog.md) · [Tracker.md](Tracker.md) · [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) · [TestPlan.md](TestPlan.md) · [OpenQuestions.md](OpenQuestions.md) · [Decisions.md](Decisions.md)

---

## 0. What this document is, and what it is not

Three documents describe history here and they answer different questions. Reaching for the wrong
one is the usual way a fact gets recorded twice and then drifts.

| Document | Question it answers | Unit |
|---|---|---|
| [Changelog.md](Changelog.md) | *What shipped, at which milestone gate?* | version tag (`0.1.0` → `1.0.0`) |
| [Tracker.md](Tracker.md) | *Which planned task is done, and what proves it?* | `T-###` row |
| **This document** | *What was actually built, commit by commit, and what did measuring it reveal?* | git commit |

This is the engineering record. It is the one place where a change that was **wrong** and later
corrected stays visible, because "this was measured, found wrong, and fixed" is a different and more
useful claim than "this was always right" — the same reasoning `Rules.md` `AP-14` applies to
placeholder constants, applied to the build itself.

**On authorship.** The whole of the `build/axiom-implementation` branch — every module in §2, every
commit in §3, every measurement in §4 and §5 — **was developed by Anish Grover**, who set the
architecture and scope, made every engineering decision recorded here, and reviewed and accepted each
change. Every commit on the branch is authored and committed in his name alone. The specification the
branch implements is the team's prior work (§1); the implementation is Anish's.

---

## 1. Where the branch starts

The repository began as **documentation only** — 22 files, 10,748 lines, specifying Axiom in full:
PRD, tech spec, schema contract, API surface, test plan, rules, non-goals. There was **no source
code**.

| | Commit | Date | Author |
|---|---|---|---|
| Base | `bccece0` | 23 Sep 2026 | HarshdeepAthawale — docs; deadline moved to 27 Sep 2026 |

Everything in §3 below is the implementation branch built on top of that specification.

---

## 2. What exists now

| | Count |
|---|---|
| Source modules (`src/axiom/`) | 55 files, 21,291 lines |
| Tests (`tests/`) | 11 modules, 7,195 lines, **654 passing, 0 skipped** |
| Documentation (`docs/` + root) | 24 files, 13,698 lines, 0 dead links |
| Operational scripts (`scripts/`) | 8 |
| Commits on the branch | 16 |

### Package breakdown

| Package | Files | Lines | Responsibility |
|---|---|---|---|
| `schema/` | 7 | 653 | The frozen Pydantic contract every other package codes against |
| `core/` | 5 | 451 | Hashing, errors, logging, timing primitives |
| `chunking/` | 4 | 2,243 | AST chunking with a regex degradation rung |
| `indexing/` | 6 | 2,706 | Embedder, dense/sparse/structural index builders, manifests |
| `retrieval/` | 6 | 2,338 | The three retrieval signals and weighted RRF fusion |
| `rerank/` | 2 | 1,021 | ONNX cross-encoder with a passthrough ladder |
| `agent/` | 7 | 2,304 | Classifier, planner, sufficiency evaluator, bounded loop |
| `versioning/` | 5 | 1,667 | Incremental reindex, git diff, worktrees, snippet families |
| `eval/` | 3 | 1,900 | MTEB adapter and metrics |
| `api/` | 4 | 1,782 | Five HTTP endpoints over the one pipeline |
| `ui/` | 2 | 1,114 | Streamlit surface |
| `cli.py`, `pipeline.py` | 2 | 2,840 | Typer CLI and the orchestrator all three surfaces share |

---

## 3. Commit history

Newest last. Every row is a real commit on `build/axiom-implementation`.

| # | Commit | Date | Scope | Change |
|---|---|---|---|---|
| 1 | `405a4a7` | 23 Sep | 31 files, +1,342 | Foundation: schema contract, core primitives, config profiles |
| 2 | `d821e61` | 23 Sep | 91 files, +26,424 | Full pipeline: chunking, three signals, fusion, rerank, agent loop, surfaces |
| 3 | `618ed6f` | 23 Sep | 32 files, +6,748 −1,509 | Version-aware indexing, demo corpus generator, doc reconciliation |
| 4 | `0411122` | 23 Sep | 19 files, +1,467 −126 | Restore `PROJECT_OVERVIEW.md` as the front-door document |
| 5 | `5c1d76d` | 23 Sep | 5 files, +131 −75 | Hybrid ablation arm: +0.19 NDCG, no recall gain |
| 6 | `e2e3879` | 23 Sep | 20 files, +640 −89 | Resolve `OQ-02`, `OQ-05`, `OQ-09`, `OQ-11`; make the `AP-14` guard reachability-aware |
| 7 | `76aa89e` | 23 Sep | 8 files, +422 −38 | Fix dense tie-break determinism; submission kit; real-stack reconciliation |
| 8 | `8d22680` | 23 Sep | 2 files, +26 −13 | Surface the hybrid backend's encoder so the `NFR-09` guard can see it |
| 9 | `63cd7d2` | 23 Sep | 1 file, +11 −9 | Record the reportable eval artifact with the encoder named |
| 10 | `a6e8ac4` | 23 Sep | 12 files, +336 −37 | Close the `index --json` contract drift; add opt-in CORS |
| 11 | `7714308` | 23 Sep | 9 files, +535 −129 | Expose `FR-21` families over HTTP; one loader behind both surfaces |
| 12 | `0e139a8` | 23 Sep | 2 files, +16 −8 | Re-baseline the test counts against the measured suite |
| 13 | `c3c0522` | 24 Sep | 2 files, +15 −11 | Re-measure the suite in the project virtualenv, not the system interpreter |
| 14 | `d71a3c3` | 24 Sep | 3 files, +445 −2 | Fix two latent reranker bugs that made the demo's cross-encoder a no-op |
| 15 | `0771d0d` | 24 Sep | 3 files, +405 −4 | Measure `OQ-10`'s first half; make new sweep evidence committable |
| 16 | `17d8de6` | 24 Sep | 3 files, +405 | Add this document, plus the `OQ-10` refinement evidence |

### 3.1 Phase one — build it end to end (commits 1–4)

The scope decision was *full skeleton, then depth*: wire every stage so a demo always runs, then
deepen. `405a4a7` laid the frozen `schema/` contract first, deliberately, because four workstreams
code against it and a contract that changes under them costs more than it saves. `d821e61` is the
bulk of the system — chunking through to all three presentation surfaces. `618ed6f` added
version-aware indexing (`git worktree`-based historical indexing) and the deterministic multi-version
demo corpus generator, since no repository we do not control can satisfy `NFR-02`, `FR-19` and
`FR-21` simultaneously.

### 3.2 Phase two — measure it (commits 5–9)

This is where claims became numbers. `5c1d76d` added the hybrid ablation arm. `e2e3879` closed four
open questions against real sweeps and made the `AP-14` placeholder guard *reachability-aware* — a
placeholder that a profile cannot read is not a placeholder that profile rests on, and conflating the
two blocked reportable runs for no reason.

`8d22680` and `63cd7d2` are a correction worth keeping visible. The first reportable artifact was
stamped while the `NFR-09` provenance check could not see the hybrid backend's encoder — the numbers
were right, the *evidence that they were right* was not. The guard was fixed, proved to fire, and the
artifact regenerated.

### 3.3 Phase three — make it consumable (commits 10–13)

Preparation for a browser frontend, which surfaced three contract defects:

- `axiom index --json` emitted `index_kind` where the contract says `dense_index_kind`, and a nested
  timings record where the contract types a flat `dict[str, float]`. A client typed against the docs
  could read neither.
- `API.md` claimed `reindex` emits `IndexSummary`. It does not, and should not — that shape would
  drop `blobs_reused` and `embed_calls`, which are how `FR-19`'s zero-cost-rename claim is *measured*
  rather than asserted.
- `POST /v1/query` accepted `all_versions: true` and had nowhere to return the families, so the
  `FR-21` evolutionary feature was reachable from the CLI and nowhere else.

`a6e8ac4` also added opt-in CORS (`AXIOM_API_CORS_ORIGINS`) — off by default, so the `NG-08` posture
is unchanged when unset, because a browser frontend on its own port is the one thing no amount of
correct backend code can work around.

`0e139a8` and `c3c0522` are a second visible correction. Test counts were quoted from the system
interpreter, which has none of the optional extras. The project's `.venv` has all of them, and is
where the reported eval numbers came from. A bare-install figure of "577 passed, 68 skipped" was also
withdrawn: it was measured, but the measurement was invalid — blocking `starlette` to simulate a
missing `serve` extra also breaks `streamlit`, turning clean skips into 66 errors.

### 3.4 Phase four — run the real models (commits 14–15)

`onnxruntime` was installed but the ONNX **exports** were not, so the cross-encoder had never
actually run. Exporting `ms-marco-MiniLM-L-6-v2` to INT8 ([Setup.md §6.1](Setup.md#61-export)) and
running it surfaced two bugs that no test could have caught, because every existing rerank test
drives a fake:

1. **The token budget ignored which ladder rung loaded.** `AXIOM_RERANK_MAX_TOKENS` is one number
   (1536), but `bge-reranker-v2-m3` accepts 8192 while `ms-marco-MiniLM-L-6-v2` stops at 512. Feeding
   1536 to the smaller model does not truncate — it fails the ONNX graph, which `rerank_detailed`
   catches and turns into a passthrough. Since the demo profile promotes `ms-marco` to primary, **the
   headline cross-encoder stage would have read as configured-and-running while scoring nothing.**
   The budget is now capped by the checkpoint's own `max_position_embeddings`.
2. **`truncation="only_second"` raised on long queries.** It encodes "the query is short, spend the
   budget on the document" — true for a search box, false for a benchmark whose queries are whole
   problem statements. When false the tokenizer raises rather than truncating, which also surfaced as
   a silent passthrough. Now `longest_first`.

Nine tests now cover the real ONNX path.

---

## 4. What was measured

### 4.1 Retrieval — full test split, 3,765 queries / 8,765 documents

| Arm | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| Sparse only | 0.91 | — | — |
| Dense only | 7.59 | 6.39 | 27.22 |
| **Hybrid RRF @ 0.15** | **7.78** | **6.60** | 27.17 |

Encoder `all-MiniLM-L6-v2` (the 0.6B primary has no ONNX export), rerank in passthrough. Artifact
`appsretrieval_results.json`, `REPORTABLE` at `8d22680`, dataset revision `f22508f96b`.

**The binding constraint is Recall@100 = 27.2.** Fusion reorders the candidate pool without enlarging
it (+0.19 NDCG, +0.00 recall), so reranking and agent refinement — both strictly downstream — cannot
reach the ~73% of relevant documents that never enter the pool. The lever is a stronger first-stage
embedder, not a better reranker.

### 4.2 Incremental reindex (`FR-19`, `NFR-02`)

| Metric | Measured | Budget |
|---|---|---|
| Reindex wall clock | 295 ms | 45 s |
| Embedding calls | 0 | — |
| Blobs reused | 101 | — |
| Blobs written for 303 chunk-instances | 201 | — |
| Families carrying real diffs | 50 of 107 | — |

### 4.3 Tuned constants

| Constant | Value | Evidence |
|---|---|---|
| `eval_sparse_weight` | 0.15 | `OQ-02` — swept 0.0–0.5 on 1,500 train queries; 0.15 gives NDCG@10 26.02 vs 25.06 dense-only |
| `dedupe_cosine` | 0.95 | `OQ-11` — 45,753 pairs over a 3-version corpus; minimum-error threshold |

All tuning is on the **train** split. `NG-29` fences the test split off absolutely.

---

## 5. Open questions, and why they are open

### 5.1 `OQ-10` — the agent sufficiency thresholds

Measured in two halves on the train split, with the real INT8 cross-encoder.

**Can the top-1 rerank score tell a weak result list from a strong one?** Yes, modestly.
Spearman(top-1, NDCG@10) = **0.293** over 1,500 queries, with separation rising monotonically with
the threshold. *(An earlier 120-query run put ρ at 0.062 with a non-monotonic curve and was read as
"this predicate cannot discriminate." That was a small-sample artifact; the full run is the one to
believe.)*

**Does refining actually help?** No — it is a wash. Over 592 refinable queries:

| | Count |
|---|---|
| Unchanged by refinement | 541 |
| Helped | 25 |
| Hurt | 26 |
| Mean NDCG@10 delta | **−0.037** |

At the placeholder `0.35`, refinement fires on **90.7%** of queries for a corpus-wide delta of
**−0.041** — a second pass on nearly every query, for slightly worse results.

**Conclusion: the constant is not the problem, the treatment is.** Picking a "better" threshold would
be dressing up noise. Two caveats belong on this finding: the refinement path here joins the
rewritten plan's `effective_queries` into one retrieval rather than issuing the per-sub-query fan-out
`agent.loop` does, and `refine_plan` ran on its **heuristic rung** because no LLM is configured — an
LLM-written rewrite could plausibly do better.

Evidence: `data/sweeps/oq10_sufficiency.json`, `data/sweeps/oq10_refinement.json`.

### 5.2 `stability_bonus` (`OQ-11`) — unmeasurable with what exists

Tuning it needs a corpus that is both multi-version **and** labelled. AppsRetrieval is labelled but
has no versions; the demo corpus has versions but no relevance ground truth. Inventing labels would
manufacture the appearance of measurement, which is worse than leaving the constant flagged.

### 5.3 `chunk_min_tokens` / `chunk_target_tokens` (`OQ-09`) — unreachable on this benchmark

The eval harness maps one corpus document to one chunk, so chunking never runs on the reported path.
`run_eval.py` records them as `placeholders_excluded_as_unreachable` for exactly this reason. Bounds
were measured on the synthetic corpus (0 chunks below floor, 0 above target) and marked indicative.

### 5.4 Which placeholders actually bind

| Profile | `agent_enabled` | `evolutionary_enabled` | Live placeholders |
|---|---|---|---|
| `eval` | off | off | the two chunk bounds — both unreachable |
| `demo` | on | on | all five |

**The reported NDCG rests on zero live placeholders.** `AP-14` is satisfied on the merits, not by
exclusion. All three genuinely open placeholders affect the demo path only, and the eval profile
never runs the agent loop at all.

---

## 6. Corrections kept on the record

Listed because a build log that only records successes is a marketing document.

| What was claimed | What was true | Fixed in |
|---|---|---|
| First reportable artifact was sound | The `NFR-09` encoder check was inert when it was stamped | `8d22680`, `63cd7d2` |
| Suite is "611 / 642 passing" | Counted on the system interpreter, not the project `.venv` | `0e139a8`, `c3c0522` |
| Bare install is "577 passed, 68 skipped" | Measurement invalid — blocking `starlette` breaks `streamlit` | `c3c0522` |
| Sufficiency predicate cannot discriminate (ρ=0.06) | 120-query artifact; ρ=0.293 at 1,500 | §5.1 |
| Cross-encoder loads in `.venv` | `onnxruntime` was installed, the ONNX *exports* were not | `d71a3c3` |
| `is_multi_version` is a `GET /v1/families` response field | It is a Python property, never serialised | `7714308` |

A seventh belongs here from before this branch: an external comparison table cited "BM25 4.8 /
BGE-M3 7.37" favourably while an unfavourable 14.7 from the *same unverified source* had been
quarantined. Citing one and not the other is selection, not evidence. All three were withdrawn and
replaced with the three measured arms in §4.1.

---

## 7. State and what is not done

**The branch is local.** 16 commits on `build/axiom-implementation`, **nothing pushed** to
`HarshdeepAthawale/Samsung-Prism-Hack` (since renamed `HarshdeepAthawale/AXIOM`).

| Item | State |
|---|---|
| Backend | Complete — 5 endpoints, CLI, Streamlit, opt-in CORS, OpenAPI 3.1 with 20 schemas |
| Test suite | 654 passing, 0 skipped, in `.venv` |
| Eval artifact | `REPORTABLE`, pinned to git SHA and dataset revision |
| Frontend (JS) | Not started — framework and hosting model undecided |
| Push to GitHub | **Not done** — outward-facing, awaiting the owner's call |
| Release tag `PRISM_GENAI_HACKATHON_Y2026` | Not cut |
| Slide deck, demo video | Not produced |

### 7.1 Environment notes

- The project `.venv` (Python 3.11.14, uv-managed) has every optional extra. Test counts and eval
  numbers are only meaningful from it.
- `optimum` was installed into that venv to export the reranker. It is an export-time tool, not a
  runtime dependency.
- `data/models/onnx/ms-marco-minilm-l-6-v2-int8/` (23 MB) is gitignored per
  [Setup.md §6](Setup.md#6-onnx-export-and-int8-quantisation) and must be re-exported on a fresh box.
