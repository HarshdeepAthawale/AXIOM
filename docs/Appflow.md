# Application Flow

Eight end-to-end runtime flows through PRISM: sequence diagrams, stage-by-stage module traces, latency budgets, and the degradation rungs each flow exercises on its unhappy path.

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-16
**Status:** Draft

Related: [Design.md](Design.md) · [TechSpecifications.md](TechSpecifications.md) · [Schema.md](Schema.md) · [PRD.md](PRD.md) · [TestPlan.md](TestPlan.md) · [Rules.md](Rules.md) · [NonGoals.md](NonGoals.md)

---

## How to read this document

Every flow below is a concrete instance of the six-stage pipeline in
[Design.md §3.1](Design.md#31-the-pipeline). The stage names (QUERY UNDERSTANDING, FIRST-STAGE
RETRIEVAL, FUSION, RERANK, SUFFICIENCY CHECK, FORMAT) are identical across all eight flows and
across [Design.md](Design.md) — if a flow skips a stage or takes a fallback rung, that is called out
explicitly rather than silently omitted, so the eight flows read as one system observed from eight
angles, not eight unrelated diagrams.

Latency numbers are the canonical budget breakdown from
[NonGoals.md NG-10](NonGoals.md#ng-10--no-production-sla-uptime-or-ha-guarantee) unless a flow states
otherwise; they are engineering budgets, not promises, per that same entry. Schema objects named on
each arrow (`QueryPlan`, `ScoredChunk`, `FusedResult`, `RetrievalResult`, `Chunk`,
`VersionManifest`, `SnippetFamily`) are defined normatively in [Schema.md](Schema.md).

| Flow | Scenario | Primary budget |
|---|---|---|
| [1](#flow-1--cold-index-build) | Cold index build | `NFR-01`, ≤ 12 min / 10k chunks |
| [2](#flow-2--standard-query-no-agent-refinement) | Standard query, no agent refinement | `NFR-03`, ≤ 900 ms p50 |
| [3](#flow-3--query-triggering-agent-refinement) | Query triggering agent refinement | `NFR-04`, ≤ 5 s p95 |
| [4](#flow-4--degraded-query-with-the-llm-disabled) | Degraded query, `AXIOM_LLM_ENABLED=false` | `NFR-07` |
| [5](#flow-5--incremental-reindex) | Incremental reindex | `NFR-02`, ≤ 45 s / 50 changed files |
| [6](#flow-6--version-scoped-query) | Version-scoped query | `FR-20` |
| [7](#flow-7--evolutionary-all-versions-query) | Evolutionary / all-versions query | `FR-21` |
| [8](#flow-8--reranker-failure-cascade) | Reranker failure cascade | `NFR-07` |

---

## Flow 1 — Cold index build

**Scenario:** `axiom index data/demo_repo --version-id v1`. First-time index of a repository with no
prior `.axiom/` state. Exercises the offline path in [Design.md §3.2](Design.md#32-what-is-offline-vs-online).

```
 operator     cli.py      chunking.        indexing.       indexing.       indexing.       indexing.
              index cmd   ast_chunker      dense           sparse          structural      manifest
    │            │             │               │               │               │               │
    │──index────>│             │               │               │               │               │
    │            │──discover──>│               │               │               │               │
    │            │  (.js files, ignore list)    │               │               │               │
    │            │<──list[Path]│               │               │               │               │
    │            │──chunk_file (per file)──────>│               │               │               │
    │            │<──list[Chunk]────────────────│               │               │               │
    │            │──────────────────build_index (embed, L2-norm, FAISS)────────>│               │
    │            │<─────────────────────dense.faiss, dense.idmap.json───────────│               │
    │            │────────────────────────────build_index (bm25s tokens)────────────────────────>│
    │            │<───────────────────────────────────────sparse.bm25s/────────────────────────│
    │            │──────────────────────────────────────────build_index (symbols/calls/imports)─────────>│
    │            │<─────────────────────────────────────────────structural.sqlite────────────────────────│
    │            │────────────────────────────────────────────────────────────write(manifest, registry)──>│
    │            │<───────────────────────────────────────────────────manifest.json, registry.json────────│
    │<──done─────│             │               │               │               │               │
```

| Step | Module : function | What it produces | Notes |
|---|---|---|---|
| 1 | `cli.py:index` | file list | Applies the ignore list from [NonGoals.md NG-26](NonGoals.md#ng-26--no-indexing-of-build-output-vendored-or-non-source-assets) (`node_modules/`, `dist/`, minified, etc.) before chunking begins |
| 2 | `chunking.ast_chunker:chunk_file` | `list[Chunk]` per file | Degrades per-file to the line-window fallback on a parse error ([Design.md §6.2](Design.md#62-why-degradation-composes-differently-depending-on-where-it-happens)); never aborts the whole index |
| 3 | `indexing.dense:build_index` | `dense.faiss`, `dense.idmap.json` | Embedding is the dominant cost — batched forward passes at `AXIOM_EMBEDDING_BATCH_SIZE=64` |
| 4 | `indexing.sparse:build_index` | `sparse.bm25s/` | Reuses the identical `Chunk` corpus dense just consumed; no re-chunking |
| 5 | `indexing.structural:build_index` | `structural.sqlite` | Skipped entirely under the `eval` profile — see [Design.md §7](Design.md#7-two-profiles-one-pipeline) |
| 6 | `indexing.manifest:write` | `manifest.json`, `registry.json` | Atomic write via `index/<version>.tmp/` + `os.replace`, per `TC-034` |

**Budget:** `NFR-01`, ≤ 12 minutes for 10,000 chunks on the 8-core / 16 GB reference box. Dense
embedding dominates wall clock; sparse and structural build concurrently with each other but after
chunking completes (both need the full `list[Chunk]`, not a streaming subset).

**Degradation rungs exercised on the unhappy path:** a syntactically broken file falls back to the
regex identifier splitter at step 2 (`TC-021`); this does not fail the whole index. A file that
cannot be read at all (permissions, encoding) is skipped with a `WARNING`, not fatal. Step 6's
manifest write is one of the three cases where raising *is* correct
([Design.md §6.3](Design.md#63-where-raising-is-still-correct)) — a partially-written index tree at
process end is a `PrismIndexError` state the next load must refuse, not silently tolerate.

---

## Flow 2 — Standard query, no agent refinement

**Scenario:** `axiom query "How is the input preprocessed before going to the main function?"` — a
`Q1`-shaped archetype query ([PRD.md §1.1](PRD.md#11-the-three-query-archetypes)) whose first pass
already satisfies the sufficiency predicate. This is the canonical happy-path query and the flow
that reproduces the p50 breakdown in [NonGoals.md NG-10](NonGoals.md#ng-10--no-production-sla-uptime-or-ha-guarantee)
exactly.

```
 user      agent.        agent.         retrieval.*     retrieval.      rerank.          agent.        formatter
           classifier    planner        (dense/sparse/  fusion          cross_encoder    evaluator
                                         structural,
                                         concurrent)
   │           │             │               │               │               │               │             │
   │──query───>│             │               │               │               │               │             │
   │           │──classify──>│               │               │               │               │             │
   │           │<─QueryType──│               │               │               │               │             │
   │           │             │──build_plan──>│               │               │               │             │
   │           │             │<─QueryPlan────│               │               │               │             │
   │           │             │───────────────┼──search (x3, concurrent)─────>│               │             │
   │           │             │               │<──list[ScoredChunk] x3────────│               │             │
   │           │             │               │───────────────>rrf(weights,k=60)              │             │
   │           │             │               │               │<──list[FusedResult] N=25──────│             │
   │           │             │               │               │───────────────>rerank(top 25) │             │
   │           │             │               │               │               │<─rerank_score, top 10────── │
   │           │             │               │               │               │───────────────>assess──────>│
   │           │             │               │               │               │               │<sufficient──│
   │           │             │               │               │               │               │             │──format──>│
   │<──────────────────────────────────────────────────────────────────────────────────────list[RetrievalResult]───────│
```

| Step | Module : function | Latency contribution | Cumulative |
|---|---|---|---|
| 1 | `agent.classifier:classify` | 60 ms | 60 ms |
| 2 | `agent.planner:build_plan` → `QueryPlan` (includes query embedding for the dense leg) | 35 ms | 95 ms |
| 3 | `retrieval.dense:search` + `retrieval.sparse:search` + `retrieval.structural:search`, concurrent (§5.1 of [Design.md](Design.md#51-why-the-three-signals-run-concurrently-not-in-a-process-pool)) | 28 ms (wall clock of the slowest of the three, not the sum) | 123 ms |
| 4 | `retrieval.fusion:reciprocal_rank_fusion` → `list[FusedResult]`, N=25 | 3 ms | 126 ms |
| 5 | Hydrate: resolve the 25 `FusedResult.chunk_id`s to full `Chunk` bodies for the reranker | 12 ms | 138 ms |
| 6 | `rerank.cross_encoder:rerank`, one batched call over 25 pairs | 620 ms | 758 ms |
| 7 | `agent.evaluator:assess_sufficiency` → sufficient, no refinement pass | included in step 6's return | 758 ms |
| 8 | formatter → `list[RetrievalResult]`, top 10 | 10 ms | **768 ms** |

**Budget:** `NFR-03`, ≤ 900 ms p50. 768 ms measured leaves **132 ms of headroom**. The reranker (step
6) is overwhelmingly the dominant cost at 620 of 768 ms — this is why
[TechSpecifications.md](TechSpecifications.md) treats reranker throughput as the single highest-value
latency optimisation target, and why `AXIOM_RERANKER_ENABLED=false` (skip Stage 4 entirely) is the
fastest available ablation for isolating a regression elsewhere in the pipeline.

**What flows on each arrow:** query text → `agent.classifier` → `QueryType` → `agent.planner` →
`QueryPlan` → three parallel `list[ScoredChunk]` → `retrieval.fusion` → `list[FusedResult]` (25) →
hydrated `Chunk` bodies attached → `rerank.cross_encoder` → `list[FusedResult]` with `rerank_score`
set (10) → `agent.evaluator` (sufficient) → formatter → `list[RetrievalResult]` (10).

**Manual test coverage:** [TestPlan.md M-01](TestPlan.md#7-manual-test-script-demo-day) is this exact
flow for a `Q1` query; `TC-065` asserts the well-formedness of the resulting `RetrievalResult` list.

---

## Flow 3 — Query triggering agent refinement

**Scenario:** A vague single-word query, e.g. `axiom query "auth"`, whose first pass fails the
sufficiency predicate (top-1 `rerank_score < 0.35`, per `_CONTRACT.md §5`), triggering one
refinement pass.

```
 user   [ Stage 1–4 as Flow 2, pass 1 ]   agent.evaluator   agent.planner   [ Stage 2–4 again, pass 2 ]   agent.evaluator   formatter
   │              │                             │                │                    │                        │              │
   │──query──────>│                             │                │                    │                        │              │
   │        (pass 1: classify/plan/fan-out/fuse/rerank)           │                    │                        │              │
   │              │────────────────────────────>│                │                    │                        │              │
   │              │                     top1=0.22 < 0.35          │                    │                        │              │
   │              │                             │───insufficient─>│                    │                        │              │
   │              │                             │                │──refine(plan,best)─>│                        │              │
   │              │                             │                │<──revised QueryPlan─│                        │              │
   │              │                             │                │  (expansion_terms widened, sub_queries added) │              │
   │              │                             │                │────────────────────>│                        │              │
   │              │                             │                │        (pass 2: fan-out/fuse/rerank on revised plan)         │
   │              │                             │                │                    │───────────────────────>│              │
   │              │                             │                │                    │                top1=0.61 >= 0.35        │
   │              │                             │                │                    │                         │──sufficient─>│
   │              │                             │                │                    │                         │              │──format─>│
   │<─────────────────────────────────────────────────────────────────────────────────────────────────────────────list[RetrievalResult]────│
```

| Step | Module : function | Latency | Cumulative |
|---|---|---|---|
| 1 | Pass 1: classify → plan → fan-out → fuse → rerank (identical to Flow 2, steps 1–6) | 758 ms | 758 ms |
| 2 | `agent.evaluator:assess_sufficiency` → insufficient (top-1 `rerank_score` 0.22 < 0.35) | ~1 ms | 759 ms |
| 3 | `agent.planner:refine` — widens `expansion_terms`, may add `sub_queries` (`FR-03`) | ~5 ms | 764 ms |
| 4 | Pass 2: fan-out → fuse → rerank on the revised `QueryPlan` (query classification is *not* re-run; `query_type` is carried over) | ~700 ms (no re-classification step) | 1,464 ms |
| 5 | `agent.evaluator:assess_sufficiency` → sufficient (top-1 now 0.61) | ~1 ms | 1,465 ms |
| 6 | formatter → `list[RetrievalResult]` | 10 ms | **~1.48 s** |

**Budget:** `NFR-04`, ≤ 5 s p95 with up to 2 passes. A two-pass query at ~1.5 s sits comfortably
inside budget; the 5 s ceiling is sized for the worst case (large candidate hydration, cold model
cache, slower hardware than the reference box), not the typical two-pass cost shown here.

**What differs from Flow 2:** the loop carries the `QueryPlan.original_query` unchanged across
passes ([Schema.md §10](Schema.md#10-queryplan): "identical across all passes of one request") while
`sub_queries`, `extracted_identifiers`, and `expansion_terms` are revised by `agent.planner:refine`.
`best` tracks the higher-scoring of the two passes' results (§5.2 of
[Design.md](Design.md#52-how-the-agent-loop-composes-with-the-fan-out)) — pass 2 is not assumed
better by construction, only preferred if it measurably is.

**Degradation rungs exercised on the unhappy path:** if pass 2's rewrite is byte-identical to pass
1's (the planner has nothing new to try), the loop stops early with `stop_reason="no_new_query"`
rather than paying for an identical retrieval twice — `TC-089`. If the 5 s deadline is crossed mid-pass
2, `agent.loop:run` returns pass 1's `best` rather than waiting for an incomplete pass 2 —
`TC-088`.

---

## Flow 4 — Degraded query, with the LLM disabled

**Scenario:** `AXIOM_LLM_ENABLED=false axiom query "How is the input preprocessed?"` — the
configuration a network-restricted judge's machine may be running under, per
[PRD.md US-12](PRD.md#us-12--nfr-07). Exercises the heuristic path of `NFR-07`'s degradation
requirement, not a runtime failure — this is a deliberately selected mode, not an error.

```
 user    agent.classifier          agent.planner            [ Stage 2–6 identical to Flow 2 ]
   │            │                        │                              │
   │──query────>│                        │                              │
   │      AXIOM_LLM_ENABLED=false        │                              │
   │      -> heuristic rule engine       │                              │
   │      (identifier/keyword regexes)   │                              │
   │            │───QueryType (heuristic)│                              │
   │            │                        │──build_plan (no LLM call)───>│
   │            │                        │  sub_queries = [original]    │
   │            │                        │  (FR-03 decomposition skipped;
   │            │                        │   LLM-only capability)       │
   │<───────────────────────────────────────────────list[RetrievalResult]│
```

| Step | Module : function | Behaviour under `AXIOM_LLM_ENABLED=false` |
|---|---|---|
| 1 | `agent.classifier:classify` | Heuristic rule engine (identifier/keyword regexes) instead of LLM classification. This is rung 2 of the classifier's declared ladder in [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade), selected deliberately rather than reached by failure |
| 2 | `agent.planner:build_plan` | `sub_queries` defaults to `[original_query]` — decomposition (`FR-03`) is an LLM-only capability with no heuristic equivalent, so it is simply not attempted, not degraded-and-retried |
| 3 | `agent.evaluator:assess_sufficiency` | Unaffected — sufficiency is judged from rerank scores, never from the LLM (`NG-23`), so this stage behaves identically whether the LLM is on or off |
| 4 | Stages 2–4, 6 | Identical to Flow 2 — the LLM never participates in retrieval, fusion, or reranking regardless of `AXIOM_LLM_ENABLED` |

**Budget:** Typically *faster* than Flow 2, not slower — no LLM inference call is on the critical
path. [Setup.md §8 Rung 5](Setup.md#rung-5--smoke-search-returning-ranked-results-2-s) shows a
measured 214 ms with both the LLM and the reranker disabled, versus 712 ms with both enabled.

**Test coverage:** `TC-067` — asserts `FakeLLM.call_count == 0` for the full pipeline under this
flag, and that all three archetype queries still return well-formed, non-empty results.

---

## Flow 5 — Incremental reindex

**Scenario:** `axiom reindex --to <new-rev>` after a commit touching 50 files in a previously-indexed
repository. Exercises `FR-18`/`FR-19` and the `NFR-02` budget.

```
 operator   cli.py       versioning.       versioning.       chunking.        indexing.*        indexing.
            reindex cmd  gitdiff           incremental       ast_chunker      (dense/sparse/     manifest
                                                                               structural)
    │           │             │                 │                 │                 │                │
    │──reindex─>│             │                 │                 │                 │                │
    │           │──diff_versions(old,new)──────>│                 │                 │                │
    │           │<──{A: [..], M: [..], D: [..], R: [..]}──────────│                 │                │
    │           │             │──resolve(A ∪ M)────────────────────>│                 │                │
    │           │             │  (chunk only touched files)         │                 │                │
    │           │             │<──list[Chunk] for touched files────│                 │                │
    │           │             │──for each Chunk: content_hash lookup in .axiom/blobs/──────────────────>│
    │           │             │   (hit -> reuse vector; miss -> embed)                                  │
    │           │             │──drop D-file chunks from all three indexes───────────>│                │
    │           │             │──update structural edges for touched files only──────>│                │
    │           │             │─────────────────────────────────────────────────write(manifest, parent_version=old)──>│
    │<──done────│             │                 │                 │                 │                │
```

| Step | Module : function | What it produces | Notes |
|---|---|---|---|
| 1 | `versioning.gitdiff:diff_versions` | `{A, M, D, R}` file sets | `git diff --name-status <old>..<new>`, parsed per `TC-073` including similarity-scored rename lines (`R087 old new`) |
| 2 | `chunking.ast_chunker:chunk_file` | `list[Chunk]` for A ∪ M files only | Unchanged files never re-enter the chunker — this is the entire mechanism behind the ≤45 s budget |
| 3 | Blob-store lookup (`.axiom/blobs/<content_hash>.npy`) | reused vector or a new embed call | A pure rename (`R`, no content change) produces new `chunk_id`s (path changed) but hits the existing blob by `content_hash` — zero embedding calls, `TC-075` |
| 4 | `indexing.*:build_index` (incremental mode) | updated `dense.faiss`, `sparse.bm25s/`, `structural.sqlite` | D-file chunks are dropped from all three indexes; structural graph edges are updated only for touched files |
| 5 | `indexing.manifest:write` | new `manifest.json` with `parent_version` set | Forms the version chain `versioning.evolutionary` later walks (Flow 7) |

**Budget:** `NFR-02`, ≤ 45 s for a 50-changed-file diff, versus the ~12-minute full-rebuild budget
(`NFR-01`) for the same corpus size — the entire point of `FR-18`/`FR-19`.

**Degradation rungs exercised on the unhappy path:** if `git diff` fails (not a git repo, or a
corrupted `.git`), `versioning.incremental:reindex` falls back to whole-file-hash comparison against
the parent `VersionManifest.file_hashes`, and if that is also unavailable, falls back to a full
reindex (Flow 1) rather than erroring — [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade).
A version with zero actual changes is a valid no-op outcome (`TC-077`), not an edge case treated as
an error.

**Manual test coverage:** [TestPlan.md M-08](TestPlan.md#7-manual-test-script-demo-day).

---

## Flow 6 — Version-scoped query

**Scenario:** `axiom query "deeplink" --version v1` against a repository with multiple indexed
versions. Exercises `FR-20` and [PRD.md US-7](PRD.md#us-7--fr-16-fr-20).

```
 user      cli.py        indexing.       [ Stage 1–6, scoped to v1's index files ]
           query cmd     manifest
                          (registry)
   │           │              │                        │
   │──query───>│              │                        │
   │      --version v1        │                        │
   │           │──resolve(v1)>│                        │
   │           │<─manifest path for v1─                │
   │           │  (or the active version from
   │           │   registry.json if --version omitted)  │
   │           │──────────────────────load .axiom/index/v1/*──────────>│
   │           │                        (Flow 2's Stage 1-6, but every
   │           │                         retrieval.* call is scoped to
   │           │                         v1's dense.faiss / sparse.bm25s /
   │           │                         structural.sqlite)
   │<───────────────────────────────────────list[RetrievalResult], every
   │                                         chunk.metadata.version_id == "v1"
```

| Step | Module : function | Notes |
|---|---|---|
| 1 | `cli.py:query --version v1` → registry resolution | Looks up `v1` in `registry.json`; absent the flag, resolves the registry's active version instead |
| 2 | `indexing.manifest:load` | Loads `v1`'s `VersionManifest`; asserts `embedding_model`/`embedding_dim` match the runtime embedder before any FAISS call (`TC-030`) |
| 3 | Stages 1–6 (identical to Flow 2) | Every `retrieval.*:search` call opens `v1`'s index files specifically — `dense.faiss`, `sparse.bm25s/`, `structural.sqlite` under `.axiom/index/v1/`, never a mix of versions |

**Budget:** Identical to Flow 2 — version scoping changes *which* index files are opened, not the
pipeline shape or latency profile. Switching the active version pointer itself
(`axiom version use v1`) is O(1), `TC-079` — it never triggers a reindex.

**What must never happen (Rules.md AP-13):** the version filter operates on
`ChunkMetadata.version_id`, a declared schema field, never on string surgery over `chunk_id` — see
[Rules.md AP-13](Rules.md#ap-13--filtering-versions-by-string-surgery-instead-of-metadata) for the
rejected pattern this flow deliberately avoids.

**Manual test coverage:** [TestPlan.md M-09](TestPlan.md#7-manual-test-script-demo-day).

---

## Flow 7 — Evolutionary / all-versions query

**Scenario:** `axiom query "handleDeeplink" --all-versions` against a repository with 3+ indexed
versions. Exercises `FR-21`, [PRD.md US-9](PRD.md#us-9--fr-21), and
[TestPlan.md TC-081](TestPlan.md#310-category-j--evolutionary-retrieval-tc-081--tc-085)–`TC-085`.

```
 user   cli.py        [ Stage 1-4, run once per     versioning.       versioning.       formatter
        query cmd      indexed version, concurrent ] evolutionary      evolutionary
                                                       :build_families  :ranking_bonus
   │        │                    │                          │                 │             │
   │──query─>│                   │                           │                 │             │
   │    --all-versions            │                           │                 │             │
   │        │──fan-out Stage 1-4 per version (v1, v2, v3, concurrent)─────────>│             │
   │        │<──list[FusedResult] per version, unioned──────────────────────────             │
   │        │                    │──group by (symbol, file_path), cosine >= 0.95────────────>│             │
   │        │                    │<──list[SnippetFamily]──────────────────────────────────────             │
   │        │                    │                          │──apply bonus (multi-version)──>│             │
   │        │                    │                          │  final = base*(1+0.10*stability)│             │
   │        │                    │                          │<─────re-sorted, deduped─────────│             │
   │        │                    │                          │                 │             │──format────>│
   │<────────────────────────────────────────────────────────────────────list[RetrievalResult],────────────│
   │                                                          one row per SnippetFamily, expandable to members
```

| Step | Module : function | Notes |
|---|---|---|
| 1 | Stages 1–4, run once per indexed version, concurrently across versions (an extension of the same fan-out pattern in [Design.md §5.1](Design.md#51-why-the-three-signals-run-concurrently-not-in-a-process-pool)) | Each version's fusion runs against that version's own index files, as in Flow 6 |
| 2 | `versioning.evolutionary:build_families` | Groups chunks with cosine ≥ 0.95 sharing `(symbol, file_path)` — never by similarity alone, and never across different `(symbol, file_path)` pairs (`TC-082`) |
| 3 | `versioning.evolutionary:ranking_bonus` | `final = base * (1 + 0.10 * stability)` applied only to families spanning ≥ 2 versions (`SnippetFamily.is_multi_version`); a single-version family is neutral, not penalised (`TC-084`) |
| 4 | formatter | One `RetrievalResult` row per `SnippetFamily`, `representative` = newest member, expandable to `members` with per-transition `diffs` |

**What flows on each arrow:** the union of per-version `FusedResult`s → grouped into
`list[SnippetFamily]` (each carrying `family_id`, `representative`, `members`, `versions`,
`stability`, `diffs`) → stability-bonus-adjusted `score` → `RetrievalResult`s whose `chunk` is each
family's `representative`.

**Degradation rungs exercised on the unhappy path:** if version metadata is missing for a chunk
(malformed manifest), `build_families` falls back to identity families — one member each, per
[Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s evolutionary-dedupe row — rather
than guessing at grouping without reliable version data.

**Manual test coverage:** [TestPlan.md M-10](TestPlan.md#7-manual-test-script-demo-day).

---

## Flow 8 — Reranker failure cascade

**Scenario:** Mid-demo-session, the cross-encoder ONNX session fails to (re)load — a corrupted model
cache entry, an out-of-memory condition, or a deliberately induced failure for the degradation
rehearsal in [Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before).
This is the flow that makes [Design.md §6](Design.md#6-the-degradation-ladder-as-a-structural-concept)'s
abstract description of a Stage-4 degradation concrete, end to end, on one example.

```
 user   [ Stage 1-3 identical to Flow 2 ]    rerank.cross_encoder     agent.evaluator      formatter
   │              │                                  │                       │                │
   │──query──────>│                                  │                       │                │
   │        (classify/plan/fan-out/fuse -> 25 FusedResult, RRF order, rrf_score set)            │
   │              │─────────────────────────────────>│                       │                │
   │              │                            _ensure_session() raises      │                │
   │              │                            PrismModelError                │                │
   │              │                            (ONNX session construction    │                │
   │              │                             fails: corrupt cache /       │                │
   │              │                             OOM / missing artefact)      │                │
   │              │                            WARNING logged: stage=rerank, │                │
   │              │                            reason=PrismModelError         │                │
   │              │                            degradation counter += 1      │                │
   │              │                            -> pass RRF order through    │                │
   │              │                               unchanged, rerank_score=None│                │
   │              │<─────────────────────────────────│                       │                │
   │              │──────────────────────────────────────────────>assess_sufficiency           │
   │              │                                                   (RRF-score predicate,     │
   │              │                                                    since rerank scores       │
   │              │                                                    are absent — Rules.md §3   │
   │              │                                                    "Sufficiency check" row)   │
   │              │<──────────────────────────────────────────────────sufficient (never loops     │
   │              │                                                    blindly on a passthrough)  │
   │              │                                                                       │────format──>│
   │<─────────────────────────────────────────────────────────list[RetrievalResult], every ─────────────│
   │                                                             match_reason="rerank_passthrough",
   │                                                             every rerank_score is None,
   │                                                             score == rrf_score
```

| Step | Module : function | Behaviour |
|---|---|---|
| 1 | Stages 1–3 (classify, plan, fan-out, fuse) | Identical to Flow 2 — the failure has not happened yet; `list[FusedResult]` with `rrf_score` set, `rerank_score=None` (unset), top N=25 |
| 2 | `rerank.cross_encoder:rerank` | `_ensure_session()` raises `PrismModelError` on ONNX session construction. Caught inside the reranker stage, per [Rules.md AP-04](Rules.md#ap-04--silent-exception-swallowing-rule-3)'s pattern: `WARNING` logged naming `stage="rerank"` and the exception type, the run's degradation counter incremented, never silently swallowed |
| 3 | Degradation rung | RRF order passed through unchanged. Every `FusedResult.rerank_score` stays `None`; ordering is by `rrf_score` via `final_score` ([Schema.md §8](Schema.md#8-fusedresult): "`final_score` is `rrf_score` when `rerank_score` is `None`") |
| 4 | `agent.evaluator:assess_sufficiency` | Falls back to the **RRF-score predicate** rather than the rerank-score predicate — [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s "Sufficiency check" row names exactly this rung, and specifies it must "declare sufficient" rather than loop blindly, since an RRF-based predicate has no principled way to trigger a rewrite that a rerank-based one would have |
| 5 | formatter | Every `RetrievalResult.match_reason` states `"rerank_passthrough"` so the degradation is visible to the caller, not hidden behind a plausible-looking score — [Design.md §6.2](Design.md#62-why-degradation-composes-differently-depending-on-where-it-happens) |

**Budget impact:** *faster* than Flow 2, not slower — Stage 4's 620 ms (the dominant cost in the
happy-path breakdown) is skipped entirely. A degraded query is cheap; it is simply less accurate,
which is the honest trade [NonGoals.md NG-10](NonGoals.md#ng-10--no-production-sla-uptime-or-ha-guarantee)
already states for the reranker-disabled ablation path.

**Test coverage:** `TC-062` — asserts non-empty results in pure RRF order, every `rerank_score is
None`, `match_reason` notes the degradation, and exactly one `ERROR`-adjacent log entry (at
`WARNING`, per [Rules.md §9.1](Rules.md#91-logging-discipline) item 3 — a *degradation* is never
logged at `ERROR`), not one per candidate pair.

**Why this is the flow that makes §6 concrete:** [Design.md §6.2](Design.md#62-why-degradation-composes-differently-depending-on-where-it-happens)
states in the abstract that "a Stage 4 degradation changes the *meaning* of the final score, and that
change must be visible to the caller." This flow is the literal trace of that sentence: `score`
silently becomes `rrf_score` instead of a calibrated `[0,1]` rerank probability, and the *only* thing
that keeps that change from being silent is `match_reason` and the degradation counter — both of
which this flow shows firing at the exact moment the type-preserving-but-less-informative substitution
happens.

---

## Related documents

| Document | Relationship |
|---|---|
| [Design.md](Design.md) | The architecture these eight flows are concrete instances of |
| [Schema.md](Schema.md) | Normative definition of every object named on the arrows above |
| [TestPlan.md](TestPlan.md) | `TC-###` cases and `M-##` manual steps each flow's assertions map to |
| [Rules.md](Rules.md) | The degradation-ladder table flows 4, 5, 8 trace concretely |
| [PRD.md](PRD.md) | The user stories (`US-1`–`US-12`) each flow demonstrates the acceptance criteria for |
