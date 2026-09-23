# Design

Architecture, design principles, the concurrency model, the degradation ladder as a structural concept, and the extension points Axiom was built to accept.

**Owner:** Anish Grover
**Last updated:** 2026-09-23
**Status:** Draft

Related: [PRD.md](PRD.md) · [TechSpecifications.md](TechSpecifications.md) · [Schema.md](Schema.md) · [Appflow.md](Appflow.md) · [Rules.md](Rules.md) · [Decisions.md](Decisions.md) · [NonGoals.md](NonGoals.md) · [TestPlan.md](TestPlan.md)

---

## 1. Scope and audience

This document explains **why the system is shaped the way it is** — the pipeline topology, the
concurrency model, and the failure-handling philosophy — as distinct from
[TechSpecifications.md](TechSpecifications.md), which pins the exact constants, algorithms, and
per-module behavioural contracts, and from [Appflow.md](Appflow.md), which walks the same
architecture through eight concrete runtime scenarios with real latency numbers. Read this document
to understand the shape; read the other two for the numbers and the walkthroughs.

Every noun used below (`Chunk`, `ScoredChunk`, `FusedResult`, `QueryPlan`, `RetrievalResult`) is
defined once, normatively, in [Schema.md](Schema.md). This document does not redefine them.

---

## 2. Design principles

Four principles, each traceable to a binding rule in [Rules.md](Rules.md) or a locked choice in
`_CONTRACT.md`, that explain most of the architectural decisions below without re-deriving them
each time.

| Principle | Binding source | What it rules out |
|---|---|---|
| **Retrieve, do not generate.** The system never emits a token of code it did not read from disk. | [NonGoals.md NG-01](NonGoals.md#ng-01--no-code-generation)–`NG-03` | LLM-authored snippets, prose summaries, autofix diffs |
| **Rank in rank space, never in an invented shared score space.** Three signals with three incomparable score domains fuse by position, not by magnitude. | [Rules.md Rule 4](Rules.md#rule-4--higher-is-better-lists-are-sorted-descending); [`ADR-002`](Decisions.md#adr-002--weighted-reciprocal-rank-fusion-over-score-space-fusion) | Min-max normalisation, learned score blending, any cross-signal score comparison |
| **Every stage is pure and every failure degrades.** Same input plus same config yields the same output, forever; a stage that cannot complete returns a typed empty result instead of dying mid-run. | [Rules.md Rule 2](Rules.md#rule-2--stages-are-pure), [Rule 3](Rules.md#rule-3--never-raise-on-bad-input-degrade) | Hidden mutable state, unbounded retries, an eval run that can die at query 3,200 of 3,765 |
| **Corpus size must never reach the LLM.** The query LLM classifies, expands, decomposes, and judges sufficiency from scores — never from code. | [NonGoals.md NG-23](NonGoals.md#ng-23--no-llm-ingestion-of-retrieved-code); [`ADR-008`](Decisions.md#adr-008--the-query-llm-never-reads-code) | Any code path where a bigger repo makes the agent loop slower or less deterministic |

These four principles are why the architecture below has the shape it has: a wide, parallel
first-stage fan-out (principle 2 demands independent signals that only compare by rank); a narrow,
bounded refinement loop (principle 3 and 4 together); and a formatter that is the *only* place a
`Chunk`'s text is allowed to reach a human (principle 1).

---

## 3. Architecture overview

### 3.1 The pipeline

Every query, from CLI, API, or UI, passes through the same six stages. The diagram names the actual
module and entrypoint each stage runs, per the package layout locked in `_CONTRACT.md §3` — not the
informal sketch in the original proposal.

```
                                   raw query: str
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
    Stage 1          │  agent.classifier:classify            │
    QUERY             │  agent.planner:build_plan             │
    UNDERSTANDING     │  -> QueryPlan (query_type,            │
                      │     strategy_weights, sub_queries,     │
                      │     extracted_identifiers,             │
                      │     expansion_terms)                   │
                      └──────────────────┬─────────────────────┘
                                         │  QueryPlan
                     ┌───────────────────┼───────────────────┐
                     ▼                   ▼                   ▼
    Stage 2   retrieval.dense:search   retrieval.sparse:search  retrieval.structural:search
    FIRST-STAGE  (FAISS, K=100)         (bm25s, K=100)           (structural.sqlite, K=50)
    RETRIEVAL    -> list[ScoredChunk]   -> list[ScoredChunk]     -> list[ScoredChunk]
    (concurrent) DENSE                  SPARSE                   STRUCTURAL
                     │                   │                       │
                     └───────────────────┼───────────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
    Stage 3           │  retrieval.fusion:                    │
    FUSION            │  reciprocal_rank_fusion(weights, k=60) │
                      │  -> list[FusedResult], top N=25         │
                      └──────────────────┬─────────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
    Stage 4           │  rerank.cross_encoder:rerank           │
    RERANK            │  -> list[FusedResult] with              │
                      │     rerank_score set, top 10            │
                      └──────────────────┬─────────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
    Stage 5           │  agent.evaluator:assess_sufficiency    │
    SUFFICIENCY       │  -> sufficient | insufficient           │
    CHECK             └────────┬─────────────────┬─────────────┘
                          insufficient        sufficient
                                │                  │
                     ┌──────────▼──────────┐        │
                     │ agent.planner:refine │        │
                     │ -> new QueryPlan     │        │
                     │ loop to Stage 2      │        │
                     │ (max 2 total passes) │        │
                     └──────────────────────┘        │
                                                      ▼
                                    ┌──────────────────────────────────────┐
                     Stage 6         │  formatter (inside agent.loop:run)    │
                     FORMAT          │  -> list[RetrievalResult]              │
                                    └──────────────────────────────────────┘
```

Stages 1–4 are a single pass. Stage 5 decides whether to loop back to Stage 2 with a revised
`QueryPlan` or to proceed to Stage 6. The whole Stage 2–5 cycle is `agent.loop:run`'s body, bounded
by `AXIOM_AGENT_MAX_PASSES` and `AXIOM_AGENT_WALL_CLOCK_MS` — see
[§5](#5-concurrency-and-the-agent-loop). The concrete latency contribution of each stage on the
happy path is worked out in [Appflow.md flow 2](Appflow.md#flow-2--standard-query-no-agent-refinement).

### 3.2 What is offline vs. online

The diagram in §3.1 is the **online** (query-time) path. It depends entirely on artefacts built by a
separate **offline** (index-time) path that never runs inside a query:

```
   repo on disk
        │
        ▼
  chunking.ast_chunker:chunk_file  (tree-sitter, per-file)
        │  -> list[Chunk]
        ▼
  ┌─────────────┬──────────────────┬────────────────────┐
  ▼             ▼                  ▼
indexing.dense  indexing.sparse    indexing.structural
:build_index    :build_index       :build_index
-> dense.faiss  -> sparse.bm25s/   -> structural.sqlite
  dense.idmap.json
        │             │                  │
        └─────────────┴──────────────────┘
                       ▼
          indexing.manifest:write
          -> manifest.json, registry.json
```

This offline/online split is why [Rules.md AP-12](Rules.md#ap-12--re-chunking-or-re-embedding-on-the-read-path-rule-2-96)
is a merge blocker: a query handler that re-parses or re-embeds the repo has collapsed the two paths
into one, which both explodes p50 latency and risks producing `chunk_id`s that do not match the
indexed ones. The online path in §3.1 only ever *reads* artefacts the offline path already wrote.

---

## 4. The Three-Signal Rationale

Dense, sparse, and structural retrieval answer three different questions about the same corpus, and
each one's score is meaningful only against other scores from the *same* signal.

| Signal | Question it answers | Score domain | Why it cannot be compared to the others |
|---|---|---|---|
| **Dense** (`retrieval.dense:search`) | "What code *behaves like* this description?" | Cosine similarity in `[-1, 1]`, practically `[0, 1]` for L2-normalised Qwen3 embeddings | Bounded and smooth — a 0.02 gap between two candidates is a small, real difference in a tight distribution near the query's nearest neighbours. |
| **Sparse** (`retrieval.sparse:search`) | "What code *contains these exact tokens*?" | Unbounded positive BM25 score, driven by term rarity (IDF) | A single rare-identifier hit can score an order of magnitude higher than five common-token matches. The scale is corpus-vocabulary-dependent, not fixed. |
| **Structural** (`retrieval.structural:search`) | "What code has *this graph relationship* to a named symbol?" | Integer/graph-derived proximity, unbounded | Not a relevance estimate at all in the statistical sense — it is closer to a boolean "reachable within N hops" turned into a rank. |

Three domains this different cannot be added, averaged, or even min-max normalised into one number
without smuggling in an implicit, unjustified exchange rate between "very lexically exact" and
"very semantically close" and "two call-graph hops away." [Rules.md AP-02](Rules.md#ap-02--fusing-in-score-space-instead-of-rank-space-rule-4-contract-5)
shows the concrete failure mode of trying anyway. The only operation that is well-defined across all
three is **rank**: "this chunk was signal X's Nth choice" is comparable across signals because every
signal produces a total order over its own candidates, and Reciprocal Rank Fusion consumes exactly
that — nothing else. The full arithmetic, the alternatives considered, and why RRF specifically (and
not CombSUM, CombMNZ, or a learned blend) is the fusion rule is recorded once, in
[`ADR-002`](Decisions.md#adr-002--weighted-reciprocal-rank-fusion-over-score-space-fusion); this
section exists so [Schema.md §3.2](Schema.md#32-signalkind-semantics) has a narrative home to point
at when it says the three score domains are "mutually incomparable... exactly why fusion is
rank-based."

One consequence worth stating plainly: because fusion only sees rank, a signal's *absolute*
usefulness for a query type is expressed entirely through its **weight** in `QueryPlan.strategy_weights`,
never through its raw score. A `STRUCTURAL` query weights the structural signal at 0.6 not because
its scores are "bigger" in some sense, but because a chunk it ranks #1 should out-influence the
fused result more than a chunk dense or sparse ranks #1 — that is a modelling choice made once, at
the weight vector, not re-derived per query from score magnitudes.

---

## 5. Concurrency and the agent loop

### 5.1 Why the three signals run concurrently, not in a process pool

Stage 2 issues three independent lookups — FAISS, `bm25s`, and a SQLite query — against the same
`QueryPlan`. None of the three depends on another's output, so they run **concurrently on a bounded
thread pool**, not sequentially and not in separate processes.

Thread pool, not multiprocessing, for a specific reason tied to [Rules.md AP-06](Rules.md#ap-06--loading-models-at-import-time):
the dense retriever holds a live ONNX Runtime session (for the query embedding) and the reranker
holds another, both loaded lazily and kept warm for the process's lifetime because construction costs
hundreds of milliseconds and hundreds of megabytes of RSS. Forking a process per query — or even a
worker pool of processes — means every worker pays that load cost independently, which is exactly
the pathology [Rules.md AP-06](Rules.md#ap-06--loading-models-at-import-time) exists to prevent.
None of the three per-signal calls is CPU-bound in a way that needs true parallelism across cores
either: FAISS's C++ search kernel and SQLite's C implementation both release the GIL for the
duration of the call, and `bm25s`'s numpy-vectorised scoring is fast enough in absolute terms
(single-digit milliseconds at corpus scale) that the concurrency here is about **hiding I/O and
kernel-call latency**, not about spreading CPU-bound work across cores. A bounded `ThreadPoolExecutor`
sized to 3 (one per signal) inside `agent.loop:run` is therefore the correct shape: one process, one
set of warm model sessions, three overlapping calls.

### 5.2 How the agent loop composes with the fan-out

The bounded refinement loop from [`ADR-007`](Decisions.md#adr-007--bounded-agent-loop-hard-caps-not-convergence)
is not a separate concurrency mechanism layered on top of §5.1 — it is a **sequential outer loop
around the same concurrent fan-out**. Each pass through Stage 2–5 in §3.1 is the identical
three-way concurrent lookup, re-run with a revised `QueryPlan`. Concretely:

```
deadline = Deadline(settings.agent_wall_clock_ms)                # 5000 ms, monotonic
max_passes = max(1, agent_max_passes if agent_enabled else 1)    # 2; the eval profile forces 1
best = None
for pass_no in range(1, max_passes + 1):                         # 2 TOTAL cycles, not 2 + 1
    if pass_no > 1 and deadline.expired:                         # pass 1 is EXEMPT
        break
    scored_lists = concurrent_fan_out(plan)      # Stage 2, thread pool, per effective_query
    fused = reciprocal_rank_fusion(scored_lists, plan.strategy_weights)  # Stage 3
    if pass_no > 1 and deadline.expired:         # abandon BEFORE the reranker, ~80% of the cost
        break
    chunks = hydrate(fused)
    reranked = cross_encoder.rerank(plan.original_query, fused, chunks)  # Stage 4
    if best is None or quality(reranked) > quality(best):
        best = reranked                          # quality = (cross_encoder_ran, top1)
    if evaluator.is_sufficient(reranked):         # Stage 5
        break
    plan = planner.next_plan(plan, reranked)      # only reached if insufficient
    if plan is unchanged:                         # nothing new to try -> stop
        break
return formatter(best)                             # Stage 6
```

Three details of that block are load-bearing and were wrong in the previous revision:

1. **`agent_max_passes = 2` means two total cycles**, one initial retrieval and at most one
   refinement. It is not two refinements on top of an initial pass.
   [Rules.md AP-03](Rules.md#ap-03--unbounded-agent-loop-rule-3-contract-5)'s reference block seeds
   `best` from a retrieval taken *outside* the loop and so implies three; on this point Rules.md is
   the document that must change, not this one. Since rerank dominates a pass's cost, the difference
   is a ~50% swing in p95.
2. **Pass 1 is exempt from the deadline.** Checking the budget at the top of *every* pass, as the
   previous block did, means a query issued with an already-exhausted budget returns nothing at all.
   A query must return something; a zero-length budget degrades to one pass, not to an empty answer.
3. **Passes are compared on `(cross_encoder_ran, top1)`, not on `top1` alone.** A passthrough pass's
   RRF score (~0.016) and a cross-encoder pass's score (~0.71) are different quantities, so a
   raw-magnitude comparison would discard a better-but-uncalibrated pass every time.

This is why the concurrency model and the agent loop are one design decision, not two: a widened
`agent_max_passes` multiplies the cost of the *entire* concurrent fan-out plus rerank, not just one
stage, which is precisely why [Rules.md §9.6](Rules.md#96-performance-regression-policy) item 4
requires a re-measured p95 and a new ADR before that cap ever moves. Every pass pays the full Stage
2–4 cost; there is no cheaper "just retry the weak signal" path, because a weak result is a property
of the *fused, reranked* list, not attributable to one signal in isolation.

### 5.3 What is never concurrent

The reranker's cross-encoder batch (Stage 4) scores all 25 fused candidates in one model call, not
25 concurrent single-pair calls — batching inside one ONNX Runtime session is both faster (one
forward pass, not 25) and what makes [TestPlan.md TC-061](TestPlan.md) ("batching does not change
results") a meaningful invariant to test. Chunking and indexing (§3.2, offline path) use a bounded
worker pool internally for embedding batches, but that is entirely separate machinery from the
query-time thread pool described above and is out of scope for online latency budgets — see
[TechSpecifications.md](TechSpecifications.md) for the indexing-time concurrency shape.

---

## 6. The degradation ladder as a structural concept

[Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade) is the authoritative, per-stage
table of every fallback rung in the system — this section does not repeat it. What belongs here is
the *shape* of the pattern and why it is a load-bearing piece of the architecture, not a defensive
afterthought bolted onto each module independently.

### 6.1 The shape

Every stage in §3.1 and §3.2 that can fail declares an ordered list of fallback rungs. Each rung is
strictly cheaper and strictly less capable than the one above it, ending in a **typed, empty, but
valid** result — never an exception. Descending a rung is always:

1. Attempted at the preferred rung first.
2. On failure, logged once at `WARNING` naming the stage and the reason (never at `ERROR` — an
   `ERROR` in this codebase means a contract, config, or startup failure per
   [Rules.md §9.2](Rules.md#92-error-taxonomy), not a degraded-but-handled input).
3. Counted in the run's degradation counter, which [Tracker.md §5](Tracker.md#5-eval-metrics-log)
   reports in aggregate so "how many of 3,765 queries degraded, and on which rung" is answerable
   after any eval run, not just anecdotally observed during one.
4. Never retried at the same rung — a rung either succeeds or hands off to the next one immediately.

This is exactly the shape [Rules.md AP-11](Rules.md#ap-11--raising-on-malformed-query-input-rule-3)
shows for the sparse tokenizer (code-aware regex → whitespace split → empty list) and
[AP-04](Rules.md#ap-04--silent-exception-swallowing-rule-3) shows for structural search
(SQL lookup → identifier substring match, logged and counted, never silently swallowed).

### 6.2 Why degradation composes differently depending on where it happens

Not every stage's degradation has the same *blast radius*, and understanding that is what makes the
ladder an architectural property rather than thirteen independent try/except blocks:

- **A chunker degradation is invisible past indexing.** If `chunking.ast_chunker:chunk_file` falls
  back to the line-window splitter for one broken file (per [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s
  table), every downstream stage — dense, sparse, structural, fusion, rerank — simply sees `Chunk`
  objects with `ChunkKind.BLOCK` instead of `ChunkKind.FUNCTION`. Nothing downstream needs to know a
  degradation happened; the schema absorbs it. This is why `Chunk`'s shape is uniform regardless of
  which chunking rung produced it — see [Schema.md §6](Schema.md#6-chunk).
- **A first-stage retrieval degradation (Stage 2) shrinks the fusion input, but fusion tolerates it
  by construction.** If `retrieval.structural:search` returns `[]` because no identifier resolved,
  `retrieval.fusion:reciprocal_rank_fusion` renormalises the remaining weights over the signals that
  *did* return something — this is `TC-055` in [TestPlan.md](TestPlan.md), and it means a Stage 2
  degradation never needs special-casing at Stage 3. The degradation is absorbed one stage later by a
  property fusion already has, not by new code reacting to the specific failure.
- **A Stage 4 (rerank) degradation changes the *meaning* of the final score, and that change must be
  visible to the caller.** When the cross-encoder is unavailable, Stage 4 passes the RRF order through
  unchanged with `rerank_score=None` — this is not silently equivalent to a low-confidence rerank
  score, so [Schema.md §8](Schema.md#8-fusedresult) makes `rerank_score` an explicit `float | None`
  and `RetrievalResult.match_reason` names the degradation (`"rerank_passthrough"`) so a consumer can
  always distinguish "reranked and scored low" from "not reranked at all." [Appflow.md flow
  8](Appflow.md#flow-8--reranker-failure-cascade) walks this exact case end to end.
- **A Stage 5 (agent loop) degradation is a budget exhaustion, not a component failure**, and its
  fallback is simply "stop looping, return the best pass seen so far" — there is no lower rung below
  "the single-pass result," because a single pass is already Stages 1–4 running correctly. This is
  why the agent loop's degradation table entry in [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)
  has only one rung, unlike the multi-rung chunker or sparse-tokenizer ladders.

The general principle: **a degradation's blast radius is bounded by how far downstream its output
shape still satisfies the schema contract it was supposed to produce.** A degraded `Chunk` is still a
valid `Chunk`; a degraded `ScoredChunk` list is still a valid (possibly empty) `list[ScoredChunk]`
fusion already knows how to renormalise around; a degraded rerank result is a valid `FusedResult`
with one field explicitly nulled. Nothing in the pipeline needs a special "degraded mode" branch,
because degradation never leaves the type system — it only ever narrows within it. This is the same
principle [Rules.md Rule 1](Rules.md#rule-1--ids-are-sacred) protects from the opposite direction:
just as an id must never be transformed, a degraded value must never be a *different shape* than the
value it replaces, only a less informative one.

### 6.3 Where raising is still correct

Three categories never degrade — see [Rules.md Rule 3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s
exception list: contract violations (`AxiomContractError`), configuration errors (a Pydantic
`ValidationError` at startup), and missing index artefacts at startup (`IndexNotFoundError`). These
are not points on the ladder at all; they are pre-conditions for the ladder existing in the first
place. (`src/axiom/core/errors.py` defines exactly four classes — `AxiomError`,
`AxiomContractError`, `IndexNotFoundError`, `DegradationExhaustedError`; where `Rules.md §9.2`
enumerates a seven-class taxonomy, that document names classes the code does not define. Flagged for
the Rules.md owner.) An `AxiomContractError` mid-run means the type-preservation guarantee in §6.2 has already been
violated somewhere upstream, and continuing to degrade past that point would just be manufacturing a
plausible-looking wrong answer — which is precisely the `0.0`-NDCG@10-with-no-error failure mode
[Rules.md Rule 1](Rules.md#rule-1--ids-are-sacred) spends its longest passage warning about.

---

## 7. Two profiles, one pipeline

The `eval` and `demo` profiles ([`ADR-001`](Decisions.md#adr-001--two-first-class-evaluation-profiles),
full rationale in [PRD.md §2.2](PRD.md#22-two-evaluation-contexts-two-profiles)) are not two
different architectures — they are the *same* pipeline in §3.1 with different `QueryPlan.strategy_weights`
and, for `eval`, one stage skipped outright:

```
                              configs/eval.yaml              configs/demo.yaml
                              ─────────────────              ──────────────────
Stage 1  Query understanding  same                            same
Stage 2  Dense retrieval      runs (weight .85)                runs (per-QueryType weight)
         Sparse retrieval     runs (weight .15)                runs (per-QueryType weight)
         Structural retrieval SKIPPED — structural.sqlite       runs (per-QueryType weight)
                              is never built for this profile
Stage 3  Fusion                two-signal RRF                  three-signal RRF
Stage 4  Rerank                same                            same
Stage 5  Sufficiency/refine    same                             same
Stage 6  Format                same                             same
```

The structural skip under `eval` is not a runtime `if` branch inside `retrieval.structural:search` —
it is a decision made once, at index-build time: `indexing.structural:build_index` is never invoked
for an `eval`-profile index, so `structural.sqlite` never exists for that corpus, and Stage 2's
concurrent fan-out (§5.1) launches only two tasks instead of three. This keeps the "does the
structural signal contribute" question answerable from the *index directory contents* alone, not
from tracing a conditional through the query path — an evaluator inspecting `.axiom-apps/index/*/`
can see there is no `structural.sqlite` file and know immediately that the signal was never in play,
consistent with the `.axiom-apps/` vs. `.axiom-demo/` split named in [Setup.md §5.5](Setup.md#55-dataset--coir-appsretrieval).

---

## 8. Extension points

The module boundaries in §3.1/§3.2 were chosen so that every item in
[NonGoals.md §4, "Deferred, not rejected"](NonGoals.md#4-deferred-not-rejected) is an **additive**
change — a new module and a new weight, not a rewrite. This section makes that concrete for the
three extension shapes most likely to matter after 27 September.

### 8.1 A fourth retrieval signal (e.g. learned sparse retrieval / SPLADE)

| Change | Where |
|---|---|
| New enum variant | `SignalKind.LEARNED_SPARSE` added to [Schema.md §3](Schema.md#3-enums) |
| New retrieval module | `retrieval/learned_sparse.py`, same interface shape as `retrieval/dense.py`: `search(plan) -> list[ScoredChunk]` |
| New indexing module | `indexing/learned_sparse.py`, invoked alongside the other three in the offline path (§3.2) |
| Fusion change | One more entry in `QueryPlan.strategy_weights` per `QueryType`; `reciprocal_rank_fusion` already iterates `dict[SignalKind, float]`, so it needs no code change, only a config change |
| Schema change | **None.** `ScoredChunk`, `FusedResult`, `Chunk` are all signal-agnostic already |

What does *not* change: `Chunk`'s shape, `FusedResult`'s shape, the reranker, the agent loop, the
formatter. This is exactly why [NonGoals.md §4](NonGoals.md#4-deferred-not-rejected) can say a
SPLADE-style fourth signal is deferred on time budget alone (the index-time expansion pass is the
actual cost) rather than deferred because the architecture cannot hold it.

### 8.2 A new source-language grammar (e.g. TypeScript, JSX)

`ChunkMetadata.language` already exists as a field precisely for this
([Schema.md §5](Schema.md#5-chunkmetadata): "the field exists so adding a grammar needs no
migration"). The chunker's module boundary is a per-language grammar adapter behind
`chunking.ast_chunker:chunk_file`, dispatching on file extension to a `tree-sitter` `Language`
instance. Adding TypeScript means:

1. A new grammar dependency (`tree-sitter-typescript`) and a node-kind mapping table (which
   TS-specific AST nodes count as `FUNCTION`/`METHOD`/`CLASS`, since TS's grammar has additional node
   kinds JS's does not — interfaces, type aliases, decorators).
2. Export-resolution rules for `is_exported` that understand TS's `export type`/`export interface`
   forms in addition to the ESM/CommonJS forms already handled.
3. No change to `indexing/`, `retrieval/`, `rerank/`, `agent/`, or the schema — every downstream
   stage already treats `language` as an opaque string and chunks generically.

This is why [NonGoals.md §4](NonGoals.md#4-deferred-not-rejected) can state the TypeScript cost as
"a day per language" rather than an open-ended unknown: the day is entirely inside `chunking/`.

### 8.3 A different reranker or embedder

Both are already behind the lazy-loading adapter pattern in
[Rules.md AP-06](Rules.md#ap-06--loading-models-at-import-time) — `rerank/cross_encoder.py` and the
embedding adapter inside `retrieval/dense.py` construct their ONNX Runtime session on first use, keyed
off `Settings.reranker_model` / `Settings.embedding_model`. Swapping the primary model, or promoting
the currently-declared fallback (`ms-marco-MiniLM-L-6-v2`, `all-MiniLM-L6-v2`) to primary, is a config
change plus a re-export/re-quantise step ([Setup.md §6](Setup.md#6-onnx-export-and-int8-quantisation)),
not a code change — this is precisely the mechanism `NFR-07`'s degradation requirement already relies
on, since "swap to the fallback model" and "swap to a different primary model" are the same code path
exercised for two different reasons.

### 8.4 What is *not* a clean extension point, on purpose

Two things in [NonGoals.md §4](NonGoals.md#4-deferred-not-rejected) are deferred for reasons that are
architectural, not merely a time-budget line item, and are worth naming so a future contributor does
not assume they are as cheap as §8.1–§8.3:

- **Call-graph reachability beyond depth 1.** The current `structural.sqlite` schema
  ([`ADR-011`](Decisions.md#adr-011--sqlite-for-the-structural-index-not-a-graph-database)) answers
  depth-1 traversals (callers-of, callees-of, one ordered-pair check) with a plain join. Multi-hop
  reachability needs a recursive CTE and, more importantly, a new relevance model — "two hops away" is
  not obviously more or less relevant than "one hop away in the other direction," and that scoring
  question has no answer yet, unlike the additive signal/language cases above where the *mechanism*
  is well understood and only the *content* is new.
- **Multi-hop agent planning beyond 2 total passes.** As §5.2 shows, this is not "add a third loop
  iteration" — it multiplies the full concurrent-fan-out-plus-rerank cost, and the 5-second wall-clock
  budget in `NFR-04` cannot fund it without a faster reranker first ([NonGoals.md §4](NonGoals.md#4-deferred-not-rejected)
  names this dependency explicitly). This extension point is gated on a *different* extension (a
  cheaper Stage 4) landing first, which is why it is not listed alongside §8.1–§8.3 as freely
  additive.

---

## 9. Related documents

| Document | Relationship |
|---|---|
| [Appflow.md](Appflow.md) | Eight concrete runtime traces through the pipeline described in §3 |
| [TechSpecifications.md](TechSpecifications.md) | Exact constants, algorithms, and per-module behavioural specs this document deliberately omits |
| [Schema.md](Schema.md) | Normative definition of every model named above |
| [Rules.md](Rules.md) | The binding invariants (purity, degradation, id integrity) this design exists to satisfy |
| [Decisions.md](Decisions.md) | `ADR-###` records for the specific choices §4–§7 summarise narratively |
| [NonGoals.md](NonGoals.md) | The scope fence §8 shows this architecture was built to respect without foreclosing |
