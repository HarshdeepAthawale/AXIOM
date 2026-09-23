# Axiom — Agentic Code Intelligence

**Samsung PRISM GenAI Hackathon 3rd Edition (2026-27) | Theme 01**

> **Team:** Incognito
> **Members:** Prabinder Singh, Anish Grover, Harshdeep Athawale, Parth Deshmukh
> **Institute:** Thapar Institute of Engineering & Technology, Patiala
> **Submission Deadline:** 27 September 2026

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-23
**Status:** Active — front-door overview

This document is the entry point to the project. It describes the system **as built**.
Where a number here is measured, it says so; where it is a projection, it says that too.
Authority for any detail: [`docs/_CONTRACT.md`](docs/_CONTRACT.md) →
[`docs/PRD.md`](docs/PRD.md) / [`docs/Schema.md`](docs/Schema.md) →
[`docs/TechSpecifications.md`](docs/TechSpecifications.md).

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Our Solution (One Paragraph)](#2-our-solution-one-paragraph)
3. [Why This Is Hard](#3-why-this-is-hard)
4. [System Architecture](#4-system-architecture)
5. [Component Deep-Dive](#5-component-deep-dive)
6. [The Agentic Loop](#6-the-agentic-loop)
7. [Version-Aware Retrieval (P1)](#7-version-aware-retrieval-p1)
8. [Evolutionary Retrieval (Bonus)](#8-evolutionary-retrieval-bonus)
9. [Evaluation & Metrics](#9-evaluation--metrics)
10. [Tech Stack](#10-tech-stack)
11. [Submission Checklist](#11-submission-checklist)
12. [Team & Responsibilities](#12-team--responsibilities)
13. [Timeline (10 Days)](#13-timeline-10-days)
14. [Key Risks & Mitigations](#14-key-risks--mitigations)

---

## 1. The Problem

Voice-assistant codebases are massive — thousands of JavaScript files, dozens of agents and tools spread across deeply nested directories. When a developer asks:

- *"How is the input preprocessed before going to the main function?"*
- *"Which files call tool XYZ before tool ABC?"*
- *"Where is the Bluetooth-settings deeplink used?"*

...they need the exact code snippets and file locations. But:

- **No LLM can hold the entire repo in its context window** — the codebase is too large
- **Simple text search misses semantic meaning** — `normalize()` won't match the query "preprocessing"
- **Structural questions need code understanding** — you can't answer "which files call X before Y" with keyword search alone

### The Core Task

> **Given a library of code and a natural-language query, provide a ranking of code snippets in order of their relevance to the query.**

This is a **code retrieval** problem — not code generation. We retrieve, rank, and locate. Generating answers or explanations is explicitly out of scope.

### The Three Submission Goals

| Priority        | Goal                          | Description                                                                                                                      |
| --------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **P0** (must)   | **Retrieval Accuracy**        | Retrieve the most relevant code snippets for a given query. Evaluated on the CoIR `AppsRetrieval` dataset using NDCG@10 and MRR. |
| **P1** (should) | **Retrieval Across Versions** | Support different versions of the codebase. Indexes/caches must rebuild in reasonable time when the code changes.                |
| **Bonus**       | **Evolutionary Retrieval**    | Retrieve code snippets *across all versions* simultaneously — even when snippets across versions are very similar.               |

### Constraints

- **Must run on CPU** — minimal GPU use allowed
- **Single language:** JavaScript
- **Output:** snippets + file/line locations (not full code generation; optimization suggestions are a bonus)
- **The codebase is far larger than any LLM context window** — must work agentically

---

## 2. Our Solution (One Paragraph)

We build a **multi-pass agentic code retrieval system** that combines three retrieval signals — **dense code embeddings**, **sparse lexical search (BM25)**, and **structural AST/call-graph indexing** — fused with **Reciprocal Rank Fusion** and refined by a **cross-encoder reranker**. An **agentic orchestrator** classifies each query (semantic vs. structural vs. usage), routes it to the right retrieval strategy, evaluates whether the initial results are sufficient, and if not, rewrites the query and re-retrieves — all without a GPU. For version support, we use **git-diff-based incremental reindexing** so only changed files are re-embedded, and for evolutionary retrieval, we tag every snippet with its version and use version-aware deduplication at ranking time.

---

## 3. Why This Is Hard

| Challenge                       | Why it matters                                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Code isn't natural language** | Variable names, control flow, and semantics don't follow English grammar. General-purpose embeddings miss code structure.            |
| **Thousands of snippets**       | You can't feed them all to an LLM. First-stage retrieval must be fast and high-recall.                                               |
| **Structural queries**          | "Which files call X before Y?" requires understanding call graphs and execution order — pure embedding similarity won't answer this. |
| **Versioned code**              | Snippets across versions are nearly identical (one-line diffs), making them hard to rank/deduplicate.                                |
| **CPU-only**                    | Rules out large GPU-dependent models. Must use small, quantized, or CPU-optimized models.                                            |
| **Agentic, not single-pass**    | Simple retrieve-and-return won't work on complex queries. The system must plan, search, read context, and refine.                    |

---

## 4. System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                  │
│   "How is the input preprocessed before going to the main function?"│
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Query Classifier   │
                    │  semantic/structural  │
                    │  /usage/hybrid        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────┐   ┌────────────┐   ┌────────────────┐
     │   Dense     │   │   Sparse   │   │   Structural   │
     │  Embedding  │   │   BM25     │   │   AST / Call   │
     │  Search     │   │   Index    │   │   Graph        │
     └──────┬─────┘   └──────┬─────┘   └───────┬────────┘
            │                │                  │
            └────────────────┼──────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Reciprocal Rank     │
                  │  Fusion (RRF)        │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Cross-Encoder       │
                  │  Reranker            │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Agentic Evaluator   │◄──── "Are these results good enough?"
                  │  (Plan → Search →    │         │
                  │   Assess → Refine)   │         │  NO → rewrite query, re-retrieve
                  └──────────┬──────────┘         │
                             │ YES                 │
                             ▼                     │
                  ┌─────────────────────┐          │
                  │  RANKED RESULTS      │◄─────────┘
                  │  snippet + file +    │
                  │  line location       │
                  └─────────────────────┘
```

### Why Three Retrieval Signals?

| Signal                    | What it catches                                    | What it misses                                   |
| ------------------------- | -------------------------------------------------- | ------------------------------------------------ |
| **Dense embeddings**      | Semantic meaning ("preprocessing" ↔ `normalize()`) | Exact identifier names, structural relationships |
| **BM25 (sparse lexical)** | Exact keywords, function/variable names            | Synonyms, semantic similarity                    |
| **AST / call graph**      | "X calls Y", import chains, control flow order     | Natural-language intent                          |

**No single signal is sufficient.** Fusing all three with Reciprocal Rank Fusion consistently outperforms any individual signal on code retrieval benchmarks (CoIR, CORE-Bench, SweRank).

---

## 5. Component Deep-Dive

### 5.1 Query Understanding

Before retrieval, we classify the query to route it to the right strategy:

| Query Type     | Example                                                            | Routing                                                     |
| -------------- | ------------------------------------------------------------------ | ----------------------------------------------------------- |
| **Semantic**   | "How is input preprocessed?"                                       | Dense embeddings (primary) + BM25                           |
| **Structural** | "Which files call tool XYZ before tool ABC?"                       | AST/call-graph traversal (primary) + embeddings for context |
| **Usage**      | "Where is the Bluetooth-settings deeplink used?"                   | BM25 exact match (primary) + dense for related uses         |
| **Hybrid**     | "How does the auth module validate tokens before calling the API?" | All three signals, equal weight                             |

**Query preprocessing:**

1. **Identifier extraction** — pull out function/variable/class names from the query (e.g., "tool XYZ" → `XYZ`)
2. **Query expansion** — add synonyms/related terms (e.g., "preprocess" → `normalize`, `sanitize`, `transform`)
3. **Intent classification** — determine whether the query needs semantic, structural, or usage retrieval

### 5.2 Indexing (Offline, One-Time per Version)

#### Dense Index (Vector Store)

- **Embedding model:** `Qwen/Qwen3-Embedding-0.6B` INT8 (primary), falling back to `sentence-transformers/all-MiniLM-L6-v2`, then to a seeded hashing embedder so the pipeline runs with zero models downloaded. **Measured results below come from the MiniLM rung.** Qwen3 pools the *last* token and takes an instruction prefix on the query side only; both are implemented in `indexing/embedder.py` and neither is optional for correctness.
- **Chunking:** AST-aware — split by function/class/module boundaries using **tree-sitter** (not naive line-count chunking). Each chunk = one function or logical block with its docstring/comments.
- **Vector store:** FAISS CPU. `IndexFlatIP` below 50,000 vectors, `IndexIVFPQ` at or above — exact search is cheap at demo scale and avoids folding ANN recall loss into a reported number. ChromaDB is not a dependency.
- **Metadata per chunk:** file path, line range, function name, exported/private, imports

#### Sparse Index (BM25)

- **Library:** `bm25s` (`rank-bm25` is ruled out — `ADR-003`), with a pure-Python BM25 fallback so the sparse leg still runs on a bare install
- **Tokenization:** code-aware — split on camelCase, snake_case, dots; keep identifiers intact
- **Corpus:** same AST-chunked snippets as the dense index

#### Structural Index (AST + Call Graph)

- **Parser:** `tree-sitter` with the JavaScript grammar
- **Indexes built:**
  - **Call graph** — which function calls which (direct calls, method calls, callback chains)
  - **Import graph** — which file imports from which
  - **Symbol table** — every function/class/variable with its file + line + scope
  - **Export map** — what each module exports (for "where is X used?" queries)

### 5.3 Retrieval (Online, Per Query)

#### Pass 1: Hybrid Retrieval

1. **Dense search** — embed the query, retrieve top-K (K=100) nearest neighbors from FAISS
2. **BM25 search** — tokenize the query, retrieve top-K from the sparse index
3. **Structural search** (if query is structural) — traverse the AST/call-graph index
4. **Reciprocal Rank Fusion (RRF)** — merge all ranked lists:

```
RRF_score(d) = Σ  w_i / (k + rank_i(d))
               i∈{dense, bm25, structural}
```

where `k = 60` and `w_i` is the per-`QueryType` weight vector, not a flat 1. A usage query should
not weight dense the same as a semantic one: `SEMANTIC` is `(.60, .30, .10)`, `STRUCTURAL` is
`(.20, .20, .60)`, `USAGE` is `(.25, .55, .20)`, `HYBRID` is flat. A signal that returns nothing has
its weight dropped and the rest renormalised. Rank-only, so no score normalisation is needed.
(Appendix B derives the *unweighted* form as an illustration of the mechanism.)

#### Pass 2: Cross-Encoder Reranking

Take the top-N from RRF — `fusion_top_n`, which is **25** on the demo profile and **5** on the eval profile, where the heavier `bge-reranker-v2-m3` is used and the candidate chain is narrowed to pay for it. Score each (query, snippet) pair with a **cross-encoder reranker** (candidates: `Qwen3-Reranker-0.6B` quantized, `bge-reranker-v2-m3`, `ms-marco-MiniLM-L-6-v2`).

Cross-encoders are more accurate than bi-encoders because they see both query and document together — but too slow for first-stage retrieval over thousands of snippets. Using them only on the top-N is the sweet spot.

#### Pass 3: Agentic Refinement (see §6)

If the agent evaluates the results as insufficient, it rewrites the query and loops back.

### 5.4 Result Formatting

Each result includes:

- **Code snippet** — the relevant function/block
- **File path + line range** — exact location in the codebase
- **Relevance score** — the final reranked score
- **Match reason** — which signal contributed most (semantic, lexical, structural)
- **(Bonus) Optimization suggestion** — if the agent spots obvious issues in the surfaced code

---

## 6. The Agentic Loop

This is what makes the system **agentic** rather than a static retrieval pipeline. The agent can:

1. **Plan** — decompose a complex query into sub-queries
2. **Search** — run hybrid retrieval for each sub-query
3. **Assess** — judge sufficiency from the *scores* (top-1 rerank score, and how many
   results clear the floor). The LLM never sees snippet text: it classifies, expands,
   decomposes and judges sufficiency, and that boundary is enforced by module structure,
   not by convention. This is the project's central claim and `agent/evaluator.py`
   implements exactly this.
4. **Refine** — if results are poor, rewrite the query (add identifiers, change terms, broaden/narrow scope)
5. **Combine** — merge results from multiple sub-queries into a coherent ranking

### Example: Complex Query

**Query:** *"Which files call tool XYZ before tool ABC?"*

**Agent plan:**

1. Search the call graph for all callers of `XYZ` → set A
2. Search the call graph for all callers of `ABC` → set B
3. Find the intersection: files in both A and B
4. For each intersecting file, check execution order (does `XYZ` call appear before `ABC`?)
5. Return the matching code blocks with line locations

**Query:** *"How is the input preprocessed before going to the main function?"*

**Agent plan:**

1. Semantic search for "input preprocessing" → initial results
2. Identify function names in results (e.g., `normalize`, `sanitize`)
3. Structural search: trace call graph backwards from `main()` to find functions that run before it
4. Intersect: which of the semantic hits are also in the call chain leading to `main()`?
5. Rerank and return

### Why Not Just Use an LLM for Everything?

The problem statement explicitly says:

> *"The number of snippets and their length are too long to fit in the context window of any LLM."*

The agent uses an LLM only for:

- Query classification and rewriting (short inputs, fast)
- Deciding if results are sufficient (short evaluation)
- Decomposing complex queries into sub-queries

**Not** for reading the entire codebase.

---

## 7. Version-Aware Retrieval (P1)

> *"Codebases are hardly static — they keep changing with new commits."*

### Incremental Reindexing

When the codebase changes (new commit/version):

1. **Diff detection** — compare the new version against the indexed version (git diff or file hash comparison)
2. **Selective re-embedding** — only re-embed changed files. Unchanged files keep their existing embeddings.
3. **BM25 update** — add/remove/update entries for changed files
4. **AST update** — re-parse changed files, update call graph edges

**Goal:** Rebuild the index in seconds-to-minutes, not hours. A full re-index of the entire codebase is the fallback.

### Version Tagging

Every indexed snippet carries metadata:

```json
{
  "file": "src/agents/bluetooth.js",
  "lines": [42, 67],
  "function": "handleDeeplink",
  "version": "v2.3.1",
  "commit": "a1b2c3d",
  "last_modified": "2026-09-10"
}
```

When the user specifies a version, retrieval filters to that version's index.

### Storage Strategy

- **Per-version indexes** — each version gets its own FAISS index + BM25 index + AST graph
- **Shared base** — unchanged files share embeddings across versions (deduplication by content hash)
- **Trade-off:** more disk space for faster version switching vs. single index with version filtering

---

## 8. Evolutionary Retrieval (Bonus)

> *"Retrieve code snippets across all versions — even when snippets across versions are very similar."*

This is hard because:

- A function changed by one line across 10 versions produces 10 near-identical embeddings
- Standard ranking surfaces all 10 as "relevant," drowning out genuinely different code

### Our Approach

1. **Cross-version index** — all versions' snippets in one unified index, each tagged with version metadata
2. **Version-aware deduplication** — group near-identical snippets (cosine similarity **>= 0.95**, sharing `symbol` + `file_path`) across versions into "snippet families"
3. **Representative selection** — for each family, surface the most recent version by default, but allow expanding to see all versions
4. **Diff highlighting** — when showing a snippet family, highlight what changed between versions
5. **Ranking signal:** snippets that appear in *more* versions get a stability boost; snippets unique to a single version get a novelty signal

### Result Format for Evolutionary Queries

```
Result 1: handleDeeplink() in src/agents/bluetooth.js
  ├── v2.3.1 (current) — lines 42-67 [★ shown]
  ├── v2.3.0 — lines 42-65           [diff: +2 lines, error handling added]
  └── v2.2.0 — lines 38-60           [diff: refactored from switch to map]

Result 2: preprocessInput() in src/utils/normalize.js
  └── v2.3.1 (only version) — lines 10-25
```

---

## 9. Evaluation & Metrics

### Screening Round (Automated)

Performance on the **test split of CoIR** `AppsRetrieval` **dataset**:

| Metric      | What it measures                                 | How it's computed                                  |
| ----------- | ------------------------------------------------ | -------------------------------------------------- |
| **NDCG@10** | Are the most relevant results at the top?        | Normalized Discounted Cumulative Gain at cutoff 10 |
| **MRR**     | How early does the first relevant result appear? | Mean Reciprocal Rank                               |

**Reference baselines on CoIR Apps — quarantined, do not quote.**

This section previously carried a seven-row table of published NDCG@10 figures (BM25 4.8,
UniXcoder 1.4, E5-PT 10.6, BGE 0.6B 14.7, E5-Mistral 23.5, Voyage-Code-2 26.5, Revela 26.6) and a
target of "beat BGE (14.7)". **The table is withdrawn and the target with it.** An adversarial
audit of this suite could not locate 14.7 in either paper the figure was attributed to, and no one
on the team has since re-opened those papers to check. The rest of the rows inherited the same
provenance and are no better attested.

The rule, per [`docs/PRD.md` §2.0.3](docs/PRD.md#203-external-comparison-table--quarantined-pending-per-row-citation):
**no row returns to this table without a specific paper, table and page read by a human.** A
number that cannot be sourced is not a baseline, it is an anchor — and anchoring our own gates to
it is how the 14.7 error nearly reached a slide.

**Our target is expressed against our own measured baseline instead** — see *Measured Results*
below and [`docs/PRD.md` §2](docs/PRD.md#2-goals-and-success-metrics), which states the milestone gates as
multiples of `B`, the dense-only NDCG@10 we measured ourselves, rather than as absolute numbers
borrowed from elsewhere.

### How to Generate the Evaluation JSON

```python
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

class OurPipeline(AbsEncoder):
    # Wrap our full pipeline (hybrid retrieval + reranking) as an MTEB-compatible encoder
    ...

model = OurPipeline()
task = mteb.get_task("AppsRetrieval")
result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})
task_result = list(result.task_results)[0]
with open("appsretrieval_results.json", "w") as f:
    json.dump(task_result.to_dict(), f, indent=2)
```

Upload this JSON as a GitHub Release artifact tagged `PRISM_GENAI_HACKATHON_Y2026`.

### Measured Results

Everything below was produced by `scripts/run_eval.py` on this repository. Nothing here is a
projection, a citation, or a target. Where a run is not reportable, it says so and why.

**Screening benchmark — CoIR `AppsRetrieval`, full `test` split.**
3,765 judged queries over an 8,765-document corpus. No truncation. `degraded: False`.

| Arm | NDCG@10 | MRR@10 | Recall@100 | Wall clock |
|---|---|---|---|---|
| Dense only, `all-MiniLM-L6-v2` (22M params) | 7.59 | 6.39 | 27.22 | 39 min |
| Dense + BM25, weighted RRF (.85/.15) | **7.78** | **6.60** | 27.17 | 17 min (cached) |
| Target in `docs/PRD.md` | 20.0 | 22.0 | 65.0 | — |

**What the ablation shows, and it is not what we assumed.** Adding the sparse leg moves NDCG@10 by
`+0.19` and MRR@10 by `+0.21` — a real but small gain — while Recall@100 moves by `-0.05`, which is
to say not at all. BM25 is **reordering** the candidate pool, not enlarging it. Every relevant
document it surfaces, the dense leg had already found.

That localises the problem precisely. The 27.2 recall ceiling is a property of the dense embedder
alone, and fusion cannot lift it; neither can the reranker or the agent loop, which act even later
in the pipeline. The only lever that moves this number is a stronger first-stage embedder — the
`Qwen3-Embedding-0.6B` primary, which is 27x larger and has not yet been exported to ONNX and run.
The architecture is sound and measured; the model underneath it is the fallback.

For scale, published CoIR figures put BM25 at 4.8 and BGE-M3 (568M params) at 7.37 on this task.
A 22M-parameter model reaching 7.59 dense-only — no sparse leg, no reranker, no agent loop — is
therefore roughly at the level of a model **25x its size**. That is the honest framing of this
number, and it is also why the previously-quoted "BGE 0.6B = 14.7" baseline was retracted: an
adversarial audit could not locate it in either cited paper.

**The binding constraint is `Recall@100 = 27.22`, not NDCG.** Reranking and agentic refinement can
only reorder what the first stage already retrieved, so with roughly three-quarters of the relevant
documents never entering the candidate pool, no amount of second-stage work reaches NDCG@10 of 20.
First-stage recall — a stronger embedder, and fusing the sparse leg — is the only lever that moves
it. This is measured, not argued.

**Not yet reportable.** `scripts/run_eval.py` refuses to mark a run reportable while any
`PLACEHOLDER` constant is active or the git tree is dirty (`Rules.md` AP-14). Seven constants are
still unmeasured. The run above is therefore a **baseline**, not a submission number.

**Version-aware and evolutionary (P1 + Bonus).** Measured on a generated 60-file, 3-version corpus
(`scripts/make_demo_repo.py`), real stack — tree-sitter, faiss, bm25s, MiniLM:

| Property | Measured | Budget |
|---|---|---|
| Incremental reindex, 50 changed files | **295 ms** | 45 s (`NFR-02`) |
| Embedding calls for that reindex | **0** (101 blobs reused) | `FR-19` |
| Cross-version storage | 201 blobs for 303 chunk-instances | — |
| Snippet families over 3 versions | 107, of which 50 carry real diffs | `FR-21` |

### Hands-On Evaluation (Live Demo)

The jury will:

1. Review our PPT and demo video
2. Run our code on specific queries
3. Evaluate P1 (version retrieval) and Bonus (evolutionary retrieval)

**They want to see:**

- The solution working live, not just numbers
- Responses for given queries
- How fast the solution runs

### Jury Scoring Breakdown

| Criterion                         | Weight  | What they look for                     |
| --------------------------------- | ------- | -------------------------------------- |
| Working prototype & functionality | **30%** | Does it actually work?                 |
| Technical depth & feasibility     | **25%** | Is the approach sound?                 |
| Innovation & originality          | **20%** | Would a real user want this?           |
| Relevance to theme                | **15%** | Can it become a Samsung PRISM worklet? |
| Presentation & documentation      | **10%** | Clear PPT, demo, README                |

---

## 10. Tech Stack

### Core Retrieval

| Component           | Library/Tool                                                  | Why                               |
| ------------------- | ------------------------------------------------------------- | --------------------------------- |
| **Embedding model** | `Qwen3-Embedding:0.6B` (INT8 quantized) or `all-MiniLM-L6-v2` | CPU-friendly, strong on code      |
| **Vector store**    | FAISS CPU — `IndexFlatIP` < 50k vectors, `IndexIVFPQ` at/above | Exact at demo scale; ANN only when the corpus needs it |
| **BM25**            | `bm25s` (+ pure-Python fallback)                              | Materially faster than `rank-bm25` on a 10k-chunk corpus (`ADR-003`) |
| **Reranker**        | `bge-reranker-v2-m3` or `ms-marco-MiniLM-L-6-v2`              | Small cross-encoder, CPU-friendly |
| **AST parsing**     | `tree-sitter` + `tree-sitter-javascript`                      | Industry-standard JS parser, fast |

### Agentic Orchestration

| Component                      | Library/Tool                                                   | Why                                                         |
| ------------------------------ | -------------------------------------------------------------- | ----------------------------------------------------------- |
| **Agent framework**            | Custom — no framework dependency                               | The loop is ~100 lines with hard caps; a framework would add surface without adding capability (`ADR-007`) |
| **LLM (query rewriting only)** | `Qwen2.5-1.5B-Instruct` Q4_K_M GGUF, **local only** — never a third-party API (`NG-17`) | Classification, expansion, decomposition and sufficiency only. Never reads code. Fully optional: the heuristic rule engine is the default path |

### Infrastructure

| Component                   | Library/Tool             | Why                               |
| --------------------------- | ------------------------ | --------------------------------- |
| **Language**                | Python                   | MTEB compatibility, ML ecosystem  |
| **Evaluation**              | MTEB library             | Required by the problem statement |
| **Version control parsing** | Shell out to `git` — no library dependency | `git diff --name-status` and `git worktree` are exactly the two things needed |
| **API (if needed)**         | FastAPI                  | Lightweight, async                |

### Key Constraint: CPU-Only

All embedding, retrieval, and reranking must run efficiently on CPU:

- Use **INT8 quantized models** (ONNX Runtime or `optimum`)
- Use **FAISS CPU**, flat inner-product below 50k vectors and IVF-PQ above
- Reranker on the top `fusion_top_n` only (25 demo / 5 eval), never thousands
- BM25 is inherently CPU-fast
- Tree-sitter is C-based, extremely fast

---

## 11. Submission Checklist

### Registration (by 16 Sep 2026, 11:59 PM)

- [ ] Team registered via [Google Form](https://forms.gle/NxN6TWXLpcXmTnv66)
- [ ] Team name: `Incognito`
- [ ] All 4 members: name, email, phone, year, branch

### Final Submission (by 27 Sep 2026, 11:59 PM)

| Deliverable                  | Status | Notes                                    |
| ---------------------------- | ------ | ---------------------------------------- |
| **Working prototype**        |        | Public GitHub repo                       |
| **README**                   |        | Reproducible setup, Docker, requirements |
| **GitHub Release tag**       |        | `PRISM_GENAI_HACKATHON_Y2026`            |
| `appsretrieval_results.json` |        | Attached to the release                  |
| **Demo video** (max 5 min)   |        | YouTube or Drive link                    |
| **PPT**                      |        | Named `Incognito_Submission_ppt`      |
| **Google Form submission**   |        | All links included                       |

### PPT Must Include

- Theme ID, project title, team details
- Problem statement in our own words
- Solution and architecture diagram
- Tools and tech stack used
- Innovation highlights
- Results (NDCG@10, MRR numbers)
- Limitations

### Demo Video Must Show

- The solution working live (not just numbers)
- Responses for given queries
- How fast the solution runs
- Version-aware retrieval in action (P1)

---

## 12. Team & Responsibilities

> Authority for per-requirement ownership is the `Owner` column of
> [`docs/PRD.md §5`](docs/PRD.md#5-functional-requirements).

| Member                 | Focus Area                                     | Key Deliverables                                                                           |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Prabinder Singh**    | Retrieval Core — Embedding + BM25 + RRF fusion | Dense index, BM25 index, hybrid fusion pipeline, MTEB evaluation wrapper                   |
| **Anish Grover**       | Structural Intelligence — AST + Call Graph     | Tree-sitter JS parser, call/import graph builder, structural query handler                 |
| **Harshdeep Athawale** | Agentic Orchestration + Reranking              | Query classifier, agent loop (plan/search/refine), cross-encoder reranker, demo UI         |
| **Parth Deshmukh**     | Version Support + Evaluation + Submission      | Git-diff incremental reindexing, evolutionary retrieval, MTEB eval runner, PPT, demo video |

### Collaboration Rules

- All code in one shared GitHub repo
- Daily sync (brief stand-up)
- Integration test after each component lands
- Final 2 days: integration, eval, PPT, video

---

## 13. Timeline (10 Days)

**Re-baselined 2026-09-23.** The 15–25 Sep day plan this section used to carry **is void**, and
so is every "Day 10 = 24 Sep" reference that went with it. The build window is now five days:
**Day 1 = 2026-09-23, submission Day 5 = 2026-09-27 (23:59).** The authority for the plan is
[`docs/ImplementationPlan.md` §0](docs/ImplementationPlan.md#0-re-baseline-notice--read-this-before-anything-else);
this table is a summary of it and loses to it on any disagreement.

The re-baseline is a change of subject, not a slipped schedule. The system is built — 69 modules,
611 tests passing. What remains is the part that is actually scored: measuring it, and producing a
demo and a submission artifact from real numbers.

| Day | Date | Milestone | Who |
| --- | --- | --- | --- |
| **1** | 23 Sep | Reproducible clone, green suite, real model stack installed and smoked | All |
| **2** | 24 Sep | Full-split dense-only baseline `B` measured; ablation arms recorded in `artifacts/experiments.csv` | Prabinder + Parth |
| **3** | 25 Sep | Tagged demo repo (10–50 files); reranker weights exported; rerank + agent deltas measured against `B` | Harshdeep + Anish |
| **4** | 26 Sep | Demo video, deck and README built from measured numbers only; tuning frozen (train split only, `NG-29`) | Harshdeep + Parth |
| **5** | 27 Sep | Final reportable eval run, `appsretrieval_results.json`, GitHub release `PRISM_GENAI_HACKATHON_Y2026`, Google Form | All |

---

## 14. Key Risks & Mitigations

> Summary view. The canonical register is `RISK-01`–`RISK-12` in
> [`docs/ImplementationPlan.md`](docs/ImplementationPlan.md#5-risk-register).

| Risk                                        | Impact                              | Mitigation                                                                                                                         |
| ------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Embedding model too slow on CPU**         | Can't index the full corpus in time | Pre-compute and cache all embeddings. Use INT8 quantization. Fall back to `all-MiniLM-L6-v2` (fastest).                            |
| **NDCG@10 score too low**                   | Won't pass screening                | Hybrid retrieval (BM25 + dense) consistently beats either alone. Reranker adds 5-15 points of MRR. Iterate on query preprocessing. |
| **Tree-sitter parsing fails on edge cases** | Structural queries break            | Graceful degradation — if AST fails, fall back to regex-based identifier extraction. Test on the sample codebase early.            |
| **Agent loop is too slow**                  | Demo shows poor latency             | Cap the agent to max 2 refinement passes. Cache query classifications. The agentic loop is a bonus on top of the static pipeline.  |
| **CoIR dataset format mismatch**            | MTEB wrapper doesn't work           | Test the MTEB integration on day 1-2 with a dummy model. The format is standard BEIR-compatible.                                   |
| **Version indexing takes too long**         | P1 incomplete                       | Incremental indexing (only changed files) is the mitigation itself. Worst case: full reindex is acceptable for small codebases.    |

---

## Appendix A: Understanding the CoIR Apps Dataset

The evaluation dataset is the **test split of CoIR** `AppsRetrieval`:

| Split                | Entries                                          |
| -------------------- | ------------------------------------------------ |
| Corpus               | 8,765 code snippets                              |
| Queries              | 8,765 natural-language queries                   |
| Test relevance pairs | 3,765 query-document pairs with relevance scores |
| Train pairs          | 5,000 (can be used for fine-tuning or few-shot)  |

Each corpus entry has:

- `_id` — unique identifier
- `text` — the code snippet
- `title` — brief description
- `language` — programming language
- `meta_information` — starter code, URL

The task: for each test query, rank the corpus entries by relevance. The MTEB library handles the evaluation — we just need to wrap our retrieval pipeline as an `AbsEncoder`.

---

## Appendix B: How Reciprocal Rank Fusion Works

Given ranked lists from multiple retrievers:

```
RRF_score(doc) = Σ  1 / (k + rank_in_list_i(doc))
```

- `k = 60` (standard, prevents top-1 from dominating)
- A document ranked #1 in two lists gets: `1/61 + 1/61 = 0.0328`
- A document ranked #1 in one list and #10 in another: `1/61 + 1/70 = 0.0307`
- A document ranked #5 in all three lists: `1/65 × 3 = 0.0462` ← **higher!**

**Key property:** Documents that appear in multiple lists get boosted, even if they're not #1 in any single list. This is exactly what we want — snippets that match semantically AND lexically AND structurally are the most relevant.

---

## Appendix C: Connection to AXIOM (Previous Hackathon)

> Note: this project is now itself named **Axiom**. The section below refers to the
> team's earlier Samsung ennovateX AX Hackathon 2026 entry of the same name; Axiom
> here is its successor, not the same codebase.

This team's two founding members (Prabinder and Anish) built **AXIOM** for the Samsung ennovateX AX Hackathon 2026 (Problem Statement 06: Enhancing Reasoning in SLMs). AXIOM was a verifier-centric framework with a 5-head Process Reward Model (XD-PRM).

**What carries over:**

- Experience with embedding models, vector search, and retrieval pipelines
- Engineering discipline: shared contracts, type safety, config-driven architecture
- Agentic system design (AXIOM's adaptive-depth controller = agentic reasoning)

**What's new:**

- Code-specific domain (JavaScript, AST, call graphs) instead of math/commonsense reasoning
- Retrieval-focused (not generation or RL)
- CPU constraint (AXIOM required a GPU)
- Two new team members (Harshdeep and Parth) bring fresh perspectives

---

*Team Incognito — Prabinder Singh, Anish Grover, Harshdeep Athawale, Parth Deshmukh*
*Thapar Institute of Engineering & Technology, Patiala*
*Samsung PRISM GenAI Hackathon 3rd Edition, Theme 01: Agentic Code Intelligence*
