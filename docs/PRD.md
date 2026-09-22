# PRISM — Product Requirements Document

Defines what PRISM must do, for whom, and how success is measured for the Samsung PRISM GenAI Hackathon 3rd Edition (Theme 01) submission.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-15
**Status:** Draft

Related: [TechSpecifications.md](TechSpecifications.md) · [Design.md](Design.md) · [Schema.md](Schema.md) · [NonGoals.md](NonGoals.md) · [OpenQuestions.md](OpenQuestions.md) · [Decisions.md](Decisions.md) · [TestPlan.md](TestPlan.md) · [ImplementationPlan.md](ImplementationPlan.md) · [Tracker.md](Tracker.md) · [API.md](API.md) · [README.md](README.md)

---

## 1. Problem Statement

A modern voice-assistant codebase is roughly 10,000 JavaScript files spread across deeply
nested directories: dozens of agent modules, tool adapters, deeplink handlers, intent
routers, and preprocessing utilities. Nobody on the team has read all of it. Nobody can.

When an engineer needs to change behaviour, the first question is never "write me code" —
it is **"where does this already happen?"** Answering that question today means grepping
for a guess, opening twenty files, and reconstructing control flow by hand.

Three things make this specifically hard, and they are the three things PRISM is built around:

| Obstacle | Concrete failure today |
|---|---|
| The codebase does not fit in any LLM context window | 10k files is ~50M tokens. Pasting the repo into a model is not an option, so the system must retrieve before it reasons. |
| Lexical search misses meaning | The engineer types "preprocessing". The function is called `normalize()`. `grep preprocessing` returns nothing. |
| Semantic search misses structure | "Which files call `XYZ` before `ABC`?" is a question about the call graph and statement order. No embedding of any size answers it, because the answer is not in the text of any single snippet. |

**The task, stated precisely:** given a library of code and a natural-language query,
produce a ranking of code snippets ordered by relevance to the query, each located by
file path and line range.

This is a **retrieval** problem. PRISM retrieves, ranks, and locates. It does not generate
code and it does not explain code — see [NonGoals.md](NonGoals.md) (`NG-01`, `NG-02`, `NG-03`).

### 1.1 The Three Query Archetypes

Every design decision in PRISM traces back to one of these three queries, taken verbatim
from the problem statement:

| # | Query | Archetype | Why the other signals fail |
|---|---|---|---|
| Q1 | "How is the input preprocessed before going to the main function?" | Semantic + structural | "preprocessed" has no lexical match. Dense retrieval finds `normalize()`, `sanitize()`, `transform()`; only the call graph proves they run *before* `main()`. |
| Q2 | "Which files call tool XYZ before tool ABC?" | Structural | Requires two call-graph queries, a set intersection, and a source-order comparison on `ChunkMetadata.calls`. Pure similarity returns files that *mention* both tools, which is the wrong answer. |
| Q3 | "Where is the Bluetooth-settings deeplink used?" | Usage / lexical | The literal string `bluetooth-settings` is the highest-precision signal that exists. Dense retrieval dilutes it by returning every Bluetooth-adjacent function. BM25 nails it. |

Q1, Q2, and Q3 map directly onto `QueryType.SEMANTIC`, `QueryType.STRUCTURAL`, and
`QueryType.USAGE` in [Schema.md](Schema.md#querytype), and onto the three default
`strategy_weights` vectors locked in the contract.

---

## 2. Goals and Success Metrics

PRISM has exactly three goals, inherited from the problem statement's own priority ladder.

| Priority | Goal | Primary metric | Target | Baseline to beat |
|---|---|---|---|---|
| **P0** (must) | Retrieval accuracy on CoIR `AppsRetrieval` test split | NDCG@10 | **≥ 20.0** | BGE 0.6B = 14.7 |
| **P0** (must) | Early relevance | MRR | **≥ 22.0** | — |
| **P0** (must) | First-stage recall (the ceiling the reranker works under) | Recall@100 | **≥ 65.0** | — |
| **P1** (should) | Version-aware incremental reindexing | Wall clock for 50 changed files | **≤ 45 s** | Full reindex (~12 min) |
| **Bonus** | Evolutionary cross-version retrieval | Snippet-family dedup demonstrably collapses ≥ 2 versions of the same symbol into one ranked result with diffs | Qualitative + live demo | No rival submission does this |

Secondary, non-scored but gating metrics:

| Metric | Target | Source |
|---|---|---|
| Cold index, 10k chunks | ≤ 12 min | `_CONTRACT.md` §7 |
| Query p50 (no agent loop) | ≤ 900 ms | `_CONTRACT.md` §7 |
| Query p95 (2 agent passes) | ≤ 5 s | `_CONTRACT.md` §7 |
| Peak RSS during query | ≤ 4 GB | `_CONTRACT.md` §7 |

Reference leaderboard for context (NDCG@10 on CoIR `AppsRetrieval`, sources: CoIR
arXiv:2407.02883 and the Revela comparison table arXiv:2506.16552):

| Model | Size | NDCG@10 |
|---|---|---|
| UniXcoder | 0.1B | 1.4 |
| BM25 | — | 4.8 |
| E5-PT | 0.3B | 10.6 |
| BGE | 0.6B | 14.7 |
| **PRISM target** | **0.6B, CPU-only** | **≥ 20.0** |
| E5-Mistral | 7B | 23.5 |
| Voyage-Code-2 | — | 26.5 |
| Revela | 3B | 26.6 |

Best reported score on this task is 26.52. Our ≥ 20.0 target therefore sits **between
BGE-0.6B and E5-Mistral-7B** — above every sub-1B model on the board, below the 3B/7B and
proprietary-API tier. That is the honest claim and it is the one we will make in the PPT:
a 0.6B model on CPU closing most of the gap to a 7B model through architecture (three
signals, fusion, reranking, refinement) rather than through parameters.

Two numbers in that table govern the whole eval strategy and are easy to misread:

- **BM25 = 4.8.** Lexical retrieval is nearly useless on this benchmark, because the query
  is English prose describing a programming problem and the document is a Python solution
  with almost no shared surface vocabulary. Fusing a 4.8-scoring list at equal weight with
  a 14.7-scoring list is a **net negative**: it injects noise into the top ranks that the
  reranker then has to spend its 25 candidate slots on. The eval profile must therefore
  weight sparse **low**, not equally. The exact value is tuned, not guessed — see `OQ-02`
  in [OpenQuestions.md](OpenQuestions.md).
- **UniXcoder 0.1B = 1.4 < BM25 = 4.8.** Being a code model is not sufficient; NL-to-code
  alignment is what matters. This is why the embedder choice is a measured decision
  (`OQ-03`) and not a preference.

### 2.1 Definition of Done

The submission is done when all of the following are simultaneously true:

1. `axiom index <repo>` completes on the demo repo inside the cold-index budget.
2. `axiom query "<Q1|Q2|Q3>"` returns file+line results for all three archetypes.
3. `scripts/run_eval.py` produces `appsretrieval_results.json` with NDCG@10 ≥ 20.0.
4. `axiom reindex --to <newer-commit>` completes inside 45 s for a 50-file diff.
5. `axiom query --all-versions "<query>"` returns `SnippetFamily`-collapsed results.
6. The whole pipeline runs green with `AXIOM_LLM_ENABLED=false`.
7. Release `PRISM_GENAI_HACKATHON_Y2026` exists with the eval JSON attached.

### 2.2 Two Evaluation Contexts, Two Profiles

This is the single most important structural fact about the project, and it is a
**deliberate design split, not a gap**. PRISM is scored in two different places, on two
different corpora, and they are not the same language.

| | Screening benchmark | Live demo |
|---|---|---|
| Corpus | CoIR `AppsRetrieval` — built on the APPS dataset. Queries are English competitive-programming problem statements; corpus documents are **Python** solutions. | The ~10k-file **JavaScript** voice-assistant codebase from the problem statement. |
| Document shape | Standalone single-file solutions. No imports across documents, no cross-file call graph, no module boundaries. | Deeply nested multi-module repo: imports, exports, call chains, tool dispatch. |
| What wins | NL-to-code semantic alignment. Dense retrieval plus reranking. | All three signals. Structural queries (Q2) are only answerable with the call graph. |
| Profile | `configs/eval.yaml` — dense-heavy, sparse down-weighted, **structural disabled** | `configs/demo.yaml` — all three signals at the locked per-`QueryType` weights |
| Jury weight it earns | Screening gate; feeds the "Results" slide | 30% prototype + 25% depth + 20% innovation = **75%** of the scored rubric |

**Consequence, stated plainly:** the tree-sitter AST/call-graph signal contributes
approximately nothing to NDCG@10, because APPS snippets have no cross-file structure for it
to index. Running it there would cost latency and inject an empty or near-random third list
into fusion. So we do not run it there. The structural signal earns its keep on the live
demo — which is where three quarters of the jury score lives, and which is the deliverable
the problem statement's JavaScript constraint actually describes.

The JavaScript constraint in the problem statement applies to the **codebase being
indexed**, not to the screening benchmark. Two profiles is the correct engineering response
to two corpora; one profile would be a compromise that is wrong for both. This is tracked
as `OQ-01`, the highest-priority open question, in
[OpenQuestions.md](OpenQuestions.md#oq-01) and will be promoted to an ADR in
[Decisions.md](Decisions.md) once the profile weights are measured.

Profile selection is explicit and logged: every run prints the active profile name, and
`appsretrieval_results.json` is only ever generated under `configs/eval.yaml`.

---

## 3. Personas

### P-1 — Jury Evaluator (Samsung PRISM reviewer)

| Attribute | Detail |
|---|---|
| Context | Reviews dozens of submissions. Has ~10 minutes per team. Has never seen our repo. |
| Environment | Their own laptop, CPU-only, no pre-warmed caches, possibly offline-restricted. |
| Wants | A `README` they can follow in 5 commands; a demo video that shows live behaviour, not slides; numbers that are reproducible from the attached JSON. |
| Fails us if | Setup breaks, a model download stalls, latency is visibly bad, or the P1/Bonus claims are only described and never demonstrated. |
| Requirements they drive | `FR-22`, `FR-23`, `FR-25`, `NFR-01`, `NFR-04`, `NFR-06`, `NFR-07`, `NFR-09` |

### P-2 — Codebase Developer (the actual end user)

| Attribute | Detail |
|---|---|
| Context | Works daily in a ~10k-file JavaScript voice-assistant repo. Joined 3 weeks ago. |
| Environment | 8-core laptop, 16 GB RAM, repo already cloned, index built once overnight. |
| Wants | To type Q1/Q2/Q3 in English and get the 3–5 exact functions with file paths and line numbers they can jump to. |
| Fails us if | Results are plausible but wrong, line numbers are off, or the answer is a paragraph of prose instead of a location. |
| Requirements they drive | `FR-01`–`FR-14`, `FR-20`, `FR-21`, `NFR-03`, `NFR-08` |

### P-3 — CI / Automation Consumer

| Attribute | Detail |
|---|---|
| Context | A pipeline step, not a human. Runs after every merge to keep the index fresh; or a script that batch-queries for an audit. |
| Environment | Headless container, no TTY, JSON in / JSON out, exit codes matter. |
| Wants | Deterministic output, stable field names, machine-readable errors, incremental reindex that costs seconds not minutes. |
| Fails us if | Output shape drifts between runs, the CLI prints tables instead of JSON, or reindex re-embeds unchanged files. |
| Requirements they drive | `FR-17`, `FR-18`, `FR-19`, `FR-23`, `FR-24`, `NFR-02`, `NFR-08`, `NFR-10` |

---

## 4. User Stories

Each story carries the `FR-##` that implements it. Acceptance criteria are observable from
outside the system — a CLI invocation, an HTTP response, or an eval number.

### US-1 → `FR-01`, `FR-02`
**As a** codebase developer, **I want** the system to figure out on its own whether my
question is about meaning, structure, or usage, **so that** I do not have to learn a query syntax.

*Acceptance:* `axiom classify "Which files call XYZ before ABC?"` prints
`query_type=STRUCTURAL` with `extracted_identifiers=["XYZ","ABC"]`. All three archetype
queries classify correctly with `AXIOM_LLM_ENABLED=false`.

### US-2 → `FR-06`, `FR-08`, `FR-11`
**As a** codebase developer, **I want** lexical and semantic matches merged into one list,
**so that** a snippet that matches both ranks above one that matches either.

*Acceptance:* For a query where chunk A is dense-rank 5 / sparse-rank 5 and chunk B is
dense-rank 1 only, the fused output ranks A above B (RRF `k=60`, HYBRID weights).

### US-3 → `FR-09`, `FR-10`
**As a** codebase developer, **I want** to ask ordering questions about function calls,
**so that** I can trace execution without reading every file.

*Acceptance:* `axiom query --type structural "which files call parseIntent before dispatch"`
returns only chunks whose `metadata.calls` contains `parseIntent` at a lower index than
`dispatch`, each with its file path and line range.

### US-4 → `FR-12`
**As a** codebase developer, **I want** the top candidates re-scored by a model that sees
my query and the code together, **so that** the best answer is at position 1, not position 7.

*Acceptance:* On a held-out set of 20 hand-labelled queries against the demo repo,
reranking moves the gold chunk into the top 3 for ≥ 15 of them; measured MRR improves over
the RRF-only ordering.

### US-5 → `FR-03`, `FR-13`
**As a** codebase developer, **I want** the system to notice when its own results are weak
and try again with a better query, **so that** I get an answer on the first ask instead of
rephrasing manually.

*Acceptance:* When top-1 rerank score < 0.35 or fewer than 3 results exceed 0.20, the loop
runs a refinement pass; total passes never exceed 2; total wall clock never exceeds 5 s;
the response records `passes_used` and the rewritten query.

### US-6 → `FR-14`
**As a** codebase developer, **I want** every result to carry the exact file and line range
and a reason it matched, **so that** I can open it and trust it.

*Acceptance:* Every `RetrievalResult` has non-empty `chunk.location.file_path`,
`start_line ≤ end_line`, and a `match_reason` naming the `dominant_signal`. Slicing the
file at `[start_line, end_line]` reproduces `chunk.text` byte-for-byte.

### US-7 → `FR-16`, `FR-20`
**As a** codebase developer, **I want** to pin a query to a specific release,
**so that** I can answer "how did this work in v2.2.0?" without checking out old code.

*Acceptance:* `axiom query --version v2.2.0 "<q>"` returns only chunks whose
`metadata.version_id == "v2.2.0"`; results for `v2.3.1` differ where the code differs.

### US-8 → `FR-18`, `FR-19`
**As a** CI consumer, **I want** reindexing to cost time proportional to the diff, not the
repo, **so that** I can run it on every merge.

*Acceptance:* After a commit touching 50 files, `axiom reindex --to HEAD` completes in
≤ 45 s and the log reports embeddings computed ≈ chunks in changed files only; a pure file
rename with unchanged content computes 0 embeddings.

### US-9 → `FR-21`
**As a** codebase developer, **I want** ten near-identical versions of one function shown as
one result I can expand, **so that** they do not crowd out genuinely different code.

*Acceptance:* `axiom query --all-versions "<q>"` returns a `SnippetFamily` whose
`representative` is the newest member, `versions` lists every version containing it,
`stability` ∈ (0,1], and `diffs` describes each successive change.

### US-10 → `FR-22`
**As a** jury evaluator, **I want** a standard MTEB results file, **so that** I can verify
the score without running anything.

*Acceptance:* `python scripts/run_eval.py --task AppsRetrieval --split test` writes
`appsretrieval_results.json` in MTEB `TaskResult` shape, containing `ndcg_at_10` and
`mrr_at_10`, attached to the release tag.

### US-11 → `FR-24`, `FR-25`
**As a** jury evaluator, **I want** to click through a query and see signal contributions,
**so that** I can tell the hybrid architecture is real and not a slide.

*Acceptance:* Streamlit on `:8501` shows, per result, the per-signal ranks from
`FusedResult.contributions` and the rerank delta; the API on `:8000` returns the same data
as JSON from `POST /query`.

### US-12 → `NFR-07`
**As a** jury evaluator on a locked-down machine, **I want** the system to work without the
LLM and without the primary models, **so that** a download failure is not a demo failure.

*Acceptance:* With `AXIOM_LLM_ENABLED=false` and the fallback model profile
(`configs/fast.yaml`), all three archetype queries still return ranked results; the log
states which fallback was used.

---

## 5. Functional Requirements

Priority column: **P0** = screening-critical, **P1** = version support, **B** = bonus,
**INF** = infrastructure needed by any of the above.

| ID | Requirement | Priority | Owner |
|---|---|---|---|
| `FR-01` | Classify every incoming query into `SEMANTIC`, `STRUCTURAL`, `USAGE`, or `HYBRID` and emit a `QueryPlan` with `strategy_weights`. Heuristic rule engine is the default path; the query LLM refines only when `AXIOM_LLM_ENABLED=true`. | P0 | Harshdeep |
| `FR-02` | Extract candidate code identifiers from the raw query (camelCase/snake_case/dotted tokens, quoted literals, `foo()` forms) into `QueryPlan.extracted_identifiers`, and produce `expansion_terms` from a curated code-synonym map (`preprocess → normalize, sanitize, transform, clean`). | P0 | Harshdeep |
| `FR-03` | Decompose a multi-clause query into ≤ 3 ordered `sub_queries` when the plan requires more than one retrieval (e.g. "calls X before Y" → one sub-query per identifier plus an ordering check). | P0 | Harshdeep |
| `FR-04` | Chunk source files on tree-sitter AST node boundaries into `Chunk` records targeting 64–512 tokens; split oversized functions at statement boundaries with 1-statement overlap; merge chunks < 16 tokens into their parent. Fall back to a line-window splitter when parsing fails. | P0 | Anish |
| `FR-05` | Build the dense index: embed every chunk with the locked embedder, L2-normalise, persist `dense.faiss` (`IndexFlatIP` under 50k vectors, `IndexIVFPQ` at or above 50k) plus `dense.idmap.json`. | P0 | Prabinder |
| `FR-06` | Dense retrieval: embed the query (or each sub-query) and return top **K=100** `ScoredChunk` records with `signal=DENSE`, inner-product scores, 1-indexed ranks. | P0 | Prabinder |
| `FR-07` | Build the sparse index with `bm25s` over the identical chunk corpus, using a code-aware tokenizer that splits camelCase/snake_case/dots while also retaining the original identifier token. | P0 | Prabinder |
| `FR-08` | Sparse retrieval: return top **K=100** `ScoredChunk` records with `signal=SPARSE`, using the query's raw terms plus `extracted_identifiers` and `expansion_terms`. | P0 | Prabinder |
| `FR-09` | Build the structural index into `structural.sqlite` with four relations: symbols (name, kind, file, line span, scope), calls (caller → callee, source order), imports (file → module), exports (module → symbol). | P0 | Anish |
| `FR-10` | Structural retrieval: answer callers-of, callees-of, imports-of, exports-of, and ordered-call-pair queries by SQL traversal; return top **K=50** `ScoredChunk` with `signal=STRUCTURAL`. Return an empty list rather than an error when no identifier resolves. | P0 | Anish |
| `FR-11` | Fuse the per-signal ranked lists with weighted RRF: `score(d) = Σ_i w_i / (60 + rank_i(d))`, weights from `QueryPlan.strategy_weights`; emit `FusedResult` carrying `contributions` and `dominant_signal`; truncate to **N=25**. | P0 | Prabinder |
| `FR-12` | Rerank the 25 fused candidates with the locked cross-encoder, write `rerank_score`, re-sort, and truncate to top **10**. Reranking must be skippable by config for ablation runs. | P0 | Harshdeep |
| `FR-13` | Bounded agentic refinement: after reranking, evaluate sufficiency (trigger: top-1 `rerank_score` < 0.35 **or** fewer than 3 results above 0.20). On trigger, rewrite/broaden the query and re-run retrieval. Hard caps: **2 passes** (`AXIOM_AGENT_MAX_PASSES`) and **5 s** wall clock, enforced by a monotonic deadline checked before each pass. | P0 | Harshdeep |
| `FR-14` | Format each survivor as a `RetrievalResult`: chunk text, `file_path`, `start_line`–`end_line`, final `score`, human-readable `match_reason` naming the dominant signal and the matched identifiers, and the per-signal rank map. | P0 | Harshdeep |
| `FR-15` | Populate `optimization_hint` for a result when a static pattern-check fires on the surfaced chunk (sync I/O in an async path, `await` inside a loop, unbounded `for..in` over a network payload). Hint is a fixed string from a rule table, never LLM prose. | B | Harshdeep |
| `FR-16` | Stamp every chunk with `version_id`, `commit_sha`, and `last_modified` at index time, resolved from git when the target is a repo and from a config override otherwise. | P1 | Parth |
| `FR-17` | Maintain `.axiom/registry.json` mapping `version_id` → manifest path plus the active version, and write one `VersionManifest` per version including `file_hashes`, `embedding_model`, `embedding_dim`, `index_kind`, and `parent_version`. | P1 | Parth |
| `FR-18` | Incremental reindex: resolve `git diff --name-status <old>..<new>` into {A,M,D,R}; re-chunk and re-embed only A and M files; drop D chunks from all three indexes; update the structural graph edges for touched files only. Full rebuild remains available as `--full`. | P1 | Parth |
| `FR-19` | Content-addressed embedding reuse: persist vectors at `.axiom/blobs/<content_hash>.npy` and reuse them across versions, so an unchanged-content rename or a revert costs zero embedding compute. | P1 | Parth |
| `FR-20` | Version-scoped query: `--version <id>` restricts retrieval to that version's index; absent the flag, the registry's active version is used. | P1 | Parth |
| `FR-21` | Evolutionary retrieval: with `--all-versions`, search across every indexed version, group chunks with cosine ≥ 0.95 sharing `symbol` + `file_path` into a `SnippetFamily`, choose the newest member as `representative`, compute `stability = members / total_versions`, apply the ranking bonus `final = base * (1 + 0.10 * stability)` to families spanning ≥ 2 versions, and attach per-transition `diffs`. | B | Parth |
| `FR-22` | MTEB adapter: expose the full pipeline as an MTEB-compatible encoder, run `AppsRetrieval` test split, and export `appsretrieval_results.json` containing `ndcg_at_10` and `mrr_at_10` via `scripts/run_eval.py`. | P0 | Parth |
| `FR-23` | Typer CLI `prism` with subcommands: `index`, `reindex`, `query`, `classify`, `versions`, `families`, `eval`, `serve`, `ui`. Every subcommand supports `--json` for machine-readable output and returns non-zero on failure. | INF | Prabinder |
| `FR-24` | FastAPI service on port 8000: `POST /query`, `GET /versions`, `GET /chunk/{chunk_id}`, `GET /health`, all typed by the same Pydantic models as the CLI. Dev-only; no auth (see `NG-08`). | INF | Harshdeep |
| `FR-25` | Streamlit UI on port 8501: query box, query-type badge, result cards with syntax-highlighted snippet and `file:line` header, per-signal rank breakdown, agent-pass indicator, version selector, and an expandable snippet-family view. | INF | Harshdeep |
| `FR-26` | Supervised tuning on the `AppsRetrieval` **train** split. The 5,000 train query-document pairs are the only corpus used to tune RRF signal weights per profile, the rerank sufficiency thresholds (0.35 / 0.20), and query-preprocessing variants. Split 4,000 tune / 1,000 dev, seeded and committed as an id list under `data/splits/`. No gradient training of any model (see `NG-06`). The test split is never used for tuning; each candidate configuration is scored on test at most once and every such run is logged in [Tracker.md](Tracker.md). | P0 | Parth |

---

## 6. Non-Functional Requirements

| ID | Requirement | Measurement | Owner |
|---|---|---|---|
| `NFR-01` | Cold index of 10,000 chunks completes in **≤ 12 min** on the reference box (8-core CPU, 16 GB RAM). | `scripts/bench_latency.py --phase index` wall clock | Prabinder |
| `NFR-02` | Incremental reindex of a 50-changed-file diff completes in **≤ 45 s**. | `axiom reindex` wall clock, 3-run median | Parth |
| `NFR-03` | Query p50 latency with the agent loop disabled is **≤ 900 ms**. | 100-query sample, `bench_latency.py --phase query` | Prabinder |
| `NFR-04` | Query p95 latency with up to 2 agent passes is **≤ 5 s**, enforced by a hard deadline, not by hope. | Same harness with `AXIOM_AGENT_MAX_PASSES=2` | Harshdeep |
| `NFR-05` | Peak resident set size during query serving stays **≤ 4 GB**. | `psutil` RSS sampled at 100 ms during the query benchmark | Prabinder |
| `NFR-06` | CPU-only execution. No CUDA, ROCm, or accelerator dependency anywhere in the install or runtime path; ONNX Runtime uses the CPU execution provider exclusively. | `pip check` on a CPU-only image plus a smoke run in `python:3.11-slim-bookworm` | Prabinder |
| `NFR-07` | Mandatory graceful degradation. Every model has a declared fallback (embedder → MiniLM, reranker → ms-marco-MiniLM-L-6-v2, LLM → heuristic rule engine, tree-sitter → regex identifier extraction). The full pipeline must produce ranked results with `AXIOM_LLM_ENABLED=false` and with any single primary model unavailable. A file that fails to parse degrades to window chunking and never aborts the index. | Smoke matrix: 4 degradation scenarios × 3 archetype queries | Harshdeep |
| `NFR-08` | Determinism. Identical query + identical index + identical config yields byte-identical ranked `chunk_id` order. All sampling temperatures are 0; FAISS `IndexIVFPQ` training is seeded; `chunk_id` and `content_hash` are pure blake2b-128 functions of their inputs. | Repeat-run diff over 50 queries | Prabinder |
| `NFR-09` | Reproducibility. A clean clone reaches a working index with `uv sync` + one documented command; `uv.lock` is committed; a Dockerfile builds the same environment; model revisions are pinned by name and revision in `configs/`. | Fresh-clone rehearsal on day 9 | Parth |
| `NFR-10` | Observability. Every stage (chunk, embed, dense, sparse, structural, fuse, rerank, agent pass) emits a structured timing record; `--json` output includes a `timings` block so latency claims are auditable. | Inspect one `--json` response per archetype | Harshdeep |
| `NFR-11` | Code health. `ruff` lint + format clean at line length 100; `mypy --strict` clean on `src/axiom/core`, `src/axiom/retrieval`, `src/axiom/schema`; `pytest` green in CI; CI order is ruff → mypy → pytest → smoke index. | GitHub Actions run | Prabinder |
| `NFR-12` | Storage and first-run footprint. On-disk index for 10k chunks stays under 1.5 GB including shared blobs; total model download for the primary profile stays under 2.5 GB; the fallback profile stays under 500 MB so a bandwidth-limited evaluator can still run the demo. | `du -sh .prism` plus model cache size after a clean run | Parth |

---

## 7. Prioritisation (MoSCoW)

Mapped onto the P0/P1/Bonus ladder and the 2026-09-15 → 2026-09-27 build window.

| MoSCoW | Scope | Requirements | Land by |
|---|---|---|---|
| **Must have** | P0 screening path: chunking, all three indexes, all three retrievers, RRF, reranker, result formatting, CLI, MTEB export, train-split tuning | `FR-01`, `FR-02`, `FR-04`–`FR-12`, `FR-14`, `FR-22`, `FR-23`, `FR-26`; `NFR-01`, `NFR-03`, `NFR-05`, `NFR-06`, `NFR-08`, `NFR-09`, `NFR-11` | Day 7 (21 Sep) |
| **Must have** | Bounded agent loop — the word "agentic" is in the theme title; without it we are a hybrid search box | `FR-03`, `FR-13`; `NFR-04`, `NFR-07` | Day 6 (20 Sep) |
| **Should have** | P1 version support: tagging, manifests, git-diff incremental reindex, blob reuse, version-scoped query | `FR-16`–`FR-20`; `NFR-02`, `NFR-12` | Day 8 (22 Sep) |
| **Should have** | Demo surfaces the jury actually touches | `FR-24`, `FR-25`; `NFR-10` | Day 9 (23 Sep) |
| **Could have** | Bonus evolutionary retrieval with snippet families and stability bonus | `FR-21` | Day 8 (22 Sep), after P1 lands |
| **Could have** | Optimization hints on surfaced code | `FR-15` | Day 9 (23 Sep), cut first if behind |
| **Won't have** | Everything in [NonGoals.md](NonGoals.md) `NG-01`–`NG-20` | — | Never, this cycle |

### 7.1 Cut Order Under Time Pressure

If the schedule slips, features are dropped in exactly this order. This list is agreed in
advance so nobody negotiates it at 2 a.m. on 23 Sep.

1. `FR-15` optimization hints.
2. `FR-25` Streamlit polish (keep a minimal single-column UI; the CLI carries the demo).
3. `FR-21` stability ranking bonus (keep family grouping, drop the score adjustment).
4. `FR-24` REST API (CLI covers P-3's needs).
5. `FR-03` query decomposition (keep single-pass refinement in `FR-13`).

Nothing above `FR-03` in this list may be cut. `FR-13` is never cut — it is the theme.

---

## 8. Jury Scoring Alignment

The five criteria and the specific artefacts that earn each one.

| Criterion | Weight | What earns it | Owning requirements |
|---|---|---|---|
| Working prototype and functionality | **30%** | A live CLI and UI answering Q1/Q2/Q3 on a real 10k-file JS repo, plus `axiom reindex` running in front of the jury and `--all-versions` collapsing a snippet family. The demo video shows wall-clock timings on screen. Degradation matrix proves it also works on the evaluator's own locked-down laptop. | `FR-06`, `FR-08`, `FR-10`–`FR-14`, `FR-18`, `FR-21`, `FR-23`, `FR-25`, `NFR-01`–`NFR-07` |
| Technical depth and feasibility | **25%** | Three genuinely different signals with a principled fusion rule (weighted RRF, `k=60`, per-query-type weights), a two-stage retrieve-then-rerank architecture with declared candidate widths (100/100/50 → 25 → 10), content-addressed incremental indexing, and a locked Pydantic data contract in [Schema.md](Schema.md). Every constant in [TechSpecifications.md](TechSpecifications.md) is justified, not guessed. Ablation table (dense only / +sparse / +structural / +rerank / +agent) shows each component's contribution in NDCG@10. | `FR-04`, `FR-05`, `FR-07`, `FR-09`, `FR-11`, `FR-12`, `FR-19`, `NFR-08`, `NFR-11` |
| Innovation and originality | **20%** | The two things no comparable submission has: (a) a **structural AST/call-graph signal** as a first-class retrieval list inside the fusion, which is the only way Q2-class ordering queries are answerable at all; (b) a **bounded agentic refinement loop** with an explicit sufficiency predicate and a hard 5 s deadline — agentic in a way that is measurable rather than decorative. Plus snippet-family evolutionary retrieval with stability-weighted ranking. | `FR-09`, `FR-10`, `FR-13`, `FR-21`, `FR-15` |
| Relevance to theme | **15%** | Theme 01 asks for agentic code intelligence under a CPU constraint on a repo larger than any context window. PRISM never feeds code to an LLM — the LLM only classifies, expands, decomposes, and judges sufficiency. Everything is retrieval over pre-built indexes. The P1 story (codebases keep changing with new commits) is answered with real git-diff incremental reindexing, not a content-hash cache bolted onto embeddings. This is directly shippable as a PRISM worklet: index a Samsung repo, serve queries to its owning team. | `FR-13`, `FR-16`–`FR-21`, `NFR-06` |
| Presentation and documentation | **10%** | This `docs/` suite: [PRD.md](PRD.md), [TechSpecifications.md](TechSpecifications.md), [Design.md](Design.md), [Schema.md](Schema.md), [Appflow.md](Appflow.md), [TestPlan.md](TestPlan.md), [Decisions.md](Decisions.md) (ADRs), [ImplementationPlan.md](ImplementationPlan.md) (day plan + `RISK-01`–`RISK-12`), [Tracker.md](Tracker.md), [NonGoals.md](NonGoals.md), [OpenQuestions.md](OpenQuestions.md), [Setup.md](Setup.md), [API.md](API.md), [Glossary.md](Glossary.md). A README with a 5-command setup and a Docker path. `Incognito_Submission_ppt` covering problem, architecture, stack, innovation, results, limitations. A ≤ 5-minute demo video. Release `PRISM_GENAI_HACKATHON_Y2026` with `appsretrieval_results.json` attached. | `FR-22`, `NFR-09`, `NFR-10` |

### 8.1 Where the Marks Are Actually Won

30% + 25% = 55% of the score is *prototype works* and *approach is sound*. That is why the
cut order in §7.1 protects the core pipeline and the agent loop above everything else, and
why `NFR-07` (degradation) is a Must rather than a nicety: a prototype that fails to start
on the evaluator's machine scores zero on 30% of the rubric regardless of its NDCG.

---

## 9. Competitive Positioning

A rival public submission exists at `github.com/DeshnaDey/Samsung-PRISM` — same theme, same
project name, same intended release tag — working the same dataset with dense + BM25 + RRF
+ cross-encoder reranking. It is a competently engineered two-signal hybrid, and it is worth
being precise about both what it does well and what it does not attempt.

**What they do well, and we should match rather than ignore:** frozen stage interfaces with
a no-op passthrough pattern (every stage is wired end-to-end before any stage is smart), an
`experiments.md` discipline for logging runs, pinned and clean-room-verified dependencies,
and documented MTEB v2 API traps. Our equivalents are the locked contract in
[Schema.md](Schema.md), the run log in [Tracker.md](Tracker.md), the committed `uv.lock`
(`NFR-09`), and the MTEB adapter notes in [TechSpecifications.md](TechSpecifications.md).

**What they do not attempt** (verified against the public repo as of 2026-09-14): no
structural/AST signal; no agent loop despite shipping under the word *Agentic*; no
version-aware retrieval — `src/versioning/cache.py` is a content-hash embedding cache,
which is a rerun speedup, not the ability to query a version; no Bonus evolutionary
retrieval; and **no measured result at all** — every `ENABLE_*` flag is `False` and
`experiments.md` is entirely `TBD`. They use `rank-bm25` and a FAISS `Flat` index.

The table below is not a criticism of their engineering; it is the precise statement of
what PRISM adds, and it is the source of the 20% innovation argument in §8.

| Dimension | Rival approach | PRISM | Why it matters to the rubric |
|---|---|---|---|
| Retrieval signals | Dense + BM25 (2) | Dense + BM25 + **structural AST/call-graph** (3) | Query archetype Q2 ("calls X before Y") is unanswerable with 2 signals. The third list is the only thing that makes structural queries work, and it is the headline innovation claim. |
| Fusion | RRF | Weighted RRF, `k=60`, weights selected per `QueryType` | A usage query should not weight dense the same as a semantic query. Per-type weights are a measurable ablation, not a stylistic difference. |
| Reranking | Cross-encoder | Cross-encoder, with reranking made ablatable so its contribution is reportable | Same capability; we additionally quantify it. |
| Agentic behaviour | None — single-pass pipeline | **Bounded agent loop:** sufficiency predicate (top-1 < 0.35 or < 3 results > 0.20), query rewrite, max 2 passes, 5 s deadline | The theme is titled *Agentic* Code Intelligence. A single-pass pipeline is a strong retriever but concedes theme relevance (15%) and innovation (20%). |
| Query understanding | Query text as given | Classification into 4 types, identifier extraction, code-synonym expansion, decomposition into ≤ 3 sub-queries | Directly drives the per-type fusion weights and the structural routing. |
| Versioning (P1) | Content-hash embedding cache | Per-version manifests + registry, `git diff --name-status` A/M/D/R handling, content-addressed shared embedding blobs, version-scoped retrieval | A hash cache avoids recompute but does not let you *query a version*. P1 asks for retrieval across versions, not just cheaper reindexing. |
| Evolutionary (Bonus) | Not addressed | `SnippetFamily` grouping at cosine ≥ 0.95 on `symbol`+`file_path`, newest-as-representative, `stability` score, `final = base * (1 + 0.10 * stability)`, per-transition diffs | This is the stated bonus goal. Attempting it credibly is differentiating on its own. |
| Degradation story | Not stated | Every model has a declared fallback; pipeline runs with `AXIOM_LLM_ENABLED=false`; parse failures degrade to window chunking | Protects the 30% prototype score on an unknown evaluator machine. |
| Fusion weighting on the benchmark | Default `FUSION_WEIGHTS = (0.5, 0.5)` | Profile-specific weights tuned on the 5,000 train pairs (`FR-26`); sparse deliberately down-weighted in `configs/eval.yaml` | BM25 scores 4.8 on APPS against BGE's 14.7. A 50/50 fusion actively depresses the dense signal. Evidence-driven weighting is a measurable point of difference, and one we can show as an ablation row. |
| Libraries | `rank-bm25`, FAISS `Flat` | `bm25s` (locked), FAISS `IndexFlatIP` under 50k vectors / `IndexIVFPQ` at or above 50k | `bm25s` is materially faster on a 10k-chunk corpus; the index-kind switch keeps the cold-index budget reachable at demo scale. |
| Reported results | None yet — all feature flags off, `experiments.md` all `TBD` | Ablation table with a number in every row, plus `appsretrieval_results.json` attached to the release | An unmeasured pipeline cannot claim a score. Having numbers is itself a 30%/25% differentiator. |

**Honest assessment of our risk relative to the rival:** they have fewer moving parts, so
their P0 number could land earlier and more safely once they turn their flags on. Our
mitigation is sequencing — the dense + sparse hybrid plus reranker is scheduled green by
Day 4 (18 Sep), before the structural signal and the agent loop are integrated, so we
always hold a submittable baseline. See the risk register at
[ImplementationPlan.md#risk-register](ImplementationPlan.md#risk-register) and the day plan
in [ImplementationPlan.md](ImplementationPlan.md).

**Name collision.** They ship under the identical project name and the identical release
tag `PRISM_GENAI_HACKATHON_Y2026`. The tag is prescribed by the organisers, so it cannot be
changed; the project name can. Tracked as `OQ-04` in
[OpenQuestions.md](OpenQuestions.md#oq-04), owner Harshdeep, deadline 2026-09-20.

---

## 10. Assumptions

| # | Assumption | If false |
|---|---|---|
| A-1 | CoIR `AppsRetrieval` is reachable through the MTEB/HuggingFace path and is BEIR-shaped (`_id`, `text`, `title`, `language`, `meta_information`). | Vendor a cached copy under `data/` early; `scripts/run_eval.py` supports a local corpus path. |
| A-2 | A suitable multi-version JavaScript repository can be sourced for the P1/Bonus demo. | Tracked as `OQ-07` in [OpenQuestions.md](OpenQuestions.md); fallback is a synthesised version history over a chosen OSS JS repo's real tags. |
| A-3 | The reference box (8-core CPU, 16 GB RAM) approximates the evaluator's machine. | `configs/fast.yaml` fallback profile targets a 4-core / 8 GB floor. |
| A-4 | `Qwen3-Embedding-0.6B` INT8 throughput is sufficient for a 10k-chunk cold index inside 12 min. | Tracked as `OQ-02`; declared fallback is `all-MiniLM-L6-v2`. |
| A-5 | The structural signal transfers value to the **live demo** corpus (JavaScript, multi-module). It is **not** assumed to transfer to the APPS eval corpus — see §2.2. | If it fails on the demo corpus too, the innovation claim collapses to the agent loop alone. Tracked as `OQ-01`; see [OpenQuestions.md](OpenQuestions.md#oq-01). |
| A-6 | The 5,000 `AppsRetrieval` train pairs are usable for weight and threshold tuning without any gradient training. | If the train split is unavailable or mis-shaped, fall back to a 500-query holdout carved from the dev portion of test and declare it in [Tracker.md](Tracker.md). Tracked as `OQ-05`. |

---

## 11. Out of Scope

Deliberate exclusions are enumerated with rationale in [NonGoals.md](NonGoals.md) as
`NG-01`–`NG-20`. The headline ones: no code generation, no answer synthesis, no
natural-language explanation of code, no language beyond JavaScript, no GPU path, no model
training, no auth or multi-tenancy, no IDE plugin, no hosted deployment.

Unresolved decisions are tracked in [OpenQuestions.md](OpenQuestions.md) as `OQ-##` and are
promoted to `ADR-###` entries in [Decisions.md](Decisions.md) when settled.
