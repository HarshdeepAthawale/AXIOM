# Test Plan

How Axiom is verified: what is unit tested, what is integration tested, what is only smoke tested, and what is measured by evaluation rather than asserted by a test.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-23
**Status:** Active — reconciled against the implementation

Related: [_CONTRACT.md](_CONTRACT.md) · [Schema.md](Schema.md) · [TechSpecifications.md](TechSpecifications.md) · [API.md](API.md) · [Decisions.md](Decisions.md) · [ImplementationPlan.md](ImplementationPlan.md) · [Tracker.md](Tracker.md) · [OpenQuestions.md](OpenQuestions.md) · [Rules.md](Rules.md)

---

## 0. What changed on 2026-09-23, and why

This plan was written before the code. The code now exists — `src/axiom/` plus `tests/`, 403 tests
collected, **397 passing and 6 failing** — and a walk of this document against it found that **at
least nine P0 cases could not pass against a spec-conformant implementation.** They asserted shapes the schema
rejects, columns the DDL does not define, an inequality that is arithmetically false, and an
exception class that does not exist in the error taxonomy. Those nine are corrected below, each with
a note naming what was wrong.

Three further structural gaps were closed:

1. **This document contained zero `FR-`/`NFR-` references.** A requirement with no test case is not
   real, and there was no way to check that claim. §2.2 is now a requirement → coverage matrix.
2. **`FR-10` — the structural signal, the requirement
   [PRD.md §8](PRD.md#8-how-this-maps-to-the-jury-scoring-rubric) credits with the Innovation
   score — had no test case at all.** Category L (`TC-091`–`TC-098`) covers its five query forms,
   the K clamp, the empty-list contract and the degradation ladder.
3. **`FR-22` — the shape of `appsretrieval_results.json`, the single artifact attached to the
   release — had no test case.** Category M (`TC-099`–`TC-105`) covers it, along with
   `NFR-06`, `NFR-10`, `FR-14` and `FR-15`, none of which had one either.

**Where a real test already exists, this plan cites it by file and test name** rather than inventing
a parallel numbering. A `TC-###` whose `Implemented by` cell is empty is work that has not been
done; a `TC-###` with a citation is a claim you can check in thirty seconds.

---

## 1. Testing Philosophy

Axiom is a ranking system. Ranking quality is a **measurement**, not an assertion. The single most
common way retrieval projects waste a sprint is writing brittle tests like
`assert results[0].chunk.metadata.symbol == "normalizeInput"`. That test fails the first time
someone retunes an RRF weight, and its failure carries no information about whether the system got
better or worse.

So we split responsibilities hard:

| Concern | Verified by | Never verified by |
|---|---|---|
| Schema shape, ID derivation, serialisation round-trips | `pytest` unit tests, strict | eyeballing JSON |
| Arithmetic (RRF, NDCG, MRR, stability bonus) | `pytest` unit tests against hand-computed constants | a golden file nobody re-derived |
| Invariants (sorted order, rank contiguity, dedup, determinism) | `pytest` property/invariant tests | manual demo |
| Wiring across module boundaries | integration tests on a fixture repo | unit tests with mocks of our own code |
| Whole-pipeline liveness, CLI, API, UI | smoke tests (does it run, is output well-formed) | assertions on which chunk ranked first |
| **Retrieval accuracy** | **MTEB run on CoIR `AppsRetrieval`, NDCG@10 / MRR@10 / Recall@100** | **any `pytest` assertion** |
| Latency and memory budgets | `scripts/bench_latency.py` benchmark harness | a `time.time()` assert inside a unit test |

### 1.1 The pyramid

```
                      ┌──────────────────────────────┐
                      │  MTEB eval (gates)           │  quality, not pass/fail
                      ├──────────────────────────────┤
                      │  Benchmarks (bench_latency)  │  budgets, threshold-gated
                      ├──────────────────────────────┤
                      │  Smoke: CLI / API / UI       │  liveness only
                      ├──────────────────────────────┤
                      │  Integration: fixture repo   │  test_pipeline.py, test_scripts.py
                      ├──────────────────────────────┤
                      │  Unit: schema, hashing,      │  the bulk of the 400
                      │  chunking, fusion, structural│
                      └──────────────────────────────┘
```

As built, the suite is 403 tests across eight modules: `test_schema.py` (71), `test_agent.py` (46),
`test_pipeline.py` (43), `test_scripts.py` (42), `test_fusion.py` (41), `test_structural.py` (39),
`test_hashing.py` (34), `test_chunking.py` (33). **Six currently fail**, listed in §3.14.

### 1.2 Rules of engagement

1. **No unit test asserts a document's identity or position** unless the input is a synthetic
   fixture built specifically to make that ordering arithmetically forced (see TC-049..TC-052).
2. **No test downloads a model.** Real model weights appear only in `slow`-marked tests and in the
   benchmark/eval harnesses. Unit tests use `FakeEmbedder` (deterministic hash-seeded vectors,
   configurable dim) and `FakeCrossEncoder` (score = fixed lookup table).
3. **No test needs network.** `tests/conftest.py` sets `HF_HUB_OFFLINE=1`,
   `AXIOM_LLM_ENABLED=false`, and `TRANSFORMERS_OFFLINE=1` for the default run.
4. **Every test is full-suite-safe**: `tmp_path`-scoped `.axiom/` directories, no shared global
   state, no ordering dependency, no reliance on wall-clock except the explicit budget tests.
5. **A test that pins a log string, a docstring, or a message template gets deleted.** We assert
   observable contract only: return values, file formats, exit codes, HTTP status and body shape.
6. **A test may not assert something the implementation does not do.** When this plan and the code
   disagree, one of them is wrong and the disagreement is resolved in writing — in §0's change log
   or in the case's own note — never by quietly loosening the assertion.

### 1.3 Fixtures

| Fixture | Location | Contents | Exists |
|---|---|---|---|
| `repo_v1` | `tests/fixtures/repo_v1/` | JS files covering: exports, a class with methods, an oversized generated function, a syntactically broken file, an empty file, a constants-only file, a circular import pair, duplicated content at two paths, 200-deep nesting, i18n text | yes |
| `repo_v2` | `tests/fixtures/repo_v2/` | `repo_v1` + modified, added, deleted, pure-rename and rename-with-edit files | yes |
| `fixture_git_repo` | built by `tests/conftest.py` | real `git init`, two commits, so `git diff --name-status` is exercised for real | yes — `tests/test_scripts.py::TestGitFixture` |
| `bench_queries` | `tests/fixtures/bench_queries.txt` | **100** queries, fixed order — the single latency fixture | yes |
| `queries` | `tests/fixtures/queries.txt` | natural-language queries for the identifier-extraction stopword check | yes |
| `FakeEmbedder` | `tests/fakes.py` | `blake2b(text)` → deterministic unit vector, dim configurable | yes |
| `FakeCrossEncoder` | `tests/fakes.py` | scores from an explicit lookup table | yes |
| `FakeLLM` | `tests/fakes.py` | scripted responses; raises if called when `AXIOM_LLM_ENABLED=false` | yes |
| `gen_v2` | `tests/fixtures/gen_v2.py` | synthetic 50-changed-file diff generator — the **only** source for the `NFR-02` measurement | **no — `T-163`** |

---

## 2. Test Categories

Categories mirror the package layout in [_CONTRACT.md](_CONTRACT.md) §3. The `Owner` column is
regenerated from [PRD.md §5](PRD.md#5-functional-requirements)'s `Owner` column, which is
authoritative.

| Cat | Area | Module(s) under test | Requirements | Owner |
|---|---|---|---|---|
| A | Query preprocessing: classification, identifier extraction, expansion, decomposition | `agent/classifier.py`, `agent/planner.py`, `agent/synonyms.py` | `FR-01`–`FR-03` | Harshdeep |
| B | Chunking and snippet preprocessing | `chunking/` | `FR-04` | Anish |
| C | Indexing: dense/sparse/structural builders, manifest, blob store | `indexing/` | `FR-05`, `FR-07`, `FR-09`, `FR-17`, `FR-19` | Prabinder |
| D | Dense retrieval | `retrieval/dense.py` | `FR-06` | Prabinder |
| E | Sparse retrieval | `retrieval/sparse.py`, `retrieval/tokenizer.py` | `FR-08` | Prabinder |
| F | Fusion (weighted RRF) | `retrieval/fusion.py` | `FR-11` | Prabinder |
| G | Reranking | `rerank/cross_encoder.py` | `FR-12` | Harshdeep |
| H | End-to-end pipeline, CLI, API, UI | `pipeline.py`, `cli.py`, `api/`, `ui/` | `FR-14`, `FR-23` (Prabinder); `FR-24`, `FR-25` (Harshdeep) | split — see note |
| I | Versioning and incremental reindex | `versioning/gitdiff.py`, `versioning/incremental.py` | `FR-16`, `FR-18`, `FR-20` | Parth |
| J | Evolutionary retrieval | `versioning/evolutionary.py` | `FR-21` | Parth |
| K | Agent loop | `agent/loop.py`, `agent/evaluator.py`, `agent/llm.py` | `FR-13` | Harshdeep |
| **L** | **Structural retrieval** | `retrieval/structural.py` | `FR-10` | Anish |
| **M** | **Eval harness and submission artifact** | `eval/`, `scripts/run_eval.py` | `FR-22`, `NFR-06`, `NFR-10` | Parth |

Category H is split by requirement, not by module: `FR-23` (the CLI) is Prabinder's per PRD §5,
because every milestone-gate command is a CLI invocation; `FR-24`/`FR-25` (API and UI) are
Harshdeep's.

Priorities: **P0** blocks the merge and blocks submission. **P1** blocks submission but not a
feature branch merge. **P2** is nice-to-have; may be deferred past 27 Sep.

### 2.1 CLI surface, as built

Several cases below invoke the CLI, and three earlier revisions of this plan invoked subcommands
that do not exist. The surface as implemented in `src/axiom/cli.py` is:

| Subcommand | In `FR-23`? | Notes |
|---|---|---|
| `index`, `reindex`, `query`, `classify`, `versions`, `families`, `eval`, `serve`, `ui` | yes | the nine `FR-23` names |
| `version` | no | **hidden alias** for `versions`, registered with `hidden=True` so it does not appear in `--help` |
| `gc` | **no** | a tenth command no requirement defines; `TC-080` depends on it. Either fold it into `FR-23` or drop that dependency — raised in [Tracker.md §7](Tracker.md#7-open-defects-this-board-raises-against-src-and-the-repo) |

There is **no `axiom search`**. Any document invoking it is stale; the command is `axiom query`.
Exit codes are `0` ok, `1` internal, `2` usage/validation, `3` index-not-found.

### 2.2 Requirement → coverage matrix

Every `FR-##`/`NFR-##` in [PRD.md](PRD.md), with its covering cases and whether a real test exists
today. "measured, not tested" means the requirement is verified by a benchmark or an eval run, per
§1's division of labour, and an assertion would be the wrong tool.

| Req | Covering cases | Implemented by | Gap |
|---|---|---|---|
| `FR-01` classify | TC-001..TC-006 | `tests/test_agent.py::TestClassification` | — |
| `FR-02` identifiers + expansion | TC-007, TC-008 | `tests/test_agent.py::TestPlanner::test_expansion_terms_are_appended_for_a_semantic_query` | — |
| `FR-03` ≤ 3 sub-queries | TC-011 | `tests/test_agent.py::TestPlanner::test_sub_queries_stay_within_the_fr03_cap` | — |
| `FR-04` chunking | TC-012..TC-024 | `tests/test_chunking.py` (33 tests) | — |
| `FR-05` dense index | TC-025..TC-029 | `tests/test_pipeline.py::TestIndexBuild` | TC-028's 50k threshold is untested |
| `FR-06` dense retrieval K=100 | TC-035..TC-041 | partial | TC-035..TC-039 have no dedicated test |
| `FR-07` sparse index | TC-025, TC-048 | `tests/test_pipeline.py::TestIndexBuild` | TC-048 round-trip untested |
| `FR-08` sparse retrieval K=100 | TC-042..TC-047 | `tests/test_structural.py::TestIdentifierHelpers::test_tc047_*` | TC-042..TC-046 untested |
| `FR-09` structural index | TC-032 | `tests/test_pipeline.py::TestIndexBuild`, `tests/test_structural.py` | — |
| **`FR-10` structural retrieval** | **TC-091..TC-098** | `tests/test_structural.py` (39 tests) | — |
| `FR-11` weighted RRF, N=25 | TC-049..TC-058 | `tests/test_fusion.py` (41 tests) | — |
| `FR-12` rerank top-25 → 10 | TC-059..TC-064 | — | **no rerank test module exists** |
| `FR-13` bounded agent loop | TC-086..TC-090 | `tests/test_agent.py::TestLoop`, `::TestSufficiency`, `::TestRefinement` | — |
| `FR-14` result formatting | TC-065, TC-104 | `tests/test_pipeline.py::TestQuery::test_every_result_explains_itself` | — |
| `FR-15` optimisation hint | TC-105 | — | B-priority; rule table exists in `pipeline.optimization_hint` |
| `FR-16` version stamping | TC-073, TC-079 | `tests/test_pipeline.py::TestVersions::test_the_manifest_chain_records_the_parent` | — |
| `FR-17` registry + manifest | TC-029, TC-079 | `tests/test_schema.py::TestVersionManifest`, `tests/test_pipeline.py::TestIndexBuild::test_the_version_is_registered_and_made_active` | — |
| `FR-18` incremental reindex | TC-073..TC-078 | `tests/test_pipeline.py::TestReindexProducesAQueryableVersion`, `tests/test_scripts.py::TestGitFixture` | TC-076 byte-equivalence missing (`T-162`) |
| `FR-19` blob reuse | TC-031, TC-075 | `tests/test_pipeline.py::TestVersions::test_a_pure_rename_between_versions_reuses_the_blob` | — |
| `FR-20` version-scoped query | TC-040 | `tests/test_pipeline.py::TestVersions::test_version_scoping_is_structural_not_a_filter` | — |
| `FR-21` evolutionary retrieval | TC-081..TC-085 | `tests/test_schema.py::TestSnippetFamily` only | **no test module for `versioning/evolutionary.py`** |
| **`FR-22` MTEB adapter + results JSON** | **TC-099..TC-101** | `tests/test_scripts.py` (42 tests) | results-JSON shape assertion missing |
| `FR-23` CLI | TC-069 | — | **no CLI test module exists** |
| `FR-24` FastAPI | TC-070, TC-071 | — | **no API test module exists** |
| `FR-25` Streamlit UI | TC-072 | — | import-only smoke missing |
| `FR-26` train-split tuning | — | — | process requirement; enforced by the reportability gate (TC-100) and the §6.4 log |
| `NFR-01` cold index ≤ 12 min | PB-01 | — | measured, not tested |
| `NFR-02` reindex ≤ 45 s | PB-02 | — | measured, not tested; fixture `gen_v2.py` missing |
| `NFR-03` p50 ≤ 900 ms | PB-03 | `tests/test_scripts.py::TestBudgetGate` (gate logic only) | measured, not tested |
| `NFR-04` p95 ≤ 5 s | PB-04, TC-086 | `tests/test_agent.py::TestLoop` (bound, not wall-clock) | measured, not tested |
| `NFR-05` peak RSS ≤ 4 GB | PB-05 | — | measured, not tested |
| **`NFR-06` CPU-only** | **TC-102** | — | was verified nowhere at all |
| `NFR-07` degradation | TC-062, TC-067, TC-068 | `tests/test_pipeline.py::TestQueryDegradation`, `tests/test_chunking.py::TestDegradation`, `tests/test_structural.py::TestStoreDegradation` | — |
| `NFR-08` determinism | TC-052, TC-066 | `tests/test_pipeline.py::TestQuery::test_tc066_*`, `tests/test_fusion.py::TestDeterminism` | — |
| `NFR-09` reproducibility | — | — | `T-017` fresh-clone rehearsal; no `uv.lock` exists yet |
| **`NFR-10` observability / `timings`** | **TC-103** | `tests/test_pipeline.py::TestQuery::test_the_response_serialises_for_the_json_mode` | — |
| `NFR-11` code health | CI stages 1–3 | `ruff` + `mypy --strict` + 400 tests green locally | CI workflow does not exist (`T-002`) |
| `NFR-12` storage footprint | PB-06 | — | measured, not tested |

Seven rows above say a test module does not exist. That is the honest state of coverage on
2026-09-23 and it is why §9's project gate is not yet met.

---

## 3. Test Cases

### 3.1 Category A — Query preprocessing (TC-001 .. TC-011)

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-001 | Semantic query classifies as `SEMANTIC` | `AXIOM_LLM_ENABLED=false` | `classify("How is the input preprocessed before the main function?")` | `QueryPlan.query_type == QueryType.SEMANTIC` | P0 | `test_agent.py::TestClassification` |
| TC-002 | Structural query classifies as `STRUCTURAL` | as above | `classify("Which files call tool XYZ before tool ABC?")` | `query_type == STRUCTURAL`; `extracted_identifiers` contains `XYZ`, `ABC` | P0 | `test_agent.py::TestClassification` |
| TC-003 | Usage query classifies as `USAGE` | as above | `classify("Where is the Bluetooth-settings deeplink used?")` | `query_type == USAGE` | P0 | `test_agent.py::TestClassification` |
| TC-004 | Multi-intent query classifies as `HYBRID` | as above | `classify("How does the auth module validate tokens before calling the API?")` | `query_type == HYBRID` | P1 | `test_agent.py::TestClassification` |
| TC-005 | Strategy weights match the locked table exactly | none | For each `QueryType`, build a plan; compare `strategy_weights` to `config.DEFAULT_STRATEGY_WEIGHTS` | SEMANTIC `{.60,.30,.10}`, STRUCTURAL `{.20,.20,.60}`, USAGE `{.25,.55,.20}`, HYBRID `{.34,.33,.33}` | P0 | `test_agent.py::TestPlanner` |
| TC-006 | Weights always normalise to 1.0 | none | For all four types and every override in `configs/*.yaml`, sum the weights | `abs(sum - 1.0) <= 1e-9`; the `QueryPlan` model validator enforces `1e-6` | P0 | `test_schema.py::TestQueryPlan` |
| TC-007 | Identifier extraction handles camelCase, snake_case, dotted, and quoted forms | none | `extract_identifiers("call handleDeeplink, parse_input, utils.normalize and 'MAX_RETRY'")` | All four returned, order preserved, no duplicates | P0 | `test_agent.py::TestPlanner` |
| TC-008 | Identifier extraction emits no English stopwords | none | Run the extractor over `tests/fixtures/queries.txt` | No returned identifier is in the stopword list; none is single-character unless quoted | P1 | `test_agent.py::TestPlanner` |
| TC-009 | **Empty and whitespace-only queries degrade; they never raise** | none | `normalise_query("")`, `normalise_query("   \t\n ")`, `build_plan("")`, then the CLI and the API boundary | Both normalise to `""`. `build_plan("")` returns a **well-formed plan** whose weights sum to 1.0. The pipeline returns an empty-but-valid result set with `stop_reason == "empty_query"`. The **boundaries** reject: CLI exits 2 before touching the index (`cli.query_command`), API returns 422 from `QueryRequest`'s `min_length=1`. No exception escapes any layer | P0 | `test_agent.py::TestPlanner::test_tc009_*`, `test_pipeline.py::TestQueryDegradation::test_an_empty_query_returns_a_warning_not_a_traceback` |
| TC-010 | 10,000-character query is truncated, not fatal | none | `build_plan("x" * 10000)` then run full retrieval | Query truncated to `settings.max_query_chars` (2048); pipeline returns a well-formed result list; no exception | P1 | `test_agent.py::test_tc010_*`, `test_pipeline.py::TestQueryDegradation::test_tc010_*` |
| TC-011 | Unicode/emoji survive, and `FR-03`'s sub-query cap holds | `FakeEmbedder` | `build_plan("où est le déeplink 🔵 Bluetooth ?")`; separately, plan a compound query | No `UnicodeError`; `len(plan.sub_queries) <= 3` (`planner.MAX_SUB_QUERIES`), matching `FR-03` | P1 | `test_agent.py::test_tc011_*`, `::TestPlanner::test_sub_queries_stay_within_the_fr03_cap` |

> **TC-009 was rewritten.** It previously required `plan("")` to raise
> `axiom.core.errors.EmptyQueryError`. No such class exists — the taxonomy is `AxiomError`,
> `AxiomContractError`, `IndexNotFoundError`, `DegradationExhaustedError` — and
> [Rules.md](Rules.md) Cardinal Rule 3 forbids raising on user input in the first place. The
> implementation resolves this the way API.md already describes: **degrade in the pipeline, reject
> at the boundary.** Both halves are now asserted.

### 3.2 Category B — Chunking (TC-012 .. TC-024)

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-012 | Function boundaries become chunk boundaries | tree-sitter installed | Chunk a fixture file with top-level functions | One chunk per function, `kind == FUNCTION`; `location.start_line`/`end_line` bracket the declaration | P0 | `test_chunking.py::TestAstTaxonomy`, `::TestSpanInvariants` |
| TC-013 | Class methods get `kind == METHOD` and a `parent_symbol` | as above | Chunk `repo_v1/src/agents/bluetooth.js` | Method chunks carry `parent_symbol`; one `CLASS` chunk for the shell | P0 | `test_chunking.py::TestAstTaxonomy` |
| TC-014 | `chunk_id` is stable across repeated chunking of identical input | none | Chunk the same file twice; compare `chunk_id` sets | Sets identical; ids are 32-hex blake2b-128 | P0 | `test_hashing.py::TestBlake2b128` |
| TC-015 | **`chunk_id` round-trips unchanged through every stage** | fixture repo indexed | Capture the id set from the chunker, then from `chunks.jsonl`, `dense.idmap.json`, the bm25s doc-id map, `structural.sqlite`, each `ScoredChunk`, each `FusedResult`, each reranked result, and the final `RetrievalResult.chunk.chunk_id` | All sets equal for the retrieved subset; no stage rewrites, truncates, lowercases, or re-derives an id | P0 | `test_pipeline.py::TestQuery::test_tc015_chunk_ids_round_trip_unchanged_through_every_stage` |
| TC-016 | **`content_hash` is invariant under location change** | none | Hash a function; move it to another file and another line offset; hash again | `content_hash` identical all three times; `chunk_id` differs all three times | P0 | `test_hashing.py::TestDigestAsymmetry`, `::TestDedupMechanisms` |
| TC-017 | **`content_hash` is sensitive to real content change** | none | Mutate by one token, one rename, one added statement | `content_hash` differs for each; no collisions | P0 | `test_hashing.py::TestDigestAsymmetry` |
| TC-018 | `content_hash` normalisation ignores only declared-insignificant whitespace | none | Reindent, CRLF→LF, add a trailing newline; then delete a blank line inside a template literal | Unchanged for the first three; changed for the fourth | P0 | `test_hashing.py::TestNormaliseForHash` |
| TC-019 | Oversized function splits at statement boundaries with 1-statement overlap | oversize fixture | Chunk it | Multiple chunks, each within the token ceiling except the last; consecutive chunks share exactly one statement; byte ranges cover the function with no gaps | P0 | `test_chunking.py::TestOversizeSplit` |
| TC-020 | Chunks under 16 tokens merge into the parent | none | Chunk a file with a 2-line arrow function inside a class | No emitted chunk is under `settings.chunk_min_tokens`; the tiny function's text is inside its parent's `text` | P1 | `test_chunking.py::TestTinyChunkMerge` |
| TC-021 | Syntactically broken file falls back to the regex identifier splitter | `repo_v1/src/broken/syntax_error.js` | Chunk it | No exception; ≥1 chunk; `kind == BLOCK`; a `WARNING` records the parse failure; `metadata.calls` populated from the fallback | P0 | `test_chunking.py::TestDegradation`, `::TestLowerRungs` |
| TC-022 | Empty file and constants-only file produce zero or one MODULE chunk, never a crash | none | Chunk `repo_v1/src/empty.js` and `repo_v1/src/constants.js` | Empty file → 0 chunks. Constants file → exactly 1 `MODULE` chunk | P0 | `test_chunking.py::TestRepoWalk` |
| TC-023 | Minified bundle is chunked without pathological blow-up | minified fixture | Chunk with a timeout | Completes well inside the timeout; chunk count ≤ `settings.max_chunks_per_file`; no chunk has `end_byte <= start_byte` | P1 | `test_chunking.py::TestSpanInvariants` |
| TC-024 | Deeply nested closures do not exceed recursion limits | `repo_v1/src/deep/` | Chunk it | No `RecursionError`; traversal is iterative; nesting beyond `settings.max_ast_depth` becomes a single `BLOCK` chunk with a `WARNING` | P1 | `test_chunking.py::TestLowerRungs` |

### 3.3 Category C — Indexing (TC-025 .. TC-034)

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-025 | On-disk layout matches the contract exactly | `tmp_path` | `axiom index tests/fixtures/repo_v1 --version v1` | `.axiom/registry.json`, `.axiom/blobs/`, `.axiom/index/v1/{manifest.json,chunks.jsonl,dense.faiss,dense.idmap.json,sparse.bm25s/,structural.sqlite}` all exist; no extra top-level entries | P0 | `test_pipeline.py::TestIndexBuild::test_a_cold_build_produces_the_contracted_layout` |
| TC-026 | **`dense.idmap.json` is a total bijection with the FAISS row space** | index built | Load `dense.faiss` and the idmap | `idmap["count"] == index.ntotal == len(idmap["rows"])`; `rows` is a **positional array**, entries unique and all 32-hex, each present in `chunks.jsonl`; `idmap["embedding_model"]`, `["embedding_dim"]`, `["index_kind"]` equal the manifest's; `idmap["backend"]` names the backend that actually ran | P0 | — |
| TC-027 | Embeddings are L2-normalised | `FakeEmbedder` | Reconstruct vectors from the index | `abs(norm - 1.0) <= 1e-5` for all | P0 | — |
| TC-028 | Index kind is chosen by the `faiss_ivf_threshold` | none | Build just under and just at the threshold | `manifest.index_kind == "flat_ip"` then `"ivf_pq"`; the FAISS object type matches. Threshold read from `settings.faiss_ivf_threshold`, not hard-coded | P0 | — |
| TC-029 | Manifest records model identity and dimension | `FakeEmbedder` | Read `manifest.json` | `embedding_model` and `embedding_dim` match the active embedder; `chunk_count == len(chunks.jsonl)`; `created_at` is ISO-8601 | P0 | `test_pipeline.py::TestIndexBuild::test_the_manifest_records_the_rung_that_actually_ran` |
| TC-030 | **Dimension mismatch between index and query embedder is a hard error** | index built at one dim | Query with an embedder of a different dim | `axiom.core.errors.AxiomContractError` raised before any FAISS call; the message names both dims, both model ids and the index directory. A model-identity mismatch **at equal dim** degrades instead: empty dense result, signal marked absent | P0 | — |
| TC-031 | Blob store is content-addressed and deduped | duplicated content at two paths | Index `repo_v1` | `.axiom/blobs/` holds one file per distinct `content_hash`; the duplicated pair maps to one blob; embed-call count equals distinct content hashes | P0 | `test_pipeline.py::TestVersions::test_a_pure_rename_between_versions_reuses_the_blob` |
| TC-032 | **Structural SQLite schema and referential integrity** | index built | Query `structural.sqlite` | Tables `symbols`, `calls`, `imports`, `exports` exist. **Every `calls.caller_id` resolves to a `symbols.symbol_id`**, and every `symbols.chunk_id` resolves in `chunks.jsonl`. `calls.callee_id` is **nullable by design** — an unresolved callee still carries `callee_name` — so it is asserted as "NULL or resolving", never as "present". `PRAGMA foreign_key_check` returns empty | P0 | `test_structural.py::TestGraphQueries` (indirectly) |
| TC-033 | Circular imports are indexed without infinite traversal | `repo_v1/src/circular/{a,b}.js` | Build the import graph | Completes; both edges present; each node visited once | P1 | `test_chunking.py::TestRepoWalk` |
| TC-034 | Index build is idempotent and atomic | index built once | Re-run on the same tree with the same version id; kill a build midway and re-run | Re-run produces an identical `manifest.json` except `created_at`; an interrupted build leaves no partial version in `registry.json` (writes go to a `.tmp` sibling then `os.replace`) | P0 | `test_pipeline.py::TestIndexBuild::test_the_build_is_deterministic` |

> **TC-026 was rewritten.** It asserted `index.ntotal == len(idmap)` with "keys `0..ntotal-1`",
> i.e. an object keyed by stringified integers. [Schema.md §14.4](Schema.md) explicitly rejects that
> shape in favour of a positional `rows` array, and `indexing/dense.py::_write_idmap` implements the
> array. The old assertion would have failed on a conformant index.
>
> **TC-032 was rewritten.** It asserted a `calls.caller_chunk_id` column. The DDL in
> `indexing/structural.py` defines `caller_id INTEGER REFERENCES symbols(symbol_id)` and has no such
> column; the old query raises `OperationalError`.
>
> **TC-030 was rewritten.** It named `IndexDimensionMismatch`, which is not in the taxonomy. The
> implementation raises `AxiomContractError` from `retrieval/dense.py`, which is exactly the case
> [Rules.md §3](Rules.md) category 1 calls "an embedding of the wrong dimension".

### 3.4 Category D — Dense retrieval (TC-035 .. TC-041)

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-035 | Returns exactly K candidates with 1-indexed contiguous ranks | index of ≥ K chunks | `dense_search(q, k=100)` | `len == 100`; ranks `1..100`; all `signal == DENSE` | P0 | — |
| TC-036 | Scores are non-increasing | as above | inspect scores | Descending-sort invariant asserted by the shared `assert_ranked_list()` helper in `tests/helpers.py` | P0 | — |
| TC-037 | K larger than the corpus returns the whole corpus, not padding | index of 12 chunks | `dense_search(q, k=100)` | `len == 12`; ranks `1..12`; no sentinel ids | P0 | — |
| TC-038 | Exact-duplicate query of an indexed chunk's text retrieves that chunk at rank 1 | `FakeEmbedder` | Embed chunk text verbatim as the query | Rank-1 `chunk_id` equals the source chunk | P0 | — |
| TC-039 | Inner-product search is equivalent to cosine for normalised vectors | none | Compare FAISS scores to `numpy` cosine on the same pairs | `max abs diff <= 1e-5` | P1 | — |
| TC-040 | Version filter restricts results to one version | two versions indexed | Query with `--version v1` | Every returned chunk's `metadata.version_id == "v1"`; scoping is structural (a different index directory), not a post-filter | P0 | `test_pipeline.py::TestVersions::test_version_scoping_is_structural_not_a_filter` |
| TC-041 | IVF-PQ `nprobe` is read from config and affects recall monotonically | large synthetic index | Search with increasing `nprobe`; compare recall against flat ground truth | Recall non-decreasing in `nprobe`; value read from `settings.faiss_nprobe` | P2 | — |

### 3.5 Category E — Sparse retrieval (TC-042 .. TC-048)

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-042 | Code tokenizer splits camelCase, snake_case, and dots while keeping the whole identifier | none | `tokenize("utils.handleDeeplink_v2")` | Contains the split parts **and** the intact identifier; lowercased | P0 | — |
| TC-043 | Exact identifier query retrieves the defining chunk | fixture indexed | `sparse_search("handleDeeplink", k=100)` | The defining chunk appears near the top | P0 | — |
| TC-044 | Ranked list obeys the descending-sort and contiguous-rank invariants | as above | `assert_ranked_list(results)` | Passes; `signal == SPARSE` on every element | P0 | — |
| TC-045 | Out-of-vocabulary query returns an empty list, not an exception | as above | `sparse_search("zzqqxx_not_a_token")` | `[]` returned; fusion handles the empty signal (TC-055) | P0 | `test_pipeline.py::TestQueryDegradation::test_an_out_of_vocabulary_query_yields_an_honest_empty_result` |
| TC-046 | Stopword-only query degrades gracefully | as above | `sparse_search("the of and is to")` | `[]` or an all-zero-score list; the pipeline still returns results via the dense signal | P1 | — |
| TC-047 | **Regex tokenizer is not vulnerable to ReDoS** | none | Feed adversarial inputs (`"a"*50000`, `"("*5000`, `("ab"*10000)+"!"`) with a timeout | Every input completes in linear time; pattern constants carry no nested unbounded quantifier | P0 | `test_structural.py::TestIdentifierHelpers::test_tc047_mine_identifiers_is_linear_in_input_length`, `::test_tc047_long_runs_are_bounded_rather_than_scanned`, `::test_tc047_prose_and_symbols_are_unchanged_by_the_prefilter` |
| TC-048 | bm25s index survives a save/load round-trip | index built | Save, load in a fresh process, re-run queries | Identical `(chunk_id, score)` sequences to 1e-6 | P0 | — |

### 3.6 Category F — Fusion (TC-049 .. TC-058)

The highest-value unit-test cluster in the project: pure arithmetic with a published reference, so
it can be pinned exactly.

**The unweighted RRF derivation TC-049 rests on**, inlined here so the case does not depend on
another document. With all weights 1.0 and `k = 60`, a document at rank `r` in a list contributes
`1/(60 + r)`. A document at rank 5 in **three** lists scores `3/65 = 0.046154`; one at rank 1 in
**two** lists scores `2/61 = 0.032787`. Broad agreement beats one strong signal — that is the
property weighted RRF is chosen for, and the reason fusion never reads a raw score.

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-049 | **RRF arithmetic matches the hand-computed worked example** | all weights 1.0 (unweighted illustration; production fusion is weighted per `QueryType`) | Doc `A` at rank 5 in all three lists; `B` at rank 1 in two, absent from the third; `C` at rank 1 in one and rank 10 in another. Fuse | `score(A) ≈ 3/65 = 0.046154`; `score(B) ≈ 2/61 = 0.032787`; `score(C) ≈ 1/61 + 1/70 = 0.030679`. Order `[A, B, C]` — **A beats B** | P0 | `test_fusion.py::TestWorkedExample::test_tc049_rrf_arithmetic_matches_the_hand_computed_example` |
| TC-050 | Weighted RRF equals the contract formula term by term | default SEMANTIC weights | Fuse the TC-049 lists with `{dense .60, sparse .30, struct .10}`; compute `Σ w_i/(60+rank_i)` by hand in the test | Every `rrf_score` matches to 1e-9; `k` read from `settings.rrf_k`, default 60 | P0 | `test_fusion.py::TestWorkedExample::test_tc050_*`, `::test_rrf_k_is_read_from_settings_not_hard_coded` |
| TC-051 | **Rank-space fusion is invariant to monotonic score rescaling** | none | Fuse; rescale each list's raw scores by a different strictly-increasing map preserving order; re-fuse | `rrf_score` byte-identical between runs and order identical. Proves fusion never reads raw scores | P0 | `test_fusion.py::test_tc051_fusion_is_invariant_to_monotonic_score_rescaling` |
| TC-052 | Fusion output order is fully determined and ties break deterministically | none | Fuse lists engineered to tie exactly on `rrf_score` | Tie broken by ascending `chunk_id`; two runs agree; reversing the input-list order gives the same output | P0 | `test_fusion.py::TestDeterminism::test_tc052_exact_ties_break_by_ascending_chunk_id`, `::test_tc052_reversed_input_order_gives_the_same_output` |
| TC-053 | `contributions` records the true per-signal rank | none | Fuse the TC-049 lists | `contributions[DENSE] == 5` etc.; a signal the doc did not appear in is **absent from the dict**, not `0` or `-1` | P0 | `test_fusion.py::TestWorkedExample::test_tc053_contributions_record_the_true_per_signal_rank` |
| TC-054 | **`dominant_signal` is the signal with the largest weighted term** | SEMANTIC weights, `rrf_k = 60` | Doc at rank 1 in sparse, rank 40 in dense | `dominant_signal == DENSE`, because `.60/(60+40) = 0.006000 > .30/(60+1) = 0.004918`. The test asserts the **comparison**, not a hard-coded signal, and separately asserts the crossover: at dense rank ≥ 62 the sparse term wins | P0 | `test_fusion.py::TestDominantSignal::test_tc054_dominant_signal_is_the_largest_weighted_term`, `::test_dominant_signal_flips_past_the_crossover_rank` |
| TC-055 | **Empty-signal weight renormalisation** | none | Fuse with the structural list empty under SEMANTIC weights | Remaining weights renormalise to `{dense .6/.9, sparse .3/.9}` summing to 1.0; scores equal a directly hand-computed two-signal fusion; a doc's score does not shrink merely because a signal was unavailable; the logged plan carries the renormalised vector | P0 | `test_fusion.py::TestEmptySignals::test_tc055_structural_empty_renormalises_dense_and_sparse`, `::test_tc055_the_logged_plan_carries_the_renormalised_vector` |
| TC-056 | All signals empty yields an empty fused list, not a crash | none | Fuse three empty lists | Returns `[]`; the pipeline converts this into an empty-but-valid `list[RetrievalResult]` and a degradation note naming "no candidates" | P0 | `test_fusion.py::TestEmptySignals::test_tc056_all_signals_empty_yields_an_empty_list` |
| TC-057 | Fused list obeys the ranked-list invariants and top-N truncation | 300 candidates across 3 lists | Fuse, then truncate | `assert_ranked_list` passes; length `== min(fusion_top_n, unique_candidates)`; no duplicate `chunk_id` | P0 | `test_fusion.py::TestInvariantsAndWidths::test_tc057_300_candidates_truncate_to_25_with_no_duplicates` |
| TC-058 | Candidate widths come from config, not from a literal | fixture indexed | Run with tracing on | Dense K=100, sparse K=100, structural K=50, fusion N=25, rerank→10; every value read from `Settings`, asserted against `_CONTRACT.md §5`. Note `configs/eval.yaml` deliberately overrides `fusion_top_n` to 5 | P0 | `test_fusion.py::TestInvariantsAndWidths::test_tc058_candidate_widths_come_from_the_contract` |

> **TC-054's expected value was corrected.** The case asserted `dominant_signal == SPARSE` "because
> `.3/61 > .6/100`". That inequality is false: `0.004918 < 0.006000`. The row's own instruction —
> "assert the comparison, not a hard-coded signal" — was the right instinct and is what the
> implementation does; only the stated conclusion was wrong.

### 3.7 Category G — Reranking (TC-059 .. TC-064)

No test module exists for `rerank/cross_encoder.py`. Every row below is unimplemented.

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-059 | Reranker reorders by `rerank_score` and preserves the candidate set | `FakeCrossEncoder` | Rerank the fused candidates | Output is a permutation of the input before truncation; sorted descending by `rerank_score`; `rrf_score` preserved on each item | P0 | — |
| TC-060 | Reranker is applied to exactly `fusion_top_n` pairs and returns `top_k` | as above | Count cross-encoder calls | Call count `== settings.fusion_top_n`; output length `== settings.top_k_default`. **Both read from config** — `eval.yaml` sets `fusion_top_n: 5`, so a literal 25 here would fail on the eval profile | P0 | — |
| TC-061 | Batching does not change results | as above | Rerank with several batch sizes | Identical ordered `(chunk_id, rerank_score)` sequences | P0 | — |
| TC-062 | Reranker failure degrades to the RRF order | `FakeCrossEncoder` raising | Run the pipeline | Non-empty results in pure RRF order; the degradation is recorded once, not per pair; the response's `score_field` names which score the order came from | P0 | — |
| TC-063 | Long chunk text is truncated to the window, not rejected | 20k-character chunk | Rerank | No exception; truncation at `settings.rerank_max_chars` applied to the **document side only**, never to the query. Value is profile-dependent: 4096 on `demo.yaml`, 1024 on `eval.yaml` | P1 | — |
| TC-064 | Reranker model swap keeps the interface | `AXIOM_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`, marked `slow` | Rerank a candidate set | Same output type and invariants; scores finite; no dim or tokenizer assumption leaks. **This is not an exotic path** — it is `demo.yaml`'s primary reranker | P1 | — |

### 3.8 Category H — Pipeline, CLI, API, UI (TC-065 .. TC-072)

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-065 | Full pipeline returns well-formed `RetrievalResult` objects | fixture indexed, fakes | `axiom query "how is input preprocessed" --json` | Valid JSON; each item validates against `RetrievalResult`; `chunk.location.start_line <= end_line`; `signals` non-empty; `score` finite | P0 | `test_pipeline.py::TestQuery::test_every_archetype_returns_results_with_real_locations` |
| TC-066 | **Determinism: same input + config + seed produces identical output** | fixture indexed, `AXIOM_SEED` at its configured default | Run the same query twice in separate processes, `--json` | Payloads are equal **after removing `elapsed_ms` and the entire `timings` block**. Both are wall-clock measurements and neither can be stable; every other value — results, ranks, scores, plan, `passes_used`, `stop_reason`, `version_ids`, `degradations`, `warnings` — is a pure function of query, config and index and is compared exactly. `elapsed_ms` is separately asserted present and non-negative | P0 | `test_pipeline.py::TestQuery::test_tc066_two_runs_agree_on_everything_but_the_clock` |
| TC-067 | **Degradation with the LLM disabled produces a valid, non-empty result set** | `AXIOM_LLM_ENABLED=false`; `FakeLLM` asserts it is never called | Run the archetype queries | Every query returns results; `FakeLLM.call_count == 0`; classification came from the heuristic engine; **`QueryPlan.sub_queries == []`** and `QueryPlan.effective_queries == [original_query]` | P0 | `test_pipeline.py::TestQueryDegradation`, `test_agent.py::TestClassification` |
| TC-068 | Degradation cascade: every declared fallback is individually exercised | none | Force each of {embedder, reranker, LLM, tree-sitter} to fail, one at a time | A non-empty well-formed result in every case; exactly one `WARNING`/`ERROR` per failure; the manifest or response records which rung actually ran | P0 | `test_pipeline.py::TestQueryDegradation`, `test_pipeline.py::TestIndexBuild::test_the_degraded_rungs_are_visible_in_the_report` |
| TC-069 | **CLI surface contract** | installed package | `axiom --help` plus `--help` on each of the nine `FR-23` subcommands: `index`, `reindex`, `query`, `classify`, `versions`, `families`, `eval`, `serve`, `ui` | Exit 0 for each; the documented flags of [API.md](API.md)/[Setup.md](Setup.md) are present; an unknown flag exits 2 with a usage message. `axiom version` resolves as a hidden alias of `versions` but is **not** part of the asserted surface. `axiom search` does not exist and is not invoked | P0 | — |
| TC-070 | API contract: `POST /v1/query` | uvicorn on an ephemeral port, fixture index | Post `{"query":"...","top_k":10}` | 200; body has `results`, `query_plan`, `elapsed_ms`, `timings`; `len(results) <= 10`; validates against `api/models.py`. The unprefixed `/query` alias answers identically but is absent from `/openapi.json` | P0 | — |
| TC-071 | API input validation rejects malformed and hostile input | as above | Empty query; `top_k=0`; `top_k` far past the cap; `version="../../etc"`; an oversized body | 422 for each (413 for the oversized body); no stack trace in the body; **no filesystem access attempted**, asserted with a patched `open` — validation runs before any path is built | P0 | — |
| TC-072 | Streamlit UI smoke: app imports and renders without a live index | none | Import-only test, plus a headless run | Import succeeds with no optional dependency pulled in; with no index present the UI shows an actionable "no index found" state rather than a traceback | P1 | — |

> **TC-066 was corrected.** It excluded only `elapsed_ms`, but the same `--json` payload carries
> `timings`, a nested block of wall-clock floats per stage. As written it would have failed on its
> first run and every run after. The implementation and the test now exclude both.
>
> **TC-067 was corrected.** It asserted `sub_queries == [original_query]`.
> [Schema.md](Schema.md) and `schema/plan.py` both say `sub_queries` stays **empty** when no
> decomposition happened, and `effective_queries` is the property that falls back to the original.
> Schema is canonical.
>
> **TC-069 was corrected.** It invoked `axiom version --help`; `FR-23` defines `versions`. The
> singular form exists in `cli.py` only as a hidden alias, so asserting it would pin an
> undocumented surface.

### 3.9 Category I — Versioning and incremental reindex (TC-073 .. TC-080)

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-073 | `git diff --name-status` parsing covers A/M/D/R | `fixture_git_repo` | `diff_versions(old_sha, new_sha)` | Exact expected sets for added, modified, deleted; renames parsed with both paths captured; similarity-scored `R###` forms handled | P0 | `test_scripts.py::TestGitFixture` |
| TC-074 | Only A and M files are re-embedded | fakes, embed-call counter | Incremental v1→v2 | Embed calls equal the number of chunks in added+modified files whose `content_hash` is new; zero for deleted and unchanged files | P0 | `test_pipeline.py::TestReindexProducesAQueryableVersion` |
| TC-075 | **A pure rename costs zero embedding calls** | v2 differs only by a rename | Incremental reindex | `embed_call_count == 0`; renamed chunks get a new `chunk_id` (the path changed) but reuse the existing blob by `content_hash`; `.axiom/blobs/` gains no files | P0 | `test_pipeline.py::TestVersions::test_a_pure_rename_between_versions_reuses_the_blob` |
| TC-076 | **Incremental reindex is byte-equivalent to a full rebuild of the same tree** | none | (a) full-build v2 from scratch into `.axiom_full/`; (b) build v1 then incrementally reach v2 in `.axiom_inc/` | `chunks.jsonl` equal after sorting by `chunk_id`; idmap equal as a mapping; reconstructed vectors equal to 1e-6; sparse query results identical over a probe set; `structural.sqlite` equal after a canonical ordered dump; `manifest.json` equal except `created_at` and `parent_version` | P0 | — (`T-162`) |
| TC-077 | A version with zero changes is a no-op | v2 == v1 | Incremental reindex | New manifest with `parent_version == "v1"` and the same `chunk_count`; zero embed calls; zero new blobs; the new directory is independently readable | P1 | `test_pipeline.py::TestReindexProducesAQueryableVersion::test_an_unchanged_tree_costs_no_embedding_calls` |
| TC-078 | A version that deletes every file yields a valid empty index | v2 is an empty tree | Incremental reindex | `chunk_count == 0`; `chunks.jsonl` present and empty; the dense index has `ntotal == 0`; querying returns `[]` with no exception; v1's blobs are retained | P0 | `test_pipeline.py::TestIndexBuild::test_an_empty_repository_still_publishes_a_version` |
| TC-079 | Registry tracks versions and the active pointer | two versions indexed | `axiom versions`, then switch the active version | Both listed with chunk counts; the pointer switches; switching costs no embedding calls | P0 | `test_pipeline.py::TestVersions::test_each_response_names_the_versions_it_searched` |
| TC-080 | Blob garbage collection removes only unreferenced blobs | 3 versions, one removed | `axiom gc`, and `axiom gc --dry-run` | Blobs referenced by a surviving version are kept; blobs referenced only by the removed version are deleted; `--dry-run` mutates nothing. **`gc` is not an `FR-23` subcommand** — see §2.1 | P1 | — |

> **The stray token in TC-074's id cell** (`| triangle TC-074 |`) is removed. The id is `TC-074`.
> **`.prism_full/`/`.prism_inc/` in TC-076 are now `.axiom_full/`/`.axiom_inc/`**; the index
> directory is `.axiom/` per `ADR-015`.
> **TC-079 now invokes `axiom versions`**, not `axiom version list`; `cli.py` defines `versions`.

### 3.10 Category J — Evolutionary retrieval (TC-081 .. TC-085)

`versioning/evolutionary.py` is implemented but has **no dedicated test module**; only
`tests/test_schema.py::TestSnippetFamily` touches the data shape.

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-081 | **Family formation groups near-identical snippets** | 3 versions of one function differing by a line each; `FakeEmbedder` tuned to pairwise cosine above threshold | `build_families()` | Exactly one `SnippetFamily`; all three members; `versions` is the three ids; members sorted newest→oldest; `representative` is the newest | P0 | — |
| TC-082 | **Family formation does NOT group genuinely different snippets** | a same-symbol pair below threshold, plus a near-identical pair with **different** `symbol` | `build_families()` | The below-threshold pair yields two families. The different-symbol pair also yields two — grouping requires matching `symbol` **and** `file_path`, not similarity alone | P0 | — |
| TC-083 | Threshold boundary behaviour at `dedupe_cosine` | synthetic pairs just below, at, and just above the threshold | `build_families()` | At and above group (the comparison is `>=`); just below does not. The **operator** is asserted, not assumed. Threshold read from `settings.dedupe_cosine`, still `# PLACEHOLDER` | P0 | — |
| TC-084 | `stability` is computed correctly and the bonus applies only to multi-version families | 4 total versions; a family present in 3 | Rank with the evolutionary ranker | `stability == 0.75`; `final == base * (1 + stability * settings.stability_bonus)`; a single-version family receives `final == base`, because the bonus requires `len(versions) >= 2` | P0 | — |
| TC-085 | Cross-version query returns one row per family | 3 versions indexed | `axiom query "…" --all-versions` | No two results share a family; each exposes its member list and per-transition `diffs`; count ≤ requested `top_k`. Note `RetrievalResult` carries no family fields — the collapse happens in `cli._collapse_families`, and the family block is what carries `members`/`diffs` | P1 | — |

### 3.11 Category K — Agent loop (TC-086 .. TC-090)

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-086 | **Loop terminates within `max_passes` and within the wall-clock budget on adversarial input** | `FakeLLM` always answering "insufficient"; `FakeCrossEncoder` never satisfying sufficiency; 10k-char query; empty corpus | Run the loop repeatedly | `passes_executed <= settings.agent_max_passes` every time — **the bound counts total retrieve→fuse→rerank cycles, initial pass included**; wall clock under budget every time; `stop_reason` in `{max_passes, budget_exhausted, sufficient, no_new_query, empty_query, pass_failed}`; never raises, never loops forever | P0 | `test_agent.py::TestLoop` |
| TC-087 | Sufficiency thresholds fire exactly as specified | fake reranker with controlled scores | (a) top-1 just below `sufficiency_top1`; (b) top-1 above it with enough results above the floor; (c) a high top-1 with too few results above the floor | (a) refine; (b) stop; (c) refine. Thresholds read from `settings.agent_sufficiency_top1` / `_floor` / `_min_results`, all three currently `# PLACEHOLDER` except `min_results` | P0 | `test_agent.py::TestSufficiency` |
| TC-088 | Budget exhaustion mid-pass returns the best results so far, not an error | clock monkeypatched to advance past the deadline | Run the loop | Returns the completed pass's result set; `stop_reason == "budget_exhausted"`; the check happens before starting a new pass and before reranking a new pass | P0 | `test_agent.py::TestLoop` |
| TC-089 | Refinement changes the query and never repeats an identical retrieval | `FakeLLM` returning a rewrite | Run a 2-pass loop; capture the queries issued | Pass-2 query differs from pass-1; an identical rewrite stops early with `stop_reason == "no_new_query"` | P0 | `test_agent.py::TestRefinement` |
| TC-090 | Results across passes are merged without duplicates and with stable provenance | 2-pass loop, overlapping candidates | Inspect the merged output | No duplicate `chunk_id`; merge is order-stable; `assert_ranked_list` passes on the final list | P0 | `test_agent.py::TestLoop` |

### 3.12 Category L — Structural retrieval, `FR-10` (TC-091 .. TC-098)

**New on 2026-09-23.** `FR-10` is P0 and is the requirement
[PRD.md §8](PRD.md#8-how-this-maps-to-the-jury-scoring-rubric) credits with the Innovation score. It
previously had no category and no case. The implementation is `retrieval/structural.py`; the tests
below already exist in `tests/test_structural.py` (39 tests) and are now given ids so the matrix can
cite them.

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-091 | **callers-of** returns every chunk containing a call to the symbol | structural index built | `callers_of("resolveTool", limit)` | Every returned chunk's symbol has a `calls` row with that `callee_name`; matching is on `callee_name`, so **unresolved edges still count** — which is exactly what "where is X used" wants | P0 | `test_structural.py::TestGraphQueries::test_callers_of_finds_every_call_site` |
| TC-092 | **callees-of** walks the other direction | as above | `callees_of("main", limit)` | Returns the chunks defining functions the symbol's body invokes, joined through the resolved `callee_id`; an unresolved callee contributes nothing, because there is no chunk to point at | P0 | `test_structural.py::TestGraphQueries::test_callees_of_walks_the_other_direction` |
| TC-093 | **imports-of and exports-of** resolve all three module spellings | as above | `imports_of("./normalize")`, `imports_of("src/normalize.js")`, `imports_of("normalize")`; then `exports_of` both ways | All three spellings match the same file. One chunk per importing file is returned, so a popular utility cannot flood the candidate list. `exports_of` serves both readings — what a module exports, and who exports a named symbol | P0 | `test_structural.py::TestGraphQueries::test_imports_of_resolves_a_relative_module_specifier`, `::test_exports_of_reads_the_is_exported_flag` |
| TC-094 | **ordered-call-pair — query archetype Q2** | as above | `ordered_call_pair("XYZ", "ABC", limit)` | Matches a chunk when **some** call to `XYZ` precedes **some** call to `ABC`: `MIN(ordinal of first) < MAX(ordinal of last)`. Directional — the reversed pair does not match. A chunk calling only one of the pair does not match. `ordered_call_pair(X, X)` means "calls it at least twice", which is the honest reading | P0 | `test_structural.py::TestOrderedCallPair` (7 tests) |
| TC-095 | **An unresolvable identifier returns an empty list, never an exception** | as above | Run every query shape with an unknown identifier, then with an empty string | `[]` from every shape in both cases. This is `FR-10`'s stated contract and [Rules.md](Rules.md) Rule 3; fusion then drops the structural weight and renormalises the other two | P0 | `test_structural.py::TestGraphQueries::test_every_shape_returns_empty_for_an_unresolvable_identifier`, `::test_every_shape_survives_an_empty_identifier` |
| TC-096 | **K is clamped and the ranked list is well formed** | as above | `search(query, plan, k)` with `k` above `settings.structural_top_k` | Length clamped to the configured width (50); ranks 1-indexed and contiguous; `signal == STRUCTURAL` throughout; `assert_ranked_list` passes; evidence survives onto the result. Repeated searches are byte-identical | P0 | `test_structural.py::TestSearch::test_the_width_is_clamped`, `::test_search_returns_a_well_formed_ranked_list`, `::test_repeated_searches_are_deterministic` |
| TC-097 | **Store degradation: a missing, corrupt, or wrong-schema database returns empty** | none | Point the retriever at a missing file, a corrupt file, and a database with the wrong `user_version` | `[]` in all three cases, with one degradation record each. The `eval` profile skips the structural build entirely, and that is also not an error | P0 | `test_structural.py::TestStoreDegradation` (5 tests) |
| TC-098 | **A SQL metacharacter in an identifier is not an injection** | as above | Query with `'`, `%`, `_`, `;--` in the identifier | Parameterised throughout; `LIKE` patterns escaped; no error and no unintended match | P0 | `test_structural.py::TestStoreDegradation::test_a_sql_metacharacter_in_an_identifier_is_not_an_injection` |

### 3.13 Category M — Eval harness and submission artifact (TC-099 .. TC-105)

**New on 2026-09-23.** `FR-22` produces the single file attached to the release and had no case.

| ID | Description | Preconditions | Steps | Expected result | Pri | Implemented by |
|---|---|---|---|---|---|---|
| TC-099 | **`appsretrieval_results.json` has the required shape** | a completed run | Load the file `scripts/run_eval.py` writes | Top level carries `dataset_revision`, `task_name`, `mteb_version`, `evaluation_time`, `kg_co2_emissions`, `scores`, `axiom_provenance`. `scores[split]` is a list of score blocks whose `main_metric` is `ndcg_at_10`. `axiom_provenance` carries `reportable`, `non_reportable_because`, `mode`, `profile`, `config_hash`, `placeholders_active`, `models`, `weights`, `backend`, `dataset_source`, `limit`, `top_k`, `coverage`, `reported_percent`, `git_sha`, `generated_at`, `command`, `python`, `timings`, `resolved_config`. `reported_percent` holds `ndcg_at_10`, `mrr_at_10`, `recall_at_100`, `map_at_10` **as percentages** | P0 | — |
| TC-100 | **The reportability gate fires on every non-reportable condition** | none | Produce a run that is (a) `--limit`ed, (b) made with an active placeholder, (c) made on a dirty git tree, (d) made with a degraded retrieval backend, (e) made with a resolved embedder that is not the configured one | `axiom_provenance.reportable is False` in every case, and `non_reportable_because` names the specific reason. A run passing all five has `reportable is True` and an empty reason list. **This case currently cannot pass**: `Settings.active_placeholders()` returns the static `PLACEHOLDER_FIELDS` frozenset rather than the fields still at their placeholder value, so condition (b) never clears — see [Tracker.md §7](Tracker.md#7-open-defects-this-board-raises-against-src-and-the-repo) | P0 | `test_scripts.py` (partial) |
| TC-101 | Metric arithmetic matches hand-computed values | none | Compute NDCG@10, MRR@10, Recall@100 and MAP@10 over a small hand-labelled run against `eval/metrics.py` | Every metric matches the hand-derived constant to 1e-9. If NDCG is wrong, every number this project reports is wrong and no other test would reveal it | P0 | — |
| TC-102 | **CPU-only: no accelerator reference anywhere in the tree** (`NFR-06`) | none | Grep `src/`, `scripts/`, `configs/` and the lockfile for `.cuda()`, `.to("cuda")`, `device="cuda"`, `torch.cuda`, `faiss-gpu`, `nvidia-` | Zero matches. Wired into CI stage 1. `NFR-06` previously had **no automated verification at all**, while [Rules.md](Rules.md) calls this "the check that will actually save us" against a failure it calls unrecoverable on submission day | P0 | — |
| TC-103 | **`timings` block shape** (`NFR-10`) | fixture indexed | Inspect one `--json` response per archetype, and one index report | `timings` is present and non-empty, carries a `stages` collection, and is JSON-serialisable. The API layer flattens it to `dict[str, float]` keyed by stage via `api.models.flatten_timings`; the CLI emits the nested ledger. Both shapes are asserted, because they differ deliberately | P0 | `test_pipeline.py::TestQuery::test_the_response_serialises_for_the_json_mode`, `test_scripts.py::TestBenchRun` |
| TC-104 | `RetrievalResult` formatting contract (`FR-14`) | fixture indexed | Format a result set | Each result carries chunk text, `file_path`, `start_line`–`end_line`, a finite `score`, a `match_reason` that names the dominant signal or a declared degradation token, and the per-signal rank map. Every result explains itself | P0 | `test_pipeline.py::TestQuery::test_every_result_explains_itself`, `::test_formatted_lines_carry_a_location_and_a_reason` |
| TC-105 | `optimization_hint` fires from the rule table only (`FR-15`) | a chunk matching a rule | Retrieve it | `optimization_hint` is a fixed string from the rule table, never model prose, and is `None` when no rule fires. It is a static check and does **not** depend on `AXIOM_LLM_ENABLED` | P2 | — |

### 3.14 Currently failing — six tests, two causes

Recorded here rather than in a bug tracker because this plan is what the milestone gates read, and
`M0` is not green while these fail. On the current tree 403 tests are collected, 397 pass and six
fail. They are **two unrelated defects**, and conflating them would hide the smaller one.

#### Cause A — chunk spans disagree between bytes and lines (4 failures) · `T-019`

A chunk that does not begin at column 0 — every method, every nested function — is given a
`start_byte` at its declaration token but a `start_line` naming the whole line. `Chunk.text` is the
byte slice, so it lacks the leading indentation that
`source.split("\n")[start_line-1 : end_line]` includes. `assert_byte_equivalent` passes;
`assert_line_equivalent` fails on the indentation. Schema §4 requires both to hold.

Why it is worse than four red tests: `compute_chunk_id` binds `file_path` **and `start_line`**, so
the defect sits on the identity function the entire index is keyed by, and every `file:line` header
in the CLI, the UI and the demo is rendered from the same pair. A span that is right in bytes and
wrong in lines shows the jury a snippet whose header does not match the file on disk.

| Failing test | What it was protecting |
|---|---|
| `test_chunking.py::TestSpanInvariants::test_byte_and_line_spans_agree_across_the_whole_fixture_repo` | The invariant itself, across every fixture file — TC-012, TC-013 |
| `test_pipeline.py::TestQuery::test_every_archetype_returns_results_with_real_locations` (3 parametrisations) | TC-065 — that a returned location is a **real** location |

The fix is a span decision, not a patch to the assertion: either snap `start_byte` back to the start
of its line (the text gains the indentation, the spans agree, and the text is what a reader would
copy), or narrow `start_line`/`end_line` to describe the byte span exactly. **The first is almost
certainly right** — a snippet shown at `file:line` should be the whole line — but it changes
`chunk.text` for every indented chunk, and therefore `content_hash`, and therefore every blob key.
That makes it a breaking change under
[Changelog.md §2](Changelog.md#2-what-counts-as-a-breaking-change), and it is far cheaper today,
with no index built, than after `M2`.

#### Cause B — two tests encode "faiss is not installed" as an unguarded precondition (2 failures) · `T-020`

| Failing test | Assertion | Why it fails |
|---|---|---|
| `test_pipeline.py::TestIndexBuild::test_the_degraded_rungs_are_visible_in_the_report` | `report.dense_backend == "numpy"`, with the message *"faiss cannot have loaded on a bare install"* | `faiss-cpu` **is** installed in the working environment, so the real rung runs and the backend is `faiss` |
| `test_pipeline.py::TestReindexProducesAQueryableVersion::test_every_artefact_lands_in_the_version_directory` | `dense.npy` is present in the version directory | The faiss rung writes `dense.faiss`, not the numpy fallback's `dense.npy` |

Both tests are *right about the degraded path* and wrong to assume it. The install shape is not a
property of the code, and §1.2 rule 4 requires every test to be full-suite-safe on any supported
install — a test whose result flips when an optional extra is present is not. The fix is to assert
against the rung that actually ran (`report.dense_backend` selects the expected artefact name)
rather than to pin one rung, or to guard the case with `pytest.importorskip("faiss")` inverted. Do
**not** uninstall `faiss-cpu` to make them pass: that trades a red test for an untested primary
path, and the primary path is the one the eval run uses.

This one matters out of proportion to its size, because it means **the dense retrieval path with
faiss present has been exercised less than the run count suggests** — the previously-reported green
suite was green partly on the fallback.

Nothing in this plan may be loosened to make any of the six pass. Rule 6 of §1.2 exists for exactly
this moment.

---

**Count: 105 cases, TC-001 through TC-105, all unique** — 90 inherited, 15 added on 2026-09-23
(Category L's eight for `FR-10`, Category M's seven). Roughly half carry a citation in the
`Implemented by` column; the empty cells are the real coverage backlog, itemised per requirement in
§2.2.

---

## 4. Edge-Case Catalogue

Each row is covered by the listed case. No row is aspirational.

| Edge case | Where it bites | Covered by |
|---|---|---|
| Empty query | classifier, embedder, CLI and API boundaries | TC-009 |
| Whitespace-only query | strips to empty after normalisation | TC-009 |
| 10,000-character query | embedder window, prompt size, latency | TC-010, TC-086 |
| Query of only stopwords | BM25 returns nothing; fusion sees an empty signal | TC-046, TC-055 |
| Unicode / emoji in query | tokenizer, byte-offset math, terminal output | TC-011 |
| Minified JS file | AST node count explosion, chunk count explosion | TC-023 |
| File with syntax errors | tree-sitter ERROR nodes | TC-021 |
| Empty file (0 bytes) | zero-chunk path, empty index add | TC-022, TC-078 |
| File with no functions | MODULE-kind fallback | TC-022 |
| Single oversized function | statement split, overlap, token budget | TC-019 |
| Deeply nested closures | recursion limit, `parent_symbol` chains | TC-024 |
| Circular imports | import-graph traversal | TC-033 |
| Duplicate file content at two paths | content-hash dedup vs `chunk_id` uniqueness | TC-016, TC-031 |
| Version with zero changes | incremental no-op path | TC-077 |
| Version that deletes every file | empty index, blob retention | TC-078 |
| Out-of-vocabulary identifier query | BM25 empty result | TC-045 |
| Adversarial regex input | ReDoS in the identifier miner | TC-047 |
| **Unresolvable identifier in a structural query** | **every one of the five `FR-10` shapes** | **TC-095** |
| **Corrupt or missing `structural.sqlite`** | **structural signal absent, fusion renormalises** | **TC-097** |
| **SQL metacharacter in an identifier** | **structural query parameterisation** | **TC-098** |
| Path-like `version` or `file_path` from the API | traversal (see [Security.md](Security.md)) | TC-071 |
| Cross-encoder raising mid-batch | rerank degradation | TC-062 |
| Interrupted index build | partial-state atomicity | TC-034 |
| **Dimension mismatch between index and runtime embedder** | **the one deliberately fatal case** | **TC-030** |
| **A degraded backend producing a number that looks reportable** | **the submission artifact** | **TC-100** |

---

## 5. Performance Test Plan

### 5.1 Harness

`scripts/bench_latency.py` is the only sanctioned performance measurement. It is **not** a pytest
test; it emits a JSON row and a human table to stdout.

```bash
python scripts/bench_latency.py \
  --phase query \
  --index-root .axiom --version v1 \
  --queries tests/fixtures/bench_queries.txt \   # 100 queries, fixed order
  --repeat 5 --warmup 3 \
  --profile default \
  --out artifacts/bench/
```

Rules:

- **100 queries, from `tests/fixtures/bench_queries.txt`, in fixed order.** This is the single
  latency fixture; any other count is not comparable with any recorded row. Documents quoting a
  50-query sample are stale against this section.
- 3 discarded warm-up iterations per query (model load, mmap page-in, OS cache).
- 5 measured repeats; report **p50 and p95 across all (query, repeat) samples**, not per-query means.
- Peak RSS is sampled during the measurement and reported as a peak.
- Every row carries `git_sha`, `profile`, `model_id`, `chunk_count`, `index_kind`, `python`, `cpu`,
  `threads`.
- Thread counts are pinned so results are comparable; `--no-pin-threads` exists and makes them
  incomparable, which is why it is not used for a recorded row.

**Nothing in §5 has ever been executed against a real model.** The harness is unit-tested
(`tests/test_scripts.py::TestBenchRun`, `::TestPercentiles`, `::TestBudgetGate`,
`::TestRegressionGate`) and has no baseline. Every budget below is a **budget**, not a measurement,
and none may be quoted as one.

### 5.2 Reference hardware

The budgets in [_CONTRACT.md](_CONTRACT.md) §7 are defined on an **8-core x86-64 CPU with 16 GB
RAM, no GPU, SSD, Linux**. That is the box the numbers in the PPT must come from. Developer laptops
(including arm64 machines) may be faster or slower; a laptop measurement is never quoted as a budget
result — it is used only for relative before/after comparisons on the same machine. Every reported
number states the box it came from.

### 5.3 Budgets and measurement

| ID | Budget (from `_CONTRACT.md` §7) | Measured by | Method |
|---|---|---|---|
| PB-01 | Cold index, 10k chunks | `scripts/build_index.py --time-it` | Wall clock from process start to registry write, empty `.axiom/`, cold page cache |
| PB-02 | Incremental reindex, 50 changed files | `scripts/build_index.py --incremental --time-it` | Synthetic v2 with exactly 50 modified files from `tests/fixtures/gen_v2.py`, warm model cache. **Never measured on the demo repo**, which is 10–50 files total and cannot produce this diff |
| PB-03 | Query p50, agent loop disabled | `bench_latency.py --no-agent` | p50 over 100 queries × 5 repeats, model already loaded |
| PB-04 | Query p95, agent loop at its pass cap | `bench_latency.py --agent --force-passes 2` | p95 over the same set |
| PB-05 | Peak RSS during query | `bench_latency.py --measure-rss` | Peak RSS of the worker process during PB-04 |
| **PB-06** | **Storage footprint (`NFR-12`)** | `du -sh .axiom` plus the model cache size after a clean run | On-disk index for the eval corpus, and total model download for the active profile. **New — `NFR-12` had no measurement row** |

A profile note that matters for PB-03/PB-04: `configs/eval.yaml` runs `bge-reranker-v2-m3` with a
deliberately narrowed candidate chain and is offline and untimed. **No number from the `eval`
profile may be quoted against `NFR-03` or `NFR-04`.** The latency profile is `demo.yaml`, whose
primary reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### 5.4 Regression gate

`scripts/bench_latency.py --compare artifacts/bench/<baseline_sha>.json` exits non-zero when:

| Condition | Action |
|---|---|
| Any budget in `_CONTRACT.md` §7 exceeded | CI fails (the `bench` job) |
| p50 or p95 regressed past the configured threshold vs the last baseline | CI fails |
| A smaller regression | CI warns; merge allowed |
| Peak RSS regressed past its threshold | CI fails |
| Index build time regressed past its threshold | CI warns |

The baseline is refreshed manually by the owner after an intentional, explained regression.
Refreshing requires a line in [Changelog.md §3](Changelog.md#3-bench-baseline-refresh-log). There is
no baseline today.

---

## 6. Evaluation Protocol

Retrieval quality is not a test. It is an experiment, and it follows experiment discipline.

### 6.1 Runs

| Mode | Command | Use |
|---|---|---|
| Smoke | `python scripts/run_eval.py --task AppsRetrieval --limit 200` | Did the harness wire up? |
| Dev | `python scripts/run_eval.py --task AppsRetrieval --limit 1000` | Directional signal while tuning |
| **Reportable** | `python scripts/run_eval.py --task AppsRetrieval --split test` (full split, no `--limit`) | The only run whose number may be quoted anywhere |

`scripts/run_eval.py` is the entrypoint; `axiom eval` is the CLI wrapper over the same code path.
Either may be used, but the recorded `command` field in the results JSON is what the experiment log
cites.

### 6.2 Why a limited run is never a reportable score

`--limit N` truncates the **corpus and/or query set**. That changes the task, in two ways that both
inflate the number:

1. A smaller corpus means fewer distractors, so NDCG@10 rises mechanically — with 200 documents
   instead of 8,765, random ranking already scores far above its full-split expectation.
2. Truncation is not a random sample; it is the first N rows in file order, which correlates with
   dataset construction order.

So a `--limit` number is not a pessimistic estimate of the real score — it is a **different,
unrelated, usually higher** number. Rule: **any NDCG/MRR figure in the PPT, the README, the release
JSON, or a commit message must come from a full-split run, and must cite its git SHA.** A limited
run is stamped `mode: "SMOKE"` by the harness and is never averaged with full runs.

### 6.3 Repeats and averaging

The pipeline is deterministic given the configured seed (TC-066), so a repeat of the same SHA and
config should reproduce exactly. We therefore run:

- **1 run** to produce the number, plus **1 independent re-run** to confirm identical metrics. If
  the two disagree, that is a determinism bug — fix it; do not average it away.
- **3 runs** only when a component is genuinely stochastic — currently just IVF-PQ training. For
  `index_kind == "ivf_pq"` we report mean ± range over 3 index builds with successive seeds.

### 6.4 Experiment log

`artifacts/experiments.csv` is append-only, committed, one row per run. **The path changed from
`data/experiments.csv`**: [Rules.md §9.5](Rules.md#95-what-may-and-may-not-be-committed) lists
`data/` under "never commit" and `.gitignore` enforces it, so a runbook that committed from `data/`
was scheduled to fail under time pressure on submission day. The committed artefacts —
`experiments.csv`, `splits/`, `bench/` — now live under `artifacts/`, which is not ignored.

| Column | Example |
|---|---|
| `run_id` | `2026-09-24T14:03Z-a1b2c3d` |
| `git_sha` | `a1b2c3d` (dirty tree → suffix `-dirty`, and the run is marked non-reportable) |
| `mode` | `FULL` / `SMOKE` |
| `task` / `split` | `AppsRetrieval` / `test` |
| `limit` | empty for FULL |
| `profile` | `eval` |
| `embedding_model`, `reranker`, `llm_enabled` | from `axiom_provenance.models` |
| `weights` | from `axiom_provenance.weights` |
| `agent_passes` | `0` / `2` |
| `ndcg@10`, `mrr`, `recall@100`, `map` | from `axiom_provenance.reported_percent` |
| `wall_clock_s`, `box` | — |
| `notes` | free text |

Discipline: **no number exists unless it has a row.** A screenshot is not a result. The submission
artifact `appsretrieval_results.json` must be reproducible from its row's SHA, profile, and command
— which is why the harness stamps `git_sha`, `config_hash` and the verbatim `command` into the file
itself.

**As of 2026-09-23 this log has zero rows.** No eval run of any kind has been executed.

### 6.5 Quality gates by milestone

These gate the milestone, not the merge. The gate ladder itself is
[ImplementationPlan.md §3](ImplementationPlan.md#3-milestone-gates-m0m7); the budgets are
`_CONTRACT.md` §7 and the target is `_CONTRACT.md` §8.

| Gate | Requirement |
|---|---|
| `0.1.0` foundation | Tests green, lint clean. No quality claim |
| `0.2.0` dense baseline | Full-split NDCG@10 / MRR@10 / Recall@100 recorded, any value; this is the project's own baseline |
| `0.3.0` hybrid + RRF | Full-split delta vs `0.2.0` recorded with its sign |
| `0.4.0` rerank | Full-split delta vs `0.3.0` recorded; paired ablation run recorded |
| `0.5.0` demo | Demo repo indexed; Q1/Q2/Q3 answered live; `NFR-02` measured on the synthetic diff |
| `0.6.0` tuned | Placeholders replaced from tune/dev splits; one full-split re-run afterwards; ablation table assembled |
| `1.0.0` submission | All P0 tests green; all `_CONTRACT.md` §7 budgets measured per §5.3; the reported metric set complete and reproducible from its SHA |

**No row above states an absolute NDCG@10 threshold.** The `≥ 18.0` / `≥ 20.0` thresholds earlier
revisions carried were calibrated against an unsourced baseline and are withdrawn; see
[ImplementationPlan.md §3.2](ImplementationPlan.md#32-no-gate-carries-an-absolute-ndcg-threshold-any-more).
An absolute target may be restored once `0.2.0` measures the real baseline, by the PRD owner, in
writing.

---

## 7. Manual Test Script (Demo Day)

Run against the demo repo (`data/demo_repo/`, ≥ 2 versions), profile `demo`. Executed by the
presenter before recording and again before the live session. Each step has a binary judgement — no
"looks fine".

| # | Query / action | Expected qualitative behaviour | Pass criterion |
|---|---|---|---|
| M-01 | `axiom query "How is the input preprocessed before going to the main function?"` | Surfaces the preprocessing chain and the caller of `main()`. `match_reason` names the dense signal | ≥2 of the top 5 are functions in the preprocessing chain, and the top result's file is one a human would open first |
| M-02 | `axiom query "Which files call tool XYZ before tool ABC?"` | Classified `STRUCTURAL`; results contain both call sites in order; `signals` shows structural dominant | Every returned file genuinely contains both call sites, and ordering respects call order |
| M-03 | `axiom query "Where is the Bluetooth-settings deeplink used?"` | Classified `USAGE`; sparse-dominant; returns call sites, not the definition alone | The definition and ≥1 call site both appear in the top 5 |
| M-04 | `axiom query "auth"` (one-word query) | Broad but non-empty; no crash | ≥5 results, all related |
| M-05 | `axiom query ""` | **Clean error message, exit code 2, no traceback.** The CLI rejects at the boundary before touching the index — it does not raise | Exit code is exactly 2 and stderr carries a remediation line |
| M-06 | `axiom query "$(python -c 'print("x"*10000)')"` | Truncation warning; still returns results | Returns results within budget, no crash |
| M-07 | `AXIOM_LLM_ENABLED=false axiom query "How is the input preprocessed?"` | Identical output shape, heuristic classification, visibly faster | Non-empty results; output notes LLM-disabled mode |
| M-08 | `axiom reindex data/demo_repo --version v3 --from v2` | Console shows changed-file count, re-embedded chunk count, elapsed time | Re-embedded chunks are a small fraction of the total. **The 45 s `NFR-02` figure is PB-02's, measured on the synthetic diff — it is not claimed here** |
| M-09 | `axiom versions`, then the same query at two versions | Line numbers and code differ between versions as the git history says they should | Both return results; the two snippets are visibly the historical and current forms |
| M-10 | `axiom query "handleDeeplink" --all-versions` | One family row, expandable to its versions with per-transition diffs, newest first | Exactly one family; member order newest→oldest; diffs non-empty for ≥1 member |
| M-11 | Streamlit UI: the M-01 query, then click a result | Snippet with file path and line range, signal badges, expandable version history | Clicking opens the snippet with line numbers matching the file on disk |
| M-12 | Streamlit UI with the index deleted | Actionable empty state telling the user to run `axiom index` | No traceback in the browser |
| M-13 | Stopwatch on M-01 and M-02 | M-01 sub-second, M-02 within the agent budget | Visible timer shows values within budget |

Record all 13 outcomes in [Tracker.md §6](Tracker.md#6-demo-day-readiness-checklist) the day before
recording. Any FAIL on M-01..M-03 or M-08..M-10 blocks recording the video.

> **M-07 was corrected**: it invoked `prism query`, which does not exist — the entrypoint is
> `axiom`. As written, the demo script was scripted to fail live in front of the jury.
> **M-05 was corrected**: it required exit 2 while TC-009 required an exception. Exit 2 is right, and
> §3.1's note explains why both can be true without contradiction.
> **M-09 was corrected**: `axiom version list` is not a subcommand; `axiom versions` is.

---

## 8. CI Pipeline

`.github/workflows/ci.yml`, Ubuntu, Python 3.11 and 3.12 matrix. **This file does not exist yet —
`T-002`.** Everything below is the specification it must satisfy; today the equivalent is run by
hand.

| Stage | Command | Blocks merge |
|---|---|---|
| 1. Lint | `ruff check . && ruff format --check .`, plus TC-102's accelerator grep | yes |
| 2. Types | `mypy src/axiom/core src/axiom/retrieval src/axiom/schema` (strict, per `pyproject.toml`) | yes |
| 3. Fast tests | `pytest -m "not slow and not bench" --cov=src/axiom` | yes |
| 4. Smoke index + query | builds an index over `tests/fixtures/repo_v1` with fakes and runs the archetype queries | yes |
| 5. Dependency audit | `pip-audit` over the resolved environment (`NFR-06`'s `pip check` runs here too) | yes |
| 6. Slow tests | `pytest -m slow` — real INT8 models, cached | no |
| 7. Benchmarks | `python scripts/bench_latency.py --compare <baseline>` | no on PRs |
| 8. Eval smoke | `python scripts/run_eval.py --task AppsRetrieval --limit 200` | no; result appended to the experiment log |

Stages 1–5 are the merge gate. Anything slower gets a marker.

### 8.1 Marker convention

Declared in `pyproject.toml` under `[tool.pytest.ini_options] markers`. As committed, the file
declares `slow` and `integration`; the markers below that it does not yet declare must be added
before the stages that select them can run.

| Marker | Meaning | Runs in the merge gate | Declared |
|---|---|---|---|
| (none) | Fast, hermetic, offline | yes | n/a |
| `integration` | Crosses module boundaries with fakes | yes | yes |
| `slow` | Downloads or loads real model weights | no | yes |
| `smoke` | Exercises a real subsystem end to end with fakes | yes | **no** |
| `bench` | Timing/memory sensitive; must not run concurrently | no | **no** |
| `needs_git` | Shells out to a real `git` binary | yes | **no** |

Enforcement: `pytest --strict-markers` is already set in `pyproject.toml`, which means a stage
selecting an undeclared marker fails immediately rather than silently selecting nothing.

### 8.2 What a merge requires

1. Stages 1–5 green on both Python versions.
2. At least one P0 test covering the changed behaviour, or an explicit "no behaviour change" note in
   the PR description.
3. Schema changes (`src/axiom/schema/`): a [Changelog.md](Changelog.md) entry and a version bump per
   the breaking-change definition there, plus four-member sign-off.
4. Any change to fusion arithmetic, hashing, or the on-disk layout: the affected P0 test updated in
   the *same* commit, with its hand-computed expected value re-derived in the test body.

---

## 9. Coverage Targets

Line coverage is a smoke detector, not a goal. We set the bar high where a bug is silent and
catastrophic, and low where behaviour is measured by evaluation instead.

| Module | Line | Branch | Rationale |
|---|---|---|---|
| `src/axiom/schema/` | 95% | 90% | Pure data contract shared by four people. A wrong field name or a bad validator corrupts every artifact silently and is cheap to test exhaustively |
| `src/axiom/retrieval/fusion.py` | 95% | 90% | Deterministic arithmetic with a published reference. Empty-signal renormalisation and tie-breaking are exactly where bugs hide |
| `src/axiom/core/` | 90% | 85% | `chunk_id`/`content_hash` derivation is the backbone of dedup and incremental reindex. A hashing bug is invisible until the index is wrong |
| `src/axiom/eval/metrics.py` | 90% | 85% | If NDCG is wrong, every number we report is wrong and the pipeline will not tell us |
| `src/axiom/versioning/` | 85% | 75% | Diff parsing and blob reuse are logic-heavy and testable; TC-076 carries most of the weight and does not exist yet |
| `src/axiom/chunking/` | 80% | 70% | Heavy branching over AST shapes. We cover the taxonomy, the oversize path, the fallback path, and the catalogued edge cases |
| `src/axiom/retrieval/structural.py` | 80% | 70% | Raised from 70% on 2026-09-23: this is the innovation claim, and Category L now tests it properly |
| `src/axiom/agent/` | 75% | 65% | Termination, budget and threshold logic are P0-tested. The *quality* of a rewrite is not testable — it shows up in NDCG |
| `src/axiom/indexing/` | 75% | 65% | Correctness is asserted structurally (layout, bijection, dedup); FAISS/bm25s internals are third-party |
| `src/axiom/rerank/` | 70% | 60% | The interesting behaviour is ordering and degradation. **Currently near zero — no test module exists** |
| `src/axiom/retrieval/{dense,sparse}.py` | 70% | 60% | Thin adapters. Invariant tests catch the real bugs; the rest is delegation |
| `src/axiom/api/`, `src/axiom/ui/`, `src/axiom/cli.py` | 60% | — | Presentation layers, contract-tested and validated by §7. **Currently near zero** |
| **Project gate** | **80%** | — | `--cov-fail-under=80` in the merge gate |

The project gate is **not met today.** Four modules — `rerank/`, `api/`, `ui/`, `cli.py` — have no
test module at all, and `versioning/evolutionary.py` is covered only at the schema level. That is
the honest state, and §2.2's matrix lists every gap individually rather than burying them in an
aggregate percentage.

### 9.1 Why retrieval-quality modules get a lower bar

Two different failure modes need two different tools.

A bug in `schema/` or `fusion.py` is a **logic defect**: it has a right answer, that answer is
computable by hand, and a passing test proves the code produces it. Coverage there is meaningful
because an uncovered branch is a branch whose right answer nobody checked.

A "bug" in `agent/` or `rerank/` is usually a **quality deficit**: the code ran correctly and
returned a worse ranking. No coverage percentage detects that, and the tests written chasing 95% in
those modules are exactly the tests §1.2 bans — asserting that a mock was called, that a prompt
contains a substring, or that a particular chunk ranked first. Those tests break on every retune and
never catch a real regression.

So for those modules we test the **contract** (terminates, stays in budget, degrades, preserves
invariants, never crashes) at P0, accept 70–75% line coverage, and delegate "is it good?" to
NDCG@10 on the full split with a logged git SHA. That is the honest division of labour — and it only
works if the full-split run actually happens, which as of 2026-09-23 it has not.
