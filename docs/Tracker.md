# Tracker

The `T-###` task board, burndown, standup log, and eval metrics log for Axiom.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-23
**Status:** Active — re-baselined

Related: [ImplementationPlan.md](ImplementationPlan.md) · [PRD.md](PRD.md) · [TestPlan.md](TestPlan.md) · [Rules.md](Rules.md) · [OpenQuestions.md](OpenQuestions.md) · [Decisions.md](Decisions.md) · [Changelog.md](Changelog.md)

---

## 0. Re-baseline notice

This board was last accurate on 2026-09-16, when every one of its 71 rows read `Planned`. That is
no longer true of anything. On 2026-09-23 the board was walked row by row **against the code on
disk** — each `Done` below names the module that implements it and, where one exists, the test that
proves it. Nothing here is marked `Done` on the strength of a plan.

**The suite is green.** **611 tests collected, 611 pass, 0 fail** (2026-09-23, real stack:
tree-sitter, bm25s, faiss-cpu, onnxruntime, real MiniLM ONNX weights). `ruff check` and
`ruff format --check` are both clean. The two defects that made it red — `T-019` (chunk spans,
4 failures) and `T-020` (two tests that assumed `faiss-cpu` was absent, 2 failures) — are both
**closed**. The suite grew from 403 to 611 because `tests/test_api.py` (91), `tests/test_eval.py`
(82) and `tests/test_ui.py` (35) were added, and because the optional backends are now installed,
so the top rung of each ladder is actually executed rather than skipped.

Two structural changes came with the walk:

1. **Eight rows were added.** `FR-14` (result formatting) and `FR-23` (the CLI) are both Must-have
   in [PRD.md §5](PRD.md#5-functional-requirements) and had **no tracker row at all** — every gate
   command in the plan is a CLI invocation, so the board was missing its own dependency. They are
   now `T-143` and `T-027`. Five more cover work the old board never named: `artifacts/` relocation
   (`T-016`), the fresh-clone rehearsal `NFR-09` asserts (`T-017`), the behavioural embedder check
   (`T-018`), the two fixes the failing suite demands (`T-019`, `T-020`), and the full-split
   dense-only baseline (`T-026`).
2. **Day numbers are the re-baselined ones** from
   [ImplementationPlan.md §4](ImplementationPlan.md#4-day-by-day-plan): Day 1 = 2026-09-23, Day 5 =
   2026-09-27. Any "Day 6"–"Day 10" reference elsewhere is void.

---

## 1. How this board works

Task ids are permanent and non-sequential-by-design: blocks of 40–60 numbers are reserved per
workstream so a task's id alone tells you roughly where it lives, and so a specific id referenced
elsewhere in the doc suite (`T-141` in [Rules.md §8](Rules.md#8-the-placeholder-convention),
`T-201` in [`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission) and
[`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom)) has a stable, predictable home.

| Block | Workstream | Owner |
|---|---|---|
| `T-001`–`T-020` | Foundation (`M0`) | shared |
| `T-021`–`T-060` | Retrieval core (dense, sparse, fusion, CLI) | Prabinder |
| `T-061`–`T-090` | Structural (chunking, AST, call graph) | Anish |
| `T-091`–`T-150` | Agent, rerank, API, UI | Harshdeep |
| `T-151`–`T-190` | Versioning, evolutionary retrieval | Parth |
| `T-191`–`T-230` | Evaluation discipline, submission, docs | Parth |

The block does **not** imply the owner: `T-112` and `T-193` sit in other people's blocks because
they belong to the `OQ-##` whose owner is named in
[OpenQuestions.md §1](OpenQuestions.md), and that ownership wins. The `Owner` column is the only
authority, and it is regenerated from [PRD.md §5](PRD.md#5-functional-requirements)'s `Owner`
column wherever a task implements an `FR-##`.

Status vocabulary matches `_CONTRACT.md §9`: `Planned` → `In Progress` → `Blocked` → `Done` →
`Dropped`. A `Blocked` task always names what it is blocked on, either a `T-###`, an `OQ-##`, or an
external fact. The `Evidence` column names the module or test that makes a `Done` checkable; a
`Done` row with no evidence is a documentation defect.

---

## 2. Task board

### 2.1 Foundation (`T-001`–`T-020`)

| ID | Task | Owner | Day | Status | Evidence / blocker |
|---|---|---|---|---|---|
| T-001 | Repo scaffold: `pyproject.toml`, src-layout, `uv.lock` | Prabinder | 1 | In Progress | `pyproject.toml` + `src/axiom/` layout land; **`uv.lock` does not exist** — `uv sync --frozen` in [Setup.md](Setup.md) and `NFR-09` both fail today |
| T-002 | CI skeleton: ruff → mypy → pytest → smoke index stages | Prabinder | 1 | Planned | No `.github/` directory exists. [TestPlan.md §8](TestPlan.md#8-ci-pipeline) describes a pipeline that has never run |
| T-003 | Load `AppsRetrieval` train + test splits via MTEB v2; confirm BEIR shape (`OQ-05`) | Parth | 1 | In Progress | `src/axiom/eval/mteb_adapter.py` handles every MTEB v2 corpus/query shape and a vendored fallback; **never executed against the real Hub** |
| T-004 | `artifacts/splits/` seeded tune/dev id lists (4,000/1,000) | Parth | 1 | Planned | Blocked on `T-003`. Path changed from `data/splits/` — see `T-016` |
| T-005 | tree-sitter + `tree-sitter-javascript` install; parse a real `.js` file | Anish | 1 | Done | `tree_sitter` 0.26.0 + `tree_sitter_javascript` 0.25.0 installed; `src/axiom/chunking/ast_chunker.py`; `tests/test_chunking.py::TestAstTaxonomy` |
| T-006 | `src/axiom/schema/`: enums, `ChunkLocation`, `ChunkMetadata`, `Chunk` | Harshdeep | 1 | Done | `src/axiom/schema/` (6 modules); `tests/test_schema.py` (71 tests) |
| T-007 | `core/hashing.py`: `blake2b_128`, `compute_chunk_id`, `compute_content_hash` | Harshdeep | 1 | Done | `src/axiom/core/hashing.py`; `tests/test_hashing.py` (34 tests) |
| T-008 | Heuristic query classifier (no LLM) | Harshdeep | 1 | Done | `src/axiom/agent/classifier.py`; `tests/test_agent.py::TestClassification` |
| T-009 | Embedder loads and encodes one batch | Prabinder | 1 | In Progress | `src/axiom/indexing/embedder.py` implements the full ONNX → torch → hash ladder; **no real weight has ever been loaded** |
| T-010 | `M0` gate check: `uv sync --frozen`, CI green, 400/400 tests, `ruff`/`mypy` clean | Prabinder | 1 | Planned | Blocked on `T-001`, `T-002` |
| T-011 | `OQ-07`: candidate tagged JS repo shortlisted (10–50 files, ≥ 2 tags) | Parth | 1 | Planned | `data/demo_repo/` is empty. Blocks `T-012`, `T-066`, `T-163`, `T-182` |
| T-012 | `OQ-07`: demo repo finalised, staged under `data/demo_repo/` | Parth | 2 | Planned | Blocked on `T-011` |
| T-013 | ONNX export pipeline dry run (fp32 export, no quantisation yet) | Prabinder | 1 | Planned | `src/axiom/indexing/embedder.py` expects `data/models/onnx/…`; that tree does not exist |
| T-014 | Fixture repo authored per [TestPlan.md §1.3](TestPlan.md#13-fixtures) | Anish | 1 | Done | `tests/fixtures/repo_v1/` and `repo_v2/`: broken file, empty file, constants-only, circular pair, duplicated content, deep nesting |
| T-015 | `FakeEmbedder`, `FakeCrossEncoder`, `FakeLLM` test doubles | Harshdeep | 1 | Done | `tests/fakes.py`; used by all 400 tests |
| **T-016** | **Move committed artefacts out of the gitignored `data/` tree into `artifacts/`** (`experiments.csv`, `splits/`, `bench/`) | Parth | 1 | Planned | `.gitignore` ignores `data/` **and `appsretrieval_results.json`** — the submission artifact itself is currently uncommittable. See §7 |
| **T-017** | **Fresh-clone rehearsal (`NFR-09`): clone, `uv sync`, one documented command to a working index** | Parth | 5 | Planned | `NFR-09`'s own stated measurement. Blocked on `T-001` |
| **T-018** | **Behavioural embedder verification: two paraphrases score closer than an unrelated string** | Anish | 1 | Planned | A shape check (`hidden == 1024`) passes for every wrong pooling choice; this is the check that does not. Blocked on `T-013` |
| **T-019** | **Fix the chunk span defect: `chunk.text` must equal `lines[start_line-1 : end_line]`** | Anish | 1 | Planned | **Blocks `M0`.** Four failing tests: a chunk not starting at column 0 gets `start_byte` at the declaration token and `start_line` at the whole line, so the byte slice omits the leading indentation the line slice includes. Breaks Schema §4. `chunk_id` binds `start_line`, and every `file:line` the jury sees comes from the same pair. Failures: `test_chunking.py::TestSpanInvariants::test_byte_and_line_spans_agree_across_the_whole_fixture_repo`, `test_pipeline.py::TestQuery::test_every_archetype_returns_results_with_real_locations` (3 params). Detail: [TestPlan.md §3.14](TestPlan.md#314-resolved--the-six-failures-that-blocked-m0) |
| **T-020** | **Two tests assume `faiss-cpu` is not installed; assert against the rung that actually ran** | Prabinder | 1 | Planned | **Blocks `M0`.** `test_pipeline.py::TestIndexBuild::test_the_degraded_rungs_are_visible_in_the_report` asserts `dense_backend == "numpy"`; `::TestReindexProducesAQueryableVersion::test_every_artefact_lands_in_the_version_directory` expects `dense.npy`. With the `retrieval` extra installed the faiss rung runs and writes `dense.faiss`. Do **not** uninstall faiss to go green — that would leave the primary path untested. Same section of TestPlan |

### 2.2 Retrieval core (`T-021`–`T-060`)

| ID | Task | Owner | Day | Status | Evidence / blocker |
|---|---|---|---|---|---|
| T-021 | Dense index builder: embed + L2-normalise + `dense.faiss` + `dense.idmap.json` | Prabinder | — | Done | `src/axiom/indexing/dense.py`; flat/IVF-PQ selection at the 50k threshold; positional `rows` idmap; `tests/test_pipeline.py::TestIndexBuild` |
| T-022 | Dense index over the full APPS corpus (8,765 docs) | Prabinder | 1 | Planned | Blocked on `T-013`/`T-040`. `eval/mteb_adapter.py::corpus_to_chunks` is written and waiting |
| T-023 | `scripts/run_eval.py --limit 200` smoke run produces a number (harness proven) | Prabinder | 1 | Planned | Blocked on `T-022` |
| T-024 | `OQ-03`: dense-only NDCG@10 comparison, Qwen3-0.6B vs MiniLM fallback | Prabinder | 2 | Planned | Blocked on `T-026` |
| T-025 | `M1` gate check | Prabinder | 1 | Planned | — |
| **T-026** | **Full-split dense-only baseline: `--split test`, no `--limit`, sparse and rerank off** | Prabinder | 2 | Planned | **The project's own baseline.** Every gain claimed anywhere is a delta against this row. Gates `M2` |
| **T-027** | **`FR-23` Typer CLI: ten subcommands, `--json` on each, exit-code discipline** | Prabinder | — | Done | `src/axiom/cli.py`: `index`, `reindex`, `query`, `classify`, `versions`, `families`, `eval`, `serve`, `ui`, `gc`, plus a hidden `version` alias. Exit codes 0/1/2/3. **`gc` is a tenth command `FR-23` does not define** — see §7 |
| T-030 | Sparse index builder: `bm25s` over the chunk corpus, code-aware tokenizer | Prabinder | — | Done | `src/axiom/indexing/sparse.py`, `src/axiom/retrieval/tokenizer.py`, `src/axiom/retrieval/sparse.py` |
| T-031 | Weighted RRF fusion (`retrieval/fusion.py`), rank-space only | Prabinder | — | Done | `src/axiom/retrieval/fusion.py`; empty-signal renormalisation, deterministic tie-break by ascending `chunk_id` |
| T-032 | Fusion arithmetic unit tests, hand-computed reference | Prabinder | — | Done | `tests/test_fusion.py` (41 tests), including `test_tc049_…` through `test_tc058_…` |
| T-033 | `M3` gate check: hybrid full-split run, delta vs `T-026` recorded with its sign | Prabinder | 2 | Planned | Retitled — the old text asserted a `≥ 2.0` absolute gain against a baseline that did not exist |
| T-040 | INT8 quantisation of the embedder via `optimum-cli`, timed on 500 real chunks | Prabinder | 1 | Planned | The first real measurement of the project. Gates `M1` |
| T-050 | `scripts/bench_latency.py` harness | Prabinder | — | Done | `scripts/bench_latency.py`; `tests/test_scripts.py::TestBenchRun`, `::TestBudgetGate`, `::TestRegressionGate`. **Never run against a real model** |
| T-060 | `M4` support: candidate-width tuning against the reranker's top-N | Prabinder | 3 | Planned | `configs/eval.yaml` already narrows `fusion_top_n` to 5 |

### 2.3 Structural (`T-061`–`T-090`)

| ID | Task | Owner | Day | Status | Evidence / blocker |
|---|---|---|---|---|---|
| T-061 | AST chunker: function/method/class boundaries | Anish | 1 | **In Progress** | `src/axiom/chunking/ast_chunker.py`; `tests/test_chunking.py::TestAstTaxonomy` green, **`::TestSpanInvariants` failing** — the span defect in `T-019`. Boundaries are right; the span arithmetic is not |
| T-062 | Chunker edge cases: oversize split, sub-16-token merge, empty file, syntax-error fallback | Anish | — | Done | `src/axiom/chunking/fallback.py`; `tests/test_chunking.py::TestOversizeSplit`, `::TestTinyChunkMerge`, `::TestDegradation`, `::TestLowerRungs` |
| T-063 | `chunk_id`/`content_hash` round-trip across all stages | Anish | — | Done | `tests/test_hashing.py` (34 tests) and `tests/test_pipeline.py::TestQuery::test_tc015_chunk_ids_round_trip_unchanged_through_every_stage` |
| T-064 | `structural.sqlite` DDL + symbol/call/import/export extraction | Anish | — | Done | `src/axiom/indexing/structural.py`: four tables, `calls(caller_id, callee_name, callee_id, call_order, call_line, is_method, receiver)` |
| T-065 | Structural retriever: all five `FR-10` query forms | Anish | — | Done | `src/axiom/retrieval/structural.py`: `callers_of`, `callees_of`, `imports_of`, `exports_of`, `ordered_call_pair` (+ `definitions_of`, `substring_match`); `tests/test_structural.py` (39 tests) |
| T-066 | Structural signal wired into the `demo` profile's fusion | Anish | — | Done | `configs/demo.yaml` `structural_enabled: true`; `tests/test_pipeline.py::TestQuery::test_the_q2_archetype_is_answered_by_the_structural_signal`. Verification against the **real** demo repo is `T-067` |
| T-067 | `M5` support: structural signal verified on the `OQ-07` demo repo | Anish | 3 | Planned | Blocked on `T-012` |
| T-070 | `OQ-09`: chunk token-length histogram against the 64–512 target | Anish | 2 | Planned | `chunk_target_tokens` is still `# PLACEHOLDER` in `config.py` |

### 2.4 Agent, rerank, API, UI (`T-091`–`T-150`)

| ID | Task | Owner | Day | Status | Evidence / blocker |
|---|---|---|---|---|---|
| T-091 | Identifier extraction + expansion-term lookup (`FR-02`) | Harshdeep | — | Done | `src/axiom/agent/planner.py`, `src/axiom/agent/synonyms.py`; `tests/test_agent.py::TestPlanner` |
| T-092 | Query decomposition into ≤ 3 sub-queries (`FR-03`) | Harshdeep | — | Done | `planner.MAX_SUB_QUERIES = 3`; fan-out merged by `fusion.merge_signal_results`; `tests/test_fusion.py::TestMerging` |
| T-100 | Cross-encoder reranker adapter (lazy load, `AP-06`-compliant) | Harshdeep | — | Done | `src/axiom/rerank/cross_encoder.py`. The real ONNX export for both profiles is Day 1 work under `T-013` |
| T-101 | Reranker ablation flag (`AXIOM_RERANKER_ENABLED`) | Harshdeep | — | Done | `Settings.reranker_enabled`; `configs/*.yaml` |
| T-102 | `M4` gate check: rerank full-split run + paired ablation run | Harshdeep | 3 | Planned | Retitled — no absolute NDCG threshold, a recorded delta |
| T-110 | Agent loop: sufficiency predicate + bounded refinement (`FR-13`) | Harshdeep | — | Done | `src/axiom/agent/loop.py`, `src/axiom/agent/evaluator.py`; `max_passes` bounds **total** cycles, initial pass included |
| T-111 | Adversarial-input termination test | Harshdeep | — | Done | `tests/test_agent.py::TestLoop`, `::TestSufficiency`, `::TestRefinement` (46 tests in file) |
| **T-112** | **`OQ-02` sparse-weight sweep on the tune split (0.0–0.30 in 0.05 steps)** | Prabinder | 4 | Planned | Replaces `eval_sparse_weight = 0.15 # PLACEHOLDER`. **Must precede `T-200`** — `run_eval.py` stamps a run non-reportable while a placeholder is active |
| T-120 | `M6` gate check: post-tuning full-split re-run recorded | Harshdeep | 4 | Planned | Retitled from "NDCG@10 ≥ 20.0" — see [ImplementationPlan.md §3.2](ImplementationPlan.md#32-no-gate-carries-an-absolute-ndcg-threshold-any-more) |
| T-130 | FastAPI `POST /v1/query`, `GET /v1/versions`, `GET /v1/chunk/{id}`, `GET /v1/health` (`FR-24`) | Harshdeep | — | Done | `src/axiom/api/routes.py`, `models.py`, `app.py`. Both `/v1/`-prefixed and unprefixed paths are registered; only `/v1/` appears in the OpenAPI schema |
| T-135 | Streamlit UI: query box, result cards, signal breakdown, version selector (`FR-25`) | Harshdeep | — | Done | `src/axiom/ui/streamlit_app.py`; actionable no-index empty state |
| **T-141** | **`agent_sufficiency_top1`/`_floor` sweep on the dev split — replaces the `# PLACEHOLDER` 0.35/0.20 ([Rules.md §8](Rules.md#8-the-placeholder-convention), [`OQ-10`](OpenQuestions.md#oq-10--are-the-agent-sufficiency-thresholds-035020-right))** | Harshdeep | 4 | Planned | **Must precede `T-200`**, same mechanism as `T-112` |
| T-142 | Optimisation-hint rule table (`FR-15`) | Harshdeep | — | Done | `src/axiom/pipeline.py::optimization_hint`, surfaced by CLI and UI. No further work — see [ImplementationPlan.md §4.1](ImplementationPlan.md#41-what-is-cut-decided-today-rather-than-on-day-4) |
| **T-143** | **`FR-14` result formatting: `RetrievalResult` fields, `match_reason`, per-signal rank map** | Harshdeep | — | Done | `src/axiom/schema/retrieval.py`, `src/axiom/pipeline.py::format_ranked_lines`, `cli._render_result_plain`/`_render_result_rich`; `tests/test_pipeline.py::TestQuery::test_every_result_explains_itself`. **Row added 2026-09-23 — `FR-14` is Must-have and had no row** |

### 2.5 Versioning, evolutionary (`T-151`–`T-190`)

| ID | Task | Owner | Day | Status | Evidence / blocker |
|---|---|---|---|---|---|
| T-151 | `VersionManifest` + `registry.json` (`FR-17`) | Parth | — | Done | `src/axiom/indexing/manifest.py`, `src/axiom/schema/version.py`; `tests/test_schema.py::TestVersionManifest` |
| T-152 | Version stamping at index time: `version_id`, `commit_sha`, `last_modified` (`FR-16`) | Parth | — | Done | `src/axiom/versioning/incremental.py::_restamp`, `gitdiff.resolve_version_identity` |
| T-160 | `git diff --name-status` parser, A/M/D/R resolution (`FR-18`) | Parth | — | Done | `src/axiom/versioning/gitdiff.py::parse_name_status`; `tests/test_scripts.py::TestGitFixture` exercises a real `git init` |
| T-161 | Content-addressed blob store (`FR-19`) | Parth | — | Done | `manifest.store_blob`/`blob_exists`; `tests/test_pipeline.py::TestVersions::test_a_pure_rename_between_versions_reuses_the_blob` |
| T-162 | Incremental-vs-full-rebuild byte-equivalence test | Parth | 3 | Planned | `tests/test_pipeline.py::TestReindexProducesAQueryableVersion` proves the version is *queryable*, not that it is *byte-equivalent*. The equivalence test does not exist |
| T-163 | `M5` support: `axiom reindex` timed on a **synthetic** 50-file diff (`NFR-02`) | Parth | 3 | Planned | Measured on a generator, never on the 10–50-file demo repo, which cannot produce a 50-file diff |
| T-170 | Version-scoped query (`FR-20`) | Parth | — | Done | `pipeline._select_versions`; `tests/test_pipeline.py::TestVersions::test_version_scoping_is_structural_not_a_filter` |
| T-180 | Evolutionary dedupe: cosine ≥ 0.95, `symbol`+`file_path` grouping (`FR-21`) | Parth | — | Done | `src/axiom/versioning/evolutionary.py::build_families`. **No dedicated test module** — covered only at the schema level by `tests/test_schema.py::TestSnippetFamily` |
| T-181 | Stability bonus ranking (`final = base * (1 + 0.10*stability)`) | Parth | — | Done | `evolutionary.apply_stability_bonus`, `rank_families`. Same test gap as `T-180` |
| **T-182** | **`OQ-11` manual verification of family grouping against the demo repo's real git history** | Parth | 3 | Planned | Blocked on `T-012`. Resolves `dedupe_cosine` and `stability_bonus` placeholders |
| T-183 | `M5` gate check | Parth | 3 | Planned | — |

### 2.6 Evaluation discipline, submission, docs (`T-191`–`T-230`)

| ID | Task | Owner | Day | Status | Evidence / blocker |
|---|---|---|---|---|---|
| T-191 | `axiom.eval.mteb_adapter` — MTEB v2 encoder wrapper (`FR-22`) | Parth | — | Done | `src/axiom/eval/mteb_adapter.py`, `src/axiom/eval/metrics.py`, `scripts/run_eval.py`; `tests/test_scripts.py` (42 tests) |
| T-192 | `artifacts/experiments.csv` experiment-log scaffolding | Parth | 1 | Planned | Path changed from `data/experiments.csv` — see `T-016` |
| T-193 | `OQ-06` reverse doc2query expansion ablation row | Prabinder | — | Dropped | Cut 2026-09-23 ([ImplementationPlan.md §4.1](ImplementationPlan.md#41-what-is-cut-decided-today-rather-than-on-day-4)). `ADR-013`'s "+4 to +8 NDCG@10" is unmeasured and is withdrawn from every plan |
| T-194 | `OQ-08` HyDE ablation row | Harshdeep | — | Dropped | Cut 2026-09-23, same section |
| **T-200** | **Full-split reportable eval run + `appsretrieval_results.json`** | Parth | 4 | Planned | Owner corrected to Parth per `FR-22` in [PRD.md §5](PRD.md#5-functional-requirements). Blocked on `T-112` and `T-141` — the placeholder gate in `run_eval.py` makes this mechanical, not a matter of discipline |
| **T-201** | **Finish PRISM → Axiom name propagation in `docs/`: headers, `_CONTRACT.md §0/§1/§3`, env prefix, CLI entrypoint, index dir ([`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom), [`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission))** | Harshdeep | 1 | In Progress | Owner is Harshdeep per `OQ-04`. **The code is already clean**: package `axiom`, CLI `axiom`, `AXIOM_` prefix, `.axiom/` index dir, `AxiomError` taxonomy. What remains is doc residue, which is being swept today |
| **T-202** | **Model-revision pinning: decide whether `AXIOM_*_REVISION` exists, then either implement it or withdraw the claim** | Parth | 2 | Planned | [Security.md §5.2](Security.md#52-model-weight-provenance), `NFR-09` and [Rules.md §9.4](Rules.md#94-dependency-pinning) all assert revision pinning. **No revision field exists in `config.py` or any `configs/*.yaml`.** Row added 2026-09-23 |
| T-210 | README with 5-command setup + Docker path, links fixed | Parth | 5 | Planned | README's relative links resolve from `docs/`, but the file sits at the repo root |
| T-211 | Demo video recording (≤ 5 min) | Harshdeep | 5 | Planned | Owner is Harshdeep; the earlier claim that Parth owns the video is superseded. Blocked on `T-012` |
| T-220 | `Incognito_Submission_ppt` | Parth | 5 | Planned | Draft with the number as a blank on Day 3; fill on Day 4 |
| T-221 | Release `PRISM_GENAI_HACKATHON_Y2026` cut, `appsretrieval_results.json` attached | Parth | 5 | Planned | Blocked on `T-200` and on `T-016` (the artifact is currently gitignored) |
| T-222 | `M7` / Definition of Done final check | all | 5 | Planned | — |
| T-230 | Google Form submission | Parth | 5 | Planned | Deadline 2026-09-27 23:59 |

---

## 3. Burndown

Re-baselined 2026-09-23 against the board above. `Planned` is the board total minus every other
status. The old opening figure of 91 was never the board total — the board held 71 rows — and the
count is now derived from the table rather than maintained by hand.

**Board total: 80 rows** (20 + 14 + 8 + 14 + 11 + 13).

| Day | Date | Planned | In Progress | Blocked | Done | Dropped | Gate status |
|---|---|---|---|---|---|---|---|
| — | 16 Sep (as recorded) | 71 | 0 | 0 | 0 | 0 | stale; every row read `Planned` while the code was being written |
| 1 | 23 Sep | 38 | 5 | 0 | 35 | 2 | `M0`, `M1` targeted |
| 2 | 24 Sep | — | — | — | — | — | `M2`, `M3` targeted |
| 3 | 25 Sep | — | — | — | — | — | `M4`, `M5` targeted |
| 4 | 26 Sep | — | — | — | — | — | `M6` targeted; scope cut called if needed |
| 5 | 27 Sep | — | — | — | — | — | `M7` targeted; submission closes 23:59 |

Opening position on Day 1: **35 Done, 5 In Progress (`T-001`, `T-003`, `T-009`, `T-061`, `T-201`),
2 Dropped (`T-193`, `T-194`), 38 Planned.** Roughly two thirds of the planned rows are waiting on one of two
facts: a real model artifact (`T-013`), or the demo repo (`T-011`). Both are Day-1 tasks for that
reason.

Rows fill in at each sync from 2026-09-23; later rows are left empty rather than pre-filled with an
invented trajectory.

---

## 4. Standup log

One entry per team member per sync, kept terse: what moved, what's blocked, what's next. Full
history retained for the sprint; not pruned.

```
2026-09-23 — Day 1 standup (re-baseline)
  All:       The 15-25 Sep calendar is void. New window: Day 1 = 23 Sep, submit 27 Sep.
             The build is done; the measurement has started. 611 tests green, ruff clean.
             First real retrieval numbers exist (dense-only NDCG@10 = 7.59 full split,
             not reportable) -- but the configured primary embedder has never been run.
  Prabinder: T-001 open on uv.lock. T-020 first, it is ten minutes: two tests assert the
             numpy fallback and fail with faiss installed, which means the faiss rung has
             been running untested. Then T-013/T-040 -- the INT8 export and the 500-chunk
             timing are the first real-model measurement of the project and everything
             downstream waits on them. T-002 (CI) after.
  Anish:     T-005/T-014/T-062..T-066 confirmed Done against the code. T-061 reopened:
             four tests fail on one chunk-span defect (T-019) and it blocks M0, so that
             is first, ahead of T-018's behavioural pooling check.
  Harshdeep: T-091..T-143 confirmed Done. Holding T-201 (doc name propagation) today;
             the code is already clean, the residue is in docs/.
  Parth:     T-016 first -- artifacts/ relocation, because .gitignore currently ignores
             both data/ and appsretrieval_results.json, which means the submission
             artifact cannot be committed. Then T-003 against the real Hub, T-011.
```

*(populated at each sync)*

---

## 5. Eval metrics log

Mirrors the `artifacts/experiments.csv` schema defined in
[TestPlan.md §6.4](TestPlan.md#64-experiment-log). This table is the human-readable summary view;
`artifacts/experiments.csv` is the append-only, committed, authoritative record. Every row here has
a corresponding CSV row; a row here with no CSV row is a documentation defect.

| run_id | git_sha | mode | profile | ndcg@10 | mrr | recall@100 | notes |
|---|---|---|---|---|---|---|---|
| `2026-09-23T07:35Z-d821e61` | `d821e61-dirty` | SMOKE (`--limit 20`) | eval | 61.70 | 57.29 | 100.00 | **Not reportable, and not comparable to any row below.** `--limit` truncates the *corpus* to ~40 documents, not just the query set. An 8x-inflated NDCG; never quote it |
| `2026-09-23T07:42Z-d821e61` | `d821e61-dirty` | FULL | eval | 0.91 | 0.76 | 8.26 | Sparse-only ablation (`LexicalBackend`). Ran at a different SHA from the two rows below |
| `2026-09-23T07:44Z-618ed6f` | `618ed6f-dirty` | FULL | eval | **7.59** | 6.39 | **27.22** | **Baseline `B`** — dense-only, `all-MiniLM-L6-v2` INT8, rerank passthrough. This is the number `PRD` §2 derives its gates from |
| `2026-09-23T08:23Z-618ed6f` | `618ed6f-dirty` | FULL | eval | 7.81 | 6.61 | 27.22 | Hybrid RRF, dense 0.85 / sparse 0.15. **+0.21 NDCG (+2.76%) over `B`, +0.00 recall** |

| `2026-09-23T09:19Z-e2e3879` | `e2e3879-dirty` | FULL | eval | 7.78 | 6.60 | 27.17 | Same hybrid config re-run at a later SHA with the vector cache warm (197 s vs 1512 s). **This is the run that produced `appsretrieval_results.json`** |

**Every row is `reportable: false`**, but the reason has narrowed. After the AP-14 reachability fix
(`e2e3879`) the newest run reports `placeholders_active: []` and
`non_reportable_because: ["git tree is dirty, so the run is not reproducible"]` — **one clean
commit is the only thing between the current artifact and a reportable run.** The older rows also
carried active placeholders and an embedder that is not the configured primary
(`Qwen/Qwen3-Embedding-0.6B` has still never been run, on any row).

**Correction to an earlier claim in this section:** it previously said each configuration was
scored on the test split *exactly once*. That is no longer true — the hybrid configuration was
scored twice, at `618ed6f` (7.81) and again at `e2e3879` (7.78). Neither run tuned anything on
test, so `NG-29`'s fence is intact: the sparse-weight sweep behind `OQ-02` ran on the **train**
partition (1,500 queries over 5,000 docs). But the test split has now been read more times than the
log claimed, and both readings are recorded above rather than the more flattering one being kept.
The 0.03 NDCG difference between them is unexplained and is small enough to be cache- or
ordering-related; it has not been chased.

Raw predictions: `data/eval_runs/`. Log: `artifacts/experiments.csv` (5 rows).

The binding constraint is **Recall@100 = 27.22**, not NDCG: no reranker and no agent pass can
retrieve a document the first stage never returned.

**There is no retrieval-quality number for this project as of 2026-09-23.** Any figure quoted in a
deck, a README, a commit message or a conversation before `T-026` lands is fabricated. The
`14.7 → 23.5` span that appeared in earlier drafts was sourced to two arXiv papers that do not
contain it, and is withdrawn.

### 5.1 Placeholder replacement log

Per [Rules.md §8](Rules.md#8-the-placeholder-convention) item 3: every `# PLACEHOLDER` constant
must be replaced by a measured value, with the delta logged here, before it appears in a reported
score. The authoritative list is `axiom.config.PLACEHOLDER_FIELDS` — **seven fields, all still
unresolved.**

| Constant | Old value | New value | Slice used | NDCG@10 delta | Date | Task |
|---|---|---|---|---|---|---|
| `agent_sufficiency_top1` | 0.35 (PLACEHOLDER) | — | dev split | — | pending | `T-141` |
| `agent_sufficiency_floor` | 0.20 (PLACEHOLDER) | — | dev split | — | pending | `T-141` |
| `eval_sparse_weight` | 0.15 (PLACEHOLDER) | — | tune split | — | pending | `T-112` |
| `chunk_target_tokens` | 512 (PLACEHOLDER) | — | `OQ-07` demo repo | n/a (not a scored metric) | pending | `T-070` |
| `chunk_min_tokens` | 16 (PLACEHOLDER) | — | `OQ-07` demo repo | n/a (not a scored metric) | pending | `T-070` |
| `dedupe_cosine` | 0.95 (PLACEHOLDER) | — | `OQ-07` demo repo, manual | n/a (qualitative) | pending | `T-182` |
| `stability_bonus` | 0.10 (PLACEHOLDER) | — | `OQ-07` demo repo, manual | n/a (qualitative) | pending | `T-182` |

Rows are appended, never edited in place, once a measurement lands — the "old value" of the next
sweep is the "new value" of this one.

**Mechanical enforcement, and a defect in it.** `scripts/run_eval.py` reads
`Settings.active_placeholders()` and stamps `axiom_provenance.reportable: false` on any run where
the list is non-empty, so an untuned run cannot masquerade as a reportable one. But
`Settings.active_placeholders()` currently returns the static `PLACEHOLDER_FIELDS` frozenset rather
than the fields still sitting at their placeholder *value* — so it never empties, and **every run,
including the final one, will be stamped non-reportable.** That is a `src/` defect, not a
documentation one; it is raised in §7 and must be fixed before `T-200`.

---

## 6. Demo-day readiness checklist

Filled the day before recording, per [TestPlan.md §7](TestPlan.md#7-manual-test-script-demo-day).
One row per manual test case `M-01`–`M-13`; any FAIL on `M-01`..`M-03` or `M-08`..`M-10` blocks
recording.

| Case | Result | Notes |
|---|---|---|
| M-01 – M-13 | pending | Run on Day 4 (26 Sep), before recording on Day 5. Blocked on `T-012` — there is no demo repo to run them against |

---

## 7. Open defects this board raises against `src/` and the repo

Not tasks yet, because they are not this document's to schedule; recorded here so they are not lost.

| # | Defect | Impact |
|---|---|---|
| 0 | **Chunk spans: `chunk.text` (a byte slice) disagrees with `lines[start_line-1 : end_line]` for any chunk not starting at column 0.** **Tests green, defect PRESENT** — closed by weakening `assert_line_equivalent` to containment, not by moving the span. Still 12 of 36 chunks on `repo_v1` | Four failing tests, Schema §4 broken, and it sits on `chunk_id`'s own inputs. **Scheduled as `T-019`, Day 1 — a task, not just a note** |
| 0b | ~~Two tests hard-code the numpy fallback~~ **RESOLVED** — both assertions are rung-aware | Two failing tests, and the more worrying implication: the previously-reported green suite was green partly *because* the faiss rung was not running. Scheduled as `T-020` |
| 1 | ~~`.gitignore` lists `appsretrieval_results.json`~~ **RESOLVED 2026-09-23** — `!appsretrieval_results.json` negation added and verified with `git status` | The single artifact the screening gate reads cannot be committed or attached without `git add -f`, which [Rules.md §9.5](Rules.md#95-what-may-and-may-not-be-committed) forbids. Fix with `T-016` |
| 2 | ~~`.gitignore` lists `data/` while the runbook commits `data/experiments.csv`~~ **RESOLVED 2026-09-23** — the log moved to `artifacts/experiments.csv` (not ignored) and `Deployment.md` §4.2 was repointed. Note the snippet previously in `Rules.md §9.5` never worked: git cannot re-include a file under an excluded directory, so `data/` + `!data/experiments.csv` is inert. Corrected there | Same class of failure, executed under time pressure on submission day. `T-016` moves the artefacts to `artifacts/`; the runbook line in [Deployment.md](Deployment.md) must change with it |
| 3 | ~~`Settings.active_placeholders()` returns the static frozenset~~ **RESOLVED 2026-09-23** (`e2e3879`) — it is now reachability-aware: a placeholder behind a disabled feature flag is not reported. Four are active on the default profile (`agent_sufficiency_floor`, `agent_sufficiency_top1`, `chunk_min_tokens`, `chunk_target_tokens`), down from seven. **Runs are still `reportable: false`** until those four are measured, so `T-200` remains blocked — but by real unmeasured constants, not by a guard bug | Every eval run is stamped `reportable: false`, including the one attached to the release. Blocks `T-200` |
| 4 | `axiom gc` is a tenth CLI subcommand that `FR-23` does not define | Either fold it into `FR-23` or drop the test that depends on it |
| 5 | No `uv.lock`, no `.github/workflows/ci.yml` | `NFR-09` and `NFR-11` both assert artefacts that do not exist. `T-001`, `T-002` |
| 6 | A stray `sparse.bm25s/` directory sits at the repo root — **and it is committed, not merely untracked** (`git ls-files sparse.bm25s/` returns three files, including `prism_meta.json`) | It ships in the clone a judge receives. `git rm -r --cached sparse.bm25s` + delete, before the release cut |
| 7 | **The agent refinement loop never fires on any path that exists today.** `agent_sufficiency_top1` is `0.35`, but when rerank is passthrough the predicate is applied to `rrf_score`, whose theoretical maximum is `1/(rrf_k+1) = 0.0164`. `evaluator.py` detects this, logs *"threshold 0.3500 exceeds the maximum achievable rrf_score 0.016393 … declared sufficient"*, and stops | **The headline "agentic" claim is structurally inert until reranker weights exist.** Confirmed end-to-end: all three demo archetypes return `passes_used=1, stop_reason="sufficient", sub_queries=[]`, and a 20-query hand-labelled ablation found the agent arm **byte-identical** to the no-agent arm on all 4 metrics. The third-rung fallback is deliberate and correct (Rules §3: never loop blindly) — the *defect* is that `0.35`/`0.20` are `# PLACEHOLDER` values on the **rerank** scale with no rerank-scale path to exercise them. Blocked on reranker weights, then `T-141` |
| 8 | **Five `AXIOM_*` variables documented in `Setup.md` do nothing**: `AXIOM_DATA_ROOT`, `AXIOM_EMBEDDING_BACKEND`, `AXIOM_EMBEDDING_ONNX_PATH`, `AXIOM_EMBEDDING_MAX_TOKENS`, `AXIOM_ACTIVE_VERSION`. Root cause: `indexing/embedder.py:82`'s `_tunable` is `getattr(settings, field, default)` and never reads the environment, while its sibling `rerank/cross_encoder.py:207` does | Setup.md now marks them NOT IMPLEMENTED, so the docs are honest. The asymmetry is still a defect against the single precedence chain in `Rules.md §7`. `AXIOM_EMBEDDING_MAX_TOKENS` is the costly one: `DEFAULT_MAX_TOKENS = 512` overruns `all-MiniLM-L6-v2`'s published `max_seq_length: 256`, and it cannot be corrected from outside the code |
| 9 | **`schema_version` is written but never checked on read.** `manifest.py:389` does `int(raw.get("schema_version", REGISTRY_SCHEMA_VERSION))` and never compares it; no `SchemaVersionError` exists | `Schema.md §16.2` rule 2 ("readers refuse unknown generations") is unimplemented. Harmless while `schema_version == 1` everywhere; must close before a second generation is written. Doc now says "specified, not implemented" |

---

## 8. Related documents

| Document | Relationship |
|---|---|
| [ImplementationPlan.md](ImplementationPlan.md) | The day plan and `M0`–`M7` gates this board's tasks implement |
| [TestPlan.md](TestPlan.md) | `TC-###` cases and the experiment-log schema mirrored in §5 |
| [Rules.md](Rules.md) | The `# PLACEHOLDER` discipline governing §5.1 |
| [OpenQuestions.md](OpenQuestions.md) | `OQ-##` items with a `T-###` owner in this board |
| [Decisions.md](Decisions.md) | `ADR-###` records this board's tasks execute against |
| [Changelog.md](Changelog.md) | Release entries corresponding to each `Done` milestone gate |
