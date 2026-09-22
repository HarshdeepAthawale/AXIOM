# Tracker

The `T-###` task board, burndown, standup log, and eval metrics log for PRISM.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-16
**Status:** Draft

Related: [ImplementationPlan.md](ImplementationPlan.md) · [PRD.md](PRD.md) · [TestPlan.md](TestPlan.md) · [Rules.md](Rules.md) · [OpenQuestions.md](OpenQuestions.md) · [Decisions.md](Decisions.md) · [Changelog.md](Changelog.md)

---

## 1. How this board works

Task ids are permanent and non-sequential-by-design: blocks of 40–60 numbers are reserved per
workstream so a task's id alone tells you roughly where it lives, and so a specific id referenced
elsewhere in the doc suite (`T-141` in [Rules.md §8](Rules.md#8-the-placeholder-convention),
`T-201` in [`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission) and
[`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom)) has a stable, predictable home.

| Block | Workstream | Owner |
|---|---|---|
| `T-001`–`T-020` | Foundation (M0) | shared |
| `T-021`–`T-060` | Retrieval core (dense, sparse, fusion, eval harness) | Prabinder |
| `T-061`–`T-090` | Structural (chunking, AST, call graph) | Anish |
| `T-091`–`T-150` | Agent, rerank, API, UI | Harshdeep |
| `T-151`–`T-190` | Versioning, evolutionary retrieval | Parth |
| `T-191`–`T-230` | Evaluation discipline, submission, docs | Parth |

Status vocabulary matches `_CONTRACT.md §9`: `Planned` → `In Progress` → `Blocked` → `Done` →
`Dropped`. A `Blocked` task always names what it is blocked on, either a `T-###`, an `OQ-##`, or an
external fact.

---

## 2. Task board

Seeded from the day plan in [ImplementationPlan.md §4](ImplementationPlan.md#4-day-by-day-plan).
This is the live board; update status here as work lands rather than treating the day plan as the
source of truth for current state — the day plan is the *intent*, this table is the *record*.

### 2.1 Foundation (`T-001`–`T-020`)

| ID | Task | Owner | Target day | Status |
|---|---|---|---|---|
| T-001 | Repo scaffold: `pyproject.toml`, src-layout, `uv.lock` | Prabinder | 1 | Planned |
| T-002 | CI skeleton: ruff → mypy → pytest → smoke index stages | Prabinder | 1 | Planned |
| T-003 | Load `AppsRetrieval` train + test splits via MTEB v2; confirm BEIR shape (`OQ-05`) | Parth | 1 | Planned |
| T-004 | `data/splits/` seeded tune/dev id lists (4,000/1,000) | Parth | 2 | Planned |
| T-005 | tree-sitter + `tree-sitter-javascript` install; parse one real `.js` file | Anish | 1 | Planned |
| T-006 | `src/axiom/schema/` skeleton: enums, `ChunkLocation`, `ChunkMetadata`, `Chunk` | Harshdeep | 1 | Planned |
| T-007 | `src/axiom/core/hashing.py`: `blake2b_128`, `compute_chunk_id`, `compute_content_hash` | Harshdeep | 1 | Planned |
| T-008 | Heuristic query classifier stub (no LLM) | Harshdeep | 1 | Planned |
| T-009 | Embedder loads and encodes one batch (fp32 first, ONNX export deferred) | Prabinder | 2 | Planned |
| T-010 | `M0` gate check: `uv sync --frozen`, `pytest --collect-only`, MTEB v2 API check all green | Prabinder | 2 | Planned |
| T-011 | `OQ-07`: candidate multi-version JS repo shortlisted | Parth | 2 | Planned |
| T-012 | `OQ-07`: demo repo finalised, staged under `data/demo_repo/` | Parth | 3 | Planned |
| T-013 | ONNX export pipeline dry run (fp32 export, no quantisation yet) | Prabinder | 2 | Planned |
| T-014 | Fixture repo (`tests/fixtures/repo_v1/`) authored per [TestPlan.md §1.3](TestPlan.md#13-fixtures) | Anish | 2 | Planned |
| T-015 | `FakeEmbedder`, `FakeCrossEncoder`, `FakeLLM` test doubles (`tests/fakes.py`) | Harshdeep | 2 | Planned |

### 2.2 Retrieval core (`T-021`–`T-060`)

| ID | Task | Owner | Target day | Status |
|---|---|---|---|---|
| T-021 | Dense index builder: embed + L2-normalise + `dense.faiss` + `dense.idmap.json` | Prabinder | 3 | Planned |
| T-022 | Dense index over full APPS corpus (8,765 docs) | Prabinder | 3 | Planned |
| T-023 | `axiom eval --limit 200` smoke run produces a number (harness proven) | Prabinder | 3 | Planned |
| T-024 | `OQ-03`: dense-only NDCG@10 comparison, Qwen3-0.6B vs MiniLM fallback | Prabinder | 4 | Planned |
| T-025 | `M1` gate check | Prabinder | 3 | Planned |
| T-030 | Sparse index builder: `bm25s` over the chunk corpus, code-aware tokenizer | Prabinder | 4 | Planned |
| T-031 | RRF fusion implementation (`retrieval/fusion.py`), rank-space only | Prabinder | 4 | Planned |
| T-032 | Fusion arithmetic unit tests, hand-computed reference (`TC-049`–`TC-058`) | Prabinder | 4 | Planned |
| T-033 | `M2` gate check: full-split NDCG@10 vs. `M1` baseline, ≥ 2.0 absolute gain | Prabinder | 4 | Planned |
| T-040 | INT8 quantisation of the embedder via `optimum-cli` | Prabinder | 5 | Planned |
| T-050 | `scripts/bench_latency.py` harness | Prabinder | 5 | Planned |
| T-060 | `M3` support: candidate-width tuning against the reranker's top-25 | Prabinder | 5 | Planned |

### 2.3 Structural (`T-061`–`T-090`)

| ID | Task | Owner | Target day | Status |
|---|---|---|---|---|
| T-061 | AST chunker: function/method/class boundaries | Anish | 2 | Planned |
| T-062 | Chunker edge cases: oversize split, sub-16-token merge, empty file, syntax error fallback | Anish | 3 | Planned |
| T-063 | `chunk_id`/`content_hash` round-trip tests across all stages (`TC-014`–`TC-018`) | Anish | 3 | Planned |
| T-064 | `structural.sqlite` DDL + symbol/call/import/export extraction | Anish | 4 | Planned |
| T-065 | Structural retriever: callers-of, callees-of, ordered-call-pair queries (`FR-10`) | Anish | 5 | Planned |
| T-066 | Structural signal wired into `demo.yaml` fusion | Anish | 6 | Planned |
| T-067 | `M4` gate check: no regression on `eval` profile, Recall@100 ≥ 65.0 on `demo` | Anish | 6 | Planned |
| T-070 | `OQ-09`: chunk token-length histogram against the 64–512 target | Anish | 7 | Planned |

### 2.4 Agent, rerank, API, UI (`T-091`–`T-150`)

| ID | Task | Owner | Target day | Status |
|---|---|---|---|---|
| T-091 | Identifier extraction + expansion-term lookup (`FR-02`) | Harshdeep | 3 | Planned |
| T-092 | Query decomposition into ≤ 3 sub-queries (`FR-03`) | Harshdeep | 3 | Planned |
| T-100 | Cross-encoder reranker adapter (lazy load, `AP-06`-compliant) | Harshdeep | 5 | Planned |
| T-101 | Reranker ablation flag (`AXIOM_RERANKER_ENABLED`) | Harshdeep | 5 | Planned |
| T-102 | `M3` gate check: full-split NDCG@10 ≥ 18.0 | Harshdeep | 5 | Planned |
| T-110 | Agent loop: sufficiency predicate + bounded refinement (`FR-13`) | Harshdeep | 6 | Planned |
| T-111 | Adversarial-input termination test (`TC-086`) | Harshdeep | 6 | Planned |
| **T-112** | **`OQ-02` sparse-weight sweep on the tune split (0.0–0.30 in 0.05 steps)** | Prabinder | 7 | Planned |
| T-120 | `M5` gate check: full-split NDCG@10 ≥ 20.0 | Harshdeep | 6 | Planned |
| T-130 | FastAPI `POST /query`, `GET /versions`, `GET /chunk/{chunk_id}`, `GET /health` (`FR-24`) | Harshdeep | 8 | Planned |
| T-135 | Streamlit UI: query box, result cards, signal breakdown, version selector (`FR-25`) | Harshdeep | 9 | Planned |
| **T-141** | **`AXIOM_AGENT_SUFFICIENCY_TOP1`/`_FLOOR` sweep on the 300-query dev slice — replaces the `# PLACEHOLDER` values 0.35/0.20 with measured ones ([Rules.md §8](Rules.md#8-the-placeholder-convention), [`OQ-10`](OpenQuestions.md#oq-10--are-the-agent-sufficiency-thresholds-035020-right))** | Harshdeep | 7 | Planned |
| T-142 | Optimisation-hint rule table (`FR-15`) — first on the cut list, build only if `M5` lands early | Harshdeep | 9 | Planned |

### 2.5 Versioning, evolutionary (`T-151`–`T-190`)

| ID | Task | Owner | Target day | Status |
|---|---|---|---|---|
| T-151 | `VersionManifest` + `registry.json` (`FR-17`) | Parth | 4 | Planned |
| T-152 | Version stamping at index time: `version_id`, `commit_sha`, `last_modified` (`FR-16`) | Parth | 4 | Planned |
| T-160 | `git diff --name-status` parser, A/M/D/R resolution (`FR-18`) | Parth | 5 | Planned |
| T-161 | Content-addressed blob store (`FR-19`) | Parth | 6 | Planned |
| T-162 | `TC-076`: incremental-vs-full-rebuild byte-equivalence test | Parth | 6 | Planned |
| T-163 | `M6` support: incremental reindex ≤ 45 s for a 50-file diff on the `OQ-07` demo repo | Parth | 8 | Planned |
| T-170 | Version-scoped query (`FR-20`) | Parth | 6 | Planned |
| T-180 | Evolutionary dedupe: cosine ≥ 0.95, `symbol`+`file_path` grouping (`FR-21`) | Parth | 8 | Planned |
| T-181 | Stability bonus ranking (`final = base * (1 + 0.10*stability)`) | Parth | 8 | Planned |
| **T-182** | **`OQ-11` manual verification of family grouping against the `OQ-07` repo's real git history** | Parth | 8 | Planned |
| T-183 | `M6` gate check | Parth | 8 | Planned |

### 2.6 Evaluation discipline, submission, docs (`T-191`–`T-230`)

| ID | Task | Owner | Target day | Status |
|---|---|---|---|---|
| T-191 | `axiom.eval.mteb_adapter` — `AbsEncoder` wrapper (`FR-22`) | Parth | 5 | Planned |
| T-192 | `data/experiments.csv` experiment log scaffolding ([TestPlan.md §6.4](TestPlan.md#64-experiment-log)) | Parth | 5 | Planned |
| **T-193** | **`OQ-06` reverse doc2query expansion ablation row** | Prabinder | 7 | Planned |
| T-194 | `OQ-08` HyDE ablation row (only if `T-112`/`T-141`/`T-193` land early) | Harshdeep | 8 | Planned |
| T-200 | Full-split reportable eval run + `appsretrieval_results.json` | Parth | 9 | Planned |
| **T-201** | **Finish PRISM → Axiom name propagation: doc headers, `Prism*` error classes → `Axiom*`, `_CONTRACT.md` CLI entrypoint name ([`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom), [`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission))** | Harshdeep | 9 | Planned |
| T-210 | README with 5-command setup + Docker path | Parth | 9 | Planned |
| T-211 | Demo video recording (≤ 5 min) | Harshdeep | 9 | Planned |
| T-220 | `Incognito_Submission_ppt` | Parth | 10 | Planned |
| T-221 | Release `PRISM_GENAI_HACKATHON_Y2026` cut, `appsretrieval_results.json` attached | Parth | 10 | Planned |
| T-222 | `M7` / Definition of Done final check ([PRD.md §2.1](PRD.md#21-definition-of-done)) | all | 10 | Planned |
| T-230 | Google Form submission | Parth | 10 | Planned |

---

## 3. Burndown

Updated at each daily standup. `Planned` count is the board total minus `Done` minus `Dropped`.

| Day | Date | Planned | In Progress | Blocked | Done | Dropped | Gate status |
|---|---|---|---|---|---|---|---|
| 1 | 15 Sep | 91 | 0 | 0 | 0 | 0 | `M0` not yet reached |
| 2 | 16 Sep | — | — | — | — | — | — |
| 3 | 17 Sep | — | — | — | — | — | `M1` target |
| 4 | 18 Sep | — | — | — | — | — | `M2` target |
| 5 | 19 Sep | — | — | — | — | — | `M3` target |
| 6 | 20 Sep | — | — | — | — | — | `M4`, `M5` target |
| 7 | 21 Sep | — | — | — | — | — | tuning window |
| 8 | 22 Sep | — | — | — | — | — | `M6` target |
| 9 | 23 Sep | — | — | — | — | — | demo readiness |
| 10 | 24-25 Sep | — | — | — | — | — | `M7` target |

Rows fill in daily starting 2026-09-15; this table is intentionally empty ahead of build start rather
than pre-filled with an invented trajectory.

---

## 4. Standup log

One entry per team member per sync, kept terse: what moved, what's blocked, what's next. Full
history retained for the sprint; not pruned.

```
2026-09-15 — Day 1 standup
  Prabinder: starting T-001/T-002. No blockers.
  Anish:     starting T-005. No blockers.
  Harshdeep: starting T-006/T-007. No blockers.
  Parth:     starting T-003. Watching OQ-05 (train split shape).
```

*(populated daily from build start)*

---

## 5. Eval metrics log

Mirrors the `data/experiments.csv` schema defined in
[TestPlan.md §6.4](TestPlan.md#64-experiment-log). This table is the human-readable summary view;
`data/experiments.csv` is the append-only, committed, authoritative record. Every row here has a
corresponding CSV row; a row here with no CSV row is a documentation defect.

| run_id | git_sha | mode | profile | ndcg@10 | mrr | recall@100 | notes |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | populated from `M1` (Day 3) onward |

### 5.1 Placeholder replacement log

Per [Rules.md §8](Rules.md#8-the-placeholder-convention) item 3: every `# PLACEHOLDER` constant
must be replaced by a measured value, with the delta logged here, before it appears in a reported
score.

| Constant | Old value | New value | Slice used | NDCG@10 delta | Date | Task |
|---|---|---|---|---|---|---|
| `sufficiency_top1_threshold` | 0.35 (PLACEHOLDER) | — | 300-query dev slice | — | pending | `T-141` |
| `sufficiency_floor` | 0.20 (PLACEHOLDER) | — | 300-query dev slice | — | pending | `T-141` |
| sparse weight (`eval.yaml`) | 0.15 (PLACEHOLDER) | — | tune split | — | pending | `T-112` |
| `chunk_target_tokens` | 512 (PLACEHOLDER) | — | `OQ-07` demo repo | n/a (not a scored metric) | pending | `T-070` |
| `dedupe_cosine` | 0.95 (PLACEHOLDER) | — | `OQ-07` demo repo, manual | n/a (qualitative) | pending | `T-182` |
| `stability_bonus` | 0.10 (PLACEHOLDER) | — | `OQ-07` demo repo, manual | n/a (qualitative) | pending | `T-182` |

Rows are appended, never edited in place, once a measurement lands — the "old value" of the next
sweep is the "new value" of this one.

---

## 6. Demo-day readiness checklist

Filled the day before recording, per [TestPlan.md §7](TestPlan.md#7-manual-test-script-demo-day).
One row per manual test case `M-01`–`M-13`; any FAIL on `M-01`..`M-03` or `M-08`..`M-10` blocks
recording.

| Case | Result | Notes |
|---|---|---|
| M-01 – M-13 | pending | run once `M7` prerequisites are green, before recording |

---

## 7. Related documents

| Document | Relationship |
|---|---|
| [ImplementationPlan.md](ImplementationPlan.md) | The day plan and `M0`–`M7` gates this board's tasks implement |
| [TestPlan.md](TestPlan.md) | `TC-###` cases and the experiment-log schema mirrored in §5 |
| [Rules.md](Rules.md) | The `# PLACEHOLDER` discipline governing §5.1 |
| [OpenQuestions.md](OpenQuestions.md) | `OQ-##` items with a `T-###` owner in this board |
| [Decisions.md](Decisions.md) | `ADR-###` records this board's tasks execute against |
| [Changelog.md](Changelog.md) | Release entries corresponding to each `Done` milestone gate |
