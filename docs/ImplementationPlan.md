# Implementation Plan

Workstreams, the day-by-day build plan, milestone gates `M0`–`M7`, and the risk register `RISK-01`–`RISK-12` for the re-baselined **2026-09-23 → 2026-09-27** window.

**Owner:** Prabinder Singh
**Last updated:** 2026-09-23
**Status:** Active — re-baselined

Related: [PRD.md](PRD.md) · [Design.md](Design.md) · [TechSpecifications.md](TechSpecifications.md) · [Schema.md](Schema.md) · [Tracker.md](Tracker.md) · [TestPlan.md](TestPlan.md) · [Decisions.md](Decisions.md) · [OpenQuestions.md](OpenQuestions.md) · [NonGoals.md](NonGoals.md) · [Deployment.md](Deployment.md) · [Changelog.md](Changelog.md)

---

## 0. Re-baseline notice — read this before anything else

The 15–25 Sep calendar that every earlier revision of this document described **is void**. It is
replaced by a five-day window: **Day 1 = 2026-09-23, submission Day 5 = 2026-09-27.** The three
conflicting dates previously given for "Day 10" (24 Sep, 24–25 Sep, 25 Sep) are all void with it;
there is no Day 10 any more. Anything in another document still citing a Day 6–10 date or a
"10-day sprint" is stale against this section.

The re-baseline is not a slip being absorbed — it is a change of subject. **The system is built.**
As of 2026-09-23 the repository contains 69 Python modules (~27k lines) under `src/axiom/`,
`scripts/` and `tests/`, with **611 tests collected, 611 passing and 0 failing** (verified
2026-09-23 on the real stack: tree-sitter, bm25s, faiss-cpu, onnxruntime and real MiniLM ONNX
weights all present), `ruff check` clean and `ruff format --check` clean:

| Landed | Where |
|---|---|
| Schema contract, core primitives, config + 5 profiles | `src/axiom/schema/`, `src/axiom/core/`, `src/axiom/config.py`, `configs/` |
| AST chunker, degradation ladder, token accounting | `src/axiom/chunking/` |
| Dense, sparse and structural **index builders** | `src/axiom/indexing/` |
| Dense, sparse and structural **retrievers** + weighted RRF | `src/axiom/retrieval/` |
| Cross-encoder rerank with degradation to RRF order | `src/axiom/rerank/cross_encoder.py` |
| Bounded agent loop, classifier, planner, sufficiency evaluator | `src/axiom/agent/` |
| Version registry, git-diff incremental reindex, blob reuse, `SnippetFamily` | `src/axiom/versioning/` |
| MTEB v2 adapter, metrics, `scripts/run_eval.py` | `src/axiom/eval/`, `scripts/` |
| Typer CLI, FastAPI service, Streamlit UI | `src/axiom/cli.py`, `src/axiom/api/`, `src/axiom/ui/` |

What is **not** done is the part that is actually scored:

1. **One real model has been run; the configured primary has not.**
   `sentence-transformers/all-MiniLM-L6-v2` INT8 ONNX is live at
   `data/models/onnx/all-minilm-l6-v2-int8/` and has been used for a full eval run. Its pooling was
   verified three ways (against the model's own `1_Pooling/config.json`, against a
   `sentence_transformers` fp32 reference at min per-vector cosine 0.9958, and behaviourally).
   **`Qwen/Qwen3-Embedding-0.6B` — the configured primary on every profile — has never been
   downloaded or run.** No reranker weights exist either, so every number below was produced with
   rerank in passthrough. The `optimum` export tooling named in `Setup.md` §7.2 is still not
   installed and still not declared in any `pyproject.toml` extra.
2. **A retrieval-quality number now exists, and it is a long way below the gate.** On the **full**
   CoIR `AppsRetrieval` test split (8,765 documents, 3,765 queries, no `--limit`):

   | arm | NDCG@10 | MRR@10 | Recall@100 |
   |---|---|---|---|
   | sparse only (`LexicalBackend`) | 0.91 | 0.76 | 8.26 |
   | **dense only — baseline `B`** | **7.59** | 6.39 | **27.22** |
   | hybrid RRF (dense 0.85 / sparse 0.15) | 7.80 | 6.61 | 27.22 |

   Hybrid over our own dense-only baseline is **+0.21 NDCG@10 = +2.76% relative**. All three runs
   are correctly stamped `reportable: false` (7 active placeholders, embedder ≠ configured
   primary, dirty tree). Rows are logged in `artifacts/experiments.csv`; raw predictions are under
   `data/eval_runs/`. There is still no `appsretrieval_results.json`.

   Two facts the gates have to absorb: **Recall@100 is 27.22%**, so roughly three-quarters of
   relevant documents never enter the candidate pool and no reranker or agent pass can reach them;
   and the **sparse leg adds zero recall** (27.224 → 27.224), which is the first evidence bearing
   on `OQ-02`'s placeholder `eval_sparse_weight = 0.15`.
3. **No demo repo exists.** `data/demo_repo/` is empty; `OQ-07` is unresolved.
4. **No PPT and no demo video exist.**
5. **The repo is not reproducible from a clean clone.** There is no `uv.lock` and no
   `.github/workflows/ci.yml`, both of which `NFR-09`/`NFR-11` assert.
6. **The suite is green — and the two defects that made it red are resolved.** Recorded here
   because the history matters for what the suite is worth. As of 2026-09-23 it is **611/611
   passing**, exercising the top rung of every ladder rather than the bare install:
   - **`T-019` — chunk spans (4 failures).** A chunk that does not begin at column 0 — every
     method, every nested function — gets a `start_byte` at the declaration token but a
     `start_line` covering the whole line, so `chunk.text` is the byte slice while the line span is
     wider. The two disagree by the leading indentation, breaking Schema §4. **Fixed.** It
     mattered beyond the test: `chunk_id` binds `file_path` and `start_line`, and every
     `file:line` header the jury sees comes from the same pair. Anything that renders a chunk by
     re-reading the source file between those line numbers — the Streamlit UI, any "jump to line"
     — must still be checked against `chunk.text`, which is what the CLI renders and what was
     scored.
   - **`T-020` — two tests assume `faiss-cpu` is absent (2 failures).** They assert the numpy
     fallback (`dense_backend == "numpy"`, `dense.npy`) and fail once the optional `retrieval`
     extra is installed and the real rung runs. **Fixed** — both assertions are now rung-aware.
     The uncomfortable implication stands and is worth keeping in view: the "400 tests passing"
     reported before 2026-09-23 was green partly because none of the optional backends were
     installed, so the faiss, bm25s and ONNX paths were not being executed at all.

The gates below are therefore re-derived to gate **measurement and packaging**, not construction.
Where a gate used to say "component X lands," it now says "component X produces a number that is
written down."

---

## 1. How this plan is organised

Two axes, kept deliberately separate:

- **Workstreams** (§2) — who owns what area, mapped to the `FR-##`/`NFR-##` requirements in
  [PRD.md §5](PRD.md#5-functional-requirements) each person drives. **[PRD.md §5](PRD.md#5-functional-requirements)'s
  `Owner` column is authoritative**; this section is generated from it and never the reverse.
- **Day plan** (§4) — what must be true by the end of each calendar day, expressed as milestone
  gates `M0`–`M7`, each of which corresponds one-to-one with a version tag in
  [TestPlan.md §6.5](TestPlan.md#65-quality-gates-by-milestone) and a release entry in
  [Changelog.md](Changelog.md).

A milestone gate is **a fact about the system**, checkable by running a command, not a date on a
calendar. If Day 3 arrives and `M4`'s gate is not yet true, the day plan has slipped and the cut
order in [PRD.md §7.1](PRD.md#71-cut-order-under-time-pressure) is what governs the response — not
a renegotiation of what "done" means for `M4`.

---

## 2. Workstreams

Regenerated 2026-09-23 from the `Owner` column of
[PRD.md §5](PRD.md#5-functional-requirements)–[§6](PRD.md#6-non-functional-requirements). Where an
earlier revision of this table disagreed with PRD — it assigned `FR-22` and `FR-26` to Prabinder —
PRD wins and the table below is the corrected version.

| Workstream | Owner | Modules | Requirements driven |
|---|---|---|---|
| Retrieval core | Prabinder Singh | `indexing/dense.py`, `indexing/sparse.py`, `retrieval/dense.py`, `retrieval/sparse.py`, `retrieval/fusion.py`, `cli.py` | `FR-05`–`FR-08`, `FR-11`, `FR-23`; `NFR-01`, `NFR-03`, `NFR-05`, `NFR-06`, `NFR-08`, `NFR-11` |
| Structural intelligence | Anish Grover | `chunking/`, `indexing/structural.py`, `retrieval/structural.py` | `FR-04`, `FR-09`, `FR-10` |
| Agentic orchestration + reranking + surfaces | Harshdeep Athawale | `agent/`, `rerank/`, `api/`, `ui/` | `FR-01`–`FR-03`, `FR-12`–`FR-15`, `FR-24`, `FR-25`; `NFR-04`, `NFR-07`, `NFR-10` |
| Versioning + evaluation + submission | Parth Deshmukh | `versioning/`, `eval/`, `scripts/run_eval.py`, release packaging | `FR-16`–`FR-22`, `FR-26`; `NFR-02`, `NFR-09`, `NFR-12` |

`FR-23` (the Typer CLI) is Prabinder's per PRD §5, even though the CLI is a surface — every gate
command below is a CLI invocation, so it sits with the person who runs the gates.

Cross-cutting, owned jointly and reviewed by all four per [Rules.md §10](Rules.md#10-code-review-rules):
`src/axiom/schema/`, `src/axiom/core/`, `configs/`, this `docs/` suite.

### 2.1 Single-owner submission chain

One person owns the unbroken chain from eval run to submitted form, and may cut scope unilaterally
from Day 4 (26 Sep) onward: **Parth Deshmukh**, per `FR-22` and `FR-26` in PRD §5. The chain is
`scripts/run_eval.py` → `appsretrieval_results.json` → release `PRISM_GENAI_HACKATHON_Y2026` →
Google Form. `T-200`, `T-221` and `T-230` in [Tracker.md](Tracker.md) are all his.

---

## 3. Milestone gates `M0`–`M7`

Gate ids and version tags are unchanged — they are cited by
[TestPlan.md §6.5](TestPlan.md#65-quality-gates-by-milestone), [Changelog.md](Changelog.md) and
[Deployment.md](Deployment.md). Their **conditions** are re-derived, because the code each one used
to gate is already written. What remains to gate is a measurement, written down, on the full split.

| Gate | Tag | Target day | Gate condition | Checked by |
|---|---|---|---|---|
| `M0` | `0.1.0` | Day 1 (23 Sep) | Reproducible from a clean clone: `uv.lock` committed, `uv sync --frozen` green, `.github/workflows/ci.yml` runs stages 1–5 of [TestPlan.md §8](TestPlan.md#8-ci-pipeline), `pytest` 400/400 green, `ruff` + `mypy --strict` clean, **the suite green** (`T-019`, `T-020` — it is not today), `artifacts/` scaffolded. **The code is written; this gate is about the clone and the six failures.** | `T-001`, `T-002`, `T-010`, `T-016`, `T-019` |
| `M1` | — | Day 1 (23 Sep) | **One real model has run.** INT8 embedder exported, 500 real chunks embedded and timed, the behavioural pooling check in [Setup.md](Setup.md) green (two paraphrases score closer than an unrelated string), `scripts/run_eval.py --limit 200` emits a `SMOKE` row. No quality claim attaches to this number. | `T-013`, `T-040`, `T-023` |
| `M2` | `0.2.0` | Day 2 (24 Sep) | **The project's own dense-only baseline exists.** `scripts/run_eval.py --task AppsRetrieval --split test` on `eval.yaml` with `sparse_enabled=false`, `reranker_enabled=false`, full split, no `--limit`; NDCG@10 / MRR@10 / Recall@100 recorded in [Tracker.md §5](Tracker.md#5-eval-metrics-log) with a git SHA. **This number, not a paper's, is the denominator for every gain claimed afterwards.** | `T-026` |
| `M3` | `0.3.0` | Day 2 (24 Sep) | Sparse + weighted RRF on the full split; the delta against `M2`'s recorded number is written down with its sign, whatever the sign is. No absolute threshold. | `T-033` |
| `M4` | `0.4.0` | Day 3 (25 Sep) | Cross-encoder rerank on the full split under `eval.yaml` (`bge-reranker-v2-m3` primary, `fusion_top_n: 5`, `rerank_max_chars: 1024`, as committed in `configs/eval.yaml`); the delta against `M3` recorded. Ablation proven by a paired `AXIOM_RERANKER_ENABLED=false` run. | `T-102` |
| `M5` | `0.5.0` | Day 3 (25 Sep) | **The demo is real.** A small tagged JS repo (10–50 files, `OQ-07`) indexed at ≥ 2 versions under `demo.yaml` (`ms-marco-MiniLM-L-6-v2` primary); the three archetype queries Q1/Q2/Q3 answered live; `axiom reindex` timed against the **synthetic** 50-file diff from `tests/fixtures/gen_v2.py` for `NFR-02`; `--all-versions` collapses one family. | `T-012`, `T-163`, `T-183` |
| `M6` | `0.6.0` | Day 4 (26 Sep) | **Tuning precedes quotation.** The tune-split sweeps (`T-112` sparse weight, `T-141` sufficiency thresholds) have run and their constants are no longer `# PLACEHOLDER`; the full split is re-run once afterwards; the ablation table (dense → +sparse → +rerank → +agent) is assembled from the recorded rows. `scripts/run_eval.py` enforces this mechanically: it stamps `reportable: false` while any placeholder is active. | `T-112`, `T-141`, `T-200` |
| `M7` | `1.0.0` | Day 5 (27 Sep) | Every item of [PRD.md §2.1 Definition of Done](PRD.md#21-definition-of-done) that survived the Day-4 cut is simultaneously true; `appsretrieval_results.json` attached to release `PRISM_GENAI_HACKATHON_Y2026`; PPT and ≤ 5-minute video done; Google Form submitted. | [Deployment.md](Deployment.md) submission runbook |

### 3.1 Why `M2` was inserted where the old `M1` used to sit

The previous ladder made `M1`'s gate an `--limit 200` smoke run and then had `M2` require beating
"the `M1` dense-only baseline by ≥ 2.0." [TestPlan.md §6.2](TestPlan.md#62-why-a-limited-run-is-never-a-reportable-score)
explicitly disqualifies a `--limit` number from being a baseline, so the old `M2` compared against a
number that was not allowed to exist. The chain now separates them: `M1` proves the harness is
wired, `M2` produces the baseline, `M3` measures against it.

### 3.2 No gate carries an absolute NDCG threshold any more

`M3`, `M4` and `M5` previously read `≥ 2.0 absolute`, `≥ 18.0` and `≥ 20.0`. Those were all
calibrated off an unsourced "BGE 0.6B = 14.7" figure that appears in neither cited paper. The
figure is withdrawn. Each gate now requires a **recorded delta against our own measured baseline**,
and the public claim becomes a relative gain plus the `M6` ablation table. Once `M2` lands, the
absolute target may be re-derived from it — in [PRD.md §2](PRD.md#2-goals-and-success-metrics), by
the PRD owner, in writing, with the measurement cited. Until then no absolute number is a gate.

---

## 4. Day-by-day plan

Five days. Each cell is a deliverable, not an area of activity.

| Day | Date | Gate(s) | Prabinder | Anish | Harshdeep | Parth |
|---|---|---|---|---|---|---|
| 1 | 23 Sep | → `M0`, `M1` | Faiss-rung test fix (`T-020`); `uv.lock` + CI workflow (`T-001`, `T-002`); INT8 embedder export and 500-chunk timing (`T-040`); `--limit 200` smoke (`T-023`) | **Chunk-span fix (`T-019`) — four failing tests, blocks `M0`**; then the behavioural pooling check (`T-018`); support `T-022`'s corpus encode | Rerank ONNX export for both profiles (supports `T-013`); doc reconciliation | `artifacts/` relocation (`T-016`); `artifacts/splits/` 4,000/1,000 id lists (`T-004`); `OQ-07` demo repo shortlist (`T-011`) |
| 2 | 24 Sep | → `M2`, `M3` | Full-split **dense-only** baseline run (`T-026`); then sparse + RRF full-split run (`T-033`); `OQ-03` embedder comparison (`T-024`) | Chunk-length histogram against the real corpus (`T-070`) | Degradation matrix rehearsed against real models, not fakes | Record both runs in [Tracker.md §5](Tracker.md#5-eval-metrics-log) and `artifacts/experiments.csv`; `OQ-07` repo finalised and tagged (`T-012`) |
| 3 | 25 Sep | → `M4`, `M5` | Rerank full-split run + paired ablation (`T-102`); `bench_latency.py` real-model run (`T-050`) | Structural signal verified on the real demo repo (`T-067`) | Demo rehearsal on `demo.yaml`; UI and API smoked against a real index; `AXIOM_OFFLINE=true` rehearsed (`RISK-11`) | `axiom reindex` timed on the synthetic 50-file diff (`T-163`); incremental-vs-full equivalence test (`T-162`); `--all-versions` family check (`T-182`) |
| 4 | 26 Sep | → `M6` | Sparse-weight sweep on the **tune** split (`T-112`) | — (support) | Sufficiency-threshold sweep on the **dev** split (`T-141`) | Post-tuning full-split re-run and `appsretrieval_results.json` (`T-200`); ablation table assembled; **scope cut called here if needed** |
| 5 | 27 Sep | → `M7` | Fresh-clone rehearsal (`NFR-09`, `T-017`) | — | Demo video recording (`T-211`) | PPT (`T-220`); release cut + JSON attached (`T-221`); DoD check (`T-222`); Google Form (`T-230`) |

### 4.1 What is cut, decided today rather than on Day 4

Declared dropped up front so nobody spends a day on them:

- **The "real 10k-file JS repo" claim is deleted.** The demo corpus is a small tagged JS repo of
  10–50 files. `NFR-02`'s 50-file diff is measured on the synthetic generator
  (`tests/fixtures/gen_v2.py`), not on the demo repo, and the two are never conflated in the deck.
- **`FR-15`** (`optimization_hint`) ships as whatever the already-written rule table in
  `pipeline.optimization_hint` does. No further work.
- **HyDE (`OQ-08`), doc2query (`OQ-06`/`ADR-013`)** — not attempted. `ADR-013` stays `Proposed` and
  the "+4 to +8 NDCG@10" estimate it carries is withdrawn from every plan; it is an unimplemented,
  unmeasured guess and nothing may be banked against it.
- **Docker** is a nice-to-have on Day 5 only. The submission path is a clean clone plus `uv sync`.

### 4.2 The test-split fence is absolute

`NG-29` is not negotiable under time pressure. The tune split (4,000) and dev split (1,000) come
from `AppsRetrieval` **train**. No holdout is ever carved from test — not on Day 4, not if the
sweeps run out of time. If tuning time runs out, the reported number is the untuned one and the
deck says so. `PRD.md`'s A-6 fallback is being rewritten by the PRD owner to make this
structurally impossible to violate; until that edit lands, this section governs.

---

## 5. Risk register

Twelve risks, each with a category consistent with [NonGoals.md](NonGoals.md)'s category vocabulary
where applicable, an owner, and a mitigation that is a concrete engineering choice already reflected
elsewhere in the docs — not a hope. Re-scored 2026-09-23 against the five-day window.

| ID | Risk | Category | Owner | Status | Mitigation |
|---|---|---|---|---|---|
| `RISK-01` | Embedding model too slow on CPU to hit the cold-index budget | CPU budget | Prabinder | **Open, now the top risk** | INT8 dynamic quantisation via ONNX Runtime; declared fallback `all-MiniLM-L6-v2` at ~40x fewer parameters ([`ADR-012`](Decisions.md#adr-012--onnx-runtime-int8--llamacpp-gguf-for-a-cpu-only-stack)). Day 1 `M1` gate is *exactly* this measurement, ahead of everything else, so the fallback decision is made on data by end of Day 1 |
| `RISK-02` | NDCG@10 lands below a competitive screening score | accuracy | Prabinder | **Open — no number exists yet** | Hybrid retrieval plus reranking is the architecture, not an add-on, and it is already built. `M2`–`M4` record a delta per stage, so a shortfall is attributable to the stage that caused it. The honest fallback is the `M6` ablation table and a relative gain, which is what the 25% technical-depth criterion rewards |
| `RISK-03` | tree-sitter parsing fails on real-world JS (minified bundles, syntax errors, deep nesting) | robustness | Anish | **Mitigated** | Degradation ladder AST → statement split → line window → module chunk is implemented in `chunking/` and covered by `tests/test_chunking.py::TestDegradation` and `::TestLowerRungs` |
| `RISK-04` | Agent loop runs unbounded or blows the query-latency budget | CPU budget | Harshdeep | **Mitigated in code, unmeasured on real models** | Hard caps, not convergence: total passes and a monotonic 5 s deadline ([`ADR-007`](Decisions.md#adr-007--bounded-agent-loop-hard-caps-not-convergence)), implemented in `agent/loop.py` and covered by `tests/test_agent.py::TestLoop`. Residual risk is the real-model constant factor, measured on Day 3 |
| `RISK-05` | CoIR dataset shape or MTEB API version mismatch breaks the eval harness | integration | Parth | **Open** | `eval/mteb_adapter.py` accepts every MTEB v2 corpus/query shape and falls back to a vendored BEIR-shaped directory under `data/datasets/`. Must be exercised against the real Hub on Day 1, not Day 4 |
| `RISK-06` | Incremental reindex too slow to demonstrate live | CPU budget | Parth | **Mitigated in code** | Content-addressed blob reuse (`FR-19`) means cost scales with changed files; `tests/test_pipeline.py::TestReindexProducesAQueryableVersion` proves the path. The 45 s figure is measured on the synthetic generator on Day 3 |
| `RISK-07` | A rival submission lands a measured P0 score before we do | schedule | Prabinder | **Open** | Sequencing: `M2`'s dense-only full-split number is scheduled for Day 2, before any tuning or polish, so a submittable number exists three days before the deadline regardless of how the rest lands |
| `RISK-08` | Project name collision with a rival submission causes jury confusion | presentation | Harshdeep | **Closed in decision, open in propagation** | Renamed PRISM → Axiom ([`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom)); package, CLI, env prefix and index dir already land as `axiom`/`AXIOM_`/`.axiom` in code. Residual doc residue is `T-201`, due before `M7` |
| `RISK-09` | Tuning discipline slips under deadline pressure and the test split gets touched more than once per config | integrity | Parth | **Open — highest reputational risk** | [`NG-29`](NonGoals.md#ng-29--no-tuning-on-the-benchmark-test-split) fence plus §4.2 above plus the experiment log in [TestPlan.md §6.4](TestPlan.md#64-experiment-log): every test-split run is logged with its config hash, so an at-most-once violation is visible in the log itself |
| `RISK-10` | `faiss-cpu` / `llama-cpp-python` native wheel unavailable blocks a team member's dev environment | tooling | Anish | **Mitigated** | Every heavy dependency is optional in `pyproject.toml` and every import is lazy — `tests/test_pipeline.py::test_no_optional_dependency_is_imported_by_importing_axiom` asserts it. A missing wheel degrades a signal, it does not block a machine |
| `RISK-11` | Judge's machine is offline or network-restricted and a model download stalls mid-demo | demo-day | Harshdeep | **Open** | Mandatory pre-download of primary and fallback weights the day before ([Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before)); `AXIOM_OFFLINE=true` rehearsed on Day 3 so a stalled fetch fails fast and visibly |
| `RISK-12` | A late change to `src/axiom/schema/` breaks multiple workstreams at once | integration | Prabinder | **Largely retired** | The schema landed first and 71 tests in `tests/test_schema.py` pin it. Four-member sign-off still applies to any diff under `src/axiom/schema/` per [Rules.md §10](Rules.md#10-code-review-rules) item 7 |

### 5.1 Risk severity and review cadence

| Severity | Definition | Review cadence |
|---|---|---|
| High | Blocks a Must-have gate (`M0`–`M4`) if it fires | Reviewed at both daily syncs until closed or the gate passes |
| Medium | Blocks `M5`/`M6` or a demo-quality property | Reviewed once daily |
| Low | Affects presentation only | Reviewed on Day 4 |

`RISK-01`, `RISK-02`, `RISK-05`, `RISK-09` are High. `RISK-04`, `RISK-06`, `RISK-07`, `RISK-11` are
Medium. `RISK-03`, `RISK-08`, `RISK-10`, `RISK-12` are Low, having been substantially retired by the
code that now exists. `RISK-11` remains zero-tolerance on demo day specifically — a Low-severity
risk that fires during the five minutes the jury is watching costs disproportionately more than its
rating implies, which is why its mitigation is rehearsed rather than merely documented.

### 5.2 The risk this register does not contain

The dominant failure mode for the next five days is not on the table above, because it is not a
technical risk: **it is spending the window improving a system that already works instead of
measuring it.** Every hour before `M2` lands that is not spent getting a real embedder to produce a
real full-split number is an hour spent on upside. The gate order in §3 exists to make that
explicit.

---

## 6. Dependencies between workstreams

The construction dependency graph is now historical — all of it is built. What remains has a much
shorter chain, and it is almost entirely serial through one person:

```
Real INT8 embedder export  (Day 1, Prabinder)
        │
        ▼
Full-split dense-only baseline  ── M2 ──▶ the denominator for every claim
        │
        ├──▶ +sparse/RRF run     ── M3
        │          │
        │          └──▶ +rerank run  ── M4
        │                     │
        │                     └──▶ tune-split sweeps ──▶ post-tuning re-run ── M6
        │                                                        │
        └──▶ demo repo + demo.yaml ── M5                         ▼
                                                appsretrieval_results.json ── M7

(independent, Parth)  artifacts/{experiments.csv,splits,bench}  ──▶ experiment-log discipline
(independent, Harshdeep)  PPT + video  ──▶ needs only M5's demo, not M6's number
```

The one blocking dependency is the embedder export: nothing downstream of it can start until a real
vector exists. That is why it is the first cell of Day 1 and why `all-MiniLM-L6-v2` is pre-declared
as the fallback rather than debated when the timing comes back.

The PPT and video depend on `M5`'s live demo, not on `M6`'s final number — draft them with the
number as a blank to be filled on Day 4, so a late measurement does not also delay the deck.

---

## 7. Related documents

| Document | Relationship |
|---|---|
| [Tracker.md](Tracker.md) | `T-###` tasks that implement each cell of the day plan in §4, with real statuses against the code on disk |
| [TestPlan.md](TestPlan.md) | The gate conditions in §3 are the same version-tagged rows as `§6.5` there |
| [Decisions.md](Decisions.md) | `ADR-###` records for the architectural choices the mitigations in §5 rely on |
| [OpenQuestions.md](OpenQuestions.md) | `OQ-##` items scheduled into the day plan in §4 |
| [Changelog.md](Changelog.md) | Release entries corresponding to each `M0`–`M7` version tag |
| [Deployment.md](Deployment.md) | The `M7` submission runbook this plan builds toward |
