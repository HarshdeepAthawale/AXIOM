# Implementation Plan

Workstreams, the day-by-day build plan, milestone gates `M0`–`M7`, and the risk register `RISK-01`–`RISK-12` for the 2026-09-15 → 2026-09-27 build window.

**Owner:** Prabinder Singh
**Last updated:** 2026-09-16
**Status:** Draft

Related: [PRD.md](PRD.md) · [Design.md](Design.md) · [TechSpecifications.md](TechSpecifications.md) · [Schema.md](Schema.md) · [Tracker.md](Tracker.md) · [TestPlan.md](TestPlan.md) · [Decisions.md](Decisions.md) · [OpenQuestions.md](OpenQuestions.md) · [NonGoals.md](NonGoals.md) · [Deployment.md](Deployment.md) · [Changelog.md](Changelog.md)

---

## 1. How this plan is organised

Two axes, kept deliberately separate:

- **Workstreams** (§2) — who owns what area, mapped to the `FR-##`/`NFR-##` requirements in
  [PRD.md](PRD.md) each person drives.
- **Day plan** (§3) — what must be true by the end of each calendar day, expressed as milestone
  gates `M0`–`M7`, each of which corresponds one-to-one with a version tag in
  [TestPlan.md §6.5](TestPlan.md#65-quality-gates-by-milestone) and a release entry in
  [Changelog.md](Changelog.md).

A milestone gate is **a fact about the system**, checkable by running a command, not a date on a
calendar. If Day 6 arrives and `M5`'s gate is not yet true, the day plan has slipped and the cut
order in [PRD.md §7.1](PRD.md#71-cut-order-under-time-pressure) is what governs the response — not
a renegotiation of what "done" means for `M5`.

---

## 2. Workstreams

Matches the ownership table in `_CONTRACT.md §0` and the `Owner` column throughout
[PRD.md §5](PRD.md#5-functional-requirements)–[§6](PRD.md#6-non-functional-requirements).

| Workstream | Owner | Modules | Requirements driven |
|---|---|---|---|
| Retrieval core | Prabinder Singh | `indexing/dense.py`, `indexing/sparse.py`, `retrieval/dense.py`, `retrieval/sparse.py`, `retrieval/fusion.py`, `eval/` | `FR-05`–`FR-08`, `FR-11`, `FR-22`, `FR-23`, `FR-26`; `NFR-01`, `NFR-03`, `NFR-05`, `NFR-06`, `NFR-08`, `NFR-11` |
| Structural intelligence | Anish Grover | `chunking/`, `indexing/structural.py`, `retrieval/structural.py` | `FR-04`, `FR-09`, `FR-10` |
| Agentic orchestration + reranking + UI | Harshdeep Athawale | `agent/`, `rerank/`, `api/`, `ui/`, `FR-15` optimisation hints | `FR-01`–`FR-03`, `FR-12`–`FR-15`, `FR-24`, `FR-25`; `NFR-04`, `NFR-07`, `NFR-10` |
| Versioning + evaluation + submission | Parth Deshmukh | `versioning/`, eval-run discipline, release packaging, docs | `FR-16`–`FR-21`; `NFR-02`, `NFR-09`, `NFR-12` |

Cross-cutting, owned jointly and reviewed by all four per [Rules.md §10](Rules.md#10-code-review-rules):
`src/axiom/schema/`, `src/axiom/core/`, `configs/`, this `docs/` suite.

---

## 3. Milestone gates `M0`–`M7`

Each gate's version tag is the same tag used in
[TestPlan.md §6.5](TestPlan.md#65-quality-gates-by-milestone); a gate is not "mostly true," it is
checked by the stated command and is either green or not yet reached.

| Gate | Tag | Target day | Gate condition | Checked by |
|---|---|---|---|---|
| `M0` | — | Day 2 (16 Sep) | `uv sync --frozen` green on a clean clone; `pytest --collect-only` green; MTEB v2 API check passes ([Setup.md §4.4](Setup.md#44-mteb-v2-is-required--not-v1)); tree-sitter parses one real `.js` file; embedder encodes one batch | `T-001`–`T-010` |
| `M1` | `0.2.0` | Day 3 (17 Sep) | Dense index built over the full APPS corpus; `axiom eval --limit 200` smoke run produces *any* NDCG@10 number (harness proven, value not yet judged) | [TestPlan.md §6.5](TestPlan.md#65-quality-gates-by-milestone) row `0.2.0` |
| `M2` | `0.3.0` | Day 4 (18 Sep) | Sparse index + weighted RRF wired end to end; full-split NDCG@10 exceeds the `M1` dense-only baseline by ≥ 2.0 absolute | same table, row `0.3.0` |
| `M3` | `0.4.0` | Day 5 (19 Sep) | Cross-encoder reranker integrated and ablatable (`AXIOM_RERANKER_ENABLED=false` still runs); full-split NDCG@10 ≥ 18.0 | same table, row `0.4.0` |
| `M4` | `0.5.0` | Day 6 (20 Sep) | Structural signal (`structural.sqlite`, `retrieval/structural.py`) wired into the `demo` profile's fusion; no regression on the `eval` profile vs. `M3`; Recall@100 ≥ 65.0 | same table, row `0.5.0` |
| `M5` | `0.6.0` | Day 6 (20 Sep) | Bounded agent loop live (heuristic path and, when enabled, LLM path both green under `NFR-07`'s degradation matrix); full-split NDCG@10 ≥ 20.0 — the `PRD.md §2` target | same table, row `0.6.0` |
| `M6` | — | Day 8 (22 Sep) | P1 acceptance: `US-7`, `US-8` pass (`axiom reindex` ≤ 45 s for a 50-file diff, version-scoped query correct). Bonus acceptance: `US-9` passes (`--all-versions` returns `SnippetFamily`-collapsed results with diffs). Train-split tuning (`FR-26`) has produced at least one recorded weight/threshold delta | [PRD.md §2.1](PRD.md#21-definition-of-done) items 4–5; [Tracker.md](Tracker.md) eval log |
| `M7` | `1.0.0` | Day 10 (25 Sep) | All seven items of [PRD.md §2.1 Definition of Done](PRD.md#21-definition-of-done) simultaneously true; demo surfaces, docs, PPT, and video complete; release `PRISM_GENAI_HACKATHON_Y2026` cut | [PRD.md §2.1](PRD.md#21-definition-of-done); [Deployment.md](Deployment.md) submission runbook |

`M4` and `M5` share a target day because structural indexing (Anish) and the agent loop
(Harshdeep) are independent workstreams that only need to have *landed*, not be sequenced against
each other — they integrate on Day 6 per §4.

---

## 4. Day-by-day plan

Extends the 10-day table already circulated in `PROJECT_OVERVIEW.md §13` with gate references and
explicit per-person deliverables.

| Day | Date | Milestone(s) | Prabinder | Anish | Harshdeep | Parth |
|---|---|---|---|---|---|---|
| 1 | 15 Sep | → `M0` | Repo scaffold, `pyproject.toml`, `uv.lock`, CI skeleton | tree-sitter install + parse a sample `.js` file | Schema skeleton (`src/axiom/schema/`), classifier stub | Dataset reachability check (`OQ-05`), `data/splits/` scaffolding |
| 2 | 16 Sep | `M0` closes | Embedder loads + encodes a batch; ONNX export dry run | AST chunker: function/method/class boundaries on the fixture repo | Heuristic classifier (no LLM) passes `TC-001`–`TC-004` | `OQ-07` demo repo candidate selected |
| 3 | 17 Sep | → `M1` | Dense index over full APPS corpus; `axiom eval --limit 200` smoke | Chunker handles oversize/empty/broken-file edge cases (`TC-019`–`TC-024`) | Query planner: identifier extraction + expansion (`FR-02`) | `OQ-07` demo repo finalised; version tagging (`FR-16`) design |
| 4 | 18 Sep | → `M2` | Sparse index (`bm25s`) + weighted RRF fusion (`TC-049`–`TC-058`) | Structural SQLite schema + symbol/call/import/export extraction | — (supports fusion integration) | Registry + `VersionManifest` (`FR-17`) implementation |
| 5 | 19 Sep | → `M3` | RRF integration hardening, candidate-width tuning infra | Structural retriever: callers-of/callees-of/ordered-pair queries (`FR-10`) | Cross-encoder reranker integrated (`FR-12`), ablation flag | Incremental reindex: `git diff` A/M/D/R resolution (`FR-18`) |
| 6 | 20 Sep | → `M4`, `M5` | Support structural+fusion integration | Structural signal wired into `demo.yaml` fusion; degradation ladder (`TC-021`) | Bounded agent loop live (`FR-13`), sufficiency predicate, `TC-086`–`TC-090` | Content-addressed blob reuse (`FR-19`); `TC-076` byte-equivalence test |
| 7 | 21 Sep | Train-split tuning window | `OQ-02` sparse-weight sweep, `OQ-03` embedder comparison | `OQ-09` chunk-size histogram | `OQ-10` sufficiency-threshold sweep (`T-141`) | `OQ-06` doc2query ablation |
| 8 | 22 Sep | → `M6` | — | `OQ-11` family-grouping manual verification | FastAPI `POST /query` (`FR-24`) | Evolutionary retrieval (`FR-21`) `SnippetFamily` + stability bonus |
| 9 | 23 Sep | Demo readiness | Full-split reportable eval run, `appsretrieval_results.json` | Streamlit result-card polish support | Streamlit UI (`FR-25`), demo video recording | README, Docker, `T-201` name-propagation cleanup |
| 10 | 24–25 Sep | → `M7` | Final full-split eval re-confirmation | — | Final demo rehearsal | Release cut, PPT, Google Form submission |

The `M4`/`M5` split across Day 6 in the table above matches
`PROJECT_OVERVIEW.md §13`'s "Day 5-6: Agentic Loop" and "Day 5-6: Integration" rows, expressed here
as two independently-checkable gates rather than one fuzzy "integration" milestone.

---

## 5. Risk register

Twelve risks, each with a category consistent with [NonGoals.md](NonGoals.md)'s category vocabulary
where applicable, an owner, and a mitigation that is a concrete engineering choice already reflected
elsewhere in the docs — not a hope.

| ID | Risk | Category | Owner | Mitigation |
|---|---|---|---|---|
| [`RISK-01`](#risk-01--embedding-model-too-slow-on-cpu) | Embedding model too slow on CPU to hit the cold-index budget | CPU budget | Prabinder | INT8 dynamic quantisation via ONNX Runtime; declared fallback `all-MiniLM-L6-v2` at ~4x the throughput if the primary underperforms ([`ADR-012`](Decisions.md#adr-012--onnx-runtime-int8--llamacpp-gguf-for-a-cpu-only-stack)) |
| [`RISK-02`](#risk-02--ndcg10-score-too-low-to-pass-screening) | NDCG@10 lands below a competitive screening score | accuracy | Prabinder | Hybrid retrieval (dense+sparse) plus reranking is the architecture, not an add-on; `M1`–`M5` gates each require a measured improvement over the prior gate, so a shortfall is caught at the milestone that caused it, not on Day 10 |
| [`RISK-03`](#risk-03--tree-sitter-parsing-fails-on-real-world-edge-cases) | tree-sitter parsing fails on real-world JS (minified bundles, syntax errors, deep nesting) | robustness | Anish | Declared degradation ladder: AST → statement-boundary split → line-window fallback → whole-file module chunk, enforced by `NFR-07` and tested by `TC-019`–`TC-024` |
| [`RISK-04`](#risk-04--agent-loop-runs-unbounded-or-too-slow) | Agent loop runs unbounded or blows the query-latency budget | CPU budget | Harshdeep | Hard caps, not convergence: 2 passes, 5 s wall clock, checked before each pass ([`ADR-007`](Decisions.md#adr-007--bounded-agent-loop-hard-caps-not-convergence)); adversarial-input termination is `TC-086`, a P0 test |
| [`RISK-05`](#risk-05--coir-dataset-or-mteb-api-mismatch) | CoIR dataset shape or MTEB API version mismatch breaks the eval harness | integration | Parth | MTEB v2 API confirmed in `M0` (Day 1–2), not discovered on Day 9; [Setup.md §4.4](Setup.md#44-mteb-v2-is-required--not-v1) documents the exact v1-vs-v2 symptom so a mismatch is diagnosed in minutes, not hours |
| [`RISK-06`](#risk-06--incremental-reindex-too-slow-to-demo) | Incremental reindex too slow to demonstrate live | CPU budget | Parth | The `NFR-02` 45-second budget is the mitigation itself — content-addressed blob reuse (`FR-19`) means the cost scales with changed files, not corpus size; `TC-075` proves a pure rename costs zero embedding calls |
| [`RISK-07`](#risk-07--rival-submission-lands-its-p0-score-first) | The public rival submission ([PRD.md §9](PRD.md#9-competitive-positioning)) lands a measured P0 score before we do, since it has fewer moving parts | schedule | Prabinder | Sequencing: dense + sparse hybrid + reranker scheduled green by `M3` (Day 5), *before* the structural signal and agent loop land, so a submittable baseline exists early regardless of how the innovation features land |
| [`RISK-08`](#risk-08--project-name-collision-causes-jury-confusion) | Identical project name and release tag as the rival submission causes jury confusion | presentation | Harshdeep | Resolved: renamed PRISM → Axiom ([`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom)); residual propagation tracked as `T-201`, due before `M7` |
| [`RISK-09`](#risk-09--test-split-overfitting-under-day-910-time-pressure) | Tuning discipline slips under Day 9–10 pressure and the test split gets touched more than once per config | integrity | Parth | [`NG-29`](NonGoals.md#ng-29--no-tuning-on-the-benchmark-test-split) fence plus the experiment log in [TestPlan.md §6.4](TestPlan.md#64-experiment-log) — every test-split run is logged with its config hash, making an at-most-once violation visible in the log itself, not just in policy |
| [`RISK-10`](#risk-10--native-wheel-availability-blocks-a-dev-environment) | `faiss-cpu` / `llama-cpp-python` native wheel unavailable blocks a team member's dev environment | tooling | Anish | Documented WSL2 route for Windows-arm64 ([`NG-28`](NonGoals.md#ng-28--no-native-windows-arm64-support), [Setup.md §2.3](Setup.md#23-windows)); no team member develops on an unsupported platform without this fallback known in advance |
| [`RISK-11`](#risk-11--judge-machine-is-offline-and-a-model-download-stalls-live) | Judge's machine is offline or network-restricted and a model download stalls mid-demo | demo-day | Harshdeep | Mandatory pre-download of primary and fallback weights the day before ([Setup.md §5.4](Setup.md#54-pre-download-for-an-offline-demo-do-this-the-day-before)); `AXIOM_OFFLINE=true` rehearsed so a stalled fetch fails fast and visibly instead of hanging in front of the jury |
| [`RISK-12`](#risk-12--a-late-schema-change-breaks-multiple-workstreams-at-once) | A late change to `src/axiom/schema/` breaks multiple workstreams simultaneously, since all four people code against it | integration | Prabinder | Four-member sign-off gate on any schema diff, enforced as review checklist item 7 in [Rules.md §10](Rules.md#10-code-review-rules); schema changes are logged in [Changelog.md](Changelog.md) with a breaking/non-breaking flag |

### 5.1 Risk severity and review cadence

| Severity | Definition | Review cadence |
|---|---|---|
| High | Blocks a Must-have gate (`M0`–`M5`) if it fires | Daily standup, every day until closed or the gate passes |
| Medium | Blocks a Should-have gate (`M6`) or a demo-quality property | Reviewed every 2 days |
| Low | Affects presentation or a Could-have feature only | Reviewed at the Day 7 and Day 9 checkpoints |

`RISK-01`, `RISK-02`, `RISK-04`, `RISK-05`, `RISK-12` are High (they can block a Must-have gate).
`RISK-03`, `RISK-06`, `RISK-07`, `RISK-09`, `RISK-10` are Medium. `RISK-08`, `RISK-11` are Low
severity but zero-tolerance on demo day specifically — a Low-severity risk that fires during the
five minutes the jury is watching costs disproportionately more than its severity rating implies,
which is why `RISK-11`'s mitigation is rehearsed explicitly rather than merely documented.

---

## 6. Dependencies between workstreams

```
Schema (all four)
   │
   ├──> Retrieval core (Prabinder): dense + sparse index/retrieve, fusion
   │         │
   ├──> Structural (Anish): chunker ──> structural index/retrieve
   │         │                              │
   │         └──────────────┬───────────────┘
   │                        ▼
   │              Fusion consumes all three signals (M2 needs dense+sparse;
   │              M4 additionally needs structural)
   │                        │
   ├──> Rerank + Agent (Harshdeep): reranker (needs fused candidates, M3)
   │         agent loop (needs rerank, M5)
   │
   └──> Versioning (Parth): registry/manifest ──> incremental reindex ──> evolutionary (M6)
              (needs the chunker + all three indexers to exist first)
```

The chunker (Anish, Day 1–3) is the one true blocking dependency for everyone else — dense, sparse,
and structural indexing, and therefore fusion, all consume `Chunk` records. This is why chunking is
scheduled Day 1–3 and the fixture repo (`tests/fixtures/repo_v1/`) exists specifically so the other
three workstreams are never blocked waiting on the real `OQ-07` demo repo.

---

## 7. Related documents

| Document | Relationship |
|---|---|
| [Tracker.md](Tracker.md) | `T-###` tasks that implement each cell of the day plan in §4 |
| [TestPlan.md](TestPlan.md) | The gate conditions in §3 are the same version-tagged rows as `§6.5` there |
| [Decisions.md](Decisions.md) | `ADR-###` records for the architectural choices the mitigations in §5 rely on |
| [OpenQuestions.md](OpenQuestions.md) | `OQ-##` items scheduled into the day plan in §4 |
| [Changelog.md](Changelog.md) | Release entries corresponding to each `M1`–`M7` version tag |
| [Deployment.md](Deployment.md) | The `M7` submission runbook this plan builds toward |
