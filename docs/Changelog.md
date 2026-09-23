# Changelog

What shipped and when for Axiom: breaking vs non-breaking changes, and the release ladder from the foundation commit to submission.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-23
**Status:** Active — re-baselined

Related: [ImplementationPlan.md](ImplementationPlan.md) · [Tracker.md](Tracker.md) · [TestPlan.md](TestPlan.md) · [Decisions.md](Decisions.md) · [Schema.md](Schema.md) · [Rules.md](Rules.md) · [Setup.md](Setup.md) · [Deployment.md](Deployment.md)

---

## How to read this document

This changelog does not track a shipping product with users on old versions — it tracks a build
against a fixed jury deadline. Its two jobs are still real ones:

1. **Every version tag here is a milestone gate, not an independent release decision.** The tag
   sequence `0.1.0` → `1.0.0` is the `M0`–`M7` gate ladder defined in
   [ImplementationPlan.md §3](ImplementationPlan.md#3-milestone-gates-m0m7) and asserted as fact by
   [TestPlan.md §6.5](TestPlan.md#65-quality-gates-by-milestone). A version bumps exactly when, and
   only when, its milestone gate condition becomes true.
2. **A schema or on-disk-format change gets an entry and a version bump here regardless of which
   milestone it lands inside**, per the breaking-change definition in §2 below — the behaviour
   [TestPlan.md §8.2](TestPlan.md#82-what-a-merge-requires) item 3 requires.

**What changed on 2026-09-23.** The ladder used to gate *construction*: `0.4.0` meant "the reranker
is integrated," `0.6.0` meant "the agent loop is live." All of that is now built and merged. The
ladder therefore gates *measurement* instead: from `0.2.0` on, a tag means a number was produced on
the full test split and written down with its git SHA. §5's entries are restated on that basis, and
their target dates come from the re-baselined five-day window (Day 1 = 2026-09-23, submission
2026-09-27). The 15–25 Sep dates every earlier revision carried are void.

---

## 1. Version scheme

| Tag | Milestone gate | What it means |
|---|---|---|
| `0.1.0` | `M0` — foundation | The system exists and is internally consistent: schema, chunking, three signals, fusion, rerank, agent loop, versioning, eval harness, CLI, API, UI. **Says nothing about quality**, and does not yet mean the suite is green |
| `0.2.0` | `M2` — measured dense baseline | A full-split dense-only NDCG@10 / MRR@10 / Recall@100 exists, produced by this project, recorded with a git SHA. The denominator for every later claim |
| `0.3.0` | `M3` — hybrid + RRF | Sparse + weighted RRF measured on the full split; the delta against `0.2.0` recorded with its sign |
| `0.4.0` | `M4` — rerank | Cross-encoder rerank measured on the full split under `eval.yaml`; delta against `0.3.0` recorded; ablation proven by a paired `AXIOM_RERANKER_ENABLED=false` run |
| `0.5.0` | `M5` — demo | A real tagged JS repo indexed under `demo.yaml`; Q1/Q2/Q3 answered live; `reindex` timed on the synthetic 50-file diff; `--all-versions` collapses a family |
| `0.6.0` | `M6` — tuned + ablated | Tune-split sweeps complete, placeholders replaced, full split re-run once afterwards, ablation table assembled |
| `1.0.0` | `M7` — submission | Every surviving item of [PRD.md §2.1 Definition of Done](PRD.md#21-definition-of-done) simultaneously true |

`M1` (first real model run) carries no tag: it produces a `--limit 200` smoke number, and
[TestPlan.md §6.2](TestPlan.md#62-why-a-limited-run-is-never-a-reportable-score) forbids a limited
run from being quoted, so there is nothing to tag.

**No tag in this ladder asserts an absolute NDCG@10 threshold any more.** The earlier `≥ 18.0` and
`≥ 20.0` gates were calibrated against a "BGE 0.6B = 14.7" baseline that appears in neither paper it
was sourced to. That figure is withdrawn — see §5's `0.2.0` entry. Each tag from `0.3.0` on requires
a recorded delta against this project's own measured baseline.

---

## 2. What counts as a breaking change

A change is **breaking** if it invalidates something another workstream or an already-built index
artifact depends on without a migration path. Concretely:

| Breaking | Non-breaking |
|---|---|
| Renaming, retyping, or making required any field in `src/axiom/schema/` (any model in [Schema.md](Schema.md)) | Adding a new **optional** field with a default to an existing schema model |
| Changing the on-disk index layout under `.axiom/` — e.g. renaming `dense.idmap.json`, changing `chunks.jsonl`'s line shape, adding a required file to the version directory | Adding a new **optional** file to the version directory that older code can ignore |
| Changing the *meaning* of a locked algorithm constant — e.g. redefining what `rrf_k` weights, or changing `contributions` from "signal → rank" to "signal → score" | Changing a locked constant's **value** while its meaning is unchanged — e.g. retuning `eval_sparse_weight` after a measured delta (this is exactly what the `# PLACEHOLDER` replacement discipline in [Rules.md §8](Rules.md#8-the-placeholder-convention) does, and it is explicitly *not* breaking on its own) |
| Changing `compute_chunk_id`/`compute_content_hash`'s inputs or hash algorithm — invalidates every already-built index | Adding a new CLI subcommand or a new HTTP endpoint |
| Changing an HTTP response shape in a way existing callers cannot parse (removing/retyping a field in [API.md](API.md)) | Adding a new config profile under `configs/` |
| Removing or retyping an environment variable in [Setup.md §7](Setup.md#7-environment-variables) | Adding a new environment variable with a default that preserves current behaviour |

A breaking change requires: a version bump per §1's table (even mid-milestone), a Changelog entry
under §5 tagged **BREAKING**, and — if it touches `src/axiom/schema/` — the four-member sign-off
required by [Rules.md §10](Rules.md#10-code-review-rules) item 7, cited again here because this is
the document that makes that sign-off's consequence visible.

---

## 3. Bench-baseline refresh log

Distinct from a version-tag release entry. [TestPlan.md §5.4](TestPlan.md#54-regression-gate) states
that the latency/RSS regression baseline "is refreshed manually by the owner after an intentional,
explained regression... Refreshing requires a line in Changelog.md." Each refresh gets one row here.

| Date | Baseline SHA (new) | Reason | Regression accepted | Owner |
|---|---|---|---|---|
| — | — | *(none — `scripts/bench_latency.py` has never been run against a real model; there is no baseline to refresh. First row expected from `T-050` on Day 3)* | — | — |

---

## 4. Unreleased

Tracks what is actively being built toward the next milestone gate. A pointer, not a duplicate —
the live task list is [Tracker.md's task board](Tracker.md#2-task-board).

**Targeting `0.2.0` (`M2` — measured dense baseline), due 2026-09-24.**

Requires, in order: an INT8 embedder export and a 500-chunk timing (`T-013`, `T-040`); the
behavioural pooling check (`T-018`); a dense index over the full 8,765-document APPS corpus
(`T-022`); a `--limit 200` smoke run to prove the harness (`T-023`); then the full-split dense-only
run itself (`T-026`). No breaking changes anticipated: the schema has been stable since `0.1.0` and
[Schema.md §1](Schema.md#1-scope-and-authority)'s "nothing may be renamed without an ADR" rule
holds.

---

## 5. Release entries

### `0.1.0` — foundation

**Status: Released.** **Landed:** 2026-09-23. **Breaking:** n/a (first tag).

This is the first entry in this document that describes something that actually happened. Three
commits (`bccece0`, `405a4a7`, `d821e61`) put 69 Python modules and roughly 27,000 lines under
`src/axiom/`, `scripts/` and `tests/`. `ruff` is clean. **At the moment `0.1.0` was cut the suite
was not: 403 tests collected, 397 passing, 6 failing** — see the closing note on this entry.

> **State as of 2026-09-23, after `0.1.0`.** Everything in the "does not contain" list below has
> moved. The suite is **611/611 passing** with `ruff check` and `ruff format --check` clean; a real
> embedder (`all-MiniLM-L6-v2` INT8 ONNX) has been loaded and run; and a first retrieval number
> exists — dense-only **NDCG@10 = 7.59** on the full CoIR `AppsRetrieval` test split, not
> reportable. The live status is [Tracker.md](Tracker.md) and
> [ImplementationPlan.md §0](ImplementationPlan.md#0-re-baseline-notice--read-this-before-anything-else),
> not this entry; this entry is kept as the record of what `0.1.0` itself contained.

Landed:

| Area | Modules |
|---|---|
| Schema contract — 9 models, `extra="forbid"`, frozen | `src/axiom/schema/` (6 modules), 71 tests |
| Core primitives — blake2b-128 identity, timing ledger, structured logging, error taxonomy | `src/axiom/core/` (4 modules), 34 hashing tests |
| Config — `Settings` plus five profiles (`default`, `fast`, `accurate`, `eval`, `demo`) | `src/axiom/config.py`, `configs/` |
| Chunking — tree-sitter AST boundaries with a four-rung degradation ladder | `src/axiom/chunking/` (3 modules), 33 tests |
| Indexing — dense (FAISS flat/IVF-PQ + positional idmap), sparse (`bm25s`), structural (SQLite, four relations), manifest and blob store | `src/axiom/indexing/` (5 modules) |
| Retrieval — dense, sparse, structural (five `FR-10` query shapes), weighted RRF with empty-signal renormalisation | `src/axiom/retrieval/` (5 modules), 41 fusion + 39 structural tests |
| Rerank — cross-encoder adapter degrading to RRF order | `src/axiom/rerank/cross_encoder.py` |
| Agent — heuristic classifier, planner, sufficiency evaluator, bounded loop, optional GGUF LLM | `src/axiom/agent/` (6 modules), 46 tests |
| Versioning — registry, git-diff incremental reindex, content-addressed blob reuse, `SnippetFamily` grouping | `src/axiom/versioning/` (3 modules) |
| Evaluation — MTEB v2 adapter, metrics, `scripts/run_eval.py`, `scripts/build_index.py`, `scripts/bench_latency.py` | `src/axiom/eval/`, `scripts/`, 42 tests |
| Surfaces — Typer CLI (ten subcommands), FastAPI service, Streamlit UI | `src/axiom/cli.py`, `src/axiom/api/`, `src/axiom/ui/` |

**What `0.1.0` explicitly does not contain, stated plainly because it is the fact that matters
most:**

- **No retrieval-quality number** *(resolved after `0.1.0` — see the note above)*. At `0.1.0`,
  NDCG@10, MRR@10 and Recall@100 were unmeasured, with no `appsretrieval_results.json`, no row in
  [Tracker.md §5](Tracker.md#5-eval-metrics-log) and no row in `artifacts/experiments.csv`. Any
  figure quoted for this project before that first measured run is fabricated.
  `appsretrieval_results.json` still does not exist.
- **No real model has ever been loaded** *(partly resolved after `0.1.0`)*. At `0.1.0` all 403
  tests ran against `FakeEmbedder`, `FakeCrossEncoder` and `FakeLLM`. A real embedder has since
  been run; **no reranker or LLM weight has**, and the configured primary embedder
  `Qwen/Qwen3-Embedding-0.6B` has still never been downloaded. Every latency, throughput and
  memory figure in the doc suite therefore remains a budget, not a measurement.
- **No latency or RSS measurement.** `scripts/bench_latency.py` exists and is unit-tested; it has
  never been pointed at a real index.
- **No demo corpus.** `data/demo_repo/` is empty; `OQ-07` is open.
- **Not reproducible from a clean clone.** `uv.lock` and `.github/workflows/ci.yml` do not exist, so
  `NFR-09` and `NFR-11` are both currently unmet.

Seven constants remain `# PLACEHOLDER` in `axiom.config.PLACEHOLDER_FIELDS` and are listed in
[Tracker.md §5.1](Tracker.md#51-placeholder-replacement-log).

**Six failing tests, two defects.** Recorded here rather than quietly fixed, because a changelog
that reports a green build it does not have is worth nothing.

- **Chunk spans (4 failures, `T-019`).** A chunk that does not start at column 0 — every method,
  every nested function — receives a `start_byte` at the declaration token but a `start_line`
  naming the whole line. `chunk.text` is the byte slice, so it lacks the leading indentation that
  `lines[start_line-1 : end_line]` includes, and Schema §4's contract that the two agree is broken.
  What it touches is why it is listed first: `chunk_id` binds `file_path` and `start_line`, so the
  defect sits on the identity function the whole index is keyed by, and every `file:line` header
  shown in the demo and the UI is rendered from the same pair. The likely fix changes `chunk.text`
  for every indented chunk and therefore `content_hash` — a **breaking** change under §2, and one
  that is far cheaper now, with no index built, than after `0.2.0`.
- **Two tests assume `faiss-cpu` is absent (2 failures, `T-020`).** They assert the numpy fallback
  (`dense_backend == "numpy"`, `dense.npy`) and fail once the optional `retrieval` extra is
  installed and the faiss rung runs. Small, but it means the dense primary path has been exercised
  less than the run count implied.

Detail in [TestPlan.md §3.14](TestPlan.md#314-resolved--the-six-failures-that-blocked-m0). **`0.1.0`
is tagged as the honest record of what landed, not as a green build.**

### `0.2.0` — measured dense baseline

**Status:** Planned. **Target:** 2026-09-24 (Day 2). **Breaking:** No.

Will contain: the first number this project has ever produced. `scripts/run_eval.py --task
AppsRetrieval --split test`, full split, no `--limit`, sparse and rerank disabled, under
`configs/eval.yaml`. NDCG@10, MRR@10 and Recall@100 land in
[Tracker.md §5](Tracker.md#5-eval-metrics-log) with a git SHA.

**This entry replaces a claim rather than adding one.** Earlier drafts calibrated the whole ladder
against "BGE 0.6B = 14.7 on CoIR AppsRetrieval," sourced to arXiv:2407.02883 and arXiv:2506.16552.
Neither paper contains that figure. The 14.7 baseline and the 14.7 → 23.5 span built on it are
withdrawn from this document and from every plan; what replaces them is whatever `0.2.0` measures,
plus the `0.6.0` ablation table, presented as a relative gain over our own baseline.

### `0.3.0` — hybrid + RRF

**Status:** Planned. **Target:** 2026-09-24 (Day 2). **Breaking:** No.

Will contain: the `bm25s` sparse signal and weighted RRF fusion
([`ADR-002`](Decisions.md#adr-002--weighted-reciprocal-rank-fusion-over-score-space-fusion))
measured on the full split. Gate: the delta against `0.2.0` is recorded with its sign, whatever the
sign turns out to be. The code has been merged since `0.1.0`; what this tag adds is evidence that it
helps.

### `0.4.0` — rerank

**Status:** Planned. **Target:** 2026-09-25 (Day 3). **Breaking:** No.

Will contain: cross-encoder reranking measured on the full split under `configs/eval.yaml`, where
`BAAI/bge-reranker-v2-m3` is primary with the candidate chain narrowed (`fusion_top_n: 5`,
`rerank_max_chars: 1024`) rather than the model weakened. The paired
`AXIOM_RERANKER_ENABLED=false` run is part of the gate, so the rerank contribution is isolated
rather than assumed.

**Note on the 620 ms figure.** `TechSpecifications.md §8`'s 620 ms rerank budget is
`ms-marco-MiniLM-L-6-v2`'s number recorded against `bge-reranker-v2-m3`'s row. The two profiles now
differ deliberately: `eval.yaml` keeps bge for accuracy on an offline, untimed run;
`demo.yaml` promotes `cross-encoder/ms-marco-MiniLM-L-6-v2` to primary, because latency is what the
jury watches. No number from `eval.yaml` may be quoted against `NFR-03`/`NFR-04`.

### `0.5.0` — demo

**Status:** Planned. **Target:** 2026-09-25 (Day 3). **Breaking:** Possibly.

Will contain: a small tagged JavaScript repository (10–50 files, ≥ 2 versions — `OQ-07`) indexed
under `configs/demo.yaml` with all three signals live; the three archetype queries answered live;
`axiom reindex` timed on the **synthetic** 50-file diff from `tests/fixtures/gen_v2.py` for
`NFR-02`; `--all-versions` collapsing one `SnippetFamily`.

Flagged possibly breaking for one reason: this is the first time `FusedResult.contributions` and
`dominant_signal` carry a third live `SignalKind` in a real run, so any consumer that assumed
exactly two keys is affected. The flag is confirmed or cleared when the entry moves out of Planned.

**The "real 10k-file JS repo" claim is deleted.** A 10k-file repo is ~69k chunks at the documented
6.9 chunks/file, which crosses the 50k IVF threshold and invalidates every index-build budget in the
suite. The demo repo is small and the 50-file diff is synthetic, and the deck says so.

### `0.6.0` — tuned + ablated

**Status:** Planned. **Target:** 2026-09-26 (Day 4). **Breaking:** No.

Will contain: the tune-split sparse-weight sweep (`T-112`) and the dev-split sufficiency-threshold
sweep (`T-141`), their measured constants replacing the `# PLACEHOLDER` values, one full-split re-run
afterwards, and the ablation table (dense → +sparse → +rerank → +agent) assembled from the recorded
rows. Both sweeps draw exclusively from the `AppsRetrieval` **train** split; `NG-29`'s test-split
fence is absolute and no holdout is ever carved from test.

This ordering is enforced mechanically, not by discipline: `scripts/run_eval.py` stamps
`axiom_provenance.reportable: false` on any run made while a placeholder is active.

### `1.0.0` — submission

**Status:** Planned. **Target:** 2026-09-27 (Day 5). **Breaking:** No (freeze).

Will contain: every surviving item of
[PRD.md §2.1 Definition of Done](PRD.md#21-definition-of-done) simultaneously true; README, PPT and
≤ 5-minute demo video finished; `appsretrieval_results.json` attached to the release. This entry,
once real, records the final measured `ndcg_at_10` / `mrr_at_10` and the
[Decisions.md](Decisions.md) ADR count as of release.

**Dependency freeze note:** per [Rules.md §9.4](Rules.md#94-dependency-pinning) item 4, no
dependency may be added in the final window except to fix a submission-blocking defect — so `1.0.0`
cannot introduce a breaking dependency change; anything landing there is a bug fix, not a feature.

---

## 6. Where this history starts, and what it deliberately excludes

This changelog's history begins with the first commit toward the Samsung PRISM GenAI Hackathon 3rd
Edition (Theme 01) submission. It does not include, and should never be read as including, the
history of the team's **earlier and entirely unrelated** project also named Axiom: a
verifier-centric framework with a 5-head Process Reward Model (XD-PRM), built by two of this team's
four members for the separate Samsung ennovateX AX Hackathon 2026 (Problem Statement 06), described
in [`PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md) Appendix C.

The name collision is deliberate and explained in full in
[`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom): this project was originally named PRISM
and renamed to Axiom to resolve a project-name collision with a rival public submission
([`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission)); the new name
signals lineage from that earlier, different codebase — not shared code, shared history, or a shared
changelog. Nothing in this document, and nothing under `src/axiom/`, descends from it.

---

## 7. Release tag vs. project version — these are two different things

`1.0.0` (§5, `M7`) is **this project's own version**, assigned by this changelog under the scheme in
§1. `PRISM_GENAI_HACKATHON_Y2026` is a **GitHub Release tag**, prescribed by the hackathon
organisers as the name of the single submission slot every team's final artifact attaches to. It is
not part of this project's version scheme, and it does not change if `1.0.0` slips or is re-cut.
"PRISM" in that tag is the **event**, not this project's former name — the two senses are
distinguished in [Glossary.md](Glossary.md#project-specific-proper-nouns).

Per [ImplementationPlan.md `M7`](ImplementationPlan.md#3-milestone-gates-m0m7) and
[Tracker.md `T-221`](Tracker.md#26-evaluation-discipline-submission-docs-t-191t-230), the release
process is: reach `1.0.0` (confirmed by the Definition of Done), then cut the GitHub Release under
the organiser-mandated tag and attach `appsretrieval_results.json`. The tag names the submission
event; `1.0.0` names the state of the system at the moment it was submitted. Only one release under
that tag will ever be cut.

One prerequisite that is currently unmet: `.gitignore` lists `appsretrieval_results.json`, so the
artifact the release exists to carry cannot be committed. `T-016` moves the committed artefacts to
`artifacts/` and fixes the ignore rules; it must land before `T-221`.
