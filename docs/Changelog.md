# Changelog

What shipped and when for PRISM: breaking vs non-breaking changes, and the planned-release ladder from the first dense baseline to submission.

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-16
**Status:** Draft

Related: [ImplementationPlan.md](ImplementationPlan.md) · [Tracker.md](Tracker.md) · [TestPlan.md](TestPlan.md) · [Decisions.md](Decisions.md) · [Schema.md](Schema.md) · [Rules.md](Rules.md) · [Setup.md](Setup.md) · [Deployment.md](Deployment.md)

---

## How to read this document

This changelog does not track a shipping product with users on old versions — it tracks a 10-day
hackathon build against a fixed jury deadline. Its two jobs are still real ones:

1. **Every version tag here is a milestone gate, not an independent release decision.** The tag
   sequence `0.2.0` → `1.0.0` is the exact `M1`–`M7` gate ladder defined in
   [ImplementationPlan.md §3](ImplementationPlan.md#3-milestone-gates-m0m7) and asserted as fact by
   [TestPlan.md §6.5](TestPlan.md#65-quality-gates-by-milestone)'s "Quality gates by milestone"
   table. A version bumps here exactly when, and only when, its milestone gate condition becomes
   true — there is no separate "cut a release" decision to make.
2. **A schema or on-disk-format change gets an entry and a version bump here regardless of which
   milestone it lands inside**, per the breaking-change definition in §2 below. This is the specific
   behaviour [TestPlan.md §8.2](TestPlan.md#82-what-a-merge-requires) item 3 requires when it says
   "Schema changes... a Changelog.md entry and a version bump per the breaking-change definition
   there."

This document starts at **this project's Day 1, 2026-09-15**. See §6 for why that boundary matters
and what it deliberately excludes.

---

## 1. Version scheme

| Tag | Milestone gate | What it means |
|---|---|---|
| `0.1.0` | `M0` (implicit) | Foundation only — no version tag is cut for `M0` itself; it is a prerequisite state, not a scored capability. Listed here only so the ladder is visually complete. |
| `0.2.0` | `M1` — dense baseline | Dense index over the full APPS corpus; eval harness proven with *a* NDCG@10 number, value not yet judged |
| `0.3.0` | `M2` — hybrid + RRF | Sparse index + weighted RRF fusion wired end to end; NDCG@10 improves ≥ 2.0 absolute over `0.2.0` |
| `0.4.0` | `M3` — rerank | Cross-encoder reranker integrated and ablatable; NDCG@10 ≥ 18.0 |
| `0.5.0` | `M4` — structural | Structural signal wired into the `demo` profile's fusion; no regression on `eval` vs. `0.4.0`; Recall@100 ≥ 65.0 |
| `0.6.0` | `M5` — agent loop | Bounded agent loop live, heuristic and LLM paths both green; NDCG@10 ≥ 20.0 (the [PRD.md §2](PRD.md#2-goals-and-success-metrics) target) |
| *(unversioned)* | `M6` — P1 + Bonus | Incremental reindex, version registry, evolutionary retrieval land between `0.6.0` and `1.0.0`. Deliberately carries no version tag of its own — see note below. |
| `1.0.0` | `M7` — submission | [PRD.md §2.1 Definition of Done](PRD.md#21-definition-of-done), all seven items, simultaneously true |

**Why `M6` gets no version tag.** `M1`–`M5` are Must-have gates on the P0 screening path and the
agent loop ([PRD.md §7](PRD.md#7-prioritisation-moscow) "Must have" rows) — each one is a
qualitatively new capability worth naming. `M6` bundles two *Should-have*/*Could-have* items (P1
version support, Bonus evolutionary retrieval) that, per the cut-order discipline in
[PRD.md §7.1](PRD.md#71-cut-order-under-time-pressure), might not both land intact. Minting a
version tag for a milestone whose scope could still shrink under time pressure would mean either
renumbering later or shipping a tag that overpromises. `M6`'s two capabilities are instead tracked
directly as acceptance criteria (`US-7`–`US-9` in [PRD.md §4](PRD.md#4-user-stories)) and get their
own entries below the moment each one is actually done — versioned together only once both are
confirmed, folded into the `1.0.0` entry's changelog body rather than as a separate tag.

---

## 2. What counts as a breaking change

A change is **breaking** if it invalidates something another workstream or an already-built index
artifact depends on without a migration path. Concretely:

| Breaking | Non-breaking |
|---|---|
| Renaming, retyping, or making required any field in `src/axiom/schema/` (any model in [Schema.md](Schema.md)) | Adding a new **optional** field with a default to an existing schema model |
| Changing the on-disk index layout under `.axiom/` (`_CONTRACT.md §6`) — e.g. renaming `dense.idmap.json`, changing `chunks.jsonl`'s line shape, adding a required file to the version directory | Adding a new **optional** file to the version directory that older code can ignore |
| Changing the *meaning* of a locked algorithm constant — e.g. redefining what `rrf_k` weights, or changing `contributions` from "signal → rank" to "signal → score" | Changing a locked constant's **value** while its meaning is unchanged — e.g. retuning `rrf_k` from 60 to 50 after a measured delta (this is exactly what the `# PLACEHOLDER` replacement discipline in [Rules.md §8](Rules.md#8-the-placeholder-convention) does, and it is explicitly *not* a breaking change on its own) |
| Changing `compute_chunk_id`/`compute_content_hash`'s inputs or hash algorithm ([Schema.md §13](Schema.md#13-identity-and-hashing)) — invalidates every already-built index | Adding a new CLI subcommand or a new HTTP endpoint |
| Changing an HTTP response shape in a way existing callers cannot parse (removing/retyping a field in [API.md](API.md)) | Adding a new config profile under `configs/` |
| Removing or retyping an environment variable in [Setup.md §7](Setup.md#7-environment-variables) | Adding a new environment variable with a default that preserves current behaviour |

A breaking change requires: a version bump per §1's table (even mid-milestone, ahead of that
milestone's own gate date if the breaking change lands early), a Changelog entry under
§4 tagged **BREAKING**, and — if it touches `src/axiom/schema/` specifically — the four-member
sign-off already required by [Rules.md §10](Rules.md#10-code-review-rules) item 7, cited again here
because this is the document that makes that sign-off's consequence (a version bump) visible.

---

## 3. Bench-baseline refresh log

Distinct from a version-tag release entry. [TestPlan.md §5.4](TestPlan.md#54-regression-gate) states
that the latency/RSS regression baseline "is refreshed manually by the owner after an intentional,
explained regression (e.g. adding the structural signal)... Refreshing requires a line in
Changelog.md." Each refresh gets one row here — not a full changelog entry, because it records an
infrastructure decision (accept a new baseline), not a shipped capability.

| Date | Baseline SHA (new) | Reason | Regression accepted | Owner |
|---|---|---|---|---|
| — | — | *(none yet — first refresh expected around `0.5.0`, when the structural signal's index-build and query cost are added deliberately)* | — | — |

---

## 4. Unreleased

Tracks what is actively being built toward the next milestone gate. This section is a pointer, not
a duplicate — the live task list is [Tracker.md's task board](Tracker.md#2-task-board); this section
names only the gate being targeted and what closing it requires, updated as the "next" gate changes.

**Targeting `0.2.0` (`M1` — dense baseline), due 2026-09-17.**

Requires (see [Tracker.md §2.2](Tracker.md#22-retrieval-core-t-021t-060) `T-021`–`T-025`): dense
index builder complete, full APPS corpus indexed, `axiom eval --limit 200` smoke run producing a
number. No breaking changes anticipated in this window — the schema (`Chunk`, `ChunkLocation`,
`ChunkMetadata`) is expected to be stable from `M0` onward per [Schema.md §1](Schema.md#1-scope-and-authority)'s
"nothing may be renamed without an ADR" rule.

---

## 5. Release entries

Every entry below is currently **Planned** — the build window opened 2026-09-15 and this document
was last updated 2026-09-16, so nothing has shipped yet. Each entry states its target date (from
[ImplementationPlan.md §3](ImplementationPlan.md#3-milestone-gates-m0m7)), what it will contain, its
breaking-change flag, and where the real numbers land once measured. An entry moves from Planned to
an actual dated record the day its gate condition is confirmed true; **the target date is not
retroactively edited to match** — if a gate lands late, the entry keeps its original target date
alongside the actual date, so the plan-vs-actual delta stays visible rather than being smoothed away.

### `0.2.0` — dense baseline

**Status:** Planned. **Target:** 2026-09-17 (Day 3). **Breaking:** No (first tagged schema use).

Will contain: `dense.faiss` + `dense.idmap.json` builder over the full 8,765-document APPS corpus;
`axiom eval --limit 200` smoke run proving the MTEB v2 harness is wired correctly. The NDCG@10 value
from this run is explicitly **not reportable** — [TestPlan.md §6.2](TestPlan.md#62-why-a-limited-run-is-never-a-reportable-score)
excludes any `--limit` run from being quoted anywhere. Full-split dense-only baseline number lands in
[Tracker.md §5 eval metrics log](Tracker.md#5-eval-metrics-log) once measured.

### `0.3.0` — hybrid + RRF

**Status:** Planned. **Target:** 2026-09-18 (Day 4). **Breaking:** No.

Will contain: `bm25s` sparse index over the identical chunk corpus; weighted RRF fusion
(`retrieval/fusion.py`) per [`ADR-002`](Decisions.md#adr-002--weighted-reciprocal-rank-fusion-over-score-space-fusion).
Gate: full-split NDCG@10 exceeds `0.2.0`'s dense-only baseline by ≥ 2.0 absolute. Delta recorded in
[Tracker.md §5](Tracker.md#5-eval-metrics-log) the day this gate is confirmed.

### `0.4.0` — rerank

**Status:** Planned. **Target:** 2026-09-19 (Day 5). **Breaking:** No.

Will contain: cross-encoder reranking over the top-25 fused candidates, ablatable via
`AXIOM_RERANKER_ENABLED`. Gate: full-split NDCG@10 ≥ 18.0.

### `0.5.0` — structural

**Status:** Planned. **Target:** 2026-09-20 (Day 6). **Breaking:** Possibly — `structural.sqlite`
(a new index artifact under `.axiom/index/<version_id>/`) is additive to the on-disk layout, which
is non-breaking per §2's table (older readers that never look for it are unaffected); however, if
`FusedResult.contributions`/`dominant_signal` gain a third live `SignalKind` value in practice for
the first time here, any consumer that assumed exactly two keys would be affected. Flagged possibly
breaking pending a real audit of API/UI consumers at implementation time; confirmed flag recorded
when this entry moves out of Planned.

Will contain: `structural.sqlite` build (symbols, calls, imports, exports), structural retriever,
structural signal wired into the `demo.yaml` profile's fusion only — the `eval.yaml` profile keeps
`struct: 0.0` per [`ADR-001`](Decisions.md#adr-001--two-first-class-evaluation-profiles). Gate: no
regression vs. `0.4.0` on the `eval` profile; Recall@100 ≥ 65.0. **This is also the anticipated
first bench-baseline refresh** — see §3.

### `0.6.0` — agent loop

**Status:** Planned. **Target:** 2026-09-20 (Day 6). **Breaking:** No.

Will contain: bounded agent loop (`agent/loop.py`), sufficiency predicate, heuristic classifier path
and LLM path both proven green under the `NFR-07` degradation matrix. Gate: full-split NDCG@10 ≥
20.0 — the headline target stated in [PRD.md §2](PRD.md#2-goals-and-success-metrics). This is the
number that, once real, replaces every placeholder NDCG@10 figure quoted informally elsewhere in
this doc suite.

### P1 + Bonus (folds into `1.0.0`, see §1's note on `M6`)

**Status:** Planned. **Target:** 2026-09-22 (Day 8), confirmed and versioned together at `1.0.0`.
**Breaking:** No.

Will contain: `VersionManifest` + `registry.json`, git-diff incremental reindex (`FR-18`),
content-addressed blob reuse (`FR-19`), version-scoped query (`FR-20`), evolutionary
`SnippetFamily` grouping with the stability bonus (`FR-21`). Acceptance: `US-7`–`US-9` in
[PRD.md §4](PRD.md#4-user-stories). Train-split tuning deltas (`FR-26`) — sparse weight, sufficiency
thresholds, doc2query ablation — are recorded individually in
[Tracker.md §5.1](Tracker.md#51-placeholder-replacement-log) as they land, not batched into this
entry.

### `1.0.0` — submission

**Status:** Planned. **Target:** 2026-09-25 (Day 10). **Breaking:** No (freeze — see below).

Will contain: all seven items of [PRD.md §2.1 Definition of Done](PRD.md#21-definition-of-done)
simultaneously true; demo surfaces (`FR-24`, `FR-25`) complete; README, Docker, PPT, and demo video
finished; the `M6` capabilities above confirmed and folded in. This entry, once real, records the
final measured `ndcg_at_10` / `mrr_at_10` from `appsretrieval_results.json`
([Tracker.md T-200](Tracker.md#26-evaluation-discipline-submission-docs-t-191t-230)) and the
[Decisions.md](Decisions.md) ADR count as of release.

**Dependency freeze note:** per [Rules.md §9.4](Rules.md#94-dependency-pinning) item 4, no
dependency may be added after the Day 9 feature freeze except to fix a submission-blocking defect —
so `1.0.0` cannot itself introduce a breaking dependency change; anything landing in this final
window is a bug fix, not a feature.

---

## 6. Where this history starts, and what it deliberately excludes

This changelog's history begins at **this project's Day 1, 2026-09-15** — the first commit toward
the Samsung PRISM GenAI Hackathon 3rd Edition (Theme 01) submission. It does not include, and should
never be read as including, the history of the team's **earlier and entirely unrelated** project
also named Axiom: a verifier-centric framework with a 5-head Process Reward Model (XD-PRM), built by
two of this team's four members for the separate Samsung ennovateX AX Hackathon 2026
(Problem Statement 06, Enhancing Reasoning in SLMs), documented in `PROJECT_OVERVIEW.md` Appendix C.

The name collision is deliberate and explained in full in
[`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom): this project was originally named PRISM,
renamed to Axiom to resolve a project-name collision with a rival public submission
([`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission)), and the new
name was chosen specifically to signal lineage from that earlier, different codebase — not to imply
shared code, shared history, or a shared changelog. If a reader encounters a reference to "Axiom" in
a context that predates 2026-09-15, it is the other project. Nothing in this document, and nothing
under `src/axiom/`, descends from it.

---

## 7. Release tag vs. project version — these are two different things

`1.0.0` (§5, `M7`) is **this project's own version**, assigned by this changelog under the scheme in
§1. `PRISM_GENAI_HACKATHON_Y2026` is a **GitHub Release tag**, assigned once, by the hackathon
organisers, as the name of the single submission slot every team's final artifact is attached to —
it is not part of this project's version scheme at all, and it does not change if `1.0.0` slips or
is re-cut. Per [ImplementationPlan.md `M7`](ImplementationPlan.md#3-milestone-gates-m0m7) and
[Tracker.md `T-221`](Tracker.md#26-evaluation-discipline-submission-docs-t-191t-230), the release
process is: reach `1.0.0` (this project's version, confirmed by the Definition of Done), then cut
the GitHub Release under the organiser-mandated tag `PRISM_GENAI_HACKATHON_Y2026` and attach
`appsretrieval_results.json` to it. The tag names the submission event; `1.0.0` names the state of
the system at the moment it was submitted. Only one release under that tag will ever be cut.
