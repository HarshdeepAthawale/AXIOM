# Axiom — Product Requirements Document

Defines what Axiom must do, for whom, and how success is measured for the Samsung PRISM GenAI Hackathon 3rd Edition (Theme 01) submission. ("Samsung PRISM" is the event; "Axiom" is the project — [ADR-015](Decisions.md#adr-015--rename-prism-to-axiom).)

**Owner:** Parth Deshmukh
**Last updated:** 2026-09-23
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

Three things make this specifically hard, and they are the three things Axiom is built around:

| Obstacle | Concrete failure today |
|---|---|
| The codebase does not fit in any LLM context window | 10k files is ~50M tokens. Pasting the repo into a model is not an option, so the system must retrieve before it reasons. |
| Lexical search misses meaning | The engineer types "preprocessing". The function is called `normalize()`. `grep preprocessing` returns nothing. |
| Semantic search misses structure | "Which files call `XYZ` before `ABC`?" is a question about the call graph and statement order. No embedding of any size answers it, because the answer is not in the text of any single snippet. |

**The task, stated precisely:** given a library of code and a natural-language query,
produce a ranking of code snippets ordered by relevance to the query, each located by
file path and line range.

This is a **retrieval** problem. Axiom retrieves, ranks, and locates. It does not generate
code and it does not explain code — see [NonGoals.md](NonGoals.md) (`NG-01`, `NG-02`, `NG-03`).

### 1.1 The Three Query Archetypes

Every design decision in Axiom traces back to one of these three queries, taken verbatim
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

Axiom has exactly three goals, inherited from the problem statement's own priority ladder.

> **Number-provenance rule (binding as of 2026-09-23).** Every quantity in this document says which
> of three things it is: **measured** (a run exists and is named), **estimated** (derived by stated
> arithmetic), or **`# PLACEHOLDER`** (a guess awaiting a measurement, with an owning `OQ-##`), per
> [Rules.md §8](Rules.md#8-the-placeholder-convention). **No accuracy or latency figure in this
> project is currently measured.** Anything below that looks like a result is a target or a
> placeholder, and is labelled as such.

### 2.0 The baseline is our own, and it does not exist yet

The previous revision of this section anchored every accuracy target to a row reading
**"BGE 0.6B = 14.7"** on CoIR `AppsRetrieval`, sourced to arXiv:2407.02883 and arXiv:2506.16552, and
built the deck's central claim on a 14.7 → 23.5 span. **That row is retracted.** An adversarial audit
of this suite could not locate 14.7 in either cited paper, and we have not re-verified the papers
ourselves. We are therefore in the worst possible position for a number that gates three milestones:
we cannot show where it came from, and a jury member can open the cited arXiv link during the
presentation.

The replacement is not a better citation. It is **our own run**:

| | |
|---|---|
| **Baseline `B`** | NDCG@10 of a **bare dense-only pipeline** — no sparse, no structural, no reranker, no agent loop — on the CoIR `AppsRetrieval` **test** split, under `configs/eval.yaml` with every other signal disabled. |
| **Command** | `axiom eval --task AppsRetrieval --split test` with `dense_enabled: true` and all other signals off. |
| **Status** | `# PLACEHOLDER — unmeasured`, owned by `OQ-16`, produced by `T-200a`. |
| **Priority** | **This is the first eval of the project.** It runs before sparse, before structural, before rerank, before the agent loop, before the UI and the API. Nothing else in §2 has a value until it does. |

`B` is a better baseline than any published number for three reasons, and the reasons are themselves
part of the technical-depth argument in §8: it is measured on *our* chunking, *our* pooling and *our*
corpus preparation, so a delta against it isolates the thing we actually built; it cannot be
falsified by opening a paper; and it is reproducible by the evaluator with one command.

#### 2.0.1 Goals, restated as relative gains

| Priority | Goal | Primary metric | Target | Status |
|---|---|---|---|---|
| **P0** (must) | Retrieval accuracy on CoIR `AppsRetrieval` test split | NDCG@10 | **≥ 1.36 x `B`** | derived, see below |
| **P0** (must) | Early relevance | MRR@10 | **≥ 1.36 x `B_mrr`** | derived |
| **P0** (must) | First-stage recall (the ceiling the reranker works under) | Recall@100 | **≥ 1.20 x `B_recall`** | derived |
| **P1** (should) | Version-aware incremental reindexing | Wall clock for a 50-changed-file diff on the synthetic generator | **≤ 45 s** | `# PLACEHOLDER`, `OQ-14` |
| **Bonus** | Evolutionary cross-version retrieval | Snippet-family dedup collapses ≥ 2 versions of the same symbol into one ranked result with diffs | Qualitative + live demo | binary, not a number |

**How the multipliers were derived, so they are auditable rather than asserted.** The retired ladder
was absolute: `M2 ≥ 16.7`, `M3 ≥ 18.0`, `M5 ≥ 20.0`, all calibrated against the retracted 14.7. The
*shape* of that ladder — how much each stage was expected to add — was a reasonable engineering
judgement even though its anchor was not. So the shape is preserved and the anchor is removed:

| Retired absolute gate | As a ratio to the retired 14.7 anchor | New gate |
|---|---|---|
| `M2` ≥ 16.7 (dense + sparse fusion) | 1.136 | **≥ 1.14 x `B`** |
| `M3` ≥ 18.0 (+ cross-encoder rerank) | 1.224 | **≥ 1.22 x `B`** |
| `M5` ≥ 20.0 (full pipeline) | 1.361 | **≥ 1.36 x `B`** |

These multipliers are **estimates, not measurements** — they inherit the retired ladder's judgement
about stage contributions and nothing more. They are re-set once `B` exists and the first ablation
rows land, and `ImplementationPlan.md`'s `M2`/`M3`/`M5` gate rows must be regenerated from this
table.

**The MRR and Recall multipliers are weaker still, and should be read that way.** The retired
absolutes `MRR ≥ 22.0` and `Recall@100 ≥ 65.0` had **no stated baseline at all** — their "Baseline to
beat" cells were empty — so there is no ratio to inherit. `1.36 x B_mrr` simply carries the NDCG
multiplier across on the reasoning that reranking moves the same chunks that move NDCG; `1.20 x
B_recall` is lower because recall@100 is set almost entirely by first-stage width, which the
reranker and the agent loop do not change, and only the sparse and structural signals can lift. Both
are engineering judgement with the arithmetic shown, not derived targets, and both are re-set from
the first ablation. **A separate consequence of these two having been unbaselined:** they were P0
contract targets that appeared in neither the Definition of Done nor the `M7` gate. They are in the
Definition of Done now (§2.1 item 4).

**What goes in the deck.** A relative gain plus an ablation table, not an absolute anchored to a
citation we cannot produce:

> *"Dense-only baseline `B` on CoIR AppsRetrieval test, measured by us, reproducible with one
> command. Adding sparse fusion: +X%. Adding the cross-encoder: +Y%. Adding the structural signal:
> +Z%. Adding the bounded agent loop: +W%. Total: +N% over our own baseline, on CPU, at 0.6B
> parameters."*

That is a stronger claim than "we scored 20.0" and it is the claim the 25%-weighted technical-depth
criterion (§8) actually rewards. It also survives a low `B`: an honest ablation over a weak baseline
is a real result, whereas a missed absolute is a red gate.

#### 2.0.2 Secondary, non-scored but gating metrics

Every row is a *requirement*, and every row is currently unmeasured.

| Metric | Requirement | Status |
|---|---|---|
| Cold index, 10k chunks | ≤ 12 min | `# PLACEHOLDER`, `OQ-14`. **Projected to be missed by 1.7–3.3x** — [TechSpecifications.md §8.5](TechSpecifications.md#85-cold-index-placeholder-oq-14) |
| Query p50 (agent loop disabled, LLM off, `demo` profile) | ≤ 900 ms | `# PLACEHOLDER`, `OQ-13`. Projected ~310–770 ms |
| Query p95 (up to 2 **total** agent passes) | ≤ 5 s | `# PLACEHOLDER`, `OQ-13`. Enforced by a monotonic deadline regardless |
| Peak RSS during query | ≤ 4 GB | `# PLACEHOLDER`, `OQ-15`. Component tally projects 2.7–4.1 GB — [TechSpecifications.md §8.6](TechSpecifications.md#86-peak-rss-placeholder-oq-15) |

There is **no latency budget for the `AXIOM_LLM_ENABLED=true` path** and none is claimed; see
TechSpecifications §8.3.

#### 2.0.3 External comparison table — quarantined pending per-row citation

A leaderboard-style comparison is worth having, and the previous one was not safe to ship. It is
therefore **empty rather than wrong**, with one binding rule attached:

> **Every row of any external comparison table in this project must carry a citation naming the
> paper, the table number, and the column** — e.g. "CoIR (arXiv:2407.02883), Table 3, `Apps`
> column". A row without all three does not go in the table, the deck, or the README. Anyone filling
> a row must have opened the paper and read the cell, not inherited the number from another
> document.

| Model | Size | NDCG@10 | Source (paper · table · column) |
|---|---|---|---|
| *(unfilled)* | | | *no row may be added without all three citation fields* |
| **Axiom, dense-only** | 0.6B, CPU-only | `B` = `# PLACEHOLDER` | our own run, `axiom eval --task AppsRetrieval --split test`, `T-200a` |
| **Axiom, full pipeline** | 0.6B, CPU-only | target ≥ 1.36 x `B` | our own run, `T-200` |

Only the last two rows are ones we can stand behind, and they are the two that matter. **Unverified
claims removed from this table on 2026-09-23:** the UniXcoder 0.1B = 1.4, BM25 = 4.8, E5-PT 0.3B =
10.6, BGE 0.6B = 14.7, E5-Mistral 7B = 23.5, Voyage-Code-2 = 26.5 and Revela 3B = 26.6 rows, the
"best reported score is 26.52" claim, and the derived positioning statement that our target "sits
between BGE-0.6B and E5-Mistral-7B." None of these was verified against a source by anyone on this
team. Some may well be correct — the point is that we do not know which, and a jury can check.

**One consequence worth stating, because a real decision rested on the deleted rows.** The eval
profile down-weights sparse to 0.15 (`configs/eval.yaml`), and the justification given was
"BM25 = 4.8 against BGE = 14.7 on this benchmark, so equal-weight fusion is a net negative." The
*mechanism* is sound and independent of those numbers: APPS queries are English prose describing a
programming problem and the documents are Python solutions, so there is very little shared surface
vocabulary for a lexical signal to match on, and a weak list fused at equal weight spends the
reranker's candidate slots on noise. But the **magnitude** of the down-weighting is no longer
supported by anything, so `0.15` remains what it always was — a `# PLACEHOLDER` awaiting the sweep in
[`OQ-02`](OpenQuestions.md#oq-02--what-is-the-exact-sparse-weight-in-configseval-yaml) (`T-112`),
and the sweep's range must include `0.0`.

The related claim that "being a code model is not sufficient; NL-to-code alignment is what matters"
rested on UniXcoder 0.1B (1.4) scoring below BM25 (4.8). With both rows removed, the claim is a
hypothesis we hold about this benchmark, not an established fact, and it is stated that way wherever
it appears.

### 2.1 Definition of Done

The submission is done when all of the following are simultaneously true:

1. `axiom index <repo>` completes on the demo repo (the small tagged JS repo of §2.2).
2. `axiom query "<Q1|Q2|Q3>"` returns file+line results for all three archetypes.
3. **`axiom eval --task AppsRetrieval --split test` has been run once dense-only, producing the
   baseline `B` (§2.0).** Until this exists, nothing else in this list is worth doing.
4. `scripts/run_eval.py` produces `appsretrieval_results.json` containing `ndcg_at_10`,
   `mrr_at_10` and `recall_at_100` for the full pipeline, with **`B` recorded alongside** so the
   relative gain is computable from the artefact itself without reading this document.
5. An **ablation table** with a number in every row: dense-only (`B`) / +sparse / +rerank /
   +structural / +agent. This is the deliverable the technical-depth criterion rewards (§8.1).
6. `axiom reindex --to <newer-commit>` completes inside 45 s for a 50-changed-file diff **on the
   synthetic generator** (`NFR-02`; the demo repo is too small to produce one — §2.2).
7. `axiom query --all-versions "<query>"` returns `SnippetFamily`-collapsed results.
8. The whole pipeline runs green with `AXIOM_LLM_ENABLED=false`.
9. Release `PRISM_GENAI_HACKATHON_Y2026` (organiser-prescribed tag) exists with the eval JSON and
   the ablation table attached.

Items 6 and 7 are the first two to cut under time pressure (§7.1). Items 3, 4 and 5 are the
submission; nothing may displace them.

### 2.2 Two Evaluation Contexts, Two Profiles

This is the single most important structural fact about the project, and it is a
**deliberate design split, not a gap**. Axiom is scored in two different places, on two
different corpora, and they are not the same language.

| | Screening benchmark | Live demo |
|---|---|---|
| Corpus | CoIR `AppsRetrieval` — built on the APPS dataset. Queries are English competitive-programming problem statements; corpus documents are **Python** solutions. | A **small, real, tagged JavaScript repository**: ≈10–50 source files, ≥3 tags spanning genuine refactors, ≥1 pure rename (`OQ-07`). Voice-assistant-adjacent where possible, but *small* is the binding constraint. |
| Document shape | Standalone single-file solutions. No imports across documents, no cross-file call graph, no module boundaries. | Multi-module: imports, exports, call chains, tool dispatch. Depth matters here, not breadth — Q2 needs a genuine call graph, not 10,000 files. |
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
to two corpora; one profile would be a compromise that is wrong for both. This was
[`OQ-01`](OpenQuestions.md#oq-01--is-the-two-profile-eval--demo-split-the-final-shape), **closed
2026-09-16** and recorded as
[ADR-001](Decisions.md#adr-001--two-first-class-evaluation-profiles) (Accepted). It is settled, not
open.

#### The demo repo is small, and the "10k-file" claim is deleted

**Binding decision, 2026-09-23.** An earlier revision of §8 promised the jury "a live CLI and UI
answering Q1/Q2/Q3 on **a real 10k-file JS repo**." That claim is **deleted everywhere it appears**,
and it was never costed:

| | |
|---|---|
| 10k files at Setup.md's own ~6.9 chunks/file | ≈ 69,000 chunks |
| vs. the `IndexFlatIP` → `IndexIVFPQ` switch at | 50,000 vectors |
| Consequence | forces IVF-PQ, which [NonGoals.md NG-18](NonGoals.md#ng-18--no-ann-index-on-the-benchmark-corpus) explicitly rules out for this cycle |
| Cold index at the §8.5 projection | hours, against a 12-minute requirement |
| Calendar available | 4 days |

The demo corpus is a small tagged repo. What that costs us, stated plainly rather than hidden: the
"works at scale" story becomes an argument from architecture (content-addressed blobs, an index-kind
switch, per-version manifests) rather than a demonstration, and `NFR-02`'s 50-changed-file diff
cannot be produced on a 10–50 file repo at all.

**So `NFR-02` is measured on a synthetic diff generator**, not on the demo repo — the generator in
`TestPlan.md §5.4`, which emits a controlled A/M/D/R changeset of a stated size against a generated
corpus. The live demo shows the *mechanism* on the real repo (a few changed files, a real rename, a
real deletion); the *number* comes from the generator. The two are reported separately and labelled
separately, and the demo script says which is which out loud. A 45-second figure presented as if it
came from the repo on screen would be the same class of defect as the retracted 14.7.

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
| `FR-03` | Decompose a multi-clause query into **≤ 3** ordered `sub_queries` when the plan requires more than one retrieval (e.g. "calls X before Y" → one sub-query per identifier plus an ordering check). The bound is **3**, matching `MAX_SUB_QUERIES` in `agent/planner.py`; `Schema.md §10`'s `len <= 4` is a defect and must be lowered. On the single-query path `sub_queries` stays **empty** — it is never `[original_query]`; `QueryPlan.effective_queries` is the derived property the fan-out iterates. Fusion arithmetic across sub-queries is specified normatively in [TechSpecifications.md §5.1.4](TechSpecifications.md#514-sub-query-fan-out-and-its-fusion-arithmetic): per-signal RRF merge (equal weight per sub-query, dedupe by `chunk_id`, truncate to that signal's own width, contiguous re-ranking) **before** cross-signal fusion, so `strategy_weights`, `contributions` and Schema invariant 10 are untouched. | P0 | Harshdeep |
| `FR-04` | Chunk source files on tree-sitter AST node boundaries into `Chunk` records targeting 64–512 tokens; split oversized functions at statement boundaries with 1-statement overlap; merge chunks < 16 tokens into their parent. Fall back to a line-window splitter when parsing fails. | P0 | Anish |
| `FR-05` | Build the dense index: embed every chunk with the locked embedder, L2-normalise, persist `dense.faiss` (`IndexFlatIP` under 50k vectors, `IndexIVFPQ` at or above 50k) plus `dense.idmap.json`. | P0 | Prabinder |
| `FR-06` | Dense retrieval: embed the query (or each sub-query) and return top **K=100** `ScoredChunk` records with `signal=DENSE`, inner-product scores, 1-indexed ranks. | P0 | Prabinder |
| `FR-07` | Build the sparse index with `bm25s` over the identical chunk corpus, using a code-aware tokenizer that splits camelCase/snake_case/dots while also retaining the original identifier token. | P0 | Prabinder |
| `FR-08` | Sparse retrieval: return top **K=100** `ScoredChunk` records with `signal=SPARSE`, using the query's raw terms plus `extracted_identifiers` and `expansion_terms`. | P0 | Prabinder |
| `FR-09` | Build the structural index into `structural.sqlite` with four relations: symbols (name, kind, file, line span, scope), calls (caller → callee, source order), imports (file → module), exports (module → symbol). | P0 | Anish |
| `FR-10` | Structural retrieval: answer callers-of, callees-of, imports-of, exports-of, and ordered-call-pair queries by SQL traversal; return top **K=50** `ScoredChunk` with `signal=STRUCTURAL`. Return an empty list rather than an error when no identifier resolves. | P0 | Anish |
| `FR-11` | Fuse the per-signal ranked lists with weighted RRF: `score(d) = Σ_i w_i / (60 + rank_i(d))`, weights from `QueryPlan.strategy_weights`; emit `FusedResult` carrying `contributions` and `dominant_signal`; truncate to **N=25**. | P0 | Prabinder |
| `FR-12` | Rerank the 25 fused candidates with the locked cross-encoder, write `rerank_score`, re-sort, and truncate to top **10**. Reranking must be skippable by config for ablation runs. | P0 | Harshdeep |
| `FR-13` | Bounded agentic refinement: after reranking, evaluate sufficiency (trigger: top-1 `rerank_score` < 0.35 **or** fewer than 3 results above 0.20). On trigger, rewrite/broaden the query and re-run retrieval. Hard caps: **2 TOTAL passes** (`AXIOM_AGENT_MAX_PASSES` — the initial retrieval counts as pass 1, so this permits at most **one** refinement, not two) and **5 s** wall clock, enforced by a monotonic deadline checked before each *new* pass and again before a new pass's rerank. Pass 1 is exempt from the deadline so an exhausted budget degrades to one pass, not to an empty answer. Best-of, not last: passes are compared on `(cross_encoder_ran, top1)`. Full semantics in [TechSpecifications.md §5.3](TechSpecifications.md#53-agent-loop-bound). | P0 | Harshdeep |
| `FR-14` | Format each survivor as a `RetrievalResult`: chunk text, `file_path`, `start_line`–`end_line`, final `score`, human-readable `match_reason` naming the dominant signal and the matched identifiers, and the per-signal rank map. | P0 | Harshdeep |
| `FR-15` | Populate `optimization_hint` for a result when a static pattern-check fires on the surfaced chunk (sync I/O in an async path, `await` inside a loop, unbounded `for..in` over a network payload). Hint is a fixed string from a rule table, never LLM prose. | B | Harshdeep |
| `FR-16` | Stamp every chunk with `version_id`, `commit_sha`, and `last_modified` at index time, resolved from git when the target is a repo and from a config override otherwise. | P1 | Parth |
| `FR-17` | Maintain `.axiom/registry.json` mapping `version_id` → manifest path plus the active version, and write one `VersionManifest` per version including `file_hashes`, `embedding_model`, `embedding_dim`, `index_kind`, and `parent_version`. | P1 | Parth |
| `FR-18` | Incremental reindex: resolve `git diff --name-status <old>..<new>` into {A,M,D,R}; re-chunk and re-embed only A and M files; drop D chunks from all three indexes; update the structural graph edges for touched files only. Full rebuild remains available as `--full`. | P1 | Parth |
| `FR-19` | Content-addressed embedding reuse: persist vectors at `.axiom/blobs/<content_hash>.npy` and reuse them across versions, so an unchanged-content rename or a revert costs zero embedding compute. | P1 | Parth |
| `FR-20` | Version-scoped query: `--version <id>` restricts retrieval to that version's index; absent the flag, the registry's active version is used. | P1 | Parth |
| `FR-21` | Evolutionary retrieval: with `--all-versions`, search across every indexed version, group chunks with cosine ≥ 0.95 sharing `symbol` + `file_path` into a `SnippetFamily`, choose the newest member as `representative`, compute `stability = members / total_versions`, apply the ranking bonus `final = base * (1 + 0.10 * stability)` to families spanning ≥ 2 versions, and attach per-transition `diffs`. | B | Parth |
| `FR-22` | MTEB adapter: expose the full pipeline as an MTEB-compatible encoder, run `AppsRetrieval` test split, and export `appsretrieval_results.json` containing `ndcg_at_10`, `mrr_at_10`, `recall_at_100`, the **dense-only baseline `B`** (§2.0), the active profile name, the config hash and the seed — so the relative gain is computable from the artefact alone. This JSON is the single file attached to the release; its shape needs a P0 test case, which `TestPlan.md` does not currently have. | P0 | Parth |
| `FR-23` | Typer CLI **`axiom`** (not `prism` — [ADR-015](Decisions.md#adr-015--rename-prism-to-axiom), and `_CONTRACT.md §1` is the stale document on this axis) with subcommands: `index`, `reindex`, `query`, `classify`, `versions`, `families`, `eval`, `serve`, `ui`, `gc`, plus `version` as a hidden alias of `versions`. These eleven are what `src/axiom/cli.py` actually registers. There is **no `search` subcommand** — documents that invoke `axiom search` are wrong and will fail live. Every subcommand supports `--json` and returns non-zero on failure. | INF | Prabinder |
| `FR-24` | FastAPI service on port 8000: `POST /query`, `GET /versions`, `GET /chunk/{chunk_id}`, `GET /health`, all typed by the same Pydantic models as the CLI. Dev-only; no auth (see `NG-08`). | INF | Harshdeep |
| `FR-25` | Streamlit UI on port 8501: query box, query-type badge, result cards with syntax-highlighted snippet and `file:line` header, per-signal rank breakdown, agent-pass indicator, version selector, and an expandable snippet-family view. | INF | Harshdeep |
| `FR-26` | Supervised tuning on the `AppsRetrieval` **train** split. The 5,000 train query-document pairs are the only corpus used to tune RRF signal weights per profile, the rerank sufficiency thresholds (0.35 / 0.20), and query-preprocessing variants. Split 4,000 tune / 1,000 dev, seeded and committed as an id list under `data/splits/`. No gradient training of any model (see `NG-06`). **The test split is never used for tuning, under any circumstance, including schedule pressure** ([NonGoals.md NG-29](NonGoals.md#ng-29--no-tuning-on-the-benchmark-test-split), confirmed absolute 2026-09-23; see also Assumption A-6, whose fallback no longer has a path to the test split). Each candidate configuration is scored on test at most once and every such run is logged in [Tracker.md](Tracker.md). | P0 | Parth |

---

## 6. Non-Functional Requirements

Every `NFR` below is a **requirement**. The **Status** column says whether it has been measured.
As of 2026-09-23, none has.

| ID | Requirement | Measurement | Status | Owner |
|---|---|---|---|---|
| `NFR-01` | Cold index of 10,000 chunks completes in **≤ 12 min** on the reference box (8-core CPU, 16 GB RAM). | `scripts/bench_latency.py --phase index` wall clock | `# PLACEHOLDER`, `OQ-14`. **Projected 20–40 min** — [TechSpec §8.5](TechSpecifications.md#85-cold-index-placeholder-oq-14). NonGoals' "measured at 636 s" is retracted. | Prabinder |
| `NFR-02` | Incremental reindex of a 50-changed-file diff completes in **≤ 45 s**. | `axiom reindex` wall clock, 3-run median, **on the synthetic diff generator** (`TestPlan.md §5.4`) — the demo repo is 10–50 files and cannot produce a 50-file diff (§2.2) | `# PLACEHOLDER`, `OQ-14` | Parth |
| `NFR-03` | Query p50 latency with the agent loop disabled is **≤ 900 ms**, on the `demo`/`default` profile with `AXIOM_LLM_ENABLED=false`. **No number from `configs/eval.yaml` may be quoted against this** (that profile is offline and untimed, TechSpec §8.2), and there is no p50 claim for the LLM-on path. | 100-query sample, `bench_latency.py --phase query` | `# PLACEHOLDER`, `OQ-13`. Projected ~310–770 ms | Prabinder |
| `NFR-04` | Query p95 latency with up to 2 **total** agent passes (initial retrieval included — `FR-13`) is **≤ 5 s**, enforced by a hard monotonic deadline, not by hope. | Same harness with `AXIOM_AGENT_MAX_PASSES=2` | `# PLACEHOLDER`, `OQ-13`. The deadline bounds it regardless of the projection | Harshdeep |
| `NFR-05` | Peak resident set size during query serving stays **≤ 4 GB**. | `psutil` RSS sampled at 100 ms during the query benchmark | `# PLACEHOLDER`, `OQ-15`. Component tally projects **2.7–3.5 GB (`demo`) / 3.3–4.1 GB (`eval`)** — [TechSpec §8.6](TechSpecifications.md#86-peak-rss-placeholder-oq-15). If the measurement exceeds 4 GB the honest fix is lazy-load-and-release, or raising this NFR to 6 GB and saying why | Prabinder |
| `NFR-06` | CPU-only execution. No CUDA, ROCm, or accelerator dependency anywhere in the install or runtime path; ONNX Runtime uses the CPU execution provider exclusively. | `pip check` on a CPU-only image plus a smoke run in `python:3.11-slim-bookworm`, **plus an automated source-and-lockfile grep for `nvidia-`/`cu12`/`cuda` in CI** — `Rules.md` calls this "the check that will actually save us" but cites a test id that does not exist; it needs a real one | unverified — no automated check exists today | Prabinder |
| `NFR-07` | Mandatory graceful degradation. Every model has a declared fallback (embedder → MiniLM, reranker → ms-marco-MiniLM-L-6-v2, LLM → heuristic rule engine, tree-sitter → regex identifier extraction). The full pipeline must produce ranked results with `AXIOM_LLM_ENABLED=false` and with any single primary model unavailable. A file that fails to parse degrades to window chunking and never aborts the index. | Smoke matrix: 4 degradation scenarios × 3 archetype queries | implemented; unmeasured end to end | Harshdeep |
| `NFR-08` | Determinism. Identical query + identical index + identical config yields byte-identical ranked `chunk_id` order. All sampling temperatures are 0; FAISS `IndexIVFPQ` training is seeded; `chunk_id` and `content_hash` are pure blake2b-128 functions of their inputs. The comparison must exclude `elapsed_ms` **and every value under `timings`** — both are wall-clock and neither is deterministic. | Repeat-run diff over the **100**-query `tests/fixtures/bench_queries.txt` (one fixture, one size; `Rules.md`'s 50 is a restatement and should link instead) | unmeasured | Prabinder |
| `NFR-09` | Reproducibility. A clean clone reaches a working index with `uv sync` + one documented command; `uv.lock` is committed; a Dockerfile builds the same environment; model revisions are pinned by name **and revision** in `configs/`. **No revision field is defined anywhere in the suite today** — either add `AXIOM_*_REVISION` settings and resolve the SHAs on Day 1, or drop the revision half of this claim rather than shipping it unbacked. | Fresh-clone rehearsal, Day 4 (2026-09-26) | partially unbacked — see note | Parth |
| `NFR-10` | Observability. Every stage emits a structured timing record; `--json` output includes a `timings` block so latency claims are auditable. The stage tags the code actually emits at query time are `plan`, `agent.fan_out`, `fuse`, `hydrate`, `rerank` (there is no `classify` tag — it is inside `plan` — and no `format` tag); index-time adds `walk`, `chunk`, `blob`, `index`, `manifest`, `write`, `diff`, `evolutionary`. Any document listing a different set is wrong. | Inspect one `--json` response per archetype | implemented; the tag list above is read from `src/` | Harshdeep |
| `NFR-11` | Code health. `ruff` lint + format clean at line length 100; `mypy --strict` clean on `src/axiom/core`, `src/axiom/retrieval`, `src/axiom/schema`; `pytest` green in CI; CI order is ruff → mypy → pytest → smoke index. | GitHub Actions run | **met** — ruff clean, 400 tests passing as of 2026-09-23 | Prabinder |
| `NFR-12` | Storage and first-run footprint. On-disk index for 10k chunks stays under 1.5 GB including shared blobs. **Model download: the 2.5 GB primary-profile ceiling is retracted** — Setup.md's own model table sums to ~4.6 GB downloaded before any ONNX export, and Setup.md itself says "budget ~7 GB for models in total," so the requirement contradicted its own source document by 1.85x. Restated: **≤ 5 GB downloaded, ≤ 7 GB transient on disk during export, ~1.2 GB steady-state** after the `-fp32/` directories are deleted. The `fast`/fallback profile's **≤ 500 MB** promise is kept — that one is genuinely met. | `du -sh .axiom` plus model cache size after a clean run | index size unmeasured; model sizes derived from Setup.md §5.1 | Parth |

---

## 7. Prioritisation (MoSCoW)

Mapped onto the P0/P1/Bonus ladder and the **re-baselined 2026-09-23 → 2026-09-27 build window**
(Day 1 = 2026-09-23, submission = 2026-09-27). The 15–25 September calendar that previously appeared
here and in `ImplementationPlan.md` is **void**.

| MoSCoW | Scope | Requirements | Land by |
|---|---|---|---|
| **Must have** | **The reportable number first:** the dense-only baseline `B` (§2.0), then P0 screening path: chunking, all three indexes, all three retrievers, RRF, reranker, result formatting, CLI, MTEB export, train-split tuning | `FR-01`, `FR-02`, `FR-04`–`FR-12`, `FR-14`, `FR-22`, `FR-23`, `FR-26`; `NFR-01`, `NFR-03`, `NFR-05`, `NFR-06`, `NFR-08`, `NFR-09`, `NFR-11` | Day 3 (25 Sep) |
| **Must have** | Bounded agent loop — the word "agentic" is in the theme title; without it we are a hybrid search box | `FR-03`, `FR-13`; `NFR-04`, `NFR-07` | Day 3 (25 Sep) |
| **Should have** | P1 version support: tagging, manifests, git-diff incremental reindex, blob reuse, version-scoped query | `FR-16`–`FR-20`; `NFR-02`, `NFR-12` | Day 4 (26 Sep) |
| **Should have** | Demo surfaces the jury actually touches | `FR-24`, `FR-25`; `NFR-10` | Day 4 (26 Sep) |
| **Could have** | Bonus evolutionary retrieval with snippet families and stability bonus | `FR-21` | Day 4 (26 Sep), after P1 lands |
| **Could have** | Optimization hints on surfaced code | `FR-15` | Day 4 (26 Sep), cut first if behind |
| **Won't have** | Everything in [NonGoals.md](NonGoals.md), `NG-01`–`NG-30` — the *full* range NonGoals actually defines, not the `NG-01`–`NG-20` this row previously fenced. The range is open-ended: a new `NG-##` is inside this fence the moment it is written, without an edit here. Note `NG-28` and `NG-29` are cited as `RISK-09`/`RISK-10` mitigations, so truncating the fence at 20 silently un-fenced two risk mitigations. | — | Never, this cycle |

### 7.1 Cut Order Under Time Pressure

If the schedule slips, features are dropped in exactly this order. This list is agreed in
advance so nobody negotiates it at 2 a.m. on 26 Sep.

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
| Working prototype and functionality | **30%** | A live CLI and UI answering Q1/Q2/Q3 on the **small tagged JavaScript demo repo** (≈10–50 source files, ≥3 tags — §2.2), plus `axiom reindex` running in front of the jury and `--all-versions` collapsing a snippet family. The demo video shows wall-clock timings on screen, **labelled as single-run observations, not p50s**. Degradation matrix proves it also works on the evaluator's own locked-down laptop. *(The "real 10k-file JS repo" this cell previously promised is deleted — §2.2 gives the arithmetic that killed it. Scale is argued from architecture here, not demonstrated.)* | `FR-06`, `FR-08`, `FR-10`–`FR-14`, `FR-18`, `FR-21`, `FR-23`, `FR-25`, `NFR-01`–`NFR-07` |
| Technical depth and feasibility | **25%** | Three genuinely different signals with a principled fusion rule (weighted RRF, `k=60`, per-query-type weights), a two-stage retrieve-then-rerank architecture with declared candidate widths (100/100/50 → 25 → 10), content-addressed incremental indexing, and a locked Pydantic data contract in [Schema.md](Schema.md). Every constant in [TechSpecifications.md](TechSpecifications.md) is justified, not guessed. Ablation table (dense only / +sparse / +structural / +rerank / +agent) shows each component's contribution in NDCG@10 **as a relative gain over our own measured dense-only baseline `B`** (§2.0) rather than against an external number we cannot verify. Every external comparison row carries paper + table + column or it is not shown. | `FR-04`, `FR-05`, `FR-07`, `FR-09`, `FR-11`, `FR-12`, `FR-19`, `NFR-08`, `NFR-11` |
| Innovation and originality | **20%** | The two things no comparable submission has: (a) a **structural AST/call-graph signal** as a first-class retrieval list inside the fusion, which is the only way Q2-class ordering queries are answerable at all; (b) a **bounded agentic refinement loop** with an explicit sufficiency predicate, a hard 5 s monotonic deadline, and a 2-**total**-pass cap — agentic in a way that is measurable rather than decorative. Plus snippet-family evolutionary retrieval with stability-weighted ranking. | `FR-09`, `FR-10`, `FR-13`, `FR-21`, `FR-15` |
| Relevance to theme | **15%** | Theme 01 asks for agentic code intelligence under a CPU constraint on a repo larger than any context window. Axiom never feeds code to an LLM — the LLM only classifies, expands, decomposes, and judges sufficiency. Everything is retrieval over pre-built indexes. The P1 story (codebases keep changing with new commits) is answered with real git-diff incremental reindexing, not a content-hash cache bolted onto embeddings. This is directly shippable as a Samsung PRISM worklet: index a Samsung repo, serve queries to its owning team. | `FR-13`, `FR-16`–`FR-21`, `NFR-06` |
| Presentation and documentation | **10%** | This `docs/` suite: [PRD.md](PRD.md), [TechSpecifications.md](TechSpecifications.md), [Design.md](Design.md), [Schema.md](Schema.md), [Appflow.md](Appflow.md), [TestPlan.md](TestPlan.md), [Decisions.md](Decisions.md) (ADRs), [ImplementationPlan.md](ImplementationPlan.md) (day plan + `RISK-01`–`RISK-12`), [Tracker.md](Tracker.md), [NonGoals.md](NonGoals.md), [OpenQuestions.md](OpenQuestions.md), [Setup.md](Setup.md), [API.md](API.md), [Glossary.md](Glossary.md). A README with a 5-command setup and a Docker path. `Incognito_Submission_ppt` covering problem, architecture, stack, innovation, results, limitations. A ≤ 5-minute demo video. Release `PRISM_GENAI_HACKATHON_Y2026` (organiser-prescribed tag) with `appsretrieval_results.json` and the ablation table attached. | `FR-22`, `NFR-09`, `NFR-10` |

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
what Axiom adds, and it is the source of the 20% innovation argument in §8.

| Dimension | Rival approach | Axiom | Why it matters to the rubric |
|---|---|---|---|
| Retrieval signals | Dense + BM25 (2) | Dense + BM25 + **structural AST/call-graph** (3) | Query archetype Q2 ("calls X before Y") is unanswerable with 2 signals. The third list is the only thing that makes structural queries work, and it is the headline innovation claim. |
| Fusion | RRF | Weighted RRF, `k=60`, weights selected per `QueryType` | A usage query should not weight dense the same as a semantic query. Per-type weights are a measurable ablation, not a stylistic difference. |
| Reranking | Cross-encoder | Cross-encoder, with reranking made ablatable so its contribution is reportable | Same capability; we additionally quantify it. |
| Agentic behaviour | None — single-pass pipeline | **Bounded agent loop:** sufficiency predicate (top-1 < 0.35 or < 3 results > 0.20, both `# PLACEHOLDER`/`OQ-10`), query rewrite, max 2 **total** passes, 5 s monotonic deadline | The theme is titled *Agentic* Code Intelligence. A single-pass pipeline is a strong retriever but concedes theme relevance (15%) and innovation (20%). |
| Query understanding | Query text as given | Classification into 4 types, identifier extraction, code-synonym expansion, decomposition into ≤ 3 sub-queries with a defined per-signal RRF merge ([TechSpec §5.1.4](TechSpecifications.md#514-sub-query-fan-out-and-its-fusion-arithmetic)) | Directly drives the per-type fusion weights and the structural routing. |
| Versioning (P1) | Content-hash embedding cache | Per-version manifests + registry, `git diff --name-status` A/M/D/R handling, content-addressed shared embedding blobs, version-scoped retrieval | A hash cache avoids recompute but does not let you *query a version*. P1 asks for retrieval across versions, not just cheaper reindexing. |
| Evolutionary (Bonus) | Not addressed | `SnippetFamily` grouping at cosine ≥ 0.95 on `symbol`+`file_path`, newest-as-representative, `stability` score, `final = base * (1 + 0.10 * stability)`, per-transition diffs | This is the stated bonus goal. Attempting it credibly is differentiating on its own. |
| Degradation story | Not stated | Every model has a declared fallback; pipeline runs with `AXIOM_LLM_ENABLED=false`; parse failures degrade to window chunking | Protects the 30% prototype score on an unknown evaluator machine. |
| Fusion weighting on the benchmark | Default `FUSION_WEIGHTS = (0.5, 0.5)` | Profile-specific weights tuned on the 5,000 train pairs (`FR-26`); sparse deliberately down-weighted in `configs/eval.yaml` | APPS pairs English prose queries with Python solution documents, so a lexical signal has very little shared surface vocabulary to match on; fusing a weak list at equal weight spends the reranker's candidate slots on noise. The *mechanism* is the point of difference. The **magnitude** (0.15) is a `# PLACEHOLDER` pending `OQ-02`'s sweep, whose range must include 0.0 — the "BM25 4.8 vs BGE 14.7" figures that previously justified it are retracted (§2.0.3). |
| Libraries | `rank-bm25`, FAISS `Flat` | `bm25s` (locked), FAISS `IndexFlatIP` under 50k vectors / `IndexIVFPQ` at or above 50k | `bm25s` is materially faster on a 10k-chunk corpus; the index-kind switch keeps the cold-index budget reachable at demo scale. |
| Reported results | None yet — all feature flags off, `experiments.md` all `TBD` | Ablation table with a number in every row, plus `appsretrieval_results.json` attached to the release | An unmeasured pipeline cannot claim a score. Having numbers is itself a 30%/25% differentiator. **As of 2026-09-23 we have none either** — the implementation exists and is tested, but no eval has been run. This row is an intention until `T-200a` produces `B`. |

**Honest assessment of our risk relative to the rival:** they have fewer moving parts, so
their P0 number could land earlier and more safely once they turn their flags on. Our
mitigation is sequencing — the dense-only baseline `B` runs **first**, before sparse,
structural, rerank, the agent loop, the UI and the API, so a submittable number exists on
Day 1 and everything after it is upside rather than prerequisite. See the risk register at
[ImplementationPlan.md#risk-register](ImplementationPlan.md#risk-register) and the day plan
in [ImplementationPlan.md](ImplementationPlan.md).

**Name collision.** They ship under the identical project name and the identical release
tag `PRISM_GENAI_HACKATHON_Y2026`. The tag is prescribed by the organisers, so it cannot be
changed; the project name can, and did. **Closed:** the project is **Axiom**
([`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission), closed
2026-09-16; [ADR-015](Decisions.md#adr-015--rename-prism-to-axiom), Accepted). Package and import
root `axiom`, CLI `axiom`, env prefix `AXIOM_`, index root `.axiom/`. Only the release tag and the
event name keep the word PRISM.

---

## 10. Assumptions

| # | Assumption | If false |
|---|---|---|
| A-1 | CoIR `AppsRetrieval` is reachable through the MTEB/HuggingFace path and is BEIR-shaped (`_id`, `text`, `title`, `language`, `meta_information`). | Vendor a cached copy under `data/` early; `scripts/run_eval.py` supports a local corpus path. |
| A-2 | A **small** (≈10–50 source files) real JavaScript repository with ≥3 meaningful tags can be sourced for the P1/Bonus demo (§2.2). | Tracked as [`OQ-07`](OpenQuestions.md#oq-07--which-multi-version-javascript-repo-anchors-the-p1bonus-demo); fallback is a synthesised version history over a chosen OSS JS repo's real tags, or a hand-authored fixture repo. Either way `NFR-02`'s 50-file diff comes from the synthetic generator, not from this repo. |
| A-3 | The reference box (8-core CPU, 16 GB RAM) approximates the evaluator's machine. | `configs/fast.yaml` fallback profile targets a 4-core / 8 GB floor. |
| A-4 | `Qwen3-Embedding-0.6B` INT8 throughput is sufficient for a 10k-chunk cold index inside 12 min. **This assumption is probably false**: the arithmetic in [TechSpec §8.5](TechSpecifications.md#85-cold-index-placeholder-oq-14) projects 20–40 min. | Tracked as [`OQ-14`](OpenQuestions.md#oq-14--what-is-the-real-embedder-throughput-and-cold-index-time) (not `OQ-02`, which is the sparse weight — the previous citation was wrong). Declared fallback is `all-MiniLM-L6-v2` for the demo index, keeping Qwen3 for the one-off APPS encode. Measure on Day 1 before anything else. |
| A-5 | The structural signal transfers value to the **live demo** corpus (JavaScript, multi-module). It is **not** assumed to transfer to the APPS eval corpus — see §2.2. | If it fails on the demo corpus too, the innovation claim collapses to the agent loop alone. Not separately tracked as an `OQ`; it is answered the first time a Q2 query runs against the resolved [`OQ-07`](OpenQuestions.md#oq-07--which-multi-version-javascript-repo-anchors-the-p1bonus-demo) repo. (It previously cited `OQ-01`, which is a different question and has been Closed since 2026-09-16.) |
| A-6 | The 5,000 `AppsRetrieval` train pairs are usable for weight and threshold tuning without any gradient training. | **See §10.1 — the fallback for this assumption is constrained and is the one place in this document where the wording matters more than the content.** Tracked as [`OQ-05`](OpenQuestions.md#oq-05--is-the-5000-row-train-split-usable-for-tuning-as-assumed). |

### 10.1 A-6's fallback, rewritten so it cannot reach the test split

**This replaces the previous fallback text, which read: "fall back to a 500-query holdout carved
from the dev portion of *test*."** That sentence proposed exactly the action
[NonGoals.md NG-29](NonGoals.md#ng-29--no-tuning-on-the-benchmark-test-split) calls "the one
non-goal whose violation is invisible in the artefact and fatal to the claim" — written into a
requirements document as the sanctioned contingency, where it would have been reached precisely when
people were tired and behind schedule. It is deleted, not softened.

**The fence is absolute.** The `AppsRetrieval` test split is used for exactly one thing: the
reportable run. It is never split, sampled, held out from, eyeballed per-query, or used to compare
two configurations. There is no circumstance — including running out of time on 26 September — in
which a tuning holdout is carved from it.

If the train split is unavailable or mis-shaped, the fallback ladder is, in order:

| Rung | Fallback | What we then claim |
|---|---|---|
| 1 | Tune on the **demo corpus** with a hand-written query set (20–40 queries over the tagged JS repo, authored by us, qrels by inspection). | "Constants tuned on a held-out corpus disjoint from the benchmark." Weaker transfer, fully honest. |
| 2 | Tune on a **synthetic query set** generated from the APPS *corpus documents only* — never from test queries or qrels. | Same claim as rung 1, with a caveat about synthetic-query realism. |
| 3 | **Ship the locked default constants untuned** and say so in the results slide and in `Tracker.md`. | "Sparse weight 0.15, sufficiency 0.35/0.20 — defaults, unswept; the ablation below isolates each component at those defaults." |

Rung 3 is a perfectly acceptable submission. A number obtained at rung 3 is lower and true; a number
obtained by carving the test split is higher and unusable, and only we would ever know. If the
schedule forces a choice, take rung 3 and write it down.

---

## 11. Out of Scope

Deliberate exclusions are enumerated with rationale in [NonGoals.md](NonGoals.md) as
`NG-01`–`NG-30` — the full range NonGoals defines, and open-ended: a new `NG-##` is out of scope the
moment it is written. The headline ones: no code generation, no answer synthesis, no
natural-language explanation of code, no language beyond JavaScript, no GPU path, no model
training, no auth or multi-tenancy, no IDE plugin, no hosted deployment.

Unresolved decisions are tracked in [OpenQuestions.md](OpenQuestions.md) as `OQ-##` and are
promoted to `ADR-###` entries in [Decisions.md](Decisions.md) when settled.
