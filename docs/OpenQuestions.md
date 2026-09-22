# Open Questions

Unresolved design and product questions for PRISM: tracked with an owner and a deadline, never silently dropped, and promoted to an `ADR-###` in [Decisions.md](Decisions.md) the moment they are settled.

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-15
**Status:** Draft

Related: [PRD.md](PRD.md) · [NonGoals.md](NonGoals.md) · [Decisions.md](Decisions.md) · [TechSpecifications.md](TechSpecifications.md) · [ImplementationPlan.md](ImplementationPlan.md) · [Tracker.md](Tracker.md) · [Rules.md](Rules.md) · [TestPlan.md](TestPlan.md)

---

## How to read this document

An `OQ-##` is a question with a real answer that changes what we ship, and the answer is not yet
known. Three things this document is not:

- **Not a non-goal.** A settled exclusion is an `NG-##` in [NonGoals.md](NonGoals.md). An `OQ-##`
  that resolves to "we will not do this" gets closed and becomes an `NG-##`, with a note here
  pointing at it.
- **Not a task.** "Implement the structural retriever" is a `T-###` in [Tracker.md](Tracker.md).
  An `OQ-##` is a question whose answer a task cannot make true by itself — it needs a measurement,
  a decision among alternatives, or an external fact we do not yet control.
- **Not permanently open.** Every entry has an owner and a deadline. An entry with no deadline is a
  defect in this document, not a feature of the question.

IDs are permanent, like `NG-##`. A closed question keeps its number with a one-line resolution note
and a pointer to the `ADR-###` that closed it, because other documents cite `OQ-##` by number
([PRD.md](PRD.md), [NonGoals.md](NonGoals.md), [Rules.md](Rules.md) all do).

## Status vocabulary

| Status | Meaning |
|---|---|
| `Open` | Not yet measured or decided. |
| `Measuring` | An experiment or sweep is in flight; a `T-###` in [Tracker.md](Tracker.md) owns it. |
| `Closed` | Resolved. An `ADR-###` records the decision and the evidence. |

---

## Index

| ID | Question | Owner | Deadline | Status | Priority |
|---|---|---|---|---|---|
| [`OQ-01`](#oq-01--is-the-two-profile-eval--demo-split-the-final-shape) | Is the two-profile eval / demo split the final shape? | Harshdeep | 2026-09-17 | Closed | P0 |
| [`OQ-02`](#oq-02--what-is-the-exact-sparse-weight-in-configseval-yaml) | What is the exact sparse weight in `configs/eval.yaml`? | Prabinder | 2026-09-21 | Open | P0 |
| [`OQ-03`](#oq-03--is-qwen3-embedding-06b-the-right-embedder-for-apps) | Is Qwen3-Embedding-0.6B the right embedder for APPS? | Prabinder | 2026-09-19 | Open | P0 |
| [`OQ-04`](#oq-04--project-name-collision-with-a-rival-submission) | Project name collision with a rival submission | Harshdeep | 2026-09-20 | Closed | P0 |
| [`OQ-05`](#oq-05--is-the-5000-row-train-split-usable-for-tuning-as-assumed) | Is the 5,000-row train split usable for tuning as assumed? | Parth | 2026-09-16 | Open | P0 |
| [`OQ-06`](#oq-06--does-reverse-doc2query-expansion-earn-its-index-time-cost) | Does reverse doc2query expansion earn its index-time cost? | Prabinder | 2026-09-21 | Open | P1 |
| [`OQ-07`](#oq-07--which-multi-version-javascript-repo-anchors-the-p1bonus-demo) | Which multi-version JavaScript repo anchors the P1/Bonus demo? | Parth | 2026-09-17 | Open | P0 |
| [`OQ-08`](#oq-08--is-hyde-worth-its-latency-on-any-profile) | Is HyDE worth its latency on any profile? | Harshdeep | 2026-09-22 | Open | P2 |
| [`OQ-09`](#oq-09--are-the-chunking-placeholders-64512-tokens-16-token-floor-right) | Are the chunking placeholders (64–512 tokens, 16-token floor) right? | Anish | 2026-09-20 | Open | P1 |
| [`OQ-10`](#oq-10--are-the-agent-sufficiency-thresholds-035020-right) | Are the agent sufficiency thresholds (0.35/0.20) right? | Harshdeep | 2026-09-21 | Open | P0 |
| [`OQ-11`](#oq-11--are-the-evolutionary-dedupe-cosine-095-and-stability-bonus-010-right) | Are the evolutionary dedupe cosine (0.95) and stability bonus (0.10) right? | Parth | 2026-09-22 | Open | P1 |
| [`OQ-12`](#oq-12--does-the-jury-need-remote-access-to-the-demo-surfaces) | Does the jury need remote access to the demo surfaces? | Harshdeep | 2026-09-18 | Closed | P1 |

---

## Entries

### `OQ-01` — Is the two-profile eval / demo split the final shape?

**Question:** [PRD.md §2.2](PRD.md#22-two-evaluation-contexts-two-profiles) commits to `configs/eval.yaml`
(dense-heavy, structural disabled) and `configs/demo.yaml` (all three signals) as separate,
permanent profiles rather than one profile with a config flag. Is the split itself — not the
weights inside it, that is `OQ-02` — the right shape, or does a single profile with a
`structural_enabled` toggle serve both corpora just as well with less config surface?

**Why it matters:** Every other decision in the retrieval stack — candidate widths, the structural
signal's very existence in the fusion, the jury-facing story in [PRD.md §8](PRD.md#8-jury-scoring-alignment)
— assumes two profiles exist. Reversing this after `T-018`–`T-040` (dense + sparse hybrid,
scheduled Day 4) land would mean re-threading profile selection through the CLI, API, and eval
runner.

**What would resolve it:** Confirming that a single-profile design cannot express "structural
signal contributes ~0 to NDCG@10 and costs latency and index-build time to compute at all" without
an escape hatch that is functionally a second profile anyway.

**Resolution:** **Closed 2026-09-16.** Two profiles is correct: a single profile with a toggle still
needs two candidate-width vectors, two weight vectors, and two "does this index exist" branches —
it is the same complexity with worse discoverability (`axiom index --profile eval` states the
active profile in every log line; a boolean flag does not). Recorded as
[`ADR-001`](Decisions.md#adr-001--two-first-class-evaluation-profiles).

---

### `OQ-02` — What is the exact sparse weight in `configs/eval.yaml`?

**Question:** [PRD.md §2](PRD.md#2-goals-and-success-metrics) states BM25 scores 4.8 NDCG@10 on APPS
against BGE-0.6B's 14.7, and that fusing a 4.8-scoring signal at equal weight is a net negative — but
does not commit to a number. Is `{dense 0.85, sparse 0.15, struct 0.0}` (the placeholder quoted in
[NonGoals.md NG-04](NonGoals.md#ng-04--no-language-support-beyond-javascript-demo-and-python-benchmark))
the measured optimum, or a guess that happens to appear in two other documents because it was typed
once and copied?

**Why it matters:** This is the single highest-leverage knob for the P0 screening score. A wrong
guess costs NDCG@10 points that no amount of reranker tuning recovers, because a bad first-stage
fusion never puts the right chunk inside the top-25 the reranker sees.

**What would resolve it:** A sweep over `{0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}` sparse weight on
the 4,000-row tune split (`FR-26`), holding structural at 0.0, reporting NDCG@10 for each point.
Owned as `T-112` in [Tracker.md](Tracker.md).

**Current status:** Open, `Measuring` from Day 7 (21 Sep) once the dense+sparse+rerank pipeline is
green. `0.15` ships as the `# PLACEHOLDER` default in `configs/eval.yaml` until then, per the
[Rules.md §8](Rules.md#8-the-placeholder-convention) convention.

---

### `OQ-03` — Is Qwen3-Embedding-0.6B the right embedder for APPS?

**Question:** [PRD.md Assumption A-4](PRD.md#10-assumptions) assumes Qwen3-Embedding-0.6B INT8
throughput is sufficient for a 10k-chunk cold index inside 12 minutes, and
[PRD.md §2](PRD.md#2-goals-and-success-metrics) notes that a code-specialised model
(UniXcoder 0.1B, NDCG@10 = 1.4) can score *below* plain BM25 (4.8) on this exact benchmark, because
NL-to-code semantic alignment — not "being a code model" — is what APPS actually rewards. Does
Qwen3-Embedding-0.6B, a general-purpose multilingual embedder, actually align better with
English-prose-to-Python-solution retrieval than a code-pretrained alternative of similar size?

**Why it matters:** The whole "0.6B closing most of the gap to a 7B model" claim in
[PRD.md §2](PRD.md#2-goals-and-success-metrics) depends on this model choice being the strongest
available at its parameter budget for *this specific* task shape, not merely the strongest at code
retrieval in general.

**What would resolve it:** A dense-only (no fusion, no rerank) NDCG@10 comparison on the 1,000-row
dev slice between Qwen3-Embedding-0.6B and the declared fallback `all-MiniLM-L6-v2`, plus one
additional candidate if time allows (`OQ-06`'s doc2query work touches the same slice and should be
sequenced after this). Owned as `T-024` in [Tracker.md](Tracker.md).

**Current status:** Open. The model stack in `_CONTRACT.md §2` is locked for engineering purposes
(everything downstream — ONNX export, INT8 quantisation, `VersionManifest.embedding_dim` — is built
against it), so this question resolves to "confirm or swap before Day 4" rather than "keep both
options live indefinitely."

---

### `OQ-04` — Project name collision with a rival submission

**Question:** [PRD.md §9](PRD.md#9-competitive-positioning) documents a rival public submission at
`github.com/DeshnaDey/Samsung-PRISM` — same theme, same project name, same release tag
`PRISM_GENAI_HACKATHON_Y2026`. The release tag is prescribed by the organisers and cannot change.
Should we rename the project to remove ambiguity for the jury?

**Why it matters:** A jury reviewing dozens of submissions under time pressure conflating two
differently-scoped entries under the identical name is a presentation risk with no engineering fix.

**Resolution:** **Closed 2026-09-16.** Renamed **PRISM → Axiom**. `src/axiom/` is the import root,
`AXIOM_` is the environment-variable prefix, and the package layout in `_CONTRACT.md §0/§3` already
reflects the new name. The release tag remains `PRISM_GENAI_HACKATHON_Y2026` (organiser-mandated,
unrelated to the project's own name). Recorded as
[`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom).

**Known residue — not yet fully propagated, tracked as `T-201`:** this document suite (`docs/`) and
the error taxonomy in `src/axiom/core/errors.py` (`PrismError`, `PrismConfigError`,
`PrismContractError`, `PrismIndexError`, `PrismModelError`, `PrismParseError`, `PrismBudgetError` —
see [Rules.md §9.2](Rules.md#92-error-taxonomy)) still carry the old name in headers and identifiers
minted before the rename. `_CONTRACT.md §1` also still names the CLI entrypoint `prism`, while
[Setup.md](Setup.md) and [TestPlan.md](TestPlan.md) already invoke it as `axiom`. This is a real,
observed inconsistency in the current doc set, not a hypothetical — do not "fix" it by silently
renaming across files outside a reviewed PR; it is `T-201`'s job, scheduled for Day 9 polish, after
which every doc title, CLI reference, and error class name reads `Axiom`/`axiom`/`AxiomError`
uniformly. Until `T-201` lands, prefer whichever name the *file you are editing* already uses.

---

### `OQ-05` — Is the 5,000-row train split usable for tuning as assumed?

**Question:** [PRD.md Assumption A-6](PRD.md#10-assumptions) and
[NonGoals.md NG-29](NonGoals.md#ng-29--no-tuning-on-the-benchmark-test-split) both assume the
`AppsRetrieval` train split is reachable, BEIR-shaped, and large enough to support a 4,000/1,000
tune/dev split without the two halves being pathologically different in difficulty.

**Why it matters:** `FR-26` — the entire tuning methodology that keeps us honest about `NG-29` — has
no fallback path documented anywhere except a vague "carve a 500-query holdout from dev" note in
Assumption A-6. If the train split turns out to be a different shape than the test split (e.g.
shorter problem statements, a narrower difficulty band), weights tuned on it may not transfer, and
we would not know until the one permitted test-split run.

**What would resolve it:** Load the train split on Day 1–2 (`T-003` in
[Tracker.md](Tracker.md)), confirm row count, BEIR field shape (`_id`, `text`, `title`, `language`,
`meta_information`), and spot-check five queries against the qrels for sanity. Compare basic
statistics (query length distribution, corpus document length distribution) between train and test.

**Current status:** Open, `Measuring` from Day 1. This blocks nothing on the critical path — the
pipeline can run against synthetic fixtures — but a `Closed` resolution is required before `T-112`
(`OQ-02`'s sweep) can be trusted.

---

### `OQ-06` — Does reverse doc2query expansion earn its index-time cost?

**Question:** [NonGoals.md NG-06](NonGoals.md#ng-06--no-model-fine-tuning-or-training-of-any-kind)
names reverse doc2query document expansion in `axiom.indexing.expansion` as "the one accepted
accuracy intervention that touches the corpus," with an estimated +4 to +8 NDCG@10, but the estimate
is unsourced inside our own docs and the feature is not listed as a `FR-##` in
[PRD.md](PRD.md#5-functional-requirements). Is this in scope, and if so, does it actually deliver
inside the `NFR-01` 12-minute cold-index budget?

**Why it matters:** Doc2query expansion runs a generation pass over every corpus document at index
time — on APPS's 8,765 short Python solutions this is cheap, but the estimate needs a real number
before it can be relied on for the PPT's "Results" slide, and before it is decided whether it needs
its own `FR-##`.

**What would resolve it:** A single ablation row — eval profile with and without expansion, same
weights, same seed — logged in [Tracker.md §Eval metrics log](Tracker.md#eval-metrics-log).

**Current status:** Open. Not on the Must-have path; scheduled for Day 7 (21 Sep) alongside `OQ-02`'s
sweep, using the same tune-split infrastructure. If it misses the window it is deferred, not
abandoned — see [NonGoals.md §4](NonGoals.md#4-deferred-not-rejected).

---

### `OQ-07` — Which multi-version JavaScript repo anchors the P1/Bonus demo?

**Question:** [PRD.md Assumption A-2](PRD.md#10-assumptions) requires "a suitable multi-version
JavaScript repository" for the P1 incremental-reindex and Bonus evolutionary-retrieval demo, with a
declared fallback of synthesising a version history over a chosen OSS repo's real tags. No repo has
been named yet.

**Why it matters:** `data/demo_repo/` is referenced directly by the manual test script
([TestPlan.md §7](TestPlan.md#7-manual-test-script-demo-day), steps M-08 through M-10) and by
`FR-16`–`FR-21`. Everything about the P1/Bonus demo is blocked on this repo existing, because it is
the only corpus where the structural signal and the version registry have anything real to operate
on — the CoIR benchmark corpus explicitly does not, per
[PRD.md §2.2](PRD.md#22-two-evaluation-contexts-two-profiles).

**Candidates under consideration:** a mid-sized OSS voice-assistant-adjacent JS project with a clean
tag history (target: 10 source files, 3+ tagged versions spanning genuine refactors, not just
version bumps); or a synthetic repo hand-authored to exercise every `ChunkKind` and every
`git diff` status code (`A`/`M`/`D`/`R`) deliberately, trading realism for coverage guarantees.

**What would resolve it:** A repo (real or synthetic) checked into `tests/fixtures/` (small) or
staged under `data/demo_repo/` (gitignored, larger), satisfying: ≥3 versions, ≥1 pure rename,
≥1 function edited across ≥2 versions with a real semantic diff (for `SnippetFamily` demo value),
and a call graph deep enough that query archetype Q2 has a genuine multi-hop answer.

**Current status:** Open. Owned by Parth, targeted for Day 1–2 so the structural and versioning
workstreams (Anish, Parth) are not blocked waiting on it past Day 3.

---

### `OQ-08` — Is HyDE worth its latency on any profile?

**Question:** [NonGoals.md NG-19](NonGoals.md#ng-19--no-hyde-in-the-default-path) keeps Hypothetical
Document Embeddings off by default and behind `AXIOM_ENABLE_HYDE`, but frames it as "a measured
ablation row, not a default" rather than a closed door. Is there a profile — even a non-default one
— where HyDE's latency is affordable and its NDCG@10 gain is worth carrying the extra code path?

**Why it matters:** If the answer is a clean no, `AXIOM_ENABLE_HYDE` and its generation pass should
be deleted rather than carried as dead, untested config surface past Day 8.

**What would resolve it:** One ablation run with HyDE forced on against the eval profile's dev
slice, reporting the NDCG@10 delta and the added p50 latency in the same row.

**Current status:** Open, lowest priority (P2) of the accuracy-tuning questions — scheduled only if
`OQ-02` and `OQ-06` land early. If Day 8 arrives and this has not been measured, it resolves to "no"
by default per the cut-order discipline in [PRD.md §7.1](PRD.md#71-cut-order-under-time-pressure).

---

### `OQ-09` — Are the chunking placeholders (64–512 tokens, 16-token floor) right?

**Question:** `_CONTRACT.md §5` locks a 64–512 token chunk target and a 16-token merge floor for the
AST chunker, but [Rules.md §8](Rules.md#8-the-placeholder-convention) explicitly lists these among
"design estimates... honest guesses" chosen from intuition on day 1. Do they hold up against the
actual token-length distribution of real JavaScript functions in the demo repo (`OQ-07`) and against
the reranker's `rerank_max_chars` window (`4096` chars, ≈ 1000–1300 tokens per
[TestPlan.md TC-063](TestPlan.md)), or does a function-heavy real repo produce a token-length
distribution that makes 512 too aggressive a split point?

**Why it matters:** Chunk boundaries are irreversible once indexed — every downstream signal, and
every user-visible `file:line` result, is shaped by this one decision. A bad split point either
fragments a coherent function across multiple weakly-related chunks (hurting dense recall) or lets
oversized chunks dilute the embedding (hurting precision).

**What would resolve it:** A token-length histogram over the chunker's output on the `OQ-07` demo
repo and on a sample of 200 APPS corpus documents, checked against both bounds.

**Current status:** Open. Not gating the P0 critical path — the chunker must ship with *some*
target — but the values carry `# PLACEHOLDER` markers per [Rules.md §8](Rules.md#8-the-placeholder-convention)
until this measurement exists, and per that same rule may not appear in a reported score unmarked.

---

### `OQ-10` — Are the agent sufficiency thresholds (0.35/0.20) right?

**Question:** `FR-13` and `_CONTRACT.md §5` lock the refinement trigger at "top-1 rerank score < 0.35
OR fewer than 3 results above 0.20." [Rules.md §8](Rules.md#8-the-placeholder-convention) uses this
exact pair as its worked example of an unvalidated placeholder, and explicitly assigns it tracker id
`T-141` — see [Tracker.md](Tracker.md#t-141) for the live task.

**Why it matters:** This is the trigger that decides whether the agent loop — the requirement that
is never cut, per [PRD.md §7.1](PRD.md#71-cut-order-under-time-pressure) — actually fires when it
should. A threshold set too low means the loop rarely refines, and "agentic" becomes a label rather
than a behaviour the demo can show; too high means every query burns a second pass and risks the
5-second budget under `NFR-04`.

**What would resolve it:** A sweep on the 300-query dev slice, per the worked example already
recorded in [Rules.md AP-14](Rules.md#ap-14--reporting-a-number-built-on-a-placeholder-8): the
Day-21-September entry there (`0.35 → 0.42`, `NDCG@10 20.1 → 21.3`) is the target *shape* of the
resolution, not yet a real measurement.

**Current status:** Open, `Measuring`, `T-141` in [Tracker.md](Tracker.md), scheduled Day 7
(21 Sep) alongside the rest of the train-split tuning pass (`FR-26`).

---

### `OQ-11` — Are the evolutionary dedupe cosine (0.95) and stability bonus (0.10) right?

**Question:** `_CONTRACT.md §5` locks family-membership cosine at `≥ 0.95` and the ranking bonus at
`final = base * (1 + 0.10 * stability)`. Both are placeholders per
[Rules.md §8](Rules.md#8-the-placeholder-convention). Does 0.95 correctly separate "the same
function, lightly edited" from "a different function that happens to look similar" on the `OQ-07`
demo repo's real version history, and does a 0.10 bonus move the needle on the live demo without
distorting single-version results?

**Why it matters:** Set the cosine threshold too low and unrelated functions collapse into one
family — a correctness bug that is highly visible in a live demo (`TestPlan.md` M-10). Set it too
high and genuinely-evolved snippets that should be one family fragment into several, which quietly
defeats the entire Bonus feature.

**What would resolve it:** Running `build_families()` over the resolved `OQ-07` repo and manually
verifying every family against the git history by eye — this is a qualitative check, not a metric
sweep, because there is no labelled ground truth for "these are the same logical snippet."

**Current status:** Open. Blocked on `OQ-07`. Scheduled Day 8 (22 Sep), the same day P1 and Bonus
land per [ImplementationPlan.md](ImplementationPlan.md).

---

### `OQ-12` — Does the jury need remote access to the demo surfaces?

**Question:** `FR-24`'s API and `FR-25`'s Streamlit UI are dev-only and unauthenticated by design
(`NG-08`, `NG-10`, `NG-13`). Is a fully local, evaluator-runs-it-themselves demo sufficient, or does
the live jury session (per [PRD.md §9](PRD.md#9-jury-scoring-alignment) — actually §8) need a
presenter-hosted instance the jury can reach without cloning the repo?

**Resolution:** **Closed 2026-09-16.** Fully local only. `NG-13` already forecloses hosted
deployment as a risk category, and the jury scoring rubric ([PRD.md §8](PRD.md#8-jury-scoring-alignment))
rewards a working prototype the evaluator can run themselves, not a link. The demo video plus a
live, presenter-run local session (screen-shared or in-room) is the format; no tunnel, no hosted
instance, no exception. No `ADR-###` required — this restates `NG-13` rather than adding new scope.

---

## Changing this document

| Situation | Correct action |
|---|---|
| A new question surfaces mid-sprint | Add the next `OQ-##`, an owner, and a deadline in the same PR that surfaces it. Do not leave owner or deadline blank. |
| A question resolves | Move its Status to `Closed`, write the Resolution paragraph, add the `ADR-###` in [Decisions.md](Decisions.md) if the resolution is a binding decision (most are), and update the Index table row. |
| A deadline passes with no resolution | It does not silently roll over. Restate the deadline explicitly in the entry with a one-line reason, in the daily sync, and as a flagged row in [Tracker.md](Tracker.md). |
| A question turns out to already be settled by `_CONTRACT.md` | Close it immediately citing the contract section; do not leave a live-looking question open when the answer already exists. |
