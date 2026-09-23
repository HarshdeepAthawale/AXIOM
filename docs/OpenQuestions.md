# Open Questions

Unresolved design and product questions for Axiom: tracked with an owner and a deadline, never silently dropped, and promoted to an `ADR-###` in [Decisions.md](Decisions.md) the moment they are settled.

**Owner:** Harshdeep Athawale
**Last updated:** 2026-09-23
**Status:** Draft

Related: [PRD.md](PRD.md) · [NonGoals.md](NonGoals.md) · [Decisions.md](Decisions.md) · [TechSpecifications.md](TechSpecifications.md) · [ImplementationPlan.md](ImplementationPlan.md) · [Tracker.md](Tracker.md) · [Rules.md](Rules.md) · [TestPlan.md](TestPlan.md)

---

## Schedule re-baseline, 2026-09-23

**Every deadline in the previous revision of this document was in the past.** The build calendar it
referenced (Day 1 = 2026-09-15, Day 10 = 24/25 Sep, stated three different ways across three
documents) is **void**. The re-baselined calendar, binding for this document and for
`ImplementationPlan.md` and `Tracker.md`:

| Day | Date |
|---|---|
| Day 1 | **2026-09-23** (today) |
| Day 2 | 2026-09-24 |
| Day 3 | 2026-09-25 |
| Day 4 | 2026-09-26 |
| Day 5 — **submission** | **2026-09-27** |

Five days, not twelve. Six questions were closed in this pass rather than re-dated, because the
audit showed they were already decided in practice or because there is no longer a window in which
to answer them; three new questions (`OQ-13`–`OQ-15`) were opened because
[TechSpecifications.md §8](TechSpecifications.md#8-latency-and-memory-budget-placeholder-unmeasured)'s
`# PLACEHOLDER` figures had no owning question, and `OQ-16` was opened because the accuracy baseline
the whole project is calibrated against does not exist yet. A `# PLACEHOLDER` with no owning `OQ` is
a defect per [Rules.md §8](Rules.md#8-the-placeholder-convention), and there were fourteen of them.

**A question that cannot be answered in five days is not re-dated — it is closed with the
conservative answer.** That is a decision, not a failure, and it is recorded as one.

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
| [`OQ-01`](#oq-01--is-the-two-profile-eval--demo-split-the-final-shape) | Is the two-profile eval / demo split the final shape? | Harshdeep | — | **Closed** 2026-09-16 (ADR-001) | P0 |
| [`OQ-02`](#oq-02--what-is-the-exact-sparse-weight-in-configsevalyaml) | What is the exact sparse weight in `configs/eval.yaml`? | Prabinder | 2026-09-26 (Day 4) | Open | P0 |
| [`OQ-03`](#oq-03--is-qwen3-embedding-06b-the-right-embedder-for-apps) | Is Qwen3-Embedding-0.6B the right embedder for APPS? | Prabinder | — | **Closed** 2026-09-23 — superseded by `OQ-14` and `OQ-16` | P0 |
| [`OQ-04`](#oq-04--project-name-collision-with-a-rival-submission) | Project name collision with a rival submission | Harshdeep | — | **Closed** 2026-09-16 (ADR-015) | P0 |
| [`OQ-05`](#oq-05--is-the-5000-row-train-split-usable-for-tuning-as-assumed) | Is the 5,000-row train split usable for tuning as assumed? | Parth | 2026-09-24 (Day 2) | Open | P0 |
| [`OQ-06`](#oq-06--does-reverse-doc2query-expansion-earn-its-index-time-cost) | Does reverse doc2query expansion earn its index-time cost? | Prabinder | — | **Closed** 2026-09-23 — no (ADR-013 withdrawn) | P1 |
| [`OQ-07`](#oq-07--which-multi-version-javascript-repo-anchors-the-p1bonus-demo) | Which multi-version JavaScript repo anchors the P1/Bonus demo? | Parth | 2026-09-23 (Day 1) | Open — **scope now fixed**, only the repo identity is open | P0 |
| [`OQ-08`](#oq-08--is-hyde-worth-its-latency-on-any-profile) | Is HyDE worth its latency on any profile? | Harshdeep | — | **Closed** 2026-09-23 — no, this cycle | P2 |
| [`OQ-09`](#oq-09--are-the-chunking-placeholders-64512-tokens-16-token-floor-right) | Are the chunking placeholders (64–512 tokens, 16-token floor) right? | Anish | 2026-09-25 (Day 3) | Open | P1 |
| [`OQ-10`](#oq-10--are-the-agent-sufficiency-thresholds-035020-right) | Are the agent sufficiency thresholds (0.35/0.20) right? | Harshdeep | 2026-09-26 (Day 4) | Open | P0 |
| [`OQ-11`](#oq-11--are-the-evolutionary-dedupe-cosine-095-and-stability-bonus-010-right) | Are the evolutionary dedupe cosine (0.95) and stability bonus (0.10) right? | Parth | 2026-09-26 (Day 4) | Open — blocked on `OQ-07` | P1 |
| [`OQ-12`](#oq-12--does-the-jury-need-remote-access-to-the-demo-surfaces) | Does the jury need remote access to the demo surfaces? | Harshdeep | — | **Closed** 2026-09-16 | P1 |
| [`OQ-13`](#oq-13--what-is-the-real-reranker-throughput-on-the-reference-box) | What is the real reranker throughput on the reference box? | Prabinder | 2026-09-23 (Day 1) | **Open, new** | **P0** |
| [`OQ-14`](#oq-14--what-is-the-real-embedder-throughput-and-cold-index-time) | What is the real embedder throughput and cold-index time? | Prabinder | 2026-09-23 (Day 1) | **Open, new** | **P0** |
| [`OQ-15`](#oq-15--what-is-the-real-peak-rss-during-query-serving) | What is the real peak RSS during query serving? | Prabinder | 2026-09-24 (Day 2) | **Open, new** | P1 |
| [`OQ-16`](#oq-16--what-is-our-own-dense-only-ndcg10-baseline-on-appsretrieval-test) | What is our own dense-only NDCG@10 baseline on `AppsRetrieval` test? | Parth | 2026-09-23 (Day 1) | **Open, new** | **P0 — the single highest-priority question in the project** |

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

**Question:** Is `{dense 0.85, sparse 0.15, struct 0.0}` (the placeholder quoted in
[NonGoals.md NG-04](NonGoals.md#ng-04--no-language-support-beyond-javascript-demo-and-python-benchmark))
the measured optimum, or a guess that happens to appear in two other documents because it was typed
once and copied?

**Changed 2026-09-23.** The justification for down-weighting sparse used to be numeric —
"BM25 = 4.8 against BGE-0.6B = 14.7 on APPS." Both figures are retracted
([PRD.md §2.0.3](PRD.md#203-external-comparison-table--quarantined-pending-per-row-citation)): they
could not be located in either cited source. The *mechanism* survives — English prose queries against
Python solution documents share very little surface vocabulary, so a lexical signal is weak here and
a weak list fused at equal weight costs the reranker candidate slots. The *magnitude* does not.
`0.15` is now an unjustified placeholder rather than a poorly-sourced one, **and the sweep's range
must include `0.0`** — "drop sparse entirely on the eval profile" is a live possibility that the
retracted numbers made look unnecessary to test.

**Why it matters:** This is the single highest-leverage knob for the P0 screening score. A wrong
guess costs NDCG@10 points that no amount of reranker tuning recovers, because a bad first-stage
fusion never puts the right chunk inside the top-25 the reranker sees.

**What would resolve it:** A sweep over `{0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}` sparse weight on
the **4,000-row tune split** (`FR-26`), holding structural at 0.0, reporting NDCG@10 for each point.
Never on test ([NG-29](NonGoals.md#ng-29--no-tuning-on-the-benchmark-test-split)). Owned as `T-112`
in [Tracker.md](Tracker.md).

**Current status:** Open. Deadline **2026-09-26 (Day 4)**, `Measuring` once the dense+sparse+rerank
pipeline is green and `OQ-16`'s baseline exists. `0.15` ships as the `# PLACEHOLDER` default in
`configs/eval.yaml` until then. **If Day 4 arrives unswept, this resolves to "keep 0.15, declare it
unswept"** — rung 3 of [PRD.md §10.1](PRD.md#101-a-6s-fallback-rewritten-so-it-cannot-reach-the-test-split).
It does not roll over into the submission window.

---

### `OQ-03` — Is Qwen3-Embedding-0.6B the right embedder for APPS?

**Question:** Does Qwen3-Embedding-0.6B, a general-purpose multilingual embedder, align better with
English-prose-to-Python-solution retrieval than a code-pretrained alternative of similar size?

**Resolution: Closed 2026-09-23 — superseded, not answered.** Closing it rather than re-dating it,
because the question as posed can no longer change what ships and two sharper questions have replaced
it.

- **The comparative half is moot.** The entire stack — ONNX export, INT8 quantisation, pooling,
  `VersionManifest.embedding_dim`, 400 passing tests — is built against Qwen3-Embedding-0.6B. With
  four days left, swapping the primary embedder on the strength of a dev-slice comparison is not an
  action anyone would take. `all-MiniLM-L6-v2` remains the declared fallback (`NFR-07`) and the
  candidate for the *demo* index specifically, on throughput grounds, not accuracy grounds.
- **The throughput half is now [`OQ-14`](#oq-14--what-is-the-real-embedder-throughput-and-cold-index-time)**,
  which is a Day-1 measurement rather than a model comparison.
- **The accuracy half is now [`OQ-16`](#oq-16--what-is-our-own-dense-only-ndcg10-baseline-on-appsretrieval-test)**:
  what matters is not "is this the best 0.6B embedder" but "what does *our* dense-only pipeline
  actually score," which is the baseline every gate is now derived from.

**Note on this entry's own sourcing.** The previous text argued from "UniXcoder 0.1B scores 1.4,
below BM25's 4.8, therefore NL-to-code alignment beats code-pretraining." Both figures are retracted
([PRD.md §2.0.3](PRD.md#203-external-comparison-table--quarantined-pending-per-row-citation)). The
underlying intuition may well be right; it is no longer evidence.

**No `ADR-###`.** This closes as superseded, not as a decision.

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

**Residue, and the line that caused it — corrected 2026-09-23.**

This entry previously ended with the sentence: *"Until `T-201` lands, prefer whichever name the file
you are editing already uses."* **That sentence is deleted.** It was written as pragmatic advice and
it functioned as a standing instruction to widen the divergence with every edit: by the time the
audit ran, the environment prefix stood at 249 `AXIOM_` occurrences against 0 `PRISM_`-prefixed
settings while the binding contract still specified `PRISM_`, and the CLI entrypoint was `prism` in
`_CONTRACT.md §1` and in `FR-23` while roughly forty invocations across the suite typed `axiom`. A
demo-day test case (`TestPlan.md` M-07) was scripted to invoke `prism query` in front of the jury.
That is what "prefer whichever name the file already uses" produces, and it is worth recording as a
lesson rather than quietly removing: **a naming inconsistency has no correct local resolution, only
a global one.**

**The rule now, with no exceptions and no transition period:** the name is **Axiom**. Package and
import root `axiom`, CLI entrypoint `axiom`, env prefix `AXIOM_`, index root `.axiom/`, exception
base `AxiomError`, schema base `AxiomModel`. Five literals keep the word PRISM — the organiser-prescribed
release tag `PRISM_GENAI_HACKATHON_Y2026`, the event name "Samsung PRISM GenAI Hackathon", the
rival's repository URL, **our own repository URL** (`Samsung-Prism-Hack`) and
**`sparse.bm25s/prism_meta.json`** on disk. The complete, authoritative list with the reasoning for
each is [`_CONTRACT.md §0`](_CONTRACT.md#0-identity); earlier revisions of this
answer said "three" and missed the last two.

`_CONTRACT.md §0/§1/§3` carried `prism` / `PRISM_` until 2026-09-23 and **has since been corrected**
to match the other twenty-one docs and all of `src/`, notwithstanding its own "if a doc contradicts
this file, the doc is wrong" clause — that clause presumes the contract is current, and on this axis
it was eight days out of date. See
[ADR-015](Decisions.md#adr-015--rename-prism-to-axiom) for the binding table.

The error taxonomy has been reconciled against the code, not against prose: `src/axiom/core/errors.py`
defines exactly `AxiomError`, `AxiomContractError`, `IndexNotFoundError`, `DegradationExhaustedError`
— four classes, all `Axiom`-prefixed or plain. Documents referring to `PrismConfigError`,
`PrismModelError`, `PrismParseError` or `PrismBudgetError` were naming classes that do not exist in
any revision of the code; `Rules.md §9.2`'s seven-class taxonomy over-enumerates and is flagged for
its owner.

---

### `OQ-05` — Is the 5,000-row train split usable for tuning as assumed?

**Question:** [PRD.md Assumption A-6](PRD.md#10-assumptions) and
[NonGoals.md NG-29](NonGoals.md#ng-29--no-tuning-on-the-benchmark-test-split) both assume the
`AppsRetrieval` train split is reachable, BEIR-shaped, and large enough to support a 4,000/1,000
tune/dev split without the two halves being pathologically different in difficulty.

**Why it matters:** `FR-26` — the entire tuning methodology that keeps us honest about `NG-29` — used
to have, as its only documented fallback, Assumption A-6's "carve a 500-query holdout from the dev
portion of **test**." That was a written instruction to violate `NG-29` under exactly the conditions
(late, behind, tired) in which it would have been followed. **A-6 has been rewritten**
([PRD.md §10.1](PRD.md#101-a-6s-fallback-rewritten-so-it-cannot-reach-the-test-split)) with a
three-rung ladder — demo-corpus query set, synthetic query set from corpus documents only, or ship
the constants untuned — **none of which can reach the test split.** This question therefore no
longer gates the integrity of the claim; it only decides which rung we stand on.

If the train split turns out to be a different shape than the test split (shorter problem
statements, a narrower difficulty band), weights tuned on it may not transfer, and we would not know
until the one permitted test-split run. That is a transfer risk we accept and state, not a licence
to peek.

**What would resolve it:** Load the train split on Day 1–2 (`T-003` in
[Tracker.md](Tracker.md)), confirm row count, BEIR field shape (`_id`, `text`, `title`, `language`,
`meta_information`), and spot-check five queries against the qrels for sanity. Compare basic
statistics (query length distribution, corpus document length distribution) between train and test.

**Current status:** Open. Deadline **2026-09-24 (Day 2)**. This blocks nothing on the critical path
— the pipeline can run against synthetic fixtures, and `OQ-16`'s dense-only baseline does not need
the train split at all — but a `Closed` resolution is required before `T-112` (`OQ-02`'s sweep) can
be trusted. **If it resolves "no", go to rung 1 of PRD §10.1.** Not to the test split.

---

### `OQ-06` — Does reverse doc2query expansion earn its index-time cost?

**Question:** [NonGoals.md NG-06](NonGoals.md#ng-06--no-model-fine-tuning-or-training-of-any-kind)
named reverse doc2query document expansion in `axiom.indexing.expansion` as "the one accepted
accuracy intervention that touches the corpus," with an estimated +4 to +8 NDCG@10. Is it in scope,
and does it deliver inside the `NFR-01` cold-index budget?

**Resolution: Closed 2026-09-23 — no, not this cycle.**
[`ADR-013`](Decisions.md#adr-013--reverse-doc2query-expansion-as-the-one-corpus-touching-intervention)
is **withdrawn**. Three independent reasons, any one sufficient:

1. **It contradicted the contract and the contradiction was never resolved.**
   [`ADR-008`](Decisions.md#adr-008--the-query-llm-never-reads-code), `_CONTRACT.md §2` and
   [`NG-23`](NonGoals.md#ng-23--no-llm-ingestion-of-retrieved-code) all state the query LLM never
   receives chunk text. Reverse doc2query *is* the LLM reading chunk text — that is its mechanism —
   and the proposal reused the same query LLM, so it was not a differently-governed component. The
   fix would have been to amend `ADR-008` to "never **at query time**" and carve a bounded
   index-time exception. That amendment was never made, and "the LLM never reads code" is the
   project's central innovation claim and the sentence that earns the 15% theme-relevance criterion.
   A carve-out the jury has to be walked through is worth less than the claim it weakens.
2. **The +4 to +8 NDCG@10 was fabricated.** Unsourced — it appears in no paper cited in this suite
   and in no run of ours — and it was roughly a quarter of the stated path to the accuracy target.
   A gate calibrated against it was calibrated against nothing. Retracted from `NonGoals.md NG-06`.
3. **It does not exist and cannot be built in four days.** There is no
   `src/axiom/indexing/expansion.py`. It adds one generation pass per corpus document to a cold
   index already projected to miss `NFR-01` by 1.7–3.3x unaided
   ([TechSpec §8.5](TechSpecifications.md#85-cold-index-placeholder-oq-14)); on 8,765 APPS
   documents that is hours, never priced into `NFR-01`.

**Consequence:** `NG-06` is absolute for this cycle — nothing writes to the corpus, by gradient or
by generation. To reinstate later, do it in this order: amend `ADR-008` and `_CONTRACT.md §2`
explicitly *first*, then produce a real ablation row to replace the withdrawn estimate.

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

**Scope fixed by the project owner, 2026-09-23 — only the repo's identity is still open.** The demo
corpus is a **small, real, tagged JavaScript repository: ≈10–50 source files, ≥3 tags**. The "real
10k-file JS repo" that [PRD.md §8](PRD.md#8-jury-scoring-alignment) previously promised the jury is
**deleted**: at Setup.md's own ~6.9 chunks/file it is ~69,000 chunks, which crosses the 50k
`IndexFlatIP` → `IndexIVFPQ` threshold that
[NG-18](NonGoals.md#ng-18--no-ann-index-on-the-benchmark-corpus) rules out,
and indexes in hours against a 12-minute requirement. See
[PRD.md §2.2](PRD.md#22-two-evaluation-contexts-two-profiles) for the arithmetic.

**Consequence already absorbed elsewhere:** a 10–50 file repo cannot produce `NFR-02`'s
50-changed-file diff, so that number is measured on the **synthetic diff generator**
(`TestPlan.md §5.4`) and reported separately from the live demo, labelled as such.

**Candidates under consideration:** a small OSS voice-assistant-adjacent JS project with a clean tag
history (3+ tagged versions spanning genuine refactors, not just version bumps); or a repo
hand-authored to exercise every `ChunkKind` and every `git diff` status code (`A`/`M`/`D`/`R`)
deliberately, trading realism for coverage guarantees. Real is preferred — a jury can tell — but
**availability today beats realism**: this is the Day-1 blocker for two workstreams.

**What would resolve it:** A repo (real or synthetic) checked into `tests/fixtures/` (small) or
staged under `data/demo_repo/` (gitignored, larger), satisfying: ≥3 versions, ≥1 pure rename,
≥1 function edited across ≥2 versions with a real semantic diff (for `SnippetFamily` demo value),
and a call graph deep enough that query archetype Q2 has a genuine multi-hop answer.

**Current status:** Open, scope fixed. Deadline **2026-09-23 (Day 1)** — with four days left there
is no version of this that can slip. If no real repo is chosen by end of Day 1, the hand-authored
fixture ships and the decision is recorded in [Tracker.md](Tracker.md); `OQ-11` is blocked behind
this and cannot start until it lands.

---

### `OQ-08` — Is HyDE worth its latency on any profile?

**Question:** [NonGoals.md NG-19](NonGoals.md#ng-19--no-hyde-in-the-default-path) keeps Hypothetical
Document Embeddings off by default and behind `AXIOM_ENABLE_HYDE`. Is there a profile — even a
non-default one — where HyDE's latency is affordable and its NDCG@10 gain is worth the extra code
path?

**Resolution: Closed 2026-09-23 — no, this cycle.** Two reasons, one of them a correction.

1. **The arithmetic that previously rejected it was fabricated, and so was the arithmetic that kept
   it alive.** `NG-19` rejected HyDE by comparing "hundreds of milliseconds" of generation against
   "132 ms of headroom in the 768 ms p50 budget." There is no 768 ms p50 and there is no 132 ms of
   headroom — both are retracted
   ([TechSpec §8](TechSpecifications.md#8-latency-and-memory-budget-placeholder-unmeasured)). A
   false number was driving a real scope decision in both directions: it made the rejection look
   quantitative, and it made "measure it later and maybe promote it" look reachable.
2. **The exclusion stands on its own merits, stated without numbers.** HyDE puts a generation pass
   at the *head* of the query path in a project whose entire thesis is that the LLM never generates
   on the retrieval path. It is unimplemented, untested, and there are four calendar days in which
   the dense-only baseline has not yet been run once. There is no profile on which it earns a slot
   ahead of `OQ-13`, `OQ-14` or `OQ-16`.

**Consequence:** `AXIOM_ENABLE_HYDE` is dead config. Per this entry's own previous wording — "if the
answer is a clean no, it should be deleted rather than carried as dead, untested config surface" —
**delete the flag and its code path**, or leave it documented as permanently off and untested, but
do not carry it as a deferred measurement that will not happen. `NG-19` updated accordingly.

**No `ADR-###`.** This restates `NG-19` on honest grounds rather than adding scope.

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

**Current status:** Open. Deadline **2026-09-25 (Day 3)**. Not gating the P0 critical path — the
chunker ships with *some* target and already does — but the values carry `# PLACEHOLDER` markers per
[Rules.md §8](Rules.md#8-the-placeholder-convention) until this measurement exists, and per that
same rule may not appear in a reported score unmarked. **There is one sequencing interaction with
`OQ-14`:** if the cold-index measurement comes back at the projected 20–40 min, the lever most
likely to be pulled is dropping the embedder's max-token bound from 512 toward 192–256, which moves
the upper chunk bound with it. Do not close `OQ-09` before `OQ-14` reports, or it will be reopened
the same day.

---

### `OQ-10` — Are the agent sufficiency thresholds (0.35/0.20) right?

**Question:** `FR-13` and `_CONTRACT.md §5` lock the refinement trigger at "top-1 rerank score < 0.35
OR fewer than 3 results above 0.20." [Rules.md §8](Rules.md#8-the-placeholder-convention) uses this
exact pair as its worked example of an unvalidated placeholder, and explicitly assigns it tracker id
`T-141` — see [Tracker.md](Tracker.md#24-agent-rerank-api-ui-t-091t-150) for the live task.

**Why it matters:** This is the trigger that decides whether the agent loop — the requirement that
is never cut, per [PRD.md §7.1](PRD.md#71-cut-order-under-time-pressure) — actually fires when it
should. A threshold set too low means the loop rarely refines, and "agentic" becomes a label rather
than a behaviour the demo can show; too high means every query burns a second pass and risks the
5-second budget under `NFR-04`.

**What would resolve it:** A sweep on the **1,000-row dev split** defined by `FR-26` — not a
"300-query dev slice". That 300-query slice appears in four documents and is defined in none of
them; `FR-26` commits 4,000 tune / 1,000 dev and defines no 300-row subset. Either use the 1,000, or
define a seeded, committed 300-id subset of it under `data/splits/` and say so in `FR-26`. Use the
1,000.

[Rules.md AP-14](Rules.md#ap-14--reporting-a-number-built-on-a-placeholder-8)'s worked entry
(`0.35 → 0.42`, `NDCG@10 20.1 → 21.3`) is the *shape* a resolution takes, not a measurement — and
note its illustrative NDCG values are on the retired absolute scale (§2.0 of PRD now derives targets
as relative gains over our own baseline), so do not read them as targets.

**Current status:** Open. Deadline **2026-09-26 (Day 4)**, `T-141` in [Tracker.md](Tracker.md),
alongside the rest of the train-split tuning pass (`FR-26`). **If Day 4 arrives unswept it resolves
to "keep 0.35/0.20, declare them unswept"** — rung 3 of
[PRD.md §10.1](PRD.md#101-a-6s-fallback-rewritten-so-it-cannot-reach-the-test-split). The agent loop
still demonstrably fires either way; what is lost is the claim that the trigger is tuned.

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

**Current status:** Open. Blocked on `OQ-07`. Deadline **2026-09-26 (Day 4)**, the same day P1 and
Bonus land. This is a qualitative eyeball check against git history, so it costs an hour once the
repo exists — but it cannot start before then, which is why `OQ-07` is a Day-1 blocker. If the
Bonus feature is cut under [PRD.md §7.1](PRD.md#71-cut-order-under-time-pressure), this closes as
"not applicable, feature cut".

---

### `OQ-12` — Does the jury need remote access to the demo surfaces?

**Question:** `FR-24`'s API and `FR-25`'s Streamlit UI are dev-only and unauthenticated by design
(`NG-08`, `NG-10`, `NG-13`). Is a fully local, evaluator-runs-it-themselves demo sufficient, or does
the live jury session (per [PRD.md §8](PRD.md#8-jury-scoring-alignment) — actually §8) need a
presenter-hosted instance the jury can reach without cloning the repo?

**Resolution:** **Closed 2026-09-16.** Fully local only. `NG-13` already forecloses hosted
deployment as a risk category, and the jury scoring rubric ([PRD.md §8](PRD.md#8-jury-scoring-alignment))
rewards a working prototype the evaluator can run themselves, not a link. The demo video plus a
live, presenter-run local session (screen-shared or in-room) is the format; no tunnel, no hosted
instance, no exception. No `ADR-###` required — this restates `NG-13` rather than adding new scope.

### `OQ-13` — What is the real reranker throughput on the reference box?

**Question:** How long does one rerank batch actually take? Specifically: 25 pairs at 4096
`rerank_max_chars` through `cross-encoder/ms-marco-MiniLM-L-6-v2` (the `demo`/`default` primary),
and 5 pairs at 1024 chars through `BAAI/bge-reranker-v2-m3` (the `eval` primary), both INT8 ONNX on
the 8-core reference box.

**Why it matters:** The reranker is the dominant query-time cost under every assumption, so this one
number decides whether `NFR-03` (≤ 900 ms p50) and `NFR-04` (≤ 5 s p95) are met, missed, or need
restating. It is also the number the previous revision of the docs got wrong in the most damaging
possible way: **620 ms was recorded against `bge-reranker-v2-m3` when it was `ms-marco-MiniLM`'s
figure**, off by roughly 25x. That single transcription propagated into a "768 ms measured p50" in
four documents, a "132 ms headroom" argument that was used to reject HyDE in `NG-19`, and a table
that would have been on a slide. The arithmetic showing why 620 ms is impossible for bge — 8,375
GFLOP needing 13.5 TOPS against a ~5.1 TOPS theoretical peak — is in
[TechSpecifications.md §8.1](TechSpecifications.md#81-why-the-reranker-line-was-wrong-the-arithmetic).

**What would resolve it:** `scripts/bench_latency.py --phase rerank` against the real exported INT8
artifacts, 20 warm iterations after 3 discarded, reporting p50 and p95 per configuration. Half an
hour of work. **Do it before writing anything else on Day 1 except `OQ-16`.**

**Current status:** **Open, new.** Deadline **2026-09-23 (Day 1)**, owner Prabinder. Owns the
`# PLACEHOLDER` markers on `TechSpec §8.1`, `§8.2`, `§8.3`, `NFR-03`, `NFR-04`, every latency row in
[Appflow.md](Appflow.md), and the profile-split decision in
[TechSpec §3.2](TechSpecifications.md#32-cross-encoder-reranker). Until it closes, **no latency
number may appear in the deck, the README, or a reported result** — [Rules.md §8](Rules.md#8-the-placeholder-convention).

---

### `OQ-14` — What is the real embedder throughput and cold-index time?

**Question:** How many chunks per second does INT8 `Qwen3-Embedding-0.6B` actually encode on the
reference box, and what does that make the cold index for 10,000 chunks?

**Why it matters:** `NFR-01` requires ≤ 12 min and the arithmetic says that needs ~3.24 TOPS
sustained against a ~5.1 TOPS *theoretical* peak — about 63% of theory, sustained for twelve minutes,
from an ONNX Runtime dynamic-INT8 path that typically realises a fifth to two-fifths of peak. At that
efficiency the honest projection is roughly **20 to 40 minutes**, i.e. 1.7–3.3x over budget
([TechSpec §8.5](TechSpecifications.md#85-cold-index-placeholder-oq-14)). `NonGoals.md NG-07`'s
"cold index is **measured** at 636 s" is retracted — no such run exists, and that fabricated figure
was load-bearing for the "no distributed indexing" non-goal and for `PRD` Assumption A-4.

The practical stake is throughput of *attempts*: a cold index is the inner loop of the whole project,
and the difference between a 12-minute index and a 40-minute one is the difference between roughly
ten end-to-end attempts in the remaining window and three.

**What would resolve it:** Export one INT8 artifact and time **500 real chunks**, then extrapolate.
Not a full index — 500 chunks answers it in minutes. If the measurement lands where the arithmetic
predicts, the levers in priority order are (a) drop `AXIOM_EMBEDDING_MAX_TOKENS` from 512 toward
192–256 and re-centre the chunk band (interacts with `OQ-09`), (b) use `all-MiniLM-L6-v2` (~40x
cheaper per token) for the **demo** index while keeping Qwen3 for the one-off 8,765-document APPS
encode. Note the APPS encode is the *smaller* job, so `NFR-01` is on the critical path to the live
demo but not to the reportable number.

**Current status:** **Open, new.** Deadline **2026-09-23 (Day 1)**, owner Prabinder. Owns the
`# PLACEHOLDER` on `NFR-01`, `NFR-02`, `PRD` Assumption A-4, `TechSpec §8.5`, `NonGoals NG-07` and
Appflow Flow 1's budget. Supersedes the throughput half of the closed `OQ-03`.

---

### `OQ-15` — What is the real peak RSS during query serving?

**Question:** What is the actual peak resident set size with a query in flight, per profile?

**Why it matters:** `NFR-05` claims ≤ 4 GB and **no memory accounting existed anywhere in 10,747
lines of documentation** — the number was asserted, never derived. The first component tally
([TechSpec §8.6](TechSpecifications.md#86-peak-rss-placeholder-oq-15)) lands at 2.7–3.5 GB on the
`demo` profile and 3.3–4.1 GB on `eval`, i.e. *at or over* the stated ceiling, with the ONNX Runtime
arena allocators as the least certain row. A 4 GB claim that OOMs on the evaluator's 8 GB laptop
mid-demo costs the 30% prototype criterion outright.

**What would resolve it:** `psutil` RSS sampled at 100 ms through one query under each profile, LLM
on and off, reported as peak. Twenty minutes once a query runs end to end.

**What to do with the answer:** if it exceeds 4 GB, the first lever is **not holding all three models
resident** — the LLM is needed only before retrieval and the reranker only after fusion, so a bounded
lazy-load-and-release policy keeps peak near 2.5 GB. If they must stay resident for latency, **raise
`NFR-05` to 6 GB and say why.** A stated 6 GB is defensible; a 4 GB claim that fails is not.

**Current status:** **Open, new.** Deadline **2026-09-24 (Day 2)**, owner Prabinder. Owns the
`# PLACEHOLDER` on `NFR-05` and `TechSpec §8.6`.

---

### `OQ-16` — What is our own dense-only NDCG@10 baseline on `AppsRetrieval` test?

**Question:** What does a bare dense-only pipeline — no sparse, no structural, no reranker, no agent
loop — actually score on the CoIR `AppsRetrieval` **test** split, on our chunking, our pooling and
our corpus preparation?

**Why it matters — this is the highest-priority question in the project.** Every accuracy gate in
`PRD.md §2` is now expressed as a multiple of this number (`≥ 1.14 x B`, `≥ 1.22 x B`, `≥ 1.36 x B`),
and every ablation row is a delta against it. It exists because the previous baseline did not: the
suite anchored its entire ladder to a row reading **"BGE 0.6B = 14.7"**, sourced to arXiv:2407.02883
and arXiv:2506.16552, which an adversarial audit could not locate in either paper. We have not
re-verified the papers ourselves, which is precisely the problem — the number gates three milestones
and a slide, and a jury member can open the cited link during the presentation.

`B` is strictly better than any published figure for our purposes: it is measured on the system we
actually built, it cannot be falsified by opening a paper, and the evaluator can reproduce it with
one command.

**What would resolve it:** `axiom eval --task AppsRetrieval --split test` with `dense_enabled: true`
and every other signal off, under `configs/eval.yaml`. Record `ndcg_at_10`, `mrr_at_10`,
`recall_at_100`, the config hash and the seed into `appsretrieval_results.json` and
[Tracker.md](Tracker.md)'s eval metrics log. Owned as `T-200a`.

**This is a test-split run, and it is permitted.** `NG-29` forbids *tuning* on test, not reporting
on it. `B` is a reported measurement of a fixed configuration, made once, logged, and never used to
select a parameter. It is the baseline arm of the ablation, not a tuning signal.

**Sequencing — this runs first.** Before sparse, before structural, before the reranker, before the
agent loop, before the UI, before the API, before Docker. Nothing else in `PRD.md §2` has a value
until it does, and if the remaining window produces exactly one artefact, this is the one that has
to exist.

**One risk worth naming.** Qwen3-Embedding-0.6B requires **last-token pooling with left padding** and
an instruction prefix on the query side. A right-padded batched call returns the PAD token's hidden
state for most of the batch and yields well-formed, correctly-dimensioned, near-useless vectors that
pass every shape assertion in the codebase. If `B` comes back implausibly low, check pooling and
padding before concluding the model is weak on APPS.

**Current status:** **Open, new.** Deadline **2026-09-23 (Day 1)**, owner Parth. Owns the
`# PLACEHOLDER` on `PRD §2.0`, `§2.0.1`, `§2.0.3`, and every `M2`/`M3`/`M5` gate threshold in
[ImplementationPlan.md](ImplementationPlan.md).

---

---

## Changing this document

| Situation | Correct action |
|---|---|
| A new question surfaces mid-sprint | Add the next `OQ-##`, an owner, and a deadline in the same PR that surfaces it. Do not leave owner or deadline blank. |
| A question resolves | Move its Status to `Closed`, write the Resolution paragraph, add the `ADR-###` in [Decisions.md](Decisions.md) if the resolution is a binding decision (most are), and update the Index table row. |
| A deadline passes with no resolution | It does not silently roll over, **and inside a five-day window it does not get re-dated either.** Every open question above carries an explicit default resolution — the conservative answer it takes if its deadline arrives unanswered. Record that resolution, say it is a default rather than a measurement, and move on. The 2026-09-23 re-baseline exists because the previous revision let twelve deadlines pass into the past without either. |
| A question turns out to already be settled by `_CONTRACT.md` | Close it immediately citing the contract section; do not leave a live-looking question open when the answer already exists. |
