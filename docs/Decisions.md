# Decisions

`ADR-###` log for Axiom: what was chosen, why, what alternatives were rejected, and what it would take to reverse the decision.

**Owner:** Prabinder Singh
**Last updated:** 2026-09-23
**Status:** Draft

Related: [PRD.md](PRD.md) · [NonGoals.md](NonGoals.md) · [OpenQuestions.md](OpenQuestions.md) · [TechSpecifications.md](TechSpecifications.md) · [Design.md](Design.md) · [Schema.md](Schema.md) · [Rules.md](Rules.md) · [ImplementationPlan.md](ImplementationPlan.md) · [Changelog.md](Changelog.md)

---

## How to read this document

An Architecture Decision Record here answers one question: **why is it this way and not some other
way a reasonable engineer would have tried first?** Every ADR states the alternatives that were on
the table and why they lost — a decision log that only records the winner teaches nobody anything.

Format, one section per decision:

| Field | Meaning |
|---|---|
| Status | `Proposed` → `Accepted` → `Superseded` (never silently deleted) |
| Context | The problem, in the terms it actually presented itself |
| Decision | What we chose, stated as an imperative |
| Alternatives considered | What else was viable, and the concrete reason each lost |
| Consequences | What this decision costs us, not just what it buys |
| Reversal cost | What it would take to change our mind later |

Cardinal rule changes ([Rules.md §12](Rules.md#12-rule-change-procedure)) and any schema change
([Rules.md §10](Rules.md#10-code-review-rules) item 7) require an ADR by process, not just by
convention. Numbers are permanent; a superseded ADR keeps its number with a pointer forward.

---

## Index

| ID | Decision | Status | Date |
|---|---|---|---|
| [`ADR-001`](#adr-001--two-first-class-evaluation-profiles) | Two first-class evaluation profiles, not one with a toggle | Accepted | 2026-09-16 |
| [`ADR-002`](#adr-002--weighted-reciprocal-rank-fusion-over-score-space-fusion) | Weighted Reciprocal Rank Fusion over score-space fusion | Accepted | 2026-09-15 |
| [`ADR-003`](#adr-003--bm25s-over-rank-bm25) | `bm25s` over `rank-bm25` | Accepted | 2026-09-15 |
| [`ADR-004`](#adr-004--faiss-flat-below-50k-vectors-ivf-pq-at-or-above) | FAISS Flat below 50k vectors, IVF-PQ at or above | Accepted | 2026-09-15 |
| [`ADR-005`](#adr-005--blake2b-128-for-every-content-address) | blake2b-128 for every content address | Accepted | 2026-09-15 |
| [`ADR-006`](#adr-006--two-disjoint-id-spaces-chunk_id-and-content_hash) | Two disjoint id spaces: `chunk_id` and `content_hash` | Accepted | 2026-09-15 |
| [`ADR-007`](#adr-007--bounded-agent-loop-hard-caps-not-convergence) | Bounded agent loop: hard caps, not convergence | Accepted | 2026-09-15 |
| [`ADR-008`](#adr-008--the-query-llm-never-reads-code) | The query LLM never reads code | Accepted | 2026-09-15 |
| [`ADR-009`](#adr-009--supervised-configuration-tuning-instead-of-model-fine-tuning) | Supervised configuration tuning instead of model fine-tuning | Accepted | 2026-09-15 |
| [`ADR-010`](#adr-010--git-diff-as-the-sole-versioning-source-of-truth) | `git diff` as the sole versioning source of truth | Accepted | 2026-09-15 |
| [`ADR-011`](#adr-011--sqlite-for-the-structural-index-not-a-graph-database) | SQLite for the structural index, not a graph database | Accepted | 2026-09-15 |
| [`ADR-012`](#adr-012--onnx-runtime-int8--llamacpp-gguf-for-a-cpu-only-stack) | ONNX Runtime INT8 + llama.cpp GGUF for a CPU-only stack | Accepted | 2026-09-15 |
| [`ADR-013`](#adr-013--reverse-doc2query-expansion-as-the-one-corpus-touching-intervention) | Reverse doc2query expansion as the one corpus-touching intervention | Withdrawn | 2026-09-23 |
| [`ADR-014`](#adr-014--frozen-extraforbid-pydantic-models-as-the-schema-base) | Frozen, `extra="forbid"` Pydantic models as the schema base | Accepted | 2026-09-15 |
| [`ADR-015`](#adr-015--rename-prism-to-axiom) | Rename PRISM to Axiom | Accepted | 2026-09-16 |

---

## `ADR-001` — Two first-class evaluation profiles

**Status:** Accepted, 2026-09-16. Resolves [`OQ-01`](OpenQuestions.md#oq-01--is-the-two-profile-eval--demo-split-the-final-shape).

**Context:** Axiom is scored against two corpora that share nothing but the word "code." The CoIR
`AppsRetrieval` screening benchmark is English problem statements retrieving standalone Python
solutions with no cross-file structure. The live demo corpus is a multi-module JavaScript repo with
real imports, exports, and a call graph. A structural AST/call-graph signal is the headline
innovation claim on the demo and contributes approximately nothing on APPS — see
[PRD.md §2.2](PRD.md#22-two-evaluation-contexts-two-profiles).

**Decision:** Ship two named, checked-in config profiles — `configs/eval.yaml`
(`{dense 0.85, sparse 0.15, struct 0.0}`, structural indexing skipped entirely) and
`configs/demo.yaml` (all three signals, per-`QueryType` weights). Every run prints its active
profile name; `appsretrieval_results.json` is only ever generated under `eval.yaml`.

**Alternatives considered:**
- *One profile, `structural_enabled: bool` flag.* Rejected: still needs two weight vectors and two
  candidate-width sets underneath the flag, so it is the same complexity with a less discoverable
  surface — a boolean buried in a 40-field settings object versus a config file named for its
  purpose.
- *Auto-detect corpus language and switch profiles at runtime.* Rejected: adds a detection failure
  mode (a JS file with no distinguishing syntax in a small chunk) for a choice the operator already
  knows and should state explicitly. Determinism (`NFR-08`) prefers explicit input over inference.

**Consequences:** Two YAMLs to keep in sync when a shared constant (e.g. `rrf_k`) changes; two
places an ablation table needs a column. Both costs are small and paid once; the alternative's cost
(a jury-facing structural-signal claim measured on a corpus where it cannot possibly contribute)
would have been paid on 27 September in front of the people scoring us.

**Reversal cost:** Low. Collapsing to one profile is a config-merge exercise, not an architecture
change, because both profiles already share the same code paths and only differ in weights and a
skip-structural-indexing flag internally.

---

## `ADR-002` — Weighted Reciprocal Rank Fusion over score-space fusion

**Status:** Accepted, 2026-09-15.

**Context:** Three signals — dense cosine `[-1,1]`, BM25 unbounded positive, structural graph-derived
unbounded — must merge into one ranked list. Their score distributions are not just differently
scaled, they are differently *shaped*: BM25 has a long right tail on rare-term matches, dense cosine
clusters tightly near the query's nearest neighbours.

**Decision:** Fuse in rank space with weighted RRF: `score(d) = Σ_i w_i / (60 + rank_i(d))`, weights
from `QueryPlan.strategy_weights`, per [Schema.md §8](Schema.md#8-fusedresult) and locked in
`_CONTRACT.md §5`.

**Alternatives considered:**
- *Min-max normalise each list to `[0,1]` per query, then weighted sum.* Rejected —
  [Rules.md AP-02](Rules.md#ap-02--fusing-in-score-space-instead-of-rank-space-rule-4-contract-5)
  documents why: per-query min-max makes a document's fused score depend on the worst candidate that
  happened to appear in the same list that query, which is not a property any of the three signals'
  raw scores should have.
- *Learned fusion (logistic regression over the three raw scores).* Rejected by
  [`NG-25`](NonGoals.md#ng-25--no-learning-to-rank-or-click-feedback) — needs labelled interaction
  data we do not have, and breaks `NFR-08` determinism guarantees around retraining.
- *CombSUM / CombMNZ on raw scores.* Same normalisation problem as min-max, worse: CombMNZ's
  multiply-by-hit-count term further rewards documents merely for appearing in more lists,
  independent of how well they ranked in any of them.

**Consequences:** RRF is a coarser signal than a well-calibrated score fusion could be in theory — it
only sees rank, not confidence. In practice this is corrected by the reranker, which sees the actual
query and document together for the top-25.

**Reversal cost:** Medium. Every ranked-list consumer (`FusedResult`, `TestPlan.md` Category F) is
built assuming rank-space, hand-verifiable arithmetic (`TC-049`–`TC-058`). A score-space fusion would
require re-deriving the entire test category and re-tuning every downstream threshold that assumes
RRF's `0 < score < ~0.05` range.

---

## `ADR-003` — `bm25s` over `rank-bm25`

**Status:** Accepted, 2026-09-15.

**Context:** The public rival submission ([PRD.md §9](PRD.md#9-competitive-positioning)) uses
`rank-bm25`, the most commonly reached-for BM25 library in Python. We need sparse retrieval over a
~10k-chunk corpus, and the reference box has 8 cores to use.

**Decision:** Use `bm25s`, locked in `_CONTRACT.md §2`.

**Alternatives considered:**
- *`rank-bm25`.* Rejected: pure-Python scoring loop, materially slower at 10k-chunk scale — the
  difference is large enough to matter against the `NFR-01` 12-minute cold-index budget once sparse
  index construction is added to dense embedding and structural parsing in the same wall clock.
- *Elasticsearch / OpenSearch embedded mode.* Rejected: a second process, a second failure mode on
  the evaluator's machine, and a licensing/footprint cost for a feature (full-text query DSL) we do
  not need — we run one fixed BM25 scoring function, not ad hoc search.

**Consequences:** `bm25s` is a newer, smaller-community library than `rank-bm25`; less Stack Overflow
coverage if something breaks. Mitigated by pinning an exact version (`_CONTRACT.md §1`) and keeping
the sparse retriever behind a thin typed adapter (`Rules.md §9.3`) so a swap back is contained to one
module.

**Reversal cost:** Low. `retrieval/sparse.py` and `indexing/sparse.py` are the only two files that
import `bm25s` directly, per the adapter-boundary rule.

---

## `ADR-004` — FAISS Flat below 50k vectors, IVF-PQ at or above

**Status:** Accepted, 2026-09-15.

**Context:** The dense index must serve both the 8,765-vector APPS corpus and a 10k-plus-chunk demo
repo, with headroom for a larger repo. Exact search (`IndexFlatIP`) is trivially fast and perfectly
accurate at small scale; approximate search (`IndexIVFPQ`) trades a small accuracy loss for
sublinear scaling, but only pays off once the corpus is large enough that its cluster-count
hyperparameters are well-populated.

**Decision:** `IndexFlatIP` below 50,000 vectors, `IndexIVFPQ` at or above, switch recorded in
`VersionManifest.index_kind`. Locked in `_CONTRACT.md §2`, enforced by
[`NG-18`](NonGoals.md#ng-18--no-ann-index-on-the-benchmark-corpus).

**Alternatives considered:**
- *`IndexIVFPQ` everywhere, for consistency.* Rejected: at 8,765 vectors, IVF-PQ's cluster
  populations are far below the library's own recommended minimum (`~39 * nlist` training points),
  which silently degrades recall in a way indistinguishable from a retrieval bug — exactly the
  failure [`NG-18`](NonGoals.md#ng-18--no-ann-index-on-the-benchmark-corpus) names.
- *HNSW (`IndexHNSWFlat`).* Rejected: strong recall/latency tradeoff in general, but adds a second
  index-kind branch to test and tune for a corpus size (10k–50k chunks) where Flat is already well
  under the `NFR-03` 900ms budget; the complexity is not yet earning its keep at our scale.

**Consequences:** The threshold is a single locked constant (`AXIOM_FAISS_IVF_THRESHOLD=50000`); a
demo repo that happens to land at 49,900 chunks and grows to 50,100 changes index kind on a routine
reindex, which is a valid but observable behaviour change worth a log line, not a silent one.

**Reversal cost:** Low. The threshold is one `Settings` field; both index kinds are already
implemented behind the same `dense.py` interface.

---

## `ADR-005` — blake2b-128 for every content address

**Status:** Accepted, 2026-09-15.

**Context:** `chunk_id`, `content_hash`, and `family_id` all need a fast, collision-resistant,
fixed-width digest, computed potentially millions of times across a 10-day eval and indexing
schedule on CPU-only hardware.

**Decision:** blake2b with `digest_size=16` (128 bits), rendered as 32 lowercase hex characters,
everywhere. Full derivation in [Schema.md §13](Schema.md#13-identity-and-hashing).

**Alternatives considered:**
- *SHA-256.* Rejected: slower than blake2b on hardware without SHA hardware extensions, which the
  reference box and an unknown judge's laptop cannot be assumed to have; 256 bits of collision
  resistance is unnecessary headroom for a corpus that will never exceed low millions of chunks.
- *MD5.* Rejected outright on principle — MD5's known collision weakness is an unforced liability in
  a content-addressed store, even though a deliberate collision attack is not a realistic threat
  model for this project. Using it would also read as a red flag to any reviewer who checks.
- *Python's built-in `hash()`.* Rejected: not stable across processes (salted by default since
  Python 3.3) and would silently break `TC-014`'s cross-process `chunk_id` stability requirement.

**Consequences:** 128 bits gives a collision probability under 10⁻²⁸ for a 10⁶-chunk corpus — several
orders of magnitude more headroom than this project will ever need, at negligible extra compute cost
over a narrower digest.

**Reversal cost:** High if attempted after chunks exist on disk — every `chunk_id` and
`content_hash` in every committed fixture and every built index would need re-derivation. Not
expected to be revisited.

---

## `ADR-006` — Two disjoint id spaces: `chunk_id` and `content_hash`

**Status:** Accepted, 2026-09-15.

**Context:** Two different questions need two different identities: "where is this exact text
located" (for the user-facing result) and "have I seen this exact code before, regardless of where"
(for embedding reuse and dedup). A single id trying to answer both questions forces a choice that
breaks one of the two use cases.

**Decision:** `chunk_id = blake2b128(content, file_path, start_line)` — location-sensitive.
`content_hash = blake2b128(normalise(content))` — location-insensitive. Full rationale in
[Schema.md §13.1](Schema.md#131-why-chunk_id-includes-location-and-content_hash-does-not).

**Alternatives considered:**
- *One id, location-insensitive.* Rejected: two copy-pasted functions at different paths would
  collapse to one chunk identity, and the user could not be shown both real locations — a direct
  violation of the deliverable "snippet + file + line location."
- *One id, location-sensitive, with a separate content-hash-keyed embedding cache lookup table.*
  Functionally similar to what we built, but naming it as "one id plus a side table" instead of "two
  named ids" obscures that the side table's key *is* the second identity space — the decision here
  is really about naming it explicitly so [Rules.md Rule 1](Rules.md#rule-1--ids-are-sacred)'s
  defences ("Axiom has exactly two id spaces and they never mix") have something concrete to
  enforce.

**Consequences:** Every stage that touches an id must know which of the two it is holding — this is
exactly what [Rules.md AP-01](Rules.md#ap-01--mutating-a-chunk-id-rule-1) and `TC-016`/`TC-017` exist
to police, and it is the single most dangerous rule-violation surface in the codebase per
[Rules.md Rule 1](Rules.md#rule-1--ids-are-sacred)'s own accounting.

**Reversal cost:** Very high — this is a cardinal rule. Reversing it requires an ADR and unanimous
four-member sign-off per [Rules.md §12](Rules.md#12-rule-change-procedure).

---

## `ADR-007` — Bounded agent loop: hard caps, not convergence

**Status:** Accepted, 2026-09-15.

**Context:** The theme is titled "Agentic Code Intelligence," and a refinement loop that runs "until
results look good" is the naive first design. The evaluation harness hands the system ~3,765 test
queries in one run; a loop with no termination guarantee has a nonzero chance of hanging on one
pathological query and costing the entire eval run.

**Decision:** `AXIOM_AGENT_MAX_PASSES = 2` bounds the **total** number of
retrieve → fuse → hydrate → rerank cycles, **initial pass included** — two cycles, at most one
refinement. Hard 5-second wall-clock deadline (`AXIOM_AGENT_WALL_CLOCK_MS`), checked before starting
any *new* pass and again before a new pass's rerank; on exhaustion, return the best pass seen so far
rather than erroring.

**Pass-counting semantics, settled 2026-09-23.** The suite previously carried two incompatible
readings of the same constant: this ADR, `Design.md §5.2`, `Appflow.md` Flow 3 and `API.md`'s
`passes_used` all meant *2 total*, while
[Rules.md AP-03](Rules.md#ap-03--unbounded-agent-loop-rule-3-contract-5)'s reference block seeds
`best` from a retrieval taken *outside* the loop and therefore means *3 total*. Since the reranker
dominates a pass's cost, that is roughly a 50% swing in p95 — large enough to decide whether
`NFR-04` is met. **`agent/loop.py:run` implements 2 total** (`for pass_no in range(1, max_passes+1)`,
with the whole retrieve→rerank cycle inside the loop body), so 2-total is the binding reading and
Rules.md AP-03 is the block that must be rewritten. The transcribed loop, with all five termination
conditions, is in
[TechSpecifications.md §5.3](TechSpecifications.md#53-agent-loop-bound).

Three further behaviours are part of this decision and were previously undocumented:

- **Pass 1 is exempt from the deadline check.** A query issued against an already-exhausted budget
  must still return something; a zero-length budget degrades to one pass, not to an empty answer.
- **A refinement pass is abandoned *before* its rerank** if the deadline falls mid-pass, because the
  reranker is ~80% of a pass's cost and the previous pass's result is already a good answer
  (`TC-088`).
- **Best-of, not last.** Passes are compared on `(cross_encoder_ran, top1)`: a pass whose
  cross-encoder actually ran outranks one that degraded to passthrough, and only within the same
  rung does raw top-1 decide. Comparing an RRF score against a cross-encoder score on magnitude
  alone would discard a better-but-uncalibrated pass every time.
- **`agent_enabled = false`** (the `eval` profile) collapses the loop to exactly one pass. Refinement
  is the ablation, not the retrieval.

**Alternatives considered:**
- *Loop until a sufficiency predicate passes, no cap.* Rejected outright — this is the exact failure
  mode the decision exists to prevent, and it is explicitly called out as a cardinal-rule violation
  pattern in [Rules.md](Rules.md#ap-03--unbounded-agent-loop-rule-3-contract-5).
- *A larger fixed pass count (e.g. 4) for a more thorough refinement.* Rejected on the CPU-only
  budget: each pass re-runs retrieval and a rerank batch; a 4-pass loop cannot fit inside the
  `NFR-04` 5-second p95 budget on the reference box without either shrinking the reranker's candidate
  width (hurting accuracy every query, not just the ones that need a second pass) or accepting a
  budget breach. *Note that this rejection is an estimate, not a measurement* — no p95 exists yet
  (`OQ-13`). What makes the cap safe regardless is the monotonic deadline, which bounds the loop
  whether or not the pass-count arithmetic was right.

**Consequences:** A query that would genuinely benefit from a second refinement (a third cycle) does
not get one. This is judged an acceptable trade because the two-pass gain is expected to be
front-loaded
(the largest jump in result quality is pass 1 → pass 2; further passes have rapidly diminishing
returns on a corpus this size), and because the trade is stated plainly rather than hidden.

**Reversal cost:** Medium — raising the cap needs a re-measured p95 and is explicitly gated behind a
new ADR by [Rules.md §9.6](Rules.md#96-performance-regression-policy) item 4.

---

## `ADR-008` — The query LLM never reads code

**Status:** Accepted, 2026-09-15.

**Context:** The problem statement's core constraint is that the codebase is larger than any LLM
context window. A design that feeds retrieved snippets back into an LLM for re-ranking or
summarisation reintroduces exactly the problem the retrieval architecture exists to solve, just
deferred to a later stage.

**Decision:** The query LLM has exactly four permitted jobs — classify, expand, decompose, judge
sufficiency — and never receives chunk text as input. Sufficiency is judged from **scores**
(`top-1 rerank_score < 0.35` or fewer than 3 results above `0.20`), not from reading the candidates.
Locked in `_CONTRACT.md §2` and enforced as [`NG-23`](NonGoals.md#ng-23--no-llm-ingestion-of-retrieved-code).

**Alternatives considered:**
- *LLM-as-reranker (score each candidate by prompting with the query and the snippet).* Rejected:
  this is exactly the pattern `NG-23` forecloses — it would make agent behaviour depend on corpus
  size (more/longer candidates means a bigger prompt), the opposite of the property we want, and it
  duplicates the cross-encoder reranker's job with a slower, less consistent tool.
- *LLM-generated answer synthesis over top results.* Rejected by
  [`NG-02`](NonGoals.md#ng-02--no-answer-synthesis) — out of task scope entirely; the deliverable is
  a ranking, not a paragraph.

**Consequences:** The agent cannot reason about *why* a result is weak beyond its numeric scores —
it cannot notice "these five results are all about the wrong subsystem" the way a human skimming
them could. This is accepted because the alternative reintroduces the context-window problem the
whole architecture is built to avoid, and because it is exactly what keeps
`AXIOM_LLM_ENABLED=false` a first-class, not crippled, mode (`NFR-07`).

**Reversal cost:** High — this shapes the entire agent module boundary
(`agent/classifier.py`, `agent/planner.py`, `agent/evaluator.py`) and the design argument in
[PRD.md §6](PRD.md) (Why Not Just Use an LLM for Everything, in `PROJECT_OVERVIEW.md §6`).

---

## `ADR-009` — Supervised configuration tuning instead of model fine-tuning

**Status:** Accepted, 2026-09-15.

**Context:** The `AppsRetrieval` train split (5,000 pairs) is available. The naive way to use labelled
pairs is to fine-tune the embedder or reranker on them. `_CONTRACT.md`'s CPU-only, 10-day-window
constraints make that expensive and risky.

**Decision:** `FR-26` — use the 5,000 train pairs (split 4,000 tune / 1,000 dev, seeded id lists
under `data/splits/`) to tune **configuration**, not weights: RRF signal weights per profile,
sufficiency thresholds, query-preprocessing variants. Zero gradient steps anywhere. Enforced as
[`NG-06`](NonGoals.md#ng-06--no-model-fine-tuning-or-training-of-any-kind).

**Alternatives considered:**
- *Contrastive fine-tune of the embedder on the 5,000 train pairs.* Rejected: a single epoch over
  5,000 pairs with a 0.6B encoder on the 8-core reference box does not fit inside the build window
  once export/quantisation/testing time is also accounted for, and a half-finished fine-tune
  checkpoint is worse than the released base model, not better.
- *LoRA adapter, cheaper than a full fine-tune.* Rejected for the same time-budget reason, plus it
  reintroduces a training pipeline (data loader, loss, checkpointing, eval-during-training) that has
  to be built and debugged from nothing, for an accuracy gain that is unmeasured and could easily be
  smaller than what configuration tuning already captures for free.
- *Do nothing with the train split; tune everything on intuition.* Rejected — this is strictly worse
  than the chosen path at zero extra cost; the split is free to use for configuration and the sweep
  infrastructure (`T-112`, `T-141`) is needed anyway for the placeholder-replacement discipline in
  [Rules.md §8](Rules.md#8-the-placeholder-convention).

**Consequences:** The accuracy ceiling from configuration tuning alone is lower than a well-executed
fine-tune could reach in principle. Accepted as the correct trade for a CPU-only project inside the
2026-09-23 → 2026-09-27 window. The one exception previously carved out — doc2query expansion,
`ADR-013` — was **withdrawn on 2026-09-23**, so configuration tuning on the train split (`FR-26`) is
now the *only* accuracy lever, and `NG-06` is absolute.

**Reversal cost:** High under the current schedule (there is no time budget for it this cycle);
architecturally low, since nothing in the pipeline assumes frozen weights beyond the ONNX export
step.

---

## `ADR-010` — `git diff` as the sole versioning source of truth

**Status:** Accepted, 2026-09-15.

**Context:** P1 requires incremental reindexing that costs time proportional to the diff, not the
repo. Something has to be the authority on "what changed between version A and version B."

**Decision:** `git diff --name-status <old>..<new>` resolved into A/M/D/R, with `VersionManifest.file_hashes`
as an independent content-level cross-check. Enforced as
[`NG-24`](NonGoals.md#ng-24--no-non-git-version-sources).

**Alternatives considered:**
- *mtime-based change detection.* Rejected outright — [Rules.md AP-09](Rules.md#ap-09--cache-keyed-by-anything-other-than-content-rule-2)
  already documents why a touched-but-unchanged file must not trigger recompute, and mtime cannot
  distinguish "touched" from "changed."
- *Directory-diff heuristic (walk both trees, compare hashes) with no git dependency.* Considered
  seriously as a git-independent fallback, and partially adopted: `file_hashes` in
  `VersionManifest` *is* exactly this, kept as a cross-check rather than the primary path, because
  a pure hash-diff cannot distinguish a rename from a delete-plus-add — which matters directly for
  the "renames cost zero embeddings" property `FR-19` promises.
- *Support SVN/Mercurial as additional version sources.* Rejected on time budget — every additional
  VCS is its own diff-parsing module, its own edge cases, and a JS voice-assistant codebase in this
  problem domain is git by overwhelming likelihood.

**Consequences:** A target that is not a git repository gets no incremental path at all — always a
full index, which is correct behaviour (there is nothing to diff against) rather than a degraded
one, per [`NG-24`](NonGoals.md#ng-24--no-non-git-version-sources).

**Reversal cost:** Medium. `versioning/gitdiff.py` is the one module that shells out to git; a second
VCS backend would sit behind the same `diff_versions()` interface it already exposes.

---

## `ADR-011` — SQLite for the structural index, not a graph database

**Status:** Accepted, 2026-09-15.

**Context:** The structural signal needs to answer callers-of, callees-of, imports-of, exports-of,
and ordered-call-pair queries — a small, fixed set of traversal shapes, not open-ended graph
analytics.

**Decision:** `structural.sqlite` with four relations (symbols, calls, imports, exports), queried
with plain SQL. DDL locked in `_CONTRACT.md §6`, mirrored in [Schema.md](Schema.md). Enforced as
[`NG-27`](NonGoals.md#ng-27--no-graph-database-for-the-structural-index).

**Alternatives considered:**
- *Neo4j / a Cypher-speaking graph database.* Rejected: a second long-lived service, a second
  connection lifecycle, and a second thing that can fail to start on the judge's laptop — for query
  shapes that are all expressible as a self-join with an index, per
  [`NG-27`](NonGoals.md#ng-27--no-graph-database-for-the-structural-index)'s own accounting of the
  ordered-call-pair query.
- *In-memory graph object (e.g. `networkx`) rebuilt on load.* Rejected: rebuilding a graph from
  `chunks.jsonl` on every process start costs time proportional to corpus size on every query-serving
  startup, whereas SQLite is a file that opens instantly and is queried lazily.

**Consequences:** Deep graph algorithms (transitive closure beyond depth 1, shortest-path queries)
would be awkward in plain SQL. Accepted because none of the three query archetypes in
[PRD.md §1.1](PRD.md#11-the-three-query-archetypes) need more than depth 1 — see
[NonGoals.md §4](NonGoals.md#4-deferred-not-rejected), "call-graph reachability beyond depth 1."

**Reversal cost:** Medium. `indexing/structural.py` and `retrieval/structural.py` are the two modules
that would need a new backend; the SQL query shapes would need re-expression as graph traversals,
but the four-relation schema translates directly to a property-graph model if this is ever revisited.

---

## `ADR-012` — ONNX Runtime INT8 + llama.cpp GGUF for a CPU-only stack

**Status:** Accepted, 2026-09-15.

**Context:** `NFR-06` requires CPU-only execution with no accelerator dependency anywhere, on
hardware we do not control (the judge's laptop).

**Decision:** ONNX Runtime with `CPUExecutionProvider` exclusively, INT8 dynamic quantisation for
both the embedder and the cross-encoder, and `llama-cpp-python` with a Q4_K_M GGUF quantisation for
the query LLM. Locked in `_CONTRACT.md §2`, enforced as
[`NG-05`](NonGoals.md#ng-05--no-gpu-cuda-rocm-or-metal-path) and
[`NG-17`](NonGoals.md#ng-17--no-proprietary-model-api-on-the-core-path).

**Alternatives considered:**
- *PyTorch fp32 inference throughout.* Rejected: ~4x slower than the ONNX INT8 path, and
  materially larger download/RSS footprint against `NFR-12`. Rejected *completely* — earlier
  revisions of this ADR described `AXIOM_EMBEDDING_BACKEND=torch` as a debugging-only fallback,
  but no such variable and no torch runtime exist in `indexing/embedder.py`. There is no torch
  inference path to fall back to; below ONNX Runtime the ladder goes straight to the seeded hash
  embedder.
- *A proprietary API (OpenAI/Anthropic/Voyage) for embedding and reranking.* Rejected outright by
  [`NG-17`](NonGoals.md#ng-17--no-proprietary-model-api-on-the-core-path) — a network-restricted
  judging laptop, an expired key, or a rate limit could zero the score on judging day, which is an
  unacceptable single point of failure for a hackathon submission.
- *GGUF for every model, including the embedder and reranker (uniform stack).* Rejected: ONNX
  Runtime has materially better throughput than llama.cpp for encoder-only embedding/classification
  workloads at this scale; GGUF's strength (llama.cpp's efficient autoregressive decode) is specific
  to the generative query LLM, so using it for the encoders would trade speed for stack uniformity
  we do not need.

**Consequences:** Two different quantised-model runtimes (ONNX Runtime and llama.cpp) in one process,
rather than one. Accepted because each is the better tool for its own workload shape, and both are
already required to have declared fallbacks (`NFR-07`) regardless of runtime choice.

**Reversal cost:** High — this shapes the entire model-loading and inference boundary
(`rerank/cross_encoder.py`, `agent/llm.py`) and the export pipeline in [Setup.md §6](Setup.md).

---

## `ADR-013` — Reverse doc2query expansion as the one corpus-touching intervention

**Status:** **Withdrawn, 2026-09-23.** Was `Proposed, 2026-09-16`. Closes
[`OQ-06`](OpenQuestions.md#oq-06--does-reverse-doc2query-expansion-earn-its-index-time-cost) as
"no, not this cycle."

**Context:** [`NG-06`](NonGoals.md#ng-06--no-model-fine-tuning-or-training-of-any-kind) excludes
gradient training but named one accepted exception: reverse doc2query expansion, which generates
index-time text (candidate queries a chunk would answer, appended to the chunk's sparse-index
representation) rather than updating any model weight.

**Why it is withdrawn — three independent reasons, any one sufficient:**

1. **It contradicts `ADR-008` and the contract, and the contradiction was never resolved.**
   [`ADR-008`](Decisions.md#adr-008--the-query-llm-never-reads-code) and `_CONTRACT.md §2` state
   that the query LLM has exactly four permitted jobs — classify, expand, decompose, judge
   sufficiency — and "never receives chunk text as input." `NG-23` says the same. Reverse doc2query
   *is* the LLM reading chunk text: that is its entire mechanism. The proposal reused "the same
   query LLM already in the stack," so it was not a different component operating under a different
   rule. Two resolutions were available and neither was taken: amend `ADR-008` to read "never at
   **query time**" and carve an explicit, bounded index-time exception, or drop the proposal. We
   drop it. The "never reads code" claim is the project's central innovation argument and the line
   in `PRD.md §8` that earns the 15% theme-relevance criterion; a carve-out that a jury has to be
   walked through is worth less than the claim it weakens.
2. **The number it banked was fabricated.** `NonGoals.md NG-06` credited it with
   "**+4 to +8 NDCG@10**" — roughly a quarter of the stated path to the accuracy target — presented
   as an estimate with no source. It appears in no paper cited in this suite and in no run of ours.
   A gate calibrated on it would have been calibrated on nothing. The estimate is retracted along
   with the ADR; see [`PRD.md §2`](PRD.md#2-goals-and-success-metrics) for how accuracy targets are
   now derived (from our own measured dense-only baseline, as relative gains).
3. **It does not exist and there is no time to build it.** There is no
   `src/axiom/indexing/expansion.py`. It would add one generation pass per corpus document to a cold
   index already projected to miss `NFR-01` by 1.7–3.3x unaided
   ([TechSpecifications.md §8.5](TechSpecifications.md#85-cold-index-placeholder-oq-14)) — on the
   8,765-document APPS corpus that is hours, not minutes, and it was never priced into `NFR-01`.

**Consequence:** `NG-06` is now absolute for this cycle — **nothing writes to the corpus**, neither
by gradient nor by generation. `NonGoals.md NG-06` and `NG-19` have been corrected to remove both
the "+4 to +8" and the claim that an accepted corpus-touching intervention exists.

**Reversal cost:** Low, and deliberately kept low. Reinstating it requires exactly two things, in
this order: (a) an amendment to `ADR-008` and `_CONTRACT.md §2` that states the index-time exception
explicitly and bounds it, agreed *before* any code is written; (b) a real ablation row — eval profile
with and without expansion, same weights, same seed — replacing the withdrawn estimate. It is a
config-gated index-time pass with no schema change, so the engineering is a day; the contract
amendment is the part that must not be skipped again.

---

## `ADR-014` — Frozen, `extra="forbid"` Pydantic models as the schema base

**Status:** Accepted, 2026-09-15.

**Context:** Four people write code against a shared data model concurrently. A typo'd field name in
a hand-edited fixture or a stale `chunks.jsonl` from an older build must fail loudly, not silently
drop the unknown field and produce a subtly wrong index days later.

**Decision:** Every schema model inherits `AxiomModel`, configured `extra="forbid"`, `frozen=True`,
`str_strip_whitespace=False`. Full rationale in [Schema.md §2](Schema.md#2-module-layout).

**Alternatives considered:**
- *Plain dataclasses, validated manually at load boundaries.* Rejected: reimplements what Pydantic
  already gives for free (`field_validator`, `model_validator`, JSON round-tripping), and manual
  validation is exactly the kind of thing that gets skipped under Day 9 time pressure.
- *`extra="allow"` (permissive) models.* Rejected: a typo in a config file or a hand-edited test
  fixture would silently pass through as an ignored extra field rather than failing at load time —
  the opposite of what a shared four-person contract needs.
- *Mutable models with defensive `.copy()` everywhere a stage might need to change a field.*
  Rejected: `frozen=True` makes the purity guarantee in
  [Rules.md Rule 2](Rules.md#rule-2--stages-are-pure) structurally enforced by the type system
  instead of relying on every contributor remembering to copy — see
  [Rules.md AP-08](Rules.md#ap-08--mutating-a-caller-owned-input-rule-2) for the bug class this
  prevents outright.

**Consequences:** Every "modify one field" operation becomes `model_copy(update={...})` instead of an
in-place assignment — slightly more verbose, and the cost is paid on every mutation site in
`rerank/cross_encoder.py` and `agent/loop.py` in exchange for the purity guarantee those very modules
most need.

**Reversal cost:** Very high — this is the shared contract every workstream codes against per
[Schema.md §1](Schema.md#1-scope-and-authority); changing it requires the "four-member sign-off"
process named throughout [Rules.md](Rules.md).

---

## `ADR-015` — Rename PRISM to Axiom

**Status:** Accepted, 2026-09-16. Resolves [`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission).

**Context:** A rival public submission ships under the identical project name "PRISM" and the
identical organiser-mandated release tag `PRISM_GENAI_HACKATHON_Y2026`
([PRD.md §9](PRD.md#9-competitive-positioning)). The tag cannot change; the project's own name can.

**Decision:** Rename the project **PRISM → Axiom**. `src/axiom/` becomes the import root, `AXIOM_` the
environment-variable prefix, `axiom` the CLI entrypoint name. The release tag
`PRISM_GENAI_HACKATHON_Y2026` is unaffected — it identifies the *submission slot*, assigned by the
organisers, not the project.

**Alternatives considered:**
- *Keep the name PRISM and rely on the team name ("Incognito") and repo URL to disambiguate.*
  Rejected: a jury reviewing dozens of entries at speed is exactly the audience most likely to
  conflate two same-named submissions from a title slide alone; the disambiguation burden should not
  sit on the reader.
- *A different new name than "Axiom."* Considered and rejected in favour of continuity: the team's
  two founding members shipped an earlier hackathon project also named Axiom (a verifier-centric
  reasoning framework, unrelated codebase — see `PROJECT_OVERVIEW.md` Appendix C), so reusing the
  name signals lineage without implying the two projects share code.

**Consequences — this is the part that matters operationally.** The rename landed in code first and
in prose last, which produced eight days of drift.

**Settled 2026-09-23, by the project owner, binding on every document:** the name is **Axiom**,
everywhere, with exactly three exceptions.

| Axis | Value |
|---|---|
| Python package / import root | `axiom` (`src/axiom/`) |
| CLI entrypoint | `axiom` |
| Environment-variable prefix | `AXIOM_` |
| On-disk index root | `.axiom/` |
| Exception base class | `AxiomError` (`src/axiom/core/errors.py`) |
| **Exception 1** — release tag | `PRISM_GENAI_HACKATHON_Y2026`, organiser-prescribed, unchanged |
| **Exception 2** — event name | "Samsung PRISM GenAI Hackathon" is the *event*, not the project |
| **Exception 3** — rival repo | `github.com/DeshnaDey/Samsung-PRISM` is someone else's URL |
| **Exception 4** — our repo URL | `github.com/HarshdeepAthawale/Samsung-Prism-Hack` predates the rename; renaming it breaks every documented clone URL. Left as-is |
| **Exception 5** — on-disk sidecar | `sparse.bm25s/prism_meta.json` (`indexing/sparse.py:79`). The only `prism` literal that reaches disk. Renaming it is a format change requiring a full reindex; left as-is for this cycle |

The complete list, with the reasoning for each, is
[`_CONTRACT.md §0`](_CONTRACT.md#0-identity). Exceptions 4 and 5 were added on
2026-09-23; this ADR originally named only the first three.

`_CONTRACT.md §0/§1/§3` carried `prism` / `PRISM_` / package `prism` until 2026-09-23 and **has
since been corrected** — twenty-one of the twenty-two docs already used `axiom`, and so does all 27k
lines of `src/`. It was brought into line with them, not the reverse, notwithstanding its own "if a
doc contradicts this file, the doc is wrong" clause: that clause presumes the contract is current,
and on this axis it was not.

The error taxonomy has been reconciled against the code rather than against prose: `errors.py`
defines exactly `AxiomError`, `AxiomContractError`, `IndexNotFoundError`,
`DegradationExhaustedError`. Documents that referred to `PrismConfigError`, `PrismModelError`,
`PrismParseError` or `PrismBudgetError` were naming classes that do not exist; those references have
been rewritten to name real classes or real mechanisms (a Pydantic `ValidationError` at startup, for
instance). `Rules.md §9.2`'s seven-class taxonomy still over-enumerates and is flagged for its owner.

**The line that caused the drift has been deleted.** `OpenQuestions.md` previously advised: "Until
`T-201` lands, prefer whichever name the *file you are editing* already uses." That sentence
institutionalised the divergence — it made every subsequent edit widen the gap by design, and it is
the single identifiable cause of a 249-vs-0 split on the env prefix. It is removed, and the rule is
now simply: use `Axiom`.

**Reversal cost:** Low to reverse the naming itself (it is find-and-replace across identifiers, not
an architecture change); the cost that matters is the one-time effort of finishing the propagation,
which is exactly what `T-201` tracks.
