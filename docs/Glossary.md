# Glossary

Domain terms, acronyms, metric formulas, and project-specific jargon used across the PRISM documentation suite.

**Owner:** Anish Grover
**Last updated:** 2026-09-16
**Status:** Draft

Related: [Rules.md](Rules.md) · [Schema.md](Schema.md) · [_CONTRACT.md](_CONTRACT.md) · [PRD.md](PRD.md) · [TechSpecifications.md](TechSpecifications.md) · [README.md](README.md)

---

## How to read this document

Every term below is defined once, here, and linked to from wherever else it appears rather than
redefined inline — that is the entire point of a glossary in a doc suite this cross-referenced.
Where a term has a canonical model, formula, or table elsewhere in the suite, this entry points at
it rather than restating it; if the two ever disagree, the canonical source wins and this glossary
is wrong and should be fixed.

---

## Project-specific jargon

[Rules.md](Rules.md)'s own introduction names five terms as defined here — this section is that
promise kept. Read this section before [Rules.md](Rules.md) if the two are new to you at the same
time.

**Signal** — One of the three independent retrieval strategies: dense (embedding similarity),
sparse (BM25 lexical), or structural (AST/call-graph traversal). Each signal takes a query and
returns a ranked `list[ScoredChunk]` in its own, mutually incomparable score domain. See
[Schema.md §3.2](Schema.md#32-signalkind-semantics) for the `SignalKind` enum and
[Design.md §3](Design.md#the-three-signal-rationale) for why three signals rather than one or two.

**Stage** — Any pure callable in the pipeline that takes typed input and produces typed output:
chunker, embedder, dense retriever, sparse retriever, structural retriever, fusion, reranker, agent
pass, evolutionary dedupe. Stages are the unit [Rules.md Rule 2](Rules.md#rule-2--stages-are-pure)
governs (same input + same config ⇒ byte-identical output) and the unit
[Rules.md Rule 3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s degradation ladder is defined
per-stage.

**Candidate width** — The number of results a signal (or a fusion/rerank step) is asked to return
before the next stage narrows the list further. The locked widths: dense `K=100`, sparse `K=100`,
structural `K=50` → fused to `N=25` → reranked to top `10`. Locked in `_CONTRACT.md §5`, elaborated
in [TechSpecifications.md §5](TechSpecifications.md#5-retrieval-spec).

**Degradation ladder** — The ordered list of fallbacks a stage falls through when its primary path
fails, ending in a defined, typed, empty-but-valid result rather than an exception. The full
stage-by-stage table is [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade); the
architectural rationale for why this is a first-class return path rather than an error path is
[Design.md §5](Design.md#5-degradation-ladder-architecture-view).

**Sufficiency predicate** — The rule the agent loop uses to decide whether a pass's results are good
enough to return, or whether to refine and retry: top-1 rerank score `< 0.35` **or** fewer than 3
results score above `0.20`. Both thresholds are `# PLACEHOLDER` values pending measurement — see
[`OQ-10`](OpenQuestions.md#oq-10--are-the-agent-sufficiency-thresholds-035020-right) and
[Tracker.md `T-141`](Tracker.md#24-agent-rerank-api-ui-t-091t-150). Implemented in
`agent/evaluator.py`, consumed by `agent/loop.py`.

---

## Domain terms

General code-retrieval and information-retrieval vocabulary used throughout the doc suite,
independent of any PRISM/Axiom-specific naming. Where a fuller definition already exists elsewhere
in this glossary, the entry here points at it rather than repeating it.

**Chunk** — see [Chunking and content-addressing terms](#chunking-and-content-addressing-terms) below.

**Embedding** — A fixed-length dense vector representation of a piece of text, produced by an
embedder model such that texts with similar meaning produce vectors that are close together under
some distance or similarity measure. PRISM's dense embedder produces 1024-dim vectors
(`Qwen/Qwen3-Embedding-0.6B`) or 384-dim vectors (the `all-MiniLM-L6-v2` fallback), L2-normalised
before indexing. `_CONTRACT.md §2`.

**Dense retrieval** — Retrieval by nearest-neighbour search over embedding vectors: the query is
embedded with the same model as the corpus, and the chunks whose vectors are closest (by inner
product over L2-normalised vectors — see *Cosine similarity vs. inner product* below) are returned.
PRISM's `DENSE` signal. See [Schema.md §3.2](Schema.md#32-signalkind-semantics).

**Sparse retrieval** — Retrieval by lexical term matching and term-frequency weighting, implemented
via BM25 (below). Finds exact and near-exact vocabulary overlap between query and chunk and has no
notion of meaning beyond shared tokens. PRISM's `SPARSE` signal.

**BM25 (Best Matching 25)** — see [Acronyms](#acronyms) below.

**Cross-encoder vs. bi-encoder** — Two ways of scoring a (query, document) pair with a transformer.
A **bi-encoder** (PRISM's dense embedder) encodes the query and the document *separately* into
fixed vectors and compares them with a cheap similarity function — fast enough to run against an
entire corpus, but the model never sees the query and document together. A **cross-encoder**
(PRISM's reranker, `BAAI/bge-reranker-v2-m3`) encodes the query and document *jointly* in one
forward pass and outputs a single relevance score — more accurate because the model can attend
across the pair, but too slow to run against more than a small candidate set. This is exactly why
PRISM retrieves broadly with the bi-encoder/lexical/structural signals first (`K=100`/`100`/`50`)
and reranks only the fused top-25 with the cross-encoder. `_CONTRACT.md §2`, `_CONTRACT.md §5`.

**Reranking** — The second-stage scoring pass that re-orders a small candidate set (PRISM: the
fused top-25) using a more expensive, more accurate model (PRISM: the cross-encoder) than the
first-stage retrievers could afford to run against the whole corpus. See `FR-12`,
[Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade) for the reranker's degradation
ladder, and the **Passthrough** entry below for the well-formed degraded outcome when reranking
cannot run.

**AST (Abstract Syntax Tree)** — see [Acronyms](#acronyms) below.

**Call graph** — A directed graph whose nodes are functions/methods and whose edges represent
"caller invokes callee," derived by PRISM purely from static AST analysis (source order, never
execution) and stored as the `calls` relation in `structural.sqlite`. Answers query archetype Q2
("which files call X before Y") by a source-order comparison, never a runtime trace — see
[`NG-15`](NonGoals.md#ng-15--no-code-execution-sandboxing-or-dynamic-analysis).

**RRF (Reciprocal Rank Fusion)** — see [Retrieval and fusion terms](#retrieval-and-fusion-terms) below.

**Cosine similarity vs. inner product** — Two closely related vector similarity measures. Cosine
similarity is the inner product of two vectors divided by the product of their magnitudes — i.e. it
ignores vector length and measures only the angle between them. When vectors are **L2-normalised**
(magnitude forced to 1, as every embedding in PRISM is — `_CONTRACT.md §2`), that division is a
no-op and inner product *equals* cosine similarity exactly. This is why FAISS
`IndexFlatIP`/`IndexIVFPQ` inner-product search over PRISM's normalised vectors is already the
correct higher-is-better relevance score, with no separate cosine step — see
[Rules.md AP-10](Rules.md#ap-10--turning-a-distance-into-a-score-by-accident-rule-4).

**Quantization (INT8)** — Reducing a model's numeric precision (PRISM: from fp32/bf16 weights down
to 8-bit integers) to shrink memory footprint and speed up CPU inference, at a small, usually
imperceptible accuracy cost. PRISM applies **post-training dynamic INT8 quantisation** (no
retraining) to the embedder and reranker via `optimum-cli`, exported to ONNX. See
[Setup.md §6](Setup.md#6-onnx-export-and-int8-quantisation).

**ONNX** — see [Acronyms](#acronyms) below.

**GGUF** — see [Acronyms](#acronyms) below.

**Content-addressing** — see [Chunking and content-addressing terms](#chunking-and-content-addressing-terms) below.

**Incremental reindexing** — see *Cold index vs. incremental reindex* under
[Chunking and content-addressing terms](#chunking-and-content-addressing-terms) below.

---

## Retrieval and fusion terms

**RRF (Reciprocal Rank Fusion)** — The rank-space fusion function that merges the three signals'
ranked lists into one. Formula, locked in `_CONTRACT.md §5` and reproduced exactly (see
[Metric and scoring formulas](#metric-and-scoring-formulas) below):

```
score(d) = Σ_i  w_i / (k + rank_i(d))
```

with `k = 60` and per-`QueryType` weight vectors `w_i`. See
[Schema.md §8](Schema.md#8-fusedresult) for the `FusedResult` model and
[Decisions.md `ADR-002`](Decisions.md#adr-002--weighted-reciprocal-rank-fusion-over-score-space-fusion)
for why rank space was chosen over score space.

**Rank-space fusion vs. score-space fusion** — Rank-space fusion (RRF) consumes only each
candidate's *position* in a ranked list, never its raw score; score-space fusion would combine raw
scores directly (e.g. a weighted sum of normalised scores). PRISM uses rank-space fusion because the
three signals' raw scores are on incomparable, differently-shaped distributions (bounded cosine,
unbounded BM25, unbounded graph score) — see
[Rules.md AP-02](Rules.md#ap-02--fusing-in-score-space-instead-of-rank-space-rule-4-contract-5) for
the concrete failure mode score-space fusion would introduce.

**Dominant signal** — The signal contributing the largest weighted term to a chunk's RRF score;
recorded on `FusedResult.dominant_signal` and surfaced to the user as the leading clause of
`RetrievalResult.match_reason`. Ties resolve in the fixed order `DENSE`, `SPARSE`, `STRUCTURAL`. See
[Schema.md §8](Schema.md#8-fusedresult).

**Passthrough** — The degradation outcome when the reranker cannot run (model load failure,
timeout): the RRF order is returned unchanged, `rerank_score` is `None` on every result, and
`match_reason` records `"rerank_passthrough"`. A first-class, well-formed outcome, not an error —
see [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade), "Passthrough is a first-class
outcome, not a bug."

**Query archetype** — One of the three query shapes the problem statement is built around, each
requiring a different dominant signal. Defined with worked examples in
[PRD.md §1.1](PRD.md#11-the-three-query-archetypes):

| Archetype | Example | Dominant signal |
|---|---|---|
| Q1 — semantic | "How is the input preprocessed before going to the main function?" | dense |
| Q2 — structural | "Which files call tool XYZ before tool ABC?" | structural |
| Q3 — usage | "Where is the Bluetooth-settings deeplink used?" | sparse |

**Sub-query** — One of up to three decomposed pieces of a compound query, produced by the planner
when a single query needs more than one retrieval pass to answer (e.g. "calls X before Y"
decomposes into one sub-query per identifier plus an ordering check). See
`QueryPlan.sub_queries` in [Schema.md §10](Schema.md#10-queryplan).

---

## Chunking and content-addressing terms

**Chunk** — The atomic unit of retrieval: one function, method, class, module, or oversize-split
block, with its exact text, location, and retrieval metadata. See
[Schema.md §6](Schema.md#6-chunk) for the full `Chunk` model.

**`chunk_id`** — The location-sensitive identity of a chunk: `blake2b128(content, file_path,
start_line)`. Two byte-identical functions at different paths get different `chunk_id`s, because the
user must be shown both real locations. See
[Schema.md §13](Schema.md#13-identity-and-hashing) and
[Rules.md Rule 1](Rules.md#rule-1--ids-are-sacred).

**`content_hash`** — The location-*insensitive* identity of a chunk's normalised content:
`blake2b128(normalise(content))`. Two copy-pasted functions at different paths share one
`content_hash` and therefore one cached embedding. The mechanism behind intra-version dedup,
cross-version dedup, and zero-cost renames — see
[Schema.md §13.1](Schema.md#131-why-chunk_id-includes-location-and-content_hash-does-not).

**Content-addressing** — Identifying a piece of data by a hash of its own content rather than by
where it is stored. PRISM content-addresses chunks twice (`chunk_id`, `content_hash`) for two
different purposes — see the two entries above — and content-addresses embeddings once, by
`content_hash`, as the blob-store key.

**Blob** — A cached embedding vector stored at `.axiom/blobs/<content_hash>.npy`, keyed by
`content_hash` so it is shared across every chunk (and every version) with identical normalised
content. The one sanctioned stateful cache in the pipeline — see
[Rules.md Rule 2](Rules.md#rule-2--stages-are-pure), "the single sanctioned exception."

**Cold index vs. incremental reindex** — A cold index builds all three indexes (dense, sparse,
structural) for every chunk in a corpus from nothing, budgeted at ≤ 12 minutes for 10k chunks
(`NFR-01`). An incremental reindex re-chunks and re-embeds only files a `git diff` reports as added
or modified, drops chunks from deleted files, and reuses blobs by `content_hash` for everything
else, budgeted at ≤ 45 seconds for a 50-file diff (`NFR-02`). See
[TechSpecifications.md §7](TechSpecifications.md#7-versioning-spec).

---

## Versioning and evolutionary retrieval terms

**Snippet family** — The evolutionary-retrieval grouping unit: the set of near-identical chunks
(cosine ≥ 0.95, sharing `symbol` and `file_path`) representing one logical snippet observed across
multiple indexed versions. See [Schema.md §11](Schema.md#11-snippetfamily) for the full
`SnippetFamily` model.

**Stability** — A snippet family's version coverage: `len(members) / total_indexed_versions`, a
value in `(0, 1]`. A family present in every indexed version has `stability = 1.0`; one appearing
once in ten versions has `stability = 0.1`. Drives the ranking bonus below. See
[Schema.md §11](Schema.md#11-snippetfamily).

**Stability bonus** — The ranking multiplier applied to a multi-version snippet family's score:

```
final = base * (1 + 0.10 * stability)
```

applied only when the family spans two or more versions (`is_multi_version`); a single-version
family gets no bonus, because novelty is not the same property as instability. See
[Schema.md §11](Schema.md#11-snippetfamily) and
[Metric and scoring formulas](#metric-and-scoring-formulas) below.

**Profile** — A named, checked-in configuration bundle selecting candidate widths, signal weights,
and which signals run at all. Two first-class profiles exist: `configs/eval.yaml` (dense-heavy,
sparse down-weighted, structural signal disabled — tuned for the CoIR `AppsRetrieval` benchmark
corpus) and `configs/demo.yaml` (all three signals, per-`QueryType` weights — tuned for the live
JavaScript demo corpus). See [PRD.md §2.2](PRD.md#22-two-evaluation-contexts-two-profiles) and
[Decisions.md `ADR-001`](Decisions.md#adr-001--two-first-class-evaluation-profiles).

---

## Process and convention terms

**Placeholder constant** — A configuration value that is a design estimate rather than a measured
one (e.g. the sufficiency thresholds `0.35`/`0.20`, the dedupe cosine `0.95`, the chunk-size target
`64–512`). Marked with a `# PLACEHOLDER` comment and an owning tracker id in `config.py`; may not
appear, unmarked, in any reported score. Full convention in
[Rules.md §8](Rules.md#8-the-placeholder-convention).

**Cardinal rule** — One of the four rules in [Rules.md](Rules.md) that outrank every other rule in
the project: IDs are sacred, stages are pure, never raise on bad input (degrade instead), higher is
better with descending sort. If two rules conflict, the lower-numbered cardinal rule wins.

**Anti-pattern (`AP-##`)** — One of fourteen catalogued wrong/right code pairs in
[Rules.md §11](Rules.md#11-anti-pattern-gallery), each illustrating a specific, cheap-to-make,
expensive-to-find mistake tied to a cardinal rule (e.g. `AP-01` mutating a `chunk_id`, `AP-03` an
unbounded agent loop). Referenced by id in reviews and in this glossary's other entries.

**The reference box** — The hardware every latency and memory budget in the doc suite is measured
against: an 8-core x86-64 CPU, 16 GB RAM, no GPU, SSD, running `python:3.11-slim-bookworm`. Locked in
`_CONTRACT.md §7`; the measurement discipline (why a developer laptop's number is never quoted as a
budget result) is in [TestPlan.md §5.2](TestPlan.md#52-reference-hardware).

---

## Project-specific proper nouns

**Axiom** — The project's current name, per [`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom).
Import root `src/axiom/`, environment-variable prefix `AXIOM_`. Chosen for continuity with an
earlier, unrelated hackathon project the same two founding members shipped under the same name —
see `ADR-015`'s "Alternatives considered."

**PRISM** — Two distinct things share this name in the doc suite; do not conflate them:

| Sense | What it refers to |
|---|---|
| Samsung PRISM GenAI Hackathon | The **event** this project is submitted to — "Samsung PRISM GenAI Hackathon 3rd Edition," `_CONTRACT.md §0`. This sense of "PRISM" never changes and is not renamed; the organiser-mandated release tag `PRISM_GENAI_HACKATHON_Y2026` uses it. |
| PRISM (former project name) | The project's **own name** before [`ADR-015`](Decisions.md#adr-015--rename-prism-to-axiom) renamed it to **Axiom**. Some file headers, the `Prism*` error class names (`PrismError` and its subclasses, [Rules.md §9.2](Rules.md#92-error-taxonomy)), and the `_CONTRACT.md §1` CLI entrypoint reference still carry this old sense as unpropagated residue — see [`OQ-04`](OpenQuestions.md#oq-04--project-name-collision-with-a-rival-submission)'s "Known residue" note and `T-201` in [Tracker.md](Tracker.md). |

When a document says "PRISM" without qualification, check which sense fits the sentence — the
hackathon it never stops being submitted to, or the project name it no longer has.

**Incognito** — The team name, `_CONTRACT.md §0`.

**`eval.yaml` / `demo.yaml`** — the two named config profiles; see **Profile** under
[Versioning and evolutionary retrieval terms](#versioning-and-evolutionary-retrieval-terms) above.

**CLI subcommand naming** — Some documents show `axiom query`, others (Setup.md's worked examples)
show `axiom search`. See [API.md](API.md) for the canonical CLI subcommand name; this glossary does
not adjudicate the discrepancy.

---

## Acronyms

| Acronym | Expansion | See |
|---|---|---|
| **AST** | Abstract Syntax Tree | tree-sitter's parsed representation of source code; chunk boundaries are cut at AST node boundaries. [Schema.md §3.3](Schema.md#33-chunkkind-semantics) |
| **BM25** | Best Matching 25 | The sparse lexical ranking function, implemented via `bm25s`. [Schema.md §3.2](Schema.md#32-signalkind-semantics), [Decisions.md `ADR-003`](Decisions.md#adr-003--bm25s-over-rank-bm25) |
| **RRF** | Reciprocal Rank Fusion | See [Retrieval and fusion terms](#retrieval-and-fusion-terms) above |
| **NDCG** | Normalised Discounted Cumulative Gain | Primary P0 accuracy metric, computed at cutoff 10 (`NDCG@10`). Formula below |
| **MRR** | Mean Reciprocal Rank | Secondary P0 accuracy metric. Formula below |
| **ONNX** | Open Neural Network Exchange | The model interchange format the embedder and reranker are exported to for CPU inference via `onnxruntime`. [Setup.md §6](Setup.md#6-onnx-export-and-int8-quantisation) |
| **INT8** | 8-bit integer (quantisation) | Post-training dynamic quantisation applied to the embedder and reranker for CPU throughput. `_CONTRACT.md §2` |
| **GGUF** | GPT-Generated Unified Format | The quantised weight format for the query LLM, run via `llama-cpp-python`. `_CONTRACT.md §2` |
| **ADR** | Architecture Decision Record | One entry in [Decisions.md](Decisions.md), format `ADR-###` |
| **STRIDE** | Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege | The threat-modelling framework structuring [Security.md](Security.md)'s threat table |
| **CI** | Continuous Integration | [TestPlan.md §8](TestPlan.md#8-ci-pipeline)'s GitHub Actions pipeline |
| **DDL** | Data Definition Language | The SQL `CREATE TABLE` statements defining `structural.sqlite`'s schema. [Schema.md §14](Schema.md#14-on-disk-formats) |
| **MoSCoW** | Must have, Should have, Could have, Won't have | The prioritisation framework structuring [PRD.md §7](PRD.md#7-prioritisation-moscow) |

### Identifier prefix scheme

The doc suite's own convention, first stated in `_CONTRACT.md §9` and
[README.md](README.md#conventions):

| Prefix | Meaning | Minted in |
|---|---|---|
| `FR-##` | Functional requirement | [PRD.md](PRD.md) |
| `NFR-##` | Non-functional requirement | [PRD.md](PRD.md) |
| `ADR-###` | Architecture decision record | [Decisions.md](Decisions.md) |
| `TC-###` | Test case | [TestPlan.md](TestPlan.md) |
| `RISK-##` | Risk register entry | [ImplementationPlan.md](ImplementationPlan.md) |
| `OQ-##` | Open question | [OpenQuestions.md](OpenQuestions.md) |
| `NG-##` | Non-goal | [NonGoals.md](NonGoals.md) |
| `T-###` | Tracker task | [Tracker.md](Tracker.md) |
| `M0`–`M7` | Milestone gate | [ImplementationPlan.md](ImplementationPlan.md) |

IDs within each scheme are permanent — never renumbered, never reused, per the "Changing this
fence"/"Changing this document" sections of the documents that mint them.

---

## Metric and scoring formulas

**NDCG@10 (Normalised Discounted Cumulative Gain at cutoff 10)** — The primary P0 accuracy metric,
computed by MTEB over the CoIR `AppsRetrieval` test split. For a ranked result list of length 10
with graded relevance `rel_i` at rank `i`:

```
DCG@10  = Σ_{i=1}^{10}  rel_i / log2(i + 1)
IDCG@10 = DCG@10 of the ideal ranking (relevant documents sorted by relevance, descending)
NDCG@10 = DCG@10 / IDCG@10
```

`NDCG@10 = 0` when no relevant document appears in the top 10; `NDCG@10 = 1` for a perfect ranking.
PRISM's target is `≥ 20.0` (reported on a 0–100 scale, i.e. `NDCG@10 × 100`), against a BGE-0.6B
baseline of `14.7` — see [PRD.md §2](PRD.md#2-goals-and-success-metrics) for the full leaderboard
context.

**MRR (Mean Reciprocal Rank)** — The secondary P0 metric: for each query, the reciprocal of the rank
of the first relevant result (`0` if none appears in the returned list), averaged over all queries:

```
MRR = (1/|Q|) * Σ_{q∈Q}  1 / rank_of_first_relevant_result(q)
```

PRISM's target is `≥ 22.0` (on the same 0–100 reporting scale). See
[PRD.md §2](PRD.md#2-goals-and-success-metrics).

**Recall@100** — The first-stage recall ceiling: the fraction of queries for which at least one
relevant document appears among the top 100 candidates *before* fusion and reranking narrow the
list. This is the ceiling the reranker works under — a relevant document absent from the top 100
cannot be recovered by any later stage. PRISM's target is `≥ 65.0`. See
[PRD.md §2](PRD.md#2-goals-and-success-metrics).

**RRF score** — See [Retrieval and fusion terms](#retrieval-and-fusion-terms) above; reproduced here
for completeness:

```
score(d) = Σ_i  w_i / (k + rank_i(d)),   k = 60
```

**Stability bonus** — See [Versioning and evolutionary retrieval terms](#versioning-and-evolutionary-retrieval-terms)
above; reproduced here for completeness:

```
final = base * (1 + 0.10 * stability)   [only when the family spans ≥ 2 versions]
```

---

## Related documents

| Document | Relationship |
|---|---|
| [Rules.md](Rules.md) | The five terms this glossary's `#project-specific-jargon` section is linked from |
| [Schema.md](Schema.md) | Canonical model definitions for every schema-backed term above |
| [_CONTRACT.md](_CONTRACT.md) | Source of every locked constant and formula reproduced above |
| [PRD.md](PRD.md) | Source of the metric targets and query-archetype examples above |
| [TechSpecifications.md](TechSpecifications.md) | Full algorithmic detail behind the terms in this glossary |
