# Tech Specifications

Component-by-component engineering spec for Axiom: runtime, model stack, per-module responsibilities, algorithm constants, and the config reference. The mechanics, not the rationale — see [Design.md](Design.md) for why the system is shaped this way.

**Owner:** Prabinder Singh
**Last updated:** 2026-09-16
**Status:** Draft

Related: [_CONTRACT.md](_CONTRACT.md) · [Schema.md](Schema.md) · [Design.md](Design.md) · [Appflow.md](Appflow.md) · [Rules.md](Rules.md) · [Setup.md](Setup.md) · [PRD.md](PRD.md) · [Decisions.md](Decisions.md) · [OpenQuestions.md](OpenQuestions.md) · [TestPlan.md](TestPlan.md)

---

## 1. Scope and authority

This document is the mechanical reference: what each module does, what it consumes and produces,
and the exact value of every algorithm constant. [Design.md](Design.md) owns *why* the system has
three signals and a bounded agent loop instead of some other shape; this document owns *how* each
piece computes what it computes. Where a constant here is a measured value, it cites the ADR that
locked it. Where a constant is still a design estimate, it says so explicitly and points at the
tracking `OQ-##`/`T-###` pair — per the `# PLACEHOLDER` convention in
[Rules.md §8](Rules.md#8-the-placeholder-convention), an unmarked placeholder in this document is a
defect.

Per [README.md](README.md#where-things-live--canonical-ownership): this document owns algorithm
constants (RRF `k`, candidate widths, weights, thresholds). [Setup.md](Setup.md) owns environment
variables and installation; §6 below only summarises the config surface and links back to
[Setup.md §7](Setup.md#7-environment-variables) as the authority on how to set each one.

This is the most cross-referenced document in the suite. [PRD.md](PRD.md), [Rules.md](Rules.md),
[NonGoals.md](NonGoals.md), [Schema.md](Schema.md), [Design.md](Design.md), [Appflow.md](Appflow.md),
and [Decisions.md](Decisions.md) all point here for "the exact constant" or "the exact algorithm."
If a number appears in two documents and disagrees, the number here — traced to `_CONTRACT.md`,
an `ADR-###`, or a `# PLACEHOLDER` marker with its `OQ-##` — wins, and the other document is wrong
and should be corrected in a PR.

---

## 2. Runtime and toolchain

Locked in `_CONTRACT.md §1`; restated here as the spec every module below is built against.

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 (3.11–3.12 supported) | 3.13 untested; see [NonGoals.md NG-21](NonGoals.md#ng-21--no-python-313-support) |
| Dependency manager | `uv` primary, `pip` + `requirements.txt` fallback | `uv.lock` committed and authoritative |
| Config | `pydantic-settings` v2 + YAML profiles in `configs/` | env prefix `AXIOM_` |
| CLI | `typer` | entrypoint `axiom` (see the naming note in §4.11 below) |
| API | `fastapi` + `uvicorn` | dev only, port 8000, unauthenticated by design ([NonGoals.md NG-08](NonGoals.md#ng-08--no-authentication-authorisation-or-multi-tenancy)) |
| Demo UI | `streamlit` | port 8501 |
| Lint/format | `ruff` (lint + format) | line length 100 |
| Types | `mypy` strict on `core/`, `retrieval/`, `schema/` | other packages non-strict, must not regress |
| Tests | `pytest` + `pytest-cov` | `tests/` mirrors `src/axiom/` |
| Container | Docker, `python:3.11-slim-bookworm` base | CPU-only |
| CI | GitHub Actions: ruff → mypy → pytest → smoke index | gate defined in [TestPlan.md §8](TestPlan.md#8-ci-pipeline) |

`uv.lock` is the reproducibility anchor referenced throughout this document:
[Rules.md §7](Rules.md#7-configuration-discipline)'s "a reported score must be reproducible from a
git SHA alone" is only true because the interpreter, the dependency graph, and every constant in §5
below are all pinned artefacts, not ambient state on someone's laptop.

---

## 3. Model stack

Locked in `_CONTRACT.md §2`. Every role has a declared fallback; the whole pipeline must run with
`AXIOM_LLM_ENABLED=false` per [NFR-07](PRD.md#6-non-functional-requirements). The CPU-only stack
choice (ONNX Runtime INT8 + llama.cpp GGUF) is justified in
[ADR-012](Decisions.md#adr-012--onnx-runtime-int8--llamacpp-gguf-for-a-cpu-only-stack).

| Role | Primary | Fallback | Runtime | Dim |
|---|---|---|---|---|
| Dense embedder | `Qwen/Qwen3-Embedding-0.6B` INT8 dynamic-quantised | `sentence-transformers/all-MiniLM-L6-v2` | ONNX Runtime CPU (`onnxruntime` ≥ 1.18) | 1024 / 384 |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` INT8 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ONNX Runtime CPU | — |
| Query LLM | `Qwen2.5-1.5B-Instruct` Q4_K_M GGUF via `llama-cpp-python` | heuristic rule engine (no LLM) | local, ≤ 512 tokens out | — |
| Sparse | `bm25s` (locked over `rank-bm25`, [ADR-003](Decisions.md#adr-003--bm25s-over-rank-bm25)) | — | numpy/scipy | — |
| Vector store | FAISS CPU | — | `IndexFlatIP` below 50k vectors, `IndexIVFPQ` at/above ([ADR-004](Decisions.md#adr-004--faiss-flat-below-50k-vectors-ivf-pq-at-or-above)) | — |
| AST | `tree-sitter` ≥ 0.22 + `tree-sitter-javascript` | regex identifier fallback | C | — |

Embeddings are L2-normalised at output; similarity is inner product, equal to cosine on normalised
vectors ([Rules.md Rule 4](Rules.md#rule-4--higher-is-better-lists-are-sorted-descending)).

**The query LLM is never used to read code.** Its four permitted jobs — classify, expand,
decompose, evaluate sufficiency — are enforced structurally by module boundary (§4.8) and locked as
[ADR-008](Decisions.md#adr-008--the-query-llm-never-reads-code). Sufficiency is judged from
`FusedResult.rerank_score`/`rrf_score` values only, never from prompting the LLM with chunk text.

`OQ-03` tracks whether Qwen3-Embedding-0.6B is in fact the strongest available embedder at this
parameter budget for the APPS task shape — see
[OpenQuestions.md OQ-03](OpenQuestions.md#oq-03--is-qwen3-embedding-06b-the-right-embedder-for-apps).
The model identity is asserted at query time against `VersionManifest.embedding_model` and
`embedding_dim` ([Schema.md §12](Schema.md#12-versionmanifest)); a mismatch is a hard
`PrismContractError`, never a silent fallback.

### 3.1 Dense embedder

**Used for:** encoding both chunk text (index time) and query text (query time) into a single
1024-dimensional (Qwen3) or 384-dimensional (MiniLM) L2-normalised vector, consumed by
`retrieval/dense.py`'s FAISS search ([Schema.md §7](Schema.md#7-scoredchunk),
`SignalKind.DENSE`). Batch size at index time is `AXIOM_EMBEDDING_BATCH_SIZE` (default 64);
query time always encodes one short sequence per (sub-)query.

**Fallback:** `sentence-transformers/all-MiniLM-L6-v2`, dim 384, set via `AXIOM_EMBEDDING_MODEL`.
Switching models changes `VersionManifest.embedding_dim`, which forces a cold rebuild into a new
`version_id` — blobs are model-scoped (§7.5) and a dimension mismatch on load is a hard
`PrismContractError` ([Rules.md AP-09](Rules.md#ap-09--cache-keyed-by-anything-other-than-content-rule-2)).

**Degradation trigger** (from [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s
ladder table, `axiom.retrieval.dense:search`): bad input is *"query embedding fails, index
missing"*; the ladder is *"embed + search → empty `list[ScoredChunk]`, signal marked absent"* — the
dense signal is dropped from fusion entirely, not zero-filled, and §5.1.2's renormalisation rule
redistributes its weight over the remaining signals.

### 3.2 Cross-encoder reranker

**Used for:** scoring the 25 fused candidates jointly with the query text, producing
`FusedResult.rerank_score` in `[0, 1]` ([Schema.md §8](Schema.md#8-fusedresult)). This is the
dominant latency cost in the pipeline — 620 ms of the 768 ms measured p50 (§8) — because it runs one
forward pass per (query, candidate) pair rather than one pass total.

**Fallback:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, set via `AXIOM_RERANKER_MODEL`.

**Degradation trigger** (`axiom.rerank.cross_encoder:rerank`): bad input is *"model load failure,
sequence too long"*; the ladder is *"cross-encode → truncate to model max length and cross-encode →
pass RRF order through unchanged (`rerank_score=None`)"*. Passthrough is a first-class, well-formed
outcome — `match_reason` records `"rerank_passthrough"` and `RetrievalResult.score` falls back to
the RRF score, never a synthesised value. A separate timeout ladder rung exists at the config layer:
`AXIOM_RERANKER_TIMEOUT_MS` (default 2500 ms) triggers the same passthrough with a
`RERANKER_TIMEOUT` warning if the whole batch exceeds its wall-clock budget
([Setup.md §7.2](Setup.md#72-models)).

### 3.3 Query LLM and its fallback: the heuristic rule engine

**Used for:** exactly four jobs, structurally incapable of a fifth — classify (`QueryType`), expand
(`expansion_terms`), decompose (`sub_queries`), and evaluate sufficiency. Runs locally via
`llama-cpp-python` against the GGUF weights, `temperature=0.0`, a fixed `seed`, and a fixed `n_ctx`
([Rules.md §5](Rules.md#5-determinism-and-seeding) item 4). Output is capped at
`AXIOM_LLM_MAX_TOKENS=512`.

**Fallback:** the heuristic rule engine — no model, no inference, pure Python regex and lookup
tables. `AXIOM_LLM_ENABLED=false` routes every one of the four jobs through the heuristic path
unconditionally, and this must be a first-class mode, not a crippled one
([NG-23](NonGoals.md#ng-23--no-llm-ingestion-of-retrieved-code)). The four heuristic implementations
below are what `agent/classifier.py`, `agent/planner.py`, and `agent/evaluator.py` fall back to per
[Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s ladder, and what they run
*always* when the LLM is disabled — not a smaller, worse LLM, a structurally different code path.

#### 3.3.1 Classification heuristic

Classifies free-text queries into one of the four `QueryType` variants
([Schema.md §3.1](Schema.md#31-querytype-semantics)) by scoring the query against three disjoint
cue-word/pattern sets and picking the single strongest signal, defaulting to `HYBRID` on a tie or
when nothing fires strongly — mirroring [Schema.md §3.1](Schema.md#31-querytype-semantics)'s own
rule that "classifier ambiguity always resolves to `HYBRID`" because a wrong confident
classification costs more NDCG than a correct-but-flat weighting.

| Cue set | Fires `QueryType` | Illustrative matches |
|---|---|---|
| Structural cue words | `STRUCTURAL` | `call(s)?`, `calls? before\|after`, `caller(s)? of`, `callee(s)? of`, `import(s)?`, `export(s)?`, `which files?`, `invoke(s)? before` |
| Usage cue shape | `USAGE` | a quoted literal is present (`'...'`, `"..."`, `` `...` ``), **or** the query contains exactly one extracted identifier ([§3.3.2](#332-identifier-extraction)) and no structural/semantic cue fires — "where is `X` used", "find usages of `Y`" |
| Semantic cue words | `SEMANTIC` | `how (is\|does\|do)`, `what happens when`, `preprocess`, `validate`, `handle`, `before going to`, and generally any query with no extracted identifier and no structural cue |
| No dominant cue, or ≥ 2 cue sets fire | `HYBRID` | compound queries mixing a structural and a semantic clause; anything under-specified |

Reference implementation shape (illustrative — the engineering detail this document exists to
pin down, not a numeric contract; the *fallback path itself*, i.e. "heuristic rule engine", is what
`_CONTRACT.md §2` locks):

```python
# src/axiom/agent/classifier.py (heuristic path)
_STRUCTURAL_RE = re.compile(
    r"\b(calls?|caller|callee|import(?:s|ed)?|export(?:s|ed)?|which files?|"
    r"invoke[sd]?|before|after)\b", re.IGNORECASE,
)
_SEMANTIC_RE = re.compile(
    r"\b(how (is|does|do)|what happens|preprocess|validate|handle|"
    r"going to|transform)\b", re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"['\"`]([^'\"`]+)['\"`]")

def classify_heuristic(query: str, identifiers: list[str]) -> QueryType:
    structural_hit = bool(_STRUCTURAL_RE.search(query))
    semantic_hit = bool(_SEMANTIC_RE.search(query))
    usage_hit = bool(_QUOTED_RE.search(query)) or len(identifiers) == 1
    hits = sum([structural_hit, semantic_hit, usage_hit])
    if hits != 1:
        return QueryType.HYBRID
    if structural_hit:
        return QueryType.STRUCTURAL
    if usage_hit:
        return QueryType.USAGE
    return QueryType.SEMANTIC
```

Validated against the three archetype queries in
[PRD.md §1.1](PRD.md#11-the-three-query-archetypes): Q1 ("How is the input preprocessed before
going to the main function?") fires the semantic cue set only → `SEMANTIC`; Q2 ("Which files call
tool XYZ before tool ABC?") fires the structural cue set → `STRUCTURAL`; Q3 ("Where is the
Bluetooth-settings deeplink used?") has no quoted literal but exactly one strong identifier
candidate (`bluetooth-settings` normalised) and no structural/semantic cue → `USAGE`. All three must
classify correctly with `AXIOM_LLM_ENABLED=false` per `US-1`'s acceptance criterion
([PRD.md §4](PRD.md#4-user-stories)).

#### 3.3.2 Identifier extraction

Per [`FR-02`](PRD.md#5-functional-requirements): extract candidate code identifiers from the raw
query text into `QueryPlan.extracted_identifiers`
([Schema.md §10](Schema.md#10-queryplan)), fed to the structural retriever as exact symbol lookups
and to the sparse retriever as high-precision terms. Four token shapes are recognised:

| Shape | Pattern (illustrative) | Example match |
|---|---|---|
| camelCase | `\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b` | `resolveTool`, `handleDeeplink` |
| snake_case | `\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b` | `parse_intent`, `tool_registry` |
| dotted path | `\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b` | `tools.registry.resolve` |
| quoted literal | `['"`][^'"`]+['"`]` | `"bluetooth-settings"` |
| call form | `\b[A-Za-z_][A-Za-z0-9_]*(?=\()` | `parseIntent(` → `parseIntent` |

Matches are deduplicated preserving first-occurrence order (`QueryPlan.extracted_identifiers`'
invariant, [Schema.md §10](Schema.md#10-queryplan)) and lightly normalised — a quoted literal like
`"bluetooth-settings"` is kept both as the literal (for sparse) and split on non-word boundaries
into identifier-shaped fragments (for structural symbol matching). This is the code path the
degradation ladder's `axiom.retrieval.structural:search` entry falls further back from when it
yields nothing: *"no identifiers extractable, `structural.sqlite` missing"* → *"symbol/call lookup →
identifier substring match → empty list"* ([Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)).

#### 3.3.3 Expansion terms — the code-synonym map

Per `FR-02`: produce `expansion_terms` from a curated code-synonym map, appended to the sparse
query only (never to the dense embedding input — dense already captures semantic neighbourhoods;
duplicating the expansion there would just blur the query vector). A representative slice of the
map (the full map is a static Python dict in `agent/planner.py`, not regenerated per query — this is
deterministic per [Rules.md Rule 2](Rules.md#rule-2--stages-are-pure)):

| Query term | Expansion terms |
|---|---|
| `preprocess` | `normalize`, `sanitize`, `transform`, `clean` |
| `validate` | `verify`, `check`, `assert` |
| `auth` | `authenticate`, `authorize`, `token` |
| `fetch` | `request`, `load`, `retrieve` |
| `dispatch` | `route`, `invoke`, `handle` |
| `parse` | `decode`, `deserialize`, `extract` |
| `error` | `exception`, `fail`, `reject` |

Expansion is a fixed lookup, not a generative step, so it is available identically whether the LLM
is enabled or not — the map is the same lookup table the LLM path is permitted to *supplement* (by
adding LLM-suggested synonyms) but never to replace, keeping `AXIOM_LLM_ENABLED=false` behaviourally
close to the LLM-on path rather than a strictly worse experience.

#### 3.3.4 Decomposition and sufficiency heuristics

Two further heuristic paths complete the agent's four jobs, both quoted directly from
[Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)'s ladder table:

| Job | Stage entrypoint | Heuristic behaviour |
|---|---|---|
| Decompose (`FR-03`) | `axiom.agent.planner:build_plan` | LLM plan → **single-sub-query plan containing the original query verbatim** — `QueryPlan.sub_queries` stays empty, and `QueryPlan.effective_queries` (Schema.md §10) therefore returns `[original_query]` |
| Evaluate sufficiency (`FR-13`) | `axiom.agent.evaluator:assess_sufficiency` | rerank-score predicate → **RRF-score predicate** → declare sufficient (never loop blindly) — see §5.3 for the exact thresholds |

Neither heuristic needs an LLM call, which is precisely why the agent loop's worst-case behaviour
(no models available at all) is still a single deterministic pass rather than a crash.

### 3.4 Sparse retrieval (bm25s)

**Used for:** exact-token and near-exact-token matching over a code-aware tokenisation of chunk
text ([Schema.md §3.2](Schema.md#32-signalkind-semantics)), the dominant signal for `USAGE`-type
queries (sparse weight 0.55, §5.1.1). Locked over `rank-bm25` in
[ADR-003](Decisions.md#adr-003--bm25s-over-rank-bm25) on throughput grounds at 10k-chunk scale.

**Fallback:** none at the library level — `bm25s` has no declared substitute model, only a
tokenisation fallback (§3.4.1). This is consistent with `_CONTRACT.md §2`'s stack table, which lists
no fallback row for sparse.

**Degradation trigger** (`axiom.retrieval.sparse:search`): bad input is *"tokeniser yields zero
tokens (e.g. query is all punctuation)"*; ladder is *"code-aware tokenise → whitespace tokenise →
empty list"* — worked in [Rules.md AP-11](Rules.md#ap-11--raising-on-malformed-query-input-rule-3).

#### 3.4.1 Code-aware tokenisation

The tokenizer (`axiom.indexing.sparse:CodeTokenizer`, persisted alongside the index as
`sparse.bm25s/prism_meta.json`, [Schema.md §14.6](Schema.md#146-sparsebm25s)) splits on
camelCase and snake_case boundaries *and* keeps the original intact identifier, so a query for
`resolveTool` matches both the split tokens `resolve`/`tool` and the exact identifier token. Dots
split module paths. Lowercasing is applied uniformly to query and corpus so case differences never
cost a match. No stopword list — code identifier vocabulary does not behave like natural-language
prose, and removing short tokens like `id`, `in`, `on` would delete real signal.

### 3.5 Vector store (FAISS)

**Used for:** approximate/exact nearest-neighbour search over dense embeddings. Index kind switches
on corpus size per [ADR-004](Decisions.md#adr-004--faiss-flat-below-50k-vectors-ivf-pq-at-or-above):
`IndexFlatIP` (exact inner product) below `AXIOM_FAISS_IVF_THRESHOLD=50000` vectors,
`IndexIVFPQ` (approximate) at or above. Both the 8,765-vector APPS corpus and the ~10k-chunk demo
repo sit on the flat side; `IndexIVFPQ` exists purely as scale headroom
([NG-18](NonGoals.md#ng-18--no-ann-index-on-the-benchmark-corpus)).

**Fallback:** none — FAISS is the sole vector store; there is no declared alternative library.
Degradation on a missing/corrupt index is handled one layer up, by `retrieval/dense.py`'s ladder
(§3.1), not by a FAISS-level fallback.

**Determinism:** `IndexIVFPQ` training is seeded (`AXIOM_SEED`); a retrained index is a *new index
kind instance* and must be logged in [Tracker.md](Tracker.md) with its eval delta before any score
built on it is reported ([Rules.md §5](Rules.md#5-determinism-and-seeding) item 3).

### 3.6 AST parsing (tree-sitter)

**Used for:** parsing JavaScript source into a concrete syntax tree that `chunking/ast_chunker.py`
walks to cut `Chunk` boundaries and that `indexing/structural.py` walks to extract symbols, calls,
imports, and exports (§4.4–4.5). Grammar: `tree-sitter-javascript` only —
[NG-04](NonGoals.md#ng-04--no-language-support-beyond-javascript-demo-and-python-benchmark) is
explicit that no other grammar ships this cycle; Python corpus documents (APPS) are indexed as
opaque text with no structural extraction.

**Fallback:** regex identifier fallback — when tree-sitter cannot parse a file at all (not just an
oversized function, a genuine parse failure), chunking degrades further down its own ladder (§4.4)
to a fixed-window line splitter, and structural extraction for that file contributes nothing rather
than a guess.

**Degradation trigger** (`axiom.chunking.ast_chunker:chunk_file`): bad input is *"tree-sitter parse
error, unsupported syntax"*; ladder is *"AST node boundaries → statement-boundary split →
`axiom.chunking.fallback:split_text` fixed-window line split with overlap → whole file as one
`ChunkKind.MODULE` chunk"*. A single file's parse failure never aborts the index
([NFR-07](PRD.md#6-non-functional-requirements)).

### 3.7 Degradation ladder — full table, quoted from Rules.md §3

Reproduced here in full (not summarised) because this document is the canonical algorithmic-detail
home the other docs defer to; [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)
states the *rule* that this table is mandatory and enforced at review, this document states the
*content* of every rung for engineering reference. Stage entrypoints use the canonical names from
[Appflow.md](Appflow.md).

| Stage | Entrypoint | Bad input | Degradation ladder (top = preferred) |
|---|---|---|---|
| Query normalisation | `axiom.agent.loop:run` | empty / whitespace / control chars | strip control chars → if empty, return an empty `RetrievalResult` list with `match_reason="empty_query"` |
| Query classifier | `axiom.agent.classifier:classify` | LLM unavailable, LLM returns non-enum text | LLM classification → heuristic rule engine (§3.3.1) → `QueryType.HYBRID` with equal weights |
| Query planner | `axiom.agent.planner:build_plan` | LLM timeout, malformed JSON plan | LLM plan → single-sub-query plan containing the original query verbatim (§3.3.4) |
| Dense retrieval | `axiom.retrieval.dense:search` | query embedding fails, index missing | embed + search → empty `list[ScoredChunk]`, signal marked absent |
| Sparse retrieval | `axiom.retrieval.sparse:search` | tokeniser yields zero tokens | code-aware tokenise → whitespace tokenise → empty list |
| Structural retrieval | `axiom.retrieval.structural:search` | no identifiers extractable, `structural.sqlite` missing | symbol/call lookup → identifier substring match → empty list |
| Chunker | `axiom.chunking.ast_chunker:chunk_file` | tree-sitter parse error, unsupported syntax | AST node boundaries → statement-boundary split → fixed-window line split with overlap → whole file as one `ChunkKind.MODULE` chunk |
| Fusion | `axiom.retrieval.fusion:reciprocal_rank_fusion` | one or more signal lists empty | RRF over the non-empty lists, renormalising nothing (RRF needs no normalisation) → if all empty, empty result |
| Reranker | `axiom.rerank.cross_encoder:rerank` | model load failure, sequence too long | cross-encode → truncate to model max length and cross-encode → pass RRF order through unchanged (`rerank_score=None`) |
| Agent loop | `axiom.agent.loop:run` | wall-clock budget exhausted, a pass raises | return best results seen so far, `passes_used` recorded |
| Sufficiency check | `axiom.agent.evaluator:assess_sufficiency` | scores absent (rerank passthrough) | rerank-score predicate → RRF-score predicate → declare sufficient (never loop blindly) |
| Incremental reindex | `axiom.versioning.incremental:reindex` | `git diff` fails, no git repo | `axiom.versioning.gitdiff:diff_versions` → file-hash comparison against the parent `VersionManifest` → full reindex |
| Evolutionary dedupe | `axiom.versioning.evolutionary:build_families` | version metadata missing | family grouping → identity families (one member each) |

Passthrough and empty-but-typed results are first-class outcomes, never errors — a consumer can
always tell degradation happened because the result shape stays well-formed
([Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)). The three categories where
raising *is* correct — `PrismContractError`, `PrismConfigError`, `PrismIndexError` — are programmer
error or contract violation, never bad query input; full taxonomy in
[Rules.md §9.2](Rules.md#92-error-taxonomy).

---

## 4. Per-module spec

Package layout locked in `_CONTRACT.md §3`. Each module's degradation ladder rung below is the
corresponding row of the table in §3.7; this section states the module's responsibility and its
place in the data flow, not a re-derivation of the ladder.

### 4.1 `config.py`

**Responsibility:** the single `Settings` object (`pydantic-settings` v2) that every stage reads its
configuration from — explicitly passed, never a module-level global read from `os.environ` in a hot
path ([Rules.md Rule 2](Rules.md#rule-2--stages-are-pure)). Loads a YAML profile from `configs/`
named by `AXIOM_PROFILE`, layered under CLI flag > env var > profile YAML > field default
([Rules.md §7](Rules.md#7-configuration-discipline)).

**Inputs:** `configs/*.yaml`, process environment, CLI flags. **Outputs:** a frozen `Settings`
instance. **Degradation:** none — malformed settings raise `PrismConfigError` at startup, one of the
three categories where raising is correct ([Rules.md Rule 3](Rules.md#rule-3--never-raise-on-bad-input-degrade)).

### 4.2 `schema/`

**Responsibility:** the shared Pydantic data contract every other module codes against. Full
specification in [Schema.md](Schema.md); not restated here. Base model configuration
(`extra="forbid"`, `frozen=True`) is justified in
[ADR-014](Decisions.md#adr-014--frozen-extraforbid-pydantic-models-as-the-schema-base).

### 4.3 `core/`

**Responsibility:** cross-cutting primitives with no retrieval-domain logic of their own: logging
factory ([Rules.md §9.1](Rules.md#91-logging-discipline)), the `stage_timer` context manager, the
error taxonomy ([Rules.md §9.2](Rules.md#92-error-taxonomy)), and hashing
(`compute_chunk_id`, `compute_content_hash`, `compute_family_id`,
[Schema.md §13](Schema.md#13-identity-and-hashing)). `core/ranking.py` holds the single canonical
`rank_key` function every ranker uses ([Rules.md Rule 4](Rules.md#rule-4--higher-is-better-lists-are-sorted-descending)).

**Degradation:** N/A — `core/` never touches external input; a bug here is a `PrismContractError`
by construction.

### 4.4 `chunking/`

**Responsibility:** turn source files into `Chunk` records on AST node boundaries. Owns the
64–512-token target and 16-token merge floor (§5.4).

**Inputs:** file bytes + `file_path`. **Outputs:** `list[Chunk]` ([Schema.md §6](Schema.md#6-chunk)).

**Degradation ladder** — §3.7 row "Chunker": AST node boundaries → statement-boundary split →
`chunking/fallback.py` fixed-window line split with overlap → whole file as one `ChunkKind.MODULE`
chunk. Never aborts the index on a single file's parse failure
([NFR-07](PRD.md#6-non-functional-requirements)).

#### 4.4.1 The AST-boundary chunking algorithm

`chunking/ast_chunker.py` walks the tree-sitter parse tree top-down and classifies each candidate
node into one of the five `ChunkKind` variants per
[Schema.md §3.3](Schema.md#33-chunkkind-semantics)'s tree-sitter-origin mapping:

| `ChunkKind` | tree-sitter origin | Emission rule |
|---|---|---|
| `FUNCTION` | `function_declaration`, `function_expression`, `arrow_function`, `generator_function_declaration` | one chunk per top-level or nested standalone function; `parent_symbol` is `None` unless lexically nested |
| `METHOD` | `method_definition` inside `class_body` | one chunk per method; `parent_symbol` is always the owning class name |
| `CLASS` | `class_declaration`, `class_expression` | emitted for the class header + fields, *and* — when every method fits under the 512-token target — a second whole-class chunk so class-level queries can match |
| `MODULE` | whole-file node | only for files with no extractable top-level definitions (config objects, barrel re-export files, pure side-effect scripts), and the terminal fallback rung when nothing else parses |
| `BLOCK` | oversized-function split fragments, residual top-level statement runs | the only kind whose boundaries are not a single AST node; `symbol` carries the parent function name with a `#<n>` suffix for split fragments |

The algorithm, stage by stage:

1. **Parse.** `tree-sitter-javascript` parses the file into a CST. A parse error at this stage
   triggers the first ladder rung down (§4.4).
2. **Walk and classify.** Top-level and class-body nodes are visited; each is mapped to a
   `ChunkKind` via the table above. A node whose token count (measured on the embedder's own
   tokenizer, so the count that matters is the one the model will actually see) falls inside
   `AXIOM_CHUNK_TARGET_TOKENS`'s 64–512 range is emitted as one chunk directly.
3. **Oversized-function splitting.** A `FUNCTION`/`METHOD` node whose token count exceeds 512 is
   split at **statement boundaries** — never mid-statement, so a chunk boundary is never inside an
   expression — into consecutive `BLOCK` fragments, each targeting the same 64–512 range. Consecutive
   fragments carry a **1-statement overlap**: the last statement of fragment *n* is repeated as the
   first statement of fragment *n+1*. This exists so a query whose relevant code straddles a split
   point still has one fragment containing it whole, at the cost of duplicating that one statement's
   tokens into two chunks (an accepted, bounded redundancy — never more than one statement per
   split, never compounding across more than two adjacent fragments).
4. **Sub-floor merge.** Any chunk (from step 2 or step 3) under `AXIOM_CHUNK_MIN_TOKENS=16` tokens
   is merged into its parent node's chunk rather than emitted standalone — a one-line arrow function
   assigned to an object property, for instance, does not get its own low-information chunk.
5. **Residual statements.** Top-level statements that belong to no function/class/module
   declaration (bare side-effecting code, e.g. `registerAgent(config)` at module scope) are grouped
   into `BLOCK` chunks by contiguous run, same 64–512 target.
6. **Location and identity.** Each emitted chunk gets its `ChunkLocation` (line + byte span,
   [Schema.md §4](Schema.md#4-chunklocation)) directly from the tree-sitter node's own span (or the
   computed span for a split/merged fragment), and its `chunk_id`/`content_hash` from
   `core/hashing.py` per [Schema.md §13](Schema.md#13-identity-and-hashing) — chunking is the *only*
   place either identity is minted ([Rules.md Rule 1](Rules.md#rule-1--ids-are-sacred)).

**Both numeric bounds are `# PLACEHOLDER`.** The 64–512 token target and the 16-token merge floor
are design estimates, not measurements, flagged identically in
[Rules.md §8](Rules.md#8-the-placeholder-convention) and tracked as
[`OQ-09`](OpenQuestions.md#oq-09--are-the-chunking-placeholders-64512-tokens-16-token-floor-right),
measured by `T-070` against the `OQ-07` demo repo's real token-length distribution. The 1-statement
overlap constant is likewise unmeasured and carries the same marker. See §5.4 for the `Settings`
field table.

### 4.5 `indexing/`

Four builders, each consuming `list[Chunk]` and writing to the on-disk layout in §7 /
[Schema.md §14](Schema.md#14-on-disk-formats) / `_CONTRACT.md §6`.

| Module | Responsibility | Writes |
|---|---|---|
| `dense.py` | Embed every chunk, L2-normalise, build the FAISS index | `dense.faiss`, `dense.idmap.json` |
| `sparse.py` | Build the `bm25s` index over the identical chunk corpus | `sparse.bm25s/` |
| `structural.py` | Extract symbols, calls, imports, exports into SQLite | `structural.sqlite` |
| `manifest.py` | Write and validate `VersionManifest`; refuse a load when model identity or dim disagrees | `manifest.json` |

`manifest.py` is also where the content-addressed embedding cache is enforced — the one sanctioned
exception to stage purity ([Rules.md Rule 2](Rules.md#rule-2--stages-are-pure)): a cache entry is
invalidated by model identity, never by time, and a dimension mismatch is a hard
`PrismContractError` (see [Rules.md AP-09](Rules.md#ap-09--cache-keyed-by-anything-other-than-content-rule-2)).

`structural.py` builds the four-relation SQLite schema locked by
[ADR-011](Decisions.md#adr-011--sqlite-for-the-structural-index-not-a-graph-database): `symbols`
(name, kind, file, line span, scope), `calls` (caller → callee, source order), `imports`
(file → module), `exports` (module → symbol). The `calls` relation's source-order column is what
answers query archetype Q2 ("which files call XYZ before ABC") — it is a self-join on
`(caller, source_order)`, not a graph traversal ([NG-27](NonGoals.md#ng-27--no-graph-database-for-the-structural-index)).

### 4.6 `retrieval/`

Three independent signal retrievers plus fusion. Each retriever reads from a built index only — it
never re-chunks or re-embeds on the read path ([Rules.md AP-12](Rules.md#ap-12--re-chunking-or-re-embedding-on-the-read-path-rule-2-96)).

| Module | Reads | Candidate width | Degradation ladder |
|---|---|---|---|
| `dense.py` | `dense.faiss` + idmap | K=100 | embed + search → empty `list[ScoredChunk]`, signal marked absent |
| `sparse.py` | `sparse.bm25s/` | K=100 | code-aware tokenise → whitespace tokenise → empty list |
| `structural.py` | `structural.sqlite` | K=50 | symbol/call lookup → identifier substring match → empty list |
| `fusion.py` | the three `list[ScoredChunk]` above | → N=25 | RRF over non-empty lists (RRF needs no normalisation) → all-empty yields `[]` |

§5.1–§5.2 below give the exact fusion arithmetic and candidate-width chain.

### 4.7 `rerank/`

**Responsibility:** `cross_encoder.py` scores the top-25 fused candidates with the locked
cross-encoder, writes `rerank_score`, re-sorts, truncates to top-10. Lazily loads its ONNX session
([Rules.md AP-06](Rules.md#ap-06--loading-models-at-import-time)) so `axiom --help` and
`pytest --collect-only` never pay model-load cost.

**Degradation:** cross-encode → truncate to `AXIOM_RERANK_MAX_CHARS` and cross-encode → pass the RRF
order through unchanged with `rerank_score=None` ("rerank passthrough" is a first-class, well-formed
outcome, not an error — [Rules.md §3](Rules.md#rule-3--never-raise-on-bad-input-degrade)).

### 4.8 `agent/`

Five modules, none of which reads chunk text into the LLM (§3.3, [ADR-008](Decisions.md#adr-008--the-query-llm-never-reads-code)).

| Module | Responsibility | Degradation ladder |
|---|---|---|
| `classifier.py` | Classify into `QueryType` (§3.3.1) | LLM classification → heuristic rule engine (regexes) → `HYBRID` with equal weights |
| `planner.py` | Identifier extraction (§3.3.2), expansion (§3.3.3), decomposition (§3.3.4) into `QueryPlan` | LLM plan → single-sub-query plan containing the original query verbatim |
| `evaluator.py` | Sufficiency predicate over `FusedResult` scores (§3.3.4, §5.3) | rerank-score predicate → RRF-score predicate → declare sufficient (never loop blindly) |
| `loop.py` | Orchestrates classify → plan → retrieve → fuse → rerank → evaluate → (refine) | budget exhausted or a pass raises → return best results seen so far, `passes_used` recorded |
| `llm.py` | `llama-cpp-python` adapter, `temperature=0.0`, fixed seed and `n_ctx` | LLM unavailable → the four heuristic paths above take over per-caller |

§5.3 gives the exact loop bound and sufficiency trigger.

### 4.9 `versioning/`

| Module | Responsibility | Degradation ladder |
|---|---|---|
| `gitdiff.py` | `git diff --name-status <old>..<new>` → {A,M,D,R} | `git diff` fails / no repo → file-hash comparison against the parent `VersionManifest` → full reindex |
| `incremental.py` | Re-chunk/re-embed only A and M files; drop D chunks; reuse blobs by `content_hash` | same ladder as `gitdiff.py`, one level up the call stack |
| `evolutionary.py` | Cosine-≥-0.95 grouping into `SnippetFamily`, stability bonus | version metadata missing → identity families (one member each) |

`gitdiff.py` is the sole versioning source of truth per
[ADR-010](Decisions.md#adr-010--git-diff-as-the-sole-versioning-source-of-truth); see §5.5–§5.6 for
the exact algorithms.

### 4.10 `eval/`

**Responsibility:** `mteb_adapter.py` wraps the full pipeline as an MTEB `AbsEncoder`; `metrics.py`
computes NDCG@10/MRR/Recall@100. The adapter is the one place in the codebase where raising on a
contract violation is mandatory and unconditional: every emitted id is asserted to be a member of
the corpus id set *before* MTEB sees it, raising `PrismContractError` otherwise
([Rules.md Rule 1](Rules.md#rule-1--ids-are-sacred)). See [Setup.md §4.4](Setup.md#44-mteb-v2-is-required--not-v1)
for the exact MTEB v2 API surface this module targets.

### 4.11 `api/`, `ui/`, `cli.py`

**Responsibility:** three thin presentation layers over the same Pydantic models
([Schema.md §9](Schema.md#9-retrievalresult)). `api/app.py` + `routes.py` expose
`POST /query`, `GET /versions`, `GET /chunk/{chunk_id}`, `GET /health`
([FR-24](PRD.md#5-functional-requirements)). `ui/streamlit_app.py` renders result cards with
per-signal rank breakdown and the agent-pass indicator ([FR-25](PRD.md#5-functional-requirements)).
`cli.py` is the `typer` entrypoint with subcommands `index`, `reindex`, `query`, `classify`,
`versions`, `families`, `eval`, `serve`, `ui` — every subcommand supports `--json`
([FR-23](PRD.md#5-functional-requirements)).

**CLI naming note:** `_CONTRACT.md §1` names the entrypoint `prism`; [Setup.md](Setup.md) and
[TestPlan.md](TestPlan.md) already invoke it as `axiom`. This document uses `axiom`, consistent with
the majority of already-written docs and with the package import root `src/axiom/`. The
inconsistency is tracked and will be fully resolved by `T-201`
([ADR-015](Decisions.md#adr-015--rename-prism-to-axiom)); do not "fix" it here by editing other
files.

**Degradation:** none of the three layers degrades on its own — they surface whatever the pipeline
underneath already degraded to, and report which fallback was active
([NFR-10](PRD.md#6-non-functional-requirements)).

---

## 5. Algorithm specifications

All constants below are locked in `_CONTRACT.md §5` unless marked `PLACEHOLDER`, per
[Rules.md §8](Rules.md#8-the-placeholder-convention). A `PLACEHOLDER` value may not appear in a
reported score until its tracked measurement closes.

### 5.1 Reciprocal Rank Fusion

```
score(d) = Σ_i  w_i / (k + rank_i(d))          k = 60 (locked, AXIOM_RRF_K)
```

Rank-space, not score-space — see [ADR-002](Decisions.md#adr-002--weighted-reciprocal-rank-fusion-over-score-space-fusion)
for why. `w_i` comes from `QueryPlan.strategy_weights` ([Schema.md §10](Schema.md#10-queryplan)),
renormalised over whichever signals actually returned candidates for this query (§5.1.2). Reference
arithmetic worked in [TestPlan.md TC-049](TestPlan.md#36-category-f--fusion-tc-049--tc-058) and
`PROJECT_OVERVIEW.md` Appendix B.

#### 5.1.1 Default `strategy_weights` per `QueryType`

Order in each triple: `(dense, sparse, structural)`. Locked in `_CONTRACT.md §5` and
[Schema.md §10](Schema.md#10-queryplan); consumed by `retrieval/fusion.py`.

| `QueryType` | dense | sparse | structural | Dominant signal |
|---|---|---|---|---|
| `SEMANTIC` | 0.60 | 0.30 | 0.10 | dense |
| `STRUCTURAL` | 0.20 | 0.20 | 0.60 | structural |
| `USAGE` | 0.25 | 0.55 | 0.20 | sparse |
| `HYBRID` | 0.34 | 0.33 | 0.33 | none (flat) |

Every row sums to 1.0 ± 1e-9 ([TestPlan.md TC-006](TestPlan.md#31-category-a--query-preprocessing-tc-001--tc-011)).
On the `eval` profile these are further overridden by the eval-specific weights in §5.1.3 — the
table above is the `demo` profile's weight set, the one that actually runs the structural signal.

#### 5.1.2 Empty-signal renormalisation

When a signal returns zero candidates (e.g. structural finds no identifier to resolve), its weight
is dropped and the remaining weights are divided by their new sum before fusion runs — the logged
`QueryPlan` reflects the renormalised vector, not the nominal one
([Schema.md §10](Schema.md#10-queryplan), [TestPlan.md TC-055](TestPlan.md#36-category-f--fusion-tc-049--tc-058)).
Example: `SEMANTIC` weights with structural empty become
`{dense 0.6/0.9 = 0.6667, sparse 0.3/0.9 = 0.3333}`.

#### 5.1.3 Eval profile weights

`configs/eval.yaml` runs structural indexing off entirely (not merely weighted to zero — the
structural build step is skipped, since APPS documents have no cross-file structure to index,
[PRD.md §2.2](PRD.md#22-two-evaluation-contexts-two-profiles)). Effective weights:
`{dense 0.85, sparse 0.15, struct 0.0}`. The `0.15` sparse weight is a `PLACEHOLDER` — see
[`OQ-02`](OpenQuestions.md#oq-02--what-is-the-exact-sparse-weight-in-configseval-yaml), swept by
`T-112`.

### 5.2 Candidate width chain

Locked in `_CONTRACT.md §5`, restated in [PRD.md §8](PRD.md#8-jury-scoring-alignment) as the
"declared candidate widths" technical-depth claim.

```
dense K=100 ─┐
sparse K=100 ─┼─→ RRF fusion ─→ top N=25 ─→ cross-encoder rerank ─→ top 10
structural K=50 ┘
```

| Stage | Width | Env var |
|---|---|---|
| Dense candidate width | 100 | `AXIOM_DENSE_TOP_K` |
| Sparse candidate width | 100 | `AXIOM_SPARSE_TOP_K` |
| Structural candidate width | 50 | `AXIOM_STRUCTURAL_TOP_K` |
| Post-fusion, pre-rerank | 25 | `AXIOM_FUSION_TOP_N` |
| Final result count (default) | 10 | `AXIOM_TOP_K_DEFAULT` |

Widths are `Settings` fields, never hardcoded at a callsite
([Rules.md AP-07](Rules.md#ap-07--hardcoded-tunables-at-the-callsite-rule-7)).

### 5.3 Agent loop bound

```
deadline = monotonic() + AXIOM_AGENT_WALL_CLOCK_MS / 1000     # 5.0 s
for pass_no in 1..AXIOM_AGENT_MAX_PASSES:                     # 2
    if sufficient(best) or monotonic() >= deadline: break
    plan = refine(plan, best)
    candidate = retrieve(plan)
    if top1(candidate) > top1(best): best = candidate
return best
```

Reference implementation in [Rules.md AP-03](Rules.md#ap-03--unbounded-agent-loop-rule-3-contract-5);
bound justified in [ADR-007](Decisions.md#adr-007--bounded-agent-loop-hard-caps-not-convergence).
Termination on adversarial input is `TestPlan.md TC-086`, a P0 test.

**Sufficiency trigger** (`agent/evaluator.py`): refine when top-1 `rerank_score < 0.35` **or** fewer
than 3 results score above `0.20`. Both thresholds are `PLACEHOLDER` — this is the exact worked
example in [Rules.md §8](Rules.md#8-the-placeholder-convention) and
[Rules.md AP-14](Rules.md#ap-14--reporting-a-number-built-on-a-placeholder-8). Tracked as
[`OQ-10`](OpenQuestions.md#oq-10--are-the-agent-sufficiency-thresholds-035020-right), swept by
`T-141` ([Tracker.md](Tracker.md#t-141)). The evaluator's fallback rung (§3.3.4, §3.7) — RRF-score
predicate instead of rerank-score predicate — applies the *same* two thresholds against `rrf_score`
rather than `rerank_score` when reranking has degraded to passthrough; the thresholds are shared,
only the score field changes.

| Constant | Value | Env var | Status |
|---|---|---|---|
| Max refinement passes | 2 | `AXIOM_AGENT_MAX_PASSES` | Locked |
| Wall-clock budget | 5000 ms | `AXIOM_AGENT_WALL_CLOCK_MS` | Locked |
| Sufficiency top-1 threshold | 0.35 | `AXIOM_AGENT_SUFFICIENCY_TOP1` | `PLACEHOLDER`, `OQ-10` |
| Sufficiency floor | 0.20 | `AXIOM_AGENT_SUFFICIENCY_FLOOR` | `PLACEHOLDER`, `OQ-10` |
| Sufficiency min results above floor | 3 | `AXIOM_AGENT_SUFFICIENCY_MIN_RESULTS` | Locked |

### 5.4 Chunking targets

AST node boundaries define chunk edges (full algorithm in §4.4.1). Oversized functions split at
statement boundaries with a 1-statement overlap between consecutive fragments; chunks under the
merge floor absorb into their parent. All three numeric bounds are `PLACEHOLDER`, tracked as
[`OQ-09`](OpenQuestions.md#oq-09--are-the-chunking-placeholders-64512-tokens-16-token-floor-right),
measured by `T-070` against the `OQ-07` demo repo's real token-length distribution, cross-checked
against the reranker's `~4096`-char (≈1000–1300 token) truncation window
([TestPlan.md TC-063](TestPlan.md#37-category-g--reranking-tc-059--tc-064)) so the 512-token upper
chunk bound stays comfortably inside what the reranker will actually see whole.

| Constant | Value | Env var | Status |
|---|---|---|---|
| Target chunk size | 64–512 tokens | `AXIOM_CHUNK_TARGET_TOKENS` (upper bound) | `PLACEHOLDER`, `OQ-09` |
| Merge floor | 16 tokens | `AXIOM_CHUNK_MIN_TOKENS` | `PLACEHOLDER`, `OQ-09` |
| Oversize split overlap | 1 statement | — (chunker constant) | `PLACEHOLDER`, `OQ-09` |

### 5.5 Incremental reindex

```
{A, M, D, R} = git diff --name-status <old_commit>..<new_commit>
for f in A ∪ M:
    chunks = ast_chunk(f)
    for c in chunks:
        if blob_exists(c.content_hash): skip embed   # zero-cost reuse
        else: embed(c); write_blob(c.content_hash)
drop all chunks whose file_path ∈ D
for (old_path, new_path) in R:
    # chunk_id changes (path changed), content_hash unchanged → zero new embeddings
```

Locked in `_CONTRACT.md §5`, `FR-18`/`FR-19`. Byte-equivalence with a full rebuild of the same tree
is `TestPlan.md TC-076`; zero-embedding renames are `TC-075`. Source-of-truth choice justified in
[ADR-010](Decisions.md#adr-010--git-diff-as-the-sole-versioning-source-of-truth). Budget:
`NFR-02`, ≤ 45 s for a 50-changed-file diff — see `T-163` in [Tracker.md](Tracker.md).

When `git diff` itself fails or the target is not a git repository, `gitdiff.py` degrades to a
file-hash comparison against the parent `VersionManifest.file_hashes`
([Schema.md §12](Schema.md#12-versionmanifest)) — every file whose stored hash disagrees with its
current content hash is treated as Modified, every path present now but absent from the parent
manifest as Added, and the reverse as Deleted; renames are not detectable by this fallback (a
rename looks like a D+A pair), so it costs the zero-embedding rename property but still bounds the
diff to changed content rather than forcing a full reindex. If even that fails, the ladder's final
rung is a full reindex ([§3.7](#37-degradation-ladder--full-table-quoted-from-rulesmd-3)).

### 5.6 Evolutionary dedupe

```
family_key(c) = (c.metadata.symbol, c.location.file_path)
group chunks by family_key where pairwise cosine(embedding_i, embedding_j) >= 0.95
representative = newest member (members sorted newest → oldest)
stability = len(members) / total_indexed_versions
bonus = 1 + 0.10 * stability   if len(versions) >= 2   else 1.0
final_score = clamp(base_score * bonus, 0.0, 1.0)
```

Both `0.95` (dedupe cosine) and `0.10` (stability bonus) are `PLACEHOLDER`, tracked as
[`OQ-11`](OpenQuestions.md#oq-11--are-the-evolutionary-dedupe-cosine-095-and-stability-bonus-010-right),
verified qualitatively by `T-182` against the `OQ-07` demo repo's real version history (there is no
labelled ground truth for "same logical snippet," so this is a manual check, not a metric sweep).
Grouping requires matching `symbol` **and** `file_path`, not similarity alone
([TestPlan.md TC-082](TestPlan.md#310-category-j--evolutionary-retrieval-tc-081--tc-085)). Full
model in [Schema.md §11](Schema.md#11-snippetfamily). The bonus only applies when the family spans
`>= 2` versions (`SnippetFamily.is_multi_version`); a single-version family's `stability` is
`1/total_versions`, which is intentionally *low*, not high — a brand-new snippet is novel, not
unstable, and the multi-version gate keeps the single-version case exactly neutral
([Schema.md §11](Schema.md#11-snippetfamily)). Worked example: four indexed versions, three
members → `stability = 0.75`, bonus `= 1 + 0.10*0.75 = 1.075`, a base rerank score of 0.81 becomes
`0.871` after clamping to `[0, 1]`.

---

## 6. Config reference

[Setup.md §7](Setup.md#7-environment-variables) is the authority on every `AXIOM_*` environment
variable — installation, defaults, `.env.example`. This section indexes which config *group* each
algorithm constant above belongs to, so a reader who wants "everything that controls fusion" does
not have to scan the whole Setup.md table, and gives the subset that is algorithmically interesting
with its consuming module.

| Group | Setup.md section | Constants this document specifies |
|---|---|---|
| Core | [§7.1](Setup.md#71-core) | `AXIOM_SEED` (determinism, [Rules.md §5](Rules.md#5-determinism-and-seeding)), `AXIOM_NUM_THREADS` |
| Models | [§7.2](Setup.md#72-models) | embedder/reranker identity, `AXIOM_EMBEDDING_DIM` (§3) |
| Agent and LLM | [§7.3](Setup.md#73-agent-and-llm) | agent loop bound and sufficiency trigger (§5.3) |
| Retrieval and fusion | [§7.4](Setup.md#74-retrieval-and-fusion) | RRF `k`, candidate widths (§5.1–§5.2), chunk targets (§5.4) |
| Versioning and evolutionary | [§7.5](Setup.md#75-versioning-and-evolutionary) | dedupe cosine, stability bonus (§5.6) |
| Services | [§7.6](Setup.md#76-services) | API/UI ports — no algorithm constants |

The algorithmically-interesting subset, one row per constant, with the module that actually reads
it at runtime (as opposed to the doc section that documents it):

| Constant | Env var | Value | Consuming module |
|---|---|---|---|
| RRF constant | `AXIOM_RRF_K` | 60 | `retrieval/fusion.py` |
| Dense candidate width | `AXIOM_DENSE_TOP_K` | 100 | `retrieval/dense.py` |
| Sparse candidate width | `AXIOM_SPARSE_TOP_K` | 100 | `retrieval/sparse.py` |
| Structural candidate width | `AXIOM_STRUCTURAL_TOP_K` | 50 | `retrieval/structural.py` |
| Post-fusion width | `AXIOM_FUSION_TOP_N` | 25 | `retrieval/fusion.py` |
| Default result count | `AXIOM_TOP_K_DEFAULT` | 10 | `rerank/cross_encoder.py`, `api/routes.py` |
| FAISS index-kind threshold | `AXIOM_FAISS_IVF_THRESHOLD` | 50000 | `indexing/dense.py` |
| Chunk merge floor | `AXIOM_CHUNK_MIN_TOKENS` | 16 (`PLACEHOLDER`) | `chunking/ast_chunker.py` |
| Chunk target upper bound | `AXIOM_CHUNK_TARGET_TOKENS` | 512 (`PLACEHOLDER`) | `chunking/ast_chunker.py` |
| Agent max passes | `AXIOM_AGENT_MAX_PASSES` | 2 | `agent/loop.py` |
| Agent wall-clock budget | `AXIOM_AGENT_WALL_CLOCK_MS` | 5000 | `agent/loop.py` |
| Sufficiency top-1 threshold | `AXIOM_AGENT_SUFFICIENCY_TOP1` | 0.35 (`PLACEHOLDER`) | `agent/evaluator.py` |
| Sufficiency floor | `AXIOM_AGENT_SUFFICIENCY_FLOOR` | 0.20 (`PLACEHOLDER`) | `agent/evaluator.py` |
| Sufficiency min results | `AXIOM_AGENT_SUFFICIENCY_MIN_RESULTS` | 3 | `agent/evaluator.py` |
| Evolutionary dedupe cosine | `AXIOM_DEDUPE_COSINE` | 0.95 (`PLACEHOLDER`) | `versioning/evolutionary.py` |
| Stability bonus multiplier | `AXIOM_STABILITY_BONUS` | 0.10 (`PLACEHOLDER`) | `versioning/evolutionary.py` |
| Reranker enabled | `AXIOM_RERANKER_ENABLED` | true | `rerank/cross_encoder.py` |
| Reranker timeout | `AXIOM_RERANKER_TIMEOUT_MS` | 2500 | `rerank/cross_encoder.py` |
| LLM enabled | `AXIOM_LLM_ENABLED` | true | `agent/llm.py`, gates §3.3's fallback |
| Evolutionary retrieval enabled | `AXIOM_EVOLUTIONARY_ENABLED` | false | `versioning/evolutionary.py`, `retrieval/*` |

Precedence for every value: CLI flag > env var (`AXIOM_` prefix) > profile YAML > `Settings` field
default ([Rules.md §7](Rules.md#7-configuration-discipline)). No magic numbers at a callsite —
every constant named in §5 is a `Settings` field
([Rules.md AP-07](Rules.md#ap-07--hardcoded-tunables-at-the-callsite-rule-7)).

---

## 7. On-disk index layout

Locked in `_CONTRACT.md §6` and mirrored in full detail in
[Schema.md §14](Schema.md#14-on-disk-formats); this section restates the tree for convenience and
does not diverge from either source.

```
.axiom/
├── registry.json                     # version_id -> manifest path, active version
├── blobs/<content_hash>.npy          # shared embeddings, deduped across versions
└── index/<version_id>/
    ├── manifest.json                 # VersionManifest
    ├── chunks.jsonl                  # one Chunk per line
    ├── dense.faiss
    ├── dense.idmap.json              # faiss row -> chunk_id
    ├── sparse.bm25s/                 # bm25s native dir
    └── structural.sqlite             # symbols, calls, imports, exports tables
```

| Path | Written by | Mutability | Notes |
|---|---|---|---|
| `registry.json` | `versioning/` (registry update on every successful build) | mutable, rewritten atomically | the only mutable file in `.axiom/`; version_id → manifest path plus the active version |
| `blobs/<content_hash>.npy` | `indexing/dense.py` on a cache miss | write-once, immutable | one file per distinct `content_hash`; `float32`, L2-normalised, shape `(embedding_dim,)`; shared and never rewritten across versions or across the incremental/evolutionary paths |
| `index/<version_id>/manifest.json` | `indexing/manifest.py`, written last in the build | write-once | its presence is the build-completion marker; a directory with `chunks.jsonl` but no `manifest.json` is an aborted build |
| `index/<version_id>/chunks.jsonl` | `chunking/` output, persisted by the index builders | write-once | one `Chunk` per line, line order == FAISS row order |
| `index/<version_id>/dense.faiss` | `indexing/dense.py` | write-once | `IndexFlatIP` or `IndexIVFPQ` per `VersionManifest.index_kind` |
| `index/<version_id>/dense.idmap.json` | `indexing/dense.py` | write-once | FAISS integer row → `chunk_id`, positional array |
| `index/<version_id>/sparse.bm25s/` | `indexing/sparse.py` | write-once | native `bm25s` format plus `prism_meta.json` tokenizer record (§3.4.1) |
| `index/<version_id>/structural.sqlite` | `indexing/structural.py` | write-once | four relations: symbols, calls, imports, exports |

Every artefact under `index/<version_id>/` is write-once and content-addressed by construction — the
only mutable state in the whole tree is which version `registry.json` currently calls active. This
is what makes the incremental reindex algorithm (§5.5) and the evolutionary dedupe algorithm (§5.6)
both safely interruptible: a partially-written version directory without a `manifest.json` is
unambiguously incomplete and discarded on the next run, never half-trusted.

`.axiom/` in its entirety is gitignored — every artefact under it is derived, large, and
machine-specific ([Rules.md §9.5](Rules.md#95-what-may-and-may-not-be-committed)). Full per-file
format detail (byte layout, JSON shape, invariants) lives in
[Schema.md §14](Schema.md#14-on-disk-formats) and is not duplicated here.

---

## 8. Latency budget

The `NFR-03` p50 budget is ≤ 900 ms with the agent loop's extra passes excluded (that is `NFR-04`'s
job, ≤ 5 s p95). The canonical stage-by-stage accounting, first stated in
[NonGoals.md NG-10](NonGoals.md#ng-10--no-production-sla-uptime-or-ha-guarantee) and owned
mechanically here:

| Stage | ms | Notes |
|---|---|---|
| Classify | 60 | heuristic path; LLM path is slower and is why `AXIOM_LLM_ENABLED=false` is the faster of the two valid modes |
| Query embed | 35 | one short sequence, ONNX INT8 |
| Three signals (dense + sparse + structural), concurrent | 28 | not summed — the three retrievers run concurrently; this is the wall-clock cost of the slowest of the three |
| RRF fusion | 3 | pure arithmetic over ≤ 250 candidates, no normalisation pass |
| Hydrate (chunk_id → full `Chunk`) | 12 | `chunks.jsonl` lookup for the 25 fused candidates |
| Rerank, 25 pairs | 620 | the dominant cost; cross-encoder inference dominates the budget by more than 10x over every other stage combined |
| Format (`RetrievalResult` assembly) | 10 | template-based `match_reason`, no generation |
| **Total (measured p50)** | **768** | **132 ms headroom under the 900 ms `NFR-03` ceiling** |

The reranker is unambiguously the latency-critical stage — `AXIOM_RERANKER_ENABLED=false` is the
single largest lever for isolating whether a latency regression originates in fusion/retrieval or in
reranking ([Setup.md §7.2](Setup.md#72-models)). `NFR-10` requires every stage above to emit a
structured timing record via `core.stage_timer`, so this table is auditable from any single
`--json` response, not just asserted here.

An agent refinement pass repeats classify → plan → retrieve → fuse → hydrate → rerank in full
(§5.3), so a two-pass query's worst case is bounded by the 5000 ms wall clock, not by a simple
multiple of the 768 ms single-pass figure — the deadline check happens *before* starting a new pass,
so a pass already in flight when the deadline is crossed is allowed to finish rather than being cut
off mid-rerank batch, per the reference loop in
[Rules.md AP-03](Rules.md#ap-03--unbounded-agent-loop-rule-3-contract-5).

---

## 9. Related documents

| Document | Relationship |
|---|---|
| [Design.md](Design.md) | Architecture narrative and rationale this spec implements mechanically |
| [Schema.md](Schema.md) | Data model every module in §4 reads/writes |
| [Appflow.md](Appflow.md) | End-to-end sequence diagrams that thread through the modules and constants above |
| [Rules.md](Rules.md) | Degradation ladders (§3.7), determinism (§5), config discipline (§7) this spec assumes |
| [Decisions.md](Decisions.md) | `ADR-###` justification for every design choice stated mechanically here |
| [OpenQuestions.md](OpenQuestions.md) | `OQ-##` tracking for every constant marked `PLACEHOLDER` above |
| [Setup.md](Setup.md) | Environment-variable reference and installation |
| [TestPlan.md](TestPlan.md) | `TC-###` cases that verify the arithmetic and bounds in §5 |
