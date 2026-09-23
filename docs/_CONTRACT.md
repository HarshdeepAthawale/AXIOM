# Contract

The locked technical contract: the single source of truth every other document in `docs/` defers to.

**Owner:** Prabinder Singh
**Last updated:** 2026-09-23
**Status:** Accepted

> This file is the binding reference for every document in `docs/`. If a doc
> contradicts this file, the doc is wrong.
>
> **It is not deleted before submission.** Seventy-nine citations across the
> suite resolve into it; removing it would break every one of them. It ships as
> an ordinary document with the header convention above, like any other.

## 0. Identity

| Field | Value |
|---|---|
| Project | Axiom — Agentic Code Intelligence |
| Team | Incognito |
| Members | Prabinder Singh (Retrieval Core), Anish Grover (Structural), Harshdeep Athawale (Agentic + Rerank + UI), Parth Deshmukh (Versioning + Eval + Submission) |
| Institute | Thapar Institute of Engineering & Technology, Patiala |
| Event | Samsung PRISM GenAI Hackathon 3rd Edition (2026-27), Theme 01 |
| Build window | 2026-09-23 → 2026-09-27 (re-baselined; Day 1 = 2026-09-23, submission 2026-09-27) |
| Release tag | `PRISM_GENAI_HACKATHON_Y2026` (organiser-prescribed; see the naming note below) |
| Python package | `axiom` (distribution name `axiom`, import root `src/axiom`) |
| Repo layout | src-layout |

**The project is named Axiom, everywhere, with exactly two exceptions.** `ADR-015` is Accepted: the
package, the import root, the CLI entrypoint, the environment-variable prefix and the index
directory are all `axiom` / `AXIOM_` / `.axiom/`. The word `PRISM` survives in precisely two places
and means the Samsung programme, not this project:

| Surviving `PRISM` literal | What it is |
|---|---|
| `PRISM_GENAI_HACKATHON_Y2026` | The release tag. Organiser-prescribed; it does not change. |
| "Samsung PRISM GenAI Hackathon" | The event name. |

Any other occurrence of `prism` — in a module path, an import, a CLI invocation, an env var, an
exception class, or a directory name — is unpropagated rename residue and is a defect.

## 1. Runtime & Toolchain (LOCKED)

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 (3.11–3.12 supported) | 3.13 untested |
| Dependency manager | `uv` primary, `pip` + `requirements.txt` fallback | lockfile `uv.lock` committed |
| Config | `pydantic-settings` v2 + YAML profiles in `configs/` | env prefix `AXIOM_` |
| CLI | `typer` | entrypoint `axiom` (`[project.scripts] axiom = "axiom.cli:app"`) |
| API | `fastapi` + `uvicorn` | dev only, port 8000 |
| Demo UI | `streamlit` | port 8501 |
| Lint/format | `ruff` (lint + format) | line length 100 |
| Types | `mypy` strict on `src/axiom/core`, `src/axiom/retrieval`, `src/axiom/schema` |
| Tests | `pytest` + `pytest-cov` | `tests/` mirrors `src/axiom/` |
| Errors | `axiom.core.errors` | base `AxiomError`; see [Rules.md §9.2](Rules.md#92-error-taxonomy) |
| Seed | `Settings.seed`, default **`42`** (`AXIOM_SEED`) | the value the code carries; `1337` anywhere is stale |

Every tunable's canonical home is `src/axiom/config.py`; its canonical *documentation* is
[Setup.md §7](Setup.md#7-environment-variables). A field named in any other document and not in
those two is out of contract.
| Container | Docker, `python:3.11-slim-bookworm` base, CPU-only |
| CI | GitHub Actions: ruff → mypy → pytest → smoke index |

## 2. Model Stack (LOCKED, CPU-only)

| Role | Primary | Fallback | Runtime |
|---|---|---|---|
| Dense embedder | `Qwen/Qwen3-Embedding-0.6B` INT8 dynamic-quantized | `sentence-transformers/all-MiniLM-L6-v2` | ONNX Runtime CPU (`onnxruntime` ≥1.18) |
| Cross-encoder reranker | **profile-split, see below** | — | ONNX Runtime CPU |
| Query LLM | `Qwen2.5-1.5B-Instruct` Q4_K_M GGUF via `llama-cpp-python` | heuristic rule engine (no LLM) | local, ≤512 tok out |
| Sparse | `bm25s` (NOT `rank-bm25`) | — | numpy/scipy |
| Vector store | FAISS CPU | — | `IndexFlatIP` < 50k vectors, `IndexIVFPQ` ≥ 50k |
| AST | `tree-sitter` ≥0.22 + `tree-sitter-javascript` | regex identifier fallback | C |

**Cross-encoder reranker — the one role with no single primary.** The two profiles that produce
numbers want opposite things, so each names its own primary in `configs/`, and neither is "the
fallback":

| Profile | Reranker | Candidate chain | Why |
|---|---|---|---|
| `eval.yaml` (screening benchmark) | `BAAI/bge-reranker-v2-m3` INT8 | `fusion_top_n: 5`, `rerank_max_chars: 1024` | Accuracy is what is scored, and the run is offline and untimed. The cost is bought back by narrowing the chain (~8,375 → ~250 GFLOP), not by weakening the model. **No number from this profile is a latency claim.** |
| `demo.yaml` (live jury demo) | `cross-encoder/ms-marco-MiniLM-L-6-v2` INT8 | `fusion_top_n: 25`, `rerank_max_chars: 4096` | Latency is what the jury watches. `bge-reranker-v2-m3` is XLM-RoBERTa-large (24 layers, hidden 1024); 25 pairs of 512 tokens is ~8,375 GFLOP, which would demand ~13.5 TOPS to land in the 620 ms figure that appears in TechSpecifications.md §8 — roughly 2.6× the theoretical AVX-512-VNNI peak of the reference box. That 620 ms is MiniLM's number recorded against bge's row. |

`default.yaml`, `fast.yaml` and `accurate.yaml` inherit the `Settings` default
(`BAAI/bge-reranker-v2-m3`), except `fast.yaml`, which pins MiniLM as part of its < 500 MB budget.

**Embedding pooling and the query-side instruction envelope** are part of this lock, because every
wrong choice still produces a correctly-shaped 1024-dim vector and is therefore invisible to a shape
check. Implemented in `src/axiom/indexing/embedder.py`; specified in
[Setup.md §6.2](Setup.md#62-pooling-and-the-query-side-instruction-envelope):

| Model | Pooling | Query-side envelope |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | last **real** token, selected as `attention_mask.sum(axis=1) - 1` per row | `Instruct: {task}\nQuery:{query}`, **query side only** — documents are encoded bare |
| `sentence-transformers/all-MiniLM-L6-v2` | attention-mask-weighted **mean** | none |

A single shared pooler is itself a bug. Both rungs L2-normalise after pooling.

Embedding dim: 1024 (Qwen3-0.6B), 384 (MiniLM). Vectors are L2-normalised; similarity = inner product.
**The query LLM is NEVER used to read code.** Only: classify, expand, decompose, evaluate-sufficiency.
**Degradation is mandatory, not optional**: every model has a declared fallback and the pipeline must run with `AXIOM_LLM_ENABLED=false`.

## 3. Package Layout (LOCKED)

`axiom/` below denotes the **package-layout root**, not the checkout directory name — the GitHub
repository is `Samsung-Prism-Hack`, and the distribution installed from it is `axiom`.

```
axiom/
├── docs/
├── configs/            default.yaml, demo.yaml, eval.yaml, fast.yaml, accurate.yaml
├── data/               (gitignored) corpora, datasets
├── scripts/            build_index.py, run_eval.py, bench_latency.py
├── src/axiom/
│   ├── __init__.py
│   ├── config.py           Settings, profile loading
│   ├── pipeline.py         the one orchestrator: index(), reindex(), query()
│   ├── schema/             Pydantic models — THE shared contract
│   ├── core/               logging.py, timing.py, hashing.py, errors.py
│   ├── chunking/           ast_chunker.py, fallback.py, tokens.py
│   ├── indexing/           dense.py, sparse.py, structural.py, manifest.py, embedder.py
│   ├── retrieval/          dense.py, sparse.py, structural.py, fusion.py, tokenizer.py
│   ├── rerank/             cross_encoder.py
│   ├── agent/              classifier.py, planner.py, evaluator.py, loop.py, llm.py, synonyms.py
│   ├── versioning/         gitdiff.py, incremental.py, evolutionary.py
│   ├── eval/               mteb_adapter.py, metrics.py
│   ├── api/                app.py, routes.py, models.py
│   ├── ui/                 streamlit_app.py
│   └── cli.py
└── tests/
```

**Five profiles, not four.** `demo.yaml` is first-class (`ADR-001`, Accepted) and gates `M4`; it is
not a rename of `default.yaml`. The valid set is exactly
`default | demo | eval | fast | accurate`, and [Setup.md §7.1](Setup.md#71-core) carries the same
list.

## 4. Core Data Model (LOCKED — `src/axiom/schema/`)

All are Pydantic v2 models. Field names are binding.

```python
class QueryType(StrEnum):      SEMANTIC | STRUCTURAL | USAGE | HYBRID
class SignalKind(StrEnum):     DENSE | SPARSE | STRUCTURAL
class ChunkKind(StrEnum):      FUNCTION | METHOD | CLASS | MODULE | BLOCK

class ChunkLocation:
    file_path: str          # repo-relative, POSIX separators
    start_line: int         # 1-indexed, inclusive
    end_line: int           # 1-indexed, inclusive
    start_byte: int
    end_byte: int

class ChunkMetadata:
    symbol: str | None          # function/class name
    kind: ChunkKind
    parent_symbol: str | None
    is_exported: bool
    imports: list[str]
    calls: list[str]            # callee identifiers, in source order
    docstring: str | None
    language: str               # "javascript"
    version_id: str             # e.g. "v2.3.1" or commit sha
    commit_sha: str | None
    last_modified: str | None   # ISO-8601 date

class Chunk:
    chunk_id: str               # blake2b-128 hex of (content ⊕ file_path ⊕ start_line)
    content_hash: str           # blake2b-128 hex of normalised content only
    text: str
    location: ChunkLocation
    metadata: ChunkMetadata

class ScoredChunk:
    chunk_id: str
    score: float
    rank: int                   # 1-indexed
    signal: SignalKind

class FusedResult:
    chunk_id: str
    rrf_score: float
    rerank_score: float | None
    contributions: dict[SignalKind, int]   # signal -> rank in that list
    dominant_signal: SignalKind

class RetrievalResult:          # the thing users/API see
    chunk: Chunk
    score: float
    match_reason: str
    signals: dict[SignalKind, int]
    optimization_hint: str | None = None

class QueryPlan:
    original_query: str
    query_type: QueryType
    sub_queries: list[str]
    extracted_identifiers: list[str]
    expansion_terms: list[str]
    strategy_weights: dict[SignalKind, float]

class SnippetFamily:            # evolutionary retrieval
    family_id: str
    representative: Chunk
    members: list[Chunk]        # sorted newest → oldest
    versions: list[str]
    stability: float            # members / total_versions ∈ (0,1]
    diffs: list[str]

class VersionManifest:
    version_id: str
    commit_sha: str | None
    created_at: str
    chunk_count: int
    file_hashes: dict[str, str]     # repo-relative path -> blake2b-128
    embedding_model: str
    embedding_dim: int
    index_kind: str                 # "flat_ip" | "ivf_pq"
    parent_version: str | None
```

## 5. Algorithms (LOCKED)

- **RRF**: `score(d) = Σ_i w_i / (k + rank_i(d))`, `k = 60`, weights from `QueryPlan.strategy_weights`.
  Default weights: SEMANTIC `{dense .6, sparse .3, struct .1}`, STRUCTURAL `{dense .2, sparse .2, struct .6}`,
  USAGE `{dense .25, sparse .55, struct .2}`, HYBRID `{.34, .33, .33}`.
- **Candidate widths**: dense K=100, sparse K=100, structural K=50 → RRF → top-N=25 → rerank → top-10.
- **Agent loop**: `AXIOM_AGENT_MAX_PASSES=2` bounds **TOTAL** retrieve → fuse → hydrate → rerank
  cycles, **initial pass included** — one initial pass plus at most one refinement, never three
  cycles. The initial retrieval is inside the loop, not before it. Hard wall-clock budget
  `AXIOM_AGENT_WALL_CLOCK_MS=5000`, checked **before** starting a pass and again before the rerank
  of a refinement pass, never after finishing one; pass 1 is exempt so a query always gets one real
  attempt. `passes_used` is therefore in `0..agent_max_passes`, with `0` only for a query that went
  empty after normalisation. Sufficiency trigger: top-1 rerank score < 0.35 OR fewer than 3 results
  above 0.20.
- **Chunking**: AST node boundaries; target 64–512 tokens; oversized functions split at statement
  boundaries with 1-statement overlap; chunks < 16 tokens merged into parent.
- **Incremental reindex**: `git diff --name-status <old>..<new>` → {A,M,D,R}; re-embed only A/M;
  content-hash dedupe means an unchanged-content rename costs 0 embeddings.
- **Evolutionary dedupe**: cosine ≥ 0.95 within same `symbol`+`file_path` → one `SnippetFamily`.
  Ranking bonus: `final = base * (1 + 0.10*stability)` for families present in ≥2 versions.

## 6. On-Disk Index Layout (LOCKED)

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

`.axiom/` is the single root shape, and it is the only one. The benchmark and demo corpora are two
*instances* of it, not a second layout: they are separated by two `AXIOM_INDEX_ROOT` values
(`.axiom-apps/` and `.axiom-demo/`, [Setup.md §5.5](Setup.md#55-dataset--coir-appsretrieval)), each
holding the tree above unchanged. Any `.prism`, `.prism_full` or `.prism_inc` path in any document
or fixture is rename residue and is a defect.

## 7. Performance Budgets (LOCKED, 8-core CPU / 16 GB RAM reference box)

| Operation | Budget |
|---|---|
| Cold index, 10k chunks | ≤ 12 min |
| Incremental reindex, 50 changed files | ≤ 45 s |
| Query p50 (no agent loop) | ≤ 900 ms |
| Query p95 (with 2 agent passes) | ≤ 5 s |
| Peak RSS during query | ≤ 4 GB |

**Every row above is a budget, not a measurement.** No number here has been produced by
`scripts/bench_latency.py` on a real INT8 artifact, and until one has, none of them may be quoted
as "measured" in TechSpecifications.md §8, the deck, or to the jury —
[Rules.md §8](Rules.md#8-the-placeholder-convention) item 3 governs a latency figure exactly as it
governs an accuracy one. A budget that is missed is a scope decision; a budget reported as a
measurement is a false claim.

## 8. Targets

**The baseline is measured, not cited.** The previously-locked row `BGE 0.6B = 14.7` has been
removed: that figure appears in neither of the two papers the suite cited for it, and calibrating
`M2`/`M3`/`M5` and the deck against an unsourced number that a jury can falsify in thirty seconds is
a worse risk than a lower honest one. Its replacement is our own dense-only run on the same split,
produced by the **first** eval of the project, and the headline claim becomes a *relative* gain over
that baseline plus an ablation table.

| Metric | Baseline | Target |
|---|---|---|
| NDCG@10 (CoIR AppsRetrieval test) | our measured dense-only run — `# PLACEHOLDER`, owned by `T-200` | re-derived from the baseline once it exists; reported as a relative gain |
| MRR | same run | re-derived |
| Recall@100 (first stage) | same run | re-derived |

Until `T-200` lands, every one of these cells is a `# PLACEHOLDER` under
[Rules.md §8](Rules.md#8-the-placeholder-convention), and [Rules.md §8](Rules.md#8-the-placeholder-convention)
item 3 therefore forbids any of them from reaching the PPT, the release JSON or a claim to the jury.
The ablation table that must accompany the reported number: dense-only → +sparse → +rerank →
+agent, each row one line in `data/experiments.csv`.

## 9. Doc Conventions

- Every doc opens with a one-line purpose, an `Owner:` line, and a **maintained** `Last updated:`
  date — the date that doc last changed, not a fixed literal.
- Status vocabulary: `Planned` → `In Progress` → `Blocked` → `Done` → `Dropped`.
- Requirement IDs: `FR-##` functional, `NFR-##` non-functional, `ADR-###` decisions,
  `TC-###` test cases, `RISK-##`, `OQ-##` open questions, `NG-##` non-goals, `T-###` tracker tasks.
- Cross-reference other docs by relative link, e.g. `[Schema.md](Schema.md#6-chunk)`.
- No emoji. Tables over prose where the content is tabular. Markdown only.
