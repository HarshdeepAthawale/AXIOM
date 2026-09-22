# PRISM — Locked Technical Contract (source of truth for all docs)

> This file is the binding reference for every document in `docs/`. If a doc
> contradicts this file, the doc is wrong. Delete this file before submission;
> it is a build-time coordination artifact.

## 0. Identity

| Field | Value |
|---|---|
| Project | Axiom — Agentic Code Intelligence |
| Team | Incognito |
| Members | Prabinder Singh (Retrieval Core), Anish Grover (Structural), Harshdeep Athawale (Agentic + Rerank + UI), Parth Deshmukh (Versioning + Eval + Submission) |
| Institute | Thapar Institute of Engineering & Technology, Patiala |
| Event | Samsung PRISM GenAI Hackathon 3rd Edition (2026-27), Theme 01 |
| Build window | 2026-09-15 → 2026-09-27 |
| Release tag | `PRISM_GENAI_HACKATHON_Y2026` |
| Python package | `prism` (import root `src/axiom`) |
| Repo layout | src-layout |

## 1. Runtime & Toolchain (LOCKED)

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 (3.11–3.12 supported) | 3.13 untested |
| Dependency manager | `uv` primary, `pip` + `requirements.txt` fallback | lockfile `uv.lock` committed |
| Config | `pydantic-settings` v2 + YAML profiles in `configs/` | env prefix `PRISM_` |
| CLI | `typer` | entrypoint `prism` |
| API | `fastapi` + `uvicorn` | dev only, port 8000 |
| Demo UI | `streamlit` | port 8501 |
| Lint/format | `ruff` (lint + format) | line length 100 |
| Types | `mypy` strict on `src/axiom/core`, `src/axiom/retrieval`, `src/axiom/schema` |
| Tests | `pytest` + `pytest-cov` | `tests/` mirrors `src/axiom/` |
| Container | Docker, `python:3.11-slim-bookworm` base, CPU-only |
| CI | GitHub Actions: ruff → mypy → pytest → smoke index |

## 2. Model Stack (LOCKED, CPU-only)

| Role | Primary | Fallback | Runtime |
|---|---|---|---|
| Dense embedder | `Qwen/Qwen3-Embedding-0.6B` INT8 dynamic-quantized | `sentence-transformers/all-MiniLM-L6-v2` | ONNX Runtime CPU (`onnxruntime` ≥1.18) |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` INT8 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ONNX Runtime CPU |
| Query LLM | `Qwen2.5-1.5B-Instruct` Q4_K_M GGUF via `llama-cpp-python` | heuristic rule engine (no LLM) | local, ≤512 tok out |
| Sparse | `bm25s` (NOT `rank-bm25`) | — | numpy/scipy |
| Vector store | FAISS CPU | — | `IndexFlatIP` < 50k vectors, `IndexIVFPQ` ≥ 50k |
| AST | `tree-sitter` ≥0.22 + `tree-sitter-javascript` | regex identifier fallback | C |

Embedding dim: 1024 (Qwen3-0.6B), 384 (MiniLM). Vectors are L2-normalised; similarity = inner product.
**The query LLM is NEVER used to read code.** Only: classify, expand, decompose, evaluate-sufficiency.
**Degradation is mandatory, not optional**: every model has a declared fallback and the pipeline must run with `AXIOM_LLM_ENABLED=false`.

## 3. Package Layout (LOCKED)

```
axiom/
├── docs/
├── configs/            default.yaml, fast.yaml, accurate.yaml, eval.yaml
├── data/               (gitignored) corpora, datasets
├── scripts/            build_index.py, run_eval.py, bench_latency.py
├── src/axiom/
│   ├── __init__.py
│   ├── config.py           Settings, profile loading
│   ├── schema/             Pydantic models — THE shared contract
│   ├── core/               logging, timing, hashing, errors
│   ├── chunking/           tree-sitter AST chunker + fallback splitter
│   ├── indexing/           dense.py, sparse.py, structural.py, manifest.py
│   ├── retrieval/          dense.py, sparse.py, structural.py, fusion.py
│   ├── rerank/             cross_encoder.py
│   ├── agent/              classifier.py, planner.py, evaluator.py, loop.py, llm.py
│   ├── versioning/         gitdiff.py, incremental.py, evolutionary.py
│   ├── eval/               mteb_adapter.py, metrics.py
│   ├── api/                app.py, routes.py, models.py
│   ├── ui/                 streamlit_app.py
│   └── cli.py
└── tests/
```

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
- **Agent loop**: max 2 refinement passes (`AXIOM_AGENT_MAX_PASSES=2`), hard wall-clock budget 5 s/query.
  Sufficiency trigger: top-1 rerank score < 0.35 OR fewer than 3 results above 0.20.
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

## 7. Performance Budgets (LOCKED, 8-core CPU / 16 GB RAM reference box)

| Operation | Budget |
|---|---|
| Cold index, 10k chunks | ≤ 12 min |
| Incremental reindex, 50 changed files | ≤ 45 s |
| Query p50 (no agent loop) | ≤ 900 ms |
| Query p95 (with 2 agent passes) | ≤ 5 s |
| Peak RSS during query | ≤ 4 GB |

## 8. Targets

| Metric | Baseline to beat | Target |
|---|---|---|
| NDCG@10 (CoIR AppsRetrieval test) | BGE 0.6B = 14.7 | ≥ 20.0 |
| MRR | — | ≥ 22.0 |
| Recall@100 (first stage) | — | ≥ 65.0 |

## 9. Doc Conventions

- Every doc opens with a one-line purpose, an `Owner:` line, and `Last updated: 2026-09-15`.
- Status vocabulary: `Planned` → `In Progress` → `Blocked` → `Done` → `Dropped`.
- Requirement IDs: `FR-##` functional, `NFR-##` non-functional, `ADR-###` decisions,
  `TC-###` test cases, `RISK-##`, `OQ-##` open questions, `NG-##` non-goals, `T-###` tracker tasks.
- Cross-reference other docs by relative link, e.g. `[Schema.md](Schema.md#chunk)`.
- No emoji. Tables over prose where the content is tabular. Markdown only.
