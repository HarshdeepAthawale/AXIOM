# Schema

Canonical data-model reference for PRISM: every Pydantic model, enum, identity function, and on-disk format that other workstreams code against.

**Owner:** Prabinder Singh
**Last updated:** 2026-09-15
**Status:** Draft

---

## 1. Scope and authority

This document is the human-readable projection of `src/axiom/schema/`. The Python source in
`src/axiom/schema/` is the executable truth; this document is the normative explanation of it.
Field names, types, and ordering here are binding and are taken byte-for-byte from
`docs/_CONTRACT.md` §4. Nothing in this document may be renamed without an ADR in
[Decisions.md](Decisions.md).

Consumers of this schema:

| Consumer | Module | What it reads/writes | Owner |
|---|---|---|---|
| Chunker | `src/axiom/chunking/` | writes `Chunk`, `ChunkLocation`, `ChunkMetadata` | Anish Grover |
| Dense index | `src/axiom/indexing/dense.py` | reads `Chunk`, writes `dense.faiss` + `dense.idmap.json` | Prabinder Singh |
| Sparse index | `src/axiom/indexing/sparse.py` | reads `Chunk`, writes `sparse.bm25s/` | Prabinder Singh |
| Structural index | `src/axiom/indexing/structural.py` | reads `Chunk`, writes `structural.sqlite` | Anish Grover |
| Fusion | `src/axiom/retrieval/fusion.py` | reads `ScoredChunk`, writes `FusedResult` | Prabinder Singh |
| Reranker | `src/axiom/rerank/cross_encoder.py` | reads `FusedResult` + `Chunk`, writes `FusedResult.rerank_score` | Harshdeep Athawale |
| Agent | `src/axiom/agent/` | writes `QueryPlan`, reads `FusedResult` | Harshdeep Athawale |
| Versioning | `src/axiom/versioning/` | writes `VersionManifest`, `SnippetFamily` | Parth Deshmukh |
| API / UI | `src/axiom/api/`, `src/axiom/ui/` | reads `RetrievalResult` only | Harshdeep Athawale |

Related documents: [TechSpecifications.md](TechSpecifications.md) (component behaviour),
[Design.md](Design.md) (rationale), [APISpec.md](APISpec.md) (wire format),
[Decisions.md](Decisions.md) (ADR log).

---

## 2. Module layout

```
src/axiom/schema/
├── __init__.py       re-exports every public name below
├── enums.py          QueryType, SignalKind, ChunkKind
├── chunk.py          ChunkLocation, ChunkMetadata, Chunk
├── retrieval.py      ScoredChunk, FusedResult, RetrievalResult
├── plan.py           QueryPlan
└── version.py        SnippetFamily, VersionManifest
```

Shared configuration applied to every model in the package:

```python
# src/axiom/schema/_base.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PrismModel(BaseModel):
    """Base class for every PRISM schema model.

    `extra="forbid"` is deliberate: a typo'd field name in a config file or a
    stale `chunks.jsonl` written by an older PRISM build must fail loudly at
    load time rather than being silently dropped and producing a subtly wrong
    index. `frozen=True` makes every schema object hashable and safe to share
    across the ONNX embedding thread pool without defensive copying.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        validate_assignment=True,
        use_enum_values=False,
    )
```

`str_strip_whitespace=False` is load-bearing: `Chunk.text` must preserve leading indentation
exactly, or line offsets and the reranker's view of the code both drift.

---

## 3. Enums

All three are `StrEnum` (Python 3.11 stdlib), so they serialise to plain lowercase strings in
JSON and compare equal to those strings without an explicit `.value`.

```python
# src/axiom/schema/enums.py
from __future__ import annotations

from enum import StrEnum


class QueryType(StrEnum):
    """How the agent classified the incoming query.

    Selects the RRF weight vector (see TechSpecifications.md §6).
    """

    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    USAGE = "usage"
    HYBRID = "hybrid"


class SignalKind(StrEnum):
    """One of the three first-stage retrieval signals."""

    DENSE = "dense"
    SPARSE = "sparse"
    STRUCTURAL = "structural"


class ChunkKind(StrEnum):
    """The syntactic category of the AST node a chunk was cut from."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    BLOCK = "block"
```

### 3.1 `QueryType` semantics

| Variant | Meaning | Example query | Dominant signal |
|---|---|---|---|
| `SEMANTIC` | Intent-level question; the answer is code whose *behaviour* matches the description, and the user does not know the identifier names. | "How is the input preprocessed before going to the main function?" | dense (0.6) |
| `STRUCTURAL` | Question about relationships between code units — calls, imports, ordering, containment. | "Which files call tool XYZ before tool ABC?" | structural (0.6) |
| `USAGE` | Question about a *named* entity the user already knows; needs exact-token matching. | "Where is the Bluetooth-settings deeplink used?" | sparse (0.55) |
| `HYBRID` | Mixed intent, or the classifier's confidence is below threshold. Deliberately the safe default. | "How does the auth module validate tokens before calling the API?" | none (0.34/0.33/0.33) |

Classifier ambiguity always resolves to `HYBRID`. A wrong confident classification costs more
NDCG than a correct-but-flat weighting.

### 3.2 `SignalKind` semantics

| Variant | Produced by | Score domain before fusion |
|---|---|---|
| `DENSE` | `retrieval/dense.py` — FAISS inner-product search over L2-normalised Qwen3 embeddings | cosine in `[-1, 1]`, practically `[0, 1]` |
| `SPARSE` | `retrieval/sparse.py` — `bm25s` over code-aware tokens | unbounded positive BM25 score |
| `STRUCTURAL` | `retrieval/structural.py` — SQL over `structural.sqlite` (call graph, import graph, symbol table, export map) | integer/graph-derived score, unbounded |

The three score domains are mutually incomparable. That incomparability is exactly why fusion is
rank-based (RRF) and not score-based; see [Design.md](Design.md#the-three-signal-rationale).

### 3.3 `ChunkKind` semantics

| Variant | tree-sitter origin | Notes |
|---|---|---|
| `FUNCTION` | `function_declaration`, `function_expression`, `arrow_function`, `generator_function_declaration` | Top-level or nested standalone function. `parent_symbol` is `None` unless lexically nested. |
| `METHOD` | `method_definition` inside `class_body` | `parent_symbol` is the owning class name and is never `None`. |
| `CLASS` | `class_declaration`, `class_expression` | Emitted for the class header + fields; each method is its own chunk. A class whose methods all fit under 512 tokens is *also* emitted whole so class-level queries can match. |
| `MODULE` | whole-file node | Emitted only for files with no extractable top-level definitions (config objects, barrel re-export files, pure side-effect scripts). |
| `BLOCK` | oversized-function split fragments, and residual top-level statement runs | The only kind whose boundaries are not a single AST node. `symbol` carries the parent function name with a `#<n>` suffix for split fragments. |

---

## 4. `ChunkLocation`

### Purpose

Pins a chunk to an exact region of an exact file. This is what the user ultimately consumes — the
deliverable of the whole system is "snippet + file + line location", so this model must be exact,
not approximate. Byte offsets exist alongside line numbers because tree-sitter works in bytes and
re-slicing source by byte range is the only way to recover the chunk text without re-parsing.

```python
# src/axiom/schema/chunk.py
from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from axiom.schema._base import PrismModel
from axiom.schema.enums import ChunkKind


class ChunkLocation(PrismModel):
    """Exact source location of a chunk: repo-relative path plus line and byte span."""

    file_path: str = Field(
        ...,
        min_length=1,
        description="Repo-relative path with POSIX ('/') separators. Never absolute, "
        "never Windows-style, never prefixed with './'.",
    )
    start_line: int = Field(
        ...,
        ge=1,
        description="First line of the chunk. 1-indexed, inclusive.",
    )
    end_line: int = Field(
        ...,
        ge=1,
        description="Last line of the chunk. 1-indexed, inclusive.",
    )
    start_byte: int = Field(
        ...,
        ge=0,
        description="Byte offset of the first byte of the chunk in the UTF-8 encoded file.",
    )
    end_byte: int = Field(
        ...,
        ge=0,
        description="Byte offset one past the last byte of the chunk (half-open, "
        "matching tree-sitter's Node.end_byte).",
    )

    @field_validator("file_path")
    @classmethod
    def _posix_relative(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError(f"file_path must use POSIX separators, got {value!r}")
        if value.startswith("/") or value.startswith("./") or ":" in value[:3]:
            raise ValueError(f"file_path must be repo-relative, got {value!r}")
        return value

    @model_validator(mode="after")
    def _spans_ordered(self) -> "ChunkLocation":
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line {self.end_line} precedes start_line {self.start_line}"
            )
        if self.end_byte <= self.start_byte:
            raise ValueError(
                f"end_byte {self.end_byte} must exceed start_byte {self.start_byte}"
            )
        return self

    @property
    def line_count(self) -> int:
        """Number of source lines covered, inclusive of both endpoints."""
        return self.end_line - self.start_line + 1

    def as_ref(self) -> str:
        """Human-facing location string, e.g. 'src/agents/bluetooth.js:42-67'."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"
```

### Fields

| Name | Type | Required | Description | Invariant |
|---|---|---|---|---|
| `file_path` | `str` | yes | Repo-relative path, POSIX separators | non-empty; no `\`; no leading `/` or `./`; no drive letter |
| `start_line` | `int` | yes | First source line | `>= 1` |
| `end_line` | `int` | yes | Last source line | `>= start_line` |
| `start_byte` | `int` | yes | Byte offset of chunk start | `>= 0` |
| `end_byte` | `int` | yes | Byte offset one past chunk end | `> start_byte` |

Note the asymmetry, which trips people up: **line range is inclusive, byte range is half-open.**
Lines are inclusive because that is what a human reading `bluetooth.js:42-67` expects. Bytes are
half-open because that is what tree-sitter emits and what Python slicing wants —
`source_bytes[start_byte:end_byte]` yields exactly the chunk.

### Example

```json
{
  "file_path": "src/agents/bluetooth.js",
  "start_line": 42,
  "end_line": 67,
  "start_byte": 1284,
  "end_byte": 2011
}
```

---

## 5. `ChunkMetadata`

### Purpose

Everything the retrieval layers need to know about a chunk that is not its text or its location.
`ChunkMetadata` is the join point between the three signals: the dense index ignores it, the sparse
index folds `symbol` and `calls` into the token stream, and the structural index is built almost
entirely from it. It also carries the version triple (`version_id`, `commit_sha`,
`last_modified`) that makes P1 and the evolutionary bonus possible.

```python
class ChunkMetadata(PrismModel):
    """Retrieval-relevant facts about a chunk: symbol identity, graph edges, provenance."""

    symbol: str | None = Field(
        default=None,
        description="Declared name of the function/method/class. None for anonymous "
        "arrow functions assigned to nothing, and for MODULE chunks.",
    )
    kind: ChunkKind = Field(
        ...,
        description="Syntactic category of the originating AST node.",
    )
    parent_symbol: str | None = Field(
        default=None,
        description="Enclosing class or function name. Always set for METHOD chunks.",
    )
    is_exported: bool = Field(
        default=False,
        description="True if the symbol leaves the module via ESM export or CommonJS "
        "module.exports/exports.x. Drives 'where is X used' resolution.",
    )
    imports: list[str] = Field(
        default_factory=list,
        description="Module specifiers imported by the file this chunk came from, "
        "verbatim as written (e.g. './normalize', 'node:fs').",
    )
    calls: list[str] = Field(
        default_factory=list,
        description="Callee identifiers invoked inside the chunk, in source order. "
        "Order is significant: it answers 'calls X before Y'. Duplicates are kept.",
    )
    docstring: str | None = Field(
        default=None,
        description="Leading JSDoc or comment block attached to the symbol, comment "
        "markers stripped, whitespace-normalised.",
    )
    language: str = Field(
        default="javascript",
        description="Source language of the chunk. Only 'javascript' is produced in "
        "the hackathon scope; the field exists so adding a grammar needs no migration.",
    )
    version_id: str = Field(
        ...,
        min_length=1,
        description="Logical version this chunk belongs to, e.g. 'v2.3.1' or a commit sha. "
        "Every chunk belongs to exactly one version.",
    )
    commit_sha: str | None = Field(
        default=None,
        description="Full 40-char git commit sha, when the corpus is a git repo.",
    )
    last_modified: str | None = Field(
        default=None,
        description="ISO-8601 date (YYYY-MM-DD) of the commit that last touched the file.",
    )

    @field_validator("commit_sha")
    @classmethod
    def _sha_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 40 or not all(c in "0123456789abcdef" for c in value):
            raise ValueError(f"commit_sha must be 40 lowercase hex chars, got {value!r}")
        return value

    @field_validator("last_modified")
    @classmethod
    def _iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from datetime import date

        date.fromisoformat(value)  # raises ValueError on malformed input
        return value

    @model_validator(mode="after")
    def _method_has_parent(self) -> "ChunkMetadata":
        if self.kind is ChunkKind.METHOD and not self.parent_symbol:
            raise ValueError("METHOD chunks must carry parent_symbol")
        return self
```

### Fields

| Name | Type | Required | Description | Invariant |
|---|---|---|---|---|
| `symbol` | `str \| None` | no (`None`) | Declared symbol name | `None` only for anonymous or `MODULE` chunks |
| `kind` | `ChunkKind` | yes | Syntactic category | one of the five variants |
| `parent_symbol` | `str \| None` | no (`None`) | Enclosing class/function | non-`None` when `kind == METHOD` |
| `is_exported` | `bool` | no (`False`) | Escapes the module | if `True`, an `exports` row exists for it |
| `imports` | `list[str]` | no (`[]`) | File-level module specifiers | verbatim source text; may repeat |
| `calls` | `list[str]` | no (`[]`) | Callee identifiers | **source order preserved**, duplicates kept |
| `docstring` | `str \| None` | no (`None`) | Attached doc comment | markers stripped; never the empty string |
| `language` | `str` | no (`"javascript"`) | Source language | lowercase tree-sitter grammar name |
| `version_id` | `str` | yes | Owning version | non-empty; a key in `registry.json` |
| `commit_sha` | `str \| None` | no (`None`) | Git commit | 40 lowercase hex chars |
| `last_modified` | `str \| None` | no (`None`) | Last-touched date | valid `YYYY-MM-DD` |

`calls` ordering is not cosmetic. It is the entire data source for ordering predicates such as
"which files call tool XYZ *before* tool ABC". Any code that sorts, sets, or dedups `calls` breaks
`STRUCTURAL` retrieval.

### Example

```json
{
  "symbol": "handleDeeplink",
  "kind": "function",
  "parent_symbol": null,
  "is_exported": true,
  "imports": ["./normalize", "../tools/registry", "node:url"],
  "calls": ["preprocessInput", "resolveTool", "logEvent", "resolveTool"],
  "docstring": "Resolve an incoming bluetooth-settings deeplink to a tool invocation.",
  "language": "javascript",
  "version_id": "v2.3.1",
  "commit_sha": "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
  "last_modified": "2026-09-10"
}
```

---

## 6. `Chunk`

### Purpose

The atomic unit of retrieval. Every index row, every score, every result traces back to exactly one
`Chunk`. A `Chunk` is immutable and content-addressed twice over — once by location-sensitive
`chunk_id`, once by location-insensitive `content_hash` — and that double addressing is the
mechanism behind both incremental reindexing and cross-version embedding dedup.

```python
class Chunk(PrismModel):
    """One retrievable code snippet: text, exact location, and retrieval metadata."""

    chunk_id: str = Field(
        ...,
        pattern=r"^[0-9a-f]{32}$",
        description="blake2b-128 hex digest of (content, file_path, start_line). "
        "Globally unique per (version, file, position). Primary key everywhere.",
    )
    content_hash: str = Field(
        ...,
        pattern=r"^[0-9a-f]{32}$",
        description="blake2b-128 hex digest of NORMALISED content only. Location "
        "independent, so identical code at a different path/line shares one hash "
        "and therefore one cached embedding.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Verbatim source text of the chunk, indentation preserved. "
        "Equals source_bytes[start_byte:end_byte].decode('utf-8').",
    )
    location: ChunkLocation = Field(..., description="Where this chunk lives.")
    metadata: ChunkMetadata = Field(..., description="What this chunk is.")

    @model_validator(mode="after")
    def _ids_match_content(self) -> "Chunk":
        from axiom.core.hashing import compute_chunk_id, compute_content_hash

        expected_chunk_id = compute_chunk_id(
            self.text, self.location.file_path, self.location.start_line
        )
        if self.chunk_id != expected_chunk_id:
            raise ValueError(
                f"chunk_id {self.chunk_id} does not match recomputed {expected_chunk_id}"
            )
        expected_content_hash = compute_content_hash(self.text)
        if self.content_hash != expected_content_hash:
            raise ValueError(
                f"content_hash {self.content_hash} does not match recomputed "
                f"{expected_content_hash}"
            )
        return self

    @property
    def blob_name(self) -> str:
        """Filename of this chunk's cached embedding under `.axiom/blobs/`."""
        return f"{self.content_hash}.npy"
```

The `_ids_match_content` validator is a strict integrity check and it runs on every
`chunks.jsonl` line at load time. It is the cheapest possible defence against a corrupted or
hand-edited index: two blake2b-128 digests over a ~500-token string cost roughly 2 microseconds,
so validating a 10k-chunk index costs under 50 ms.

### Fields

| Name | Type | Required | Description | Invariant |
|---|---|---|---|---|
| `chunk_id` | `str` | yes | Location-sensitive identity | 32 lowercase hex chars; equals `compute_chunk_id(text, file_path, start_line)` |
| `content_hash` | `str` | yes | Location-insensitive identity | 32 lowercase hex chars; equals `compute_content_hash(text)` |
| `text` | `str` | yes | Verbatim snippet | non-empty; byte-identical to the source slice |
| `location` | `ChunkLocation` | yes | Source position | see §4 |
| `metadata` | `ChunkMetadata` | yes | Retrieval metadata | see §5 |

### Example

```json
{
  "chunk_id": "9f2c41d80ba7e35617c4d9a0e8b3f512",
  "content_hash": "3ad1907ceef84b2260d5c7a19e04b8f3",
  "text": "export function handleDeeplink(url) {\n  const clean = preprocessInput(url);\n  const tool = resolveTool(clean.host);\n  logEvent('deeplink', clean.host);\n  return resolveTool(tool.id).invoke(clean.params);\n}",
  "location": {
    "file_path": "src/agents/bluetooth.js",
    "start_line": 42,
    "end_line": 67,
    "start_byte": 1284,
    "end_byte": 2011
  },
  "metadata": {
    "symbol": "handleDeeplink",
    "kind": "function",
    "parent_symbol": null,
    "is_exported": true,
    "imports": ["./normalize", "../tools/registry", "node:url"],
    "calls": ["preprocessInput", "resolveTool", "logEvent", "resolveTool"],
    "docstring": "Resolve an incoming bluetooth-settings deeplink to a tool invocation.",
    "language": "javascript",
    "version_id": "v2.3.1",
    "commit_sha": "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
    "last_modified": "2026-09-10"
  }
}
```

---

## 7. `ScoredChunk`

### Purpose

One retriever's opinion about one chunk. Produced by each of the three signals independently, and
consumed only by the fusion layer. Deliberately carries `chunk_id` rather than a whole `Chunk`:
each signal returns 50-100 of these per query, and hydrating full chunk bodies before fusion has
discarded 75% of them would waste both allocation and cache.

```python
# src/axiom/schema/retrieval.py
from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from axiom.schema._base import PrismModel
from axiom.schema.chunk import Chunk
from axiom.schema.enums import QueryType, SignalKind


class ScoredChunk(PrismModel):
    """A single (chunk, score, rank) triple emitted by one retrieval signal."""

    chunk_id: str = Field(
        ..., pattern=r"^[0-9a-f]{32}$", description="Identity of the scored chunk."
    )
    score: float = Field(
        ...,
        description="Raw, signal-native score. Cosine for DENSE, BM25 for SPARSE, "
        "graph score for STRUCTURAL. NOT comparable across signals.",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Position within this signal's ranked list. 1-indexed; rank 1 is best.",
    )
    signal: SignalKind = Field(..., description="Which retriever produced this.")
```

### Fields

| Name | Type | Required | Description | Invariant |
|---|---|---|---|---|
| `chunk_id` | `str` | yes | Scored chunk | 32 hex chars; resolvable in the active version |
| `score` | `float` | yes | Signal-native score | finite; comparable only within one `signal` |
| `rank` | `int` | yes | Rank in this signal's list | `>= 1`; contiguous and unique within a list |
| `signal` | `SignalKind` | yes | Producing retriever | one of three |

A `list[ScoredChunk]` handed to fusion must satisfy: all elements share one `signal`; `rank` values
are exactly `1..len(list)` with no gaps or ties; and `score` is non-increasing as `rank` increases.
Ties in raw score are broken deterministically by ascending `chunk_id` so that two runs on the same
index produce byte-identical output.

### Example

```json
{
  "chunk_id": "9f2c41d80ba7e35617c4d9a0e8b3f512",
  "score": 0.7431,
  "rank": 3,
  "signal": "dense"
}
```

---

## 8. `FusedResult`

### Purpose

The output of RRF and the input to reranking. Keeps full provenance — which signals found this
chunk and at what rank — because that provenance is what produces the user-visible `match_reason`
and what makes fusion debuggable. When a query returns something absurd, `contributions` tells you
in one glance whether the dense, sparse, or structural leg misfired.

```python
class FusedResult(PrismModel):
    """A chunk after Reciprocal Rank Fusion, optionally after cross-encoder reranking."""

    chunk_id: str = Field(
        ..., pattern=r"^[0-9a-f]{32}$", description="Identity of the fused chunk."
    )
    rrf_score: float = Field(
        ...,
        gt=0.0,
        description="Sum over contributing signals of w_i / (60 + rank_i). Strictly "
        "positive: a chunk with no contributions is never constructed.",
    )
    rerank_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sigmoid-normalised cross-encoder relevance in [0,1]. None when "
        "the chunk fell outside top-N=25, or when reranking is disabled.",
    )
    contributions: dict[SignalKind, int] = Field(
        ...,
        min_length=1,
        description="signal -> 1-indexed rank that signal assigned this chunk. "
        "Absent key means that signal did not return this chunk at all.",
    )
    dominant_signal: SignalKind = Field(
        ...,
        description="The signal contributing the largest w_i/(60+rank_i) term. Ties "
        "resolve in the order DENSE, SPARSE, STRUCTURAL.",
    )

    @field_validator("contributions")
    @classmethod
    def _ranks_positive(cls, value: dict[SignalKind, int]) -> dict[SignalKind, int]:
        for signal, rank in value.items():
            if rank < 1:
                raise ValueError(f"contribution rank for {signal} must be >= 1, got {rank}")
        return value

    @model_validator(mode="after")
    def _dominant_is_contributor(self) -> "FusedResult":
        if self.dominant_signal not in self.contributions:
            raise ValueError(
                f"dominant_signal {self.dominant_signal} absent from contributions "
                f"{sorted(self.contributions)}"
            )
        return self

    @property
    def final_score(self) -> float:
        """Score used for the final ordering: rerank score when present, else RRF score."""
        return self.rrf_score if self.rerank_score is None else self.rerank_score

    @property
    def signal_count(self) -> int:
        """How many of the three signals independently surfaced this chunk."""
        return len(self.contributions)
```

### Fields

| Name | Type | Required | Description | Invariant |
|---|---|---|---|---|
| `chunk_id` | `str` | yes | Fused chunk | 32 hex chars |
| `rrf_score` | `float` | yes | Weighted RRF sum | `> 0`; equals `Σ w_i/(60 + rank_i)` over `contributions` |
| `rerank_score` | `float \| None` | no (`None`) | Cross-encoder score | `None` or in `[0, 1]` |
| `contributions` | `dict[SignalKind, int]` | yes | signal → rank | non-empty; `1 <= len <= 3`; all ranks `>= 1` |
| `dominant_signal` | `SignalKind` | yes | Largest contributing term | must be a key of `contributions` |

`rerank_score` being `None` is not "unknown relevance", it is "not a rerank candidate". Sorting a
mixed list must therefore use `final_score`, never `rerank_score or 0.0` — the latter would push
un-reranked chunks below genuinely irrelevant reranked ones.

### Example

```json
{
  "chunk_id": "9f2c41d80ba7e35617c4d9a0e8b3f512",
  "rrf_score": 0.0189,
  "rerank_score": 0.8142,
  "contributions": { "dense": 3, "sparse": 11, "structural": 2 },
  "dominant_signal": "structural"
}
```

---

## 9. `RetrievalResult`

### Purpose

The only schema object that crosses the API and UI boundary. Everything else in this document is
internal. `RetrievalResult` carries the whole `Chunk` (so the caller gets snippet text, file path,
and line range in one payload), a single comparable `score`, and a human-readable `match_reason`.
Keeping it separate from `FusedResult` means we can change the internals of fusion without breaking
the public contract in [APISpec.md](APISpec.md).

```python
class RetrievalResult(PrismModel):
    """A user-facing search hit: the snippet, its location, its score, and why it matched."""

    chunk: Chunk = Field(..., description="The full retrieved chunk, text included.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final relevance in [0,1]: the rerank score, multiplied by the "
        "evolutionary stability bonus when the chunk belongs to a multi-version family.",
    )
    match_reason: str = Field(
        ...,
        min_length=1,
        description="One-sentence explanation naming the dominant signal and the "
        "concrete evidence, e.g. \"structural: calls preprocessInput before main\".",
    )
    signals: dict[SignalKind, int] = Field(
        ...,
        min_length=1,
        description="signal -> 1-indexed first-stage rank. Copied verbatim from "
        "FusedResult.contributions; powers the 'why did this match' UI panel.",
    )
    optimization_hint: str | None = Field(
        default=None,
        description="Optional bonus-feature note about the surfaced code, e.g. "
        "\"awaits inside a for-loop; consider Promise.all\". Never a code rewrite.",
    )

    @property
    def location_ref(self) -> str:
        """Convenience passthrough: 'src/agents/bluetooth.js:42-67'."""
        return self.chunk.location.as_ref()
```

### Fields

| Name | Type | Required | Description | Invariant |
|---|---|---|---|---|
| `chunk` | `Chunk` | yes | Full chunk payload | valid per §6 |
| `score` | `float` | yes | Final relevance | in `[0, 1]`; monotone non-increasing across the result list |
| `match_reason` | `str` | yes | Why it matched | non-empty; names a `SignalKind` |
| `signals` | `dict[SignalKind, int]` | yes | First-stage ranks | non-empty; ranks `>= 1` |
| `optimization_hint` | `str \| None` | no (`None`) | Bonus code note | `None` when `AXIOM_LLM_ENABLED=false` |

`score` is clamped to `[0, 1]` *after* the stability bonus is applied, because
`base * (1 + 0.10 * stability)` can reach 1.1 when `base` is near 1.0 and `stability` is 1.0.
Clamping rather than rescaling keeps the top of the list stable.

### Example

```json
{
  "chunk": { "chunk_id": "9f2c41d80ba7e35617c4d9a0e8b3f512", "...": "see §6" },
  "score": 0.8956,
  "match_reason": "structural: handleDeeplink calls preprocessInput before resolveTool (rank 2 in call-graph)",
  "signals": { "dense": 3, "sparse": 11, "structural": 2 },
  "optimization_hint": "resolveTool is invoked twice with the same argument; hoist the result"
}
```

---

## 10. `QueryPlan`

### Purpose

The agent's decision record for one query. Produced by `agent/classifier.py` +
`agent/planner.py`, consumed by fusion (for `strategy_weights`) and by the structural retriever
(for `extracted_identifiers`). Rewritten in place on each refinement pass, so the plan of the
final pass is what gets logged and shown in the demo UI.

```python
# src/axiom/schema/plan.py
from __future__ import annotations

from pydantic import Field, model_validator

from axiom.schema._base import PrismModel
from axiom.schema.enums import QueryType, SignalKind


class QueryPlan(PrismModel):
    """The agent's plan for one retrieval pass: classification, expansion, and weights."""

    original_query: str = Field(
        ...,
        min_length=1,
        description="The user's query, verbatim and never mutated across passes.",
    )
    query_type: QueryType = Field(
        ...,
        description="Classification result. Selects the default weight vector.",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Decomposition of a compound query. Empty means 'run the original "
        "query as-is'. Each sub-query is retrieved independently and RRF-merged.",
    )
    extracted_identifiers: list[str] = Field(
        default_factory=list,
        description="Code identifiers lifted out of the query text, e.g. 'resolveTool'. "
        "Fed to the structural retriever as exact symbol lookups.",
    )
    expansion_terms: list[str] = Field(
        default_factory=list,
        description="Added synonyms/related code vocabulary, e.g. 'preprocess' -> "
        "['normalize', 'sanitize', 'transform']. Appended to the sparse query only.",
    )
    strategy_weights: dict[SignalKind, float] = Field(
        ...,
        min_length=1,
        description="Per-signal RRF weights. Must sum to 1.0 within 1e-6 over the "
        "signals actually available for this query.",
    )

    @model_validator(mode="after")
    def _weights_normalised(self) -> "QueryPlan":
        total = sum(self.strategy_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"strategy_weights must sum to 1.0, got {total!r}")
        for signal, weight in self.strategy_weights.items():
            if weight < 0.0:
                raise ValueError(f"weight for {signal} must be >= 0, got {weight}")
        return self

    @property
    def effective_queries(self) -> list[str]:
        """Queries to actually retrieve: the sub-queries, or the original if none."""
        return self.sub_queries or [self.original_query]
```

### Fields

| Name | Type | Required | Description | Invariant |
|---|---|---|---|---|
| `original_query` | `str` | yes | Raw user query | non-empty; identical across all passes of one request |
| `query_type` | `QueryType` | yes | Classification | one of four |
| `sub_queries` | `list[str]` | no (`[]`) | Decomposition | each non-empty; `len <= 4` |
| `extracted_identifiers` | `list[str]` | no (`[]`) | Code identifiers | valid JS identifier shape; deduped, order preserved |
| `expansion_terms` | `list[str]` | no (`[]`) | Added vocabulary | deduped; disjoint from `extracted_identifiers` |
| `strategy_weights` | `dict[SignalKind, float]` | yes | RRF weights | non-empty; all `>= 0`; sums to `1.0 ± 1e-6` |

The sum-to-one invariant holds *after* renormalisation for empty signals. If the structural
retriever returns nothing, its weight is removed and the remaining weights are divided by their
new sum — the plan that gets logged reflects the renormalised vector, not the nominal one. See
[TechSpecifications.md](TechSpecifications.md#6-retrieval-spec).

### Example

```json
{
  "original_query": "How does the auth module validate tokens before calling the API?",
  "query_type": "hybrid",
  "sub_queries": [
    "auth module token validation",
    "functions called before the API request in auth"
  ],
  "extracted_identifiers": ["auth", "validateToken"],
  "expansion_terms": ["verify", "jwt", "decode", "authorize"],
  "strategy_weights": { "dense": 0.34, "sparse": 0.33, "structural": 0.33 }
}
```

---

## 11. `SnippetFamily`

### Purpose

The evolutionary-retrieval (Bonus) unit. A family is the set of near-identical chunks that are the
same logical snippet observed across multiple versions. Without families, a function edited by one
line across ten versions occupies ten of the top-ten slots. With families, it occupies one slot,
shows its newest form, and gains a small stability bonus for having survived many versions.

```python
# src/axiom/schema/version.py
from __future__ import annotations

from pydantic import Field, model_validator

from axiom.schema._base import PrismModel
from axiom.schema.chunk import Chunk


class SnippetFamily(PrismModel):
    """A group of near-identical chunks representing one snippet across versions."""

    family_id: str = Field(
        ...,
        pattern=r"^[0-9a-f]{32}$",
        description="blake2b-128 hex of (symbol, file_path). Stable across versions, "
        "so the same logical function keeps its family_id as the repo evolves.",
    )
    representative: Chunk = Field(
        ...,
        description="The member surfaced by default: the newest version's chunk.",
    )
    members: list[Chunk] = Field(
        ...,
        min_length=1,
        description="Every chunk in the family, sorted newest version first.",
    )
    versions: list[str] = Field(
        ...,
        min_length=1,
        description="version_ids covered by members, newest first. Same length and "
        "order as members.",
    )
    stability: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="len(members) / total indexed versions. 1.0 means the snippet "
        "exists in every version; 0.1 over 10 versions means it appeared once.",
    )
    diffs: list[str] = Field(
        default_factory=list,
        description="Unified-diff text between consecutive members, newest pair first. "
        "Length is len(members) - 1 when populated, or 0 when diffing is disabled.",
    )

    @model_validator(mode="after")
    def _family_consistent(self) -> "SnippetFamily":
        if self.versions != [m.metadata.version_id for m in self.members]:
            raise ValueError("versions must mirror members' version_ids in order")
        if self.representative.chunk_id != self.members[0].chunk_id:
            raise ValueError("representative must be members[0] (the newest member)")
        symbols = {(m.metadata.symbol, m.location.file_path) for m in self.members}
        if len(symbols) != 1:
            raise ValueError(f"family spans multiple (symbol, file_path) pairs: {symbols}")
        if self.diffs and len(self.diffs) != len(self.members) - 1:
            raise ValueError(
                f"expected {len(self.members) - 1} diffs, got {len(self.diffs)}"
            )
        return self

    @property
    def is_multi_version(self) -> bool:
        """True when the stability ranking bonus applies (family spans >= 2 versions)."""
        return len(self.versions) >= 2

    def ranking_bonus(self) -> float:
        """Multiplier applied to the family's base score: 1 + 0.10*stability, or 1.0."""
        return 1.0 + 0.10 * self.stability if self.is_multi_version else 1.0
```

### Fields

| Name | Type | Required | Description | Invariant |
|---|---|---|---|---|
| `family_id` | `str` | yes | Stable family identity | 32 hex chars; `blake2b128(symbol ⊕ file_path)` |
| `representative` | `Chunk` | yes | Default-surfaced member | identical to `members[0]` |
| `members` | `list[Chunk]` | yes | All family members | non-empty; newest-first; all share one `(symbol, file_path)`; pairwise cosine `>= 0.95` |
| `versions` | `list[str]` | yes | Covered versions | mirrors `members` order and length |
| `stability` | `float` | yes | Version coverage | in `(0, 1]`; `len(members) / total_versions` |
| `diffs` | `list[str]` | no (`[]`) | Consecutive diffs | empty, or exactly `len(members) - 1` entries |

The bonus only applies when `is_multi_version` is true. A single-version family has
`stability = 1/total_versions` which is *low*, not high — a brand-new snippet is novel, not
unstable, and multiplying its score by `1 + 0.10/N` would be noise. The gate makes the
single-version case exactly neutral.

### Example

```json
{
  "family_id": "b73e0c19af52d8461205e9c7fa3b6d80",
  "representative": { "chunk_id": "9f2c41d80ba7e35617c4d9a0e8b3f512", "...": "v2.3.1 chunk" },
  "members": [
    { "chunk_id": "9f2c41d80ba7e35617c4d9a0e8b3f512", "...": "v2.3.1, lines 42-67" },
    { "chunk_id": "41b8d0c97fe2a6530d84c1b7e90f2a35", "...": "v2.3.0, lines 42-65" },
    { "chunk_id": "7c0e5a2b91df4368a05e2c8d1b6f3049", "...": "v2.2.0, lines 38-60" }
  ],
  "versions": ["v2.3.1", "v2.3.0", "v2.2.0"],
  "stability": 0.75,
  "diffs": [
    "@@ -42,24 +42,26 @@\n+  if (!clean) return null;\n+  // added error handling",
    "@@ -38,23 +42,24 @@\n-  switch (host) {\n+  const handler = HOSTS[host];"
  ]
}
```

With four indexed versions and three members, `stability = 3/4 = 0.75`, so the ranking bonus is
`1 + 0.10 * 0.75 = 1.075`: a base rerank score of 0.81 becomes 0.871.

---

## 12. `VersionManifest`

### Purpose

The per-version index descriptor written to `.axiom/index/<version_id>/manifest.json`. It answers
three questions that must be answerable without loading any index: what model built this index (so
we refuse to query a MiniLM index with Qwen vectors), what files it covered and at what hash (so
incremental reindex can diff against it), and which version it descends from (so the evolutionary
layer can order versions without a git repo).

```python
class VersionManifest(PrismModel):
    """Descriptor for one built index version: provenance, coverage, and model identity."""

    version_id: str = Field(
        ...,
        min_length=1,
        description="Logical version label. Directory name under .axiom/index/.",
    )
    commit_sha: str | None = Field(
        default=None,
        description="Full 40-char git sha this version was built from, if any.",
    )
    created_at: str = Field(
        ...,
        min_length=20,
        description="ISO-8601 UTC timestamp of index completion, e.g. "
        "'2026-09-18T14:03:22Z'.",
    )
    chunk_count: int = Field(
        ...,
        ge=0,
        description="Number of lines in chunks.jsonl; equals the FAISS vector count.",
    )
    file_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="repo-relative POSIX path -> blake2b-128 hex of the whole file's "
        "normalised content. The diff basis for incremental reindexing.",
    )
    embedding_model: str = Field(
        ...,
        min_length=1,
        description="HF id of the embedder that produced dense.faiss, e.g. "
        "'Qwen/Qwen3-Embedding-0.6B'. Mismatch at query time is a hard error.",
    )
    embedding_dim: int = Field(
        ...,
        gt=0,
        description="Vector dimensionality: 1024 for Qwen3-0.6B, 384 for MiniLM.",
    )
    index_kind: str = Field(
        ...,
        pattern=r"^(flat_ip|ivf_pq)$",
        description="FAISS index type actually built: 'flat_ip' below 50k vectors, "
        "'ivf_pq' at or above 50k.",
    )
    parent_version: str | None = Field(
        default=None,
        description="version_id this index was incrementally derived from. None for a "
        "cold build. Forms a chain the evolutionary layer walks.",
    )

    @model_validator(mode="after")
    def _manifest_consistent(self) -> "VersionManifest":
        if self.parent_version == self.version_id:
            raise ValueError("parent_version must differ from version_id")
        expected_dim = {"Qwen/Qwen3-Embedding-0.6B": 1024,
                        "sentence-transformers/all-MiniLM-L6-v2": 384}
        known = expected_dim.get(self.embedding_model)
        if known is not None and known != self.embedding_dim:
            raise ValueError(
                f"{self.embedding_model} produces dim {known}, manifest says "
                f"{self.embedding_dim}"
            )
        for path in self.file_hashes:
            if "\\" in path or path.startswith("/"):
                raise ValueError(f"file_hashes key must be POSIX-relative: {path!r}")
        return self
```

### Fields

| Name | Type | Required | Description | Invariant |
|---|---|---|---|---|
| `version_id` | `str` | yes | Version label | non-empty; matches its directory name; a key in `registry.json` |
| `commit_sha` | `str \| None` | no (`None`) | Source commit | 40 hex chars or `None` |
| `created_at` | `str` | yes | Build completion time | ISO-8601 UTC with `Z` suffix |
| `chunk_count` | `int` | yes | Chunks indexed | `>= 0`; equals `chunks.jsonl` line count and FAISS `ntotal` |
| `file_hashes` | `dict[str, str]` | no (`{}`) | path → file hash | POSIX-relative keys; 32-hex values |
| `embedding_model` | `str` | yes | Embedder HF id | non-empty; must equal the runtime model at query time |
| `embedding_dim` | `int` | yes | Vector dim | `> 0`; consistent with `embedding_model` |
| `index_kind` | `str` | yes | FAISS index type | `"flat_ip"` or `"ivf_pq"` |
| `parent_version` | `str \| None` | no (`None`) | Derivation parent | `!= version_id`; no cycles in the chain |

### Example

```json
{
  "version_id": "v2.3.1",
  "commit_sha": "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
  "created_at": "2026-09-18T14:03:22Z",
  "chunk_count": 10248,
  "file_hashes": {
    "src/agents/bluetooth.js": "5e1c8a30bd97f24601a3c5d7e8b92f14",
    "src/utils/normalize.js": "cc02f7a15b83d96420e1a7c3d5b80f62"
  },
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "embedding_dim": 1024,
  "index_kind": "flat_ip",
  "parent_version": "v2.3.0"
}
```

---

## 13. Identity and hashing

All PRISM digests are **blake2b with `digest_size=16`** (128-bit), rendered as 32 lowercase hex
characters. blake2b is chosen over SHA-256 because it is faster on ARM and x86 without
hardware SHA extensions (which the CPU-only reference box may lack), and over MD5 because MD5's
collision weakness is an unnecessary liability in a content-addressed store. 128 bits gives a
collision probability under 10⁻²⁸ for a 10⁶-chunk corpus.

```python
# src/axiom/core/hashing.py
from __future__ import annotations

import hashlib

_DIGEST_SIZE = 16
_SEP = b"\x00"


def normalise_for_hash(content: str) -> str:
    """Canonicalise code text for content hashing.

    Three transformations, in order:
      1. line endings -> "\\n"  (CRLF and lone CR both collapse)
      2. trailing whitespace stripped from every line
      3. runs of blank lines collapsed to a single blank line, and leading/
         trailing blank lines removed entirely

    Comments are NOT stripped. Comments and JSDoc carry real retrieval signal --
    they are frequently the only natural-language text in a chunk and are what
    lets a dense embedding connect "preprocessing" to `normalize()`. A snippet
    whose only change is a reworded comment is a genuinely different snippet for
    retrieval purposes and must get a new content_hash.
    """
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    blank_run = 0
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if line:
            blank_run = 0
            out.append(line)
        else:
            blank_run += 1
            if blank_run == 1 and out:
                out.append("")
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def blake2b_128(*parts: bytes) -> str:
    """Domain-separated blake2b-128 hex digest over an ordered sequence of parts."""
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for index, part in enumerate(parts):
        if index:
            digest.update(_SEP)
        digest.update(part)
    return digest.hexdigest()


def compute_chunk_id(content: str, file_path: str, start_line: int) -> str:
    """Location-sensitive chunk identity: blake2b-128(content, file_path, start_line)."""
    return blake2b_128(
        content.encode("utf-8"),
        file_path.encode("utf-8"),
        str(start_line).encode("ascii"),
    )


def compute_content_hash(content: str) -> str:
    """Location-INsensitive content identity: blake2b-128(normalise_for_hash(content))."""
    return blake2b_128(normalise_for_hash(content).encode("utf-8"))


def compute_family_id(symbol: str | None, file_path: str) -> str:
    """Stable snippet-family identity: blake2b-128(symbol, file_path)."""
    return blake2b_128((symbol or "").encode("utf-8"), file_path.encode("utf-8"))


def compute_file_hash(content: str) -> str:
    """Whole-file hash used as the incremental-reindex diff basis."""
    return compute_content_hash(content)
```

### 13.1 Why `chunk_id` includes location and `content_hash` does not

| | `chunk_id` | `content_hash` |
|---|---|---|
| Inputs | raw `content`, `file_path`, `start_line` | normalised `content` only |
| Normalisation applied | no | yes |
| Changes when code moves within a file | yes | no |
| Changes when file is renamed | yes | no |
| Changes on a whitespace-only edit | yes | no |
| Role | primary key of a chunk in one version's index | cache key of an embedding in `.axiom/blobs/` |

`chunk_id` is an *address*: "this exact text, at this path, starting on this line". Two identical
utility functions copy-pasted into two files are two distinct chunks — the user must be shown both
locations — so they get distinct `chunk_id`s.

`content_hash` is a *value*: "this code, up to formatting". Those same two copy-pasted functions
have one `content_hash`, so the embedder runs once and both chunks read the same
`.axiom/blobs/<content_hash>.npy`. This is the single mechanism behind three otherwise separate
features:

1. **Intra-version dedup** — copy-pasted code embeds once.
2. **Cross-version dedup** — a file untouched between `v2.3.0` and `v2.3.1` contributes zero new
   embeddings to the `v2.3.1` build, because every chunk's `content_hash` already has a blob.
3. **Zero-cost renames** — `git diff --name-status` reports `R src/a.js src/b.js`. Every chunk gets
   a new `chunk_id` (path changed) but the same `content_hash`, so the rename costs one directory
   of new metadata and **zero** embedding forward passes.

If `content_hash` included `file_path`, none of the three would work. This is the reason the
contract specifies "normalised content only", and the reason that phrase must not be
"helpfully" broadened.

### 13.2 Normalisation worked example

Input (`\r\n` endings, trailing spaces shown as `·`, blank runs):

```
function f(x) {··\r\n
\r\n
\r\n
  return x + 1;·\r\n
}\r\n
\r\n
```

After `normalise_for_hash`:

```
function f(x) {

  return x + 1;
}
```

`content_hash` is computed over those 4 lines joined by `\n` with no trailing newline.
`chunk_id`, by contrast, hashes the raw CRLF text — so reformatting the file changes every
`chunk_id` in it while changing no `content_hash`, which is precisely the behaviour incremental
reindexing wants: new chunk rows, no new embeddings.

---

## 14. On-disk formats

Layout is fixed by `_CONTRACT.md` §6:

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

### 14.1 `registry.json`

Small, hand-readable, rewritten atomically (write to `registry.json.tmp`, `os.replace`). The only
mutable file in `.axiom/`; everything under `index/<version_id>/` is write-once.

```json
{
  "schema_version": 1,
  "active_version": "v2.3.1",
  "versions": {
    "v2.2.0": {
      "manifest": "index/v2.2.0/manifest.json",
      "created_at": "2026-09-16T09:12:40Z",
      "chunk_count": 9871,
      "parent_version": null
    },
    "v2.3.0": {
      "manifest": "index/v2.3.0/manifest.json",
      "created_at": "2026-09-17T11:44:05Z",
      "chunk_count": 10102,
      "parent_version": "v2.2.0"
    },
    "v2.3.1": {
      "manifest": "index/v2.3.1/manifest.json",
      "created_at": "2026-09-18T14:03:22Z",
      "chunk_count": 10248,
      "parent_version": "v2.3.0"
    }
  }
}
```

| Key | Type | Description | Invariant |
|---|---|---|---|
| `schema_version` | `int` | On-disk schema generation | currently `1`; see §16 |
| `active_version` | `str` | Default version for unqualified queries | must be a key of `versions` |
| `versions` | `object` | version_id → summary record | `manifest` path is relative to `.axiom/`; `parent_version` chain is acyclic |

The summary duplicates `created_at`, `chunk_count`, and `parent_version` from each manifest on
purpose: the CLI's `axiom versions` must list every version without opening N manifest files. The
manifest is authoritative; a mismatch is a corruption error, not a merge.

### 14.2 `manifest.json`

Exactly one serialised `VersionManifest` (§12), `model_dump_json(indent=2)`, UTF-8, trailing
newline. Written last in the build so its presence is the completion marker: a directory with
`chunks.jsonl` but no `manifest.json` is an aborted build and is deleted on the next run.

### 14.3 `chunks.jsonl`

One `Chunk` per line, `model_dump_json()` with no indentation, UTF-8, `\n` terminated, **no**
trailing blank line. Line order is the canonical chunk order and is identical to FAISS row order,
so `dense.idmap.json` is strictly a convenience/verification artifact rather than the source of
truth.

```
{"chunk_id":"9f2c41d8...","content_hash":"3ad1907c...","text":"export function handleDeeplink(url) {...}","location":{...},"metadata":{...}}
{"chunk_id":"41b8d0c9...","content_hash":"cc02f7a1...","text":"function preprocessInput(raw) {...}","location":{...},"metadata":{...}}
```

| Property | Rule |
|---|---|
| Encoding | UTF-8, no BOM |
| Line terminator | `\n` on every line including the last |
| Ordering | stable: sorted by `(file_path, start_line)` |
| Uniqueness | `chunk_id` unique across the file |
| Row correspondence | line `i` (0-indexed) ↔ FAISS row `i` |
| Streaming | consumers must read line-by-line; the file may exceed 200 MB |

Reading it in full is never necessary at query time: `retrieval/*.py` hydrate chunks by
`chunk_id` through a byte-offset sidecar index built lazily on first access and memo-cached.

### 14.4 `dense.idmap.json`

Maps FAISS integer row ids to `chunk_id`s. FAISS stores only `int64` labels, so this is the
indirection layer.

```json
{
  "schema_version": 1,
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "embedding_dim": 1024,
  "index_kind": "flat_ip",
  "count": 10248,
  "rows": [
    "9f2c41d80ba7e35617c4d9a0e8b3f512",
    "41b8d0c97fe2a6530d84c1b7e90f2a35",
    "7c0e5a2b91df4368a05e2c8d1b6f3049"
  ]
}
```

| Key | Type | Description | Invariant |
|---|---|---|---|
| `schema_version` | `int` | On-disk schema generation | `1` |
| `embedding_model` | `str` | Producing model HF id | equals `manifest.embedding_model` |
| `embedding_dim` | `int` | Vector dim | equals `manifest.embedding_dim` |
| `index_kind` | `str` | FAISS type | equals `manifest.index_kind` |
| `count` | `int` | Row count | equals `len(rows)`, `manifest.chunk_count`, and FAISS `ntotal` |
| `rows` | `list[str]` | row → `chunk_id` | positional; `rows[i]` is FAISS label `i`; all 32-hex; unique |

A `rows` array rather than an object keyed by stringified integers: the array is half the bytes,
parses faster, and makes the positional invariant structural rather than conventional.

### 14.5 `blobs/<content_hash>.npy`

One NumPy `.npy` file per distinct `content_hash`, shape `(embedding_dim,)`, dtype `float32`,
L2-normalised so that FAISS inner product equals cosine similarity. Shared across every version
and never deleted while any manifest references its hash. Blob files are immutable: a blob whose
name exists is assumed correct and is never rewritten, which is what makes the incremental build
idempotent and safely interruptible.

Two models producing two dims means blobs are model-scoped in practice; the guard is
`manifest.embedding_model`. Switching models requires a cold rebuild into a new `version_id` — see
§16.

### 14.6 `sparse.bm25s/`

Native `bm25s` on-disk format, written by `BM25.save(path)` and read by `BM25.load(path,
mmap=True)`. PRISM treats the directory as opaque with one addition: PRISM writes
`sparse.bm25s/prism_meta.json` alongside it recording the tokeniser configuration, because bm25s
does not persist the tokeniser and a query tokenised differently from the corpus silently returns
garbage.

```json
{
  "schema_version": 1,
  "tokenizer": "axiom.indexing.sparse:CodeTokenizer",
  "tokenizer_version": 1,
  "split_camel_case": true,
  "split_snake_case": true,
  "split_dots": true,
  "keep_intact_identifier": true,
  "lowercase": true,
  "stopwords": "none",
  "doc_count": 10248
}
```

`doc_count` must equal `manifest.chunk_count`; bm25s document index `i` corresponds to
`chunks.jsonl` line `i`, the same correspondence as FAISS rows.

### 14.7 `structural.sqlite`

SQLite 3, one file per version, created with `PRAGMA journal_mode=WAL` during the build and
checkpointed to `DELETE` mode on completion so the shipped artifact is a single file. The schema
generation is stored in `PRAGMA user_version` (currently `1`) rather than in a metadata table, so
the four data tables below are the complete schema.

```sql
PRAGMA user_version = 1;
PRAGMA foreign_keys = ON;

-- Every named code entity in this version: the symbol table.
CREATE TABLE symbols (
    symbol_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id      TEXT    NOT NULL,
    symbol        TEXT,                      -- NULL for anonymous functions / MODULE chunks
    kind          TEXT    NOT NULL
                  CHECK (kind IN ('function', 'method', 'class', 'module', 'block')),
    parent_symbol TEXT,
    file_path     TEXT    NOT NULL,          -- repo-relative, POSIX separators
    start_line    INTEGER NOT NULL CHECK (start_line >= 1),
    end_line      INTEGER NOT NULL CHECK (end_line >= start_line),
    start_byte    INTEGER NOT NULL CHECK (start_byte >= 0),
    end_byte      INTEGER NOT NULL CHECK (end_byte > start_byte),
    is_exported   INTEGER NOT NULL DEFAULT 0 CHECK (is_exported IN (0, 1)),
    language      TEXT    NOT NULL DEFAULT 'javascript',
    version_id    TEXT    NOT NULL,
    UNIQUE (chunk_id)
);

CREATE INDEX idx_symbols_symbol      ON symbols (symbol);
CREATE INDEX idx_symbols_file        ON symbols (file_path, start_line);
CREATE INDEX idx_symbols_parent      ON symbols (parent_symbol);
CREATE INDEX idx_symbols_exported    ON symbols (is_exported) WHERE is_exported = 1;

-- Call-graph edges. One row per call expression, ordered within the caller.
CREATE TABLE calls (
    call_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    caller_id   INTEGER NOT NULL
                REFERENCES symbols (symbol_id) ON DELETE CASCADE,
    callee_name TEXT    NOT NULL,            -- identifier as written at the call site
    callee_id   INTEGER
                REFERENCES symbols (symbol_id) ON DELETE SET NULL,  -- NULL if unresolved
    call_order  INTEGER NOT NULL CHECK (call_order >= 0),  -- 0-based, source order
    call_line   INTEGER NOT NULL CHECK (call_line >= 1),
    is_method   INTEGER NOT NULL DEFAULT 0 CHECK (is_method IN (0, 1)),
    receiver    TEXT,                         -- 'client' in client.send(x); NULL if bare
    UNIQUE (caller_id, call_order)
);

CREATE INDEX idx_calls_callee_name ON calls (callee_name);
CREATE INDEX idx_calls_callee_id   ON calls (callee_id);
CREATE INDEX idx_calls_caller_ord  ON calls (caller_id, call_order);

-- Import-graph edges. One row per imported binding, not per import statement.
CREATE TABLE imports (
    import_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT    NOT NULL,           -- importing file, repo-relative POSIX
    module_spec   TEXT    NOT NULL,           -- verbatim specifier: './normalize'
    resolved_path TEXT,                        -- repo-relative target, NULL if external
    imported_name TEXT,                        -- exported name; NULL for namespace imports
    local_alias   TEXT,                        -- local binding name
    import_kind   TEXT    NOT NULL
                  CHECK (import_kind IN ('esm_default', 'esm_named', 'esm_namespace',
                                         'cjs_require', 'dynamic')),
    import_line   INTEGER NOT NULL CHECK (import_line >= 1),
    version_id    TEXT    NOT NULL
);

CREATE INDEX idx_imports_file     ON imports (file_path);
CREATE INDEX idx_imports_resolved ON imports (resolved_path);
CREATE INDEX idx_imports_name     ON imports (imported_name);

-- Export map. Answers 'where is X used?' by pairing with imports.imported_name.
CREATE TABLE exports (
    export_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT    NOT NULL,           -- exporting file, repo-relative POSIX
    exported_name TEXT    NOT NULL,           -- 'default' for default exports
    local_name    TEXT,                        -- inner name when re-aliased on export
    symbol_id     INTEGER
                  REFERENCES symbols (symbol_id) ON DELETE SET NULL,
    export_kind   TEXT    NOT NULL
                  CHECK (export_kind IN ('default', 'named', 'namespace',
                                         'cjs_module', 'cjs_property')),
    export_line   INTEGER NOT NULL CHECK (export_line >= 1),
    version_id    TEXT    NOT NULL,
    UNIQUE (file_path, exported_name)
);

CREATE INDEX idx_exports_name   ON exports (exported_name);
CREATE INDEX idx_exports_symbol ON exports (symbol_id);
```

Table-by-table:

| Table | Grain | Populated from | Primary query it serves |
|---|---|---|---|
| `symbols` | one row per `Chunk` | `Chunk` + `ChunkMetadata` | "where is `handleDeeplink` defined?" |
| `calls` | one row per call expression | `ChunkMetadata.calls`, positionally | "which functions call `X` before `Y`?" |
| `imports` | one row per imported binding | `ChunkMetadata.imports` + per-file AST pass | "which files depend on `./normalize`?" |
| `exports` | one row per exported name | per-file AST export pass | "where is the Bluetooth deeplink used?" |

Design notes worth knowing before you touch this schema:

- `symbols.symbol_id` is a local `INTEGER` surrogate rather than `chunk_id TEXT` as the primary
  key. Integer joins over 10k+ call edges are measurably faster in SQLite than 32-char text joins,
  and `UNIQUE (chunk_id)` keeps the mapping to the Pydantic layer bijective.
- `calls.callee_id` is nullable because JavaScript call resolution is undecidable in general
  (dynamic dispatch, computed member access, callbacks). `callee_name` is always present;
  `resolved` status is simply `callee_id IS NOT NULL`. Unresolved edges are still useful — a
  name-only match is exactly what a `USAGE` query needs.
- `UNIQUE (caller_id, call_order)` enforces the contiguity of source order. If this constraint ever
  fires, the AST walker emitted calls out of order and every ordering predicate downstream is
  wrong.
- `version_id` is denormalised onto `symbols`, `imports`, and `exports` even though the whole file
  is one version. It costs a few hundred kilobytes and makes it possible to `ATTACH` several
  versions' databases and run one cross-version query for the evolutionary path without rewriting
  every statement.
- No foreign key from `symbols.chunk_id` to anything: `chunks.jsonl` is not a SQL table. Integrity
  between the two is checked by `axiom index verify`.

---

## 15. Invariants

These hold at every observable boundary. Each is checkable, and `axiom index verify` checks all of
them. A violation is a bug in PRISM, never an input-data problem to be tolerated.

1. **Ids are opaque and stable.** `chunk_id`, `content_hash`, and `family_id` are exactly 32
   lowercase hex characters. No component truncates, uppercases, prefixes, URL-escapes, or
   re-hashes them. An id that appears in a log, an API response, and a `.npy` filename is
   byte-identical in all three.
2. **Ids are recomputable.** For every `Chunk`, `chunk_id == compute_chunk_id(text,
   location.file_path, location.start_line)` and `content_hash == compute_content_hash(text)`.
3. **Line ranges are 1-indexed and inclusive.** `start_line >= 1` and `end_line >= start_line`.
   Byte ranges are 0-indexed and half-open: `end_byte > start_byte`.
4. **Text matches location.** `text == source_bytes[start_byte:end_byte].decode("utf-8")` for the
   file at `file_path` in version `version_id`.
5. **Paths are repo-relative POSIX.** Every `file_path`, every `file_hashes` key, and every
   `resolved_path` uses `/`, has no leading `/` or `./`, and contains no drive letter. Windows
   paths are converted at the single ingest boundary in `chunking/walker.py`, never later.
6. **Ranked lists are descending.** Any list of `ScoredChunk`, `FusedResult`, or `RetrievalResult`
   is sorted by its score field descending. Ties break by ascending `chunk_id`, making output
   deterministic across runs.
7. **Ranks are dense and 1-indexed.** Within one signal's list, `rank` takes exactly the values
   `1..N` with no gaps and no duplicates.
8. **Vectors are L2-normalised float32.** Every blob satisfies `abs(norm(v) - 1.0) < 1e-5` and has
   shape `(manifest.embedding_dim,)`. This is what makes FAISS inner product equal cosine.
9. **Weights sum to one.** `QueryPlan.strategy_weights` sums to `1.0 ± 1e-6` over the signals
   actually used, after empty-signal renormalisation.
10. **RRF score matches contributions.** `FusedResult.rrf_score == Σ w_s / (60 + contributions[s])`
    over exactly the keys present in `contributions`, to within `1e-9`.
11. **`dominant_signal` is a contributor.** It is always a key of `contributions`, and it is the key
    maximising `w_s / (60 + rank_s)`.
12. **Row correspondence is triple.** `chunks.jsonl` line `i` ↔ FAISS row `i` ↔ bm25s document `i`.
    `manifest.chunk_count == len(idmap.rows) == faiss.ntotal == bm25s doc_count`.
13. **One version per chunk.** Every `Chunk` has exactly one `metadata.version_id`, and it equals
    the `version_id` of the directory its `chunks.jsonl` lives in.
14. **Blobs are immutable and reference-counted.** A `blobs/<hash>.npy` is written once and never
    modified. It is deletable only when no manifest in `registry.json` covers a chunk with that
    `content_hash`.
15. **Families are homogeneous and newest-first.** All members of a `SnippetFamily` share one
    `(symbol, file_path)` pair, are pairwise cosine `>= 0.95`, are ordered newest version first,
    and `representative is members[0]`.
16. **Stability is a proper fraction.** `0 < stability <= 1`, equal to `len(members) /
    len(registry.versions)`.
17. **Public scores are bounded.** `RetrievalResult.score ∈ [0, 1]` after the stability bonus is
    applied and clamped. `FusedResult.rerank_score ∈ [0, 1]` or is `None`.
18. **Model identity is enforced, not assumed.** A query against version `V` fails loudly unless the
    runtime embedder's HF id and dim equal `V`'s manifest values. There is no silent reprojection.
19. **Manifest presence means build completion.** `manifest.json` is written last. A version
    directory without it is incomplete and is not registered in `registry.json`.
20. **`extra="forbid"` everywhere.** No schema model silently absorbs an unknown field. A forward-
    compatible reader is a schema-drift generator.

---

## 16. Migration and versioning policy

Two independent version numbers exist and must not be confused:

| Number | Where | Governs | Current |
|---|---|---|---|
| `schema_version` | `registry.json`, `dense.idmap.json`, `sparse.bm25s/prism_meta.json` | the JSON on-disk formats | `1` |
| `PRAGMA user_version` | `structural.sqlite` | the SQL schema | `1` |
| `version_id` | `registry.json`, `manifest.json` | a *corpus* version (user-facing) | n/a |

`version_id` identifies code being indexed. `schema_version` identifies the format PRISM writes.
They change for entirely unrelated reasons.

### 16.1 Change classification

| Class | Example | Requires | Reindex |
|---|---|---|---|
| **Additive-optional** | new `ChunkMetadata` field with a default | bump `schema_version`; readers of gen `N` refuse gen `N+1` (`extra="forbid"`) | next cold build; existing indexes stay queryable by the matching PRISM version |
| **Additive-required** | new required field on `Chunk` | bump `schema_version`; write a migration in `src/axiom/schema/migrations/` | full reindex |
| **Rename** | `calls` → `callees` | ADR in [Decisions.md](Decisions.md) + bump | full reindex |
| **Semantic** | `end_line` becomes exclusive | ADR + bump + explicit note in this document | full reindex |
| **Hash change** | different digest, or changing `normalise_for_hash` | bump; **invalidates every blob** | full reindex, blobs discarded |
| **Model change** | Qwen3 → MiniLM fallback | no schema bump; manifest records it | new `version_id`, cold build |

### 16.2 Rules

1. **No in-place mutation of a built index.** Versions are immutable. A schema change produces a
   new `version_id`, it never rewrites `index/v2.3.1/`.
2. **Readers refuse unknown generations.** Loading a `schema_version` greater than the running
   build's raises `PrismSchemaVersionError` with both numbers and the remediation command. Silent
   best-effort parsing is prohibited: it produces indexes that are subtly wrong rather than
   loudly broken.
3. **Hash functions are frozen for the build window.** `compute_chunk_id`,
   `compute_content_hash`, `compute_family_id`, and `normalise_for_hash` cannot change between
   2026-09-15 and 2026-09-27 without an ADR, because any change invalidates every blob in
   `.axiom/blobs/` and forces a 12-minute cold rebuild across four developers' machines.
4. **`schema_version` bumps are monotonic and global.** One counter for all JSON formats. A change
   to `dense.idmap.json` alone still bumps the shared number; version skew between sidecar files
   is not worth the bookkeeping at this scale.
5. **Migrations are forward-only scripts, not runtime shims.** `axiom index migrate --from 1 --to
   2` reads the old artifacts and writes a new `version_id`. There is no compatibility layer in
   `src/axiom/schema/`; the models describe exactly one generation.
6. **Field removal is deletion, not deprecation.** Remove the field, migrate every reader in the
   same commit. No aliases, no `deprecated=True`, no dual-read. With a 10-day window and four
   developers, a deprecation path is pure carrying cost.
7. **This document is updated in the same commit as the model.** A change to
   `src/axiom/schema/` that does not touch `docs/Schema.md` fails review.

### 16.3 Backward-compatibility guarantee for the submission

For the release tagged `PRISM_GENAI_HACKATHON_Y2026`, `schema_version` is `1` and
`PRAGMA user_version` is `1`. Any index built by any PRISM commit on the release tag is readable by
any other commit on that tag. That guarantee is what lets Parth publish
`appsretrieval_results.json` from an index built on a different machine than the one that runs the
live demo.

---

## 17. See also

| Document | Relationship |
|---|---|
| [TechSpecifications.md](TechSpecifications.md) | How each module produces and consumes these models |
| [Design.md](Design.md) | Why the model is shaped this way; fusion and degradation rationale |
| [APISpec.md](APISpec.md) | Wire representation of `RetrievalResult` and `QueryPlan` |
| [Decisions.md](Decisions.md) | ADR log; every schema change needs an entry |
| [TestPlan.md](TestPlan.md) | `TC-###` cases covering the §15 invariants |
| [OpenQuestions.md](OpenQuestions.md) | `OQ-##` items that may affect this schema |
