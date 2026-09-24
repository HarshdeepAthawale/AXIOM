"""The atomic unit of retrieval and its two identities."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from axiom.schema._base import AxiomModel
from axiom.schema.enums import ChunkKind


class ChunkLocation(AxiomModel):
    """Exact source location of a chunk: repo-relative path plus line and byte span.

    Note the asymmetry: **line range is inclusive, byte range is half-open.**
    Lines are inclusive because that is what a human reading ``bluetooth.js:42-67``
    expects. Bytes are half-open because that is what tree-sitter emits and what
    Python slicing wants.
    """

    file_path: str = Field(
        ...,
        min_length=1,
        description="Repo-relative path with POSIX ('/') separators. Never absolute, "
        "never Windows-style, never prefixed with './'.",
    )
    start_line: int = Field(..., ge=1, description="First line of the chunk. 1-indexed, inclusive.")
    end_line: int = Field(..., ge=1, description="Last line of the chunk. 1-indexed, inclusive.")
    start_byte: int = Field(
        ..., ge=0, description="Byte offset of the first byte of the chunk in the UTF-8 file."
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
    def _spans_ordered(self) -> ChunkLocation:
        if self.end_line < self.start_line:
            raise ValueError(f"end_line {self.end_line} precedes start_line {self.start_line}")
        if self.end_byte <= self.start_byte:
            raise ValueError(f"end_byte {self.end_byte} must exceed start_byte {self.start_byte}")
        return self

    @property
    def line_count(self) -> int:
        """Number of source lines covered, inclusive of both endpoints."""
        return self.end_line - self.start_line + 1

    def as_ref(self) -> str:
        """Human-facing location string, e.g. 'src/agents/bluetooth.js:42-67'."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class ChunkMetadata(AxiomModel):
    """Retrieval-relevant facts about a chunk: symbol identity, graph edges, provenance."""

    symbol: str | None = Field(
        default=None,
        description="Declared name of the function/method/class. None for anonymous "
        "arrow functions assigned to nothing, and for MODULE chunks.",
    )
    kind: ChunkKind = Field(..., description="Syntactic category of the originating AST node.")
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
        description="Logical version this chunk belongs to, e.g. 'v2.3.1' or a commit sha.",
    )
    commit_sha: str | None = Field(
        default=None, description="Full 40-char git commit sha, when the corpus is a git repo."
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
    def _method_has_parent(self) -> ChunkMetadata:
        if self.kind is ChunkKind.METHOD and not self.parent_symbol:
            raise ValueError("METHOD chunks must carry parent_symbol")
        return self


class Chunk(AxiomModel):
    """One retrievable code snippet: text, exact location, and retrieval metadata.

    Content-addressed twice over -- once by location-sensitive ``chunk_id``, once
    by location-insensitive ``content_hash``. That double addressing is the
    mechanism behind both incremental reindexing and cross-version embedding
    dedup (Schema.md section 13.1).
    """

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
    def _ids_match_content(self) -> Chunk:
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
        """Filename of this chunk's cached embedding under ``.axiom/blobs/``."""
        return f"{self.content_hash}.npy"

    @classmethod
    def create(
        cls,
        text: str,
        location: ChunkLocation,
        metadata: ChunkMetadata,
    ) -> Chunk:
        """Build a chunk, deriving both identities from the text and location.

        The only sanctioned construction path outside of index deserialisation --
        it makes it impossible to mint a chunk whose ids disagree with its body.
        """
        from axiom.core.hashing import compute_chunk_id, compute_content_hash

        return cls(
            chunk_id=compute_chunk_id(text, location.file_path, location.start_line),
            content_hash=compute_content_hash(text),
            text=text,
            location=location,
            metadata=metadata,
        )
