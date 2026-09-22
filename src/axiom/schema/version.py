"""Cross-version retrieval: snippet families and per-version index descriptors."""

from __future__ import annotations

from pydantic import Field, model_validator

from axiom.schema._base import AxiomModel
from axiom.schema.chunk import Chunk


class SnippetFamily(AxiomModel):
    """A group of near-identical chunks representing one snippet across versions.

    The stability bonus only applies when :attr:`is_multi_version` is true. A
    single-version family has ``stability = 1/total_versions`` which is *low*, not
    high -- a brand-new snippet is novel, not unstable -- so the gate makes the
    single-version case exactly neutral.
    """

    family_id: str = Field(
        ...,
        pattern=r"^[0-9a-f]{32}$",
        description="blake2b-128 hex of (symbol, file_path). Stable across versions, "
        "so the same logical function keeps its family_id as the repo evolves.",
    )
    representative: Chunk = Field(
        ..., description="The member surfaced by default: the newest version's chunk."
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
    def _family_consistent(self) -> SnippetFamily:
        if self.versions != [m.metadata.version_id for m in self.members]:
            raise ValueError("versions must mirror members' version_ids in order")
        if self.representative.chunk_id != self.members[0].chunk_id:
            raise ValueError("representative must be members[0] (the newest member)")
        symbols = {(m.metadata.symbol, m.location.file_path) for m in self.members}
        if len(symbols) != 1:
            raise ValueError(f"family spans multiple (symbol, file_path) pairs: {symbols}")
        if self.diffs and len(self.diffs) != len(self.members) - 1:
            raise ValueError(f"expected {len(self.members) - 1} diffs, got {len(self.diffs)}")
        return self

    @property
    def is_multi_version(self) -> bool:
        """True when the stability ranking bonus applies (family spans >= 2 versions)."""
        return len(self.versions) >= 2

    def ranking_bonus(self) -> float:
        """Multiplier applied to the family's base score: 1 + 0.10*stability, or 1.0."""
        return 1.0 + 0.10 * self.stability if self.is_multi_version else 1.0


class VersionManifest(AxiomModel):
    """Descriptor for one built index version: provenance, coverage, and model identity.

    Answers three questions without loading any index: what model built this index,
    what files it covered and at what hash, and which version it descends from.
    """

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
        description="ISO-8601 UTC timestamp of index completion, e.g. '2026-09-18T14:03:22Z'.",
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
        ..., gt=0, description="Vector dimensionality: 1024 for Qwen3-0.6B, 384 for MiniLM."
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
    def _manifest_consistent(self) -> VersionManifest:
        if self.parent_version == self.version_id:
            raise ValueError("parent_version must differ from version_id")
        expected_dim = {
            "Qwen/Qwen3-Embedding-0.6B": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
        }
        known = expected_dim.get(self.embedding_model)
        if known is not None and known != self.embedding_dim:
            raise ValueError(
                f"{self.embedding_model} produces dim {known}, manifest says {self.embedding_dim}"
            )
        for path in self.file_hashes:
            if "\\" in path or path.startswith("/"):
                raise ValueError(f"file_hashes key must be POSIX-relative: {path!r}")
        return self
