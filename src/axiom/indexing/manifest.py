"""The on-disk index layout: registry, per-version manifests, and the blob cache.

Three artefacts live behind this module, and the split between them is the entire
reason ``.axiom/`` is safely interruptible (TechSpecifications.md section 7):

* ``index/<version_id>/manifest.json`` is written **last** in a build, so its
  presence is the completion marker. A version directory carrying
  ``chunks.jsonl`` but no manifest is an aborted build and is discarded.
* ``registry.json`` is the only mutable file in the tree. It is rewritten
  atomically (tmp + :func:`os.replace`), and it duplicates ``created_at``,
  ``chunk_count`` and ``parent_version`` from each manifest on purpose so that
  ``axiom versions`` never has to open N manifest files (Schema.md section 14.1).
* ``blobs/<content_hash>.npy`` is the content-addressed embedding cache --
  write-once, immutable, shared across versions. This is the one sanctioned
  exception to stage purity (Rules.md Rule 2): a cache entry is invalidated by
  model identity, never by time (Rules.md AP-09), which is what makes an
  unchanged-content rename cost zero embedding forward passes (FR-19, TC-075).

``numpy`` is imported lazily inside the blob functions only. Everything else here
is stdlib plus pydantic, so ``import axiom.indexing.manifest`` costs nothing and
works on a bare install (NFR-07).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from axiom.config import Settings
from axiom.core.errors import AxiomContractError, IndexNotFoundError
from axiom.core.logging import get_logger, log_degradation
from axiom.schema import Chunk, VersionManifest

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np

_LOG = get_logger("indexing.manifest")

#: Generation of the on-disk JSON formats (Schema.md sections 14.1/14.4).
REGISTRY_SCHEMA_VERSION = 1

#: A ``version_id`` is a directory name under ``.axiom/index/``, so it is also an
#: injection surface: ``--version ../../etc`` must never reach the filesystem.
#: Pattern is copied verbatim from API.md section 3.1 so the CLI, the HTTP layer
#: and this module all reject exactly the same strings (TC-071).
VERSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: Tolerance for the L2-norm invariant on a stored vector (Schema.md section 15.8).
_NORM_TOLERANCE = 1e-5


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def index_root(settings: Settings) -> Path:
    """Root of the on-disk index tree, normally ``.axiom/``."""
    return Path(settings.index_root)


def versions_dir(settings: Settings) -> Path:
    """Directory holding one subdirectory per built version."""
    return index_root(settings) / "index"


def version_dir(settings: Settings, version_id: str) -> Path:
    """Directory of one built version. Raises on a traversal-shaped ``version_id``."""
    return versions_dir(settings) / require_valid_version_id(version_id)


def manifest_path(settings: Settings, version_id: str) -> Path:
    """Path of a version's ``manifest.json`` -- the build-completion marker."""
    return version_dir(settings, version_id) / "manifest.json"


def chunks_path(settings: Settings, version_id: str) -> Path:
    """Path of a version's ``chunks.jsonl``, whose line order is FAISS row order."""
    return version_dir(settings, version_id) / "chunks.jsonl"


def registry_path(settings: Settings) -> Path:
    """Path of the registry -- the only mutable file under the index root."""
    return index_root(settings) / "registry.json"


def blobs_dir(settings: Settings) -> Path:
    """Directory of the shared, content-addressed embedding cache."""
    return index_root(settings) / "blobs"


def blob_path(settings: Settings, content_hash: str) -> Path:
    """Path of one cached embedding. Keyed by content only, never by path or mtime."""
    return blobs_dir(settings) / f"{content_hash}.npy"


def is_valid_version_id(version_id: str) -> bool:
    """True when ``version_id`` is safe to use as a directory name."""
    return bool(VERSION_ID_PATTERN.match(version_id))


def require_valid_version_id(version_id: str) -> str:
    """Return ``version_id`` unchanged, or refuse it before it reaches the filesystem.

    Refusing raises :class:`IndexNotFoundError` rather than a validation error: to
    every caller a path-traversal-shaped version is indistinguishable from a
    version that simply does not exist, and neither one is allowed to touch disk.
    """
    if not is_valid_version_id(version_id):
        raise IndexNotFoundError(f"unusable version_id {version_id!r}")
    return version_id


def index_kind_for(chunk_count: int, settings: Settings) -> str:
    """FAISS index type implied by a corpus size, per the locked 50k threshold.

    Recorded in the manifest rather than inferred at query time, because the
    loader must be able to refuse a mismatched index without opening it.
    """
    return "ivf_pq" if chunk_count >= settings.faiss_ivf_threshold else "flat_ip"


def utc_now_iso() -> str:
    """Current UTC instant as ``YYYY-MM-DDTHH:MM:SSZ``.

    Wall-clock time is allowed here and nowhere near ranking (NFR-08): a manifest
    timestamp is provenance, not an input to a score.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Atomic writes
# --------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` via a sibling tmp file and :func:`os.replace`.

    A half-written ``registry.json`` would orphan every indexed version, so the
    rename -- atomic within a filesystem -- is what guarantees a reader sees
    either the old file or the new one, never a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def commit_version_dir(tmp_dir: Path, final_dir: Path) -> Path:
    """Publish a staged version directory, replacing any stale build in place.

    Builds stage into ``index/<version>.tmp/`` and land here (Appflow.md Flow 1,
    TC-034). POSIX ``rename`` refuses a non-empty destination, so an existing
    directory is removed first; that window is acceptable because a version
    directory without ``manifest.json`` is by definition discardable.
    """
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_dir, final_dir)
    return final_dir


# --------------------------------------------------------------------------
# VersionManifest
# --------------------------------------------------------------------------


def build_manifest(
    version_id: str,
    chunk_count: int,
    settings: Settings,
    *,
    file_hashes: dict[str, str] | None = None,
    commit_sha: str | None = None,
    parent_version: str | None = None,
    created_at: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    index_kind: str | None = None,
) -> VersionManifest:
    """Assemble a :class:`VersionManifest` from a finished build.

    Model identity defaults to the *running* settings rather than to whatever the
    parent version used: a manifest must describe the vectors actually written, or
    invariant 18 (model identity is enforced, not assumed) is unenforceable.
    """
    return VersionManifest(
        version_id=require_valid_version_id(version_id),
        commit_sha=commit_sha,
        created_at=created_at or utc_now_iso(),
        chunk_count=chunk_count,
        file_hashes=dict(sorted((file_hashes or {}).items())),
        embedding_model=embedding_model or settings.embedding_model,
        embedding_dim=embedding_dim or settings.embedding_dim,
        index_kind=index_kind or index_kind_for(chunk_count, settings),
        parent_version=parent_version,
    )


def write_manifest(
    manifest: VersionManifest, settings: Settings, *, target: Path | None = None
) -> Path:
    """Serialise a manifest to ``manifest.json``. Write it LAST in a build.

    ``target`` lets a builder write into its staging directory before the whole
    version directory is renamed into place; by default the manifest lands in the
    version's published directory.
    """
    path = target if target is not None else manifest_path(settings, manifest.version_id)
    return _atomic_write_text(path, manifest.model_dump_json(indent=2) + "\n")


def read_manifest(settings: Settings, version_id: str) -> VersionManifest:
    """Load one version's manifest.

    A missing manifest raises :class:`IndexNotFoundError` -- one of the three
    sanctioned raise categories (Rules.md Rule 3), since it means the requested
    index artefact is absent, not that a user typed something odd. A manifest that
    exists but does not parse is corrupted state, which is a contract violation.
    """
    path = manifest_path(settings, version_id)
    if not path.is_file():
        raise IndexNotFoundError(f"no manifest for version {version_id!r} at {path}")
    try:
        return VersionManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except IndexNotFoundError:
        raise
    except Exception as exc:
        raise AxiomContractError(f"corrupt manifest at {path}: {exc}") from exc


def try_read_manifest(settings: Settings, version_id: str | None) -> VersionManifest | None:
    """Best-effort manifest load for degradation paths.

    The incremental reindexer asks "is there a parent I can diff against?" and
    must treat "no" as an ordinary answer, not an error -- it simply drops to the
    next rung of the ladder (a full reindex).
    """
    if not version_id:
        return None
    try:
        return read_manifest(settings, version_id)
    except (IndexNotFoundError, AxiomContractError) as exc:
        log_degradation(_LOG, "manifest.read", f"{version_id}: {exc}", "no parent manifest")
        return None


def verify_compatible(manifest: VersionManifest, settings: Settings) -> None:
    """Refuse to query an index built by a different embedder (invariant 18).

    There is no silent reprojection between a 384-dim MiniLM index and 1024-dim
    Qwen3 vectors; the failure it would otherwise produce is a plausible-looking
    ranking that is entirely noise, which is far more expensive than a crash.
    """
    if manifest.embedding_model != settings.embedding_model:
        raise AxiomContractError(
            f"version {manifest.version_id} was built with {manifest.embedding_model!r}, "
            f"runtime embedder is {settings.embedding_model!r}"
        )
    if manifest.embedding_dim != settings.embedding_dim:
        raise AxiomContractError(
            f"version {manifest.version_id} has dim {manifest.embedding_dim}, "
            f"runtime embedder produces {settings.embedding_dim}"
        )


# --------------------------------------------------------------------------
# registry.json
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionRecord:
    """One row of ``registry.json``: a manifest's summary, duplicated on purpose.

    The manifest stays authoritative (Schema.md section 14.1); this record exists
    so listing versions is one file read rather than N.
    """

    version_id: str
    manifest: str
    created_at: str
    chunk_count: int
    parent_version: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Registry-shaped dict; ``version_id`` is the key, so it is not repeated."""
        return {
            "manifest": self.manifest,
            "created_at": self.created_at,
            "chunk_count": self.chunk_count,
            "parent_version": self.parent_version,
        }


@dataclass
class Registry:
    """In-memory view of ``registry.json``."""

    active_version: str | None = None
    versions: dict[str, VersionRecord] = field(default_factory=dict)
    schema_version: int = REGISTRY_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        """Serialise with versions in sorted key order, so the file is diffable."""
        return {
            "schema_version": self.schema_version,
            "active_version": self.active_version,
            "versions": {vid: self.versions[vid].to_json() for vid in sorted(self.versions)},
        }

    def ordered(self) -> list[VersionRecord]:
        """Records oldest first, by ``(created_at, version_id)``.

        The second key is not decorative: two versions built inside the same
        second would otherwise order non-deterministically (NFR-08).
        """
        return sorted(self.versions.values(), key=lambda r: (r.created_at, r.version_id))

    def newest_first(self) -> list[str]:
        """Version ids newest first, following ``parent_version`` where it is intact.

        Walking the parent chain beats sorting by timestamp because the chain is
        what the build actually recorded; timestamps are only the tie-break for
        versions the chain does not reach (a cold rebuild of an unrelated tree).
        """
        if not self.versions:
            return []
        children = {
            r.parent_version: r.version_id for r in self.versions.values() if r.parent_version
        }
        chain: list[str] = []
        head = self.active_version if self.active_version in self.versions else None
        if head is None:
            oldest = self.ordered()
            head = oldest[-1].version_id
        # Walk forward from head in case the active pointer is not the newest.
        cursor: str | None = head
        seen: set[str] = set()
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            cursor = children.get(cursor)
            if cursor is not None:
                head = cursor
        cursor = head
        seen = set()
        while cursor is not None and cursor in self.versions and cursor not in seen:
            seen.add(cursor)
            chain.append(cursor)
            cursor = self.versions[cursor].parent_version
        remainder = [r.version_id for r in reversed(self.ordered()) if r.version_id not in seen]
        return chain + remainder


def load_registry(settings: Settings) -> Registry:
    """Read ``registry.json``, or return an empty registry when none exists yet.

    A missing registry is the normal state before the first build, so it is not an
    error. A registry that exists but does not parse is corrupted state and raises
    -- overwriting it with an empty one would silently orphan every built version.
    """
    path = registry_path(settings)
    if not path.is_file():
        return Registry()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        versions: dict[str, VersionRecord] = {}
        for vid, rec in (raw.get("versions") or {}).items():
            versions[vid] = VersionRecord(
                version_id=vid,
                manifest=rec.get("manifest", f"index/{vid}/manifest.json"),
                created_at=rec.get("created_at", ""),
                chunk_count=int(rec.get("chunk_count", 0)),
                parent_version=rec.get("parent_version"),
            )
        return Registry(
            active_version=raw.get("active_version"),
            versions=versions,
            schema_version=int(raw.get("schema_version", REGISTRY_SCHEMA_VERSION)),
        )
    except Exception as exc:
        raise IndexNotFoundError(f"corrupt registry at {path}: {exc}") from exc


def save_registry(registry: Registry, settings: Settings) -> Path:
    """Rewrite ``registry.json`` atomically."""
    return _atomic_write_text(
        registry_path(settings), json.dumps(registry.to_json(), indent=2) + "\n"
    )


def register_version(
    manifest: VersionManifest,
    settings: Settings,
    *,
    make_active: bool = True,
) -> Registry:
    """Add (or refresh) a version's registry row after its manifest landed.

    Called only once the manifest exists on disk: invariant 19 says a version
    directory without ``manifest.json`` is incomplete and must not be registered.
    """
    registry = load_registry(settings)
    relative = PurePosixPath("index") / manifest.version_id / "manifest.json"
    registry.versions[manifest.version_id] = VersionRecord(
        version_id=manifest.version_id,
        manifest=str(relative),
        created_at=manifest.created_at,
        chunk_count=manifest.chunk_count,
        parent_version=manifest.parent_version,
    )
    if make_active or registry.active_version is None:
        registry.active_version = manifest.version_id
    save_registry(registry, settings)
    return registry


def finalize_version(
    manifest: VersionManifest,
    settings: Settings,
    *,
    make_active: bool = True,
) -> Path:
    """Write the manifest and register it -- the last two steps of every build."""
    path = write_manifest(manifest, settings)
    register_version(manifest, settings, make_active=make_active)
    return path


def set_active_version(settings: Settings, version_id: str) -> Registry:
    """Point the registry at another indexed version. O(1), never a reindex (TC-079)."""
    registry = load_registry(settings)
    if version_id not in registry.versions:
        raise IndexNotFoundError(f"version {version_id!r} is not in the registry")
    registry.active_version = version_id
    save_registry(registry, settings)
    return registry


def remove_version(settings: Settings, version_id: str, *, delete_files: bool = False) -> Registry:
    """Drop a version from the registry, optionally deleting its directory.

    Blobs are deliberately left alone: they are shared, reference-counted state
    (invariant 14), and deciding which are now unreferenced is :func:`gc_blobs`'s
    job, run separately so ``axiom version rm`` stays instant and reversible.
    """
    registry = load_registry(settings)
    registry.versions.pop(version_id, None)
    if registry.active_version == version_id:
        remaining = registry.newest_first()
        registry.active_version = remaining[0] if remaining else None
    save_registry(registry, settings)
    if delete_files:
        target = version_dir(settings, version_id)
        if target.is_dir():
            shutil.rmtree(target)
    return registry


def list_versions(settings: Settings) -> list[VersionRecord]:
    """Every indexed version, oldest first -- the body of ``GET /v1/versions``."""
    return load_registry(settings).ordered()


def resolve_version(settings: Settings, version_id: str | None = None) -> str:
    """Resolve the version a query should run against (FR-20).

    Absent an explicit id, the registry's active version is used. A version whose
    directory carries a manifest still resolves even if the registry lost its row,
    because the manifest -- not the registry -- is authoritative about what was
    actually built (Schema.md section 14.1).
    """
    if version_id is not None:
        require_valid_version_id(version_id)
        if manifest_path(settings, version_id).is_file():
            return version_id
        raise IndexNotFoundError(f"version {version_id!r} has no manifest on disk")

    registry = load_registry(settings)
    active = registry.active_version
    if active and manifest_path(settings, active).is_file():
        return active
    if active:
        log_degradation(
            _LOG,
            "manifest.resolve_version",
            f"active version {active!r} has no manifest on disk",
            "newest registered version with a manifest",
        )
    for candidate in registry.newest_first():
        if manifest_path(settings, candidate).is_file():
            return candidate
    raise IndexNotFoundError(f"no built index under {index_root(settings)}")


# --------------------------------------------------------------------------
# chunks.jsonl
# --------------------------------------------------------------------------


def canonical_chunk_order(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Sort chunks into the canonical on-disk order, dropping duplicate ids.

    ``(file_path, start_line, chunk_id)`` is a total order, so an incremental
    build and a full rebuild of the same tree emit byte-identical files (TC-076).
    The id is the last key purely to break the tie two chunks starting on the same
    line would otherwise leave open.
    """
    ordered = sorted(
        chunks, key=lambda c: (c.location.file_path, c.location.start_line, c.chunk_id)
    )
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in ordered:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique.append(chunk)
    if len(unique) != len(ordered):
        _LOG.warning("dropped %d duplicate chunk_ids before write", len(ordered) - len(unique))
    return unique


def write_chunks(chunks: Sequence[Chunk], path: Path) -> int:
    """Write ``chunks.jsonl``: one chunk per line, newline-terminated, no blank tail.

    An empty corpus produces a present, zero-byte file rather than no file at all
    (TC-078) -- "the index exists and is empty" and "the build never ran" must
    stay distinguishable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json())
            handle.write("\n")
            count += 1
    os.replace(tmp, path)
    return count


def iter_chunk_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream raw chunk dicts from a ``chunks.jsonl``, skipping unparseable lines.

    Raw dicts rather than models because the callers that stream the whole file
    (blob GC, id sets) only need one or two fields, and ``chunks.jsonl`` may
    exceed 200 MB.
    """
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                log_degradation(
                    _LOG, "manifest.chunks", f"{path}:{lineno} {exc}", "skipped that line"
                )
                continue
            if isinstance(record, dict):
                yield record


def read_chunks(path: Path) -> list[Chunk]:
    """Load and validate a whole ``chunks.jsonl``.

    A line that fails validation is dropped with a loud log rather than aborting
    the load: one corrupt row must not cost the other 10,247 (Rules.md Rule 3).
    """
    chunks: list[Chunk] = []
    for index, record in enumerate(iter_chunk_records(path)):
        try:
            chunks.append(Chunk.model_validate(record))
        except Exception as exc:
            log_degradation(_LOG, "manifest.chunks", f"{path} row {index}: {exc}", "skipped row")
    return chunks


def read_version_chunks(settings: Settings, version_id: str) -> list[Chunk]:
    """Load one indexed version's chunk corpus."""
    return read_chunks(chunks_path(settings, version_id))


# --------------------------------------------------------------------------
# Content-addressed blob cache
# --------------------------------------------------------------------------


def _load_numpy() -> Any | None:
    """Import numpy, or declare the blob cache unavailable.

    numpy is a core dependency, but the cache is still written to degrade: a build
    that cannot cache vectors is slower, not wrong, and NFR-07 says the pipeline
    must survive a missing library rather than fail to import.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a core dependency
        log_degradation(_LOG, "blob_cache", f"numpy unavailable: {exc}", "cache disabled")
        return None
    return np


def blob_exists(settings: Settings, content_hash: str) -> bool:
    """True when this content already has a cached vector. The rename fast path."""
    return blob_path(settings, content_hash).is_file()


def load_blob(
    settings: Settings, content_hash: str, *, expected_dim: int | None = None
) -> np.ndarray | None:
    """Read one cached vector, or ``None`` on a miss.

    A dimension mismatch is a hard contract violation (Rules.md AP-09): it means
    blobs from two different embedders were mixed into one cache, and every score
    computed from them would be meaningless.
    """
    np_mod = _load_numpy()
    if np_mod is None:
        return None
    path = blob_path(settings, content_hash)
    if not path.is_file():
        return None
    try:
        vector = np_mod.load(path)
    except Exception as exc:
        log_degradation(_LOG, "blob_cache", f"unreadable blob {path}: {exc}", "cache miss")
        return None
    want = expected_dim if expected_dim is not None else settings.embedding_dim
    if vector.ndim != 1 or int(vector.shape[0]) != want:
        raise AxiomContractError(
            f"blob {content_hash} has shape {tuple(vector.shape)}, expected ({want},)"
        )
    return vector


def store_blob(
    settings: Settings,
    content_hash: str,
    vector: Sequence[float] | np.ndarray,
    *,
    expected_dim: int | None = None,
) -> Path | None:
    """Cache one vector, write-once. Returns ``None`` if the cache is unavailable.

    An existing blob is never rewritten -- immutability is what makes an
    interrupted build resumable and the incremental path idempotent (Schema.md
    section 14.5). The L2-norm check enforces invariant 8 here, at the single
    write point, so FAISS inner product is cosine everywhere downstream.
    """
    np_mod = _load_numpy()
    if np_mod is None:
        return None
    path = blob_path(settings, content_hash)
    if path.is_file():
        return path
    array = np_mod.asarray(vector, dtype="float32").reshape(-1)
    want = expected_dim if expected_dim is not None else settings.embedding_dim
    if int(array.shape[0]) != want:
        raise AxiomContractError(
            f"refusing to cache a {int(array.shape[0])}-dim vector as {content_hash}; "
            f"the active embedder produces {want}"
        )
    norm = float(np_mod.linalg.norm(array))
    if abs(norm - 1.0) > _NORM_TOLERANCE:
        raise AxiomContractError(
            f"vector for {content_hash} has L2 norm {norm:.6f}; blobs must be normalised"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy.save appends ".npy" unless the name already ends in it, so the staging
    # name has to keep that suffix or the rename would chase the wrong file.
    tmp = path.with_name(path.name + ".tmp.npy")
    np_mod.save(tmp, array)
    os.replace(tmp, path)
    return path


def referenced_content_hashes(settings: Settings) -> set[str]:
    """Every ``content_hash`` any registered version still depends on.

    Streams each ``chunks.jsonl`` instead of loading it: this runs over the whole
    corpus of every version, and the files are the largest artefacts we produce.
    """
    referenced: set[str] = set()
    registry = load_registry(settings)
    for version_id in sorted(registry.versions):
        for record in iter_chunk_records(chunks_path(settings, version_id)):
            digest = record.get("content_hash")
            if isinstance(digest, str):
                referenced.add(digest)
    return referenced


def gc_blobs(settings: Settings, *, dry_run: bool = True) -> tuple[list[str], int]:
    """Delete blobs no registered version references (TC-080).

    Returns the deleted hashes (sorted, so the report is deterministic) and the
    bytes reclaimed. ``dry_run`` mutates nothing, which is the default because the
    blob cache is the expensive artefact in the tree -- deleting it wrongly costs
    a full re-embed.
    """
    directory = blobs_dir(settings)
    if not directory.is_dir():
        return [], 0
    referenced = referenced_content_hashes(settings)
    removed: list[str] = []
    freed = 0
    for path in sorted(directory.glob("*.npy")):
        digest = path.stem
        if digest in referenced:
            continue
        freed += path.stat().st_size
        removed.append(digest)
        if not dry_run:
            path.unlink()
    if removed:
        _LOG.info(
            "blob gc %s %d unreferenced blobs (%d bytes)",
            "would remove" if dry_run else "removed",
            len(removed),
            freed,
        )
    return removed, freed


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "VERSION_ID_PATTERN",
    "Registry",
    "VersionRecord",
    "blob_exists",
    "blob_path",
    "blobs_dir",
    "build_manifest",
    "canonical_chunk_order",
    "chunks_path",
    "commit_version_dir",
    "finalize_version",
    "gc_blobs",
    "index_kind_for",
    "index_root",
    "is_valid_version_id",
    "iter_chunk_records",
    "list_versions",
    "load_blob",
    "load_registry",
    "manifest_path",
    "read_chunks",
    "read_manifest",
    "read_version_chunks",
    "referenced_content_hashes",
    "register_version",
    "registry_path",
    "remove_version",
    "require_valid_version_id",
    "resolve_version",
    "save_registry",
    "set_active_version",
    "store_blob",
    "try_read_manifest",
    "utc_now_iso",
    "verify_compatible",
    "version_dir",
    "versions_dir",
    "write_chunks",
    "write_manifest",
]
