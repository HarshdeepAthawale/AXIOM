"""Incremental reindex: touch only what the diff says changed.

The whole module exists to make one number small. A cold build of a 10k-file
repository is budgeted at 12 minutes (NFR-01) and is dominated by embedding
forward passes; a 50-file commit must reindex in 45 seconds (NFR-02). The only
way to get there is to never re-embed content that already has a vector, and the
mechanism is the double identity every :class:`~axiom.schema.Chunk` carries
(Schema.md section 13.1):

* ``chunk_id`` = digest(text, file_path, start_line) -- changes when a file moves,
  so a renamed file genuinely needs new chunk records;
* ``content_hash`` = digest(normalised text) -- does **not** change when a file
  moves, so those new chunk records hit blobs that already exist.

A pure ``git mv`` therefore costs re-chunking and zero embedding calls (FR-19,
TC-075), and an incremental build of a tree is byte-equivalent to a cold build of
the same tree (TC-076) because both paths chunk with the same chunker, stamp the
same ``version_id``, and write in the same canonical order.

Every heavy dependency is optional and resolved lazily. With no chunker, no
embedder and no index builders installed, ``reindex`` still produces a valid,
registered version whose ``chunks.jsonl`` and ``manifest.json`` are correct --
it just holds whole-file MODULE chunks and writes no vectors, and says so in the
report's ``degradations``.
"""

from __future__ import annotations

import importlib
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from axiom.config import Settings
from axiom.core.hashing import compute_file_hash
from axiom.core.logging import get_logger, log_degradation
from axiom.core.timing import TimingLedger
from axiom.indexing import manifest as mf
from axiom.schema import Chunk, ChunkKind, ChunkLocation, ChunkMetadata, VersionManifest
from axiom.versioning.gitdiff import (
    DiffResult,
    diff_file_hashes,
    diff_versions,
    resolve_commit_sha,
)

_LOG = get_logger("versioning.incremental")

#: Source extensions walked when no project walker is installed. The hackathon
#: scope is JavaScript only (Schema.md section 5, ``ChunkMetadata.language``);
#: ``Settings`` carries no include-list field, so this is the declared default
#: rather than a tunable.
SOURCE_EXTENSIONS: Final[frozenset[str]] = frozenset({".js", ".mjs", ".cjs", ".jsx"})

#: Directories never descended into: derived output, dependency trees, and our own
#: index root. Walking ``node_modules`` would multiply a 10k-file repo by 50.
SKIP_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {".git", ".axiom", ".venv", "node_modules", "dist", "build", "__pycache__", ".mypy_cache"}
)

#: A file with a NUL byte in its first block is binary; chunking it produces
#: garbage tokens and a useless embedding.
_BINARY_SNIFF_BYTES: Final[int] = 8192

ChunkerFn = Callable[[Path, str, str, str | None], list[Chunk]]
"""``(absolute_path, repo_relative_path, version_id, commit_sha) -> chunks``."""

EmbedderFn = Callable[[list[str]], Sequence[Sequence[float]]]
"""``(texts) -> one L2-normalised vector per text, in the same order``."""


@dataclass(frozen=True)
class EmbedderHandle:
    """An embedder plus the identity that has to reach the manifest.

    The model that *loaded* is not always the model that was *configured* -- the
    embedder ladder falls back when onnxruntime or a model file is missing. The
    manifest must record what actually produced the vectors, or invariant 18
    (model identity is enforced, not assumed) guards nothing.
    """

    encode: EmbedderFn
    model_id: str | None = None
    dim: int | None = None


@dataclass
class ReindexReport:
    """What one reindex actually did -- the audit trail behind the NFR-02 claim.

    ``embed_calls`` is the load-bearing field: TC-075 asserts it is exactly zero
    for a pure rename, and a regression there is invisible in every other metric
    because the output index is identical either way, just slower to produce.
    """

    version_id: str
    parent_version: str | None = None
    strategy: str = "full"
    chunk_count: int = 0
    files_chunked: int = 0
    files_dropped: int = 0
    chunks_carried: int = 0
    chunks_rebuilt: int = 0
    chunks_dropped: int = 0
    blobs_reused: int = 0
    blobs_written: int = 0
    embed_calls: int = 0
    missing_embeddings: int = 0
    elapsed_ms: float = 0.0
    degradations: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    manifest_path: str | None = None

    def degrade(self, reason: str) -> None:
        """Record a rung taken, once, preserving order."""
        if reason not in self.degradations:
            self.degradations.append(reason)

    def to_dict(self) -> dict[str, Any]:
        """The ``--json`` body for ``axiom reindex`` (API.md section 8).

        Deliberately not :class:`~axiom.api.models.IndexSummary`, which is
        ``axiom index``'s shape: projecting onto it would drop
        ``blobs_reused``/``embed_calls``, and those counters are how ``FR-19``'s
        rename-is-free claim is measured rather than asserted.
        """
        return {
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "strategy": self.strategy,
            "chunk_count": self.chunk_count,
            "files_chunked": self.files_chunked,
            "files_dropped": self.files_dropped,
            "chunks_carried": self.chunks_carried,
            "chunks_rebuilt": self.chunks_rebuilt,
            "chunks_dropped": self.chunks_dropped,
            "blobs_reused": self.blobs_reused,
            "blobs_written": self.blobs_written,
            "embed_calls": self.embed_calls,
            "missing_embeddings": self.missing_embeddings,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "degradations": list(self.degradations),
            "timings": self.timings,
        }


# --------------------------------------------------------------------------
# Optional collaborators, resolved lazily
# --------------------------------------------------------------------------


def _optional_module(name: str) -> Any | None:
    """Import a sibling module that may not be written or installed yet.

    Import failures are expected states here, not bugs: the chunker needs
    ``tree_sitter`` and the dense builder needs ``faiss`` plus ``onnxruntime``,
    and NFR-07 requires the pipeline to keep working when neither is present.
    """
    try:
        return importlib.import_module(name)
    except Exception as exc:
        log_degradation(_LOG, "incremental", f"{name} unavailable: {exc}", "built-in fallback")
        return None


def _fallback_chunk_file(
    path: Path, file_path: str, version_id: str, commit_sha: str | None
) -> list[Chunk]:
    """Last rung of the chunker ladder: the whole file as one MODULE chunk.

    Identical to the ladder's final fallback in Rules.md section 3. A single
    oversized chunk retrieves poorly, but it retrieves -- and it keeps the index
    complete, which is what stops one missing grammar from emptying the corpus.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        log_degradation(_LOG, "chunker", f"unreadable {file_path}: {exc}", "file skipped")
        return []
    if not data or b"\x00" in data[:_BINARY_SNIFF_BYTES]:
        return []
    text = data.decode("utf-8", errors="replace")
    if not text.strip():
        return []
    line_count = max(1, len(text.splitlines()))
    location = ChunkLocation(
        file_path=file_path,
        start_line=1,
        end_line=line_count,
        start_byte=0,
        end_byte=len(data),
    )
    metadata = ChunkMetadata(
        symbol=None,
        kind=ChunkKind.MODULE,
        version_id=version_id,
        commit_sha=commit_sha,
    )
    return [Chunk.create(text=text, location=location, metadata=metadata)]


def resolve_chunker(settings: Settings) -> ChunkerFn:
    """Return the installed AST chunker, or the whole-file fallback.

    The adapter tries several call shapes because ``chunking/ast_chunker.py`` is
    owned by another workstream and only its *name* is fixed by Appflow.md; a
    signature mismatch degrades to the fallback rather than failing the build.
    """
    module = _optional_module("axiom.chunking.ast_chunker")
    chunk_file = getattr(module, "chunk_file", None) if module is not None else None
    if chunk_file is None:
        return _fallback_chunk_file

    def adapter(path: Path, file_path: str, version_id: str, commit_sha: str | None) -> list[Chunk]:
        attempts: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
            ((path, file_path, version_id, settings), {"commit_sha": commit_sha}),
            ((path, file_path, version_id), {"commit_sha": commit_sha}),
            ((path, file_path, version_id), {}),
            ((path, file_path), {}),
        )
        for args, kwargs in attempts:
            try:
                produced = chunk_file(*args, **kwargs)
            except TypeError:
                continue
            except Exception as exc:
                log_degradation(_LOG, "chunker", f"{file_path}: {exc}", "whole-file MODULE chunk")
                return _fallback_chunk_file(path, file_path, version_id, commit_sha)
            return list(produced or [])
        log_degradation(
            _LOG, "chunker", "ast_chunker.chunk_file signature unrecognised", "whole-file chunk"
        )
        return _fallback_chunk_file(path, file_path, version_id, commit_sha)

    return adapter


def resolve_embedder(settings: Settings) -> EmbedderHandle | None:
    """Return a handle on the installed embedder, or ``None`` when there is none.

    ``indexing.embedder.load_embedder`` is preferred because it owns the model
    ladder and always yields *some* embedder; the older ``indexing.dense``
    function names are still probed so this module does not depend on one
    workstream landing before the other.

    ``None`` is a supported outcome: the version still builds, its blobs are
    simply not written, and the report says how many vectors are missing -- far
    better than aborting an index build on a machine without onnxruntime.
    """
    ladder = _optional_module("axiom.indexing.embedder")
    loader = getattr(ladder, "load_embedder", None) if ladder is not None else None
    if callable(loader):
        try:
            embedder = loader(settings)
        except Exception as exc:
            log_degradation(_LOG, "embedder", f"load_embedder failed: {exc}", "no embedder")
        else:
            return EmbedderHandle(
                encode=lambda texts: embedder.encode(texts),
                model_id=getattr(embedder, "model_id", None),
                dim=getattr(embedder, "dim", None),
            )

    module = _optional_module("axiom.indexing.dense")
    if module is None:
        return None
    for name in ("embed_texts", "embed_batch", "embed"):
        candidate = getattr(module, name, None)
        if callable(candidate):

            def adapter(texts: list[str], _fn: Any = candidate) -> Sequence[Sequence[float]]:
                try:
                    return _fn(texts, settings)
                except TypeError:
                    return _fn(texts)

            return EmbedderHandle(encode=adapter)
    log_degradation(_LOG, "embedder", "no embed entrypoint in indexing.dense", "blobs not written")
    return None


# --------------------------------------------------------------------------
# Filesystem walk and hashing
# --------------------------------------------------------------------------


def walk_source_files(root: Path, settings: Settings) -> list[tuple[Path, str]]:
    """Enumerate indexable files as ``(absolute_path, repo-relative POSIX path)``.

    Sorted, because the walk order determines chunk production order, which must
    not vary between two runs on the same tree (NFR-08). Windows separators are
    converted here, at the single ingest boundary, never later (invariant 5).
    """
    # The chunking workstream owns discovery; the module it lives in has moved
    # once already, so both homes are probed before falling back to a local walk.
    for module_name, names in (
        ("axiom.chunking.ast_chunker", ("discover_source_files",)),
        ("axiom.chunking.walker", ("walk_repo", "iter_source_files", "walk")),
    ):
        module = _optional_module(module_name)
        if module is None:
            continue
        fn = next((f for f in (getattr(module, n, None) for n in names) if callable(f)), None)
        if fn is not None:
            try:
                produced = list(fn(root, settings))
            except TypeError:
                try:
                    produced = list(fn(root))
                except Exception as exc:
                    log_degradation(_LOG, "walker", str(exc), "built-in walk")
                    continue
            except Exception as exc:
                log_degradation(_LOG, "walker", str(exc), "built-in walk")
                continue
            pairs: list[tuple[Path, str]] = []
            for item in produced:
                if isinstance(item, tuple):
                    absolute, relative = Path(item[0]), str(item[1])
                else:
                    absolute = Path(item)
                    try:
                        relative = absolute.relative_to(root).as_posix()
                    except ValueError:
                        # A discoverer that returns something outside the tree is
                        # confused, not fatal: skip the path, keep the build.
                        continue
                pairs.append((absolute, relative))
            return sorted(pairs, key=lambda pair: pair[1])

    found: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts[:-1]):
            continue
        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        found.append((path, path.relative_to(root).as_posix()))
    return sorted(found, key=lambda pair: pair[1])


def hash_files(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    """Hash whole files for ``VersionManifest.file_hashes``.

    The manifest's hash map is the diff basis when git is unavailable, so it is
    computed with the same normalisation as ``content_hash``: a pure line-ending
    or trailing-whitespace change must not read as a content change and trigger a
    pointless re-embed.
    """
    hashes: dict[str, str] = {}
    for relative in sorted(set(relative_paths)):
        path = root / relative
        try:
            data = path.read_bytes()
        except OSError as exc:
            log_degradation(_LOG, "file_hash", f"unreadable {relative}: {exc}", "path omitted")
            continue
        hashes[relative] = compute_file_hash(data.decode("utf-8", errors="replace"))
    return hashes


# --------------------------------------------------------------------------
# Core reindex
# --------------------------------------------------------------------------


def _restamp(chunk: Chunk, version_id: str, commit_sha: str | None) -> Chunk:
    """Move a carried-over chunk into the new version.

    Invariant 13 says a chunk's ``version_id`` equals the version directory it
    lives in, so a chunk reused from the parent must be re-stamped. Neither id
    changes: both are functions of text and location, and this touches neither --
    which is exactly why the reuse is free.
    """
    metadata = chunk.metadata
    if metadata.version_id == version_id and metadata.commit_sha == commit_sha:
        return chunk
    return chunk.model_copy(
        update={
            "metadata": metadata.model_copy(
                update={
                    "version_id": version_id,
                    "commit_sha": commit_sha,
                }
            )
        }
    )


def _chunk_files(
    root: Path,
    relative_paths: Sequence[str],
    version_id: str,
    commit_sha: str | None,
    chunker: ChunkerFn,
    report: ReindexReport,
) -> list[Chunk]:
    """Chunk a set of files, skipping the ones that no longer exist."""
    produced: list[Chunk] = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            continue
        chunks = chunker(path, relative, version_id, commit_sha)
        if chunks:
            report.files_chunked += 1
            produced.extend(chunks)
    return produced


def _materialise_embeddings(
    chunks: Sequence[Chunk],
    settings: Settings,
    embedder: EmbedderHandle | None,
    report: ReindexReport,
) -> int | None:
    """Ensure every chunk's content has a cached vector, embedding only the misses.

    The blob probe happens before any model is loaded, which is what makes the
    rename case cost nothing at all rather than "cheap": no batch is assembled, no
    session is created, and ``embed_calls`` stays at zero (TC-075).

    Returns the dimensionality actually observed, so the manifest can record the
    rung that ran rather than the rung that was configured.
    """
    wanted: list[str] = []
    texts: dict[str, str] = {}
    for chunk in chunks:
        digest = chunk.content_hash
        if digest in texts:
            continue
        if mf.blob_exists(settings, digest):
            report.blobs_reused += 1
            continue
        texts[digest] = chunk.text
        wanted.append(digest)

    if not wanted:
        return embedder.dim if embedder is not None else None
    if embedder is None:
        report.missing_embeddings = len(wanted)
        report.degrade("no embedder: blobs not written")
        log_degradation(
            _LOG,
            "embedder",
            f"{len(wanted)} chunks have no cached vector and no embedder is available",
            "index built without dense vectors",
        )
        return None

    observed = embedder.dim
    batch_size = max(1, settings.embedding_batch_size)
    for start in range(0, len(wanted), batch_size):
        batch = wanted[start : start + batch_size]
        payload = [texts[digest] for digest in batch]
        try:
            vectors = embedder.encode(payload)
        except Exception as exc:
            report.missing_embeddings += len(batch)
            report.degrade(f"embedder failed: {exc}")
            log_degradation(_LOG, "embedder", str(exc), "remaining chunks left unembedded")
            continue
        report.embed_calls += len(payload)
        for digest, vector in zip(batch, vectors, strict=False):
            if observed is None:
                observed = len(vector)
            if mf.store_blob(settings, digest, vector, expected_dim=observed) is not None:
                report.blobs_written += 1
            else:
                # The vector exists but could not be persisted (no numpy). The
                # build is still valid; the next one just pays to embed again.
                report.degrade("blob cache unavailable: vectors not persisted")
    return observed


#: ``(signal, module, function)`` for the three side indexes, named explicitly.
#:
#: The names are spelled out rather than discovered via ``getattr(module,
#: "build_index")`` because only two of the three modules carry that alias, so
#: discovery silently skipped the dense index and produced a version that
#: answered zero queries. Every one of these takes ``(chunks, settings,
#: out_dir)``; the order is load-bearing and is asserted by
#: :func:`_build_side_indexes`.
_SIDE_INDEX_BUILDERS: Final[tuple[tuple[str, str, str], ...]] = (
    ("dense", "axiom.indexing.dense", "build_dense_index"),
    ("sparse", "axiom.indexing.sparse", "build_sparse_index"),
    ("structural", "axiom.indexing.structural", "build_structural_index"),
)


def _dense_embedder(handle: EmbedderHandle | None) -> Any | None:
    """Adapt a reindex :class:`EmbedderHandle` to the dense builder's ``Embedder``.

    ``build_dense_index`` loads the embedder ladder when none is injected, which
    on the rename path would fire the whole fallback chain to encode nothing --
    every vector is already a blob hit. The handle already names the model that
    produced those blobs, so hand it over when it is complete enough to be
    trusted (both identity fields set); otherwise let the ladder run, which is
    correct, just louder.
    """
    if handle is None or handle.model_id is None or handle.dim is None:
        return None
    # Bound to locals first: the nested class closes over these, and a narrowed
    # attribute of an optional does not stay narrowed inside a closure.
    model_id, dim, encode = handle.model_id, handle.dim, handle.encode

    class _HandleEmbedder:
        def __init__(self) -> None:
            self.model_id = model_id
            self.dim = dim

        @staticmethod
        def encode(texts: list[str]) -> Sequence[Sequence[float]]:
            return encode(texts)

    return _HandleEmbedder()


def _build_side_indexes(
    chunks: Sequence[Chunk],
    out_dir: Path,
    settings: Settings,
    report: ReindexReport,
    builders: Mapping[str, Callable[..., Any]] | None,
    handle: EmbedderHandle | None = None,
) -> None:
    """Hand the finished chunk corpus to the three index builders, best effort.

    Each needs an optional native library (faiss, bm25s, tree-sitter). A builder
    that is absent or that raises costs its signal, not the build: retrieval
    degrades to the signals that were written (Rules.md section 3, fusion row).

    The calling convention is fixed at ``builder(chunks, settings, out_dir)``,
    for injected builders too. It used to be guessed -- ``(chunks, out_dir,
    settings)`` with a ``TypeError`` retry -- and the retry bound ``out_dir`` to
    the ``settings`` parameter, leaving the real ``out_dir`` at its ``Path(".")``
    default: ``sparse.bm25s/`` landed in the process's working directory instead
    of the version directory. Guessing an argument order is never worth it.
    """
    targets: dict[str, Callable[..., Any] | None] = {}
    enabled_by_name = {
        "dense": settings.dense_enabled,
        "sparse": settings.sparse_enabled,
        "structural": settings.structural_enabled,
    }
    if builders is not None:
        targets.update(builders)
    else:
        for name, module_path, function_name in _SIDE_INDEX_BUILDERS:
            if not enabled_by_name[name]:
                continue
            module = _optional_module(module_path)
            targets[name] = getattr(module, function_name, None) if module is not None else None

    for name, builder in targets.items():
        if builder is None:
            report.degrade(f"no {name} index builder")
            continue
        kwargs: dict[str, Any] = {}
        if name == "dense":
            embedder = _dense_embedder(handle)
            if embedder is not None:
                kwargs["embedder"] = embedder
        try:
            builder(chunks, settings, out_dir, **kwargs)
        except Exception as exc:
            report.degrade(f"{name} index build failed: {exc}")
            log_degradation(_LOG, f"indexing.{name}", str(exc), "signal omitted from this version")


def _resolve_diff(
    old_manifest: VersionManifest | None,
    root: Path,
    old_rev: str | None,
    new_rev: str | None,
    current_hashes: Callable[[], dict[str, str]],
    report: ReindexReport,
) -> DiffResult | None:
    """Walk the diff ladder: git, then file hashes, then ``None`` for a full rebuild.

    ``current_hashes`` is a thunk, not a dict: hashing the whole working tree costs
    one read per file, and when git answers -- the normal case -- that work is
    pure waste against the 45 s NFR-02 budget.
    """
    if old_manifest is None:
        return None
    if old_rev and new_rev:
        result = diff_versions(old_rev, new_rev, root)
        if result is not None:
            return result
        report.degrade("git diff unavailable: fell back to file-hash comparison")
    elif old_manifest.commit_sha and new_rev:
        result = diff_versions(old_manifest.commit_sha, new_rev, root)
        if result is not None:
            return result
        report.degrade("git diff unavailable: fell back to file-hash comparison")

    if not old_manifest.file_hashes:
        report.degrade("parent manifest carries no file_hashes: full rebuild")
        return None
    return diff_file_hashes(old_manifest.file_hashes, current_hashes())


def reindex(
    old_version: str | None,
    new_version: str,
    settings: Settings,
    *,
    repo_path: Path | str | None = None,
    old_rev: str | None = None,
    new_rev: str | None = None,
    chunker: ChunkerFn | None = None,
    embedder: EmbedderFn | EmbedderHandle | None = None,
    index_builders: Mapping[str, Callable[..., Any]] | None = None,
    commit_sha: str | None = None,
    force_full: bool = False,
    make_active: bool = True,
) -> ReindexReport:
    """Build ``new_version`` by reusing everything ``old_version`` already paid for.

    Args:
        old_version: Parent version to derive from. ``None`` forces a cold build.
        new_version: Version id to write. Becomes the directory name under
            ``.axiom/index/``, so it is validated against the traversal-safe
            pattern before any path is touched.
        settings: Resolved settings; supplies the index root, the embedder
            identity recorded in the manifest, and the batch size.
        repo_path: Working tree to index. Defaults to the current directory.
        old_rev: Revision the parent version was built from. Defaults to the
            parent manifest's ``commit_sha``.
        new_rev: Revision to build. Defaults to ``HEAD`` when the tree is a repo.
        chunker: Injected chunker, mainly for tests. Defaults to the installed AST
            chunker, then to whole-file MODULE chunks.
        embedder: Injected embedder, mainly for tests. ``None`` means "resolve
            one"; pass an explicit counting stub to assert TC-075's zero.
        index_builders: Injected ``{name: build_index}`` map. ``None`` resolves the
            three installed builders; ``{}`` skips side indexes entirely.
        commit_sha: Override for the sha stamped into every chunk (FR-16).
        force_full: Skip the diff and rebuild the whole tree.
        make_active: Point the registry at the new version when it lands.

    Returns:
        A :class:`ReindexReport`. The function does not raise on a missing parent,
        a missing git repo, a missing model or an unparseable file -- each of
        those is a rung, recorded in ``report.degradations``.
    """
    ledger = TimingLedger()
    root = Path(repo_path or ".").resolve()
    mf.require_valid_version_id(new_version)

    report = ReindexReport(version_id=new_version)
    chunker = chunker or resolve_chunker(settings)
    handle: EmbedderHandle | None
    if embedder is None:
        handle = resolve_embedder(settings)
    elif isinstance(embedder, EmbedderHandle):
        handle = embedder
    else:
        handle = EmbedderHandle(encode=embedder)

    old_manifest = None if force_full else mf.try_read_manifest(settings, old_version)
    if old_version and old_manifest is None and not force_full:
        report.degrade(f"parent {old_version} unavailable: full rebuild")
    if old_manifest is not None and old_manifest.version_id == new_version:
        report.degrade("parent and target version ids are identical: rebuilding in place")
        old_manifest = None

    resolved_sha = commit_sha
    if resolved_sha is None:
        resolved_sha = resolve_commit_sha(root, new_rev or "HEAD")

    with ledger.measure("walk") as detail:
        present = walk_source_files(root, settings)
        detail["files"] = len(present)
    present_paths = [relative for _, relative in present]

    with ledger.measure("diff") as detail:
        diff = _resolve_diff(
            old_manifest,
            root,
            old_rev,
            new_rev,
            lambda: hash_files(root, present_paths),
            report,
        )
        detail.update(diff.summary() if diff is not None else {"source": "full"})

    parent_version = (
        old_manifest.version_id if old_manifest is not None and diff is not None else None
    )
    report.parent_version = parent_version
    report.strategy = diff.source if diff is not None else "full"

    with ledger.measure("chunk") as detail:
        if diff is None:
            carried: list[Chunk] = []
            to_chunk = present_paths
            report.files_dropped = 0
        else:
            parent_chunks = mf.read_version_chunks(settings, old_manifest.version_id)  # type: ignore[union-attr]
            dropped = set(diff.paths_to_drop)
            rechunk = set(diff.paths_to_chunk)
            carried = [
                _restamp(chunk, new_version, resolved_sha)
                for chunk in parent_chunks
                if chunk.location.file_path not in dropped
                and chunk.location.file_path not in rechunk
            ]
            report.chunks_dropped = len(parent_chunks) - len(carried)
            report.files_dropped = len(dropped)
            # Only chunk paths that survive the walk: a diff can name a file that
            # is gone (deleted after the diff was taken) or filtered out here.
            allowed = set(present_paths)
            to_chunk = [p for p in diff.paths_to_chunk if p in allowed]
        rebuilt = _chunk_files(root, to_chunk, new_version, resolved_sha, chunker, report)
        report.chunks_carried = len(carried)
        report.chunks_rebuilt = len(rebuilt)
        chunks = mf.canonical_chunk_order([*carried, *rebuilt])
        report.chunk_count = len(chunks)
        detail["chunks"] = len(chunks)

    with ledger.measure("blob") as detail:
        observed_dim = _materialise_embeddings(chunks, settings, handle, report)
        detail["embed_calls"] = report.embed_calls
        detail["blobs_reused"] = report.blobs_reused

    staging = mf.versions_dir(settings) / f"{new_version}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    with ledger.measure("write") as detail:
        detail["lines"] = mf.write_chunks(chunks, staging / "chunks.jsonl")

    with ledger.measure("index"):
        _build_side_indexes(chunks, staging, settings, report, index_builders, handle)

    with ledger.measure("manifest"):
        if diff is None:
            file_hashes = hash_files(root, [c.location.file_path for c in chunks])
        else:
            file_hashes = dict(old_manifest.file_hashes)  # type: ignore[union-attr]
            for path in diff.paths_to_drop:
                file_hashes.pop(path, None)
            file_hashes.update(hash_files(root, diff.paths_to_chunk))
            still_present = set(present_paths)
            file_hashes = {k: v for k, v in file_hashes.items() if k in still_present}
        # The manifest records the embedder that actually ran, never the one that
        # was configured: the ladder may have fallen back, and invariant 18 is
        # only enforceable against the truth (indexing.embedder.load_embedder).
        manifest = mf.build_manifest(
            new_version,
            len(chunks),
            settings,
            file_hashes=file_hashes,
            commit_sha=resolved_sha,
            parent_version=parent_version,
            embedding_model=handle.model_id if handle is not None else None,
            embedding_dim=(handle.dim if handle is not None else None) or observed_dim,
        )
        # The manifest is written into the staging directory and published by the
        # rename, so a reader never sees a completion marker over a partial build.
        mf.write_manifest(manifest, settings, target=staging / "manifest.json")
        mf.commit_version_dir(staging, mf.version_dir(settings, new_version))
        mf.register_version(manifest, settings, make_active=make_active)
        report.manifest_path = str(mf.manifest_path(settings, new_version))

    report.elapsed_ms = ledger.total_ms
    report.timings = ledger.as_flat_dict()
    _LOG.info(
        "reindex %s -> %s [%s]: %d chunks, %d embed calls, %d blobs reused",
        parent_version or "(cold)",
        new_version,
        report.strategy,
        report.chunk_count,
        report.embed_calls,
        report.blobs_reused,
    )
    return report


def full_reindex(
    version_id: str,
    settings: Settings,
    *,
    repo_path: Path | str | None = None,
    **kwargs: Any,
) -> ReindexReport:
    """Cold build of a whole tree -- ``axiom index`` and ``axiom reindex --full``.

    Kept as a thin wrapper rather than a separate implementation so the two paths
    cannot drift; TC-076 asserts their outputs are equal, which is only credible
    if they share the chunking, ordering and manifest code exactly.
    """
    return reindex(None, version_id, settings, repo_path=repo_path, force_full=True, **kwargs)


__all__ = [
    "SKIP_DIRECTORIES",
    "SOURCE_EXTENSIONS",
    "ChunkerFn",
    "EmbedderFn",
    "EmbedderHandle",
    "ReindexReport",
    "full_reindex",
    "hash_files",
    "reindex",
    "resolve_chunker",
    "resolve_embedder",
    "walk_source_files",
]
