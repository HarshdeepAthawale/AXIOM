"""Resolve what changed between two revisions into {A, M, D, R} path sets.

``git diff --name-status`` is the sole versioning source of truth (ADR-010), and
rename detection is the reason: git is the only component in the stack that knows
``src/a.js`` became ``src/b.js``. A rename it reports as ``R100 old new`` costs
zero embedding forward passes downstream, because the chunk text -- and therefore
the ``content_hash`` keying the blob cache -- never changed (FR-19, TC-075).

Everything here is stdlib. Nothing raises: a missing repo, a missing ``git``
binary, a bad revision and a corrupt ``.git`` all return ``None``, which is the
signal for :mod:`axiom.versioning.incremental` to drop to the next rung of the
ladder (file-hash comparison against the parent manifest, then a full reindex).
``None`` and an empty diff are deliberately different answers: an empty
:class:`DiffResult` means "nothing changed", which is a valid no-op build
(TC-077), not a failure.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from axiom.core.logging import get_logger, log_degradation

_LOG = get_logger("versioning.gitdiff")

#: Seconds before a ``git`` invocation is abandoned. ``Settings`` has no field for
#: this (it is not a tunable that can move a score, so AP-07 does not reach it);
#: the cap exists only so a wedged git process cannot hang an index build.
GIT_TIMEOUT_S: Final[float] = 30.0

#: Status letters git can emit; the rename/copy pair carries a similarity score
#: (``R100``, ``R087``) and *two* paths, which is the only irregular row shape.
_RENAME_LIKE: Final[frozenset[str]] = frozenset({"R", "C"})


@dataclass(frozen=True)
class DiffResult:
    """Added / modified / deleted / renamed paths between two revisions.

    Paths are repo-relative POSIX strings, matching ``ChunkLocation.file_path``
    and ``VersionManifest.file_hashes`` keys (Schema.md invariant 5). Every
    collection is sorted, because these sets drive which files get re-chunked and
    therefore the order chunks are produced in -- and that order must not depend
    on git's output order or on set iteration (Rules.md AP-05).
    """

    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    renamed: tuple[tuple[str, str], ...] = ()
    source: str = "git"
    unresolved: tuple[str, ...] = field(default=())

    @property
    def paths_to_chunk(self) -> tuple[str, ...]:
        """Files that must be re-chunked: added, modified, and rename destinations.

        A rename destination is included even when its content is untouched:
        ``chunk_id`` is a digest of (content, file_path, start_line), so moving a
        file changes every id in it. The *embeddings* are still reused, because
        ``content_hash`` ignores location (Schema.md section 13.1).
        """
        return tuple(sorted({*self.added, *self.modified, *(new for _, new in self.renamed)}))

    @property
    def paths_to_drop(self) -> tuple[str, ...]:
        """Files whose existing chunks must disappear: deletions and rename sources."""
        return tuple(sorted({*self.deleted, *(old for old, _ in self.renamed)}))

    @property
    def is_empty(self) -> bool:
        """True when nothing changed -- a valid no-op reindex (TC-077)."""
        return not (self.added or self.modified or self.deleted or self.renamed)

    @property
    def changed_file_count(self) -> int:
        """Number of distinct files touched; the unit of the NFR-02 budget."""
        return len({*self.paths_to_chunk, *self.paths_to_drop})

    def summary(self) -> dict[str, int | str]:
        """Compact, loggable shape of this diff."""
        return {
            "source": self.source,
            "added": len(self.added),
            "modified": len(self.modified),
            "deleted": len(self.deleted),
            "renamed": len(self.renamed),
        }


def _run_git(args: list[str], repo_path: Path) -> str | None:
    """Run one git command, returning stdout or ``None`` on any failure.

    The argument vector is passed as a list with no shell, so a branch name like
    ``$(rm -rf /)`` is a literal revision that git rejects, not a command.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_path),
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log_degradation(_LOG, "gitdiff", f"git {' '.join(args)} failed: {exc}", "no git output")
        return None
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        log_degradation(
            _LOG,
            "gitdiff",
            f"git {' '.join(args)} exited {completed.returncode}: {detail}",
            "no git output",
        )
        return None
    return completed.stdout.decode("utf-8", errors="replace")


def is_git_repo(repo_path: Path) -> bool:
    """True when ``repo_path`` is inside a git working tree."""
    out = _run_git(["rev-parse", "--is-inside-work-tree"], repo_path)
    return out is not None and out.strip() == "true"


def resolve_commit_sha(repo_path: Path, rev: str = "HEAD") -> str | None:
    """Full 40-char sha for a revision, or ``None`` when it cannot be resolved.

    ``ChunkMetadata.commit_sha`` validates the 40-hex shape, so an abbreviated or
    decorated answer is rejected here rather than at model construction, where it
    would abort a whole index build over provenance metadata (FR-16).
    """
    out = _run_git(["rev-parse", rev], repo_path)
    if out is None:
        return None
    sha = out.strip()
    if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
        return sha
    log_degradation(_LOG, "gitdiff", f"rev-parse {rev} gave {sha!r}", "commit_sha=None")
    return None


def resolve_version_identity(
    repo_path: Path | None,
    *,
    rev: str = "HEAD",
    version_override: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve ``(version_id, commit_sha)`` for a build target (FR-16).

    Git is consulted when the target is a repository; otherwise the caller's
    config override stands alone. That ordering is what lets Axiom index a plain
    directory -- an unpacked tarball, the evaluator's scratch copy -- with a
    hand-supplied version label and no git at all.
    """
    sha: str | None = None
    if repo_path is not None and is_git_repo(repo_path):
        sha = resolve_commit_sha(repo_path, rev)
    if version_override:
        return version_override, sha
    if sha is not None:
        return sha, sha
    log_degradation(
        _LOG,
        "gitdiff",
        f"no git identity for {repo_path} and no version override",
        "caller must supply version_id",
    )
    return None, None


def parse_name_status(raw: str, *, source: str = "git") -> DiffResult:
    """Parse ``git diff --name-status`` output, NUL-separated or tab-separated.

    Two shapes reach this function. With ``-z`` (what :func:`diff_versions` asks
    for) records are NUL-terminated fields, which is the only encoding safe for
    paths containing spaces, tabs or non-ASCII bytes -- without it git escapes and
    quotes such paths and the path we parse is not the path on disk. The
    tab-separated form is still accepted because it is what a human pastes from a
    terminal and what TC-073 states its fixtures in.

    Unknown status letters are collected into ``unresolved`` rather than dropped,
    so a future git version cannot silently shrink the set of files we re-chunk.
    """
    added: set[str] = set()
    modified: set[str] = set()
    deleted: set[str] = set()
    renamed: set[tuple[str, str]] = set()
    unresolved: set[str] = set()

    tokens: list[str]
    if "\0" in raw:
        tokens = [t for t in raw.split("\0") if t != ""]
    else:
        tokens = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            tokens.extend(part for part in line.split("\t") if part != "")

    index = 0
    while index < len(tokens):
        status = tokens[index].strip()
        index += 1
        if not status:
            continue
        letter = status[0].upper()
        if letter in _RENAME_LIKE:
            if index + 1 >= len(tokens):
                unresolved.add(status)
                break
            old_path, new_path = tokens[index], tokens[index + 1]
            index += 2
            if letter == "C":
                # A copy leaves the source in place: only the destination is new.
                added.add(new_path)
            else:
                renamed.add((old_path, new_path))
            continue
        if index >= len(tokens):
            unresolved.add(status)
            break
        path = tokens[index]
        index += 1
        if letter == "A":
            added.add(path)
        elif letter == "D":
            deleted.add(path)
        elif letter in {"M", "T", "U"}:
            # T (type change, e.g. file -> symlink) and U (unmerged) are treated as
            # modifications: re-chunking them is cheap and always correct, whereas
            # ignoring them would leave stale chunks in the index.
            modified.add(path)
        else:
            unresolved.add(status)
            modified.add(path)

    if unresolved:
        _LOG.warning("unrecognised git status codes: %s", sorted(unresolved))

    return DiffResult(
        added=tuple(sorted(added)),
        modified=tuple(sorted(modified)),
        deleted=tuple(sorted(deleted)),
        renamed=tuple(sorted(renamed)),
        source=source,
        unresolved=tuple(sorted(unresolved)),
    )


def diff_versions(old_rev: str, new_rev: str, repo_path: Path | None = None) -> DiffResult | None:
    """Diff two revisions, or return ``None`` when git cannot answer.

    ``None`` is not an error state: it is the handoff to the next rung of the
    degradation ladder. Returning an empty :class:`DiffResult` instead would claim
    "nothing changed", which for a non-repo directory is a confident lie that
    would leave the new index identical to the old one.
    """
    root = Path(repo_path or ".")
    if not is_git_repo(root):
        log_degradation(_LOG, "gitdiff", f"{root} is not a git work tree", "file-hash comparison")
        return None
    raw = _run_git(["diff", "--name-status", "--find-renames", "-z", f"{old_rev}..{new_rev}"], root)
    if raw is None:
        return None
    result = parse_name_status(raw, source="git")
    _LOG.info("git diff %s..%s: %s", old_rev, new_rev, result.summary())
    return result


def diff_working_tree(rev: str, repo_path: Path | None = None) -> DiffResult | None:
    """Diff a revision against the current working tree, including untracked files.

    Used by ``axiom reindex`` with no ``--to``: the interesting diff during a demo
    is usually "what have I edited since the last index", which no commit covers.
    """
    root = Path(repo_path or ".")
    if not is_git_repo(root):
        log_degradation(_LOG, "gitdiff", f"{root} is not a git work tree", "file-hash comparison")
        return None
    raw = _run_git(["diff", "--name-status", "--find-renames", "-z", rev], root)
    if raw is None:
        return None
    result = parse_name_status(raw, source="git")
    untracked = _run_git(["ls-files", "--others", "--exclude-standard", "-z"], root)
    if untracked:
        extra = tuple(sorted(p for p in untracked.split("\0") if p))
        result = DiffResult(
            added=tuple(sorted({*result.added, *extra})),
            modified=result.modified,
            deleted=result.deleted,
            renamed=result.renamed,
            source=result.source,
            unresolved=result.unresolved,
        )
    return result


def diff_file_hashes(
    parent_hashes: Mapping[str, str], current_hashes: Mapping[str, str]
) -> DiffResult:
    """Compare two ``file_hashes`` maps -- the ladder's middle rung.

    This is what runs when git is unavailable. It bounds the work to genuinely
    changed content, which is most of the win, but it cannot see renames: a moved
    file looks like a delete plus an add. The chunks are then re-chunked *and*
    re-embedded, costing the zero-embedding rename property (TC-075) while
    remaining correct -- content-addressed blobs still dedupe the vectors if the
    embedder ran before on identical text, so in practice the cost is one blob
    lookup, not one forward pass.
    """
    parent_keys = set(parent_hashes)
    current_keys = set(current_hashes)
    modified = {p for p in parent_keys & current_keys if parent_hashes[p] != current_hashes[p]}
    return DiffResult(
        added=tuple(sorted(current_keys - parent_keys)),
        modified=tuple(sorted(modified)),
        deleted=tuple(sorted(parent_keys - current_keys)),
        renamed=(),
        source="file_hash",
    )


__all__ = [
    "GIT_TIMEOUT_S",
    "DiffResult",
    "diff_file_hashes",
    "diff_versions",
    "diff_working_tree",
    "is_git_repo",
    "parse_name_status",
    "resolve_commit_sha",
    "resolve_version_identity",
]
