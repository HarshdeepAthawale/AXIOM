"""Index a repository as it stood at a past revision.

``axiom index <repo>`` reads the working tree, and ``--version-id`` only
*labels* the result. That is correct for the common case and wrong for the one
that P1 and the Bonus actually ask for:

* `US-7` / `FR-20` -- "how did this work in v2.2.0?" needs an index whose chunks
  hold v2.2.0's code, not today's code wearing a v2.2.0 label.
* `FR-21` -- a `SnippetFamily` groups the *same* symbol across *different*
  versions. If every version was built from the same working tree, every member
  is byte-identical, every `content_hash` collides, and the families collapse
  into nothing. The bonus feature cannot be demonstrated at all.

Checking the ref out in place would be destructive: it would move the user's
HEAD and lose uncommitted work. ``git worktree`` exists for exactly this -- it
materialises a second working directory at an arbitrary ref, sharing the object
store, and detaches cleanly. Nothing in the user's tree is touched.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from axiom.core.logging import get_logger

_LOG = get_logger("versioning.checkout")

#: Seconds any single git call may take before we give up and degrade.
_GIT_TIMEOUT = 120


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    )


def is_git_repo(root: Path) -> bool:
    """True when ``root`` is inside a git work tree."""
    try:
        done = _git(root, "rev-parse", "--is-inside-work-tree", check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and done.stdout.strip() == "true"


def resolve_ref(root: Path, ref: str) -> tuple[str, str] | None:
    """Resolve ``ref`` to ``(full_sha, committed_date)``.

    Returns ``None`` rather than raising when the ref does not exist: a typo'd
    tag is bad *input*, and Rule 3 says bad input degrades.
    """
    try:
        sha = _git(root, "rev-parse", f"{ref}^{{commit}}", check=False)
        if sha.returncode != 0:
            return None
        when = _git(root, "show", "-s", "--format=%cs", sha.stdout.strip(), check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        _LOG.warning("git rev-parse %s failed: %s", ref, exc)
        return None
    return sha.stdout.strip(), when.stdout.strip() or ""


@contextmanager
def worktree_at(root: Path, ref: str) -> Iterator[Path | None]:
    """Yield a detached worktree of ``root`` at ``ref``, or ``None`` if unavailable.

    Yielding ``None`` instead of raising keeps the caller on the degradation
    ladder: no git, no such ref, or a git too old for ``worktree`` all mean
    "index the working tree instead and say so", never a crash.

    The worktree is removed on exit even if the body raises, and ``git worktree
    prune`` runs afterwards so an interrupted build cannot leave the user's
    repository carrying a stale administrative entry.
    """
    if not is_git_repo(root):
        _LOG.warning("%s is not a git work tree; cannot materialise ref %s", root, ref)
        yield None
        return

    resolved = resolve_ref(root, ref)
    if resolved is None:
        _LOG.warning("ref %r does not resolve in %s", ref, root)
        yield None
        return

    sha, _ = resolved
    tmp = Path(tempfile.mkdtemp(prefix="axiom-worktree-"))
    target = tmp / "tree"
    try:
        done = _git(root, "worktree", "add", "--detach", str(target), sha, check=False)
        if done.returncode != 0:
            _LOG.warning("git worktree add failed: %s", done.stderr.strip())
            yield None
            return
        _LOG.info("materialised %s at %s", ref, target)
        yield target
    finally:
        _git(root, "worktree", "remove", "--force", str(target), check=False)
        _git(root, "worktree", "prune", check=False)
        shutil.rmtree(tmp, ignore_errors=True)
