"""Version-aware and evolutionary retrieval (FR-16 .. FR-21).

Three modules, one story: :mod:`gitdiff` says what changed between two
revisions, :mod:`incremental` rebuilds only that, and :mod:`evolutionary` reads
the resulting chain of versions back as snippet families.

Importing this package is cheap and dependency-free -- git is invoked through
:mod:`subprocess`, numpy is reached only inside the blob cache, and the chunker,
embedder and index builders are all resolved lazily at call time (NFR-07).
"""

from __future__ import annotations

from axiom.versioning.evolutionary import (
    apply_stability_bonus,
    build_families,
    cosine,
    identity_families,
    rank_families,
    version_order_from_registry,
)
from axiom.versioning.gitdiff import (
    DiffResult,
    diff_file_hashes,
    diff_versions,
    diff_working_tree,
    is_git_repo,
    parse_name_status,
    resolve_commit_sha,
    resolve_version_identity,
)
from axiom.versioning.incremental import ReindexReport, full_reindex, reindex

__all__ = [
    "DiffResult",
    "ReindexReport",
    "apply_stability_bonus",
    "build_families",
    "cosine",
    "diff_file_hashes",
    "diff_versions",
    "diff_working_tree",
    "full_reindex",
    "identity_families",
    "is_git_repo",
    "parse_name_status",
    "rank_families",
    "reindex",
    "resolve_commit_sha",
    "resolve_version_identity",
    "version_order_from_registry",
]
