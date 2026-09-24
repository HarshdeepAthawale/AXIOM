"""Cross-version retrieval: collapse one snippet's history into one result row.

Without families, a function edited by one line across ten indexed versions takes
ten of the top ten slots and the result list is worthless. A
:class:`~axiom.schema.SnippetFamily` collapses those ten into one row that shows
the newest form, carries the older forms as members, and earns a small bonus for
having survived many versions (FR-21).

Two rules define membership, and both are load-bearing (TC-082):

1. **Identity, not similarity.** Members must share ``(symbol, file_path)``.
   Similarity alone would merge two different functions that happen to be
   near-identical -- the exact case a developer asks about when they want to know
   which of the two to call.
2. **Complete linkage at the threshold.** Every member is cosine
   ``>= Settings.dedupe_cosine`` with every other member, not merely with the
   representative. Single-linkage chaining would let A-B-C form a family where A
   and C are unrelated, and Schema.md invariant 15 states the pairwise property.

Cosine is computed in pure Python. The groups are a handful of vectors each, and
avoiding numpy here means the threshold comparison at 0.9499 / 0.9500 / 0.9501
(TC-083) is decided by one deterministic reduction rather than by a BLAS kernel
whose summation order depends on the build.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Mapping, Sequence
from typing import Final

from axiom.config import Settings
from axiom.core.hashing import compute_family_id
from axiom.core.logging import get_logger, log_degradation
from axiom.schema import Chunk, SnippetFamily

_LOG = get_logger("versioning.evolutionary")

#: Context lines in a per-transition unified diff. ``Settings`` has no field for
#: it because it is presentation, not retrieval: it cannot move a score.
DIFF_CONTEXT_LINES: Final[int] = 3

Vector = Sequence[float]
EmbeddingLookup = Mapping[str, Vector]
"""Vectors keyed by ``content_hash`` (the blob store's key) or by ``chunk_id``."""


def cosine(left: Vector, right: Vector) -> float:
    """Cosine similarity of two vectors, ``0.0`` when either has no magnitude.

    Vectors reaching this function are normally already L2-normalised (Schema.md
    invariant 8), so the denominator is 1.0 and this is an inner product. The
    normalisation is still applied rather than assumed, because a caller passing
    raw vectors would otherwise get similarities above 1.0 and every family would
    collapse into one.

    A numpy array is a valid ``Sequence[float]`` and is what
    ``VersionManifest.load_blob`` hands back, so emptiness is tested by length
    rather than by truthiness: ``not left`` raises ``ValueError`` on an ndarray
    of more than one element, which turned the natural call site into a crash.
    """
    if len(left) != len(right) or len(left) == 0:
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(math.fsum(a * a for a in left))
    norm_right = math.sqrt(math.fsum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _vector_for(chunk: Chunk, embeddings: EmbeddingLookup | None) -> Vector | None:
    """Look a chunk's vector up by ``chunk_id`` first, then by ``content_hash``.

    Both keyings are accepted because both exist in the system: the blob cache is
    content-addressed, while an in-memory retrieval pass naturally holds vectors
    per candidate chunk.
    """
    if not embeddings:
        return None
    vector = embeddings.get(chunk.chunk_id)
    if vector is None:
        vector = embeddings.get(chunk.content_hash)
    return vector


def _order_index(version_order: Sequence[str]) -> dict[str, int]:
    """Map version id to its position, newest first."""
    return {version_id: position for position, version_id in enumerate(version_order)}


def version_order_from_registry(settings: Settings) -> list[str]:
    """Indexed versions newest first, taken from the registry's parent chain.

    The chain is what the builds actually recorded, so it survives version labels
    that do not sort (``v2.10`` before ``v2.9``, a bare commit sha) -- parsing
    order out of a ``version_id`` string is forbidden (Rules.md AP-13).
    """
    from axiom.indexing import manifest as mf

    try:
        return mf.load_registry(settings).newest_first()
    except Exception as exc:
        log_degradation(_LOG, "evolutionary", f"registry unreadable: {exc}", "caller order")
        return []


def _diff_pair(newer: Chunk, older: Chunk) -> str:
    """Unified diff from the older member to the newer one.

    Direction matters for reading: additions (``+``) are what the newer version
    introduced. Labels carry the version id rather than a timestamp so the text is
    reproducible across runs (NFR-08).
    """
    return "".join(
        difflib.unified_diff(
            older.text.splitlines(keepends=True),
            newer.text.splitlines(keepends=True),
            fromfile=f"{older.location.file_path}@{older.metadata.version_id}",
            tofile=f"{newer.location.file_path}@{newer.metadata.version_id}",
            n=DIFF_CONTEXT_LINES,
        )
    )


def _make_family(
    members: Sequence[Chunk],
    total_versions: int,
    *,
    with_diffs: bool,
) -> SnippetFamily:
    """Assemble one validated :class:`SnippetFamily` from ordered members.

    ``stability`` is clamped into ``(0, 1]`` before construction: the model
    rejects anything outside it, and a caller that under-reports
    ``total_versions`` would otherwise turn a reporting mistake into an exception
    in the middle of a query.
    """
    denominator = max(total_versions, len(members), 1)
    stability = len(members) / denominator
    diffs = (
        [_diff_pair(members[i], members[i + 1]) for i in range(len(members) - 1)]
        if with_diffs and len(members) > 1
        else []
    )
    head = members[0]
    return SnippetFamily(
        family_id=compute_family_id(head.metadata.symbol, head.location.file_path),
        representative=head,
        members=list(members),
        versions=[member.metadata.version_id for member in members],
        stability=stability,
        diffs=diffs,
    )


def identity_families(
    chunks: Sequence[Chunk],
    total_versions: int = 1,
    *,
    with_diffs: bool = False,
) -> list[SnippetFamily]:
    """One family per chunk -- the degraded rung when grouping is impossible.

    Used when version metadata or embeddings are missing (Rules.md section 3,
    evolutionary-dedupe row). Every result still carries a ``family_id`` and a
    representative, so the formatter and the UI need no special case for the
    degraded path.
    """
    return [_make_family([chunk], total_versions, with_diffs=with_diffs) for chunk in chunks]


def build_families(
    chunks_by_version: Mapping[str, Sequence[Chunk]],
    embeddings: EmbeddingLookup | None = None,
    settings: Settings | None = None,
    *,
    version_order: Sequence[str] | None = None,
    total_versions: int | None = None,
    with_diffs: bool = True,
) -> list[SnippetFamily]:
    """Group chunks observed across versions into snippet families (FR-21).

    Args:
        chunks_by_version: ``version_id -> chunks``. Key order is interpreted as
            **newest first** unless ``version_order`` says otherwise; pass
            :func:`version_order_from_registry` to be explicit.
        embeddings: Vectors keyed by ``chunk_id`` or ``content_hash``. When a
            pair's vectors are absent the pair cannot be compared and does not
            group -- silently merging on identity alone would defeat TC-082.
        settings: Supplies ``dedupe_cosine``. Defaults to the default profile.
        version_order: Explicit newest-first version ordering.
        total_versions: Denominator of ``stability``. Defaults to the number of
            versions given; pass the registry's count when the query only
            returned candidates from some of them (TC-084).
        with_diffs: Attach per-transition unified diffs. Disabled diffs are a
            declared state in the model (``diffs`` is then empty, not wrong).

    Returns:
        Families sorted by descending member count, then ascending ``family_id``,
        so the list is deterministic regardless of dict iteration order.
    """
    active = settings or Settings()
    threshold = active.dedupe_cosine

    order = list(version_order) if version_order is not None else list(chunks_by_version)
    rank_of = _order_index(order)
    denominator = total_versions if total_versions is not None else max(len(order), 1)

    flat: list[Chunk] = []
    unknown_versions: set[str] = set()
    for version_id, chunks in chunks_by_version.items():
        if version_id not in rank_of:
            unknown_versions.add(version_id)
        for chunk in chunks:
            if chunk.metadata.version_id != version_id:
                # The chunk's own metadata is authoritative (invariant 13); a
                # mismatch means the caller bucketed it wrongly, and trusting the
                # bucket would produce a `versions` list the model rejects.
                unknown_versions.add(version_id)
            flat.append(chunk)

    if not flat:
        return []
    if unknown_versions:
        log_degradation(
            _LOG,
            "evolutionary",
            f"version metadata disagrees with the caller's buckets: {sorted(unknown_versions)}",
            "chunk metadata used, unknown versions ordered last",
        )

    if not embeddings:
        log_degradation(
            _LOG, "evolutionary", "no embeddings supplied", "identity families (one member each)"
        )
        return _sorted(identity_families(flat, denominator, with_diffs=False))

    missing = 0
    grouped: list[list[Chunk]] = []
    keyed: dict[tuple[str | None, str], list[Chunk]] = {}
    for chunk in flat:
        keyed.setdefault((chunk.metadata.symbol, chunk.location.file_path), []).append(chunk)

    for key in sorted(keyed, key=lambda k: (k[1], k[0] or "")):
        candidates = sorted(
            keyed[key],
            key=lambda c: (
                rank_of.get(c.metadata.version_id, len(order)),
                c.location.start_line,
                c.chunk_id,
            ),
        )
        groups: list[list[Chunk]] = []
        for chunk in candidates:
            vector = _vector_for(chunk, embeddings)
            if vector is None:
                missing += 1
                groups.append([chunk])
                continue
            placed = False
            for group in groups:
                # One member per version: two same-named functions in one file and
                # one version would otherwise repeat a version id and push
                # `stability` above 1.0, which the model rejects outright.
                if any(m.metadata.version_id == chunk.metadata.version_id for m in group):
                    continue
                if all(
                    _pairwise(chunk, member, vector, embeddings) >= threshold for member in group
                ):
                    group.append(chunk)
                    placed = True
                    break
            if not placed:
                groups.append([chunk])
        grouped.extend(groups)

    if missing:
        log_degradation(
            _LOG,
            "evolutionary",
            f"{missing} chunks have no embedding",
            "those chunks form single-member families",
        )

    families = [_make_family(group, denominator, with_diffs=with_diffs) for group in grouped]
    return _sorted(families)


def _pairwise(chunk: Chunk, member: Chunk, vector: Vector, embeddings: EmbeddingLookup) -> float:
    """Cosine between a candidate and an existing member, ``0.0`` if unknown."""
    other = _vector_for(member, embeddings)
    if other is None:
        return 0.0
    return cosine(vector, other)


def _sorted(families: Sequence[SnippetFamily]) -> list[SnippetFamily]:
    """Deterministic family order: longest history first, then by id."""
    return sorted(families, key=lambda f: (-len(f.members), f.family_id))


def apply_stability_bonus(
    base_score: float, family: SnippetFamily, settings: Settings | None = None
) -> float:
    """Apply the stability bonus to a base score and clamp it to ``[0, 1]``.

    ``final = base * (1 + stability_bonus * stability)``, and only for a family
    spanning two or more versions: a single-version family has a *low* stability
    (``1/total_versions``) because it is new, not unstable, so the multi-version
    gate keeps that case exactly neutral rather than penalising novelty
    (Schema.md section 11).

    The clamp is not defensive padding. With the default ``0.10`` bonus the
    product reaches ``1.1 * base``, so a base of 0.95 would leave the ``[0, 1]``
    range that invariant 17 puts on every public score.

    The bonus factor is read from ``Settings`` rather than from
    :meth:`SnippetFamily.ranking_bonus`, which hardcodes ``0.10``: the constant is
    a PLACEHOLDER tracked by OQ-11 and has to stay sweepable (Rules.md AP-07).
    """
    active = settings or Settings()
    if not family.is_multi_version:
        return min(max(base_score, 0.0), 1.0)
    boosted = base_score * (1.0 + active.stability_bonus * family.stability)
    return min(max(boosted, 0.0), 1.0)


def rank_families(
    scored: Sequence[tuple[SnippetFamily, float]], settings: Settings | None = None
) -> list[tuple[SnippetFamily, float]]:
    """Re-score families with the stability bonus and sort the result.

    Descending by score, ties broken by ascending representative ``chunk_id``
    (Rules.md Rule 4) -- the same total order every other ranked list in the
    system uses, so a tie cannot reorder between runs.
    """
    boosted = [(family, apply_stability_bonus(score, family, settings)) for family, score in scored]
    return sorted(boosted, key=lambda pair: (-pair[1], pair[0].representative.chunk_id))


__all__ = [
    "DIFF_CONTEXT_LINES",
    "EmbeddingLookup",
    "apply_stability_bonus",
    "build_families",
    "cosine",
    "identity_families",
    "rank_families",
    "version_order_from_registry",
]
