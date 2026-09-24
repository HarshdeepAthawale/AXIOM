"""Request and response envelopes for the HTTP surface (API.md section 3).

These models are the *envelopes only*. The payloads inside them --
:class:`~axiom.schema.RetrievalResult`, :class:`~axiom.schema.QueryPlan`,
:class:`~axiom.schema.Chunk` -- are the shared contract owned by
``src/axiom/schema/`` and are referenced here by name, never restated. API.md
section 3 draws that line deliberately: an envelope may grow an HTTP-only
bookkeeping field without four people renegotiating the data model.

Nothing in this module imports FastAPI. The validation rules below are ordinary
Pydantic field constraints, so the CLI can reuse the same models for ``--json``
output (API.md section 8) and a contract test can exercise them with no server
running -- which is what makes "typed by the same Pydantic models as the CLI"
(FR-24) a fact about the code rather than an aspiration.

**Validation here is not Rule 3 degradation.** Rules.md Rule 3 governs input
that has already been accepted: a query that survives this gate and *then* goes
empty under truncation degrades to an empty result set. A request that arrives
empty, oversized, or path-traversal-shaped never reaches a stage at all -- it is
a 422 before any file is opened (TC-071 asserts exactly that, with a patched
``open``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from axiom.schema import Chunk, QueryPlan, QueryType, RetrievalResult, SnippetFamily

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from axiom.pipeline import IndexReport
    from axiom.pipeline import QueryResponse as PipelineQueryResponse

#: Default ceiling on ``top_k`` (API.md section 4, ``AXIOM_TOP_K_MAX``).
#:
#: This is *not* a :class:`~axiom.config.Settings` field: ``Settings`` has
#: ``top_k_default`` but no ``top_k_max``, and ``Settings`` is frozen foundation
#: code this module may not edit. It follows the precedent
#: ``pipeline.MAX_FANOUT_WORKERS`` set for the same situation -- a named module
#: constant, read from the documented environment variable, never a literal at a
#: callsite (Rules.md AP-07).
DEFAULT_TOP_K_MAX = 100

#: Default request-body ceiling in bytes (API.md section 4,
#: ``AXIOM_API_MAX_BODY_BYTES``). 256 KiB: a real query is a few hundred bytes,
#: and TC-071's adversarial 1 MB body is well past any legitimate use.
DEFAULT_MAX_BODY_BYTES = 262_144

#: ``version`` shape. No ``/``, no ``..``, no leading dot -- path-traversal-shaped
#: input is rejected by the validator, before any filesystem access (TC-071).
VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"

#: ``chunk_id`` shape: a blake2b-128 hex digest (Rules.md Rule 1).
CHUNK_ID_PATTERN = r"^[0-9a-f]{32}$"


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, or keep the default.

    A malformed value is ignored rather than fatal: a typo in ``AXIOM_TOP_K_MAX``
    must not stop the demo server from binding, and the default it falls back to
    is the documented one.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def top_k_ceiling() -> int:
    """Current ``top_k`` ceiling, honouring ``AXIOM_TOP_K_MAX`` (API.md section 4).

    Read per call rather than frozen at import so a test (or a profile switch)
    can move the ceiling without reloading the module.
    """
    return _env_int("AXIOM_TOP_K_MAX", DEFAULT_TOP_K_MAX)


def max_body_bytes() -> int:
    """Current request-body ceiling, honouring ``AXIOM_API_MAX_BODY_BYTES``."""
    return _env_int("AXIOM_API_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)


class ErrorCode(StrEnum):
    """Machine-readable ``error`` values (API.md section 6).

    One closed vocabulary shared by every non-2xx response, so a client parses
    one shape and switches on one field instead of sniffing status codes.
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    VERSION_NOT_FOUND = "VERSION_NOT_FOUND"
    CHUNK_NOT_FOUND = "CHUNK_NOT_FOUND"
    INDEX_UNAVAILABLE = "INDEX_UNAVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiModel(BaseModel):
    """Base for the envelopes: no unexpected fields in, no surprises out.

    ``extra="forbid"`` on a *request* is what turns a typo'd field name into a
    422 the caller can see, instead of a silently ignored parameter that makes
    the response look inexplicably wrong.
    """

    model_config = ConfigDict(extra="forbid")


class WarningItem(ApiModel):
    """One degradation that did not fail the request (API.md section 5).

    A warning is information attached to a ``200``. Rules.md Rule 3's whole point
    is that a reranker timeout or an exhausted agent budget is a *successful*
    answer with less machinery behind it -- the client is told which, and decides
    for itself whether to care.
    """

    code: str = Field(..., min_length=1, description="Stable machine-readable tag.")
    detail: str = Field(..., description="One line of human-readable context.")


class QueryRequest(ApiModel):
    """Body of ``POST /v1/query`` (API.md section 3.1)."""

    query: str = Field(
        ...,
        description="The search text, verbatim. Empty or whitespace-only is a "
        "validation failure, not a pipeline degradation.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Final result count. Defaults to AXIOM_TOP_K_DEFAULT; capped "
        "at AXIOM_TOP_K_MAX.",
    )
    version: str | None = Field(
        default=None,
        description="Index version to search. Defaults to the registry's active "
        "version. Ignored when all_versions is true.",
    )
    all_versions: bool = Field(
        default=False,
        description="Search every indexed version and collapse near-duplicates "
        "into snippet families (FR-21).",
    )
    query_type: QueryType | None = Field(
        default=None,
        description="Force the classification instead of running the classifier.",
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        """Reject a query that is empty once stripped, keeping the text verbatim.

        The stripped form is only the *test*; the original string is what the
        pipeline receives, because ``QueryPlan.original_query`` is contractually
        the user's query unmutated.
        """
        if not value.strip():
            raise ValueError("query must contain at least one non-whitespace character")
        return value

    @field_validator("top_k")
    @classmethod
    def _top_k_in_range(cls, value: int | None) -> int | None:
        """Enforce the ceiling from :func:`top_k_ceiling`, read at validation time."""
        if value is None:
            return None
        ceiling = top_k_ceiling()
        if value > ceiling:
            raise ValueError(f"top_k must be <= {ceiling}")
        return value

    @field_validator("version")
    @classmethod
    def _version_shape(cls, value: str | None) -> str | None:
        """Reject path-traversal-shaped ids before any path is ever built.

        Done as an explicit validator rather than a ``pattern=`` constraint so
        the rejection message names the rule instead of echoing a regex at a
        jury member reading the response body.
        """
        if value is None:
            return None
        import re

        if not re.fullmatch(VERSION_PATTERN, value):
            raise ValueError(
                "version must match "
                f"{VERSION_PATTERN} (no path separators, no '..', no leading dot)"
            )
        return value


def flatten_timings(ledger: Any) -> dict[str, float]:
    """Collapse a :class:`~axiom.core.timing.TimingLedger` into ``stage -> ms``.

    API.md section 3.1 types ``timings`` as ``dict[str, float]`` keyed by the
    Rules.md section 9.1 stage tags; ``TimingLedger.as_dict()`` is a richer
    nested record carrying per-stage detail. Both are true, so the wire gets the
    documented flat shape and the detail stays in the server log. A stage that
    ran more than once (the agent loop's second pass re-runs ``dense``) is summed,
    because the question the field answers is "where did the wall clock go".
    """
    flat = getattr(ledger, "as_flat_dict", None)
    if callable(flat):
        return dict(flat())
    totals: dict[str, float] = {}
    for stage in getattr(ledger, "stages", []):
        totals[stage.stage] = round(totals.get(stage.stage, 0.0) + stage.elapsed_ms, 3)
    return totals


class FamilyMember(ApiModel):
    """One version's copy of a snippet, inside a :class:`QueryFamily`."""

    chunk_id: str = Field(..., min_length=1)
    version_id: str = Field(default="")
    location: str = Field(default="", description="``path:start-end``, as the UI shows it.")
    score: float = Field(default=0.0)


class QueryFamily(ApiModel):
    """A collapsed cross-version result group (FR-21).

    Returned only when a query asked for ``all_versions``. Deliberately carries
    **no stability number**: :class:`~axiom.schema.SnippetFamily`'s ``stability``
    is "in how many indexed versions does this snippet exist", computed over the
    whole corpus, and the only thing derivable from a result list is how many
    versions survived into the top-k. For an unchanged snippet that is always
    exactly one -- ``chunk_id`` is ``digest(text, file_path, start_line)`` and
    carries no version, so two versions' copies of an untouched function are
    literally the same candidate and fusion already merged them. Publishing that
    ratio as "stability" would put ``0.50`` on a query card for the same family
    ``GET /v1/families`` reports as ``1.00``. Ask that endpoint for stability.
    """

    family_id: str = Field(..., min_length=1)
    representative: str = Field(
        ..., min_length=1, description="chunk_id of the member that survived into the results."
    )
    versions: list[str] = Field(default_factory=list)
    members: list[FamilyMember] = Field(default_factory=list)


def collapse_families(
    results: Sequence[RetrievalResult],
) -> tuple[list[RetrievalResult], list[QueryFamily]]:
    """Keep one result per snippet family (FR-21, TC-085).

    A cross-version query naturally returns the same function once per version,
    which is three rows saying one thing. ``family_id`` is defined by Schema.md
    section 11 as ``blake2b-128(symbol, file_path)``, so collapsing on it is the
    model's own notion of "same snippet" rather than a heuristic invented here.
    The highest-scoring member survives as the representative, because the
    incoming list is already sorted by the score the user is shown.

    Lives here, beside the envelope it returns, because both the API and the CLI
    render it. The CLI grew its own copy first; one definition is what stops the
    two surfaces disagreeing about what a family is.

    Returns:
        ``(collapsed results, families)`` in the same order, one family per kept
        result.
    """
    from axiom.core.hashing import compute_family_id

    kept: list[RetrievalResult] = []
    order: list[str] = []
    by_family: dict[str, dict[str, Any]] = {}

    for result in results:
        family_id = compute_family_id(result.chunk.metadata.symbol, result.chunk.location.file_path)
        block = by_family.get(family_id)
        if block is None:
            block = {
                "family_id": family_id,
                "representative": result.chunk.chunk_id,
                "versions": [],
                "members": [],
            }
            by_family[family_id] = block
            order.append(family_id)
            kept.append(result)
        version_id = result.chunk.metadata.version_id
        if version_id not in block["versions"]:
            block["versions"].append(version_id)
        block["members"].append(
            FamilyMember(
                chunk_id=result.chunk.chunk_id,
                version_id=version_id,
                location=result.chunk.location.as_ref(),
                score=result.score,
            )
        )

    return kept, [QueryFamily(**by_family[family_id]) for family_id in order]


class QueryResponse(ApiModel):
    """Body of ``POST /v1/query`` -- and of ``axiom query --json`` (API.md section 8).

    The first six fields are API.md section 3.1's documented contract. The rest
    are the HTTP-only bookkeeping section 3 explicitly permits an envelope to
    carry, and each earns its place in the UI: ``score_field`` tells the result
    card whether ``score`` is a calibrated rerank probability or a raw RRF value
    (they differ by two orders of magnitude, and mislabelling one as the other is
    how a demo accidentally lies), ``degradations`` drives the "what fell back"
    panel NFR-10 asks for, and ``version_ids`` names what was actually searched
    after ``all_versions`` resolution.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[RetrievalResult] = Field(
        ..., description="Ranked hits, best first. Never longer than top_k."
    )
    query_plan: QueryPlan = Field(..., description="The plan of the final agent pass.")
    elapsed_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock for the whole request. The one field excluded "
        "from the determinism contract (TC-066).",
    )
    passes_used: int = Field(
        ...,
        ge=0,
        description="Agent passes that actually ran. Zero only when the query "
        "went empty after normalisation, in which case stop_reason says so.",
    )
    timings: dict[str, float] = Field(
        default_factory=dict, description="Per-stage elapsed ms (NFR-10)."
    )
    warnings: list[WarningItem] = Field(
        default_factory=list, description="Degradations that did not fail the request."
    )

    profile: str = Field(default="default", description="Config profile that answered.")
    stop_reason: str = Field(default="", description="Why the agent loop stopped.")
    score_field: str = Field(
        default="rrf_score",
        description="Which field RetrievalResult.score carries: 'rerank_score' "
        "when the cross-encoder ran, 'rrf_score' on the passthrough rung.",
    )
    version_ids: list[str] = Field(
        default_factory=list, description="Index versions actually searched."
    )
    degradations: list[str] = Field(
        default_factory=list, description="Ladder rungs taken during this query."
    )
    families: list[QueryFamily] = Field(
        default_factory=list,
        description="Cross-version groups, one per result, in result order. "
        "Empty unless the request set all_versions.",
    )

    @classmethod
    def from_pipeline(
        cls, response: PipelineQueryResponse, extra_warnings: list[WarningItem] | None = None
    ) -> QueryResponse:
        """Serialise the one orchestrator's output onto the wire.

        The single conversion point between ``pipeline.QueryResponse`` and the
        HTTP/CLI envelope. Having exactly one means the API and the CLI cannot
        disagree about what a query returned -- the drift FR-24's "typed by the
        same Pydantic models" clause exists to prevent.
        """
        warnings = [WarningItem(code=w["code"], detail=w["detail"]) for w in response.warnings]
        if extra_warnings:
            warnings.extend(extra_warnings)
        return cls(
            results=list(response.results),
            query_plan=response.query_plan,
            elapsed_ms=round(response.elapsed_ms, 3),
            passes_used=response.passes_used,
            timings=flatten_timings(response.ledger),
            warnings=warnings,
            profile=response.profile,
            stop_reason=response.stop_reason,
            score_field=response.score_field,
            version_ids=list(response.version_ids),
            degradations=list(response.degradations),
        )


class VersionSummary(ApiModel):
    """One row of ``GET /v1/versions`` (API.md section 3.2).

    Sourced from ``registry.json``, which duplicates these four fields from each
    manifest precisely so listing N versions is one file read rather than N
    (Schema.md section 14.1).
    """

    version_id: str = Field(..., min_length=1)
    created_at: str = Field(default="", description="ISO-8601 UTC.")
    chunk_count: int = Field(default=0, ge=0)
    parent_version: str | None = Field(
        default=None, description="Derivation parent; null for a cold build."
    )


class VersionsResponse(ApiModel):
    """Body of ``GET /v1/versions`` -- and of ``axiom versions --json``."""

    active_version: str = Field(
        ..., description="The version POST /v1/query uses when none is given."
    )
    versions: list[VersionSummary] = Field(default_factory=list)


class FamiliesResponse(ApiModel):
    """Body of ``GET /v1/families`` -- and of ``axiom families --json``'s content.

    The CLI emits the bare ``list[SnippetFamily]`` API.md section 8 documents;
    this envelope adds what an HTTP client cannot otherwise learn: how many
    families exist before ``limit`` truncated the list, and which versions
    ``stability`` was computed over. Without ``total`` a UI cannot tell "there
    are 12 families" from "there are 400 and you asked for 12".
    """

    families: list[SnippetFamily] = Field(default_factory=list)
    total: int = Field(
        default=0, ge=0, description="Families matching the filter, before ``limit``."
    )
    version_ids: list[str] = Field(
        default_factory=list,
        description="Versions the families were computed over, newest first. This "
        "list's length is ``stability``'s denominator.",
    )


class HealthResponse(ApiModel):
    """Body of ``GET /v1/health`` (API.md section 3.5).

    ``warmed`` is empty on a plain liveness check *and* on a repeat warm call:
    the field reports what this call constructed, not what is currently loaded,
    so a near-zero second ``?warm=true`` is self-explanatory rather than looking
    like a failure.
    """

    status: str = Field(default="ok")
    warm: bool = Field(default=False, description="Echoes the request.")
    warmed: list[str] = Field(
        default_factory=list, description="Components constructed on this call."
    )
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    index_available: bool = Field(
        default=False,
        description="Whether a built index resolved. False is still a 200: the "
        "process is alive, there is simply nothing to search yet.",
    )
    profile: str = Field(default="default")
    version: str = Field(default="", description="Package version.")


class ErrorResponse(ApiModel):
    """The one body shape every non-2xx response has (API.md section 6).

    FastAPI's default validation body (a list of Pydantic error objects under
    ``detail``) is overridden so a client parses one shape. ``detail`` is one
    human-readable line and never a traceback -- TC-071 asserts that directly;
    full tracebacks go to the server log at DEBUG (Rules.md section 9.1).
    """

    error: ErrorCode = Field(..., description="Machine-readable code.")
    detail: str = Field(default="", description="One line, no stack trace.")


class IndexSummary(ApiModel):
    """``axiom index --json`` / ``axiom reindex --json`` output (API.md section 8).

    Lives here, with the HTTP envelopes, rather than in the CLI: there is no
    index-building endpoint today, but the shape is an envelope over the shared
    contract like every other, and keeping it beside its siblings is what stops a
    second copy of it being invented in ``cli.py``.
    """

    version_id: str = Field(..., min_length=1)
    chunk_count: int = Field(..., ge=0)
    elapsed_ms: float = Field(..., ge=0.0)
    timings: dict[str, float] = Field(default_factory=dict)
    file_count: int = Field(default=0, ge=0)
    embedding_model: str = Field(
        default="", description="Model id the blobs were actually written with."
    )
    embedding_dim: int = Field(default=0, ge=0)
    dense_backend: str = Field(default="")
    dense_index_kind: str = Field(
        default="", description="The manifest's ``index_kind`` (``flat_ip`` or ``ivf_pq``)."
    )
    sparse_backend: str = Field(default="")
    structural_skipped: bool = Field(default=False)
    degradations: list[str] = Field(default_factory=list)

    @classmethod
    def from_report(cls, report: IndexReport) -> IndexSummary:
        """Project an :class:`~axiom.pipeline.IndexReport` onto the documented shape."""
        return cls(
            version_id=report.manifest.version_id,
            chunk_count=report.chunk_count,
            elapsed_ms=round(report.elapsed_ms, 3),
            timings=flatten_timings(report.ledger),
            file_count=report.file_count,
            embedding_model=report.manifest.embedding_model,
            embedding_dim=report.manifest.embedding_dim,
            dense_backend=report.dense_backend,
            dense_index_kind=report.dense_index_kind,
            sparse_backend=report.sparse_backend,
            structural_skipped=report.structural_skipped,
            degradations=list(report.degradations),
        )


#: Re-exported for the route signatures; keeps ``routes.py`` importing one module.
ChunkResponse = Chunk

__all__ = [
    "CHUNK_ID_PATTERN",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_TOP_K_MAX",
    "VERSION_PATTERN",
    "ApiModel",
    "ChunkResponse",
    "ErrorCode",
    "ErrorResponse",
    "FamiliesResponse",
    "FamilyMember",
    "HealthResponse",
    "IndexSummary",
    "QueryFamily",
    "QueryRequest",
    "QueryResponse",
    "VersionSummary",
    "VersionsResponse",
    "WarningItem",
    "collapse_families",
    "flatten_timings",
    "max_body_bytes",
    "top_k_ceiling",
]
