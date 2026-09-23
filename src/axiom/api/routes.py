"""The four endpoint handlers, and the FastAPI router that binds them to paths.

Every handler here is an ordinary function over the models in
:mod:`axiom.api.models`. FastAPI appears exactly once, inside
:func:`build_router`, and only to attach those functions to paths. That split is
not ceremony: it means ``POST /v1/query``'s behaviour can be tested with no
server, no TestClient and no ``fastapi`` installed at all, and it keeps the HTTP
layer to what it actually is -- a serialiser in front of
:mod:`axiom.pipeline`, which is the system (TechSpecifications.md section 4.11:
"three thin presentation layers over the same Pydantic models").

**Degradation versus error** (API.md section 5) is the one judgement this module
makes, and it is easy to get backwards. A query that ran and degraded -- a dead
signal, a reranker that would not load, an exhausted agent budget -- is a ``200``
with the story in ``warnings``. Only three things are HTTP errors: a request that
never should have been accepted (422/413), an index or version that does not
exist (404/503), and an internal contract violation (500). Rules.md Rule 3 does
the rest inside the pipeline, and this layer must not second-guess it by turning
an empty result set into a 404.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import lru_cache
from time import perf_counter
from typing import Any, TypeVar

from axiom import pipeline
from axiom.api.models import (
    CHUNK_ID_PATTERN,
    ErrorCode,
    FamiliesResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    VersionsResponse,
    VersionSummary,
    WarningItem,
    collapse_families,
)
from axiom.config import Settings, get_settings
from axiom.core.errors import (
    AxiomContractError,
    DegradationExhaustedError,
    IndexNotFoundError,
)
from axiom.core.logging import get_logger, log_degradation
from axiom.indexing import manifest as mf
from axiom.schema import Chunk

_LOG = get_logger("api.routes")

#: Return type of a handler run off the event loop by :func:`_offload`.
_T = TypeVar("_T")

#: Path prefix API.md section 9 keeps for a hypothetical future IDE plugin
#: (NG-11). It promises nothing about a ``/v2/``; it is one cheap path segment.
API_PREFIX = "/v1"

#: TechSpecifications.md section 4.11 names the endpoints without the prefix and
#: API.md section 3 names them with it. Both are registered rather than one of
#: them being silently declared the loser: the unprefixed aliases are hidden from
#: the OpenAPI schema so the documented surface stays exactly API.md's.
ALIAS_PREFIX = ""

#: Components ``GET /v1/health?warm=true`` constructs, in load order.
WARMABLE = ("embedder", "reranker", "llm")

#: Warning raised when a ``query_type`` override arrives that the pipeline entry
#: point cannot yet forward. See :func:`_pipeline_accepts_query_type`.
QUERY_TYPE_UNSUPPORTED = "QUERY_TYPE_UNSUPPORTED"


class ApiError(Exception):
    """A condition that must surface as a non-2xx response.

    Carries the HTTP status and the machine-readable code together so the
    handler that raises it decides the mapping once, at the site that knows what
    happened, instead of an outer layer re-deriving it from an exception type it
    only partly understands.
    """

    def __init__(self, status_code: int, code: ErrorCode, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail


@lru_cache(maxsize=1)
def _pipeline_accepts_query_type() -> bool:
    """Whether :func:`axiom.pipeline.query` can forward a forced ``QueryType``.

    ``agent.planner.build_plan`` takes a ``query_type`` override and
    ``agent.loop.run`` takes a pre-built ``plan`` -- both ends of API.md section
    3.1's ``query_type`` field exist. The orchestrator between them does not yet
    expose the kwarg, and ``pipeline.py`` belongs to another module owner.

    So the capability is detected rather than assumed: the moment the kwarg
    lands, this surface starts honouring the override with no edit here, and
    until then the request is answered by the classifier with a ``warnings``
    entry saying so. Silently ignoring a field the client set would be the one
    unacceptable option -- it makes the response look inexplicably wrong.
    """
    try:
        return "query_type" in inspect.signature(pipeline.query).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def _settings_for(settings: Settings | None) -> Settings:
    """Resolve the ambient configuration when a caller did not pin one."""
    return settings if settings is not None else get_settings()


def _index_error(exc: IndexNotFoundError, requested_version: str | None) -> ApiError:
    """Map "nothing resolved" onto 404 or 503 (API.md section 6).

    The distinction is whose fault it is. A version the *client* named that does
    not exist is a scoping error against an otherwise healthy index -- 404. No
    index at all is the server having nothing to serve -- 503, the same condition
    a missing registry produces.
    """
    if requested_version is not None:
        return ApiError(404, ErrorCode.VERSION_NOT_FOUND, str(exc))
    return ApiError(503, ErrorCode.INDEX_UNAVAILABLE, str(exc))


# ---------------------------------------------------------------------------
# POST /v1/query
# ---------------------------------------------------------------------------


def run_query(request: QueryRequest, settings: Settings | None = None) -> QueryResponse:
    """Answer one query (API.md section 3.1, Appflow.md Flows 2, 3, 6, 7).

    Args:
        request: An already-validated request body. Validation happened at the
            model, before this function was reached and before any path was
            built -- TC-071's "no filesystem access attempted" assertion depends
            on that ordering.
        settings: Active configuration; the ambient profile when omitted.

    Returns:
        The wire envelope. A degraded run -- no candidates, a dead signal, a
        reranker in passthrough -- returns normally with ``warnings`` populated.

    Raises:
        ApiError: Only for a version that does not resolve (404), an index that
            is not there at all (503), a model with no rung left (503), or an
            internal contract violation (500).
    """
    resolved = _settings_for(settings)
    extra: list[WarningItem] = []
    kwargs: dict[str, Any] = {}

    if request.query_type is not None:
        if _pipeline_accepts_query_type():
            kwargs["query_type"] = request.query_type
        else:
            log_degradation(
                _LOG,
                "api.routes:run_query",
                f"pipeline.query cannot forward query_type={request.query_type.value!r}",
                "the classifier's own decision",
            )
            extra.append(
                WarningItem(
                    code=QUERY_TYPE_UNSUPPORTED,
                    detail=(
                        f"query_type={request.query_type.value!r} was not applied; "
                        "this build's pipeline entry point classifies the query itself"
                    ),
                )
            )

    try:
        result = pipeline.query(
            request.query,
            resolved,
            version_id=request.version,
            top_k=request.top_k,
            all_versions=request.all_versions,
            **kwargs,
        )
    except IndexNotFoundError as exc:
        raise _index_error(exc, request.version) from exc
    except DegradationExhaustedError as exc:
        raise ApiError(503, ErrorCode.MODEL_UNAVAILABLE, str(exc)) from exc
    except AxiomContractError as exc:
        raise ApiError(500, ErrorCode.CONTRACT_VIOLATION, str(exc)) from exc

    body = QueryResponse.from_pipeline(result, extra_warnings=extra)
    if not request.all_versions:
        return body

    # Collapsing is the caller's answer to "the same function came back once per
    # version". It runs only on the all-versions path because on a single-version
    # query every family has exactly one member, and a families block that says
    # nothing is worse than no families block -- a UI would draw a version
    # selector with one entry in it.
    kept, families = collapse_families(body.results)
    return body.model_copy(update={"results": kept, "families": families})


# ---------------------------------------------------------------------------
# GET /v1/versions
# ---------------------------------------------------------------------------


def list_versions(settings: Settings | None = None) -> VersionsResponse:
    """List every indexed version (API.md section 3.2, FR-17, FR-20).

    Reads ``registry.json`` only. Opening N manifests to answer a listing is
    exactly the cost Schema.md section 14.1 duplicated these four fields to
    avoid.

    Raises:
        ApiError: 503 when the registry is corrupt, or when nothing has been
            indexed yet -- ``active_version`` is contractually a string, and
            there is no honest string for "no index exists".
    """
    resolved = _settings_for(settings)
    try:
        registry = mf.load_registry(resolved)
    except IndexNotFoundError as exc:
        raise ApiError(503, ErrorCode.INDEX_UNAVAILABLE, str(exc)) from exc

    records = registry.ordered()
    if not records:
        raise ApiError(
            503,
            ErrorCode.INDEX_UNAVAILABLE,
            f"no version registered under {mf.index_root(resolved)}; run 'axiom index <repo>'",
        )

    active = registry.active_version
    if active not in registry.versions:
        newest = registry.newest_first()
        active = newest[0] if newest else records[-1].version_id
        log_degradation(
            _LOG,
            "api.routes:list_versions",
            f"registry active pointer {registry.active_version!r} is not a known version",
            f"newest registered version {active!r}",
        )

    return VersionsResponse(
        active_version=active,
        versions=[
            VersionSummary(
                version_id=record.version_id,
                created_at=record.created_at,
                chunk_count=record.chunk_count,
                parent_version=record.parent_version,
            )
            for record in records
        ],
    )


# ---------------------------------------------------------------------------
# GET /v1/families
# ---------------------------------------------------------------------------

#: Families returned when a request does not say otherwise. Matches the CLI's
#: ``FAMILY_LIST_LIMIT`` so the two surfaces show the same page by default
#: (Rules.md AP-07: a documented default, never a literal at a callsite).
DEFAULT_FAMILY_LIMIT = 20

#: Ceiling on ``limit``. Building families walks every chunk of every version and
#: loads a blob per distinct content hash, so an unbounded page is a way to ask
#: this process to read the whole corpus into memory.
MAX_FAMILY_LIMIT = 500


def list_families(
    version: str | None = None,
    limit: int = DEFAULT_FAMILY_LIMIT,
    multi_only: bool = False,
    diffs: bool = False,
    settings: Settings | None = None,
) -> FamiliesResponse:
    """Browse snippet families across indexed versions (API.md section 3.3, FR-21).

    The HTTP half of ``axiom families``: same corpus, same grouping, same
    degradation behaviour, because both call :func:`axiom.pipeline.list_families`
    rather than each loading the index their own way.

    ``diffs`` is off by default and costs real work when on -- a unified diff per
    transition, for every family on the page -- so a UI should request it for the
    one family a user expanded, not for the list.

    Args:
        version: Restrict to one version id. ``None`` uses every built version.
        limit: Page size; ``0`` means every match, capped at
            :data:`MAX_FAMILY_LIMIT`.
        multi_only: Only families spanning two or more versions.
        diffs: Attach per-transition unified diffs.
        settings: Configuration; the process default when omitted.

    Raises:
        ApiError: 422 for a malformed ``version`` or a negative ``limit``; 503
            when no index has been built.
    """
    resolved = _settings_for(settings)
    _validate_version_param(version)
    if limit < 0:
        raise ApiError(422, ErrorCode.VALIDATION_ERROR, "limit must be zero or positive")
    page = MAX_FAMILY_LIMIT if limit == 0 else min(limit, MAX_FAMILY_LIMIT)

    try:
        families, version_ids = pipeline.list_families(
            resolved, version=version, multi_only=multi_only, with_diffs=diffs
        )
    except IndexNotFoundError as exc:
        raise _index_error(exc, version) from exc
    except AxiomContractError as exc:
        raise ApiError(500, ErrorCode.CONTRACT_VIOLATION, str(exc)) from exc

    # ``total`` counts what matched, not what fitted, so a client can tell
    # "there are twelve" from "there are four hundred and you asked for twelve".
    return FamiliesResponse(
        families=families[:page], total=len(families), version_ids=list(version_ids)
    )


# ---------------------------------------------------------------------------
# GET /v1/chunk/{chunk_id}
# ---------------------------------------------------------------------------


def get_chunk(chunk_id: str, version: str | None = None, settings: Settings | None = None) -> Chunk:
    """Hydrate one chunk by id (API.md section 3.4).

    Streams ``chunks.jsonl`` and validates only the matching row. The whole-file
    :func:`~axiom.indexing.manifest.read_chunks` would validate 10,000 models to
    return one, on a path the UI hits every time a jury member expands a card.

    Ids are matched by exact string equality and never re-derived (Rules.md
    Rule 1).

    Raises:
        ApiError: 404 when the id is well-formed but absent, 404 for an unknown
            version, 503 when no index resolves at all.
    """
    resolved = _settings_for(settings)
    try:
        version_id = mf.resolve_version(resolved, version)
    except IndexNotFoundError as exc:
        raise _index_error(exc, version) from exc

    path = mf.chunks_path(resolved, version_id)
    for record in mf.iter_chunk_records(path):
        if record.get("chunk_id") != chunk_id:
            continue
        try:
            return Chunk.model_validate(record)
        except Exception as exc:
            # A row whose ids disagree with its own body is corrupted state, not
            # user input: the digests are recomputed by the model itself.
            raise ApiError(
                500,
                ErrorCode.CONTRACT_VIOLATION,
                f"chunk {chunk_id} in version {version_id} failed validation: {exc}",
            ) from exc

    raise ApiError(
        404,
        ErrorCode.CHUNK_NOT_FOUND,
        f"chunk {chunk_id} is not present in version {version_id}",
    )


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------

#: Components already constructed in this process, so a repeat ``?warm=true``
#: reports an empty ``warmed`` list instead of claiming credit twice (API.md
#: section 3.5).
_WARMED: set[str] = set()


def _warm_embedder(settings: Settings) -> str | None:
    """Construct the embedder singleton; return the rung that loaded."""
    from axiom.indexing.embedder import load_embedder

    return load_embedder(settings).model_id


def _warm_reranker(settings: Settings) -> str | None:
    """Construct the cross-encoder; ``None`` when every rung failed."""
    from axiom.rerank.cross_encoder import warm_reranker

    return warm_reranker(settings)


def _warm_llm(settings: Settings) -> str | None:
    """Construct the GGUF adapter; ``None`` when the LLM is off or absent."""
    from axiom.agent.llm import get_llm

    llm = get_llm(settings)
    if llm is None or not llm.available:
        return None
    # Touching the handle is what actually mmaps the weights; an adapter that
    # only knows where the file is has not paid the cold-load cost the demo
    # runbook warms precisely to avoid paying in front of the jury.
    llm.classify("warm")
    return settings.llm_model


def warm_models(settings: Settings | None = None) -> list[str]:
    """Eagerly construct the lazy model singletons (API.md section 3.5).

    Returns:
        The components constructed *on this call*, in load order. Empty when
        everything was already warm.

    Never raises. A model that will not load is a declared degradation with a
    fallback behind it (NFR-07), and the fallback is what the next query will
    use; the 503 ``MODEL_UNAVAILABLE`` case belongs to a ladder with no rungs
    left, which the loaders signal by raising
    :class:`~axiom.core.errors.DegradationExhaustedError` -- not by returning.
    """
    resolved = _settings_for(settings)
    loaders = {
        "embedder": _warm_embedder,
        "reranker": _warm_reranker,
        "llm": _warm_llm,
    }
    warmed: list[str] = []
    for component in WARMABLE:
        if component in _WARMED:
            continue
        if component == "reranker" and not resolved.reranker_enabled:
            continue
        if component == "llm" and not resolved.llm_enabled:
            continue
        try:
            loaded = loaders[component](resolved)
        except DegradationExhaustedError:
            raise
        except Exception as exc:
            log_degradation(
                _LOG,
                f"api.routes:warm:{component}",
                f"{type(exc).__name__}: {exc}",
                "left to construct lazily on the first query",
            )
            continue
        if loaded is None:
            continue
        _WARMED.add(component)
        warmed.append(component)
        _LOG.info(
            "warmed %s (%s)",
            component,
            loaded,
            extra={"axiom_extra": {"stage": "warm", "component": component, "model": loaded}},
        )
    return warmed


def health(warm: bool = False, settings: Settings | None = None) -> HealthResponse:
    """Liveness, and the sanctioned way to pay the cold-load cost before a demo.

    A plain call touches no model and no index beyond asking the registry
    whether one exists, so it stays a true liveness probe: a process with no
    index built is alive, and reporting that as anything but ``200`` would make
    a container healthcheck fail for a condition that is not a fault.
    """
    from axiom import __version__

    resolved = _settings_for(settings)
    started = perf_counter()

    warmed: list[str] = []
    if warm:
        try:
            warmed = warm_models(resolved)
        except DegradationExhaustedError as exc:
            raise ApiError(503, ErrorCode.MODEL_UNAVAILABLE, str(exc)) from exc

    try:
        index_available = bool(mf.resolve_version(resolved))
    except Exception:
        index_available = False

    return HealthResponse(
        status="ok",
        warm=warm,
        warmed=warmed,
        elapsed_ms=round((perf_counter() - started) * 1000.0, 3),
        index_available=index_available,
        profile=resolved.profile,
        version=__version__,
    )


# ---------------------------------------------------------------------------
# FastAPI binding
# ---------------------------------------------------------------------------


def build_router(settings: Settings | None = None) -> Any:
    """Bind the handlers above to their paths and return an ``APIRouter``.

    The only place in the package that imports FastAPI, and it does so inside
    the function (Rules.md AP-06): ``import axiom.api.routes`` must succeed on a
    bare install, because ``axiom --help`` imports the CLI, which imports this,
    and an evaluator whose machine has no ``fastapi`` still needs the CLI.

    Each endpoint is registered twice -- at ``/v1/...`` per API.md section 3 and
    at the bare path per TechSpecifications.md section 4.11. The aliases are
    excluded from the OpenAPI schema, so the documented surface is exactly one
    set of paths while a client typing either gets an answer.

    One trap worth naming, because it fails in a way that looks like something
    else entirely: ``from __future__ import annotations`` turns every annotation
    into a string, and FastAPI resolves those strings against *module* globals.
    An endpoint parameter annotated with a name imported inside this function
    (``Response``, say) therefore does not resolve, and FastAPI silently reads it
    as a required query parameter -- ``GET /v1/health`` answering 422 "response:
    Field required", and ``/openapi.json`` failing to generate. Every annotation
    on the endpoints below refers to a module-level import for that reason.
    ``Path``/``Query`` are safe because they are *default values*, not
    annotations.
    """
    from fastapi import APIRouter, Path, Query

    router = APIRouter()
    resolved = settings

    def _register(method: str, path: str, endpoint: Any, **kwargs: Any) -> None:
        router.add_api_route(f"{API_PREFIX}{path}", endpoint, methods=[method], **kwargs)
        router.add_api_route(
            f"{ALIAS_PREFIX}{path}",
            endpoint,
            methods=[method],
            include_in_schema=False,
            **kwargs,
        )

    async def query_endpoint(request: QueryRequest) -> QueryResponse:
        """Run the full retrieval pipeline and return ranked results."""
        return await _offload(run_query, request, resolved)

    async def versions_endpoint() -> VersionsResponse:
        """List every indexed version and the active pointer."""
        return await _offload(list_versions, resolved)

    async def families_endpoint(
        version: str | None = Query(default=None),
        limit: int = Query(default=DEFAULT_FAMILY_LIMIT, ge=0, le=MAX_FAMILY_LIMIT),
        multi_only: bool = Query(default=False),
        diffs: bool = Query(default=False),
    ) -> FamiliesResponse:
        """Browse snippet families across indexed versions."""
        return await _offload(list_families, version, limit, multi_only, diffs, resolved)

    async def chunk_endpoint(
        chunk_id: str = Path(..., pattern=CHUNK_ID_PATTERN),
        version: str | None = Query(default=None),
    ) -> Chunk:
        """Hydrate one chunk by id."""
        _validate_version_param(version)
        return await _offload(get_chunk, chunk_id, version, resolved)

    async def health_endpoint(warm: bool = Query(default=False)) -> HealthResponse:
        """Liveness probe; ``?warm=true`` pre-loads the models."""
        return await _offload(health, warm, resolved)

    _register("POST", "/query", query_endpoint, response_model=QueryResponse)
    _register("GET", "/versions", versions_endpoint, response_model=VersionsResponse)
    _register("GET", "/families", families_endpoint, response_model=FamiliesResponse)
    _register("GET", "/chunk/{chunk_id}", chunk_endpoint, response_model=Chunk)
    _register("GET", "/health", health_endpoint, response_model=HealthResponse)
    return router


def _validate_version_param(version: str | None) -> None:
    """Apply the request model's ``version`` rule to a query parameter.

    ``GET /v1/chunk/{id}?version=../../etc`` must be rejected by the same rule
    that rejects it in a POST body, and must be rejected *before* a path is
    built (TC-071). Reusing :class:`~axiom.api.models.QueryRequest`'s validator
    keeps one definition of the rule rather than two regexes that can drift.
    """
    if version is None:
        return
    import re

    from axiom.api.models import VERSION_PATTERN

    if not re.fullmatch(VERSION_PATTERN, version):
        raise ApiError(
            422,
            ErrorCode.VALIDATION_ERROR,
            f"version must match {VERSION_PATTERN} (no path separators, no '..', no leading dot)",
        )


async def _offload(function: Callable[..., _T], *args: Any) -> _T:
    """Run a blocking handler off the event loop.

    Every handler here is synchronous and some are genuinely slow -- a cold query
    does FAISS search, SQLite traversal and an ONNX cross-encode. Running that on
    the event loop would stall the health endpoint the demo runbook polls. The
    threadpool is the right tool and not a workaround: FAISS and SQLite both
    release the GIL, which is the same reasoning Design.md section 5.1 applies to
    the fan-out.
    """
    from starlette.concurrency import run_in_threadpool

    # ``run_in_threadpool`` is untyped from this module's point of view (starlette
    # is an optional import), so the cast restores the parametrised return type
    # the endpoints' annotations depend on.
    result: _T = await run_in_threadpool(function, *args)
    return result


__all__ = [
    "ALIAS_PREFIX",
    "API_PREFIX",
    "QUERY_TYPE_UNSUPPORTED",
    "WARMABLE",
    "ApiError",
    "build_router",
    "get_chunk",
    "health",
    "list_versions",
    "run_query",
    "warm_models",
]
