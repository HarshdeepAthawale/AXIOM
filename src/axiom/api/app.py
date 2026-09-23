"""Application factory, error handling, and the ``axiom serve`` entry point.

FastAPI and uvicorn are optional dependencies (``pip install 'axiom[serve]'``).
Importing this module on a bare install must therefore succeed and must not cost
anything: every FastAPI symbol is imported inside the function that needs it
(Rules.md AP-06), and a missing dependency becomes
:class:`ServeDependencyError` carrying an install line, not an
``ImportError`` traceback in front of a jury (NFR-07).

Three responsibilities beyond wiring:

**One error body shape.** API.md section 6 promises every non-2xx response is
``{"error": CODE, "detail": "one line"}``. FastAPI's default validation body is
a list of Pydantic error objects under ``detail``, so it is overridden here.
TC-071 asserts no stack trace ever reaches a response body; full tracebacks go
to the log at DEBUG (Rules.md section 9.1).

**A body ceiling enforced before parsing.** :class:`BodyLimitMiddleware` is raw
ASGI rather than a ``BaseHTTPMiddleware`` subclass, because the latter wraps
``receive`` in a way that makes reading the body twice a known hazard -- and
reading the body is the entire job here.

**No CORS, no auth, and both stated in the OpenAPI description.** NG-08 makes
this surface dev-only and unauthenticated by deployment posture, not by
oversight; the description says so, so nobody reading ``/docs`` has to guess
whether it was forgotten.
"""

from __future__ import annotations

import json
import os
from typing import Any

from axiom.api.models import ErrorCode, max_body_bytes
from axiom.config import Settings, get_settings
from axiom.core.errors import AxiomContractError, AxiomError, IndexNotFoundError
from axiom.core.logging import get_logger

_LOG = get_logger("api.app")

#: Default bind host (API.md section 2, ``AXIOM_API_HOST``). Loopback, because
#: an unauthenticated surface (NG-08) has no business listening on a LAN. The
#: Docker compose file overrides it to ``0.0.0.0`` deliberately and in writing.
DEFAULT_API_HOST = "127.0.0.1"

#: What to tell a user who has no server extras installed.
SERVE_INSTALL_HINT = (
    "the HTTP API needs FastAPI and uvicorn: pip install 'axiom[serve]' "
    "(or: pip install fastapi uvicorn)"
)

#: Modules ``axiom serve`` cannot run without.
SERVE_REQUIREMENTS = ("fastapi", "uvicorn")

#: OpenAPI description. Stated in the schema rather than in a README nobody
#: opens next to ``/docs``.
API_DESCRIPTION = """
Agentic code retrieval over a repository index: dense + sparse + structural
retrieval, reciprocal rank fusion, a bounded agent refinement loop, and an
optional cross-encoder rerank.

**This service is dev-only and unauthenticated by design** (NG-08): no login, no
API key, no RBAC, no rate limiting, no CORS. It binds loopback by default and
grants no privilege the filesystem did not already grant -- anyone who can read
the index can already read the repository it was built from. It carries no
uptime guarantee (NG-10) and is not meant to be hosted (NG-13).

Degradation is a 200. A dead signal, a reranker in passthrough, or an exhausted
agent budget returns ranked results with the story in `warnings`. Only an
unresolvable version, an absent index, an unloadable model, or an internal
contract violation is an HTTP error.
""".strip()


class ServeDependencyError(AxiomError):
    """An optional serving dependency is missing.

    Deliberately a subclass of :class:`~axiom.core.errors.AxiomError` so the
    CLI's existing ``except AxiomError`` boundary prints the message and exits
    cleanly. Its ``str()`` is the install line and nothing else -- it is written
    to be shown to a person, not parsed.
    """


def missing_dependencies(requirements: tuple[str, ...] = SERVE_REQUIREMENTS) -> list[str]:
    """Which of ``requirements`` are not importable, without importing them.

    ``find_spec`` answers the question for the price of a path search; actually
    importing FastAPI to find out whether FastAPI exists would put a heavy import
    on ``axiom --help``.
    """
    from importlib.util import find_spec

    absent: list[str] = []
    for name in requirements:
        try:
            found = find_spec(name) is not None
        except (ImportError, ValueError):  # pragma: no cover - broken install
            found = False
        if not found:
            absent.append(name)
    return absent


def require_serve_dependencies() -> None:
    """Raise :class:`ServeDependencyError` with an install line, or return."""
    absent = missing_dependencies()
    if absent:
        raise ServeDependencyError(f"missing {', '.join(absent)} -- {SERVE_INSTALL_HINT}")


def api_host() -> str:
    """Bind host, honouring ``AXIOM_API_HOST`` (API.md section 2).

    Not a :class:`~axiom.config.Settings` field: ``Settings`` carries
    ``api_port`` but no ``api_host``, and it is frozen foundation code. Same
    treatment ``pipeline.MAX_FANOUT_WORKERS`` gets -- a named constant read from
    the documented variable, never a literal at a callsite (Rules.md AP-07).
    """
    return os.environ.get("AXIOM_API_HOST") or DEFAULT_API_HOST


def api_port(settings: Settings | None = None) -> int:
    """Bind port: ``Settings.api_port``, which already honours ``AXIOM_API_PORT``."""
    return (settings if settings is not None else get_settings()).api_port


class BodyLimitMiddleware:
    """Reject a request body larger than ``AXIOM_API_MAX_BODY_BYTES`` (413).

    Raw ASGI on purpose. Two paths:

    * ``Content-Length`` present and over the limit -- rejected without reading a
      byte, which is the case TC-071's 1 MB body exercises.
    * No ``Content-Length`` (a chunked upload) -- the body is buffered up to the
      limit and then replayed to the application, so the downstream handler sees
      a normal request. Buffering is bounded by the limit itself, which is the
      point: an unbounded read to find out whether a read was unbounded is not a
      defence.
    """

    def __init__(self, app: Any, limit: int | None = None) -> None:
        self.app = app
        self._limit = limit

    @property
    def limit(self) -> int:
        """Resolved at request time so a test can move the ceiling."""
        return self._limit if self._limit is not None else max_body_bytes()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        limit = self.limit
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            await _send_error(send, 413, ErrorCode.PAYLOAD_TOO_LARGE, limit)
            return

        if declared is not None:
            await self.app(scope, receive, send)
            return

        buffered: list[bytes] = []
        total = 0
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                # A disconnect mid-body: hand it on and let the server unwind.
                buffered.append(b"")
                break
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > limit:
                await _send_error(send, 413, ErrorCode.PAYLOAD_TOO_LARGE, limit)
                return
            buffered.append(chunk)
            more = bool(message.get("more_body", False))

        body = b"".join(buffered)
        replayed = False

        async def replay() -> Any:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)


def _content_length(scope: Any) -> int | None:
    """Parse ``Content-Length`` from a raw ASGI scope, or ``None``."""
    for key, value in scope.get("headers", []):
        if key.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_error(send: Any, status: int, code: ErrorCode, limit: int) -> None:
    """Emit the API.md section 6 error body straight onto the wire."""
    payload = json.dumps(
        {
            "error": code.value,
            "detail": f"request body exceeds the {limit} byte limit",
        }
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


def create_app(settings: Settings | None = None) -> Any:
    """Build the FastAPI application (API.md sections 2-6).

    Args:
        settings: Configuration every request is answered with. Pinned at
            construction rather than resolved per request so a profile cannot
            change underneath a running server -- an answer that depends on when
            it was asked is not reproducible, and NFR-08 is the whole point.

    Returns:
        A ``fastapi.FastAPI`` instance.

    Raises:
        ServeDependencyError: FastAPI is not installed.
    """
    require_serve_dependencies()

    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from axiom import __version__
    from axiom.api.routes import ApiError, build_router

    resolved = settings if settings is not None else get_settings()

    app = FastAPI(
        title="Axiom - Agentic Code Intelligence",
        version=__version__,
        description=API_DESCRIPTION,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved

    def _body(code: ErrorCode, detail: str, status: int) -> Any:
        return JSONResponse(status_code=status, content={"error": code.value, "detail": detail})

    async def _api_error(_: Request, exc: Exception) -> Any:
        assert isinstance(exc, ApiError)
        _LOG.info(
            "request rejected: %s",
            exc.code.value,
            extra={"axiom_extra": {"stage": "api", "code": exc.code.value, "detail": exc.detail}},
        )
        return _body(exc.code, exc.detail, exc.status_code)

    async def _validation_error(_: Request, exc: Exception) -> Any:
        """422 naming the first failing field, never a Pydantic error list."""
        detail = "request failed validation"
        errors: list[dict[str, Any]] = getattr(exc, "errors", lambda: [])()
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
            message = str(first.get("msg", "invalid value"))
            detail = f"{location or 'request'}: {message}"
        return _body(ErrorCode.VALIDATION_ERROR, detail, 422)

    async def _http_error(request: Request, exc: Exception) -> Any:
        """Give Starlette's own 404/405/413 the documented body shape.

        An unrouted path lands here. API.md section 6 defines no code for "that
        endpoint does not exist", and the closest true statement is that the
        request was malformed -- the path is part of the request -- so it is a
        ``VALIDATION_ERROR`` whose detail names the method and path rather than
        Starlette's bare "Not Found".
        """
        status = getattr(exc, "status_code", 500)
        detail = str(getattr(exc, "detail", "") or "")
        if status == 413:
            code = ErrorCode.PAYLOAD_TOO_LARGE
        elif status >= 500:
            code = ErrorCode.INTERNAL_ERROR
        else:
            code = ErrorCode.VALIDATION_ERROR
        if status in (404, 405):
            detail = f"no route for {request.method} {request.url.path}"
        return _body(code, detail or f"HTTP {status}", status)

    async def _index_error(_: Request, exc: Exception) -> Any:
        """Backstop: an index error that escaped a handler is still a 503."""
        return _body(ErrorCode.INDEX_UNAVAILABLE, str(exc), 503)

    async def _contract_error(_: Request, exc: Exception) -> Any:
        _LOG.error(
            "contract violation reached the API boundary: %s",
            exc,
            exc_info=True,
            extra={"axiom_extra": {"stage": "api", "code": ErrorCode.CONTRACT_VIOLATION.value}},
        )
        return _body(ErrorCode.CONTRACT_VIOLATION, str(exc), 500)

    async def _unhandled(_: Request, exc: Exception) -> Any:
        """Backstop. An INTERNAL_ERROR in the log is a defect to file."""
        _LOG.error(
            "unhandled exception: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
            extra={"axiom_extra": {"stage": "api", "code": ErrorCode.INTERNAL_ERROR.value}},
        )
        return _body(ErrorCode.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}", 500)

    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(IndexNotFoundError, _index_error)
    app.add_exception_handler(AxiomContractError, _contract_error)
    app.add_exception_handler(Exception, _unhandled)

    app.add_middleware(BodyLimitMiddleware)
    app.include_router(build_router(resolved))

    _LOG.info(
        "api application built (profile %s)",
        resolved.profile,
        extra={"axiom_extra": {"stage": "api", "profile": resolved.profile}},
    )
    return app


def serve(
    host: str | None = None,
    port: int | None = None,
    settings: Settings | None = None,
    *,
    log_level: str | None = None,
) -> None:
    """Run the API with uvicorn until interrupted (``axiom serve``, FR-24).

    Args:
        host: Bind address; ``AXIOM_API_HOST`` or loopback when omitted.
        port: Bind port; ``AXIOM_API_PORT`` (via ``Settings``) when omitted.
        settings: Configuration the server answers every request with.
        log_level: uvicorn's level; the project's ``AXIOM_LOG_LEVEL`` when omitted.

    Raises:
        ServeDependencyError: FastAPI or uvicorn is not installed. The message is
            an install line; a caller catching :class:`~axiom.core.errors.AxiomError`
            should print it and exit non-zero rather than let a traceback out.
    """
    require_serve_dependencies()
    import uvicorn

    resolved = settings if settings is not None else get_settings()
    bind_host = host or api_host()
    bind_port = port if port is not None else api_port(resolved)
    app = create_app(resolved)

    _LOG.info(
        "serving on http://%s:%d (docs at /docs, unauthenticated by design)",
        bind_host,
        bind_port,
        extra={"axiom_extra": {"stage": "api", "host": bind_host, "port": bind_port}},
    )
    uvicorn.run(
        app,
        host=bind_host,
        port=bind_port,
        log_level=(log_level or resolved.log_level).lower(),
    )


def __getattr__(name: str) -> Any:
    """Serve ``app`` on first access, so ``uvicorn axiom.api.app:app`` works.

    A module-level ``app = create_app()`` would import FastAPI at import time and
    break the bare-install guarantee for every consumer of this module, the CLI
    included. PEP 562 keeps the import string working without that cost.
    """
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "API_DESCRIPTION",
    "DEFAULT_API_HOST",
    "SERVE_INSTALL_HINT",
    "SERVE_REQUIREMENTS",
    "BodyLimitMiddleware",
    "ServeDependencyError",
    "api_host",
    "api_port",
    "create_app",
    "missing_dependencies",
    "require_serve_dependencies",
    "serve",
]
