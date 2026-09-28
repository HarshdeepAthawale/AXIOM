"""The ``axiom`` command line: an argument parser in front of :mod:`axiom.pipeline`.

This module decides nothing. It parses flags, resolves :class:`~axiom.config.Settings`
through the documented precedence chain, calls the one orchestrator, and renders
what comes back -- twice, in two presentations of the *same* object. That is the
whole design (TechSpecifications.md section 4.11): the CLI, the API and the UI are
three thin layers over one pipeline, so a human table and a ``--json`` body can
never tell different stories about the same query (API.md section 8).

Three properties this file owns, and which nothing underneath can provide:

**Exit codes are a contract** (API.md section 7.4, Rules.md section 9.2). Persona
P-3 is a CI consumer: it reads the exit code before it reads a byte of output. So
``0`` is success, ``2`` is a usage or config error (unknown flag, empty query,
invalid profile), ``3`` is a missing or corrupt index, and ``1`` is anything that
should never have happened. No user-facing message on any of those paths carries a
Python traceback -- tracebacks go to the log at ``DEBUG``, which is where
Rules.md section 9.1 puts them.

**Every subcommand speaks JSON** (FR-23). The ``--json`` shape per subcommand is
API.md section 8's table, verbatim: ``query`` emits ``POST /v1/query``'s body,
``classify`` emits the bare ``QueryPlan``, ``versions`` emits ``GET /v1/versions``'s
body, ``families`` emits a ``list[SnippetFamily]``, and the two build commands emit
an index-build summary. Errors emit API.md section 6's ``{"error", "detail"}``
envelope, so a client parses exactly one error shape whether it spoke HTTP or argv.

**Nothing heavy is imported to print ``--help``.** ``rich`` is optional here and the
renderer degrades to plain text without it; ``axiom.pipeline``, the registry, the
versioning package and the eval harness are all imported inside the command that
needs them. ``import axiom.cli`` costs ``typer`` plus the schema, which is what
keeps ``axiom --help`` instant on a cold, bare install (NFR-07).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from pydantic import ValidationError

from axiom import __version__
from axiom.config import Settings, get_settings
from axiom.core.errors import (
    AxiomContractError,
    AxiomError,
    DegradationExhaustedError,
    IndexNotFoundError,
)
from axiom.core.logging import configure_logging, get_logger
from axiom.schema import QueryType, RetrievalResult, SignalKind

_LOG = get_logger("cli")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Exit codes, mirroring API.md section 7.4 and Rules.md section 9.2's taxonomy.
EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_INDEX = 3

#: ``error`` codes for the ``--json`` failure envelope (API.md section 6). The CLI
#: and the HTTP layer share this vocabulary so a client parses one shape.
ERROR_CONFIG = "CONFIG_ERROR"
ERROR_VALIDATION = "VALIDATION_ERROR"
ERROR_INDEX_UNAVAILABLE = "INDEX_UNAVAILABLE"
ERROR_CONTRACT = "CONTRACT_VIOLATION"
ERROR_MODEL = "MODEL_UNAVAILABLE"
ERROR_INTERNAL = "INTERNAL_ERROR"

#: Source lines of a snippet shown per result card, before ``--snippet-lines``.
#:
#: A display cap, not a tunable: it cannot move a score or change a ranking, so it
#: is a named module constant rather than an invented ``Settings`` field, the same
#: treatment ``pipeline.MAX_FANOUT_WORKERS`` gets for the same reason (Rules.md
#: AP-07). The flag exists because the right number depends on the terminal, and a
#: terminal is not something a config profile can know.
SNIPPET_MAX_LINES = 12

#: Families listed by ``axiom families`` before ``--limit`` says otherwise. A large
#: repository has one family per function; an unbounded default would page a demo
#: terminal off the screen.
FAMILY_LIST_LIMIT = 20

#: Where ``axiom eval`` looks for the evaluation driver. It is a script rather than
#: a package module on purpose (it enforces the eval-profile and reportability
#: gates that only a results file needs), so the CLI delegates to it instead of
#: forking a second implementation of those gates.
EVAL_SCRIPT_RELATIVE = Path("scripts") / "run_eval.py"

#: Import paths tried, in order, for the FastAPI application object.
API_APP_CANDIDATES: tuple[str, ...] = ("axiom.api.app:app", "axiom.api:app")

#: Streamlit entrypoint, resolved next to the installed ``axiom.ui`` package so it
#: is found from a wheel as well as from a source checkout.
UI_APP_FILENAME = "streamlit_app.py"

app = typer.Typer(
    name="axiom",
    help="Axiom -- agentic code retrieval. CPU-only, multi-signal, version-aware.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# --------------------------------------------------------------------------
# Global state and settings resolution
# --------------------------------------------------------------------------


@dataclass
class _Globals:
    """Flags accepted before the subcommand, e.g. ``axiom --profile eval query ...``.

    Every one of these is also accepted *after* the subcommand, because that is
    where people actually type them. The subcommand's value wins when both are
    given; the two boolean kill-switches OR together, since ``--no-rerank``
    anywhere means the user wants the reranker off.
    """

    profile: str | None = None
    version: str | None = None
    top_k: int | None = None
    no_rerank: bool = False
    no_agent: bool = False
    log_level: str | None = None
    index_root: Path | None = None
    json_output: bool = False
    configs_dir: Path | None = None


def _state(ctx: typer.Context) -> _Globals:
    """The global flag block, creating an empty one when the callback was bypassed.

    ``ctx.obj`` is ``None`` when a command function is called directly -- which is
    exactly what the tests in TestPlan.md section 3.8 do -- so this never assumes
    the callback ran.
    """
    if not isinstance(ctx.obj, _Globals):
        ctx.obj = _Globals()
    return ctx.obj


def _repo_root() -> Path:
    """Repository root of a source checkout: ``src/axiom/cli.py`` -> three up."""
    return Path(__file__).resolve().parents[2]


def _configs_dir(override: Path | None = None) -> Path:
    """Directory holding the profile YAMLs.

    The working directory wins so a user can keep a tuned ``configs/`` next to the
    repository they are indexing; the checkout's own directory is the fallback, so
    ``axiom --profile eval`` works from anywhere rather than silently loading no
    profile at all and running the defaults under an eval banner.
    """
    if override is not None:
        return override
    local = Path("configs")
    if local.is_dir():
        return local
    return _repo_root() / "configs"


def _resolve_settings(
    ctx: typer.Context,
    *,
    profile: str | None = None,
    top_k: int | None = None,
    no_rerank: bool = False,
    no_agent: bool = False,
    log_level: str | None = None,
    index_root: Path | None = None,
    **overrides: Any,
) -> Settings:
    """Build :class:`Settings` from the global block plus this command's flags.

    CLI flags outrank the environment, which outranks the profile YAML, which
    outranks the field defaults (Rules.md section 7) -- ``get_settings`` owns that
    chain, so this function only decides *which* overrides exist. ``None`` means
    "not given" and is dropped rather than written as a null.
    """
    state = _state(ctx)
    resolved_profile = profile or state.profile
    effective: dict[str, Any] = {
        "top_k_default": top_k if top_k is not None else state.top_k,
        "log_level": (log_level or state.log_level or "").upper() or None,
        "index_root": index_root if index_root is not None else state.index_root,
    }
    if no_rerank or state.no_rerank:
        effective["reranker_enabled"] = False
    if no_agent or state.no_agent:
        effective["agent_enabled"] = False
    effective.update(overrides)

    settings = get_settings(
        profile=resolved_profile,
        configs_dir=_configs_dir(state.configs_dir),
        **effective,
    )
    # Honour Setup.md section 7.1's AXIOM_LOG_FORMAT here rather than in the
    # logging core: the format is a presentation choice, and the CLI is the only
    # surface that knows whether a human or a CI job is reading stderr.
    configure_logging(
        settings.log_level,
        json_output=os.environ.get("AXIOM_LOG_FORMAT", "console").lower() == "json",
    )
    return settings


# --------------------------------------------------------------------------
# Output primitives
# --------------------------------------------------------------------------


def _console() -> Any | None:
    """A ``rich`` console, or ``None`` when rich is not installed.

    ``rich`` is a declared dependency but not one of the four the degraded path is
    guaranteed (NFR-07): a judge who installed the wheel with ``--no-deps``, or ran
    from a vendored tree, still gets readable output. Every renderer below has a
    plain-text twin for that reason.
    """
    try:
        from rich.console import Console
    except ImportError:  # pragma: no cover - exercised only on a rich-less install
        return None
    return Console(stderr=False, soft_wrap=False)


def _echo(text: str = "") -> None:
    """Human-readable line to stdout, bypassing rich markup interpretation."""
    print(text)


def _emit_json(payload: Any) -> None:
    """Write one JSON document to stdout.

    Key order is insertion order, never sorted: the envelope's field order is part
    of how the document reads, and every value in it is already deterministic
    (NFR-08). ``default=str`` is the backstop for a ``Path`` that slipped into a
    summary, never a licence to serialise arbitrary objects.
    """
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _fail(
    code: int,
    error: str,
    detail: str,
    *,
    json_output: bool,
    remediation: str | None = None,
) -> NoReturn:
    """Report a failure in the caller's chosen format and exit non-zero.

    The message never contains a traceback (API.md section 7.4). The exception, if
    any, was already logged with its type; the stack lives at ``DEBUG``, reachable
    with ``--log-level DEBUG``, which is where a developer looks and where a judge
    does not have to.
    """
    if json_output:
        body: dict[str, str] = {"error": error, "detail": detail}
        if remediation:
            body["remediation"] = remediation
        _emit_json(body)
    else:
        print(f"error: {detail}", file=sys.stderr)
        if remediation:
            print(f"  try: {remediation}", file=sys.stderr)
    raise typer.Exit(code)


@contextmanager
def _boundary(json_output: bool) -> Iterator[None]:
    """Map every escaping exception onto API.md section 7.4's exit codes.

    Rule 3 means almost nothing reaches here: a bad query, a dead signal and a
    missing model all degrade inside the pipeline. What does reach here is the
    short list of states a caller must be told about rather than handed a ranking
    for -- no index, a violated internal contract, an exhausted degradation ladder
    -- plus the backstop for a genuine bug, which is reported as one.
    """
    try:
        yield
    except (typer.Exit, typer.Abort, KeyboardInterrupt):
        raise
    except IndexNotFoundError as exc:
        _LOG.debug("index error", exc_info=True)
        _fail(
            EXIT_INDEX,
            ERROR_INDEX_UNAVAILABLE,
            str(exc),
            json_output=json_output,
            remediation="axiom index <repo> --version-id <id>",
        )
    except AxiomContractError as exc:
        _LOG.error("contract violation: %s", exc)
        _LOG.debug("contract violation", exc_info=True)
        _fail(EXIT_INTERNAL, ERROR_CONTRACT, str(exc), json_output=json_output)
    except DegradationExhaustedError as exc:
        _LOG.debug("degradation ladder exhausted", exc_info=True)
        _fail(EXIT_INTERNAL, ERROR_MODEL, str(exc), json_output=json_output)
    except ValidationError as exc:
        _LOG.debug("settings validation failed", exc_info=True)
        reported = exc.errors()
        first: dict[str, Any] = dict(reported[0]) if reported else {}
        where = ".".join(str(part) for part in first.get("loc", ())) or "settings"
        _fail(
            EXIT_USAGE,
            ERROR_CONFIG,
            f"invalid configuration for {where}: {first.get('msg', exc)}",
            json_output=json_output,
        )
    except AxiomError as exc:
        _LOG.debug("axiom error", exc_info=True)
        _fail(
            EXIT_INTERNAL,
            ERROR_INTERNAL,
            f"{type(exc).__name__}: {exc}",
            json_output=json_output,
        )
    except OSError as exc:
        _LOG.debug("filesystem error", exc_info=True)
        _fail(
            EXIT_INDEX,
            ERROR_INDEX_UNAVAILABLE,
            f"{type(exc).__name__}: {exc}",
            json_output=json_output,
        )
    except Exception as exc:
        _LOG.error("unexpected %s: %s", type(exc).__name__, exc)
        _LOG.debug("unexpected error", exc_info=True)
        _fail(
            EXIT_INTERNAL,
            ERROR_INTERNAL,
            f"{type(exc).__name__}: {exc}",
            json_output=json_output,
        )


# --------------------------------------------------------------------------
# Result rendering
# --------------------------------------------------------------------------


def _signal_breakdown(signals: dict[SignalKind, int]) -> str:
    """``dense #2 · sparse #1`` -- the per-signal first-stage ranks (FR-14, FR-25).

    Ordered by :data:`axiom.retrieval.fusion.SIGNAL_ORDER` rather than by dict
    order so two runs print the same line, and so the order matches the tie-break
    order fusion itself used.
    """
    from axiom.retrieval.fusion import SIGNAL_ORDER

    parts = [f"{s.value} #{signals[s]}" for s in SIGNAL_ORDER if s in signals]
    return " · ".join(parts) if parts else "none"


def _snippet_lines(text: str, limit: int) -> tuple[str, int]:
    """Clip a chunk body to ``limit`` lines, reporting how many were hidden."""
    if limit <= 0:
        return "", text.count("\n") + 1
    lines = text.splitlines() or [text]
    if len(lines) <= limit:
        return "\n".join(lines), 0
    return "\n".join(lines[:limit]), len(lines) - limit


def _render_result_plain(
    position: int,
    result: RetrievalResult,
    *,
    snippet_lines: int,
    family: dict[str, Any] | None,
) -> None:
    """Plain-text result card. The rich card below shows exactly these fields."""
    meta = result.chunk.metadata
    symbol = meta.symbol or meta.kind.value
    _echo(f"[{position}] {result.chunk.location.as_ref()}   score {result.score:.4f}   {symbol}")
    _echo(f"    signals  {_signal_breakdown(result.signals)}")
    _echo(f"    why      {result.match_reason}")
    if result.optimization_hint:
        _echo(f"    hint     {result.optimization_hint}")
    if family is not None:
        versions = ", ".join(family["versions"])
        members = len(family["members"])
        _echo(f"    family   {family['family_id'][:8]}  {versions}  ({members} member(s) shown)")
    body, hidden = _snippet_lines(result.chunk.text, snippet_lines)
    if body:
        start = result.chunk.location.start_line
        for offset, line in enumerate(body.splitlines()):
            _echo(f"    {start + offset:>5} | {line}")
        if hidden:
            _echo(f"          | ... {hidden} more line(s)")
    _echo()


def _render_result_rich(
    console: Any,
    position: int,
    result: RetrievalResult,
    *,
    snippet_lines: int,
    family: dict[str, Any] | None,
) -> None:
    """Rich result card: ``file:line`` header, snippet, score, reason, breakdown."""
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    meta = result.chunk.metadata
    symbol = meta.symbol or meta.kind.value

    facts = Table.grid(padding=(0, 1))
    facts.add_column(style="dim", justify="right", no_wrap=True)
    facts.add_column(overflow="fold")
    facts.add_row("signals", _signal_breakdown(result.signals))
    facts.add_row("why", result.match_reason)
    if result.optimization_hint:
        facts.add_row("hint", Text(result.optimization_hint, style="yellow"))
    if family is not None:
        facts.add_row(
            "family",
            f"{family['family_id'][:8]}  {', '.join(family['versions'])}  "
            f"({len(family['members'])} member(s) shown)",
        )

    body, hidden = _snippet_lines(result.chunk.text, snippet_lines)
    renderables: list[Any] = [facts]
    if body:
        renderables.append(
            Syntax(
                body,
                meta.language or "text",
                line_numbers=True,
                start_line=result.chunk.location.start_line,
                word_wrap=False,
                theme="ansi_dark",
            )
        )
        if hidden:
            renderables.append(Text(f"... {hidden} more line(s)", style="dim"))

    group: Any
    if len(renderables) == 1:
        group = renderables[0]
    else:
        from rich.console import Group

        group = Group(*renderables)

    console.print(
        Panel(
            group,
            title=f"[bold]{position}. {result.chunk.location.as_ref()}[/bold]  [dim]{symbol}[/dim]",
            title_align="left",
            subtitle=f"[bold]score {result.score:.4f}[/bold]",
            subtitle_align="right",
            border_style="cyan",
        )
    )


def _summarise_degradations(notes: Sequence[str]) -> list[str]:
    """Collapse per-file ladder rungs into one line each, for the terminal.

    ``capture_degradations`` reports one entry per *component*, and some ladders
    name a component per file -- the chunker's is ``chunker:<path>``. A
    tree-sitter-less build of sixteen files therefore produces sixteen entries
    that differ only in the path, which is exactly right in ``--json`` and
    unreadable on a terminal, where it buries the three rungs that actually
    changed how retrieval behaves.

    Entries whose component shares a prefix before the first ``:`` and whose
    reason and fallback are identical collapse to one line with a count. Nothing
    is dropped and nothing is reworded: an entry that does not fit the
    ``component: reason -> fallback`` shape is passed through verbatim.
    """
    order: list[tuple[str, str, str]] = []
    counts: dict[tuple[str, str, str], int] = {}
    for note in notes:
        component, separator, rest = note.partition(": ")
        reason, arrow, fallback = rest.partition(" -> ")
        if not separator or not arrow:
            key = (note, "", "")
        else:
            key = (component.split(":", 1)[0], reason, fallback)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1

    lines: list[str] = []
    for key in order:
        prefix, reason, fallback = key
        line = prefix if not reason and not fallback else f"{prefix}: {reason} -> {fallback}"
        count = counts[key]
        lines.append(f"{line} (x{count})" if count > 1 else line)
    return lines


def _render_query_human(
    response: Any,
    results: Sequence[RetrievalResult],
    *,
    text: str,
    top_k: int,
    snippet_lines: int,
    families: dict[str, dict[str, Any]],
) -> None:
    """Header, result cards, then what degraded -- in that order.

    ``results`` is passed separately rather than read off the response because
    ``--all-versions`` collapses the list to one row per snippet family, and the
    response object is frozen. Everything else -- plan, passes, versions, timings
    -- is read straight off the response, so the human view and the ``--json``
    view cannot drift into showing different facts (API.md section 8).

    Degradations print last and always: NFR-07's claim is that the system keeps
    answering with fewer signals, and that claim is only checkable if the run says
    out loud which rung it took.
    """
    console = _console()
    plan = response.query_plan
    header = [
        f"query      {text!r}",
        f"type       {plan.query_type.name}   "
        f"passes {response.passes_used} ({response.stop_reason})",
        f"versions   {', '.join(response.version_ids) or 'none'}   profile {response.profile}",
        f"results    {len(results)} of top-k {top_k} in {response.elapsed_ms:.0f} ms",
    ]
    if console is not None:
        from rich.panel import Panel

        console.print(
            Panel(
                "\n".join(header),
                border_style="dim",
                title="axiom query",
                title_align="left",
            )
        )
    else:
        for line in header:
            _echo(line)
        _echo()

    if not results:
        _echo("no results. the index may be empty, or every signal was unavailable.")
    for position, result in enumerate(results, start=1):
        family = families.get(result.chunk.chunk_id)
        if console is not None:
            _render_result_rich(
                console, position, result, snippet_lines=snippet_lines, family=family
            )
        else:
            _render_result_plain(position, result, snippet_lines=snippet_lines, family=family)

    if response.score_field != "rerank_score":
        # Honesty over flattery: with no cross-encoder loaded the score *is* the
        # raw RRF sum, whose ceiling is sum(w)/(k+1) ~ 0.0164. Printing it without
        # its scale would read as "no confidence" to a judge (OQ-10).
        _echo(
            f"note: score field is {response.score_field} (no reranker loaded); "
            "raw RRF is bounded by ~1/(rrf_k+1), so small values are expected."
        )
    for warning in response.warnings:
        _echo(f"warning [{warning['code']}] {warning['detail']}")
    for note in _summarise_degradations(response.degradations):
        _echo(f"degraded: {note}")


# --------------------------------------------------------------------------
# Family collapsing for --all-versions
# --------------------------------------------------------------------------


def _collapse_families(
    results: Sequence[RetrievalResult],
) -> tuple[list[RetrievalResult], dict[str, dict[str, Any]]]:
    """Keep one result per snippet family, keyed for the renderer (FR-21, TC-085).

    The grouping itself lives in :func:`axiom.api.models.collapse_families`, so
    the CLI and ``POST /v1/query`` cannot disagree about what a family is -- this
    is only the re-keying the human renderer wants, ``chunk_id`` of the surviving
    representative to its family block, so a result card can look its own family
    up without re-deriving anything.
    """
    from axiom.api.models import collapse_families

    kept, families = collapse_families(results)
    return kept, {family.representative: family.model_dump(mode="json") for family in families}


# --------------------------------------------------------------------------
# Version-id resolution shared by index and reindex
# --------------------------------------------------------------------------


def _checked_version_id(value: str | None, *, flag: str, json_output: bool) -> str | None:
    """Refuse a traversal-shaped ``version_id`` as a usage error, not an index error.

    ``manifest.require_valid_version_id`` already refuses it -- before any path is
    built from it (Security.md's traversal row) -- but it refuses with
    ``IndexNotFoundError``, which the taxonomy maps to exit 3. A malformed flag
    value is a *usage* error, and exit 2 is what API.md section 7.4 promises for
    one, so the shape is checked here where the value is still a flag.
    """
    if value is None:
        return None
    from axiom.indexing.manifest import VERSION_ID_PATTERN, is_valid_version_id

    if not is_valid_version_id(value):
        _fail(
            EXIT_USAGE,
            ERROR_VALIDATION,
            f"{flag} {value!r} is not a usable version id",
            json_output=json_output,
            remediation=f"version ids must match {VERSION_ID_PATTERN.pattern}",
        )
    return value


def _next_version_id(settings: Settings) -> str:
    """A fresh ``vN`` label that no registered version already uses.

    Counting registered versions is not parsing order out of a label (Rules.md
    AP-13 forbids that): ordering still comes from the registry's parent chain.
    This only needs a name nothing else has taken.
    """
    from axiom.indexing import manifest as mf

    existing = set(mf.load_registry(settings).versions)
    candidate = len(existing) + 1
    while f"v{candidate}" in existing:
        candidate += 1
    return f"v{candidate}"


def _resolve_build_identity(
    repo: Path, settings: Settings, version_id: str | None, rev: str = "HEAD"
) -> tuple[str, str | None]:
    """Resolve ``(version_id, commit_sha)`` for a build target (FR-16).

    An explicit ``--version-id`` always wins. Failing that a git sha is the best
    label available, because it is the only one that means the same thing to the
    repository and to the index. A plain directory with no git falls back to
    ``vN``, which is what lets Axiom index an unpacked tarball.
    """
    from axiom.versioning.gitdiff import resolve_version_identity

    resolved, sha = resolve_version_identity(repo, rev=rev, version_override=version_id)
    if resolved is None:
        resolved = _next_version_id(settings)
    return resolved, sha


# --------------------------------------------------------------------------
# Global callback
# --------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Config profile: default, demo, fast, accurate, eval."),
    ] = None,
    version: Annotated[
        str | None, typer.Option("--version", help="Index version id to operate on.")
    ] = None,
    top_k: Annotated[int | None, typer.Option("--top-k", help="Final result count.")] = None,
    no_rerank: Annotated[
        bool, typer.Option("--no-rerank", help="Disable the cross-encoder rerank stage.")
    ] = False,
    no_agent: Annotated[
        bool, typer.Option("--no-agent", help="Disable refinement: exactly one retrieval pass.")
    ] = False,
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="DEBUG, INFO, WARNING, ERROR.")
    ] = None,
    index_root: Annotated[
        Path | None, typer.Option("--index-root", help="Root of the on-disk index tree.")
    ] = None,
    configs_dir: Annotated[
        Path | None, typer.Option("--configs-dir", help="Directory holding the profile YAMLs.")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable output for every subcommand.")
    ] = False,
    app_version: Annotated[
        bool, typer.Option("--app-version", help="Print the installed Axiom version and exit.")
    ] = False,
) -> None:
    """Global flags. Each is also accepted after the subcommand, where it wins.

    ``--version`` names an *index* version, not this program's version: API.md
    section 7.3 uses it that way for ``query`` and ``families``, and a CLI whose
    global flag meant one thing and whose subcommand flag meant another would be
    worse than the small surprise of ``--app-version``.
    """
    ctx.obj = _Globals(
        profile=profile,
        version=version,
        top_k=top_k,
        no_rerank=no_rerank,
        no_agent=no_agent,
        log_level=log_level,
        index_root=index_root,
        json_output=json_output,
        configs_dir=configs_dir,
    )
    if app_version:
        if json_output:
            _emit_json({"name": "axiom", "version": __version__})
        else:
            _echo(f"axiom {__version__}")
        raise typer.Exit(EXIT_OK)
    if ctx.invoked_subcommand is None:
        # Flags with no subcommand is a usage error, not a no-op: exiting 0 here
        # would let a typo'd CI invocation pass silently (API.md section 7.4).
        _echo(ctx.get_help())
        raise typer.Exit(EXIT_USAGE)


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------


@contextmanager
def _no_worktree() -> Iterator[None]:
    """Stand-in for :func:`worktree_at` when ``--at`` was not given.

    Lets the index path use one ``with`` statement for both cases instead of
    duplicating the build call under an if/else.
    """
    yield None


@app.command("index")
def index_command(
    ctx: typer.Context,
    repo: Annotated[Path, typer.Argument(help="Repository root to index.")],
    version_id: Annotated[
        str | None,
        typer.Option("--version-id", help="Label for this build. Default: git sha, else vN."),
    ] = None,
    at_rev: Annotated[
        str | None,
        typer.Option(
            "--at",
            help="Index the repo as it stood at this git ref (tag, branch or sha), "
            "instead of the working tree. Required to build an index of a past "
            "release (US-7, FR-20) and to give FR-21's families distinct members.",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild even if this version id already exists.")
    ] = False,
    parent: Annotated[
        str | None, typer.Option("--parent", help="Version this one descends from.")
    ] = None,
    no_activate: Annotated[
        bool, typer.Option("--no-activate", help="Leave the registry's active version alone.")
    ] = False,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    index_root: Annotated[Path | None, typer.Option("--index-root")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Cold-build all three indexes over a repository tree (FR-04..FR-10)."""
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        settings = _resolve_settings(
            ctx, profile=profile, index_root=index_root, log_level=log_level
        )
        from axiom import pipeline
        from axiom.indexing import manifest as mf

        root = repo.expanduser()
        if not root.is_dir():
            _fail(
                EXIT_USAGE,
                ERROR_VALIDATION,
                f"{root} is not a directory",
                json_output=json_output,
                remediation="axiom index <path-to-repo>",
            )

        _checked_version_id(version_id, flag="--version-id", json_output=json_output)
        _checked_version_id(parent, flag="--parent", json_output=json_output)
        resolved_id, sha = _resolve_build_identity(root, settings, version_id)
        if mf.manifest_path(settings, resolved_id).is_file() and not force:
            _fail(
                EXIT_USAGE,
                ERROR_VALIDATION,
                f"version {resolved_id!r} is already built under {mf.index_root(settings)}",
                json_output=json_output,
                remediation=f"axiom index {root} --version-id {resolved_id} --force",
            )

        from axiom.versioning.checkout import worktree_at

        # `--at` swaps the tree we read without touching the user's checkout.
        # When it cannot be honoured the context yields None and we index the
        # working tree instead, saying so -- a stale ref must not be silently
        # indexed as if it were the requested release (Rule 3).
        with worktree_at(root, at_rev) if at_rev else _no_worktree() as materialised:
            if at_rev and materialised is None:
                _fail(
                    EXIT_USAGE,
                    ERROR_VALIDATION,
                    f"could not materialise {at_rev!r} from {root}",
                    json_output=json_output,
                    remediation=f"git -C {root} tag --list   # check the ref exists",
                )
            source = materialised or root
            if materialised is not None:
                # Stamp the commit the ref actually names; otherwise every
                # historical index would claim HEAD's sha and the manifest
                # chain would be a lie.
                from axiom.versioning.checkout import resolve_ref

                resolved_ref = resolve_ref(root, at_rev) if at_rev else None
                if resolved_ref is not None:
                    sha = resolved_ref[0]
            if materialised is not None and version_id is None:
                # The label should name the release, not the detached sha the
                # worktree happens to sit on.
                resolved_id = at_rev or resolved_id
            report = pipeline.build_index_detailed(
                source,
                resolved_id,
                settings,
                make_active=not no_activate,
                commit_sha=sha,
                parent_version=parent,
            )

        if json_output:
            _emit_json(report.as_dict())
            return
        shown = f"{root}@{at_rev}" if at_rev else str(root)
        _echo(f"indexed {shown} as version {report.manifest.version_id}")
        _echo(f"  chunks     {report.chunk_count} from {report.file_count} file(s)")
        _echo(
            f"  embedder   {report.manifest.embedding_model} (dim {report.manifest.embedding_dim})"
        )
        _echo(f"  dense      {report.dense_backend} / {report.dense_index_kind}")
        _echo(f"  sparse     {report.sparse_backend}")
        _echo(f"  structural {'skipped' if report.structural_skipped else 'built'}")
        _echo(f"  elapsed    {report.elapsed_ms:.0f} ms")
        for note in _summarise_degradations(report.degradations):
            _echo(f"  degraded:  {note}")


# --------------------------------------------------------------------------
# reindex
# --------------------------------------------------------------------------


@app.command("reindex")
def reindex_command(
    ctx: typer.Context,
    repo: Annotated[Path, typer.Argument(help="Repository root.")] = Path("."),
    to_rev: Annotated[
        str | None, typer.Option("--to", help="Revision to build. Default: HEAD.")
    ] = None,
    from_rev: Annotated[
        str | None, typer.Option("--from", help="Revision the parent version was built from.")
    ] = None,
    full: Annotated[
        bool, typer.Option("--full", help="Force a full rebuild instead of a diff.")
    ] = False,
    version_id: Annotated[
        str | None, typer.Option("--version-id", help="Label for the new build.")
    ] = None,
    parent: Annotated[
        str | None, typer.Option("--parent", help="Version to derive from. Default: active.")
    ] = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    index_root: Annotated[Path | None, typer.Option("--index-root")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Incrementally rebuild only what a git diff says changed (FR-18, FR-19)."""
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        settings = _resolve_settings(
            ctx, profile=profile, index_root=index_root, log_level=log_level
        )
        from axiom.indexing import manifest as mf
        from axiom.versioning.incremental import reindex as run_reindex

        root = repo.expanduser()
        if not root.is_dir():
            _fail(
                EXIT_USAGE,
                ERROR_VALIDATION,
                f"{root} is not a directory",
                json_output=json_output,
                remediation="axiom reindex <path-to-repo> --to <rev>",
            )

        _checked_version_id(version_id, flag="--version-id", json_output=json_output)
        _checked_version_id(parent, flag="--parent", json_output=json_output)
        registry = mf.load_registry(settings)
        old_version = parent or state.version or registry.active_version
        if old_version is not None and old_version not in registry.versions:
            _fail(
                EXIT_INDEX,
                ERROR_INDEX_UNAVAILABLE,
                f"parent version {old_version!r} is not registered",
                json_output=json_output,
                remediation="axiom versions list",
            )

        new_id, _sha = _resolve_build_identity(root, settings, version_id, rev=to_rev or "HEAD")
        if new_id == old_version:
            # A rebuild of the same label would replace the parent it derives
            # from; the manifest model rejects parent_version == version_id
            # outright, so catching it here yields a usable message instead.
            _fail(
                EXIT_USAGE,
                ERROR_VALIDATION,
                f"new version {new_id!r} is also the parent; nothing changed since it was built",
                json_output=json_output,
                remediation=f"axiom reindex {root} --version-id <new-id>",
            )

        report = run_reindex(
            old_version,
            new_id,
            settings,
            repo_path=root,
            old_rev=from_rev,
            new_rev=to_rev,
            force_full=full,
        )

        if json_output:
            _emit_json(report.to_dict())
            return
        _echo(f"reindexed {root} as version {report.version_id} ({report.strategy})")
        _echo(f"  parent     {report.parent_version}")
        _echo(
            f"  chunks     {report.chunk_count} ({report.chunks_carried} carried, "
            f"{report.chunks_rebuilt} rebuilt, {report.chunks_dropped} dropped)"
        )
        _echo(f"  embeddings {report.embed_calls} call(s), {report.blobs_reused} blob(s) reused")
        _echo(f"  elapsed    {report.elapsed_ms:.0f} ms")
        for note in _summarise_degradations(report.degradations):
            _echo(f"  degraded:  {note}")


# --------------------------------------------------------------------------
# query
# --------------------------------------------------------------------------


@app.command("query")
def query_command(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="What to search for, in plain language.")],
    version: Annotated[
        str | None, typer.Option("--version", help="Query one specific indexed version.")
    ] = None,
    all_versions: Annotated[
        bool,
        typer.Option("--all-versions", help="Search every version; collapse snippet families."),
    ] = False,
    top_k: Annotated[int | None, typer.Option("--top-k", help="Final result count.")] = None,
    query_type: Annotated[
        QueryType | None, typer.Option("--type", help="Force the classification.")
    ] = None,
    snippet_lines: Annotated[
        int, typer.Option("--snippet-lines", help="Source lines shown per card; 0 hides the body.")
    ] = SNIPPET_MAX_LINES,
    no_rerank: Annotated[bool, typer.Option("--no-rerank")] = False,
    no_agent: Annotated[bool, typer.Option("--no-agent")] = False,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    index_root: Annotated[Path | None, typer.Option("--index-root")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run the full retrieval pipeline and print ranked results (FR-01..FR-14)."""
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        settings = _resolve_settings(
            ctx,
            profile=profile,
            top_k=top_k,
            no_rerank=no_rerank,
            no_agent=no_agent,
            index_root=index_root,
            log_level=log_level,
        )
        if not text.strip():
            # The pipeline degrades an empty query into an empty result set; the
            # CLI still owes its caller exit 2 (API.md section 7.4, TC-009), and
            # short-circuiting here means an empty query never touches the index.
            _fail(
                EXIT_USAGE,
                ERROR_VALIDATION,
                "query is empty after whitespace stripping",
                json_output=json_output,
                remediation='axiom query "how is input normalised before dispatch"',
            )

        _checked_version_id(version or state.version, flag="--version", json_output=json_output)

        from axiom import pipeline

        # ``_resolve_settings`` already folded --top-k into the settings object,
        # so this is the one resolved width, not a second guess at it.
        width = settings.top_k_default
        response = pipeline.query(
            text,
            settings,
            version_id=version or state.version,
            top_k=width,
            all_versions=all_versions,
            query_type=query_type,
        )

        results = list(response.results)
        families: dict[str, dict[str, Any]] = {}
        if all_versions:
            results, families = _collapse_families(results)

        if json_output:
            payload = response.as_dict()
            payload["results"] = [r.model_dump(mode="json") for r in results]
            # Always present, empty off the --all-versions path. A key that
            # appears only sometimes forces every client to branch on its
            # absence, and the two surfaces would then disagree about the
            # envelope depending on a flag (TC-085 needs the members reachable;
            # nothing needs the key to vanish).
            payload["families"] = list(families.values())
            _emit_json(payload)
            return

        _render_query_human(
            response,
            results,
            text=text,
            top_k=width,
            snippet_lines=snippet_lines,
            families=families,
        )


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------


@app.command("classify")
def classify_command(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="Query to classify and plan.")],
    query_type: Annotated[
        QueryType | None, typer.Option("--type", help="Force the classification.")
    ] = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Classify and plan a query without retrieving anything (FR-01..FR-03).

    US-1's acceptance criterion is literal about the first two lines of output --
    ``query_type=STRUCTURAL`` and ``extracted_identifiers=[...]`` -- so they are
    printed first, bare, and greppable, before anything decorative.
    """
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        settings = _resolve_settings(ctx, profile=profile, log_level=log_level)
        if not text.strip():
            _fail(
                EXIT_USAGE,
                ERROR_VALIDATION,
                "query is empty after whitespace stripping",
                json_output=json_output,
                remediation='axiom classify "which files call parseIntent before dispatch"',
            )

        from axiom.agent.planner import build_plan

        plan = build_plan(text, settings, query_type=query_type)
        if json_output:
            # API.md section 8: the QueryPlan object alone, unwrapped -- no
            # envelope, so no timings block here (see the report's note on NFR-10).
            _emit_json(plan.model_dump(mode="json"))
            return

        _echo(f"query_type={plan.query_type.name}")
        _echo(f"extracted_identifiers={json.dumps(plan.extracted_identifiers)}")
        _echo(f"expansion_terms={json.dumps(plan.expansion_terms)}")
        _echo(f"sub_queries={json.dumps(plan.sub_queries)}")
        weights = "  ".join(
            f"{signal.value}={weight:.2f}" for signal, weight in plan.strategy_weights.items()
        )
        _echo(f"strategy_weights  {weights}")


# --------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------


def _registry_payload(registry: Any) -> dict[str, Any]:
    """``GET /v1/versions``'s body (API.md section 3.2), oldest version first."""
    return {
        "active_version": registry.active_version,
        "versions": [
            {
                "version_id": record.version_id,
                "created_at": record.created_at,
                "chunk_count": record.chunk_count,
                "parent_version": record.parent_version,
            }
            for record in registry.ordered()
        ],
    }


def _versions_impl(
    ctx: typer.Context,
    action: str,
    version_id: str | None,
    purge: bool,
    profile: str | None,
    index_root: Path | None,
    log_level: str | None,
    json_output: bool,
) -> None:
    """Shared body for ``axiom versions`` and its singular alias."""
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        settings = _resolve_settings(
            ctx, profile=profile, index_root=index_root, log_level=log_level
        )
        from axiom.indexing import manifest as mf

        verb = action.lower()
        if verb not in {"list", "use", "rm"}:
            _fail(
                EXIT_USAGE,
                ERROR_VALIDATION,
                f"unknown action {action!r}",
                json_output=json_output,
                remediation="axiom versions list | use <id> | rm <id>",
            )
        target = _checked_version_id(
            version_id or state.version, flag="version id", json_output=json_output
        )
        if verb in {"use", "rm"} and not target:
            _fail(
                EXIT_USAGE,
                ERROR_VALIDATION,
                f"'{verb}' needs a version id",
                json_output=json_output,
                remediation=f"axiom versions {verb} <id>",
            )

        if verb == "use":
            registry = mf.set_active_version(settings, str(target))
        elif verb == "rm":
            registry = mf.remove_version(settings, str(target), delete_files=purge)
        else:
            registry = mf.load_registry(settings)

        payload = _registry_payload(registry)
        if json_output:
            _emit_json(payload)
            return

        if verb == "use":
            _echo(f"active version is now {registry.active_version}")
        elif verb == "rm":
            _echo(
                f"removed {target} from the registry" + (" and deleted its files" if purge else "")
            )
            _echo("blobs are shared; run 'axiom gc' to reclaim the unreferenced ones")
        if not payload["versions"]:
            _echo("no versions indexed yet")
            return
        _echo(f"{'':2}{'version':24} {'created':22} {'chunks':>8}  parent")
        for record in payload["versions"]:
            marker = "*" if record["version_id"] == payload["active_version"] else " "
            _echo(
                f"{marker} {record['version_id']:24} {record['created_at']:22} "
                f"{record['chunk_count']:>8}  {record['parent_version'] or '-'}"
            )


@app.command("versions")
def versions_command(
    ctx: typer.Context,
    action: Annotated[str, typer.Argument(help="list, use, or rm.")] = "list",
    version_id: Annotated[str | None, typer.Argument(help="Version id for use/rm.")] = None,
    purge: Annotated[
        bool, typer.Option("--purge", help="With rm: also delete the version directory.")
    ] = False,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    index_root: Annotated[Path | None, typer.Option("--index-root")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Registry operations: list (default), use <id>, rm <id> (FR-17, FR-20)."""
    _versions_impl(ctx, action, version_id, purge, profile, index_root, log_level, json_output)


@app.command("version", hidden=True)
def version_command(
    ctx: typer.Context,
    action: Annotated[str, typer.Argument(help="list, use, or rm.")] = "list",
    version_id: Annotated[str | None, typer.Argument(help="Version id for use/rm.")] = None,
    purge: Annotated[bool, typer.Option("--purge")] = False,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    index_root: Annotated[Path | None, typer.Option("--index-root")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Alias of ``axiom versions``.

    TestPlan.md TC-069 and TC-079 invoke the singular form; API.md section 7.1
    settles on the plural and records the discrepancy rather than erasing it.
    Supporting both costs one delegation and means neither committed test contract
    nor the canonical document is wrong at the terminal.
    """
    _versions_impl(ctx, action, version_id, purge, profile, index_root, log_level, json_output)


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------


@app.command("families")
def families_command(
    ctx: typer.Context,
    version: Annotated[
        str | None, typer.Option("--version", help="Only this version's chunks.")
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Families to show; 0 means all.")
    ] = FAMILY_LIST_LIMIT,
    multi_only: Annotated[
        bool, typer.Option("--multi-only", help="Only families spanning two or more versions.")
    ] = False,
    diffs: Annotated[
        bool, typer.Option("--diffs", help="Attach per-transition unified diffs.")
    ] = False,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    index_root: Annotated[Path | None, typer.Option("--index-root")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List snippet families across indexed versions -- browse, no query (FR-21).

    Grouping needs vectors: two same-named functions in one file are one family
    only if they are also near-identical (TC-082). Those come from the
    content-addressed blob store, and when numpy or the blobs are absent every
    family degrades to a single member and says so, which is the honest answer
    rather than a merge that never compared anything.
    """
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        settings = _resolve_settings(
            ctx, profile=profile, index_root=index_root, log_level=log_level
        )
        from axiom import pipeline

        target = _checked_version_id(
            version or state.version, flag="--version", json_output=json_output
        )
        families, version_ids = pipeline.list_families(
            settings, version=target, multi_only=multi_only, with_diffs=diffs
        )
        shown = families if limit <= 0 else families[:limit]

        if json_output:
            _emit_json([family.model_dump(mode="json") for family in shown])
            return

        _echo(
            f"{len(families)} family(ies) over {len(version_ids)} version(s); showing {len(shown)}"
        )
        _echo()
        for family in shown:
            head = family.representative
            symbol = head.metadata.symbol or head.metadata.kind.value
            _echo(f"{family.family_id[:8]}  {symbol}  {head.location.file_path}")
            _echo(
                f"          versions {', '.join(family.versions)}  "
                f"stability {family.stability:.2f}  "
                f"{'multi-version' if family.is_multi_version else 'single-version'}"
            )
            for text in family.diffs:
                for line in text.splitlines():
                    _echo(f"          {line}")
            _echo()


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------


def _load_eval_script() -> Any | None:
    """Import ``scripts/run_eval.py`` by path, or ``None`` when it is not there.

    It lives outside the package because it is an experiment driver, not a library:
    it enforces the eval-profile gate, the placeholder banner and the provenance
    block that only a results file needs (PRD.md section 2.2, Rules.md AP-14).
    Delegating keeps those gates in one place; a wheel installed without the
    repository simply reports that the script is missing.
    """
    import importlib.util

    candidates = [Path.cwd() / EVAL_SCRIPT_RELATIVE, _repo_root() / EVAL_SCRIPT_RELATIVE]
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("axiom_run_eval", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


@app.command("eval")
def eval_command(
    ctx: typer.Context,
    task: Annotated[str, typer.Option("--task", help="MTEB task name.")] = "AppsRetrieval",
    split: Annotated[str, typer.Option("--split", help="train or test.")] = "test",
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Truncate to N queries. Marks the run non-reportable."),
    ] = None,
    corpus_path: Annotated[
        Path | None, typer.Option("--corpus-path", help="Vendored BEIR-shaped dataset directory.")
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="'module.path:factory' returning a SearchBackend."),
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Results file path.")] = None,
    predictions: Annotated[
        Path | None, typer.Option("--predictions", help="Raw run in MTEB prediction format.")
    ] = None,
    top_k: Annotated[int | None, typer.Option("--top-k", help="Candidate depth per query.")] = None,
    profile: Annotated[
        str | None, typer.Option("--profile", help="Only 'eval' is accepted.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run the MTEB adapter against a retrieval task (FR-22, FR-26)."""
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        module = _load_eval_script()
        if module is None:
            _fail(
                EXIT_USAGE,
                ERROR_CONFIG,
                f"{EVAL_SCRIPT_RELATIVE} not found next to the working directory or the package",
                json_output=json_output,
                remediation="run axiom eval from a source checkout of the repository",
            )

        argv: list[str] = ["--task", task, "--split", split]
        argv += ["--profile", profile or state.profile or "eval"]
        if limit is not None:
            argv += ["--limit", str(limit)]
        if corpus_path is not None:
            argv += ["--corpus-path", str(corpus_path)]
        if backend is not None:
            argv += ["--backend", backend]
        if out is not None:
            argv += ["--out", str(out)]
        if predictions is not None:
            argv += ["--predictions", str(predictions)]
        if top_k is not None or state.top_k is not None:
            argv += ["--top-k", str(top_k if top_k is not None else state.top_k)]
        if state.configs_dir is not None:
            argv += ["--configs-dir", str(state.configs_dir)]
        if json_output:
            argv.append("--json")

        try:
            code = int(module.main(argv))
        except SystemExit as exc:  # the script's own _fail path, already reported
            code = int(exc.code or 0)
        raise typer.Exit(code)


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------


@app.command("serve")
def serve_command(
    ctx: typer.Context,
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int | None, typer.Option("--port", help="Bind port.")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Restart on source changes.")] = False,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    index_root: Annotated[Path | None, typer.Option("--index-root")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Start the FastAPI service (FR-24).

    ``--json`` only affects the startup banner: this is a long-running process, and
    API.md section 8 marks the flag not-applicable for it. The profile and index
    root resolved here are exported into the environment so the server process --
    which builds its own ``Settings`` -- sees the same configuration the flags
    asked for, rather than silently reverting to the ambient one.
    """
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        settings = _resolve_settings(
            ctx, profile=profile, index_root=index_root, log_level=log_level
        )
        bind_port = port if port is not None else settings.api_port

        try:
            import uvicorn
        except ImportError:
            _fail(
                EXIT_USAGE,
                ERROR_CONFIG,
                "uvicorn is not installed",
                json_output=json_output,
                remediation="pip install 'axiom[serve]'",
            )

        import importlib
        import importlib.util

        target: str | None = None
        for candidate in API_APP_CANDIDATES:
            module_name, _, attribute = candidate.partition(":")
            try:
                if importlib.util.find_spec(module_name) is None:
                    continue
                module = importlib.import_module(module_name)
            except (ImportError, ValueError):
                continue
            if hasattr(module, attribute):
                target = candidate
                break
        if target is None:
            _fail(
                EXIT_USAGE,
                ERROR_CONFIG,
                f"no FastAPI app found at any of {', '.join(API_APP_CANDIDATES)}",
                json_output=json_output,
                remediation="check that src/axiom/api/app.py defines `app`",
            )

        os.environ["AXIOM_PROFILE"] = settings.profile
        os.environ["AXIOM_INDEX_ROOT"] = str(settings.index_root)
        banner = {
            "service": "api",
            "app": target,
            "host": host,
            "port": bind_port,
            "profile": settings.profile,
        }
        if json_output:
            _emit_json(banner)
        else:
            _echo(f"serving {target} on http://{host}:{bind_port} (profile {settings.profile})")
        uvicorn.run(
            target, host=host, port=bind_port, reload=reload, log_level=settings.log_level.lower()
        )


# --------------------------------------------------------------------------
# ui
# --------------------------------------------------------------------------


@app.command("ui")
def ui_command(
    ctx: typer.Context,
    port: Annotated[int | None, typer.Option("--port", help="Streamlit port.")] = None,
    api_base_url: Annotated[
        str | None, typer.Option("--api-base-url", help="Base URL of a running axiom serve.")
    ] = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    index_root: Annotated[Path | None, typer.Option("--index-root")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Start the Streamlit UI (FR-25).

    Streamlit owns its own process model -- it re-executes the script on every
    interaction -- so this launches ``python -m streamlit run`` as a child rather
    than importing the app. Configuration crosses the boundary through the
    environment, which is the only channel a Streamlit script reads.
    """
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        settings = _resolve_settings(
            ctx, profile=profile, index_root=index_root, log_level=log_level
        )
        bind_port = port if port is not None else settings.ui_port

        import importlib.util

        if importlib.util.find_spec("streamlit") is None:
            _fail(
                EXIT_USAGE,
                ERROR_CONFIG,
                "streamlit is not installed",
                json_output=json_output,
                remediation="pip install 'axiom[serve]'",
            )

        from axiom import ui as ui_package

        script = Path(str(ui_package.__file__)).parent / UI_APP_FILENAME
        if not script.is_file():
            _fail(
                EXIT_USAGE,
                ERROR_CONFIG,
                f"{script} does not exist",
                json_output=json_output,
                remediation="check that src/axiom/ui/streamlit_app.py is present",
            )

        env = dict(os.environ)
        env["AXIOM_PROFILE"] = settings.profile
        env["AXIOM_INDEX_ROOT"] = str(settings.index_root)
        if api_base_url:
            env["AXIOM_API_BASE_URL"] = api_base_url

        banner = {
            "service": "ui",
            "script": str(script),
            "port": bind_port,
            "profile": settings.profile,
        }
        if json_output:
            _emit_json(banner)
        else:
            _echo(
                f"starting streamlit on http://127.0.0.1:{bind_port} (profile {settings.profile})"
            )

        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script),
            "--server.port",
            str(bind_port),
            # Local-only, like `axiom serve` (NG-08): Streamlit otherwise binds every
            # interface. Headless skips its first-run email prompt, which blocks a
            # fresh install until someone presses Enter; the URL is printed above.
            "--server.address",
            "127.0.0.1",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        raise typer.Exit(subprocess.call(command, env=env))


# --------------------------------------------------------------------------
# gc
# --------------------------------------------------------------------------


@app.command("gc")
def gc_command(
    ctx: typer.Context,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be deleted; mutate nothing.")
    ] = False,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    index_root: Annotated[Path | None, typer.Option("--index-root")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Delete embedding blobs no registered version references (TC-080).

    Not one of FR-23's nine subcommands, and deliberately so: TC-080 depends on it
    and API.md section 7.3 documents it for that reason, filed with the rest of the
    naming residue. ``axiom versions rm`` stays instant by leaving blobs alone;
    this is where reclaiming them happens, separately and reversibly.
    """
    state = _state(ctx)
    json_output = json_output or state.json_output
    with _boundary(json_output):
        settings = _resolve_settings(
            ctx, profile=profile, index_root=index_root, log_level=log_level
        )
        from axiom.core.timing import TimingLedger
        from axiom.indexing import manifest as mf

        ledger = TimingLedger()
        with ledger.measure("blob") as detail:
            removed, freed = mf.gc_blobs(settings, dry_run=dry_run)
            detail["blobs"] = len(removed)

        if json_output:
            _emit_json(
                {
                    "dry_run": dry_run,
                    "deleted": removed,
                    "bytes_reclaimed": freed,
                    "timings": ledger.as_dict(),
                }
            )
            return
        verb = "would delete" if dry_run else "deleted"
        _echo(f"{verb} {len(removed)} unreferenced blob(s), {freed} byte(s)")


__all__ = [
    "EXIT_INDEX",
    "EXIT_INTERNAL",
    "EXIT_OK",
    "EXIT_USAGE",
    "FAMILY_LIST_LIMIT",
    "SNIPPET_MAX_LINES",
    "app",
]
