"""The Streamlit demo surface (FR-25) and its launcher.

Streamlit is an optional dependency (``pip install 'axiom[serve]'``), so this
package imports cleanly without it: :mod:`axiom.ui.streamlit_app` keeps every
Streamlit symbol inside ``main()``, and :func:`launch` checks for the package
before starting anything. ``axiom ui`` on a machine with no Streamlit prints an
install line, not an ``ImportError`` (NFR-07).

Streamlit cannot be started in-process the way uvicorn can -- it owns its own
server, its own file watcher and its own script-rerun loop, and its supported
entry point is ``streamlit run <script>``. :func:`launch` therefore spawns that
command with the interpreter currently running, which is what keeps the child
inside the same virtualenv as the index it is about to read.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from axiom.config import Settings, get_settings
from axiom.core.errors import AxiomError
from axiom.core.logging import get_logger

_LOG = get_logger("ui")

#: What to tell a user who has no Streamlit installed.
UI_INSTALL_HINT = (
    "the demo UI needs Streamlit: pip install 'axiom[serve]' (or: pip install streamlit)"
)

#: Default base URL the UI tries before falling back to in-process retrieval
#: (API.md section 2, ``AXIOM_API_BASE_URL``).
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


class UiDependencyError(AxiomError):
    """Streamlit is not installed.

    An :class:`~axiom.core.errors.AxiomError` subclass so the CLI's existing
    error boundary prints the message and exits cleanly; ``str()`` is the
    install line, written for a person rather than a parser.
    """


def missing_dependencies() -> list[str]:
    """``["streamlit"]`` when Streamlit is absent, ``[]`` otherwise.

    Answered with ``find_spec`` rather than an import: importing Streamlit costs
    seconds and a page of warnings, which is a lot to pay to discover that it is
    installed.
    """
    from importlib.util import find_spec

    try:
        present = find_spec("streamlit") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        present = False
    return [] if present else ["streamlit"]


def require_ui_dependencies() -> None:
    """Raise :class:`UiDependencyError` with an install line, or return."""
    absent = missing_dependencies()
    if absent:
        raise UiDependencyError(f"missing {', '.join(absent)} -- {UI_INSTALL_HINT}")


def app_path() -> Path:
    """Filesystem path of the script ``streamlit run`` executes.

    Deployment.md's compose file runs ``streamlit run
    src/axiom/ui/streamlit_app.py`` directly; this resolves the same file from
    an installed package, where ``src/`` does not exist.
    """
    return Path(__file__).resolve().parent / "streamlit_app.py"


def api_base_url() -> str:
    """Configured API base URL (``AXIOM_API_BASE_URL``), default loopback."""
    return os.environ.get("AXIOM_API_BASE_URL") or DEFAULT_API_BASE_URL


def launch(
    port: int | None = None,
    api_base_url_value: str | None = None,
    settings: Settings | None = None,
    *,
    headless: bool = False,
    extra_args: tuple[str, ...] = (),
) -> int:
    """Start ``streamlit run`` on the demo app and block until it exits.

    Args:
        port: Listen port; ``Settings.ui_port`` (8501) when omitted.
        api_base_url_value: Base URL the UI should try for HTTP retrieval. It is
            passed through the environment rather than as a script argument
            because Streamlit owns the script's own argv.
        settings: Configuration to resolve the port from.
        headless: Skip opening a browser -- what a container and a smoke test
            both want (TC-072 runs exactly this).
        extra_args: Additional flags forwarded verbatim to ``streamlit run``.

    Returns:
        The child process's exit code.

    Raises:
        UiDependencyError: Streamlit is not installed.
    """
    require_ui_dependencies()

    resolved = settings if settings is not None else get_settings()
    listen_port = port if port is not None else resolved.ui_port

    env = dict(os.environ)
    env["AXIOM_API_BASE_URL"] = api_base_url_value or api_base_url()
    env.setdefault("AXIOM_UI_PORT", str(listen_port))

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path()),
        "--server.port",
        str(listen_port),
    ]
    if headless:
        command += ["--server.headless", "true"]
    command += list(extra_args)

    _LOG.info(
        "starting streamlit on port %d (api base %s)",
        listen_port,
        env["AXIOM_API_BASE_URL"],
        extra={"axiom_extra": {"stage": "ui", "port": listen_port}},
    )
    return subprocess.call(command, env=env)


def __getattr__(name: str) -> Any:
    """Serve ``main`` lazily, so importing this package never imports Streamlit."""
    if name == "main":
        from axiom.ui.streamlit_app import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_API_BASE_URL",
    "UI_INSTALL_HINT",
    "UiDependencyError",
    "api_base_url",
    "app_path",
    "launch",
    "missing_dependencies",
    "require_ui_dependencies",
]
