"""Suite-wide fixtures and the offline/hermetic guarantees.

TestPlan.md section 1.2 rules 2-4 are enforced here rather than repeated in
every module: no network, no model download, no shared state between tests, and
no ``.axiom/`` outside ``tmp_path``. The env vars are set at *session* scope and
before any Axiom import that reads them, because
:class:`~axiom.config.Settings` is a ``BaseSettings`` -- it samples the
environment at construction time, so setting ``AXIOM_LLM_ENABLED`` inside a test
body would be too late for a ``Settings`` the module already built.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

# Applied at import time: pytest imports conftest before it imports any test
# module, and a test module that builds a Settings at import scope would
# otherwise see the developer's real environment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("AXIOM_LLM_ENABLED", "false")
os.environ.setdefault("AXIOM_OFFLINE", "true")
os.environ.setdefault("AXIOM_LOG_LEVEL", "WARNING")

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _quieten_axiom_logging() -> None:
    """Hold the Axiom logger at WARNING for the whole session.

    ``AXIOM_LOG_LEVEL`` alone does not achieve this:
    :func:`~axiom.core.logging.get_logger` calls ``configure_logging()`` with
    its *default* level, and ``configure_logging`` sets the root logger's level
    before its idempotence guard -- so every lazily imported Axiom module
    resets the level back to INFO as it loads. The handler is installed once and
    never touched again, so a level set there survives. Without this, a full run
    buries real failures under several thousand INFO lines. (The ``get_logger``
    behaviour itself is reported with this workstream's findings.)
    """
    from axiom.core.logging import configure_logging

    configure_logging("WARNING")
    axiom_logger = logging.getLogger("axiom")
    axiom_logger.setLevel(logging.WARNING)
    for handler in axiom_logger.handlers:
        handler.setLevel(logging.WARNING)


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers TestPlan.md section 8.1 declares.

    ``--strict-markers`` is on in ``pyproject.toml`` but only ``slow`` and
    ``integration`` are declared there, and ``pyproject.toml`` is outside this
    workstream's ownership. Registering the rest here keeps the plan's marker
    vocabulary usable without editing a file another agent owns.
    """
    for marker, description in (
        ("smoke", "exercises a real subsystem end to end with fakes; seconds"),
        ("bench", "timing or memory sensitive; must not run concurrently"),
        ("needs_git", "shells out to a real git binary"),
        ("needs_treesitter", "requires the optional tree-sitter extra"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")
    _quieten_axiom_logging()


#: Ceiling, in seconds, for an unmarked test (TestPlan.md section 8.1).
FAST_TEST_BUDGET_S = 1.0

#: Markers that exempt a test from :data:`FAST_TEST_BUDGET_S`.
SLOW_MARKERS = frozenset({"slow", "bench", "smoke", "integration"})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """Fail an unmarked test that takes longer than a second.

    TestPlan.md section 8.1 names this hook as the thing that "actually keeps
    the gate fast": the marker convention is only real if something enforces
    it, and a suite that drifts from 4 minutes to 12 does so one unmarked
    second at a time. A test that legitimately needs longer declares
    ``slow`` / ``bench`` / ``smoke`` and says so in its name.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.passed:
        return
    if report.duration <= FAST_TEST_BUDGET_S:
        return
    if SLOW_MARKERS & {marker.name for marker in item.iter_markers()}:
        return
    report.outcome = "failed"
    report.longrepr = (
        f"{item.nodeid} took {report.duration:.2f}s, over the "
        f"{FAST_TEST_BUDGET_S:.0f}s budget for an unmarked test. Mark it "
        f"slow/bench/smoke or make it faster (TestPlan.md section 8.1)."
    )


@pytest.fixture
def axiom_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """``caplog``, but it can actually see Axiom's records.

    :func:`~axiom.core.logging.configure_logging` sets ``propagate = False`` on
    the ``axiom`` logger, and ``_quieten_axiom_logging`` calls it before the
    first test runs. ``caplog`` installs its handler on the *root* logger, so
    with propagation off no Axiom record ever reaches it -- a test asserting
    "this degradation was logged loudly" then fails, or worse passes only
    because some earlier test happened to reset the flag. Either way the
    assertion is about the ambient logging configuration rather than about the
    behaviour under test.

    This attaches ``caplog``'s handler straight to the ``axiom`` logger and
    lowers its level for the duration, so the assertion is about the record the
    code emits and nothing else. Restores both on the way out.
    """
    logger = logging.getLogger("axiom")
    previous_level = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Root of the committed synthetic fixtures."""
    return FIXTURES


@pytest.fixture(scope="session")
def repo_v1() -> Path:
    """The v1 synthetic JS repo (TestPlan.md section 1.3).

    Session-scoped and read-only. Anything that needs to *write* into a repo
    copies it into ``tmp_path`` first -- rule 4 forbids shared mutable state.
    """
    return FIXTURES / "repo_v1"


@pytest.fixture(scope="session")
def repo_v2() -> Path:
    """The v2 repo: v1 plus modifications, an addition, a deletion, a pure
    rename (``deep/nested.js`` -> ``deep/pipeline.js``, byte identical) and a
    rename-with-edit (``dup/copy_b.js`` -> ``dup/copy_c.js``)."""
    return FIXTURES / "repo_v2"


@pytest.fixture
def repo_v1_copy(repo_v1: Path, tmp_path: Path) -> Path:
    """A writable per-test copy of ``repo_v1``."""
    target = tmp_path / "repo_v1"
    shutil.copytree(repo_v1, target)
    return target


@pytest.fixture
def settings(tmp_path: Path):
    """Ambient :class:`~axiom.config.Settings` with a ``tmp_path``-scoped index root.

    Built through :func:`~axiom.config.get_settings` rather than
    ``Settings(...)`` so the profile-loading precedence chain that the CLI and
    the pipeline actually use is the one under test.
    """
    from axiom.config import get_settings

    return get_settings(
        "default",
        configs_dir=REPO_ROOT / "configs",
        index_root=tmp_path / ".axiom",
        llm_enabled=False,
        reranker_enabled=False,
    )


@pytest.fixture
def indexed_v1(repo_v1: Path, settings) -> Iterator[tuple[object, object]]:
    """Build a real index over ``repo_v1`` with no optional dependency installed.

    Returns ``(settings, IndexReport)``. This is the fixture behind every
    integration assertion in ``test_pipeline.py``: it proves the degraded ladder
    -- hash embedder, numpy dense index, pure-python BM25, sqlite structural --
    produces a *queryable* index, which is the NFR-07 promise in one object.
    """
    from axiom import pipeline

    report = pipeline.build_index_detailed(repo_v1, "v1", settings)
    yield settings, report


@pytest.fixture
def git_repo_v1_v2(repo_v1: Path, repo_v2: Path, tmp_path: Path) -> Path:
    """A real two-commit git repo: commit 1 is v1, commit 2 is v2.

    ``git diff --name-status`` is then exercised against a real binary rather
    than a fabricated diff string (TestPlan.md section 1.3, ``fixture_git_repo``).
    Skips rather than fails where git is unavailable.
    """
    git = shutil.which("git")
    if git is None:  # pragma: no cover - environment dependent
        pytest.skip("git binary not available")
    work = tmp_path / "gitrepo"
    shutil.copytree(repo_v1, work)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Axiom Tests",
        "GIT_AUTHOR_EMAIL": "tests@example.invalid",
        "GIT_COMMITTER_NAME": "Axiom Tests",
        "GIT_COMMITTER_EMAIL": "tests@example.invalid",
    }

    def run(*args: str) -> None:
        subprocess.run([git, *args], cwd=work, env=env, check=True, capture_output=True)

    run("init", "-q")
    run("add", "-A")
    run("commit", "-q", "-m", "v1")
    for entry in list(work.iterdir()):
        if entry.name != ".git":
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    for entry in repo_v2.iterdir():
        target = work / entry.name
        shutil.copytree(entry, target) if entry.is_dir() else shutil.copy2(entry, target)
    run("add", "-A")
    run("commit", "-q", "-m", "v2")
    return work
