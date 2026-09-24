#!/usr/bin/env python3
"""Build an Axiom index over a repository, with the timings PB-01/PB-02 are measured from.

Usage::

    python scripts/build_index.py data/demo_repo --version v1
    python scripts/build_index.py data/demo_repo --version v2 --parent v1 --time-it
    python scripts/build_index.py data/demo_repo --version v1 --profile fast --json

This is the harness behind two budgets in TestPlan.md section 5.3:

* **PB-01** -- cold index of 10k chunks in 12 minutes. Measured with ``--time-it``
  from process start (not from the first chunk) to the moment ``registry.json``
  is on disk, because model load and import time are part of what a user waits
  through and excluding them would flatter the number.
* **PB-02** -- incremental reindex of 50 changed files in 45 seconds, measured
  the same way with ``--parent`` naming the version being derived from.

Two things it refuses to do quietly. It never reports a duration without also
reporting the **rungs that actually ran** -- a build that fell back to the hash
embedder is four orders of magnitude faster than one that ran Qwen3 through
ONNX, and a timing that does not say which one happened is not a measurement.
And it never overwrites an existing version without ``--force``, because an
interrupted rebuild over a live index is how a demo loses its corpus.

Peak RSS is sampled here too when ``psutil`` is importable, for the same reason
``bench_latency.py`` samples it: NFR-05's 4 GB ceiling applies to the build as
well as the query, and the build is where the embedder's batch size bites.

Exit codes follow API.md section 7.4: 0 success, 2 usage/config, 3 missing
input, 1 anything unexpected.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:  # running from a clone without `pip install -e .`
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from axiom import pipeline  # noqa: E402
from axiom.config import Settings, get_settings  # noqa: E402
from axiom.core.errors import AxiomError, IndexNotFoundError  # noqa: E402
from axiom.core.logging import configure_logging, get_logger  # noqa: E402
from axiom.core.timing import TimingLedger  # noqa: E402
from axiom.indexing import manifest as mf  # noqa: E402

_EXIT_OK = 0
_EXIT_INTERNAL = 1
_EXIT_USAGE = 2
_EXIT_INPUT = 3

#: RSS sampling period, in seconds. Matches ``bench_latency.py`` and
#: TestPlan.md section 5.1's "every 100 ms" so the two harnesses' peak numbers
#: are comparable.
RSS_SAMPLE_INTERVAL_S = 0.1

_LOGGER = get_logger("scripts.build_index")


def build_parser() -> argparse.ArgumentParser:
    """CLI surface. Mirrors ``axiom index``'s flags in API.md section 7.2."""
    parser = argparse.ArgumentParser(
        prog="build_index.py",
        description="Build an Axiom index over a repository.",
    )
    parser.add_argument("repo", type=Path, help="Repository root to index.")
    parser.add_argument("--version", required=True, help="Logical version id, e.g. 'v1'.")
    parser.add_argument("--profile", default=None, help="Config profile (default, fast, eval...).")
    parser.add_argument("--index-root", type=Path, default=None, help="Override .axiom location.")
    parser.add_argument("--parent", default=None, help="Version this one is derived from.")
    parser.add_argument("--commit-sha", default=None, help="40-char git sha being indexed.")
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="Build the version but leave the registry's active version alone.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing version of this id.",
    )
    parser.add_argument(
        "--time-it",
        action="store_true",
        help="Report wall clock from process start and per-stage timings (PB-01/PB-02).",
    )
    parser.add_argument(
        "--measure-rss",
        action="store_true",
        help="Sample peak RSS every 100 ms while building (needs psutil).",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON object on stdout.")
    parser.add_argument("--out", type=Path, default=None, help="Also write the JSON row here.")
    parser.add_argument("--log-level", default=None, help="Override AXIOM_LOG_LEVEL.")
    return parser


class _RssSampler:
    """Background peak-RSS sampler.

    A daemon thread rather than a signal timer so it works identically on
    Windows and inside a container, and so a build that finishes in 200 ms is
    not charged the cost of installing and tearing down a handler. Degrades to
    ``None`` -- never an error -- when ``psutil`` is absent, which is the whole
    point: a missing optional dependency may cost a number, never a build.
    """

    def __init__(self, enabled: bool) -> None:
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any = None
        if not enabled:
            return
        try:
            import psutil
        except ImportError:
            _LOGGER.warning("psutil not installed; peak RSS will not be reported")
            return
        self._process = psutil.Process()

    def __enter__(self) -> _RssSampler:
        if self._process is None:
            return self
        self.peak_bytes = int(self._process.memory_info().rss)
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(RSS_SAMPLE_INTERVAL_S):
            try:
                rss = int(self._process.memory_info().rss)
            except Exception:  # pragma: no cover - the process is exiting
                return
            if self.peak_bytes is None or rss > self.peak_bytes:
                self.peak_bytes = rss


def pin_log_level(level: str) -> None:
    """Make ``--log-level`` stick across the lazily-imported Axiom modules.

    :func:`~axiom.core.logging.get_logger` calls ``configure_logging()`` with
    its *default* level, and ``configure_logging`` sets the root ``axiom``
    logger's level before its idempotence guard. So every module imported after
    this script chose a level -- and Axiom imports most of its modules lazily,
    inside the first query -- silently resets the root logger back to INFO.

    The handler, by contrast, is installed once and never touched again, so a
    level set *there* survives. Filtering at the handler is therefore the only
    place this script can enforce its own verbosity without editing the frozen
    foundation. (The underlying ``get_logger`` behaviour is reported with this
    workstream's findings.)
    """
    logging.getLogger("axiom").setLevel(level.upper())
    for handler in logging.getLogger("axiom").handlers:
        handler.setLevel(level.upper())


def _git_sha(repo: Path) -> str | None:
    """Short sha of ``repo``'s HEAD, with ``-dirty`` appended when it is.

    Provenance, not a feature: Rules.md section 7 says a number without the
    commit that produced it is not a result, and that applies to a build time
    as much as to an NDCG score.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def _resolve_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if args.index_root is not None:
        overrides["index_root"] = args.index_root
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    return get_settings(args.profile, configs_dir=_REPO_ROOT / "configs", **overrides)


def _existing_version(settings: Settings, version_id: str) -> bool:
    try:
        return mf.manifest_path(settings, version_id).is_file()
    except AxiomError:  # pragma: no cover - a malformed version id
        return False


def _row(
    args: argparse.Namespace,
    settings: Settings,
    report: pipeline.IndexReport,
    started_at: float,
    peak_rss: int | None,
) -> dict[str, Any]:
    """The machine-readable summary. One flat object, one line."""
    manifest = report.manifest
    return {
        "schema_version": 1,
        "repo": str(args.repo),
        "git_sha": _git_sha(args.repo),
        "version_id": manifest.version_id,
        "parent_version": manifest.parent_version,
        "profile": settings.profile,
        "chunk_count": report.chunk_count,
        "file_count": report.file_count,
        # The rungs that actually ran. A duration without these is not a
        # measurement -- see this module's docstring.
        "embedding_model": manifest.embedding_model,
        "embedding_dim": manifest.embedding_dim,
        "index_kind": manifest.index_kind,
        "dense_backend": report.dense_backend,
        "sparse_backend": report.sparse_backend,
        "structural_skipped": report.structural_skipped,
        "degradations": list(report.degradations),
        "wall_clock_s": round(time.perf_counter() - started_at, 3),
        "build_ms": round(report.elapsed_ms, 3),
        "peak_rss_mb": None if peak_rss is None else round(peak_rss / (1024 * 1024), 1),
        "timings": report.ledger.as_dict(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "placeholders": settings.active_placeholders(),
    }


def _print_human(row: dict[str, Any]) -> None:
    timings = row["timings"]
    print(
        f"indexed {row['version_id']}: {row['chunk_count']} chunks from {row['file_count']} files"
    )
    print(f"  profile        {row['profile']}")
    print(f"  embedder       {row['embedding_model']} (dim {row['embedding_dim']})")
    print(f"  dense          {row['dense_backend']} / {row['index_kind']}")
    print(f"  sparse         {row['sparse_backend']}")
    print(f"  structural     {'skipped' if row['structural_skipped'] else 'built'}")
    if row["peak_rss_mb"] is not None:
        print(f"  peak RSS       {row['peak_rss_mb']} MB")
    print(f"  wall clock     {row['wall_clock_s']:.3f} s (build {row['build_ms'] / 1000:.3f} s)")
    for stage in timings.get("stages", []):
        print(f"    {stage['stage']:<12} {stage['elapsed_ms']:>9.1f} ms")
    for note in row["degradations"]:
        print(f"  ! {note}")


def main(argv: list[str] | None = None) -> int:
    started_at = time.perf_counter()
    args = build_parser().parse_args(argv)

    settings = _resolve_settings(args)
    configure_logging(settings.log_level, json_output=args.json)
    pin_log_level(settings.log_level)

    if not args.repo.is_dir():
        print(f"error: {args.repo} is not a directory", file=sys.stderr)
        return _EXIT_INPUT
    if not mf.is_valid_version_id(args.version):
        print(f"error: {args.version!r} is not a valid version id", file=sys.stderr)
        return _EXIT_USAGE
    if _existing_version(settings, args.version) and not args.force:
        print(
            f"error: version {args.version!r} already exists under "
            f"{mf.index_root(settings)}; pass --force to overwrite",
            file=sys.stderr,
        )
        return _EXIT_USAGE
    if args.parent is not None and not _existing_version(settings, args.parent):
        print(f"error: parent version {args.parent!r} is not built", file=sys.stderr)
        return _EXIT_INPUT

    ledger = TimingLedger()
    try:
        with _RssSampler(args.measure_rss) as sampler:
            report = pipeline.build_index_detailed(
                args.repo,
                args.version,
                settings,
                make_active=not args.no_activate,
                commit_sha=args.commit_sha,
                parent_version=args.parent,
                ledger=ledger,
            )
    except IndexNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_INPUT
    except AxiomError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_INTERNAL
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted; the index was published atomically or not at all", file=sys.stderr)
        return _EXIT_INTERNAL

    row = _row(args, settings, report, started_at, sampler.peak_bytes)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(row))
    elif args.time_it:
        _print_human(row)
    else:
        print(
            f"indexed {row['version_id']}: {row['chunk_count']} chunks "
            f"from {row['file_count']} files in {row['wall_clock_s']:.1f} s"
        )
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
