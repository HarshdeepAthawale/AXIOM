#!/usr/bin/env python3
"""The only sanctioned performance measurement (TestPlan.md section 5.1).

Usage::

    python scripts/bench_latency.py --phase query \\
        --version v1 --queries tests/fixtures/bench_queries.txt \\
        --repeat 5 --warmup 3 --profile default --out data/bench/

    python scripts/bench_latency.py --phase index --repo data/demo_repo --repeat 3
    python scripts/bench_latency.py --compare data/bench/<baseline_sha>.json

It is deliberately **not** a pytest test. TestPlan.md section 1 puts latency in
the "measured by a benchmark harness, never by a ``time.time()`` assert inside a
unit test" column, for a reason worth restating: a timing assertion inside the
merge gate is a test that fails on a loaded CI runner and passes on a quiet one,
which trains everyone to rerun it rather than read it.

What this harness does that a stopwatch does not:

* **Discards warm-up iterations.** Model load, mmap page-in and OS cache are
  real costs, but they are paid once, and folding them into a p50 measures the
  first query rather than the steady state. ``--warmup 3`` per TestPlan 5.1.
* **Reports p50 and p95 over all (query, repeat) samples**, not a mean of
  per-query means -- a mean hides the tail, and PB-04 is a p95 budget.
* **Samples RSS every 100 ms on a background thread** (NFR-05 / PB-05), when
  ``psutil`` is importable. Peak, not final: the 4 GB ceiling is about the
  transient during reranking, which a post-hoc reading misses entirely.
* **Stamps what produced the number.** git sha, profile, resolved model ids,
  the *rungs that actually ran*, chunk count, index kind, python, platform,
  thread pinning. A p50 from a hash-embedder run and a p50 from a real ONNX run
  differ by orders of magnitude; a row that does not say which is not a result
  (Rules.md section 7).
* **Refuses to call a degraded run a budget result.** ``reportable`` is false
  whenever a ladder fell back, whenever a placeholder constant is still active
  (AP-14), or whenever the tree is dirty.

``--compare`` implements the section 5.4 regression gate: non-zero exit when a
budget is exceeded or p50/p95 regressed more than 15% against a baseline row.

Exit codes: 0 success, 2 usage/config, 3 missing index or query file, 4 a gate
failed under ``--compare``, 1 anything unexpected.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:  # running from a clone without `pip install -e .`
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from axiom import pipeline  # noqa: E402
from axiom.config import Settings, get_settings  # noqa: E402
from axiom.core.errors import AxiomError, IndexNotFoundError  # noqa: E402
from axiom.core.logging import configure_logging, get_logger  # noqa: E402
from axiom.indexing import manifest as mf  # noqa: E402

_EXIT_OK = 0
_EXIT_INTERNAL = 1
_EXIT_USAGE = 2
_EXIT_INPUT = 3
_EXIT_GATE = 4

#: RSS sampling period, in seconds (TestPlan.md section 5.1: "every 100 ms").
RSS_SAMPLE_INTERVAL_S = 0.1

#: Intra-op thread pinning, so two runs on the same box are comparable
#: (TestPlan.md section 5.1). Applied before any heavy import, which is why
#: this module sets the environment rather than a Settings field: ONNX Runtime
#: and OpenMP both read these at load time, not at session creation.
PINNED_THREADS = 8

#: _CONTRACT.md section 7 budgets, in the units this harness measures.
BUDGETS: dict[str, float] = {
    "query_p50_ms": 900.0,  # PB-03
    "query_p95_ms": 5000.0,  # PB-04
    "peak_rss_mb": 4096.0,  # PB-05
}

#: Section 5.4's regression thresholds, as fractions.
FAIL_REGRESSION = 0.15
WARN_REGRESSION = 0.05
RSS_FAIL_REGRESSION = 0.10

#: Fallback query set when ``--queries`` is not given. Deliberately small and
#: fixed in order: a benchmark whose input changes between runs measures the
#: input, not the code.
DEFAULT_QUERIES: tuple[str, ...] = (
    "How is the input preprocessed before going to the main function?",
    "Which files call preprocessInput before resolveTool?",
    "Where is the Bluetooth-settings deeplink used?",
    "How does the tool registry resolve a tool by name?",
    "what validates a payload before dispatch",
)

_LOGGER = get_logger("scripts.bench_latency")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class _RssSampler:
    """Background peak-RSS sampler; a no-op when ``psutil`` is absent.

    Degrading to ``None`` rather than refusing to run is deliberate: a missing
    optional dependency may cost a *number*, never the benchmark. The row then
    says ``peak_rss_mb: null`` and ``--compare`` skips the RSS gate instead of
    silently passing it.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.peak_bytes: int | None = None
        self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any = None
        if not enabled:
            return
        try:
            import psutil
        except ImportError:
            _LOGGER.warning("psutil not installed; peak RSS will not be reported (PB-05)")
            return
        self._process = psutil.Process()
        self.available = True

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


@dataclass
class Samples:
    """Measured durations in milliseconds, plus what the runs produced."""

    values: list[float] = field(default_factory=list)
    results: list[int] = field(default_factory=list)
    passes: list[int] = field(default_factory=list)
    errors: int = 0

    def percentile(self, fraction: float) -> float:
        """Nearest-rank percentile over *all* samples.

        Nearest-rank rather than an interpolating estimator so that a p95 over
        20 samples is an observation that actually happened, not a number
        between two of them. With the 500 samples TestPlan 5.1 prescribes
        (100 queries x 5 repeats) the difference is immaterial; with a dev
        machine's 25 it is the difference between a real tail and a fiction.
        """
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
        return ordered[rank - 1]

    def summary(self) -> dict[str, Any]:
        if not self.values:
            return {"samples": 0}
        return {
            "samples": len(self.values),
            "p50_ms": round(self.percentile(0.50), 3),
            "p95_ms": round(self.percentile(0.95), 3),
            "min_ms": round(min(self.values), 3),
            "max_ms": round(max(self.values), 3),
            "mean_ms": round(statistics.fmean(self.values), 3),
            "errors": self.errors,
            "mean_results": round(statistics.fmean(self.results), 2) if self.results else 0.0,
            "mean_passes": round(statistics.fmean(self.passes), 2) if self.passes else 0.0,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_latency.py",
        description="Measure Axiom index-build and query latency against the §7 budgets.",
    )
    parser.add_argument(
        "--phase",
        choices=("index", "query", "both"),
        default="query",
        help="What to measure. 'index' needs --repo; 'query' needs a built index.",
    )
    parser.add_argument("--repo", type=Path, default=None, help="Repo root for --phase index.")
    parser.add_argument("--index-root", type=Path, default=None, help="Override .axiom location.")
    parser.add_argument("--version", default=None, help="Version to query (default: active).")
    parser.add_argument("--profile", default=None, help="Config profile.")
    parser.add_argument("--queries", type=Path, default=None, help="One query per line.")
    parser.add_argument("--repeat", type=int, default=5, help="Measured repeats per query.")
    parser.add_argument("--warmup", type=int, default=3, help="Discarded warm-up iterations.")
    parser.add_argument("--top-k", type=int, default=None, help="Final result count.")
    parser.add_argument(
        "--no-agent", action="store_true", help="Single pass, for the PB-03 measurement."
    )
    parser.add_argument(
        "--agent", action="store_true", help="Force the agent loop on, for the PB-04 measurement."
    )
    parser.add_argument(
        "--force-passes", type=int, default=None, help="Pin agent_max_passes (PB-04 uses 2)."
    )
    parser.add_argument(
        "--measure-rss", action="store_true", default=True, help="Sample peak RSS (default on)."
    )
    parser.add_argument(
        "--no-measure-rss", dest="measure_rss", action="store_false", help="Skip RSS sampling."
    )
    parser.add_argument(
        "--no-pin-threads",
        dest="pin_threads",
        action="store_false",
        default=True,
        help="Leave OMP/ONNX thread counts alone (results become incomparable).",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Directory or file for the JSON row."
    )
    parser.add_argument("--compare", type=Path, default=None, help="Baseline row to gate against.")
    parser.add_argument("--json", action="store_true", help="Emit the row on stdout as JSON.")
    parser.add_argument("--log-level", default="WARNING", help="Axiom log level for the run.")
    return parser


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


def pin_threads(count: int = PINNED_THREADS) -> None:
    """Pin intra-op thread counts so two runs on one box are comparable.

    ``setdefault`` rather than assignment: an operator who exported
    ``OMP_NUM_THREADS`` deliberately -- to reproduce a 4-core box, say -- must
    not have it silently overwritten by the harness that is supposed to be
    recording their configuration.
    """
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, str(count))


def load_queries(path: Path | None) -> list[str]:
    """Read the fixed-order query set. Comments and blank lines are ignored."""
    if path is None:
        return list(DEFAULT_QUERIES)
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return lines


def _git_sha() -> str:
    """Short sha of the *Axiom* tree, ``-dirty`` when it is not clean.

    A dirty tree makes the row non-reportable: the code that produced the
    number cannot be checked out again (TestPlan.md section 6.4).
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _resolve_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {"log_level": args.log_level}
    if args.index_root is not None:
        overrides["index_root"] = args.index_root
    if args.no_agent:
        overrides["agent_enabled"] = False
    if args.agent:
        overrides["agent_enabled"] = True
    if args.force_passes is not None:
        overrides["agent_enabled"] = True
        overrides["agent_max_passes"] = args.force_passes
    if args.pin_threads:
        overrides["num_threads"] = PINNED_THREADS
    return get_settings(args.profile, configs_dir=_REPO_ROOT / "configs", **overrides)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_query(
    queries: list[str],
    settings: Settings,
    *,
    version_id: str | None,
    top_k: int | None,
    warmup: int,
    repeat: int,
) -> Samples:
    """Time ``repeat`` passes over the query set, after ``warmup`` discarded ones.

    The loop is query-major within each repeat so that consecutive samples of
    the *same* query are separated by the whole set -- otherwise repeat 2 of
    query 1 reads a cache that repeat 1 of query 1 just warmed, and the
    measurement is of the cache.

    :class:`~axiom.core.errors.IndexNotFoundError` is deliberately **not**
    caught: it is a setup failure, not a per-query one, and swallowing it would
    produce a row of zero samples that exits 0 -- a benchmark reporting success
    for having measured nothing. Any other ``AxiomError`` is one query's
    problem and is counted in ``errors`` instead.
    """
    samples = Samples()

    for _ in range(max(0, warmup)):
        for query in queries:
            try:
                pipeline.query(query, settings, version_id=version_id, top_k=top_k)
            except IndexNotFoundError:
                raise
            except AxiomError:
                pass

    for _ in range(max(1, repeat)):
        for query in queries:
            started = time.perf_counter()
            try:
                response = pipeline.query(query, settings, version_id=version_id, top_k=top_k)
            except IndexNotFoundError:
                raise
            except AxiomError as exc:
                samples.errors += 1
                _LOGGER.warning("query failed during benchmark: %s", exc)
                continue
            samples.values.append((time.perf_counter() - started) * 1000.0)
            samples.results.append(len(response.results))
            samples.passes.append(response.passes_used)
    return samples


def measure_index(
    repo: Path, settings: Settings, *, warmup: int, repeat: int
) -> tuple[Samples, pipeline.IndexReport | None]:
    """Time ``repeat`` cold builds (PB-01), each into its own version id.

    Each build gets a distinct version so no run is measuring an overwrite, and
    none is made active -- a benchmark must not repoint the registry a demo is
    about to use.
    """
    samples = Samples()
    last: pipeline.IndexReport | None = None
    for iteration in range(max(0, warmup) + max(1, repeat)):
        version_id = f"bench-{iteration:03d}"
        started = time.perf_counter()
        report = pipeline.build_index_detailed(repo, version_id, settings, make_active=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if iteration >= max(0, warmup):
            samples.values.append(elapsed_ms)
            samples.results.append(report.chunk_count)
            samples.passes.append(report.file_count)
        last = report
    return samples, last


def _index_facts(settings: Settings, version_id: str | None) -> dict[str, Any]:
    """Which index the numbers describe. Absent index degrades to empty facts."""
    try:
        resolved = mf.resolve_version(settings, version_id)
        manifest = mf.read_manifest(settings, resolved)
    except (AxiomError, OSError):
        return {}
    return {
        "version_id": manifest.version_id,
        "chunk_count": manifest.chunk_count,
        "index_kind": manifest.index_kind,
        "embedding_model": manifest.embedding_model,
        "embedding_dim": manifest.embedding_dim,
    }


def build_row(
    args: argparse.Namespace,
    settings: Settings,
    *,
    query_samples: Samples | None,
    index_samples: Samples | None,
    index_report: pipeline.IndexReport | None,
    peak_rss: int | None,
    degraded: bool,
) -> dict[str, Any]:
    sha = _git_sha()
    placeholders = settings.active_placeholders()
    row: dict[str, Any] = {
        "schema_version": 1,
        "git_sha": sha,
        "profile": settings.profile,
        "phase": args.phase,
        "agent_enabled": settings.agent_enabled,
        "agent_max_passes": settings.agent_max_passes,
        "reranker_enabled": settings.reranker_enabled,
        "llm_enabled": settings.llm_enabled,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "peak_rss_mb": None if peak_rss is None else round(peak_rss / (1024 * 1024), 1),
        "placeholders": placeholders,
        # A number is quotable only when the code is checked-out-able, no
        # ladder fell back, and no placeholder constant is load-bearing (AP-14).
        "reportable": not degraded and not sha.endswith("-dirty") and not placeholders,
        "degraded": degraded,
    }
    row.update(_index_facts(settings, args.version))
    if query_samples is not None:
        row["query"] = query_samples.summary()
    if index_samples is not None:
        row["index"] = index_samples.summary()
    if index_report is not None:
        row["dense_backend"] = index_report.dense_backend
        row["sparse_backend"] = index_report.sparse_backend
        row["structural_skipped"] = index_report.structural_skipped
    return row


# ---------------------------------------------------------------------------
# Reporting and the regression gate
# ---------------------------------------------------------------------------


def check_budgets(row: dict[str, Any]) -> list[str]:
    """_CONTRACT.md section 7 budgets. Returns the failures, empty when clean."""
    failures: list[str] = []
    query = row.get("query") or {}
    if query.get("samples"):
        if query["p50_ms"] > BUDGETS["query_p50_ms"]:
            failures.append(
                f"PB-03: query p50 {query['p50_ms']:.0f} ms > {BUDGETS['query_p50_ms']:.0f} ms"
            )
        if query["p95_ms"] > BUDGETS["query_p95_ms"]:
            failures.append(
                f"PB-04: query p95 {query['p95_ms']:.0f} ms > {BUDGETS['query_p95_ms']:.0f} ms"
            )
    peak = row.get("peak_rss_mb")
    if peak is not None and peak > BUDGETS["peak_rss_mb"]:
        failures.append(f"PB-05: peak RSS {peak:.0f} MB > {BUDGETS['peak_rss_mb']:.0f} MB")
    return failures


def compare_rows(current: dict[str, Any], baseline: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Section 5.4's gate: ``(failures, warnings)``.

    A regression is only meaningful between comparable runs, so a baseline from
    a different profile or a different embedding model is reported as a failure
    of *comparability* rather than silently compared -- the alternative is
    gating a real regression against a number from another configuration.
    """
    failures: list[str] = []
    warnings: list[str] = []

    for key in ("profile", "embedding_model"):
        if key in baseline and key in current and baseline[key] != current[key]:
            failures.append(f"not comparable: {key} was {baseline[key]!r}, is now {current[key]!r}")
    if failures:
        return failures, warnings

    for metric in ("p50_ms", "p95_ms"):
        was = (baseline.get("query") or {}).get(metric)
        now = (current.get("query") or {}).get(metric)
        if not was or now is None:
            continue
        delta = (now - was) / was
        if delta > FAIL_REGRESSION:
            failures.append(f"{metric} regressed {delta:+.1%} ({was:.0f} -> {now:.0f} ms)")
        elif delta > WARN_REGRESSION:
            warnings.append(f"{metric} regressed {delta:+.1%} ({was:.0f} -> {now:.0f} ms)")

    was_rss, now_rss = baseline.get("peak_rss_mb"), current.get("peak_rss_mb")
    if was_rss and now_rss is not None:
        delta = (now_rss - was_rss) / was_rss
        if delta > RSS_FAIL_REGRESSION:
            failures.append(f"peak RSS regressed {delta:+.1%} ({was_rss:.0f} -> {now_rss:.0f} MB)")

    was_build = (baseline.get("index") or {}).get("p50_ms")
    now_build = (current.get("index") or {}).get("p50_ms")
    if was_build and now_build is not None and (now_build - was_build) / was_build > 0.20:
        warnings.append(f"index build regressed {(now_build - was_build) / was_build:+.1%}")

    return failures, warnings


def print_table(row: dict[str, Any]) -> None:
    """The human table. Every line that carries a number also carries its context."""
    print(f"axiom bench  sha={row['git_sha']}  profile={row['profile']}  phase={row['phase']}")
    print(
        f"  index        version={row.get('version_id', '-')} chunks={row.get('chunk_count', '-')} "
        f"kind={row.get('index_kind', '-')}"
    )
    print(
        f"  models       embed={row.get('embedding_model', '-')} "
        f"rerank={'on' if row['reranker_enabled'] else 'off'} "
        f"llm={'on' if row['llm_enabled'] else 'off'}"
    )
    print(
        f"  agent        enabled={row['agent_enabled']} max_passes={row['agent_max_passes']} "
        f"threads={row['threads']}"
    )
    for phase in ("query", "index"):
        block = row.get(phase)
        if not block or not block.get("samples"):
            continue
        print(
            f"  {phase:<12} n={block['samples']:<4} "
            f"p50={block['p50_ms']:>8.1f} ms  p95={block['p95_ms']:>8.1f} ms  "
            f"min={block['min_ms']:>8.1f}  max={block['max_ms']:>8.1f}"
        )
        if phase == "query":
            print(
                f"               mean_results={block['mean_results']} "
                f"mean_passes={block['mean_passes']} errors={block['errors']}"
            )
    if row["peak_rss_mb"] is not None:
        print(f"  peak RSS     {row['peak_rss_mb']:.1f} MB")
    else:
        print("  peak RSS     not measured (psutil absent)")
    print(f"  reportable   {row['reportable']}", end="")
    if not row["reportable"]:
        reasons = []
        if row["degraded"]:
            reasons.append("a degradation ladder fell back")
        if row["git_sha"].endswith("-dirty"):
            reasons.append("dirty tree")
        if row["placeholders"]:
            reasons.append(f"{len(row['placeholders'])} placeholder constants active")
        print(f"  ({'; '.join(reasons)})")
    else:
        print()


def _write_row(out: Path, row: dict[str, Any]) -> Path:
    target = out / f"{row['git_sha']}.json" if out.is_dir() or not out.suffix else out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pin_threads:
        pin_threads()

    settings = _resolve_settings(args)
    configure_logging(settings.log_level, json_output=args.json)
    pin_log_level(settings.log_level)

    if args.phase in ("index", "both") and args.repo is None:
        print("error: --phase index requires --repo", file=sys.stderr)
        return _EXIT_USAGE
    if args.repo is not None and not args.repo.is_dir():
        print(f"error: {args.repo} is not a directory", file=sys.stderr)
        return _EXIT_INPUT
    if args.queries is not None and not args.queries.is_file():
        print(f"error: query file {args.queries} not found", file=sys.stderr)
        return _EXIT_INPUT

    queries = load_queries(args.queries)
    if args.phase in ("query", "both") and not queries:
        print("error: no queries to run", file=sys.stderr)
        return _EXIT_INPUT

    query_samples: Samples | None = None
    index_samples: Samples | None = None
    index_report: pipeline.IndexReport | None = None

    try:
        with _RssSampler(args.measure_rss) as sampler:
            if args.phase in ("index", "both"):
                index_samples, index_report = measure_index(
                    args.repo, settings, warmup=args.warmup, repeat=args.repeat
                )
            if args.phase in ("query", "both"):
                query_samples = measure_query(
                    queries,
                    settings,
                    version_id=args.version,
                    top_k=args.top_k,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
    except IndexNotFoundError as exc:
        print(f"error: {exc}; run scripts/build_index.py first", file=sys.stderr)
        return _EXIT_INPUT
    except AxiomError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_INTERNAL

    if query_samples is not None and not query_samples.values:
        # Every attempt failed. A row with no samples is not a measurement, and
        # exiting 0 on one would let a broken benchmark pass a CI gate.
        print("error: no query completed; nothing was measured", file=sys.stderr)
        return _EXIT_INPUT

    facts = _index_facts(settings, args.version)
    degraded = bool(
        (index_report is not None and (index_report.dense_backend != "faiss"))
        or facts.get("embedding_model", "").startswith("axiom/hash-embedder")
    )
    row = build_row(
        args,
        settings,
        query_samples=query_samples,
        index_samples=index_samples,
        index_report=index_report,
        peak_rss=sampler.peak_bytes,
        degraded=degraded,
    )

    if args.out is not None:
        written = _write_row(args.out, row)
        _LOGGER.info("wrote %s", written)

    if args.json:
        print(json.dumps(row))
    else:
        print_table(row)

    exit_code = _EXIT_OK
    budget_failures = check_budgets(row)
    for failure in budget_failures:
        print(f"BUDGET FAIL  {failure}", file=sys.stderr)

    if args.compare is not None:
        if not args.compare.is_file():
            print(f"error: baseline {args.compare} not found", file=sys.stderr)
            return _EXIT_INPUT
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        failures, warnings = compare_rows(row, baseline)
        for warning in warnings:
            print(f"WARN         {warning}", file=sys.stderr)
        for failure in failures:
            print(f"REGRESSION   {failure}", file=sys.stderr)
        if failures:
            exit_code = _EXIT_GATE

    if budget_failures and row["reportable"]:
        # A degraded run's numbers describe a fallback rung, not the system the
        # budgets were written for, so they do not gate.
        exit_code = _EXIT_GATE
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
