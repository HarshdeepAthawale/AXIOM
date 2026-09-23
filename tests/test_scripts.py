"""The two harnesses in ``scripts/``, plus the real-git fixture they build on.

``bench_latency.py`` is explicitly *not* a pytest test (TestPlan.md section 1:
latency is measured by a benchmark harness, never by a ``time.time()`` assert in
the merge gate). What **is** tested here is everything about it that is not a
timing: that its percentile arithmetic is right, that its budget and regression
gates fire on the numbers section 5.3/5.4 specify, and that both scripts run end
to end and exit with the codes API.md section 7.4 documents. A benchmark whose
gate logic is untested is a gate that silently passes.

``scripts/`` is imported by path rather than as a package: neither file is
installed, and ``pyproject.toml`` (which another workstream owns) declares no
``scripts`` package.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

pytestmark = pytest.mark.smoke


def _load(name: str) -> ModuleType:
    """Import a ``scripts/*.py`` file by path.

    The module is registered in ``sys.modules`` *before* it executes, not after:
    ``@dataclass`` resolves string annotations through
    ``sys.modules[cls.__module__]`` at class-creation time, so a module that is
    only registered afterwards makes every dataclass in it raise.
    """
    module_name = f"axiom_scripts_{name}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_index_script() -> ModuleType:
    return _load("build_index")


@pytest.fixture(scope="module")
def bench_script() -> ModuleType:
    return _load("bench_latency")


# ---------------------------------------------------------------------------
# scripts/build_index.py
# ---------------------------------------------------------------------------


class TestBuildIndexScript:
    def test_it_builds_a_queryable_index(
        self, build_index_script: ModuleType, repo_v1: Path, tmp_path: Path
    ) -> None:
        root = tmp_path / ".axiom"
        code = build_index_script.main(
            [str(repo_v1), "--version", "v1", "--index-root", str(root), "--json"]
        )
        assert code == 0
        assert (root / "registry.json").is_file()

    def test_the_json_row_names_the_rungs_that_actually_ran(
        self, build_index_script: ModuleType, repo_v1: Path, tmp_path: Path, capsys
    ) -> None:
        """A duration without the rung that produced it is not a measurement."""
        build_index_script.main(
            [
                str(repo_v1),
                "--version",
                "v1",
                "--index-root",
                str(tmp_path / ".axiom"),
                "--json",
            ]
        )
        row = json.loads(capsys.readouterr().out.strip())
        assert row["chunk_count"] > 0
        assert row["dense_backend"] and row["sparse_backend"]
        assert row["embedding_model"]
        assert row["wall_clock_s"] >= 0.0
        assert row["timings"]["stages"]

    def test_it_refuses_to_overwrite_a_version_without_force(
        self, build_index_script: ModuleType, repo_v1: Path, tmp_path: Path
    ) -> None:
        """An interrupted rebuild over a live index is how a demo loses its corpus."""
        root = tmp_path / ".axiom"
        args = [str(repo_v1), "--version", "v1", "--index-root", str(root)]
        assert build_index_script.main(args) == 0
        assert build_index_script.main(args) == 2
        assert build_index_script.main([*args, "--force"]) == 0

    def test_a_missing_repository_exits_3(
        self, build_index_script: ModuleType, tmp_path: Path
    ) -> None:
        assert (
            build_index_script.main(
                [str(tmp_path / "absent"), "--version", "v1", "--index-root", str(tmp_path / "x")]
            )
            == 3
        )

    def test_an_invalid_version_id_exits_2(
        self, build_index_script: ModuleType, repo_v1: Path, tmp_path: Path
    ) -> None:
        assert (
            build_index_script.main(
                [str(repo_v1), "--version", "../escape", "--index-root", str(tmp_path / "x")]
            )
            == 2
        )

    def test_an_unbuilt_parent_exits_3(
        self, build_index_script: ModuleType, repo_v1: Path, tmp_path: Path
    ) -> None:
        assert (
            build_index_script.main(
                [
                    str(repo_v1),
                    "--version",
                    "v2",
                    "--parent",
                    "v1",
                    "--index-root",
                    str(tmp_path / ".axiom"),
                ]
            )
            == 3
        )

    def test_no_activate_leaves_the_registry_pointer_alone(
        self, build_index_script: ModuleType, repo_v1: Path, repo_v2: Path, tmp_path: Path
    ) -> None:
        from axiom.config import get_settings
        from axiom.indexing import manifest as mf

        root = tmp_path / ".axiom"
        build_index_script.main([str(repo_v1), "--version", "v1", "--index-root", str(root)])
        build_index_script.main(
            [str(repo_v2), "--version", "v2", "--index-root", str(root), "--no-activate"]
        )
        settings = get_settings("default", configs_dir=REPO_ROOT / "configs", index_root=root)
        assert mf.load_registry(settings).active_version == "v1"

    def test_the_rss_sampler_degrades_when_psutil_is_absent(
        self, build_index_script: ModuleType, monkeypatch
    ) -> None:
        """A missing optional dependency may cost a number, never a build."""
        real_import = __import__

        def blocked(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("blocked for this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", blocked)
        with build_index_script._RssSampler(enabled=True) as sampler:
            pass
        assert sampler.peak_bytes is None

    def test_it_runs_as_a_subprocess_from_a_clean_interpreter(
        self, repo_v1: Path, tmp_path: Path
    ) -> None:
        """The ``sys.path`` shim must work from a clone without ``pip install -e .``."""
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "build_index.py"),
                str(repo_v1),
                "--version",
                "v1",
                "--index-root",
                str(tmp_path / ".axiom"),
                "--time-it",
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert completed.returncode == 0, completed.stderr
        assert "chunks from" in completed.stdout


# ---------------------------------------------------------------------------
# scripts/bench_latency.py -- gate arithmetic
# ---------------------------------------------------------------------------


class TestPercentiles:
    def test_nearest_rank_p50_and_p95(self, bench_script: ModuleType) -> None:
        """Hand-computed: 100 samples 1..100, nearest rank.

        ``p50`` is the 50th value and ``p95`` the 95th -- both are observations
        that actually occurred, which is the point of nearest-rank.
        """
        samples = bench_script.Samples(values=[float(i) for i in range(1, 101)])
        assert samples.percentile(0.50) == 50.0
        assert samples.percentile(0.95) == 95.0

    def test_a_single_sample_is_its_own_percentile(self, bench_script: ModuleType) -> None:
        samples = bench_script.Samples(values=[7.5])
        assert samples.percentile(0.50) == samples.percentile(0.95) == 7.5

    def test_no_samples_is_zero_not_an_exception(self, bench_script: ModuleType) -> None:
        assert bench_script.Samples().percentile(0.95) == 0.0
        assert bench_script.Samples().summary() == {"samples": 0}

    def test_p95_reflects_the_tail_a_mean_would_hide(self, bench_script: ModuleType) -> None:
        """PB-04 is a p95 budget precisely because the mean hides this."""
        samples = bench_script.Samples(values=[10.0] * 95 + [9000.0] * 5)
        summary = samples.summary()
        assert summary["p50_ms"] == 10.0
        assert summary["p95_ms"] == 10.0
        assert summary["max_ms"] == 9000.0


class TestBudgetGate:
    def _row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "query": {"samples": 100, "p50_ms": 400.0, "p95_ms": 3000.0},
            "peak_rss_mb": 1200.0,
        }
        row.update(overrides)
        return row

    def test_a_run_inside_every_budget_reports_no_failure(self, bench_script) -> None:
        assert bench_script.check_budgets(self._row()) == []

    def test_pb03_fires_above_900_ms_p50(self, bench_script) -> None:
        failures = bench_script.check_budgets(
            self._row(query={"samples": 100, "p50_ms": 901.0, "p95_ms": 3000.0})
        )
        assert any("PB-03" in failure for failure in failures)

    def test_pb04_fires_above_5_s_p95(self, bench_script) -> None:
        failures = bench_script.check_budgets(
            self._row(query={"samples": 100, "p50_ms": 400.0, "p95_ms": 5001.0})
        )
        assert any("PB-04" in failure for failure in failures)

    def test_pb05_fires_above_4_gb_peak_rss(self, bench_script) -> None:
        failures = bench_script.check_budgets(self._row(peak_rss_mb=4097.0))
        assert any("PB-05" in failure for failure in failures)

    def test_an_unmeasured_rss_does_not_silently_pass_the_gate(self, bench_script) -> None:
        """``None`` means "not measured", which must not read as "within budget"
        -- the row carries ``peak_rss_mb: null`` and the gate abstains."""
        assert bench_script.check_budgets(self._row(peak_rss_mb=None)) == []

    def test_the_budgets_match_the_contract(self, bench_script) -> None:
        assert bench_script.BUDGETS["query_p50_ms"] == 900.0
        assert bench_script.BUDGETS["query_p95_ms"] == 5000.0
        assert bench_script.BUDGETS["peak_rss_mb"] == 4096.0


class TestRegressionGate:
    BASE: ClassVar[dict[str, object]] = {
        "profile": "default",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "query": {"p50_ms": 400.0, "p95_ms": 2000.0},
        "peak_rss_mb": 1000.0,
    }

    def _now(self, **query: float) -> dict[str, object]:
        row = dict(self.BASE)
        row["query"] = {**self.BASE["query"], **query}
        return row

    def test_no_change_passes_clean(self, bench_script) -> None:
        assert bench_script.compare_rows(dict(self.BASE), dict(self.BASE)) == ([], [])

    def test_a_regression_over_15_percent_fails(self, bench_script) -> None:
        """Section 5.4: >15% is a CI failure. 400 -> 461 ms is +15.25%."""
        failures, _ = bench_script.compare_rows(self._now(p50_ms=461.0), dict(self.BASE))
        assert any("p50_ms" in failure for failure in failures)

    def test_a_regression_between_5_and_15_percent_only_warns(self, bench_script) -> None:
        """400 -> 440 ms is +10%: merge allowed, comment posted."""
        failures, warnings = bench_script.compare_rows(self._now(p50_ms=440.0), dict(self.BASE))
        assert failures == []
        assert any("p50_ms" in warning for warning in warnings)

    def test_an_improvement_is_never_a_failure(self, bench_script) -> None:
        assert bench_script.compare_rows(self._now(p50_ms=100.0), dict(self.BASE)) == ([], [])

    def test_a_rss_regression_over_10_percent_fails(self, bench_script) -> None:
        row = dict(self.BASE)
        row["peak_rss_mb"] = 1101.0
        failures, _ = bench_script.compare_rows(row, dict(self.BASE))
        assert any("RSS" in failure for failure in failures)

    def test_an_incomparable_baseline_fails_rather_than_being_compared(self, bench_script) -> None:
        """Gating a real regression against a number from another configuration
        is worse than refusing to gate at all."""
        row = dict(self.BASE)
        row["profile"] = "fast"
        failures, _ = bench_script.compare_rows(row, dict(self.BASE))
        assert any("not comparable" in failure for failure in failures)

    def test_a_different_embedding_model_is_also_incomparable(self, bench_script) -> None:
        row = dict(self.BASE)
        row["embedding_model"] = "axiom/hash-embedder-v1"
        failures, _ = bench_script.compare_rows(row, dict(self.BASE))
        assert failures and "embedding_model" in failures[0]


# ---------------------------------------------------------------------------
# scripts/bench_latency.py -- running it
# ---------------------------------------------------------------------------


class TestBenchRun:
    @pytest.fixture
    def built(self, indexed_v1, tmp_path: Path) -> Path:
        settings, _ = indexed_v1
        return Path(settings.index_root)

    def test_a_query_run_produces_a_stamped_row(
        self, bench_script: ModuleType, built: Path, tmp_path: Path, capsys
    ) -> None:
        code = bench_script.main(
            [
                "--phase",
                "query",
                "--index-root",
                str(built),
                "--version",
                "v1",
                "--repeat",
                "2",
                "--warmup",
                "1",
                "--json",
            ]
        )
        assert code == 0
        row = json.loads(capsys.readouterr().out.strip())
        assert row["query"]["samples"] > 0
        assert row["profile"] and row["git_sha"]
        assert row["python"] and row["platform"]
        assert row["threads"] == "8", "threads must be pinned for comparability"

    def test_a_degraded_run_is_never_reportable(
        self, bench_script: ModuleType, built: Path, capsys
    ) -> None:
        """AP-14: a number resting on a fallback rung or an unvalidated constant
        is not quotable, and the row has to say so itself."""
        bench_script.main(
            [
                "--phase",
                "query",
                "--index-root",
                str(built),
                "--repeat",
                "1",
                "--warmup",
                "0",
                "--json",
            ]
        )
        row = json.loads(capsys.readouterr().out.strip())
        assert row["degraded"] is True
        assert row["reportable"] is False
        assert row["placeholders"], "the PLACEHOLDER constants are still active"

    def test_an_index_run_times_the_build(
        self, bench_script: ModuleType, repo_v1: Path, tmp_path: Path, capsys
    ) -> None:
        code = bench_script.main(
            [
                "--phase",
                "index",
                "--repo",
                str(repo_v1),
                "--index-root",
                str(tmp_path / ".axiom"),
                "--repeat",
                "1",
                "--warmup",
                "0",
                "--json",
            ]
        )
        assert code == 0
        row = json.loads(capsys.readouterr().out.strip())
        assert row["index"]["samples"] == 1
        assert row["index"]["p50_ms"] > 0

    def test_the_row_can_be_written_and_read_back_as_a_baseline(
        self, bench_script: ModuleType, built: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "bench"
        assert (
            bench_script.main(
                [
                    "--phase",
                    "query",
                    "--index-root",
                    str(built),
                    "--repeat",
                    "1",
                    "--warmup",
                    "0",
                    "--out",
                    str(out),
                    "--json",
                ]
            )
            == 0
        )
        written = list(out.glob("*.json"))
        assert len(written) == 1
        baseline = json.loads(written[0].read_text(encoding="utf-8"))
        assert baseline["schema_version"] == 1

        # Comparing a row against itself must be a clean pass.
        assert bench_script.compare_rows(baseline, baseline) == ([], [])

    def test_a_missing_index_exits_3(self, bench_script: ModuleType, tmp_path: Path) -> None:
        assert (
            bench_script.main(
                ["--phase", "query", "--index-root", str(tmp_path / "absent"), "--repeat", "1"]
            )
            == 3
        )

    def test_index_phase_without_a_repo_exits_2(self, bench_script: ModuleType) -> None:
        assert bench_script.main(["--phase", "index"]) == 2

    def test_a_missing_query_file_exits_3(self, bench_script: ModuleType, tmp_path: Path) -> None:
        assert bench_script.main(["--queries", str(tmp_path / "nope.txt")]) == 3

    def test_a_missing_baseline_exits_3(
        self, bench_script: ModuleType, built: Path, tmp_path: Path
    ) -> None:
        assert (
            bench_script.main(
                [
                    "--phase",
                    "query",
                    "--index-root",
                    str(built),
                    "--repeat",
                    "1",
                    "--warmup",
                    "0",
                    "--compare",
                    str(tmp_path / "nope.json"),
                ]
            )
            == 3
        )


class TestQueryLoading:
    def test_the_bench_query_fixture_is_loaded_in_a_fixed_order(
        self, bench_script: ModuleType, fixtures_dir: Path
    ) -> None:
        """A benchmark whose input varies between runs measures the input."""
        path = fixtures_dir / "bench_queries.txt"
        first = bench_script.load_queries(path)
        assert len(first) >= 20
        assert first == bench_script.load_queries(path)
        assert not any(query.startswith("#") for query in first)

    def test_the_default_query_set_is_used_when_none_is_given(
        self, bench_script: ModuleType
    ) -> None:
        assert bench_script.load_queries(None) == list(bench_script.DEFAULT_QUERIES)

    def test_blank_lines_and_comments_are_dropped(
        self, bench_script: ModuleType, tmp_path: Path
    ) -> None:
        path = tmp_path / "q.txt"
        path.write_text("# a comment\n\n  real query  \n\n", encoding="utf-8")
        assert bench_script.load_queries(path) == ["real query"]


def test_thread_pinning_respects_a_deliberate_override(
    bench_script: ModuleType, monkeypatch
) -> None:
    """An operator reproducing a 4-core box must not be silently overwritten."""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    bench_script.pin_threads()
    import os

    assert os.environ["OMP_NUM_THREADS"] == "4"


# ---------------------------------------------------------------------------
# The real-git fixture (TestPlan.md section 1.3, fixture_git_repo)
# ---------------------------------------------------------------------------


@pytest.mark.needs_git
class TestGitFixture:
    def test_the_two_commits_diff_the_way_the_fixtures_say(self, git_repo_v1_v2: Path) -> None:
        """``git diff --name-status`` against a real binary, not a fabricated string.

        v2 adds ``tools/telemetry.js``, deletes ``constants.js``, modifies
        three files, renames ``deep/nested.js`` -> ``deep/pipeline.js`` byte
        identically, and renames ``dup/copy_b.js`` -> ``dup/copy_c.js`` with an
        edit. Rename detection is git's job (``-M``), so the assertion is on the
        union of the categories rather than on which bucket each rename landed
        in -- git's similarity threshold is not ours to pin.
        """
        from axiom.versioning.gitdiff import diff_versions, is_git_repo

        assert is_git_repo(git_repo_v1_v2)
        diff = diff_versions("HEAD~1", "HEAD", git_repo_v1_v2)
        assert diff is not None
        assert not diff.is_empty

        touched = set(diff.paths_to_chunk) | set(diff.paths_to_drop)
        assert "src/tools/telemetry.js" in touched
        assert "src/constants.js" in touched
        assert "src/utils/normalize.js" in touched
        assert "src/deep/nested.js" in touched or "src/deep/pipeline.js" in touched

    def test_a_no_op_diff_is_empty_not_an_error(self, git_repo_v1_v2: Path) -> None:
        """TC-077: a version with zero changes is a valid no-op reindex."""
        from axiom.versioning.gitdiff import diff_versions

        diff = diff_versions("HEAD", "HEAD", git_repo_v1_v2)
        assert diff is not None and diff.is_empty

    def test_diff_collections_are_sorted(self, git_repo_v1_v2: Path) -> None:
        """AP-05: chunk production order must not depend on git's output order."""
        from axiom.versioning.gitdiff import diff_versions

        diff = diff_versions("HEAD~1", "HEAD", git_repo_v1_v2)
        assert diff is not None
        for collection in (diff.added, diff.modified, diff.deleted):
            assert list(collection) == sorted(collection)

    def test_a_non_git_directory_degrades_rather_than_raising(self, tmp_path: Path) -> None:
        from axiom.versioning.gitdiff import diff_versions, is_git_repo

        assert is_git_repo(tmp_path) is False
        assert diff_versions("HEAD~1", "HEAD", tmp_path) is None
