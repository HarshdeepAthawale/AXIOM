#!/usr/bin/env python3
"""Produce ``appsretrieval_results.json`` -- the artifact the screening gate reads.

Usage::

    python scripts/run_eval.py --task AppsRetrieval --split test
    python scripts/run_eval.py --task AppsRetrieval --limit 200        # smoke only
    python scripts/run_eval.py --corpus-path data/datasets/AppsRetrieval

Three guarantees this script enforces, none of which the metric arithmetic can
enforce for itself:

**It only runs under ``configs/eval.yaml``.** PRD.md section 2.2 states plainly
that ``appsretrieval_results.json`` is only ever generated under the eval profile.
The demo profile runs the structural signal, which APPS has nothing for, and its
fusion weights are tuned for a different corpus -- a number produced under it
would be a number for a task nobody is scoring. Any other profile exits 2.

**It stamps what produced the number.** Active profile, a hash of the fully
resolved config, the resolved model identities, the backend that actually ran,
every degradation taken, and the git SHA. Rules.md section 7 puts it as: a
results file without its resolved config is not a result.

**It refuses to call a placeholder-backed number reportable.** Rules.md AP-14
forbids quoting a score that rests on an unvalidated constant.
``Settings.active_placeholders()`` is read at run time, printed as a banner, and
written into the JSON, so a reader of the file can see -- without reading any
code -- whether the score is defensible yet. A smoke run (``--limit``) and a
degraded backend are non-reportable for the same reason (TestPlan.md section 6.2).

Exit codes follow API.md section 7.4: 0 success, 2 usage/config, 3 missing index
or dataset, 1 anything unexpected.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:  # running from a clone without `pip install -e .`
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from axiom.config import Settings, get_settings  # noqa: E402
from axiom.core.errors import AxiomContractError, AxiomError, IndexNotFoundError  # noqa: E402
from axiom.core.hashing import blake2b_128  # noqa: E402
from axiom.core.logging import configure_logging, get_logger  # noqa: E402
from axiom.core.timing import TimingLedger  # noqa: E402
from axiom.eval import metrics  # noqa: E402
from axiom.eval.mteb_adapter import build_search_model, resolve_task  # noqa: E402

#: The only profile under which a results file may be written (PRD.md section 2.2).
REQUIRED_PROFILE = "eval"

#: Cut-offs written into the results file. 10 and 100 are the reported ones
#: (PRD.md section 2: NDCG@10, MRR@10, Recall@100); the rest are context.
REPORT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10, 100)

#: Schema generation of the ``axiom_provenance`` block, so a later reader can
#: tell which fields to expect (Schema.md section 16).
PROVENANCE_SCHEMA_VERSION = 1

_EXIT_OK = 0
_EXIT_INTERNAL = 1
_EXIT_USAGE = 2
_EXIT_INDEX = 3

_LOGGER = get_logger("eval.run")


def build_parser() -> argparse.ArgumentParser:
    """CLI surface. Mirrors ``axiom eval``'s flags in API.md section 7.3."""
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Run the Axiom pipeline over a CoIR retrieval task and write "
        "an MTEB TaskResult-shaped JSON.",
    )
    parser.add_argument("--task", default="AppsRetrieval", help="MTEB task name.")
    parser.add_argument(
        "--split", default="test", help="Dataset split. 'test' is the only reportable one."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate to N queries for a smoke run. Marks the result NON-REPORTABLE: "
        "a limited run is a different task, not a cheap estimate (TestPlan.md 6.2).",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Vendored BEIR-shaped dataset directory (corpus.jsonl, queries.jsonl, "
        "qrels/<split>.tsv). Preferred over the Hub; this is assumption A-1's mitigation.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Retrieval backend as 'module.path:factory', called with Settings and "
        "returning a SearchBackend. Omitted means the zero-dependency lexical fallback, "
        "which is never reportable.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Default: <task lowercased>_results.json in the working directory.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional path for the raw run in MTEB's prediction format: "
        '{"<split>": {query_id: {doc_id: score}}}.',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Candidate depth per query. Defaults to max(100, dense_top_k) so Recall@100 "
        "is measurable.",
    )
    parser.add_argument(
        "--profile",
        default=REQUIRED_PROFILE,
        help=f"Config profile. Only {REQUIRED_PROFILE!r} is accepted.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="Directory holding the profile YAMLs. Defaults to <repo>/configs.",
    )
    parser.add_argument("--json", action="store_true", help="Echo the results JSON to stdout.")
    return parser


def _git_sha(repo_root: Path) -> str | None:
    """Short git SHA of the working tree, suffixed ``-dirty`` when it is.

    TestPlan.md section 6.4 requires every experiment-log row to carry the SHA,
    and a dirty tree makes the run non-reproducible and therefore non-reportable.
    Returns ``None`` outside a git checkout rather than failing the run.
    """

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    sha = run("rev-parse", "--short", "HEAD")
    if sha is None:
        return None
    status = run("status", "--porcelain")
    return f"{sha}-dirty" if status else sha


def _config_hash(settings: Settings) -> str:
    """blake2b-128 over the fully resolved config.

    Canonical JSON with sorted keys, so two runs with the same effective settings
    hash identically regardless of how those settings were reached -- CLI flag,
    env var, or YAML (Rules.md section 7's precedence chain collapses here).
    """
    payload = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
    return blake2b_128(payload.encode("utf-8"))


def _resolve_settings(args: argparse.Namespace) -> Settings:
    """Load settings for the requested profile, or exit 2 explaining why not."""
    configs_dir = args.configs_dir or (_REPO_ROOT / "configs")
    profile_path = Path(configs_dir) / f"{args.profile}.yaml"
    if args.profile != REQUIRED_PROFILE:
        _fail(
            _EXIT_USAGE,
            f"refusing to run under profile {args.profile!r}. "
            f"{args.task} results are only ever generated under configs/"
            f"{REQUIRED_PROFILE}.yaml (PRD.md 2.2): the demo profile enables the "
            f"structural signal, which APPS has no cross-file structure to feed.",
        )
    if not profile_path.is_file():
        _fail(_EXIT_USAGE, f"profile file {profile_path} not found")

    settings = get_settings(profile=args.profile, configs_dir=Path(configs_dir))
    if settings.profile != REQUIRED_PROFILE:
        _fail(
            _EXIT_USAGE,
            f"resolved profile is {settings.profile!r}, not {REQUIRED_PROFILE!r}; "
            f"check AXIOM_PROFILE in the environment",
        )
    if settings.structural_enabled:
        _fail(
            _EXIT_USAGE,
            "structural_enabled is true under the eval profile. APPS documents are "
            "standalone files with no call graph; running the structural signal here "
            "injects a near-random third list into fusion (PRD.md 2.2).",
        )
    return settings


def _fail(code: int, message: str) -> None:
    """Print a remediation-shaped error and exit. Never prints a traceback (API.md 7.4)."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _score_block(
    scores: dict[str, float],
    *,
    hf_subset: str,
    languages: tuple[str, ...],
    main_metric: str,
) -> dict[str, Any]:
    """One entry of an MTEB ``TaskResult.scores[split]`` list.

    Fractions, not percentages: MTEB stores ``0.214`` and the leaderboard renders
    ``21.4``. Mixing the two is how a submission ends up claiming a 0.2 NDCG.
    """
    block: dict[str, Any] = {
        "hf_subset": hf_subset,
        "languages": list(languages),
        "main_score": scores.get(main_metric, 0.0),
    }
    block.update({key: value for key, value in sorted(scores.items())})
    return block


def _mteb_version() -> str | None:
    """Installed MTEB version, or ``None``. Never imported at module scope (NFR-07)."""
    try:
        import mteb
    except ImportError:
        return None
    return str(getattr(mteb, "__version__", "unknown"))


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and write the results file. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging()

    settings = _resolve_settings(args)
    ledger = TimingLedger()
    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()

    placeholders = settings.active_placeholders()
    if placeholders:
        print(
            "PLACEHOLDER WARNING (Rules.md AP-14): "
            f"{len(placeholders)} unvalidated constants are active -- "
            f"{', '.join(placeholders)}. This run is NOT reportable until each is "
            "measured and logged in Tracker.md.",
            file=sys.stderr,
        )

    try:
        with ledger.measure("dataset") as detail:
            task = resolve_task(
                args.task,
                args.split,
                local_path=args.corpus_path,
                settings=settings,
            )
            task = task.limited(args.limit)
            detail["documents"] = len(task.corpus)
            detail["queries"] = len(task.queries)
            detail["source"] = task.source

        model = build_search_model(settings, backend_spec=args.backend)
        top_k = args.top_k if args.top_k else max(100, settings.dense_top_k)

        with ledger.measure("index") as detail:
            model.index(
                task.corpus,
                hf_split=task.split,
                hf_subset=task.hf_subset,
                encode_kwargs={"batch_size": settings.embedding_batch_size},
            )
            detail["chunks"] = model.indexed_count

        with ledger.measure("search") as detail:
            run = model.search(
                task.queries,
                hf_split=task.split,
                hf_subset=task.hf_subset,
                top_k=top_k,
                encode_kwargs={"batch_size": settings.embedding_batch_size},
            )
            detail["queries"] = len(run)
            detail["top_k"] = top_k

        with ledger.measure("score") as detail:
            scores = metrics.evaluate_run(task.qrels, run, k_values=REPORT_K_VALUES)
            coverage = metrics.run_coverage(task.qrels, run)
            detail.update({str(key): value for key, value in coverage.items()})

    except IndexNotFoundError as exc:
        _fail(_EXIT_INDEX, str(exc))
        return _EXIT_INDEX
    except AxiomContractError as exc:
        # The one module where a contract violation must be fatal
        # (TechSpecifications.md 4.10). A wrong id silently scores 0.0.
        _fail(_EXIT_INTERNAL, f"contract violation: {exc}")
        return _EXIT_INTERNAL
    except AxiomError as exc:
        _fail(_EXIT_INTERNAL, str(exc))
        return _EXIT_INTERNAL

    elapsed_s = time.perf_counter() - wall_start
    backend_name = model.name
    backend_detail = model.backend_detail
    degraded_backend = model.degraded
    is_full_split = args.limit is None and args.split == "test"
    git_sha = _git_sha(_REPO_ROOT)
    dirty = bool(git_sha and git_sha.endswith("-dirty"))

    blockers: list[str] = []
    if placeholders:
        blockers.append(f"active placeholders: {', '.join(placeholders)} (Rules.md AP-14)")
    if not is_full_split:
        blockers.append("not a full test-split run (TestPlan.md 6.2)")
    if degraded_backend:
        blockers.append(f"degraded retrieval backend {backend_name!r} (NFR-07 fallback path)")
    resolved_encoder = backend_detail.get("encoder")
    if resolved_encoder is not None and resolved_encoder != settings.embedding_model:
        # NFR-09 pins model identity by name and revision. A resolved embedder
        # that is not the configured one means some rung of the embedder ladder
        # fired -- a missing ONNX export, a failed download -- and the number
        # belongs to whatever answered instead, not to the configured model.
        blockers.append(
            f"resolved embedder {resolved_encoder!r} is not the configured "
            f"{settings.embedding_model!r}"
        )
    if dirty:
        blockers.append("git tree is dirty, so the run is not reproducible")
    if git_sha is None:
        blockers.append("no git SHA available; the run cannot be logged in TestPlan.md 6.4")

    results: dict[str, Any] = {
        "dataset_revision": task.dataset_revision,
        "task_name": task.task_name,
        "mteb_version": _mteb_version(),
        "evaluation_time": round(elapsed_s, 3),
        "kg_co2_emissions": None,
        "scores": {
            task.split: [
                _score_block(
                    scores,
                    hf_subset=task.hf_subset,
                    languages=task.languages,
                    main_metric="ndcg_at_10",
                )
            ]
        },
        "axiom_provenance": {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "reportable": not blockers,
            "non_reportable_because": blockers,
            "mode": "FULL" if is_full_split else "SMOKE",
            "profile": settings.profile,
            "profile_file": str(
                (args.configs_dir or (_REPO_ROOT / "configs")) / f"{settings.profile}.yaml"
            ),
            "config_hash": _config_hash(settings),
            "placeholders_active": placeholders,
            "models": {
                "embedding_model": settings.embedding_model,
                "embedding_dim": settings.embedding_dim,
                "reranker_model": settings.reranker_model,
                "reranker_enabled": settings.reranker_enabled,
                "llm_model": settings.llm_model,
                "llm_enabled": settings.llm_enabled,
                "agent_enabled": settings.agent_enabled,
            },
            "weights": {
                "dense": settings.eval_dense_weight,
                "sparse": settings.eval_sparse_weight,
                "structural": 0.0,
            },
            "backend": backend_name,
            "backend_detail": backend_detail,
            "dataset_source": task.source,
            "limit": args.limit,
            "top_k": top_k,
            "coverage": coverage,
            "reported_percent": {
                "ndcg_at_10": round(scores.get("ndcg_at_10", 0.0) * 100.0, 4),
                "mrr_at_10": round(scores.get("mrr_at_10", 0.0) * 100.0, 4),
                "recall_at_100": round(scores.get("recall_at_100", 0.0) * 100.0, 4),
                "map_at_10": round(scores.get("map_at_10", 0.0) * 100.0, 4),
            },
            "git_sha": git_sha,
            "generated_at": started_at.isoformat().replace("+00:00", "Z"),
            "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            "python": sys.version.split()[0],
            "timings": ledger.as_dict(),
            "resolved_config": settings.model_dump(mode="json"),
        },
    }

    out_path = args.out or Path(f"{args.task.lower()}_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")

    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        args.predictions.write_text(
            json.dumps({task.split: run}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    _print_summary(results, out_path, scores, blockers)
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    return _EXIT_OK


def _print_summary(
    results: dict[str, Any],
    out_path: Path,
    scores: dict[str, float],
    blockers: list[str],
) -> None:
    """Human-readable summary to stderr; the file is the machine-readable artifact."""
    provenance = results["axiom_provenance"]
    percent = provenance["reported_percent"]
    lines = [
        f"task            {results['task_name']} / {next(iter(results['scores']))}",
        f"mode            {provenance['mode']}",
        f"profile         {provenance['profile']}  config_hash={provenance['config_hash']}",
        f"backend         {provenance['backend']}",
        f"corpus          {provenance['coverage']['qrels_queries']} judged queries, "
        f"{provenance['coverage']['retrieved_documents']} retrieved documents",
        f"NDCG@10         {percent['ndcg_at_10']:.2f}   (target >= 20.0)",
        f"MRR@10          {percent['mrr_at_10']:.2f}   (target >= 22.0)",
        f"Recall@100      {percent['recall_at_100']:.2f}   (target >= 65.0)",
        f"MAP@10          {percent['map_at_10']:.2f}",
        f"elapsed         {results['evaluation_time']:.1f}s",
        f"written         {out_path}",
    ]
    if blockers:
        lines.append("NOT REPORTABLE:")
        lines.extend(f"  - {reason}" for reason in blockers)
    else:
        lines.append(f"REPORTABLE at git {provenance['git_sha']}")
    print("\n".join(lines), file=sys.stderr)
    _LOGGER.info(
        "eval complete",
        extra={
            "axiom_extra": {
                "stage": "eval",
                "ndcg_at_10": scores.get("ndcg_at_10", 0.0),
                "reportable": provenance["reportable"],
            }
        },
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(_EXIT_INTERNAL) from None
