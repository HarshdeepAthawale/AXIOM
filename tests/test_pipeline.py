"""End to end on a synthetic repo with **no optional dependency installed**.

This is the NFR-07 test. Everything here runs on the bottom rung of every
ladder -- hash embedder, numpy dense search, pure-python BM25, sqlite structural
graph, passthrough rerank, heuristic classifier -- because that is the
configuration on an evaluator's laptop where ``pip install axiom[retrieval]``
never resolved a wheel. If this module is green on a bare install, the demo is
alive.

Following TestPlan.md section 1.2 rule 1, nothing here asserts *which* chunk
ranked first. What is asserted is that the artefacts exist in the contracted
layout, that every returned result carries a real file path and line range that
match the file on disk, that identity survives all eight stages (TC-015), and
that the whole thing degrades rather than raising.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from axiom import pipeline
from axiom.config import get_settings
from axiom.core.errors import IndexNotFoundError
from axiom.indexing import manifest as mf
from axiom.schema import RetrievalResult, SignalKind

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

Q1 = "How is the input preprocessed before going to the main function?"
Q2 = "Which files call preprocessInput before resolveTool?"
Q3 = "Where is the Bluetooth-settings deeplink used?"

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


class TestIndexBuild:
    def test_a_cold_build_produces_the_contracted_layout(self, indexed_v1) -> None:
        """_CONTRACT.md section 6: ``registry.json``, the version dir, the
        manifest, ``chunks.jsonl``, and the dense/sparse/structural artefacts."""
        settings, report = indexed_v1
        root = mf.index_root(settings)
        version = mf.version_dir(settings, "v1")

        assert (root / "registry.json").is_file()
        assert mf.manifest_path(settings, "v1").is_file()
        assert mf.chunks_path(settings, "v1").is_file()
        assert version.is_dir()
        assert report.chunk_count > 0 and report.file_count > 0

    def test_the_manifest_records_the_rung_that_actually_ran(self, indexed_v1) -> None:
        """Rules.md: record what loaded, never the configured value -- the
        configured Qwen model is absent here, so the manifest must not claim it."""
        settings, report = indexed_v1
        manifest = mf.read_manifest(settings, "v1")

        assert manifest.chunk_count == report.chunk_count
        assert manifest.embedding_dim > 0
        assert manifest.index_kind in {"flat_ip", "ivf_pq"}
        assert manifest.embedding_model != settings.embedding_model, (
            "the primary model cannot have loaded on a bare install"
        )

    def test_the_degraded_rungs_are_visible_in_the_report(self, indexed_v1) -> None:
        """Rule 3: a degrade that nobody notices is a silently wrong answer.

        ``IndexReport.degradations`` used to stay empty here -- it collected only
        whole-step *failures*, so a build that fell all the way down four ladders
        reported nothing while a dozen WARNINGs scrolled past, and ``axiom index
        --json`` could not tell an operator which rung had run (NFR-07). The
        build now listens on the ``axiom`` logger, so the list carries the rungs
        themselves.

        What this asserts is the *reporting mechanism*, not which rung happened
        to win. An earlier version hardcoded ``dense_backend == "numpy"`` with
        the note "faiss cannot have loaded on a bare install" -- which made the
        test fail the moment someone installed the extras, i.e. exactly when the
        system got better. Optional dependencies are optional in both
        directions, so the invariant is: whichever rung ran is *named*, and any
        rung below the top is *reported*.
        """
        _, report = indexed_v1
        assert report.dense_backend in {"faiss", "numpy"}, report.dense_backend
        assert report.sparse_backend in {"bm25s", "pure_python"}, report.sparse_backend

        joined = " | ".join(report.degradations)
        assert len(report.degradations) == len(set(report.degradations)), "rungs must dedupe"

        # Each ladder that did not run at its top rung must say so.
        if report.dense_backend == "numpy":
            assert "faiss" in joined, joined
        if report.sparse_backend == "pure_python":
            assert "bm25s" in joined, joined
        if "hash-embedder" in report.manifest.embedding_model:
            assert "hash-embedder" in joined, joined
        # The primary embedder needs an ONNX export that no test fixture ships,
        # so the embedder ladder always degrades at least one rung here.
        assert any("embedder" in entry for entry in report.degradations), joined

    def test_chunks_jsonl_round_trips_through_the_schema(self, indexed_v1) -> None:
        """Every line must validate, including the ``chunk_id`` integrity check."""
        settings, report = indexed_v1
        chunks = mf.read_version_chunks(settings, "v1")
        assert len(chunks) == report.chunk_count
        assert all(len(chunk.chunk_id) == 32 for chunk in chunks)

        raw = mf.chunks_path(settings, "v1").read_text(encoding="utf-8").splitlines()
        assert len(raw) == report.chunk_count
        json.loads(raw[0])

    def test_chunk_ids_are_unique_across_the_whole_index(self, indexed_v1) -> None:
        settings, _ = indexed_v1
        ids = [chunk.chunk_id for chunk in mf.read_version_chunks(settings, "v1")]
        assert len(set(ids)) == len(ids)

    def test_the_version_is_registered_and_made_active(self, indexed_v1) -> None:
        settings, _ = indexed_v1
        registry = mf.load_registry(settings)
        assert registry.active_version == "v1"
        assert "v1" in registry.versions

    def test_the_build_is_deterministic(self, repo_v1: Path, tmp_path: Path) -> None:
        """NFR-08: the same repo built twice yields the same chunk id set."""
        first = get_settings("default", configs_dir=CONFIGS, index_root=tmp_path / "a")
        second = get_settings("default", configs_dir=CONFIGS, index_root=tmp_path / "b")
        pipeline.build_index(repo_v1, "v1", first)
        pipeline.build_index(repo_v1, "v1", second)
        assert {c.chunk_id for c in mf.read_version_chunks(first, "v1")} == {
            c.chunk_id for c in mf.read_version_chunks(second, "v1")
        }

    def test_an_empty_repository_still_publishes_a_version(self, tmp_path: Path) -> None:
        """TC-078's shape: zero chunks is a legal index, not a failure."""
        settings = get_settings("default", configs_dir=CONFIGS, index_root=tmp_path / ".axiom")
        empty = tmp_path / "empty-repo"
        empty.mkdir()
        manifest = pipeline.build_index(empty, "v0", settings)
        assert manifest.chunk_count == 0
        assert mf.manifest_path(settings, "v0").is_file()

    def test_a_repo_containing_only_a_broken_file_still_builds(
        self, repo_v1: Path, tmp_path: Path
    ) -> None:
        """NFR-07: one bad file must not cost the index."""
        settings = get_settings("default", configs_dir=CONFIGS, index_root=tmp_path / ".axiom")
        lone = tmp_path / "lone"
        (lone / "src").mkdir(parents=True)
        shutil.copy2(repo_v1 / "src/broken/syntax_error.js", lone / "src" / "broken.js")
        manifest = pipeline.build_index(lone, "v1", settings)
        assert manifest.chunk_count >= 1

    def test_the_index_report_serialises(self, indexed_v1) -> None:
        _, report = indexed_v1
        json.dumps(report.as_dict(), default=str)

    def test_index_json_is_exactly_the_documented_index_summary(self, indexed_v1) -> None:
        """API.md section 8: ``axiom index --json`` emits ``IndexSummary``, whole.

        Pinned because the hand-written copy this replaced had drifted twice
        over -- an ``index_kind`` key where the contract says
        ``dense_index_kind``, and two fields the contract never listed. A key
        set compared against the model rather than against a literal means a
        new ``IndexSummary`` field cannot ship without appearing on the wire.
        """
        from axiom.api.models import IndexSummary

        _, report = indexed_v1
        body = report.as_dict()
        assert set(body) == set(IndexSummary.model_fields), (
            "the --json body and IndexSummary must not drift apart"
        )

    def test_index_json_timings_are_flat_stage_to_milliseconds(self, indexed_v1) -> None:
        """API.md section 8 types ``timings`` as ``dict[str, float]``.

        The ledger's own ``as_dict`` is a nested ``{total_ms, stages: [...]}``
        record; that is the logging shape and must not reach the wire, because a
        client typed against the contract cannot read it.
        """
        _, report = indexed_v1
        timings = report.as_dict()["timings"]
        assert isinstance(timings, dict) and timings
        assert all(isinstance(key, str) for key in timings)
        assert all(isinstance(value, float) for value in timings.values()), (
            f"nested ledger record leaked onto the wire: {timings!r}"
        )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQuery:
    @pytest.mark.parametrize("query", [Q1, Q2, Q3])
    def test_every_archetype_returns_results_with_real_locations(
        self, indexed_v1, repo_v1: Path, query: str
    ) -> None:
        """The assertion that matters for the demo: a returned ``file:line``
        actually resolves to that text in the file on disk."""
        settings, _ = indexed_v1
        response = pipeline.query(query, settings)
        assert response.results, f"{query!r} returned nothing"

        for result in response.results:
            location = result.chunk.location
            source = (repo_v1 / location.file_path).read_text(encoding="utf-8")
            assert location.start_line >= 1 and location.end_line >= location.start_line
            lines = source.split("\n")
            assert location.end_line <= len(lines), "line range runs past the end of the file"

            # The BYTE range is the exact contract (Schema.md section 6):
            # text == source_bytes[start_byte:end_byte].decode("utf-8").
            source_bytes = (repo_v1 / location.file_path).read_bytes()
            assert (
                source_bytes[location.start_byte : location.end_byte].decode("utf-8")
                == result.chunk.text
            ), f"byte range does not reproduce the chunk at {location.as_ref()}"

            # The LINE range is a human-facing locator and is a *superset* of
            # the text whenever the AST node begins mid-line -- a nested
            # function expression, an arrow function, or an object-literal
            # method all start after an indent and a `return `/`=`. PRD US-6
            # claims whole-line slicing reproduces the chunk byte-for-byte;
            # that holds only for chunks starting at column 0, so the honest
            # invariant is containment, with the byte range as the exact one.
            line_slice = "\n".join(lines[location.start_line - 1 : location.end_line])
            assert result.chunk.text.strip() in line_slice, (
                f"line range does not contain the chunk at {location.as_ref()}"
            )

    def test_every_result_explains_itself(self, indexed_v1) -> None:
        """FR-14: a non-empty ``match_reason`` naming the dominant signal, and
        ``signals`` carrying the first-stage ranks that power the UI panel."""
        settings, _ = indexed_v1
        response = pipeline.query(Q2, settings)
        assert response.results
        for result in response.results:
            assert isinstance(result, RetrievalResult)
            assert result.match_reason.strip()
            assert result.signals
            assert all(rank >= 1 for rank in result.signals.values())
            assert 0.0 <= result.score <= 1.0

    def test_the_q2_archetype_is_answered_by_the_structural_signal(self, indexed_v1) -> None:
        """The innovation claim, end to end: the ordered-call-pair evidence has
        to survive chunking, indexing, retrieval, fusion and formatting to reach
        the user as a sentence."""
        settings, _ = indexed_v1
        response = pipeline.query(Q2, settings)
        assert response.query_plan.query_type.value == "structural"
        reasons = " | ".join(result.match_reason for result in response.results)
        assert "structural" in reasons
        assert "before resolveTool" in reasons, (
            "the ordered-pair explanation did not reach the user"
        )

    def test_results_are_ranked_and_deduplicated(self, indexed_v1) -> None:
        settings, _ = indexed_v1
        response = pipeline.query(Q1, settings)
        scores = [result.score for result in response.results]
        assert scores == sorted(scores, reverse=True)
        ids = [result.chunk.chunk_id for result in response.results]
        assert len(set(ids)) == len(ids)

    def test_top_k_is_honoured(self, indexed_v1) -> None:
        settings, _ = indexed_v1
        assert len(pipeline.query(Q1, settings, top_k=2).results) <= 2
        assert pipeline.query(Q1, settings, top_k=0).results == []

    def test_tc015_chunk_ids_round_trip_unchanged_through_every_stage(self, indexed_v1) -> None:
        """TC-015: no stage rewrites, truncates, lowercases or re-derives an id.

        Checked across the stages this suite can observe from outside: the
        chunker's output as written to ``chunks.jsonl``, the dense idmap, the
        structural store, and the ``RetrievalResult`` the user receives.
        """
        settings, _ = indexed_v1
        on_disk = {chunk.chunk_id for chunk in mf.read_version_chunks(settings, "v1")}

        idmap = mf.version_dir(settings, "v1") / "dense.idmap.json"
        if idmap.is_file():
            payload = json.loads(idmap.read_text(encoding="utf-8"))
            mapped = payload if isinstance(payload, list) else payload.get("chunk_ids", [])
            assert set(mapped) <= on_disk

        returned = {
            result.chunk.chunk_id
            for query in (Q1, Q2, Q3)
            for result in pipeline.query(query, settings).results
        }
        assert returned, "no ids were returned at all"
        assert returned <= on_disk, "a returned id is not in chunks.jsonl"

    def test_the_response_serialises_for_the_json_mode(self, indexed_v1) -> None:
        """NFR-10: every ``--json`` response carries a ``timings`` block."""
        settings, _ = indexed_v1
        payload = pipeline.query(Q1, settings).as_dict()
        json.dumps(payload)
        assert "timings" in payload and payload["timings"]["stages"]
        assert payload["profile"] == "default"

    def test_tc066_two_runs_agree_on_everything_but_the_clock(self, indexed_v1) -> None:
        """TC-066: ``elapsed_ms`` is the only field excluded from determinism."""
        settings, _ = indexed_v1
        first = pipeline.query(Q2, settings).as_dict()
        second = pipeline.query(Q2, settings).as_dict()
        for payload in (first, second):
            payload.pop("elapsed_ms")
            payload.pop("timings")
        assert first == second

    def test_formatted_lines_carry_a_location_and_a_reason(self, indexed_v1) -> None:
        settings, _ = indexed_v1
        response = pipeline.query(Q2, settings)
        lines = pipeline.format_ranked_lines(response)
        assert lines
        assert any(":" in line for line in lines)
        assert len(lines) == 2 * len(response.results)


# ---------------------------------------------------------------------------
# Incremental reindex: the version it produces must be queryable
# ---------------------------------------------------------------------------


class TestReindexProducesAQueryableVersion:
    """FR-18/FR-19. The failure this guards was silent and total.

    ``versioning.incremental._build_side_indexes`` used to discover its builders
    with ``getattr(module, "build_index")`` and call them as
    ``builder(chunks, out_dir, settings)``. Only ``sparse`` and ``structural``
    carry that alias, so the dense index was never built; and the real signature
    is ``(chunks, settings, out_dir)``, so the ``except TypeError`` retry bound
    ``out_dir`` to ``settings`` and left ``out_dir`` at its ``Path(".")``
    default -- ``sparse.bm25s/`` landed in the process's working directory.

    The version still published, still registered, still reported success, and
    answered **zero** queries with all three signals absent.
    """

    def test_every_artefact_lands_in_the_version_directory(
        self, repo_v1: Path, settings, monkeypatch, tmp_path: Path
    ) -> None:
        from axiom.versioning.incremental import reindex

        pipeline.build_index(repo_v1, "v1", settings)

        # A working directory that is not the index root, so anything written
        # relative to "." is visible as pollution rather than blending in.
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        report = reindex("v1", "v2", settings, repo_path=repo_v1)
        assert report.chunk_count > 0

        version = mf.version_dir(settings, "v2")
        for artefact in ("chunks.jsonl", "manifest.json", "dense.idmap.json"):
            assert (version / artefact).is_file(), f"{artefact} missing from {version}"
        # The dense artefact is named by the rung that ran: faiss writes
        # dense.faiss, the numpy fallback writes dense.npy. Asserting one or the
        # other would make this test depend on whether the optional extras are
        # installed -- what matters is that the version carries exactly one.
        dense = [name for name in ("dense.faiss", "dense.npy") if (version / name).is_file()]
        assert len(dense) == 1, f"expected one dense artefact in {version}, found {dense}"
        assert (version / "sparse.bm25s").is_dir()
        assert (version / "structural.sqlite").is_file()

        stray = list(cwd.iterdir())
        assert stray == [], f"reindex wrote into the working directory: {stray}"
        assert not any("no dense index builder" in note for note in report.degradations), (
            report.degradations
        )

    def test_the_reindexed_version_answers_with_all_three_signals(
        self, repo_v1: Path, settings
    ) -> None:
        from axiom.versioning.incremental import reindex

        pipeline.build_index(repo_v1, "v1", settings)
        reindex("v1", "v2", settings, repo_path=repo_v1)

        response = pipeline.query(Q2, settings, version_id="v2", top_k=5)
        assert response.results, "a reindexed version must be queryable"
        signals = {signal for r in response.results for signal in r.signals}
        assert SignalKind.DENSE in signals
        assert SignalKind.SPARSE in signals
        assert SignalKind.STRUCTURAL in signals

    def test_an_unchanged_tree_costs_no_embedding_calls(self, repo_v1: Path, settings) -> None:
        """TC-075 through the real builders, not an injected map.

        The blob cache is content-addressed, so the dense builder sees 100%%
        cache hits and the embedder ladder never encodes anything -- which is
        also what proves the dense build actually ran rather than being skipped.
        """
        from axiom.versioning.incremental import reindex

        pipeline.build_index(repo_v1, "v1", settings)
        report = reindex("v1", "v2", settings, repo_path=repo_v1)
        assert report.embed_calls == 0
        assert report.blobs_reused > 0


# ---------------------------------------------------------------------------
# Concurrency: the fan-out reaches one retriever from several threads at once
# ---------------------------------------------------------------------------


class TestConcurrentFanOut:
    """NFR-08 across the cold/warm boundary, which is where it actually broke.

    ``IndexBackend.fan_out`` submits one task per (version, sub-query, signal),
    so a three-sub-query plan calls ``DenseRetriever.search`` from three threads
    simultaneously -- on a retriever nobody has loaded yet. Both retrievers
    latched their lazy load with a plain ``self._loaded = True`` written *before*
    the load ran, so the two threads that lost the race read an availability flag
    the winner had not set yet and answered an empty list.

    The consequence was not a crash and not a log line: the first query of every
    process quietly retrieved on a fraction of its sub-queries. The CLI runs one
    query per process, so that was every ``axiom query`` on the demo machine, and
    the only externally visible symptom was that a second identical query in the
    same process returned different ranks.
    """

    def _concurrent(self, retriever, query: str, plan) -> list[list[str]]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=3) as pool:
            runs = pool.map(lambda _: retriever.search(query, plan, 5), range(3))
            return [[hit.chunk_id for hit in run] for run in runs]

    def test_a_cold_dense_retriever_survives_three_simultaneous_searches(self, indexed_v1) -> None:
        from axiom.agent.planner import build_plan
        from axiom.retrieval.dense import DenseRetriever

        settings, _ = indexed_v1
        directory = mf.version_dir(settings, "v1")
        plan = build_plan(Q2, settings)

        runs = self._concurrent(DenseRetriever(directory, settings), Q2, plan)
        assert runs[0], "the cold winner must find candidates at all"
        assert all(run == runs[0] for run in runs), "a losing thread saw an unloaded index"

    def test_a_cold_sparse_retriever_survives_three_simultaneous_searches(self, indexed_v1) -> None:
        from axiom.agent.planner import build_plan
        from axiom.retrieval.sparse import SparseRetriever

        settings, _ = indexed_v1
        directory = mf.version_dir(settings, "v1")
        plan = build_plan(Q2, settings)

        runs = self._concurrent(SparseRetriever(directory, settings), Q2, plan)
        assert runs[0], "the cold winner must find candidates at all"
        assert all(run == runs[0] for run in runs), "a losing thread saw an unloaded index"

    def test_the_first_query_of_a_process_matches_the_second(self, indexed_v1) -> None:
        """The end-to-end symptom, asserted end to end.

        ``indexed_v1`` builds into a fresh ``tmp_path``, so the query below is
        genuinely the first one any retriever over this index has seen.
        """
        settings, _ = indexed_v1
        cold = pipeline.query(Q2, settings)
        warm = pipeline.query(Q2, settings)
        assert [r.chunk.chunk_id for r in cold.results] == [r.chunk.chunk_id for r in warm.results]
        assert [dict(r.signals) for r in cold.results] == [dict(r.signals) for r in warm.results]


# ---------------------------------------------------------------------------
# Degradation on the query path
# ---------------------------------------------------------------------------


class TestQueryDegradation:
    def test_an_empty_query_returns_a_warning_not_a_traceback(self, indexed_v1) -> None:
        """TC-009 as implemented: Rule 3 forbids raising on user input, so the
        CLI/API layer maps ``stop_reason`` to its exit code / 422."""
        settings, _ = indexed_v1
        response = pipeline.query("   ", settings)
        assert response.results == []
        assert response.stop_reason == "empty_query"
        assert any(w["code"] == "EMPTY_QUERY" for w in response.warnings)

    def test_tc010_a_10000_char_query_completes(self, indexed_v1) -> None:
        settings, _ = indexed_v1
        response = pipeline.query("x" * 10_000, settings)
        assert len(response.query_plan.original_query) <= 2048
        assert response.stop_reason

    def test_tc011_unicode_and_emoji_do_not_break_the_pipeline(self, indexed_v1) -> None:
        settings, _ = indexed_v1
        response = pipeline.query("où est le déeplink 🔵 Bluetooth ?", settings)
        assert isinstance(response.results, list)

    @pytest.mark.parametrize(
        "hostile",
        ["!!!???", "'; DROP TABLE symbols; --", "(" * 200, "\x00\x01", "zzqqxx_not_a_token"],
    )
    def test_hostile_queries_never_raise(self, indexed_v1, hostile: str) -> None:
        settings, _ = indexed_v1
        response = pipeline.query(hostile, settings)
        assert isinstance(response.results, list)

    def test_an_out_of_vocabulary_query_yields_an_honest_empty_result(self, indexed_v1) -> None:
        """TC-045 / TC-056 end to end: an empty fused list becomes an
        empty-but-valid response carrying ``NO_RESULTS``, never a crash."""
        settings, _ = indexed_v1
        response = pipeline.query("zzqqxx_not_a_token_anywhere", settings)
        if not response.results:
            assert any(w["code"] == "NO_RESULTS" for w in response.warnings)

    def test_a_missing_index_raises_index_not_found(self, tmp_path: Path) -> None:
        """The one thing that raises: "there is nothing to search" is not a
        ranking a caller can degrade into."""
        settings = get_settings("default", configs_dir=CONFIGS, index_root=tmp_path / "nothing")
        with pytest.raises(IndexNotFoundError):
            pipeline.query(Q1, settings)

    def test_a_deleted_sparse_index_still_answers(self, indexed_v1) -> None:
        """Fusion drops the dead signal and renormalises the survivors."""
        settings, _ = indexed_v1
        sparse = mf.version_dir(settings, "v1") / "sparse.bm25s"
        if sparse.exists():
            shutil.rmtree(sparse)
        response = pipeline.query(Q1, settings)
        assert isinstance(response.results, list)

    def test_a_corrupt_dense_idmap_still_answers(self, indexed_v1) -> None:
        settings, _ = indexed_v1
        idmap = mf.version_dir(settings, "v1") / "dense.idmap.json"
        if idmap.is_file():
            idmap.write_text("{ not json at all", encoding="utf-8")
        response = pipeline.query(Q2, settings)
        assert isinstance(response.results, list)

    def test_every_signal_disabled_returns_no_results_rather_than_an_error(
        self, indexed_v1
    ) -> None:
        settings, _ = indexed_v1
        dark = settings.model_copy(
            update={"dense_enabled": False, "sparse_enabled": False, "structural_enabled": False}
        )
        response = pipeline.query(Q1, dark)
        assert response.results == []
        assert any(w["code"] == "NO_RESULTS" for w in response.warnings)


# ---------------------------------------------------------------------------
# Versions (FR-20)
# ---------------------------------------------------------------------------


class TestVersions:
    @pytest.fixture
    def two_versions(self, repo_v1: Path, repo_v2: Path, tmp_path: Path):
        settings = get_settings(
            "default",
            configs_dir=CONFIGS,
            index_root=tmp_path / ".axiom",
            llm_enabled=False,
            reranker_enabled=False,
        )
        pipeline.build_index(repo_v1, "v1", settings)
        pipeline.build_index(repo_v2, "v2", settings, parent_version="v1")
        return settings

    def test_version_scoping_is_structural_not_a_filter(self, two_versions) -> None:
        """A query against v1 opens ``.axiom/index/v1/`` and can physically not
        see v2's rows -- ``telemetry.js`` exists only in v2."""
        v1_paths = {
            result.chunk.location.file_path
            for result in pipeline.query("record a counter", two_versions, version_id="v1").results
        }
        assert "src/tools/telemetry.js" not in v1_paths

        v2 = pipeline.query("record a counter", two_versions, version_id="v2")
        assert all(result.chunk.metadata.version_id == "v2" for result in v2.results)

    def test_each_response_names_the_versions_it_searched(self, two_versions) -> None:
        assert pipeline.query(Q1, two_versions, version_id="v1").version_ids == ["v1"]
        assert set(pipeline.query(Q1, two_versions, all_versions=True).version_ids) == {"v1", "v2"}

    def test_all_versions_searches_every_registered_version(self, two_versions) -> None:
        response = pipeline.query(Q1, two_versions, all_versions=True)
        assert response.results
        assert {result.chunk.metadata.version_id for result in response.results} <= {"v1", "v2"}

    def test_an_unknown_version_raises_index_not_found(self, two_versions) -> None:
        with pytest.raises(IndexNotFoundError):
            pipeline.query(Q1, two_versions, version_id="v99")

    def test_a_pure_rename_between_versions_reuses_the_blob(self, two_versions) -> None:
        """``deep/nested.js`` -> ``deep/pipeline.js``, byte identical (TC-075).

        Asserted on the built artefacts rather than on an embedder counter: the
        two versions' chunks share a ``content_hash``, which is precisely what
        makes the second build's forward pass unnecessary.
        """
        v1 = {
            c.content_hash
            for c in mf.read_version_chunks(two_versions, "v1")
            if c.location.file_path == "src/deep/nested.js"
        }
        v2 = {
            c.content_hash
            for c in mf.read_version_chunks(two_versions, "v2")
            if c.location.file_path == "src/deep/pipeline.js"
        }
        assert v1 and v2 and v1 == v2

        v1_ids = {
            c.chunk_id
            for c in mf.read_version_chunks(two_versions, "v1")
            if c.location.file_path == "src/deep/nested.js"
        }
        v2_ids = {
            c.chunk_id
            for c in mf.read_version_chunks(two_versions, "v2")
            if c.location.file_path == "src/deep/pipeline.js"
        }
        assert not (v1_ids & v2_ids), "a renamed chunk must be re-addressed"

    def test_the_manifest_chain_records_the_parent(self, two_versions) -> None:
        assert mf.read_manifest(two_versions, "v2").parent_version == "v1"
        assert mf.read_manifest(two_versions, "v1").parent_version is None


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def test_the_eval_profile_runs_one_pass_and_no_structural_signal(
    repo_v1: Path, tmp_path: Path
) -> None:
    """PRD.md section 2.2: on ``eval`` the structural leg is not built at all
    and the agent loop is the ablation, so exactly one pass runs."""
    settings = get_settings("eval", configs_dir=CONFIGS, index_root=tmp_path / ".axiom")
    report = pipeline.build_index_detailed(repo_v1, "v1", settings)
    assert report.structural_skipped is True

    response = pipeline.query(Q1, settings)
    assert response.passes_used <= 1
    assert SignalKind.STRUCTURAL not in response.query_plan.strategy_weights


def test_the_fast_profile_builds_and_queries(repo_v1: Path, tmp_path: Path) -> None:
    """NFR-12 / A-3: the 4-core, 8 GB, <500 MB-download floor."""
    settings = get_settings("fast", configs_dir=CONFIGS, index_root=tmp_path / ".axiom")
    assert settings.llm_enabled is False
    pipeline.build_index(repo_v1, "v1", settings)
    assert isinstance(pipeline.query(Q3, settings).results, list)


# ---------------------------------------------------------------------------
# NFR-07 -- the import contract itself
# ---------------------------------------------------------------------------


def test_no_optional_dependency_is_imported_by_importing_axiom() -> None:
    """NFR-07, asserted directly: ``import axiom.<anything>`` must succeed on a
    bare install, and must not drag a heavy wheel in as a side effect.

    Walks every module in the package in a subprocess, so a module another test
    already imported cannot mask a top-level import.
    """
    import subprocess
    import sys

    script = (
        "import importlib, pkgutil, sys\n"
        "import axiom\n"
        "failed = []\n"
        "for m in pkgutil.walk_packages(axiom.__path__, 'axiom.'):\n"
        "    try:\n"
        "        importlib.import_module(m.name)\n"
        "    except Exception as exc:\n"
        "        failed.append((m.name, repr(exc)))\n"
        "heavy = {'faiss','onnxruntime','transformers','tokenizers','bm25s',"
        "'tree_sitter','tree_sitter_javascript','llama_cpp','mteb','datasets'}\n"
        "leaked = sorted(heavy & set(sys.modules))\n"
        "print(repr((failed, leaked)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    failed, leaked = eval(completed.stdout.strip())
    assert failed == [], f"modules failed to import on a bare install: {failed}"
    assert leaked == [], f"importing axiom pulled in optional dependencies: {leaked}"


class TestModelDir:
    """Models resolve independently of the working directory.

    ``axiom reindex`` must run inside the repository being indexed, so a path
    relative to the working directory found no models there and every rung
    degraded to the hashing embedder without anyone asking for it.
    """

    def test_the_environment_variable_wins(self, tmp_path: Path, monkeypatch) -> None:
        from axiom.config import resolve_model_dir

        monkeypatch.setenv("AXIOM_MODEL_DIR", str(tmp_path / "models"))
        assert resolve_model_dir() == tmp_path / "models"

    def test_a_local_data_models_directory_is_used(self, tmp_path: Path, monkeypatch) -> None:
        from axiom.config import MODEL_DIR, resolve_model_dir

        monkeypatch.delenv("AXIOM_MODEL_DIR", raising=False)
        (tmp_path / MODEL_DIR).mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        assert resolve_model_dir() == MODEL_DIR

    def test_elsewhere_it_falls_back_to_the_checkout(self, tmp_path: Path, monkeypatch) -> None:
        from axiom import config

        monkeypatch.delenv("AXIOM_MODEL_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        checkout = Path(config.__file__).resolve().parents[2] / config.MODEL_DIR
        expected = checkout if checkout.is_dir() else config.MODEL_DIR
        assert config.resolve_model_dir() == expected

    def test_the_quantiser_output_name_is_accepted(self, tmp_path: Path) -> None:
        from axiom.config import onnx_model_file

        assert onnx_model_file(tmp_path) == tmp_path / "model.onnx"
        (tmp_path / "model_quantized.onnx").write_bytes(b"")
        assert onnx_model_file(tmp_path) == tmp_path / "model_quantized.onnx"
        (tmp_path / "model.onnx").write_bytes(b"")
        assert onnx_model_file(tmp_path) == tmp_path / "model.onnx"
