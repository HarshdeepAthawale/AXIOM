"""The evaluation harness: the arithmetic, the data path, and the id fence.

The audit recorded ``src/axiom/eval/`` (669 statements) at zero coverage. That
is the module whose output is the screening gate's number, so the absence was
the most expensive one on the list.

What is asserted here, in order of how badly a defect would hurt:

1. **The metrics are right.** ``ndcg_at_k`` and friends are hand-checked against
   values computed on paper, and against ``trec_eval``'s conventions -- linear
   gain, ideal ranking from the qrels, unjudged means zero, denominator is the
   qrels' query set. A metric that is wrong by a constant factor still looks
   plausible, which is exactly why it needs worked examples rather than
   self-consistency checks.
2. **The id fence holds.** ``AxiomSearchModel`` is the one module
   TechSpecifications.md section 4.10 says must *raise* rather than degrade: a
   mangled document id does not fail loudly, it scores 0.0 and reads as a bad
   model.
3. **The offline path works.** A vendored BEIR-shaped directory is assumption
   A-1's mitigation; it has to load without ``mteb``, ``datasets`` or a network.
4. **The degraded rung is never mistaken for a result.** ``LexicalBackend`` is a
   wiring proof, and ``degraded`` is a ``ClassVar`` specifically so no caller can
   construct one and claim otherwise.

``mteb`` and ``datasets`` are optional (NFR-07), so every test here runs on a
bare install; the handful that genuinely need MTEB use ``importorskip``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from axiom.eval.metrics import (
    DEFAULT_K_VALUES,
    dcg,
    evaluate_run,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    per_query_ap_at_k,
    per_query_mrr_at_k,
    per_query_ndcg_at_k,
    per_query_recall_at_k,
    rank_documents,
    recall_at_k,
    run_coverage,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The worked example from ``metrics.py``'s module docstring, kept here so a
#: change to either copy has to be made in both.
QRELS: dict[str, dict[str, float]] = {
    "q1": {"d1": 3, "d2": 2, "d3": 0, "d4": 1},
    "q2": {"dA": 1},
}
RUN: dict[str, dict[str, float]] = {
    "q1": {"d3": 0.9, "d1": 0.8, "d5": 0.7, "d2": 0.6},
    "q2": {},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def beir_dir(tmp_path: Path) -> Path:
    """A minimal BEIR-shaped corpus on disk -- the offline path (A-1)."""
    root = tmp_path / "TinyCode"
    (root / "qrels").mkdir(parents=True)
    documents = [
        {"_id": "d1", "title": "binary search", "text": "def bsearch(a, t):\n    lo = 0\n"},
        {"_id": "d2", "title": "quick sort", "text": "def quicksort(a):\n    return a\n"},
        {"_id": "d3", "title": "fibonacci", "text": "def fib(n):\n    return n\n"},
        {"_id": "d4", "title": "", "text": ""},
    ]
    (root / "corpus.jsonl").write_text(
        "\n".join(json.dumps(d) for d in documents) + "\n", encoding="utf-8"
    )
    queries = [
        {"_id": "q1", "text": "search a sorted array"},
        {"_id": "q2", "text": "sort a list quickly"},
    ]
    (root / "queries.jsonl").write_text(
        "\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8"
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n", encoding="utf-8"
    )
    (root / "revision.txt").write_text("fixture-rev-1\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestRanking:
    """NFR-08: an identical run must produce an identical ranking, always."""

    def test_ties_break_by_ascending_document_id(self) -> None:
        assert rank_documents({"z": 1.0, "a": 1.0, "m": 1.0}) == ["a", "m", "z"]

    def test_insertion_order_cannot_change_the_ranking(self) -> None:
        forwards = rank_documents({"a": 0.5, "b": 0.5, "c": 0.9})
        backwards = rank_documents({"c": 0.9, "b": 0.5, "a": 0.5})
        assert forwards == backwards == ["c", "a", "b"]

    def test_a_nan_score_sinks_to_the_bottom_rather_than_poisoning_the_sort(self) -> None:
        """Rule 3: bad data in a 3,765-query run is not a bug in the metric."""
        ranked = rank_documents({"good": 0.5, "bad": float("nan"), "best": 0.9})
        assert ranked == ["best", "good", "bad"]

    def test_truncation_keeps_the_best_k(self) -> None:
        assert rank_documents({"a": 0.1, "b": 0.9, "c": 0.5}, 2) == ["b", "c"]
        assert rank_documents({"a": 0.1}, 0) == []
        assert rank_documents({"a": 0.1}, -5) == []


class TestDiscountedCumulativeGain:
    """Linear gain, ``1/log2(rank+1)`` discount -- ``trec_eval``'s ``ndcg_cut``."""

    def test_the_hand_computed_values_from_the_docstring(self) -> None:
        assert dcg([0, 3, 0]) == pytest.approx(3 / math.log2(3), abs=1e-9)
        assert dcg([3, 2, 1]) == pytest.approx(3 + 2 / math.log2(3) + 0.5, abs=1e-9)

    def test_the_gain_function_is_selectable_and_changes_the_number(self) -> None:
        """Graded relevance moves by several points between the two conventions."""
        assert dcg([3], gain="linear") == pytest.approx(3.0)
        assert dcg([3], gain="exponential") == pytest.approx(7.0)

    def test_a_negative_grade_is_judged_non_relevant_not_a_penalty(self) -> None:
        assert dcg([-5, 1]) == pytest.approx(1 / math.log2(3), abs=1e-9)

    def test_an_empty_ranking_scores_zero(self) -> None:
        assert dcg([]) == 0.0


class TestPerQueryMetrics:
    """Every value below was computed on paper first (see ``metrics.py``'s header)."""

    def test_ndcg_matches_the_worked_example(self) -> None:
        assert per_query_ndcg_at_k(QRELS["q1"], RUN["q1"], 3) == pytest.approx(0.39748952, abs=1e-8)

    def test_the_ideal_ranking_comes_from_the_qrels_not_from_the_run(self) -> None:
        """``d4`` is never retrieved, yet it must still raise the denominator."""
        without_d4 = {"d1": 3, "d2": 2, "d3": 0}
        assert per_query_ndcg_at_k(without_d4, RUN["q1"], 3) > per_query_ndcg_at_k(
            QRELS["q1"], RUN["q1"], 3
        )

    def test_an_unjudged_retrieved_document_contributes_nothing(self) -> None:
        """``d5`` is in the run and absent from the qrels: grade 0, not an error."""
        assert "d5" not in QRELS["q1"]
        with_d5 = per_query_ndcg_at_k(QRELS["q1"], RUN["q1"], 3)
        without_d5 = per_query_ndcg_at_k(
            QRELS["q1"], {k: v for k, v in RUN["q1"].items() if k != "d5"}, 3
        )
        # Removing an unjudged document promotes d2 into the window, so the
        # score can only rise -- it can never have been contributing gain.
        assert without_d5 >= with_d5

    def test_a_query_with_no_positives_scores_zero_rather_than_dividing_by_zero(
        self,
    ) -> None:
        assert per_query_ndcg_at_k({"d1": 0}, {"d1": 1.0}, 10) == 0.0
        assert per_query_mrr_at_k({"d1": 0}, {"d1": 1.0}, 10) == 0.0
        assert per_query_recall_at_k({}, {"d1": 1.0}, 10) == 0.0
        assert per_query_ap_at_k({}, {"d1": 1.0}, 10) == 0.0

    def test_mrr_is_the_reciprocal_rank_of_the_first_hit(self) -> None:
        assert per_query_mrr_at_k(QRELS["q1"], RUN["q1"], 3) == 0.5
        assert per_query_mrr_at_k(QRELS["q1"], RUN["q1"], 1) == 0.0

    def test_recall_divides_by_every_positive_not_by_min_k_positives(self) -> None:
        """Capping the denominator would let 200 positives score 1.0 at k=100."""
        assert per_query_recall_at_k(QRELS["q1"], RUN["q1"], 3) == pytest.approx(1 / 3)
        many = {f"d{i}": 1 for i in range(10)}
        assert per_query_recall_at_k(many, {"d0": 1.0}, 1) == pytest.approx(0.1)

    def test_average_precision_normalises_by_the_positives_it_never_found(self) -> None:
        assert per_query_ap_at_k(QRELS["q1"], RUN["q1"], 3) == pytest.approx(1 / 6)

    def test_a_perfect_ranking_scores_one_on_every_metric(self) -> None:
        relevance = {"d1": 1.0, "d2": 1.0}
        scored = {"d1": 0.9, "d2": 0.8}
        assert per_query_ndcg_at_k(relevance, scored, 2) == pytest.approx(1.0)
        assert per_query_mrr_at_k(relevance, scored, 2) == 1.0
        assert per_query_recall_at_k(relevance, scored, 2) == 1.0
        assert per_query_ap_at_k(relevance, scored, 2) == pytest.approx(1.0)

    @pytest.mark.parametrize("k", [0, -1, -100])
    def test_a_non_positive_cut_off_degrades_to_zero_rather_than_raising(self, k: int) -> None:
        """Rule 3: a wiring mistake must show up as a visible zero, not a traceback."""
        assert per_query_ndcg_at_k(QRELS["q1"], RUN["q1"], k) == 0.0
        assert ndcg_at_k(QRELS, RUN, k) == 0.0


class TestMacroAverages:
    """The denominator is the qrels, which is the honest accounting."""

    def test_the_macro_average_matches_the_worked_example(self) -> None:
        assert ndcg_at_k(QRELS, RUN, 3) == pytest.approx(0.19874476, abs=1e-8)
        assert mrr_at_k(QRELS, RUN, 3) == pytest.approx(0.25)

    def test_an_unanswered_judged_query_scores_zero_and_drags_the_average_down(
        self,
    ) -> None:
        """``q2`` returned nothing. Dropping it would inflate every number."""
        answered_only = ndcg_at_k({"q1": QRELS["q1"]}, RUN, 3)
        assert ndcg_at_k(QRELS, RUN, 3) == pytest.approx(answered_only / 2, abs=1e-8)

    def test_a_run_query_absent_from_the_qrels_cannot_contribute_either_way(self) -> None:
        polluted = {**RUN, "q_unjudged": {"d1": 1.0}}
        assert ndcg_at_k(QRELS, polluted, 3) == pytest.approx(ndcg_at_k(QRELS, RUN, 3))

    def test_skipping_positive_free_queries_changes_the_denominator_only(self) -> None:
        qrels = {**QRELS, "q3": {"dZ": 0}}
        kept = ndcg_at_k(qrels, RUN, 3)
        skipped = ndcg_at_k(qrels, RUN, 3, skip_queries_without_positives=True)
        assert skipped == pytest.approx(kept * 3 / 2, abs=1e-8)

    def test_an_empty_qrels_is_a_visible_zero_not_a_zero_division(self) -> None:
        assert ndcg_at_k({}, {}, 10) == 0.0
        assert recall_at_k({}, {}, 10) == 0.0
        assert map_at_k({}, {}, 10) == 0.0
        assert mrr_at_k({}, {}, 10) == 0.0


class TestEvaluateRun:
    """The flat ``<metric>_at_<k>`` block ``appsretrieval_results.json`` carries."""

    def test_it_emits_four_metrics_at_every_requested_cut_off(self) -> None:
        scores = evaluate_run(QRELS, RUN)
        for cut in DEFAULT_K_VALUES:
            assert {f"ndcg_at_{cut}", f"mrr_at_{cut}", f"recall_at_{cut}", f"map_at_{cut}"} <= set(
                scores
            )
        assert len(scores) == 4 * len(DEFAULT_K_VALUES)

    def test_the_keys_match_mteb_s_task_result_naming(self) -> None:
        """So the JSON is loadable by MTEB tooling whether or not MTEB ran."""
        scores = evaluate_run({"q1": {"d1": 1}}, {"q1": {"d1": 0.9}}, k_values=(1, 10))
        assert scores["ndcg_at_10"] == 1.0
        assert scores["recall_at_1"] == 1.0

    def test_every_score_is_a_fraction_in_the_unit_interval(self) -> None:
        """PRD.md section 2 quotes percentages; the conversion is a reporting-layer job."""
        for value in evaluate_run(QRELS, RUN).values():
            assert 0.0 <= value <= 1.0

    def test_duplicate_and_non_positive_cut_offs_are_collapsed(self) -> None:
        scores = evaluate_run(QRELS, RUN, k_values=(10, 10, 0, -3))
        assert set(scores) == {"ndcg_at_10", "mrr_at_10", "recall_at_10", "map_at_10"}


class TestRunCoverage:
    """A high NDCG over eleven of 3,765 queries is not a high NDCG."""

    def test_it_publishes_the_denominators_next_to_the_number(self) -> None:
        coverage = run_coverage(QRELS, RUN)
        assert coverage == {
            "qrels_queries": 2,
            "queries_with_positives": 2,
            "answered_queries": 1,
            "empty_result_queries": 1,
            "run_only_queries": 0,
            "retrieved_documents": 4,
        }

    def test_a_run_only_query_is_counted_and_named(self) -> None:
        coverage = run_coverage(QRELS, {**RUN, "qX": {"d1": 1.0}})
        assert coverage["run_only_queries"] == 1
        assert coverage["qrels_queries"] == 2


# ---------------------------------------------------------------------------
# Optional-dependency posture
# ---------------------------------------------------------------------------


class TestBareInstallPosture:
    """NFR-07: the metric must not vanish when a 400 MB dependency tree does not resolve."""

    @pytest.mark.smoke
    def test_importing_the_eval_package_pulls_in_neither_mteb_nor_datasets(self) -> None:
        script = (
            "import sys; import axiom.eval, axiom.eval.metrics, axiom.eval.mteb_adapter; "
            "print(int(any(m.split('.')[0] in {'mteb', 'datasets', 'torch'} "
            "for m in sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "0", "importing axiom.eval pulled in an optional extra"

    def test_a_missing_mteb_degrades_to_the_local_corpus_and_names_what_it_tried(
        self, monkeypatch: pytest.MonkeyPatch, settings: Any, tmp_path: Path
    ) -> None:
        """ "dataset not found" with no paths is the least actionable error a judge can hit."""
        from axiom.core.errors import IndexNotFoundError
        from axiom.eval import mteb_adapter

        def _no_mteb(*_args: Any, **_kwargs: Any) -> Any:
            raise ImportError("No module named 'mteb'")

        monkeypatch.setattr(mteb_adapter, "load_mteb_task", _no_mteb)
        monkeypatch.setattr(mteb_adapter, "DEFAULT_LOCAL_DATA_ROOT", tmp_path / "datasets")
        offline = settings.model_copy(update={"offline": False})
        with pytest.raises(IndexNotFoundError) as caught:
            mteb_adapter.resolve_task("AppsRetrieval", "test", settings=offline)
        message = str(caught.value)
        assert "AppsRetrieval" in message
        assert "corpus.jsonl" in message and "qrels/test.tsv" in message
        assert "mteb hub" in message

    def test_offline_mode_never_reaches_for_the_hub(
        self, monkeypatch: pytest.MonkeyPatch, settings: Any, tmp_path: Path
    ) -> None:
        """``AXIOM_OFFLINE=true`` must fail fast, not hang on a socket in front of a jury."""
        from axiom.core.errors import IndexNotFoundError
        from axiom.eval import mteb_adapter

        monkeypatch.setattr(
            mteb_adapter,
            "load_mteb_task",
            lambda *_a, **_kw: pytest.fail("offline mode reached the hub"),
        )
        monkeypatch.setattr(mteb_adapter, "DEFAULT_LOCAL_DATA_ROOT", tmp_path / "datasets")
        with pytest.raises(IndexNotFoundError, match="AXIOM_OFFLINE=true"):
            mteb_adapter.resolve_task(
                "AppsRetrieval", "test", settings=settings.model_copy(update={"offline": True})
            )


# ---------------------------------------------------------------------------
# The offline data path
# ---------------------------------------------------------------------------


class TestLocalTaskLoading:
    """A vendored BEIR-shaped directory, loaded with nothing but the stdlib."""

    def test_it_loads_the_three_files_and_records_the_revision(self, beir_dir: Path) -> None:
        from axiom.eval.mteb_adapter import load_local_task

        task = load_local_task(beir_dir, task_name="TinyCode", split="test")
        assert set(task.corpus) == {"d1", "d2", "d3", "d4"}
        assert set(task.queries) == {"q1", "q2"}
        assert task.qrels == {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
        assert task.dataset_revision == "fixture-rev-1"
        assert task.source.startswith("local:")

    def test_the_header_row_of_the_qrels_tsv_is_not_a_judgment(self, beir_dir: Path) -> None:
        from axiom.eval.mteb_adapter import load_local_task

        task = load_local_task(beir_dir, task_name="TinyCode", split="test")
        assert "query-id" not in task.qrels

    def test_a_missing_file_is_an_environment_failure_not_bad_input(self, beir_dir: Path) -> None:
        """Exit 3 territory: the dataset is not there, so there is nothing to degrade to."""
        from axiom.core.errors import IndexNotFoundError
        from axiom.eval.mteb_adapter import load_local_task

        (beir_dir / "queries.jsonl").unlink()
        with pytest.raises(IndexNotFoundError, match=r"queries\.jsonl"):
            load_local_task(beir_dir, task_name="TinyCode", split="test")

    def test_a_malformed_record_inside_a_present_file_is_skipped_loudly(
        self, beir_dir: Path
    ) -> None:
        """Rule 3: one bad line must not cost a two-hour eval."""
        from axiom.eval.mteb_adapter import load_local_task

        with (beir_dir / "corpus.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
            handle.write("[1, 2, 3]\n")
            handle.write(json.dumps({"title": "no id", "text": "x"}) + "\n")
        task = load_local_task(beir_dir, task_name="TinyCode", split="test")
        assert set(task.corpus) == {"d1", "d2", "d3", "d4"}

    def test_a_non_numeric_grade_is_dropped_rather_than_crashing_the_split(
        self, beir_dir: Path
    ) -> None:
        from axiom.eval.mteb_adapter import load_local_task

        with (beir_dir / "qrels" / "test.tsv").open("a", encoding="utf-8") as handle:
            handle.write("q1\td9\tvery-relevant\n")
        task = load_local_task(beir_dir, task_name="TinyCode", split="test")
        assert task.qrels["q1"] == {"d1": 1.0}

    def test_an_explicit_local_path_beats_the_vendored_default(
        self, beir_dir: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from axiom.eval import mteb_adapter

        monkeypatch.setattr(
            mteb_adapter,
            "load_mteb_task",
            lambda *_a, **_kw: pytest.fail("a local copy existed; the hub must not be tried"),
        )
        task = mteb_adapter.resolve_task("TinyCode", "test", local_path=beir_dir, settings=settings)
        assert task.source.startswith("local:")
        assert set(task.corpus) == {"d1", "d2", "d3", "d4"}


class TestNormalisers:
    """MTEB v2, MTEB v1 and our vendored JSONL all arrive here and leave in one shape."""

    def test_every_corpus_shape_collapses_to_the_same_mapping(self) -> None:
        from axiom.eval.mteb_adapter import normalise_corpus

        expected = {"d1": {"title": "t", "text": "x"}}
        assert normalise_corpus({"d1": {"title": "t", "text": "x"}}) == expected
        assert normalise_corpus([{"_id": "d1", "title": "t", "text": "x"}]) == expected
        assert normalise_corpus([{"id": "d1", "title": "t", "text": "x"}]) == expected
        assert normalise_corpus([{"doc_id": "d1", "title": "t", "text": "x"}]) == expected

    def test_a_record_with_no_usable_id_is_dropped_not_fatal(self) -> None:
        from axiom.eval.mteb_adapter import normalise_corpus

        assert normalise_corpus([{"title": "t"}, "junk", 7]) == {}

    def test_a_conversational_query_is_joined_rather_than_stringified(self) -> None:
        """``str(list)`` would embed Python quoting into the query text."""
        from axiom.eval.mteb_adapter import normalise_queries

        assert normalise_queries({"q1": ["turn one", "turn two"]}) == {"q1": "turn one\nturn two"}
        assert normalise_queries([{"_id": "q1", "text": "plain"}]) == {"q1": "plain"}
        assert normalise_queries({"q1": {"text": "nested"}}) == {"q1": "nested"}

    def test_code_tokens_keep_the_whole_identifier_beside_its_parts(self) -> None:
        """Splitting alone loses the exact-identifier match FR-07 depends on."""
        from axiom.eval.mteb_adapter import code_tokens

        tokens = code_tokens("preprocessInput")
        assert "preprocess" in tokens and "input" in tokens
        assert "preprocessinput" in tokens

    def test_snake_case_and_punctuation_split_the_same_way(self) -> None:
        from axiom.eval.mteb_adapter import code_tokens

        assert code_tokens("resolve_tool(x)") == ["resolve", "tool", "x"]


class TestCorpusToChunks:
    """Rules.md AP-01/AP-13: the id map is explicit, never string surgery."""

    def test_one_document_becomes_one_chunk_in_document_id_order(self) -> None:
        from axiom.eval.mteb_adapter import corpus_to_chunks

        corpus = {
            "dB": {"title": "", "text": "def b(): pass"},
            "dA": {"title": "", "text": "def a(): pass"},
        }
        chunks, id_map = corpus_to_chunks(corpus)
        assert [id_map[chunk.chunk_id] for chunk in chunks] == ["dA", "dB"]
        assert len(id_map) == 2

    def test_an_empty_document_is_skipped_because_it_is_unretrievable(self) -> None:
        from axiom.eval.mteb_adapter import corpus_to_chunks

        chunks, id_map = corpus_to_chunks({"d1": {"title": "", "text": "   "}})
        assert chunks == [] and id_map == {}

    def test_byte_identical_documents_under_distinct_ids_both_survive(self) -> None:
        """The synthetic path carries the document id, so the digests differ.

        Worth pinning because the module's own comment describes the opposite
        ("two corpus documents with byte-identical text ... collide on
        chunk_id"). They collide only when their ids *also* sanitise to the same
        synthetic filename -- the case the next test covers -- and every other
        duplicate is retrievable under its own id, which is what the qrels need.
        """
        from axiom.eval.mteb_adapter import corpus_to_chunks

        body = {"title": "", "text": "def same(): pass"}
        chunks, id_map = corpus_to_chunks({"d1": dict(body), "d2": dict(body)})
        assert len(chunks) == 2
        assert set(id_map.values()) == {"d1", "d2"}

    def test_two_ids_that_sanitise_to_one_filename_drop_a_judged_document(self) -> None:
        """``a/b`` and ``a_b`` both become ``corpus/a_b.py``.

        With identical text that makes one ``chunk_id`` for two documents, and
        the second is dropped. Rule 1 forbids inventing a new id, so dropping is
        the right call -- but it silently makes the dropped document's qrels
        unsatisfiable, so the warning is the only thing standing between this and
        an unexplained recall ceiling. Asserted here so it cannot go quiet.
        """
        from axiom.eval.mteb_adapter import corpus_to_chunks

        body = {"title": "", "text": "def same(): pass"}
        chunks, id_map = corpus_to_chunks({"a/b": dict(body), "a_b": dict(body)})
        assert len(chunks) == 1
        assert len(id_map) == 1
        assert set(id_map.values()) <= {"a/b", "a_b"}

    def test_a_document_id_with_path_characters_cannot_escape_the_synthetic_path(
        self,
    ) -> None:
        from pathlib import PurePosixPath

        from axiom.eval.mteb_adapter import corpus_to_chunks

        chunks, _ = corpus_to_chunks({"../../etc/passwd": {"title": "", "text": "x = 1"}})
        path = PurePosixPath(chunks[0].location.file_path)
        assert path.parts[0] == "corpus"
        assert len(path.parts) == 2, "the id must not introduce a path separator"
        assert ".." not in path.parts, "no component may traverse upwards"

    def test_the_title_is_joined_with_a_hard_break_not_a_space(self) -> None:
        """Otherwise a title fuses into the first line of code and makes a token
        that exists in neither."""
        from axiom.eval.mteb_adapter import corpus_to_chunks

        chunks, _ = corpus_to_chunks({"d1": {"title": "Title", "text": "def f(): pass"}})
        assert chunks[0].text == "Title\n\ndef f(): pass"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class TestLexicalFallback:
    """A wiring proof, never a result."""

    def test_it_can_never_be_constructed_and_claimed_non_degraded(self) -> None:
        from axiom.eval.mteb_adapter import LexicalBackend

        backend = LexicalBackend()
        assert LexicalBackend.degraded is True
        assert backend.degraded is True
        with pytest.raises(TypeError):
            LexicalBackend(degraded=False)  # type: ignore[call-arg]

    def test_it_retrieves_the_obvious_document_with_zero_dependencies(self) -> None:
        from axiom.eval.mteb_adapter import LexicalBackend, corpus_to_chunks

        chunks, id_map = corpus_to_chunks(
            {
                "d1": {"title": "", "text": "def binary_search(array, target): pass"},
                "d2": {"title": "", "text": "def quicksort(array): pass"},
            }
        )
        backend = LexicalBackend()
        backend.index(chunks)
        hits = backend.search("binary search", 5)
        assert hits
        assert id_map[hits[0][0]] == "d1"

    def test_an_unindexed_backend_returns_nothing_rather_than_raising(self) -> None:
        from axiom.eval.mteb_adapter import LexicalBackend

        assert LexicalBackend().search("anything", 10) == []

    def test_results_are_ordered_and_truncated(self) -> None:
        from axiom.eval.mteb_adapter import LexicalBackend, corpus_to_chunks

        chunks, _ = corpus_to_chunks(
            {f"d{i}": {"title": "", "text": f"def f{i}(array): pass"} for i in range(5)}
        )
        backend = LexicalBackend()
        backend.index(chunks)
        hits = backend.search("array", 3)
        assert len(hits) <= 3
        assert [score for _, score in hits] == sorted((score for _, score in hits), reverse=True)


class TestBackendResolution:
    """A typo'd ``--backend`` at hour two of a full-split run must not be fatal."""

    @pytest.mark.parametrize("spec", [None, "", "  ", "lexical", "FALLBACK"])
    def test_the_documented_aliases_select_the_fallback_directly(
        self, spec: str | None, settings: Any
    ) -> None:
        from axiom.eval.mteb_adapter import LexicalBackend, resolve_backend

        assert isinstance(resolve_backend(spec, settings), LexicalBackend)

    @pytest.mark.parametrize(
        "spec",
        [
            "axiom_no_such_module:factory",
            "axiom.eval.mteb_adapter:no_such_attribute",
            "axiom.eval.mteb_adapter:code_tokens",
        ],
    )
    def test_an_unusable_spec_degrades_loudly_instead_of_dying(
        self, spec: str, settings: Any, axiom_caplog: pytest.LogCaptureFixture
    ) -> None:
        from axiom.eval.mteb_adapter import LexicalBackend, resolve_backend

        with axiom_caplog.at_level("WARNING"):
            backend = resolve_backend(spec, settings)
        assert isinstance(backend, LexicalBackend)
        assert any(
            "lexical BM25 fallback" in record.message for record in axiom_caplog.records
        )

    def test_a_factory_is_given_settings_only_when_it_names_the_parameter(
        self, settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arity is the wrong test: a dataclass whose first field is ``name``
        accepts one positional argument and binds ``Settings`` to it silently."""
        from axiom.eval import mteb_adapter

        seen: dict[str, Any] = {}

        def _wants_settings(settings: Any) -> Any:
            seen["settings"] = settings
            return mteb_adapter.LexicalBackend()

        def _wants_nothing() -> Any:
            seen["settings"] = "not passed"
            return mteb_adapter.LexicalBackend()

        monkeypatch.setattr(mteb_adapter, "_test_factory_a", _wants_settings, raising=False)
        monkeypatch.setattr(mteb_adapter, "_test_factory_b", _wants_nothing, raising=False)

        mteb_adapter.resolve_backend("axiom.eval.mteb_adapter:_test_factory_a", settings)
        assert seen["settings"] is settings
        mteb_adapter.resolve_backend("axiom.eval.mteb_adapter:_test_factory_b", settings)
        assert seen["settings"] == "not passed"

    def test_the_callable_adapter_re_sorts_whatever_the_pipeline_returned(self) -> None:
        """Rule 4 enforced at the seam, because the injected pipeline is not ours."""
        from axiom.eval.mteb_adapter import CallableBackend

        backend = CallableBackend(
            index_fn=lambda _chunks: None,
            search_fn=lambda _q, _k: [("bbb", 0.1), ("aaa", 0.9), ("ccc", 0.9)],
        )
        assert backend.search("q", 3) == [("aaa", 0.9), ("ccc", 0.9), ("bbb", 0.1)]
        assert backend.search("q", 1) == [("aaa", 0.9)]


class TestEncoderBackend:
    """Exact cosine, because an ANN index would fold its recall loss into the number."""

    def test_it_ranks_by_cosine_against_a_stub_encoder(self) -> None:
        numpy = pytest.importorskip("numpy")
        from axiom.eval.mteb_adapter import EncoderBackend, corpus_to_chunks

        chunks, id_map = corpus_to_chunks(
            {
                "dA": {"title": "", "text": "alpha alpha"},
                "dB": {"title": "", "text": "beta beta"},
            }
        )
        order = {chunk.text: index for index, chunk in enumerate(chunks)}

        class _Stub:
            name = "stub"

            @staticmethod
            def encode(texts: list[str]) -> Any:
                rows = []
                for text in texts:
                    if "alpha" in text:
                        rows.append([1.0, 0.0])
                    elif "beta" in text:
                        rows.append([0.0, 1.0])
                    else:
                        rows.append([0.0, 0.0])
                return numpy.asarray(rows, dtype="float32")

        backend = EncoderBackend(encoder=_Stub())
        backend.index(chunks)
        hits = backend.search("alpha", 2)
        assert id_map[hits[0][0]] == "dA"
        assert order  # the chunk order is what the matrix rows correspond to

    def test_it_reports_the_wrapped_embedder_s_own_degradation(self) -> None:
        from axiom.eval.mteb_adapter import EncoderBackend

        class _Honest:
            degraded = True

            @staticmethod
            def encode(texts: list[str]) -> Any:  # pragma: no cover - never called
                raise AssertionError

        class _Silent:
            @staticmethod
            def encode(texts: list[str]) -> Any:  # pragma: no cover - never called
                raise AssertionError

        assert EncoderBackend(encoder=_Honest()).degraded is True
        # Absence of the flag must mean "no claim made", never "not degraded".
        assert EncoderBackend(encoder=_Silent()).degraded is False

    def test_an_empty_corpus_searches_to_nothing(self) -> None:
        pytest.importorskip("numpy")
        from axiom.eval.mteb_adapter import EncoderBackend

        class _Stub:
            @staticmethod
            def encode(texts: list[str]) -> Any:  # pragma: no cover - never called
                raise AssertionError

        backend = EncoderBackend(encoder=_Stub())
        backend.index([])
        assert backend.search("q", 5) == []


# ---------------------------------------------------------------------------
# The id fence
# ---------------------------------------------------------------------------


class TestSearchModelIdFence:
    """TechSpecifications.md section 4.10: here a contract violation must raise.

    A mangled id does not fail -- MTEB scores it as a miss, so the harness being
    broken reads as the model being bad. That is why this is the one place Rule
    3's "degrade, never raise" is deliberately inverted.
    """

    @staticmethod
    def _model(hits: list[tuple[str, float]], settings: Any) -> Any:
        from axiom.eval.mteb_adapter import AxiomSearchModel, CallableBackend

        backend = CallableBackend(
            index_fn=lambda _chunks: None,
            search_fn=lambda _q, _k: hits,
            name="stub",
        )
        return AxiomSearchModel(backend=backend, settings=settings)

    def test_an_invented_chunk_id_raises_rather_than_scoring_zero(self, settings: Any) -> None:
        from axiom.core.errors import AxiomContractError

        model = self._model([("f" * 32, 0.9)], settings)
        model.index({"d1": {"title": "", "text": "def f(): pass"}})
        with pytest.raises(AxiomContractError, match="not in the corpus id map"):
            model.search({"q1": "anything"})

    def test_a_legitimate_chunk_id_maps_back_to_its_document(self, settings: Any) -> None:
        from axiom.eval.mteb_adapter import AxiomSearchModel, CallableBackend, corpus_to_chunks

        corpus = {"d1": {"title": "", "text": "def f(): pass"}}
        chunks, _ = corpus_to_chunks(corpus)
        chunk_id = chunks[0].chunk_id
        backend = CallableBackend(
            index_fn=lambda _chunks: None,
            search_fn=lambda _q, _k: [(chunk_id, 0.75)],
            name="stub",
        )
        model = AxiomSearchModel(backend=backend, settings=settings)
        model.index(corpus)
        assert model.indexed_count == 1
        assert model.search({"q1": "anything"}) == {"q1": {"d1": 0.75}}

    def test_an_empty_query_costs_only_its_own_result_set(self, settings: Any) -> None:
        """Rules.md AP-11: bad input mid-run must not lose the other 3,764 answers."""
        from axiom.eval.mteb_adapter import AxiomSearchModel, CallableBackend, corpus_to_chunks

        corpus = {"d1": {"title": "", "text": "def f(): pass"}}
        chunks, _ = corpus_to_chunks(corpus)
        chunk_id = chunks[0].chunk_id
        backend = CallableBackend(
            index_fn=lambda _chunks: None,
            search_fn=lambda _q, _k: [(chunk_id, 0.5)],
            name="stub",
        )
        model = AxiomSearchModel(backend=backend, settings=settings)
        model.index(corpus)
        run = model.search({"q_blank": "   ", "q_real": "anything"})
        assert run["q_blank"] == {}
        assert run["q_real"] == {"d1": 0.5}

    def test_a_reranking_restriction_is_applied_rather_than_ignored(self, settings: Any) -> None:
        from axiom.eval.mteb_adapter import AxiomSearchModel, CallableBackend, corpus_to_chunks

        corpus = {
            "d1": {"title": "", "text": "def one(): pass"},
            "d2": {"title": "", "text": "def two(): pass"},
        }
        chunks, id_map = corpus_to_chunks(corpus)
        hits = [(chunk.chunk_id, 0.5) for chunk in chunks]
        backend = CallableBackend(
            index_fn=lambda _chunks: None, search_fn=lambda _q, _k: hits, name="stub"
        )
        model = AxiomSearchModel(backend=backend, settings=settings)
        model.index(corpus)
        assert id_map  # both documents are in the map
        run = model.search({"q1": "x"}, top_ranked={"q1": ["d2"]})
        assert set(run["q1"]) == {"d2"}

    def test_the_degraded_flag_is_carried_up_from_the_backend(self, settings: Any) -> None:
        """``scripts/run_eval.py`` turns this into ``reportable: false``."""
        from axiom.eval.mteb_adapter import AxiomSearchModel, LexicalBackend, build_search_model

        model = AxiomSearchModel(backend=LexicalBackend(), settings=settings)
        assert model.degraded is True
        assert model.backend_detail["degraded"] is True
        assert build_search_model(settings, backend_spec="lexical").degraded is True

    def test_the_backend_detail_names_the_embedder_underneath(self, settings: Any) -> None:
        """ "encoder-exact-cosine" alone cannot tell a Qwen3 run from a hash fallback."""
        from axiom.eval.mteb_adapter import AxiomSearchModel, EncoderBackend

        class _Named:
            model_identity = "axiom/hash-embedder-v1"
            degraded = True

            @staticmethod
            def encode(texts: list[str]) -> Any:  # pragma: no cover - never called
                raise AssertionError

        model = AxiomSearchModel(backend=EncoderBackend(encoder=_Named()), settings=settings)
        assert model.backend_detail["encoder"] == "axiom/hash-embedder-v1"
        assert model.backend_detail["encoder_degraded"] is True


class TestAxiomEncoder:
    """The dense-only ablation wrapper, and its floor."""

    def test_with_no_embedder_resolvable_it_still_produces_seeded_vectors(
        self, settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        numpy = pytest.importorskip("numpy")
        from axiom.eval import mteb_adapter

        monkeypatch.setattr(mteb_adapter, "_resolve_axiom_embedder", lambda _s: None)
        encoder = mteb_adapter.AxiomEncoder(settings)
        assert encoder.degraded is True
        assert encoder.model_identity == "axiom/hashed-tokens-fallback"
        first = encoder.encode(["def preprocessInput(x): pass"])
        second = encoder.encode(["def preprocessInput(x): pass"])
        assert numpy.array_equal(first, second), "NFR-08: the fallback must be seeded"
        assert first.shape == (1, max(8, settings.embedding_dim))
        assert float(numpy.linalg.norm(first[0])) == pytest.approx(1.0, abs=1e-5)

    def test_mteb_s_batched_dataloader_shape_cannot_change_our_vectors(
        self, settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        numpy = pytest.importorskip("numpy")
        from axiom.eval import mteb_adapter

        monkeypatch.setattr(mteb_adapter, "_resolve_axiom_embedder", lambda _s: None)
        encoder = mteb_adapter.AxiomEncoder(settings)
        plain = encoder.encode(["alpha", "beta"])
        batched = encoder.encode([{"text": ["alpha", "beta"]}])
        assert numpy.allclose(plain, batched)

    def test_an_injected_embedder_is_used_and_its_degradation_believed(self, settings: Any) -> None:
        pytest.importorskip("numpy")
        from axiom.eval.mteb_adapter import AxiomEncoder

        class _Stub:
            model_id = "stub/embedder"
            degraded = False

            @staticmethod
            def encode(texts: list[str]) -> Any:
                return [[1.0, 0.0] for _ in texts]

        encoder = AxiomEncoder(settings, embedder=_Stub())
        assert encoder.degraded is False
        assert encoder.model_identity == "stub/embedder"
        assert encoder.encode(["x"]).shape == (1, 2)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTheHarnessRunsEndToEnd:
    """``scripts/run_eval.py`` over a vendored corpus, with no optional extras."""

    @staticmethod
    def _load_script() -> Any:
        import importlib.util

        path = REPO_ROOT / "scripts" / "run_eval.py"
        spec = importlib.util.spec_from_file_location("axiom_run_eval_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_smoke_run_produces_a_scored_results_file_stamped_non_reportable(
        self, beir_dir: Path, tmp_path: Path
    ) -> None:
        script = self._load_script()
        out = tmp_path / "results.json"
        predictions = tmp_path / "preds.json"
        code = script.main(
            [
                "--task",
                "TinyCode",
                "--split",
                "test",
                "--corpus-path",
                str(beir_dir),
                "--out",
                str(out),
                "--predictions",
                str(predictions),
                "--json",
            ]
        )
        assert code == 0
        body = json.loads(out.read_text(encoding="utf-8"))
        assert body["task_name"] == "TinyCode"
        assert body["dataset_revision"] == "fixture-rev-1"
        block = body["scores"]["test"][0]
        assert 0.0 <= block["ndcg_at_10"] <= 1.0
        assert block["main_score"] == block["ndcg_at_10"]

        provenance = body["axiom_provenance"]
        assert provenance["reportable"] is False
        reasons = " ".join(provenance["non_reportable_because"])
        assert "degraded retrieval backend" in reasons
        assert provenance["backend"] == "lexical-bm25-fallback"
        assert provenance["coverage"]["qrels_queries"] == 2

    def test_the_predictions_file_lets_the_number_be_re_derived_without_retrieving(
        self, beir_dir: Path, tmp_path: Path
    ) -> None:
        """TestPlan.md section 6.4: a results row must be reproducible from its file."""
        script = self._load_script()
        out = tmp_path / "results.json"
        predictions = tmp_path / "preds.json"
        script.main(
            [
                "--task",
                "TinyCode",
                "--split",
                "test",
                "--corpus-path",
                str(beir_dir),
                "--out",
                str(out),
                "--predictions",
                str(predictions),
                "--json",
            ]
        )
        run = json.loads(predictions.read_text(encoding="utf-8"))["test"]
        qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
        reported = json.loads(out.read_text(encoding="utf-8"))["scores"]["test"][0]
        assert ndcg_at_k(qrels, run, 10) == pytest.approx(reported["ndcg_at_10"], abs=1e-9)
        assert recall_at_k(qrels, run, 100) == pytest.approx(reported["recall_at_100"], abs=1e-9)

    def test_a_limited_run_is_stamped_a_different_task_not_a_cheaper_estimate(
        self, beir_dir: Path, tmp_path: Path
    ) -> None:
        """Fewer distractors inflate NDCG mechanically (TestPlan.md section 6.2)."""
        script = self._load_script()
        out = tmp_path / "results.json"
        script.main(
            [
                "--task",
                "TinyCode",
                "--split",
                "test",
                "--corpus-path",
                str(beir_dir),
                "--limit",
                "1",
                "--out",
                str(out),
                "--json",
            ]
        )
        provenance = json.loads(out.read_text(encoding="utf-8"))["axiom_provenance"]
        assert provenance["mode"] == "SMOKE"
        assert provenance["reportable"] is False
        assert any("full test-split" in reason for reason in provenance["non_reportable_because"])

    def test_a_non_test_split_can_never_be_reportable(self, beir_dir: Path, tmp_path: Path) -> None:
        """NG-29's fence: only the untouched test split may produce a quoted number."""
        (beir_dir / "qrels" / "dev.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8"
        )
        script = self._load_script()
        out = tmp_path / "results.json"
        script.main(
            [
                "--task",
                "TinyCode",
                "--split",
                "dev",
                "--corpus-path",
                str(beir_dir),
                "--out",
                str(out),
                "--json",
            ]
        )
        provenance = json.loads(out.read_text(encoding="utf-8"))["axiom_provenance"]
        assert provenance["reportable"] is False
        assert provenance["mode"] == "SMOKE"

    def test_limiting_keeps_the_qrels_internally_coherent(self, beir_dir: Path) -> None:
        from axiom.eval.mteb_adapter import load_local_task

        task = load_local_task(beir_dir, task_name="TinyCode", split="test")
        limited = task.limited(1)
        assert set(limited.queries) == {"q1"}
        assert "limit=1" in limited.source
        for query_id, judgments in limited.qrels.items():
            assert query_id in limited.queries
            for doc_id in judgments:
                assert doc_id in limited.corpus, "a judged positive fell out of the corpus"

    def test_limiting_by_none_or_zero_is_a_no_op(self, beir_dir: Path) -> None:
        from axiom.eval.mteb_adapter import load_local_task

        task = load_local_task(beir_dir, task_name="TinyCode", split="test")
        assert task.limited(None) is task
        assert task.limited(0) is task
