"""Structural retrieval: the signal nothing else can produce.

Dense retrieval answers "what does this code mean?" and sparse answers "where
does this token appear?". Neither can answer *"which files call tool XYZ before
tool ABC?"* -- the relative position of two call sites inside one function body
survives neither an embedding nor a bag of words (PRD.md section 1.1, archetype
Q2). It survives a table with an ordinal column, and that table is the project's
innovation claim, so :meth:`~axiom.retrieval.structural.StructuralRetriever.ordered_call_pair`
gets the densest coverage in this file.

The second thing asserted throughout is FR-10: an unresolvable identifier, a
missing index or a corrupt database produce an **empty list**, never an
exception. Fusion then drops the structural weight and renormalises the other
two, which is strictly better for the user than a 500.

Only ``sqlite3`` is involved, so nothing here is skipped on a bare install --
the structural signal is the one signal a missing wheel cannot disable.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from axiom.agent.planner import build_plan
from axiom.chunking import chunk_repo
from axiom.config import get_settings
from axiom.indexing.structural import INDEX_FILENAME, build_structural_index, split_callee
from axiom.retrieval.structural import (
    StructuralEvidence,
    StructuralRetriever,
    mine_identifiers,
    normalise_identifier,
)
from axiom.schema import Chunk, ChunkKind, ChunkLocation, ChunkMetadata, QueryType, SignalKind

from .fakes import make_plan
from .helpers import assert_ranked_list

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(scope="module")
def settings():
    return get_settings("default", configs_dir=CONFIGS, llm_enabled=False)


@pytest.fixture(scope="module")
def graph(repo_v1: Path, settings, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real ``structural.sqlite`` over the fixture repo.

    Module-scoped and read-only: it is built once from committed source, and no
    test in this file writes to it. Tests that need a *broken* index build their
    own under ``tmp_path``.
    """
    out = tmp_path_factory.mktemp("graph")
    chunks = chunk_repo(repo_v1, "v1", settings)
    stats = build_structural_index(chunks, settings, out)
    assert stats.skipped is False
    assert stats.symbol_count > 0 and stats.call_count > 0
    return out


@pytest.fixture
def retriever(graph: Path, settings):
    instance = StructuralRetriever(graph, settings)
    yield instance
    instance.close()


def _synthetic_chunks(version_id: str = "v1") -> list[Chunk]:
    """Chunks with hand-set metadata, for the edges the regex rung cannot mine.

    The AST rung detects exports; the regex rung (the one a bare install runs)
    does not. Rather than skip the export tests, they drive the *retriever* --
    which is what is under test -- from metadata built here directly.
    """

    def build(text: str, path: str, line: int, **meta) -> Chunk:
        fields = {"kind": ChunkKind.FUNCTION, "version_id": version_id}
        fields.update(meta)
        return Chunk.create(
            text,
            ChunkLocation(
                file_path=path,
                start_line=line,
                end_line=line + text.count("\n"),
                start_byte=0,
                end_byte=len(text.encode("utf-8")),
            ),
            ChunkMetadata(**fields),  # type: ignore[arg-type]
        )

    return [
        build(
            "export function publicHelper(x) {\n  return normalise(x);\n}",
            "src/lib/helper.js",
            1,
            symbol="publicHelper",
            is_exported=True,
            calls=["normalise"],
        ),
        build(
            "import { publicHelper } from './lib/helper.js';\nexport function consumer(y) {\n"
            "  return publicHelper(y);\n}",
            "src/consumer.js",
            1,
            symbol="consumer",
            is_exported=True,
            imports=["./lib/helper.js"],
            calls=["publicHelper"],
        ),
    ]


# ---------------------------------------------------------------------------
# The ordered-call-pair predicate -- archetype Q2
# ---------------------------------------------------------------------------


class TestOrderedCallPair:
    def test_the_predicate_finds_the_chunk_that_calls_x_before_y(self, retriever) -> None:
        """``registry.dispatch`` calls ``preprocessInput`` then ``resolveTool``.

        The returned detail carries both ordinals, which is what
        ``RetrievalResult.match_reason`` renders as the FR-14 explanation.
        """
        hits = retriever.ordered_call_pair("preprocessInput", "resolveTool", 10)
        assert hits, "the ordered pair in registry.js must be found"
        assert all(hit.evidence is StructuralEvidence.ORDERED_PAIR for hit in hits)
        assert all("before resolveTool" in hit.detail for hit in hits)
        assert all("ordinal" in hit.detail for hit in hits)

    def test_the_predicate_is_directional(self, retriever) -> None:
        """The whole claim: "X before Y" is not "Y before X".

        A signal that answered both identically would be a bag of words with
        extra steps -- it would not be answering an ordering question at all.
        """
        forward = {
            hit.chunk_id
            for hit in retriever.ordered_call_pair("preprocessInput", "resolveTool", 10)
        }
        backward = {
            hit.chunk_id
            for hit in retriever.ordered_call_pair("resolveTool", "preprocessInput", 10)
        }
        assert forward, "the forward direction must match something"
        assert not (forward & backward), "a chunk cannot satisfy both directions here"

    def test_a_chunk_that_calls_only_one_of_the_pair_does_not_match(self, retriever) -> None:
        """``normalize.js`` defines ``preprocessInput`` and never calls
        ``resolveTool``, so it must not appear."""
        pair_ids = {
            hit.chunk_id
            for hit in retriever.ordered_call_pair("preprocessInput", "resolveTool", 10)
        }
        caller_ids = {hit.chunk_id for hit in retriever.callers_of("preprocessInput", 50)}
        assert pair_ids < caller_ids, "the pair predicate must be strictly narrower"

    def test_the_predicate_is_existential_not_universal(self, tmp_path: Path, settings) -> None:
        """``connect(); send(); connect()`` must still match "connect before send".

        The universal reading (``MAX(first) < MIN(second)``) rejects the retry
        and cleanup shape, which is extremely common. The implemented reading is
        ``MIN(first) < MAX(second)``, and this fixture is the case that
        distinguishes them.
        """
        text = "function retry() {\n  connect();\n  send();\n  connect();\n}"
        chunk = Chunk.create(
            text,
            ChunkLocation(
                file_path="src/retry.js",
                start_line=1,
                end_line=5,
                start_byte=0,
                end_byte=len(text.encode("utf-8")),
            ),
            ChunkMetadata(
                symbol="retry",
                kind=ChunkKind.FUNCTION,
                version_id="v1",
                calls=["connect", "send", "connect"],
            ),
        )
        out = tmp_path / "retry"
        out.mkdir()
        build_structural_index([chunk], settings, out)
        with StructuralRetriever(out, settings) as retriever:
            assert retriever.ordered_call_pair("connect", "send", 10), (
                "the existential reading must accept the retry shape"
            )

    def test_calls_x_before_x_means_calls_it_twice(self, tmp_path: Path, settings) -> None:
        """``MIN < MAX`` requires two distinct ordinals -- the honest reading of
        "calls X before X", left in deliberately."""

        def chunk(symbol: str, calls: list[str]) -> Chunk:
            text = f"function {symbol}() {{ {' '.join(c + '();' for c in calls)} }}"
            return Chunk.create(
                text,
                ChunkLocation(
                    file_path=f"src/{symbol}.js",
                    start_line=1,
                    end_line=1,
                    start_byte=0,
                    end_byte=len(text.encode("utf-8")),
                ),
                ChunkMetadata(symbol=symbol, kind=ChunkKind.FUNCTION, version_id="v1", calls=calls),
            )

        out = tmp_path / "twice"
        out.mkdir()
        build_structural_index(
            [chunk("once", ["ping"]), chunk("twice", ["ping", "ping"])], settings, out
        )
        with StructuralRetriever(out, settings) as retriever:
            hits = {hit.chunk_id for hit in retriever.ordered_call_pair("ping", "ping", 10)}
            assert len(hits) == 1

    def test_an_unknown_identifier_pair_returns_empty(self, retriever) -> None:
        assert retriever.ordered_call_pair("zzNotACall", "alsoNotACall", 10) == []

    def test_an_empty_identifier_returns_empty_rather_than_raising(self, retriever) -> None:
        """FR-10: an empty identifier is bad input, and bad input degrades."""
        assert retriever.ordered_call_pair("", "resolveTool", 10) == []
        assert retriever.ordered_call_pair("preprocessInput", "", 10) == []
        assert retriever.ordered_call_pair("", "", 10) == []

    def test_the_limit_is_honoured(self, retriever) -> None:
        assert len(retriever.ordered_call_pair("preprocessInput", "resolveTool", 1)) <= 1


# ---------------------------------------------------------------------------
# The other four query shapes
# ---------------------------------------------------------------------------


class TestGraphQueries:
    def test_callers_of_finds_every_call_site(self, retriever) -> None:
        hits = retriever.callers_of("preprocessInput", 50)
        assert hits
        assert all(hit.evidence is StructuralEvidence.CALLER for hit in hits)

    def test_callees_of_walks_the_other_direction(self, retriever) -> None:
        hits = retriever.callees_of("dispatch", 50)
        assert all(hit.evidence is StructuralEvidence.CALLEE for hit in hits)

    def test_definitions_of_finds_the_declaring_chunk(self, retriever) -> None:
        hits = retriever.definitions_of("resolveTool", 10)
        assert len(hits) >= 1
        assert all(hit.evidence is StructuralEvidence.DEFINITION for hit in hits)

    def test_imports_of_resolves_a_relative_module_specifier(self, retriever) -> None:
        """``registry.js`` and ``main.js`` both require ``../utils/normalize.js``."""
        hits = retriever.imports_of("normalize", 50)
        assert hits, "a relative import specifier must resolve to its target module"
        assert all(hit.evidence is StructuralEvidence.IMPORT for hit in hits)

    def test_exports_of_reads_the_is_exported_flag(self, tmp_path: Path, settings) -> None:
        """Driven from hand-set metadata: the regex chunking rung a bare install
        uses does not populate ``is_exported``, but the *retriever* must read it
        when the AST rung does."""
        out = tmp_path / "exports"
        out.mkdir()
        stats = build_structural_index(_synthetic_chunks(), settings, out)
        assert stats.export_count >= 2
        with StructuralRetriever(out, settings) as retriever:
            hits = retriever.exports_of("helper", 10)
            assert hits
            assert all(hit.evidence is StructuralEvidence.EXPORT for hit in hits)

    def test_substring_match_is_the_last_rung(self, retriever) -> None:
        """Rung 2 of the declared ladder: exact lookups found nothing."""
        hits = retriever.substring_match("Deeplink", 20)
        assert hits
        assert all(hit.evidence is StructuralEvidence.SUBSTRING for hit in hits)

    @pytest.mark.parametrize(
        "method",
        [
            "callers_of",
            "callees_of",
            "definitions_of",
            "imports_of",
            "exports_of",
            "substring_match",
        ],
    )
    def test_every_shape_returns_empty_for_an_unresolvable_identifier(
        self, retriever, method: str
    ) -> None:
        """FR-10, exhaustively: no shape raises, all return ``[]``."""
        assert getattr(retriever, method)("zzqqxx_not_a_symbol", 10) == []

    @pytest.mark.parametrize(
        "method",
        [
            "callers_of",
            "callees_of",
            "definitions_of",
            "imports_of",
            "exports_of",
            "substring_match",
        ],
    )
    def test_every_shape_survives_an_empty_identifier(self, retriever, method: str) -> None:
        assert getattr(retriever, method)("", 10) == []


# ---------------------------------------------------------------------------
# search / search_with_evidence -- what fusion and the formatter consume
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_a_well_formed_ranked_list(self, retriever, settings) -> None:
        plan = build_plan("Which files call preprocessInput before resolveTool?", settings)
        assert plan.query_type is QueryType.STRUCTURAL
        results = retriever.search(plan.original_query, plan)
        assert results
        assert_ranked_list(results, expected_signal=SignalKind.STRUCTURAL)

    def test_the_ordered_pair_outranks_a_bare_caller(self, retriever, settings) -> None:
        """Evidence priority *is* the score tier: for a STRUCTURAL query the
        ordered pair sits at position 0 of ``_EVIDENCE_PRIORITY`` and therefore
        scores above a plain caller edge."""
        plan = build_plan("Which files call preprocessInput before resolveTool?", settings)
        results = retriever.search(plan.original_query, plan)
        pair_ids = {
            hit.chunk_id
            for hit in retriever.ordered_call_pair("preprocessInput", "resolveTool", 50)
        }
        by_id = {entry.chunk_id: entry for entry in results}

        pair_ranks = [by_id[cid].rank for cid in pair_ids if cid in by_id]
        other_ranks = [entry.rank for entry in results if entry.chunk_id not in pair_ids]
        assert pair_ranks, "the ordered-pair chunks must appear in the ranked list"
        assert max(pair_ranks) < min(other_ranks, default=10**6)

    def test_evidence_survives_the_return(self, retriever, settings) -> None:
        """``ScoredChunk`` is frozen with ``extra="forbid"``, so the explanation
        has nowhere to ride -- it comes back alongside, and FR-14's
        "structural: calls preprocessInput before resolveTool" is only sayable
        because of that."""
        plan = build_plan("Which files call preprocessInput before resolveTool?", settings)
        results, evidence = retriever.search_with_evidence(plan.original_query, plan)
        assert results
        assert set(evidence) >= {entry.chunk_id for entry in results[:1]}
        assert any("before resolveTool" in detail for detail in evidence.values())

    def test_a_query_with_no_identifiers_returns_empty(self, retriever, settings) -> None:
        """FR-10 at the entry point: nothing to look up is not an error."""
        plan = make_plan("the the the", query_type=QueryType.SEMANTIC)
        results, evidence = retriever.search_with_evidence("the the the", plan)
        assert results == [] and evidence == {}

    def test_identifiers_are_mined_from_the_text_when_the_plan_carries_none(
        self, retriever
    ) -> None:
        """Declared degradation: a plan without ``extracted_identifiers`` still
        gets a structural signal rather than silently contributing nothing."""
        plan = make_plan("resolveTool", query_type=QueryType.STRUCTURAL)
        assert plan.extracted_identifiers == []
        results, _ = retriever.search_with_evidence("resolveTool", plan)
        assert results, "identifiers must be mined from the query text"

    def test_the_width_is_clamped(self, retriever, settings) -> None:
        plan = build_plan("Which files call preprocessInput before resolveTool?", settings)
        assert len(retriever.search(plan.original_query, plan, k=2)) <= 2
        assert retriever.search(plan.original_query, plan, k=0) == []

    def test_usage_queries_skip_the_ordered_pair_join(self, retriever, settings) -> None:
        """ "Where is X used" is all call sites and exports; ordering is
        irrelevant, so the expensive self-join is not run at all."""
        plan = build_plan("Where is the preprocessInput helper used?", settings)
        results, evidence = retriever.search_with_evidence(plan.original_query, plan)
        assert not any("ordinal" in detail for detail in evidence.values())
        if results:
            assert_ranked_list(results, expected_signal=SignalKind.STRUCTURAL)

    def test_repeated_searches_are_deterministic(self, retriever, settings) -> None:
        plan = build_plan("Which files call preprocessInput before resolveTool?", settings)
        first = retriever.search(plan.original_query, plan)
        second = retriever.search(plan.original_query, plan)
        assert [(e.chunk_id, e.rank, e.score) for e in first] == [
            (e.chunk_id, e.rank, e.score) for e in second
        ]


# ---------------------------------------------------------------------------
# FR-10 -- degradation of the store itself
# ---------------------------------------------------------------------------


class TestStoreDegradation:
    def test_a_missing_index_returns_empty(self, tmp_path: Path, settings) -> None:
        """An absent ``structural.sqlite`` is an absent signal, not an error."""
        with StructuralRetriever(tmp_path / "nothing-here", settings) as retriever:
            assert (
                retriever.search("preprocessInput", make_plan(query_type=QueryType.STRUCTURAL))
                == []
            )
            assert retriever.callers_of("preprocessInput", 10) == []

    def test_a_corrupt_database_returns_empty(self, tmp_path: Path, settings) -> None:
        """A truncated file is exactly what an interrupted build leaves behind."""
        (tmp_path / INDEX_FILENAME).write_bytes(b"this is not a sqlite database at all")
        with StructuralRetriever(tmp_path, settings) as retriever:
            assert retriever.callers_of("preprocessInput", 10) == []

    def test_an_index_with_the_wrong_schema_returns_empty(self, tmp_path: Path, settings) -> None:
        connection = sqlite3.connect(tmp_path / INDEX_FILENAME)
        connection.execute("CREATE TABLE unrelated (a INTEGER)")
        connection.commit()
        connection.close()
        with StructuralRetriever(tmp_path, settings) as retriever:
            assert retriever.definitions_of("resolveTool", 10) == []

    def test_building_over_no_chunks_does_not_raise(self, tmp_path: Path, settings) -> None:
        """TC-078's shape: a version that deletes every file."""
        stats = build_structural_index([], settings, tmp_path)
        assert stats.symbol_count == 0
        with StructuralRetriever(tmp_path, settings) as retriever:
            assert retriever.callers_of("anything", 10) == []

    def test_the_eval_profile_skips_the_structural_build_entirely(
        self, tmp_path: Path, repo_v1: Path
    ) -> None:
        """PRD.md section 2.2: not down-weighted, *not built*."""
        eval_settings = get_settings("eval", configs_dir=CONFIGS, index_root=tmp_path / ".axiom")
        assert eval_settings.structural_enabled is False
        chunks = chunk_repo(repo_v1, "v1", eval_settings)
        stats = build_structural_index(chunks, eval_settings, tmp_path)
        assert stats.skipped is True

    def test_a_sql_metacharacter_in_an_identifier_is_not_an_injection(self, retriever) -> None:
        """Every query is parameterised; ``LIKE`` patterns are escaped."""
        for hostile in ["'; DROP TABLE symbols; --", "%", "_", "100%_x"]:
            assert retriever.substring_match(hostile, 10) is not None
        assert retriever.callers_of("preprocessInput", 10), "the table must still be there"


# ---------------------------------------------------------------------------
# Identifier normalisation helpers
# ---------------------------------------------------------------------------


class TestIdentifierHelpers:
    def test_normalise_strips_call_syntax_and_whitespace(self) -> None:
        assert normalise_identifier("  resolveTool()  ") == "resolveTool"

    def test_normalise_is_idempotent(self) -> None:
        once = normalise_identifier("utils.normalize")
        assert normalise_identifier(once) == once

    def test_split_callee_separates_the_receiver(self) -> None:
        """``a.b()`` is a call to ``b`` on ``a``; the graph stores both."""
        name, receiver = split_callee("utils.normalize")
        assert name == "normalize" and receiver == "utils"
        assert split_callee("resolveTool") == ("resolveTool", None)

    def test_mine_identifiers_lifts_code_tokens_out_of_prose(self) -> None:
        found = mine_identifiers("which files call preprocessInput before resolveTool")
        assert "preprocessInput" in found and "resolveTool" in found

    def test_mine_identifiers_on_pure_prose_is_empty_or_harmless(self) -> None:
        assert isinstance(mine_identifiers("the quick brown fox"), list)

    def test_mine_identifiers_never_raises(self) -> None:
        """Hostile shapes, all bounded by ``MAX_QUERY_CHARS`` on the real path.

        The long input is capped at the planner's 2048-character truncation
        rather than TC-047's 10,000, because that is the longest string the
        query path can actually hand this function -- and because the regex is
        quadratic in input length, so 10,000 characters costs two seconds inside
        the merge gate. The scaling itself is pinned by the ``bench``-marked
        test below.
        """
        from axiom.agent.planner import MAX_QUERY_CHARS

        for hostile in ["", "   ", "🔵", "(" * 500, "a" * MAX_QUERY_CHARS]:
            assert isinstance(mine_identifiers(hostile), list)

    @pytest.mark.bench
    def test_tc047_mine_identifiers_is_linear_in_input_length(self) -> None:
        """TC-047: adversarial regex input, after the fix.

        ``_CODE_TOKEN_RE``'s camelCase alternative,
        ``[a-z$_][A-Za-z0-9$]*[A-Z][A-Za-z0-9_$]*``, has no anchor for its
        mandatory uppercase character, so run against a whole query it scans to
        the end of a lowercase run from every start offset: 5.6 / 22 / 90 /
        340 ms for 512 / 1024 / 2048 / 4096 characters, a clean 4x per doubling.
        At the planner's 2048-character bound that was ~90 ms *per (version,
        sub-query, pass)* -- a fifth of the 5 s agent budget inside one regex.

        ``mine_identifiers`` now splits into maximal identifier-alphabet runs
        first (a pattern that cannot backtrack), skips runs over
        ``_MAX_MINED_TOKEN_CHARS``, and skips runs holding none of the three
        characters the shapes require, so the expensive pattern only ever sees a
        short token. Measured after the change: 0.007 ms at 2048 and 0.05 ms at
        20 000 characters.

        Marked ``bench`` so it stays out of the merge gate (TestPlan.md section
        8.1). The assertion has a large slack factor -- it exists to fail loudly
        if someone reintroduces super-linear growth, not to measure the machine.
        """
        import time

        def cost(size: int) -> float:
            start = time.perf_counter()
            mine_identifiers("a" * size)
            return time.perf_counter() - start

        small, large = cost(512), cost(8192)
        # 16x the input. Linear predicts 16x the time; quadratic predicts 256x.
        # 64x sits well clear of both the true ratio and runner noise.
        assert large < max(small, 1e-4) * 64, "growth is worse than linear"

    def test_tc047_long_runs_are_bounded_rather_than_scanned(self) -> None:
        """The 128-character bound is the whole reason the worst case is gone.

        A single token longer than ``_MAX_MINED_TOKEN_CHARS`` is not mined. No
        real JavaScript symbol is that long, and this is the degraded rung --
        reached only when the planner extracted no identifiers at all.
        """
        from axiom.retrieval.structural import _MAX_MINED_TOKEN_CHARS

        just_under = "a" * (_MAX_MINED_TOKEN_CHARS - 1) + "Z"
        assert mine_identifiers(just_under) == [just_under]

        just_over = "a" * _MAX_MINED_TOKEN_CHARS + "Z"
        assert mine_identifiers(just_over) == []

    def test_tc047_prose_and_symbols_are_unchanged_by_the_prefilter(self) -> None:
        """The cheap pre-filter must not change a single realistic result."""
        assert mine_identifiers("the quick brown fox") == []
        assert mine_identifiers("call preprocessInput then resolveTool") == [
            "preprocessInput",
            "resolveTool",
        ]
        assert mine_identifiers("utils.normalize and handle_tap") == [
            "utils.normalize",
            "handle_tap",
        ]
