"""The data contract: every model invariant and validator.

Coverage target for ``src/axiom/schema/`` is the highest in the plan (95% line,
90% branch, TestPlan.md section 9) for one reason: four workstreams build against
these nine models, ``extra="forbid"`` means a typo is a hard failure rather than
a silent drop, and every validator here is cheap to test exhaustively. An
untested validator is a branch whose right answer nobody checked.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from axiom.config import DEFAULT_STRATEGY_WEIGHTS, get_settings
from axiom.core.hashing import compute_chunk_id, compute_content_hash
from axiom.schema import (
    Chunk,
    ChunkKind,
    ChunkLocation,
    ChunkMetadata,
    FusedResult,
    QueryPlan,
    QueryType,
    RetrievalResult,
    ScoredChunk,
    SignalKind,
    SnippetFamily,
    VersionManifest,
)

from .fakes import make_chunk
from .helpers import hex32

TEXT = "function demo() {\n  return 1;\n}"


def _location(**overrides: object) -> ChunkLocation:
    fields: dict[str, object] = {
        "file_path": "src/demo.js",
        "start_line": 1,
        "end_line": 3,
        "start_byte": 0,
        "end_byte": len(TEXT.encode("utf-8")),
    }
    fields.update(overrides)
    return ChunkLocation(**fields)  # type: ignore[arg-type]


def _metadata(**overrides: object) -> ChunkMetadata:
    fields: dict[str, object] = {"kind": ChunkKind.FUNCTION, "version_id": "v1", "symbol": "demo"}
    fields.update(overrides)
    return ChunkMetadata(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AxiomModel base configuration
# ---------------------------------------------------------------------------


class TestBaseConfig:
    def test_models_are_frozen(self) -> None:
        """``frozen=True`` -- a stage may never mutate its input (Rules.md AP-08)."""
        location = _location()
        with pytest.raises(ValidationError):
            location.start_line = 9  # type: ignore[misc]

    def test_models_are_hashable(self) -> None:
        """Frozen models are shared across the embedding thread pool without copying."""
        assert hash(_location()) == hash(_location())

    def test_extra_fields_are_forbidden(self) -> None:
        """A stale ``chunks.jsonl`` from an older build must fail loudly, not silently."""
        with pytest.raises(ValidationError):
            ChunkLocation(
                file_path="a.js",
                start_line=1,
                end_line=1,
                start_byte=0,
                end_byte=1,
                unknown_field=True,  # type: ignore[call-arg]
            )

    def test_whitespace_is_never_stripped(self) -> None:
        """``str_strip_whitespace=False`` is load-bearing: indentation drives line offsets."""
        indented = "    return 1;\n"
        chunk = make_chunk(indented, symbol=None)
        assert chunk.text == indented


# ---------------------------------------------------------------------------
# ChunkLocation
# ---------------------------------------------------------------------------


class TestChunkLocation:
    @pytest.mark.parametrize(
        "bad_path",
        ["/abs/path.js", "./relative.js", "src\\windows.js", "C:/win.js"],
    )
    def test_rejects_non_posix_relative_paths(self, bad_path: str) -> None:
        with pytest.raises(ValidationError):
            _location(file_path=bad_path)

    def test_rejects_empty_path(self) -> None:
        with pytest.raises(ValidationError):
            _location(file_path="")

    def test_rejects_inverted_line_span(self) -> None:
        with pytest.raises(ValidationError):
            _location(start_line=10, end_line=4)

    def test_rejects_empty_byte_span(self) -> None:
        """Byte range is half-open, so ``end_byte == start_byte`` is a zero-byte chunk."""
        with pytest.raises(ValidationError):
            _location(start_byte=5, end_byte=5)

    def test_rejects_zero_start_line(self) -> None:
        """Lines are 1-indexed; 0 is the classic off-by-one that corrupts every ref."""
        with pytest.raises(ValidationError):
            _location(start_line=0, end_line=1)

    def test_line_count_is_inclusive(self) -> None:
        assert _location(start_line=42, end_line=67).line_count == 26

    def test_single_line_chunk_counts_one(self) -> None:
        assert _location(start_line=7, end_line=7).line_count == 1

    def test_as_ref_is_the_human_facing_form(self) -> None:
        assert _location(start_line=42, end_line=67).as_ref() == "src/demo.js:42-67"


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_method_requires_parent_symbol(self) -> None:
        with pytest.raises(ValidationError, match="parent_symbol"):
            _metadata(kind=ChunkKind.METHOD, parent_symbol=None)

    def test_method_with_parent_symbol_is_accepted(self) -> None:
        meta = _metadata(kind=ChunkKind.METHOD, parent_symbol="BluetoothAgent")
        assert meta.parent_symbol == "BluetoothAgent"

    @pytest.mark.parametrize("bad_sha", ["abc", "A" * 40, "g" * 40, "0" * 41])
    def test_commit_sha_must_be_40_lowercase_hex(self, bad_sha: str) -> None:
        with pytest.raises(ValidationError):
            _metadata(commit_sha=bad_sha)

    def test_valid_commit_sha_round_trips(self) -> None:
        sha = "0123456789abcdef" * 2 + "01234567"
        assert _metadata(commit_sha=sha).commit_sha == sha

    def test_last_modified_must_be_iso_date(self) -> None:
        with pytest.raises(ValidationError):
            _metadata(last_modified="18/09/2026")

    def test_last_modified_accepts_iso_date(self) -> None:
        assert _metadata(last_modified="2026-09-18").last_modified == "2026-09-18"

    def test_call_order_is_preserved_and_duplicates_kept(self) -> None:
        """``calls`` order answers "calls X before Y"; deduping it destroys Q2."""
        meta = _metadata(calls=["preprocessInput", "resolveTool", "preprocessInput"])
        assert meta.calls == ["preprocessInput", "resolveTool", "preprocessInput"]

    def test_defaults_are_the_documented_ones(self) -> None:
        meta = _metadata()
        assert meta.language == "javascript"
        assert meta.is_exported is False
        assert meta.imports == [] and meta.calls == []


# ---------------------------------------------------------------------------
# Chunk -- the integrity check
# ---------------------------------------------------------------------------


class TestChunkIntegrity:
    def test_create_derives_both_digests(self) -> None:
        chunk = Chunk.create(TEXT, _location(), _metadata())
        assert chunk.chunk_id == compute_chunk_id(TEXT, "src/demo.js", 1)
        assert chunk.content_hash == compute_content_hash(TEXT)

    def test_mismatched_chunk_id_is_rejected(self) -> None:
        """The validator is what makes it impossible to mint a chunk whose ids lie."""
        with pytest.raises(ValidationError, match="chunk_id"):
            Chunk(
                chunk_id=hex32("wrong"),
                content_hash=compute_content_hash(TEXT),
                text=TEXT,
                location=_location(),
                metadata=_metadata(),
            )

    def test_mismatched_content_hash_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="content_hash"):
            Chunk(
                chunk_id=compute_chunk_id(TEXT, "src/demo.js", 1),
                content_hash=hex32("wrong"),
                text=TEXT,
                location=_location(),
                metadata=_metadata(),
            )

    def test_text_edit_without_rederiving_ids_is_rejected(self) -> None:
        """Deserialising a tampered ``chunks.jsonl`` row must fail at load time."""
        good = Chunk.create(TEXT, _location(), _metadata())
        payload = good.model_dump()
        payload["text"] = TEXT + "\n// appended"
        with pytest.raises(ValidationError):
            Chunk.model_validate(payload)

    def test_round_trips_through_json(self) -> None:
        """``chunks.jsonl`` is the on-disk format (Schema.md section 14)."""
        chunk = Chunk.create(TEXT, _location(), _metadata(calls=["a", "b"]))
        assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk

    def test_empty_text_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Chunk.create("", _location(), _metadata())

    def test_blob_name_is_keyed_by_content_hash(self) -> None:
        """Cross-version embedding dedup keys on content, not location."""
        chunk = Chunk.create(TEXT, _location(), _metadata())
        assert chunk.blob_name == f"{chunk.content_hash}.npy"

    @pytest.mark.parametrize("bad", ["not-hex", "abc", "F" * 32, "0" * 31])
    def test_ids_must_match_the_32_hex_pattern(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id=bad,
                content_hash=compute_content_hash(TEXT),
                text=TEXT,
                location=_location(),
                metadata=_metadata(),
            )


# ---------------------------------------------------------------------------
# QueryPlan -- weight normalisation (TC-005, TC-006)
# ---------------------------------------------------------------------------


class TestQueryPlan:
    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"sum to 1\.0"):
            QueryPlan(
                original_query="q",
                query_type=QueryType.HYBRID,
                strategy_weights={SignalKind.DENSE: 0.5, SignalKind.SPARSE: 0.2},
            )

    def test_weights_within_tolerance_are_accepted(self) -> None:
        """The validator's tolerance is 1e-6, which is what float division leaves."""
        plan = QueryPlan(
            original_query="q",
            query_type=QueryType.SEMANTIC,
            strategy_weights={SignalKind.DENSE: 0.6 / 0.9, SignalKind.SPARSE: 0.3 / 0.9},
        )
        assert abs(sum(plan.strategy_weights.values()) - 1.0) < 1e-9

    def test_negative_weights_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryPlan(
                original_query="q",
                query_type=QueryType.HYBRID,
                strategy_weights={SignalKind.DENSE: 1.5, SignalKind.SPARSE: -0.5},
            )

    def test_empty_weight_map_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryPlan(original_query="q", query_type=QueryType.HYBRID, strategy_weights={})

    def test_empty_query_string_is_rejected(self) -> None:
        """``min_length=1`` is why the planner carries an EMPTY_QUERY_PLACEHOLDER."""
        with pytest.raises(ValidationError):
            QueryPlan(
                original_query="",
                query_type=QueryType.HYBRID,
                strategy_weights={SignalKind.DENSE: 1.0},
            )

    def test_effective_queries_falls_back_to_the_original(self) -> None:
        plan = QueryPlan(
            original_query="how is input preprocessed",
            query_type=QueryType.SEMANTIC,
            strategy_weights=dict(DEFAULT_STRATEGY_WEIGHTS[QueryType.SEMANTIC]),
        )
        assert plan.effective_queries == ["how is input preprocessed"]

    def test_effective_queries_prefers_sub_queries(self) -> None:
        plan = QueryPlan(
            original_query="a and b",
            query_type=QueryType.HYBRID,
            sub_queries=["a", "b"],
            strategy_weights=dict(DEFAULT_STRATEGY_WEIGHTS[QueryType.HYBRID]),
        )
        assert plan.effective_queries == ["a", "b"]

    @pytest.mark.parametrize("query_type", list(QueryType))
    def test_tc005_default_weights_match_the_locked_table(self, query_type: QueryType) -> None:
        """TC-005: the §5 table, byte-exact.

        SEMANTIC {.6,.3,.1}, STRUCTURAL {.2,.2,.6}, USAGE {.25,.55,.2},
        HYBRID {.34,.33,.33}. Hard-coded here on purpose: this test's job is to
        fail when someone retunes a weight without updating _CONTRACT.md §5.
        """
        expected = {
            QueryType.SEMANTIC: (0.60, 0.30, 0.10),
            QueryType.STRUCTURAL: (0.20, 0.20, 0.60),
            QueryType.USAGE: (0.25, 0.55, 0.20),
            QueryType.HYBRID: (0.34, 0.33, 0.33),
        }[query_type]
        weights = DEFAULT_STRATEGY_WEIGHTS[query_type]
        assert weights[SignalKind.DENSE] == expected[0]
        assert weights[SignalKind.SPARSE] == expected[1]
        assert weights[SignalKind.STRUCTURAL] == expected[2]

    @pytest.mark.parametrize("query_type", list(QueryType))
    @pytest.mark.parametrize("profile", ["default", "demo", "fast", "accurate", "eval"])
    def test_tc006_weights_normalise_for_every_profile(
        self, query_type: QueryType, profile: str
    ) -> None:
        """TC-006: ``abs(sum - 1.0) <= 1e-9`` for all four types across ``configs/*.yaml``.

        The ``eval`` profile is the interesting row: it does not down-weight the
        structural signal, it drops the leg entirely and renormalises dense and
        sparse over their own sum.
        """
        from pathlib import Path

        settings = get_settings(
            profile, configs_dir=Path(__file__).resolve().parent.parent / "configs"
        )
        weights = settings.strategy_weights_for(query_type)
        assert abs(sum(weights.values()) - 1.0) <= 1e-9
        assert all(weight >= 0.0 for weight in weights.values())

    def test_eval_profile_drops_the_structural_leg(self) -> None:
        from pathlib import Path

        settings = get_settings(
            "eval", configs_dir=Path(__file__).resolve().parent.parent / "configs"
        )
        weights = settings.strategy_weights_for(QueryType.SEMANTIC)
        assert SignalKind.STRUCTURAL not in weights
        assert weights[SignalKind.DENSE] == pytest.approx(0.85)
        assert weights[SignalKind.SPARSE] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# ScoredChunk / FusedResult
# ---------------------------------------------------------------------------


class TestScoredChunk:
    def test_rank_is_one_indexed(self) -> None:
        with pytest.raises(ValidationError):
            ScoredChunk(chunk_id=hex32("a"), score=1.0, rank=0, signal=SignalKind.DENSE)

    def test_negative_scores_are_allowed(self) -> None:
        """Signal-native scores are not normalised; a raw dot product may be negative."""
        entry = ScoredChunk(chunk_id=hex32("a"), score=-0.4, rank=1, signal=SignalKind.DENSE)
        assert entry.score == -0.4


class TestFusedResult:
    def _fused(self, **overrides: object) -> FusedResult:
        fields: dict[str, object] = {
            "chunk_id": hex32("a"),
            "rrf_score": 0.03,
            "contributions": {SignalKind.DENSE: 5, SignalKind.SPARSE: 2},
            "dominant_signal": SignalKind.DENSE,
        }
        fields.update(overrides)
        return FusedResult(**fields)  # type: ignore[arg-type]

    def test_rrf_score_must_be_strictly_positive(self) -> None:
        """A chunk with no contributions is never constructed, so 0.0 is unreachable."""
        with pytest.raises(ValidationError):
            self._fused(rrf_score=0.0)

    def test_contributions_may_not_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            self._fused(contributions={}, dominant_signal=SignalKind.DENSE)

    def test_contribution_ranks_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            self._fused(contributions={SignalKind.DENSE: 0})

    def test_dominant_signal_must_be_a_contributor(self) -> None:
        with pytest.raises(ValidationError, match="dominant_signal"):
            self._fused(dominant_signal=SignalKind.STRUCTURAL)

    def test_rerank_score_is_bounded_to_unit_interval(self) -> None:
        for bad in (-0.01, 1.01):
            with pytest.raises(ValidationError):
                self._fused(rerank_score=bad)

    def test_final_score_prefers_rerank_when_present(self) -> None:
        assert self._fused(rerank_score=0.8).final_score == 0.8

    def test_final_score_falls_back_to_rrf_when_absent(self) -> None:
        """``rerank_score=None`` means "not a rerank candidate", not "irrelevant"."""
        assert self._fused().final_score == 0.03

    def test_rerank_score_of_zero_is_not_treated_as_missing(self) -> None:
        """The ``rerank_score or 0.0`` bug: 0.0 is falsy but is a real score."""
        assert self._fused(rerank_score=0.0).final_score == 0.0

    def test_signal_count_counts_contributing_signals(self) -> None:
        assert self._fused().signal_count == 2


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def _result(self, **overrides: object) -> RetrievalResult:
        fields: dict[str, object] = {
            "chunk": make_chunk(TEXT),
            "score": 0.5,
            "match_reason": "dense: semantic match at rank 1",
            "signals": {SignalKind.DENSE: 1},
        }
        fields.update(overrides)
        return RetrievalResult(**fields)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_score", [-0.001, 1.001])
    def test_score_is_bounded_to_unit_interval(self, bad_score: float) -> None:
        """This is the only user-facing score, and the UI renders it as a percentage."""
        with pytest.raises(ValidationError):
            self._result(score=bad_score)

    def test_match_reason_may_not_be_empty(self) -> None:
        """FR-14: every result explains itself. An empty string is not an explanation."""
        with pytest.raises(ValidationError):
            self._result(match_reason="")

    def test_signals_may_not_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            self._result(signals={})

    def test_location_ref_passes_through_to_the_chunk(self) -> None:
        result = self._result()
        assert result.location_ref == result.chunk.location.as_ref()

    def test_optimization_hint_is_optional(self) -> None:
        assert self._result().optimization_hint is None


# ---------------------------------------------------------------------------
# SnippetFamily / VersionManifest
# ---------------------------------------------------------------------------


def _family_member(version: str, line: int) -> Chunk:
    return make_chunk(TEXT, file_path="src/demo.js", start_line=line, version_id=version)


class TestSnippetFamily:
    def _family(self, **overrides: object) -> SnippetFamily:
        members = [_family_member("v3", 1), _family_member("v2", 5), _family_member("v1", 9)]
        fields: dict[str, object] = {
            "family_id": hex32("family"),
            "representative": members[0],
            "members": members,
            "versions": ["v3", "v2", "v1"],
            "stability": 1.0,
        }
        fields.update(overrides)
        return SnippetFamily(**fields)  # type: ignore[arg-type]

    def test_versions_must_mirror_member_version_ids_in_order(self) -> None:
        with pytest.raises(ValidationError, match="mirror"):
            self._family(versions=["v1", "v2", "v3"])

    def test_representative_must_be_the_newest_member(self) -> None:
        members = [_family_member("v3", 1), _family_member("v2", 5)]
        with pytest.raises(ValidationError, match="representative"):
            self._family(
                representative=members[1], members=members, versions=["v3", "v2"], stability=0.66
            )

    def test_family_may_not_span_two_symbols(self) -> None:
        members = [
            _family_member("v3", 1),
            make_chunk(TEXT, file_path="src/other.js", start_line=5, version_id="v2"),
        ]
        with pytest.raises(ValidationError, match="multiple"):
            self._family(
                representative=members[0], members=members, versions=["v3", "v2"], stability=0.5
            )

    def test_diff_count_must_be_members_minus_one(self) -> None:
        with pytest.raises(ValidationError, match="diffs"):
            self._family(diffs=["only one diff"])

    def test_stability_is_bounded_and_strictly_positive(self) -> None:
        for bad in (0.0, 1.5):
            with pytest.raises(ValidationError):
                self._family(stability=bad)

    def test_multi_version_family_earns_the_bonus(self) -> None:
        """``1 + 0.10*stability``: three of three versions is a 1.10x multiplier."""
        assert self._family(stability=1.0).ranking_bonus() == pytest.approx(1.10)

    def test_single_version_family_is_exactly_neutral(self) -> None:
        """A brand-new snippet is novel, not unstable -- the gate makes it 1.0, not 1.01."""
        member = _family_member("v1", 1)
        family = self._family(
            representative=member, members=[member], versions=["v1"], stability=0.1
        )
        assert family.is_multi_version is False
        assert family.ranking_bonus() == 1.0


class TestVersionManifest:
    def _manifest(self, **overrides: object) -> VersionManifest:
        fields: dict[str, object] = {
            "version_id": "v2",
            "created_at": "2026-09-18T14:03:22Z",
            "chunk_count": 120,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": 1024,
            "index_kind": "flat_ip",
        }
        fields.update(overrides)
        return VersionManifest(**fields)  # type: ignore[arg-type]

    def test_parent_version_may_not_be_self(self) -> None:
        with pytest.raises(ValidationError, match="parent_version"):
            self._manifest(parent_version="v2")

    def test_known_model_dimension_mismatch_is_rejected(self) -> None:
        """A manifest that lies about dim silently corrupts every future search."""
        with pytest.raises(ValidationError, match="dim"):
            self._manifest(embedding_dim=384)

    def test_minilm_at_384_is_accepted(self) -> None:
        manifest = self._manifest(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=384
        )
        assert manifest.embedding_dim == 384

    def test_unknown_model_dimension_is_not_policed(self) -> None:
        """An unregistered model id carries no expectation, so any dim is legal."""
        assert self._manifest(embedding_model="fake-embedder", embedding_dim=64).embedding_dim == 64

    @pytest.mark.parametrize("bad_kind", ["flat", "hnsw", "ivf", ""])
    def test_index_kind_is_a_closed_vocabulary(self, bad_kind: str) -> None:
        with pytest.raises(ValidationError):
            self._manifest(index_kind=bad_kind)

    @pytest.mark.parametrize("bad_path", ["/abs/a.js", "src\\a.js"])
    def test_file_hash_keys_must_be_posix_relative(self, bad_path: str) -> None:
        with pytest.raises(ValidationError, match="POSIX"):
            self._manifest(file_hashes={bad_path: "0" * 32})

    def test_zero_chunk_index_is_legal(self) -> None:
        """A version that deletes every file still publishes a manifest (TC-078)."""
        assert self._manifest(chunk_count=0).chunk_count == 0

    def test_round_trips_through_json(self) -> None:
        manifest = self._manifest(file_hashes={"src/a.js": "0" * 32}, parent_version="v1")
        assert VersionManifest.model_validate_json(manifest.model_dump_json()) == manifest


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_enum_members_are_the_closed_vocabularies(self) -> None:
        """These strings appear in ``chunks.jsonl`` and in every API response."""
        assert {q.value for q in QueryType} == {"semantic", "structural", "usage", "hybrid"}
        assert {s.value for s in SignalKind} == {"dense", "sparse", "structural"}
        assert {c.value for c in ChunkKind} == {"function", "method", "class", "module", "block"}

    def test_enums_compare_equal_to_their_strings(self) -> None:
        """``StrEnum`` so a JSON round-trip needs no coercion layer."""
        assert QueryType.SEMANTIC == "semantic"
        assert SignalKind.DENSE == "dense"
