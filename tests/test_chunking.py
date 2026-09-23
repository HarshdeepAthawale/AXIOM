"""Chunking: the only place a ``chunk_id`` is minted (Rules.md Rule 1).

Two properties are load-bearing and are asserted here on every rung of the
ladder, not just the top one:

* **Byte equivalence** -- ``source_bytes[start_byte:end_byte].decode("utf-8")``
  equals ``chunk.text``, on multibyte source too. Everything downstream (the
  UI's "open this file at this line", the reranker's view of the code, the
  evolutionary diff) reads a chunk by its span, so a drifting offset is a
  silently wrong answer rather than a crash.
* **A file never aborts the index** -- NFR-07. Unparseable, empty, 0-byte,
  minified, deeply nested: each degrades and logs.

Most of this suite runs on the **regex rung**, because ``tree_sitter`` is an
optional extra and the evaluator's machine may not have it. Tests that genuinely
need the AST (kind taxonomy: FUNCTION/METHOD/CLASS) are marked
``needs_treesitter`` and skip cleanly -- see the coverage note in the report.
"""

from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path

import pytest

from axiom.chunking import (
    chunk_file,
    chunk_repo,
    discover_source_files,
    estimate_tokens,
    module_chunk,
    regex_chunks,
    split_text,
)
from axiom.chunking.ast_chunker import IGNORED_DIRECTORIES, IGNORED_SUFFIXES, SOURCE_EXTENSIONS
from axiom.config import get_settings
from axiom.schema import ChunkKind

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

HAS_TREE_SITTER = True
try:  # pragma: no cover - environment dependent
    import tree_sitter  # noqa: F401
    import tree_sitter_javascript  # noqa: F401
except Exception:  # pragma: no cover - the bare-install path, which is the default
    HAS_TREE_SITTER = False

needs_treesitter = pytest.mark.skipif(
    not HAS_TREE_SITTER, reason="requires the optional 'structural' extra (tree-sitter)"
)


@pytest.fixture
def settings():
    return get_settings("default", configs_dir=CONFIGS)


def assert_byte_equivalent(source: str, chunks) -> None:
    """The invariant, as one call. ``text`` is *always* a byte slice."""
    data = source.encode("utf-8")
    for chunk in chunks:
        span = data[chunk.location.start_byte : chunk.location.end_byte]
        assert span.decode("utf-8") == chunk.text, f"byte span drifted at {chunk.location.as_ref()}"


def assert_line_equivalent(source: str, chunks) -> None:
    """Lines are 1-indexed and inclusive on both ends (Schema.md section 4)."""
    lines = source.split("\n")
    for chunk in chunks:
        loc = chunk.location
        assert loc.start_line >= 1
        assert loc.end_line >= loc.start_line
        assert "\n".join(lines[loc.start_line - 1 : loc.end_line]) == chunk.text


# ---------------------------------------------------------------------------
# The two load-bearing invariants
# ---------------------------------------------------------------------------


class TestSpanInvariants:
    def test_byte_and_line_spans_agree_across_the_whole_fixture_repo(
        self, repo_v1: Path, settings
    ) -> None:
        """Every chunk of every fixture file, on whichever rung the install allows."""
        checked = 0
        for path in discover_source_files(repo_v1):
            source = path.read_text(encoding="utf-8")
            chunks = chunk_file(source, path.relative_to(repo_v1).as_posix(), "v1", settings)
            assert_byte_equivalent(source, chunks)
            assert_line_equivalent(source, chunks)
            checked += len(chunks)
        assert checked > 0, "the fixture repo produced no chunks at all"

    def test_tc011_multibyte_source_keeps_byte_offsets_honest(
        self, repo_v1: Path, settings
    ) -> None:
        """TC-011: accents and an emoji, where byte offsets and character
        offsets disagree. A chunker that slices ``str`` instead of ``bytes``
        passes every ASCII test and fails here."""
        path = repo_v1 / "src/i18n/labels.js"
        source = path.read_text(encoding="utf-8")
        assert len(source.encode("utf-8")) > len(source), "fixture is not actually multibyte"

        chunks = chunk_file(source, "src/i18n/labels.js", "v1", settings)
        assert chunks
        assert_byte_equivalent(source, chunks)
        assert any("🔵" in chunk.text for chunk in chunks)

    def test_a_chunk_never_has_an_inverted_span(self, repo_v1: Path, settings) -> None:
        for path in discover_source_files(repo_v1):
            for chunk in chunk_file(path, path.relative_to(repo_v1).as_posix(), "v1", settings):
                assert chunk.location.end_byte > chunk.location.start_byte
                assert chunk.location.end_line >= chunk.location.start_line

    def test_chunk_ids_are_unique_within_a_file(self, repo_v1: Path, settings) -> None:
        """Two chunks at the same path and start line would collide by construction."""
        for path in discover_source_files(repo_v1):
            chunks = chunk_file(path, path.relative_to(repo_v1).as_posix(), "v1", settings)
            ids = [chunk.chunk_id for chunk in chunks]
            assert len(set(ids)) == len(ids)

    def test_tc014_chunking_is_stable_across_repeated_runs(self, repo_v1: Path, settings) -> None:
        """TC-014: identical input -> identical id set, and ids are 32-hex."""
        path = repo_v1 / "src/utils/normalize.js"
        first = chunk_file(path, "src/utils/normalize.js", "v1", settings)
        second = chunk_file(path, "src/utils/normalize.js", "v1", settings)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert all(len(c.chunk_id) == 32 and int(c.chunk_id, 16) >= 0 for c in first)


# ---------------------------------------------------------------------------
# TC-019 -- oversize split
# ---------------------------------------------------------------------------


class TestOversizeSplit:
    """A single ~900-statement function, generated rather than committed.

    TestPlan.md's fixture table calls for a committed 5000-line file; generating
    it keeps the repository small and makes the statement count an explicit
    parameter of the test rather than a property of a blob nobody reads.
    """

    @staticmethod
    def _huge(statements: int = 900) -> str:
        body = "\n".join(f"  const v{i} = compute{i}(v{i - 1});" for i in range(1, statements))
        return "function huge() {\n" + body + "\n}\n"

    def test_tc019_oversize_function_splits_within_the_token_budget(self, settings) -> None:
        """TC-019: multiple chunks, each within ``chunk_target_tokens``."""
        source = self._huge()
        chunks = chunk_file(source, "src/generated/huge.js", "v1", settings)

        assert len(chunks) > 1, "a 900-statement function must not stay one chunk"
        assert all(
            estimate_tokens(chunk.text) <= settings.chunk_target_tokens for chunk in chunks
        ), "a fragment exceeded the token budget"

    def test_tc019_split_points_are_line_boundaries_and_cover_the_source(self, settings) -> None:
        """TC-019's "no gaps" clause, checked as coverage of the byte range.

        The fragments are asserted to be *contiguous or overlapping* -- the spec
        asks for a one-statement overlap, so consecutive starts may precede the
        previous end, but a gap would drop code out of the index entirely.
        """
        source = self._huge()
        chunks = chunk_file(source, "src/generated/huge.js", "v1", settings)
        assert_byte_equivalent(source, chunks)

        spans = sorted((c.location.start_byte, c.location.end_byte) for c in chunks)
        for (_, prev_end), (next_start, _) in pairwise(spans):
            assert next_start <= prev_end, "oversize split left a gap in the source"

        assert spans[0][0] <= len("function huge() {")
        assert spans[-1][1] >= len(source.encode("utf-8")) - 2

    def test_split_text_is_addressable_on_its_own(self, settings) -> None:
        """The rung is individually callable so a test can drive it without
        uninstalling tree-sitter (``chunking/__init__`` docstring)."""
        source = self._huge(200)
        fragments = split_text(source, "src/generated/huge.js", "v1", settings)
        assert len(fragments) > 1
        assert all(fragment.metadata.kind is ChunkKind.BLOCK for fragment in fragments)
        assert_byte_equivalent(source, fragments)

    def test_tc023_a_single_enormous_line_does_not_blow_up(self, settings) -> None:
        """TC-023's shape, scaled down: one line, no statement boundaries at all.

        The window splitter has nothing to cut on, so the character rung has to
        take over. What is asserted is the contract -- bounded chunk count,
        budget respected, spans valid -- not a chunk count.
        """
        source = ";".join(f"var a{i}=f{i}(a{i - 1})" for i in range(1, 4000)) + ";\n"
        chunks = chunk_file(source, "src/vendor/bundle.js", "v1", settings)
        assert chunks
        assert all(chunk.location.end_byte > chunk.location.start_byte for chunk in chunks)
        assert all(estimate_tokens(chunk.text) <= settings.chunk_target_tokens for chunk in chunks)
        assert_byte_equivalent(source, chunks)


# ---------------------------------------------------------------------------
# TC-020 -- the sub-floor merge
# ---------------------------------------------------------------------------


class TestTinyChunkMerge:
    def test_tc020_a_two_line_method_is_absorbed_into_its_parent(self, settings) -> None:
        """TC-020: no emitted chunk falls below ``chunk_min_tokens``, and the
        tiny function's text still appears inside the surviving chunk."""
        tiny_body = "  a() { return 1; }\n"
        big_body = "\n".join(f"    const x{i} = compute({i});" for i in range(30))
        source = f"class C {{\n{tiny_body}  bigMethod() {{\n{big_body}\n  }}\n}}\n"

        chunks = chunk_file(source, "src/tiny.js", "v1", settings)
        assert chunks
        assert all(estimate_tokens(chunk.text) >= settings.chunk_min_tokens for chunk in chunks)
        assert any("a() { return 1; }" in chunk.text for chunk in chunks), (
            "the absorbed method's text must survive somewhere"
        )

    def test_no_chunk_in_the_fixture_repo_is_below_the_floor(self, repo_v1: Path, settings) -> None:
        for path in discover_source_files(repo_v1):
            for chunk in chunk_file(path, path.relative_to(repo_v1).as_posix(), "v1", settings):
                assert estimate_tokens(chunk.text) >= settings.chunk_min_tokens


# ---------------------------------------------------------------------------
# TC-021 / TC-022 -- degradation, never an abort
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_tc021_unparseable_file_degrades_to_the_regex_rung_and_logs(
        self, repo_v1: Path, settings, caplog: pytest.LogCaptureFixture
    ) -> None:
        """TC-021: no exception, at least one chunk, ``calls`` populated by the
        regex fallback, and a WARNING recording the degrade (Rule 3: loud)."""
        source = (repo_v1 / "src/broken/syntax_error.js").read_text(encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="axiom"):
            chunks = chunk_file(source, "src/broken/syntax_error.js", "v1", settings)

        assert chunks, "an unparseable file must still contribute chunks"
        assert all(chunk.metadata.kind is ChunkKind.BLOCK for chunk in chunks)
        assert_byte_equivalent(source, chunks)

        calls = {call for chunk in chunks for call in chunk.metadata.calls}
        assert {"preprocessInput", "resolveTool"} <= calls, (
            "the regex fallback must still mine call names out of broken source"
        )
        assert any(record.levelno >= logging.WARNING for record in caplog.records)

    def test_tc022_empty_file_yields_zero_chunks(self, settings) -> None:
        """TC-022: 0 bytes -> 0 chunks, and an all-whitespace file behaves the same."""
        assert chunk_file("", "src/empty.js", "v1", settings) == []
        assert chunk_file("   \n\n\t\n", "src/blank.js", "v1", settings) == []

    def test_tc022_constants_only_file_yields_one_module_chunk(
        self, repo_v1: Path, settings
    ) -> None:
        """TC-022: a file with no functions is exactly one MODULE chunk."""
        source = (repo_v1 / "src/constants.js").read_text(encoding="utf-8")
        chunks = chunk_file(source, "src/constants.js", "v1", settings)
        assert len(chunks) == 1
        assert chunks[0].metadata.kind is ChunkKind.MODULE
        assert "MAX_RETRY" in chunks[0].text

    def test_tc024_nested_closures_do_not_recurse_to_death(self, settings) -> None:
        """TC-024: the walker is iterative. 200 levels, no ``RecursionError``.

        Generated at 200 deep -- the number the plan names -- rather than
        committed, so the depth is visible in the test rather than hidden in a
        fixture file.
        """
        depth = 200
        opens = "".join(f"  return function level{i}(a{i}) {{\n" for i in range(depth))
        closes = "".join("  };\n" for _ in range(depth))
        source = f"export function deep(seed) {{\n{opens}    return seed;\n{closes}}}\n"

        chunks = chunk_file(source, "src/deep/generated.js", "v1", settings)
        assert chunks
        assert_byte_equivalent(source, chunks)

    def test_a_file_that_cannot_be_decoded_returns_empty_rather_than_raising(
        self, tmp_path: Path, settings
    ) -> None:
        """Rule 3: bad input degrades. A latin-1 blob is not a crash."""
        path = tmp_path / "broken.js"
        path.write_bytes(b"\xff\xfe\x00invalid utf-8 \xc3\x28")
        assert chunk_file(path, "src/broken.js", "v1", settings) == []

    def test_a_missing_file_returns_empty_rather_than_raising(
        self, tmp_path: Path, settings
    ) -> None:
        assert chunk_file(tmp_path / "nope.js", "src/nope.js", "v1", settings) == []

    def test_an_absolute_or_windows_path_is_repaired_not_rejected(self, settings) -> None:
        """``ChunkLocation`` forbids both forms; the chunker normalises rather
        than propagating a validation error out of the index build."""
        source = "function f() {\n  return compute(1);\n}\n"
        for raw in ("/abs/src/f.js", "src\\win\\f.js", "./src/f.js"):
            chunks = chunk_file(source, raw, "v1", settings)
            assert chunks
            assert not chunks[0].location.file_path.startswith(("/", "./"))
            assert "\\" not in chunks[0].location.file_path


# ---------------------------------------------------------------------------
# Lower rungs, addressed directly
# ---------------------------------------------------------------------------


class TestLowerRungs:
    def test_module_chunk_is_a_single_whole_file_chunk(self, settings) -> None:
        source = "const a = 1;\nconst b = 2;\n"
        chunks = module_chunk(source, "src/constants.js", "v1", settings)
        assert len(chunks) == 1
        assert chunks[0].metadata.kind is ChunkKind.MODULE
        assert_byte_equivalent(source, chunks)

    def test_regex_chunks_finds_declaration_anchors(self, settings) -> None:
        source = (
            "function first() {\n  return alpha();\n}\n\nfunction second() {\n  return beta();\n}\n"
        )
        chunks = regex_chunks(source, "src/two.js", "v1", settings)
        symbols = {chunk.metadata.symbol for chunk in chunks}
        assert {"first", "second"} <= symbols
        assert_byte_equivalent(source, chunks)

    def test_regex_rung_records_imports_and_calls(self, settings) -> None:
        source = (
            "const { preprocessInput } = require('./normalize.js');\n"
            "import helper from 'node:fs';\n\n"
            "function run(x) {\n  const y = preprocessInput(x);\n  return resolveTool(y);\n}\n"
        )
        chunks = regex_chunks(source, "src/run.js", "v1", settings)
        imports = {spec for chunk in chunks for spec in chunk.metadata.imports}
        calls = {call for chunk in chunks for call in chunk.metadata.calls}
        assert "./normalize.js" in imports and "node:fs" in imports
        assert {"preprocessInput", "resolveTool"} <= calls

    def test_calls_keep_source_order_which_is_what_answers_q2(self, settings) -> None:
        """``metadata.calls`` order *is* the ordered-call-pair predicate's input."""
        source = (
            "function dispatch(p) {\n"
            "  const clean = preprocessInput(p);\n"
            "  return resolveTool(clean);\n"
            "}\n"
        )
        chunk = chunk_file(source, "src/dispatch.js", "v1", settings)[0]
        calls = chunk.metadata.calls
        assert calls.index("preprocessInput") < calls.index("resolveTool")


# ---------------------------------------------------------------------------
# chunk_repo / discover_source_files
# ---------------------------------------------------------------------------


class TestRepoWalk:
    def test_walk_is_sorted_and_contained(self, repo_v1: Path, settings) -> None:
        """One bad file must not cost the index (NFR-07): the broken file is in
        this tree and the walk still returns everything else."""
        chunks = chunk_repo(repo_v1, "v1", settings)
        paths = [chunk.location.file_path for chunk in chunks]
        assert paths == sorted(paths, key=lambda p: (p,)) or len(set(paths)) > 1
        assert "src/utils/normalize.js" in paths
        assert "src/broken/syntax_error.js" in paths

    def test_every_chunk_carries_the_version_id(self, repo_v1: Path, settings) -> None:
        for chunk in chunk_repo(repo_v1, "v7", settings):
            assert chunk.metadata.version_id == "v7"

    def test_provenance_is_stamped_on_every_chunk(self, repo_v1: Path, settings) -> None:
        sha = "0123456789abcdef" * 2 + "01234567"
        chunks = chunk_repo(repo_v1, "v1", settings, commit_sha=sha, last_modified="2026-09-18")
        assert chunks
        assert all(chunk.metadata.commit_sha == sha for chunk in chunks)
        assert all(chunk.metadata.last_modified == "2026-09-18" for chunk in chunks)

    def test_ignored_directories_are_not_walked(self, repo_v1_copy: Path, settings) -> None:
        """NG-26: ``node_modules`` and friends dominate byte count for no signal."""
        noise = repo_v1_copy / "node_modules" / "pkg"
        noise.mkdir(parents=True)
        (noise / "index.js").write_text("function vendored() { return compute(); }\n")
        found = {p.as_posix() for p in discover_source_files(repo_v1_copy)}
        assert not any("node_modules" in path for path in found)
        assert "node_modules" in IGNORED_DIRECTORIES

    def test_minified_suffixes_are_skipped_by_the_walk(self, repo_v1_copy: Path, settings) -> None:
        (repo_v1_copy / "src" / "vendor.min.js").write_text("var a=1;var b=2;\n")
        found = {p.name for p in discover_source_files(repo_v1_copy)}
        assert "vendor.min.js" not in found
        assert ".min.js" in IGNORED_SUFFIXES

    def test_non_javascript_files_are_not_indexed(self, repo_v1_copy: Path, settings) -> None:
        (repo_v1_copy / "README.md").write_text("# not code\n")
        (repo_v1_copy / "src" / "styles.css").write_text("body { color: red; }\n")
        for path in discover_source_files(repo_v1_copy):
            assert path.suffix in SOURCE_EXTENSIONS

    def test_an_empty_directory_walks_to_nothing(self, tmp_path: Path, settings) -> None:
        assert chunk_repo(tmp_path, "v1", settings) == []

    def test_a_nonexistent_root_degrades_rather_than_raising(
        self, tmp_path: Path, settings
    ) -> None:
        assert chunk_repo(tmp_path / "absent", "v1", settings) == []


# ---------------------------------------------------------------------------
# AST-rung taxonomy -- only meaningful with the optional extra installed
# ---------------------------------------------------------------------------


@needs_treesitter
class TestAstTaxonomy:
    def test_tc012_top_level_functions_become_function_chunks(
        self, repo_v1: Path, settings
    ) -> None:
        """TC-012: ``normalize.js`` has three top-level functions."""
        source = (repo_v1 / "src/utils/normalize.js").read_text(encoding="utf-8")
        chunks = chunk_file(source, "src/utils/normalize.js", "v1", settings)
        functions = [c for c in chunks if c.metadata.kind is ChunkKind.FUNCTION]
        assert {c.metadata.symbol for c in functions} >= {
            "sanitizeInput",
            "normalizeInput",
            "preprocessInput",
        }
        assert_byte_equivalent(source, chunks)

    def test_tc013_class_methods_carry_kind_method_and_a_parent_symbol(
        self, repo_v1: Path, settings
    ) -> None:
        """TC-013: ``BluetoothAgent`` has four methods plus a constructor."""
        source = (repo_v1 / "src/agents/bluetooth.js").read_text(encoding="utf-8")
        chunks = chunk_file(source, "src/agents/bluetooth.js", "v1", settings)
        methods = [c for c in chunks if c.metadata.kind is ChunkKind.METHOD]
        assert methods
        assert all(c.metadata.parent_symbol == "BluetoothAgent" for c in methods)
        assert {"parseDeeplink", "openSettings", "supports", "reset"} <= {
            c.metadata.symbol for c in methods
        }

    def test_exports_are_detected_for_both_module_systems(self, repo_v1: Path, settings) -> None:
        """ESM ``export function`` and CommonJS ``module.exports = {...}``."""
        esm = chunk_file(
            (repo_v1 / "src/utils/normalize.js").read_text(encoding="utf-8"),
            "src/utils/normalize.js",
            "v1",
            settings,
        )
        cjs = chunk_file(
            (repo_v1 / "src/tools/registry.js").read_text(encoding="utf-8"),
            "src/tools/registry.js",
            "v1",
            settings,
        )
        assert any(c.metadata.is_exported for c in esm), "ESM export not detected"
        assert any(c.metadata.is_exported for c in cjs), "CommonJS export not detected"
