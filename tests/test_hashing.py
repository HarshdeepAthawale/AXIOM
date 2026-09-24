"""Content addressing: the backbone of dedup and incremental reindex.

A hashing bug is invisible until the index is wrong, which is why this module
gets a 90% coverage target (TestPlan.md section 9) even though the code under
test is forty lines. The asymmetry between the two digests -- ``chunk_id``
binds location, ``content_hash`` does not -- is the single mechanism behind
intra-version dedup, cross-version dedup and zero-cost renames (Schema.md
section 13.1), so each of the three is asserted here directly rather than
inferred from an index build.
"""

from __future__ import annotations

import hashlib

import pytest

from axiom.core.hashing import (
    blake2b_128,
    compute_chunk_id,
    compute_content_hash,
    compute_family_id,
    compute_file_hash,
    normalise_for_hash,
)

from .fakes import make_chunk

FUNCTION = "function f(x) {\n\n  return x + 1;\n}"


# ---------------------------------------------------------------------------
# normalise_for_hash -- Schema.md section 13.2
# ---------------------------------------------------------------------------


class TestNormaliseForHash:
    def test_schema_13_2_worked_example(self) -> None:
        """Schema.md section 13.2, reproduced literally.

        Input has CRLF endings, trailing spaces, a three-line blank run in the
        middle and a trailing blank line. The document states the output is
        exactly four lines joined by ``\\n`` with no trailing newline.
        """
        raw = "function f(x) {  \r\n\r\n\r\n  return x + 1;\r\n}\r\n\r\n"
        assert normalise_for_hash(raw) == "function f(x) {\n\n  return x + 1;\n}"

    def test_crlf_and_lone_cr_both_collapse(self) -> None:
        assert normalise_for_hash("a\r\nb") == normalise_for_hash("a\rb") == "a\nb"

    def test_trailing_whitespace_is_stripped_per_line(self) -> None:
        assert normalise_for_hash("a   \nb\t\t\n") == "a\nb"

    def test_blank_runs_collapse_to_one(self) -> None:
        assert normalise_for_hash("a\n\n\n\n\nb") == "a\n\nb"

    def test_leading_and_trailing_blank_lines_are_removed(self) -> None:
        assert normalise_for_hash("\n\n\na\n\n\n") == "a"

    def test_all_blank_input_normalises_to_empty(self) -> None:
        assert normalise_for_hash("\n\n   \n\t\n") == ""

    def test_leading_indentation_is_preserved(self) -> None:
        """Only *trailing* whitespace is insignificant; indentation is code."""
        assert normalise_for_hash("    return x;") == "    return x;"

    def test_comments_are_not_stripped(self) -> None:
        """Deliberate: a reworded comment is a different snippet for retrieval.

        Comments are frequently the only natural-language text in a chunk and
        are what lets a dense embedding connect "preprocessing" to
        ``normalize()``.
        """
        a = compute_content_hash("// normalises input\nfunction f() {}")
        b = compute_content_hash("// sanitises input\nfunction f() {}")
        assert a != b

    def test_normalisation_is_idempotent(self) -> None:
        once = normalise_for_hash(FUNCTION + "  \r\n\r\n")
        assert normalise_for_hash(once) == once


# ---------------------------------------------------------------------------
# blake2b_128 -- the primitive
# ---------------------------------------------------------------------------


class TestBlake2b128:
    def test_digest_is_32_lowercase_hex(self) -> None:
        digest = blake2b_128(b"x")
        assert len(digest) == 32
        assert digest == digest.lower()
        int(digest, 16)  # raises if not hex

    def test_matches_hashlib_for_a_single_part(self) -> None:
        assert blake2b_128(b"abc") == hashlib.blake2b(b"abc", digest_size=16).hexdigest()

    def test_parts_are_domain_separated(self) -> None:
        """``("ab","c")`` and ``("a","bc")`` must not collide, or a symbol name
        could be confused with a path prefix."""
        assert blake2b_128(b"ab", b"c") != blake2b_128(b"a", b"bc")

    def test_separator_is_a_nul_byte_between_parts(self) -> None:
        assert blake2b_128(b"a", b"b") == hashlib.blake2b(b"a\x00b", digest_size=16).hexdigest()

    def test_empty_part_list_is_the_empty_digest(self) -> None:
        assert blake2b_128() == hashlib.blake2b(b"", digest_size=16).hexdigest()

    def test_is_stable_across_calls(self) -> None:
        assert blake2b_128(b"stable") == blake2b_128(b"stable")


# ---------------------------------------------------------------------------
# The asymmetry (TC-016, TC-017, TC-018)
# ---------------------------------------------------------------------------


class TestDigestAsymmetry:
    def test_tc016_content_hash_is_invariant_under_location_change(self) -> None:
        """TC-016: same text at three locations -> one content_hash, three chunk_ids."""
        same = compute_content_hash(FUNCTION)
        assert compute_content_hash(FUNCTION) == same

        original = compute_chunk_id(FUNCTION, "src/a.js", 10)
        moved_file = compute_chunk_id(FUNCTION, "src/b.js", 10)
        moved_line = compute_chunk_id(FUNCTION, "src/a.js", 40)
        assert len({original, moved_file, moved_line}) == 3

    def test_tc017_content_hash_is_sensitive_to_real_change(self) -> None:
        """TC-017: three single-token mutations, three distinct digests, no collisions."""
        original = "function f(x) {\n  if (x > 1) {\n    return x;\n  }\n}"
        mutants = [
            original.replace("x > 1", "x >= 1"),  # operator
            original.replace("x", "y"),  # rename
            original.replace("    return x;", "    log(x);\n    return x;"),  # added stmt
        ]
        digests = {compute_content_hash(text) for text in [original, *mutants]}
        assert len(digests) == 4

    def test_tc018_reformatting_leaves_content_hash_alone(self) -> None:
        """TC-018: CRLF conversion, trailing newline and trailing spaces are
        declared-insignificant; nothing else is."""
        base = "function f() {\n  return 1;\n}"
        assert compute_content_hash(base) == compute_content_hash(base.replace("\n", "\r\n"))
        assert compute_content_hash(base) == compute_content_hash(base + "\n")
        assert compute_content_hash(base) == compute_content_hash(
            base.replace("return 1;", "return 1;   ")
        )

    def test_tc018_indentation_width_does_change_the_hash(self) -> None:
        """Honest reading of the implementation, against TC-018's wording.

        TC-018 lists "change indentation width" among the transformations that
        must leave ``content_hash`` unchanged, but
        :func:`~axiom.core.hashing.normalise_for_hash` strips only *trailing*
        whitespace -- by design, since leading indentation is what
        ``Chunk.text`` must preserve verbatim. Stripping leading whitespace too
        would make a chunk's text unreconstructable from its normalised form.
        This test pins the implemented behaviour and flags the TC-018 row as
        overstated; see the report accompanying this suite.
        """
        two = "function f() {\n  return 1;\n}"
        four = "function f() {\n    return 1;\n}"
        assert compute_content_hash(two) != compute_content_hash(four)

    def test_reformatting_does_change_every_chunk_id(self) -> None:
        """Schema.md section 13.2: new chunk rows, no new embeddings."""
        base = "function f() {\n  return 1;\n}"
        assert compute_chunk_id(base, "a.js", 1) != compute_chunk_id(base + "\n", "a.js", 1)

    def test_start_line_is_part_of_the_address(self) -> None:
        assert compute_chunk_id(FUNCTION, "a.js", 1) != compute_chunk_id(FUNCTION, "a.js", 2)

    def test_chunk_id_is_not_derivable_from_content_hash(self) -> None:
        """They are independent digests; nothing may reconstruct one from the other."""
        assert compute_chunk_id(FUNCTION, "a.js", 1) != compute_content_hash(FUNCTION)


# ---------------------------------------------------------------------------
# The three features the asymmetry buys (Schema.md 13.1)
# ---------------------------------------------------------------------------


class TestDedupMechanisms:
    def test_intra_version_dedup_copy_pasted_code_embeds_once(self) -> None:
        """Schema.md 13.1 mechanism 1: two paths, one blob, two visible results."""
        left = make_chunk(FUNCTION, file_path="src/a.js", start_line=1)
        right = make_chunk(FUNCTION, file_path="src/b.js", start_line=1)
        assert left.content_hash == right.content_hash
        assert left.blob_name == right.blob_name
        assert left.chunk_id != right.chunk_id

    def test_cross_version_dedup_untouched_file_adds_no_blob(self) -> None:
        """Mechanism 2: identical text under a new ``version_id`` reuses the blob."""
        v1 = make_chunk(FUNCTION, file_path="src/a.js", start_line=1, version_id="v1")
        v2 = make_chunk(FUNCTION, file_path="src/a.js", start_line=1, version_id="v2")
        assert v1.content_hash == v2.content_hash
        # Same path and line, so even the address is stable when nothing moved.
        assert v1.chunk_id == v2.chunk_id

    def test_tc075_a_pure_rename_costs_zero_embeddings(self) -> None:
        """TC-075: ``git mv`` gives every chunk a new id and reuses every blob.

        Asserted at the hashing layer rather than by counting embedder calls,
        because this is where the property is *created*: if the two blob names
        agree, an incremental build physically cannot issue a forward pass for
        the renamed chunk -- the cache key already exists.
        """
        before = make_chunk(FUNCTION, file_path="src/old/name.js", start_line=12)
        after = make_chunk(FUNCTION, file_path="src/new/name.js", start_line=12)
        assert before.chunk_id != after.chunk_id, "a rename must re-address the chunk"
        assert before.content_hash == after.content_hash
        assert {before.blob_name} == {after.blob_name}, "the rename must add no blob"

    def test_rename_with_edit_does_cost_an_embedding(self) -> None:
        """The control for TC-075: a rename *plus* an edit is a genuinely new value."""
        before = make_chunk(FUNCTION, file_path="src/old.js", start_line=1)
        after = make_chunk(FUNCTION + "\n// changed", file_path="src/new.js", start_line=1)
        assert before.content_hash != after.content_hash


# ---------------------------------------------------------------------------
# compute_family_id / compute_file_hash
# ---------------------------------------------------------------------------


class TestFamilyAndFileHashes:
    def test_family_id_is_stable_across_versions(self) -> None:
        """The family key is ``(symbol, file_path)`` -- deliberately not content."""
        assert compute_family_id("handleDeeplink", "src/a.js") == compute_family_id(
            "handleDeeplink", "src/a.js"
        )

    def test_family_id_separates_symbol_from_path(self) -> None:
        assert compute_family_id("ab", "c.js") != compute_family_id("a", "bc.js")

    def test_family_id_accepts_an_anonymous_symbol(self) -> None:
        """Anonymous arrow functions carry ``symbol=None``; that must not raise."""
        assert len(compute_family_id(None, "src/a.js")) == 32

    def test_anonymous_and_empty_symbol_collide_by_design(self) -> None:
        assert compute_family_id(None, "a.js") == compute_family_id("", "a.js")

    def test_file_hash_is_the_content_hash_of_the_whole_file(self) -> None:
        """The incremental diff basis; sharing the function means sharing the
        normalisation rules, so a reformat-only commit reindexes nothing."""
        body = "const a = 1;\r\n\r\n\r\nconst b = 2;  \r\n"
        assert compute_file_hash(body) == compute_content_hash(body)

    def test_file_hash_ignores_line_ending_style(self) -> None:
        unix = "const a = 1;\nconst b = 2;\n"
        assert compute_file_hash(unix) == compute_file_hash(unix.replace("\n", "\r\n"))


# ---------------------------------------------------------------------------
# Unicode
# ---------------------------------------------------------------------------


class TestUnicode:
    @pytest.mark.parametrize(
        "text",
        [
            "const café = 'élève';",
            "const emoji = '🔵 Bluetooth';",
            "const mixed = 'naïve 🔵 déeplink';",
        ],
    )
    def test_multibyte_text_hashes_without_error(self, text: str) -> None:
        """UTF-8 is encoded explicitly, so a multibyte identifier is not a crash."""
        assert len(compute_content_hash(text)) == 32
        assert len(compute_chunk_id(text, "src/ünïcode.js", 1)) == 32

    def test_normalisation_does_not_apply_unicode_nfc_folding(self) -> None:
        """Two unicode spellings of the same glyph are different bytes and must
        stay different digests -- normalising them would be a silent edit."""
        composed = "const caf\u00e9 = 1;"  # e-acute as one code point
        decomposed = "const cafe\u0301 = 1;"  # e + combining acute
        assert compute_content_hash(composed) != compute_content_hash(decomposed)
