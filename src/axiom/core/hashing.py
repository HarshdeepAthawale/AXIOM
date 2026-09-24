"""Content addressing for Axiom.

Every digest in the system is blake2b with ``digest_size=16`` (128-bit), rendered
as 32 lowercase hex characters. See Schema.md section 13 for the rationale.
"""

from __future__ import annotations

import hashlib

_DIGEST_SIZE = 16
_SEP = b"\x00"


def normalise_for_hash(content: str) -> str:
    """Canonicalise code text for content hashing.

    Three transformations, in order:
      1. line endings -> "\\n"  (CRLF and lone CR both collapse)
      2. trailing whitespace stripped from every line
      3. runs of blank lines collapsed to a single blank line, and leading/
         trailing blank lines removed entirely

    Comments are NOT stripped. Comments and JSDoc carry real retrieval signal --
    they are frequently the only natural-language text in a chunk and are what
    lets a dense embedding connect "preprocessing" to ``normalize()``. A snippet
    whose only change is a reworded comment is a genuinely different snippet for
    retrieval purposes and must get a new content_hash.
    """
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    blank_run = 0
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if line:
            blank_run = 0
            out.append(line)
        else:
            blank_run += 1
            if blank_run == 1 and out:
                out.append("")
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def blake2b_128(*parts: bytes) -> str:
    """Domain-separated blake2b-128 hex digest over an ordered sequence of parts."""
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for index, part in enumerate(parts):
        if index:
            digest.update(_SEP)
        digest.update(part)
    return digest.hexdigest()


def compute_chunk_id(content: str, file_path: str, start_line: int) -> str:
    """Location-sensitive chunk identity: blake2b-128(content, file_path, start_line)."""
    return blake2b_128(
        content.encode("utf-8"),
        file_path.encode("utf-8"),
        str(start_line).encode("ascii"),
    )


def compute_content_hash(content: str) -> str:
    """Location-INsensitive content identity: blake2b-128(normalise_for_hash(content))."""
    return blake2b_128(normalise_for_hash(content).encode("utf-8"))


def compute_family_id(symbol: str | None, file_path: str) -> str:
    """Stable snippet-family identity: blake2b-128(symbol, file_path)."""
    return blake2b_128((symbol or "").encode("utf-8"), file_path.encode("utf-8"))


def compute_file_hash(content: str) -> str:
    """Whole-file hash used as the incremental-reindex diff basis."""
    return compute_content_hash(content)
