"""The chunker's degradation ladder, plus the source geometry both rungs share.

Rules.md section 3 fixes the ladder for ``axiom.chunking.ast_chunker:chunk_file``::

    AST node boundaries
      -> statement-boundary split
      -> axiom.chunking.fallback:split_text fixed-window line split with overlap
      -> whole file as one ChunkKind.MODULE chunk

The top two rungs need a parse tree and live in :mod:`axiom.chunking.ast_chunker`.
Everything below them lives here, together with :func:`regex_chunks` -- the
identifier-based splitter that TestPlan.md TC-021 requires a syntactically broken
file to land on, still carrying ``metadata.calls``.

This module is the dependency-free leaf of the package: ``ast_chunker`` imports
``fallback``, never the reverse. That is why the shared source-geometry helpers
(:class:`SourceIndex`, :func:`make_chunk`) live here rather than next to the AST
walker -- putting them there would make the fallback rung depend on the rung it
exists to replace.

Nothing in this file imports an optional dependency, at module level or inside a
function. That is deliberate: this is the code that has to run when nothing else
can (NFR-07).
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path

from axiom.config import Settings, get_settings
from axiom.core.logging import get_logger, log_degradation
from axiom.schema import Chunk, ChunkKind, ChunkLocation, ChunkMetadata

from .tokens import below_floor, estimate_tokens, exceeds_budget, prefix_within_budget

_LOG = get_logger("chunking.fallback")

#: Overlap between consecutive fixed-window fragments, in lines. The sibling of
#: the "1 statement" overlap in TechSpecifications.md section 5.4 -- same intent
#: (a query straddling a cut still has one fragment containing it whole), same
#: ``PLACEHOLDER`` status under OQ-09, and same reason for not being a Settings
#: field: section 5.4's own table marks the overlap constant "(chunker
#: constant)" with no env var.
FALLBACK_OVERLAP_LINES = 1

#: Statement-ish characters a hard character-level cut prefers to land on, used
#: only when a *single line* is larger than the whole token budget -- i.e. a
#: minified bundle (TC-023). Cutting after ``;`` or ``}`` keeps the fragment
#: closer to a statement boundary than an arbitrary column would.
_CUT_PREFERENCE = (";", "}")

#: JavaScript keywords that can be followed by ``(`` and are therefore not calls.
#: Without this, ``if (x)`` would contribute a phantom callee named ``if`` to
#: ``ChunkMetadata.calls`` and poison the "calls X before Y" ordering.
_NON_CALL_KEYWORDS = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "return",
        "typeof",
        "instanceof",
        "in",
        "of",
        "new",
        "delete",
        "void",
        "await",
        "yield",
        "function",
        "class",
        "do",
        "else",
        "try",
        "finally",
        "throw",
        "case",
        "with",
        "super",
        "this",
        "import",
        "export",
        "const",
        "let",
        "var",
    }
)

#: ``name(`` -- the regex-rung approximation of a call expression. Linear, no
#: nested quantifier, so it is ReDoS-free on adversarial input (TC-047).
_CALL_RE = re.compile(r"(?<![.\w$])([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")

#: ``obj.method(`` -- member calls, recorded by property name to match what the
#: AST rung records for ``this.n(q)``.
_MEMBER_CALL_RE = re.compile(r"\.\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")

#: ``require('x')``. The character class excludes quotes, so the lazy body can
#: never run past the closing quote and the match stays linear.
_REQUIRE_RE = re.compile(r"(?<![.\w$])require\s*\(\s*['\"]([^'\"\n]+)['\"]")

#: ``import ... from 'x'`` and bare ``import 'x'``.
_IMPORT_RE = re.compile(r"(?<![.\w$])import\s+(?:[^;'\"()]*?\bfrom\s*)?['\"]([^'\"\n]+)['\"]")

#: ``export ... from 'x'`` -- a re-export is an import of the file's own module
#: graph, and barrel files consist of nothing else.
_REEXPORT_RE = re.compile(r"(?<![.\w$])export\s+[^;'\"()]*?\bfrom\s*['\"]([^'\"\n]+)['\"]")

#: Definition anchors for the regex rung, in one alternation so the scan is a
#: single linear pass. Group ``name`` is the symbol when the branch has one.
_ANCHOR_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*(?P<fn>[A-Za-z_$][\w$]*)?"
    r"|^\s*(?:export\s+(?:default\s+)?)?class\s+(?P<cls>[A-Za-z_$][\w$]*)"
    r"|^\s*(?:export\s+)?(?:const|let|var)\s+(?P<var>[A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?(?:function\b|\(|[A-Za-z_$][\w$]*\s*=>)"
    r"|^\s*(?P<prop>[A-Za-z_$][\w$]*)\s*\([^()\n]*\)\s*\{",
)

#: Words that turn a following ``name(`` into a declaration rather than a call.
_DECLARATION_KEYWORDS = frozenset({"function", "class", "async", "get", "set", "static", "*"})

#: ``(args) {`` -- the tail of a shorthand method definition.
_METHOD_HEAD_RE = re.compile(r"\([^()\n]*\)\s*\{")

#: Comment markers stripped from a JSDoc block before it becomes a docstring.
_JSDOC_OPEN_RE = re.compile(r"^/\*+")
_JSDOC_CLOSE_RE = re.compile(r"\*+/$")
_JSDOC_LINE_RE = re.compile(r"^\s*\*+\s?")
_LINE_COMMENT_RE = re.compile(r"^\s*//+\s?")


class SourceIndex:
    """One source file held as bytes, with a line table over it.

    Every chunk this package emits gets its ``text`` by slicing :attr:`data`,
    never by slicing a Python string. That is the only way to guarantee
    Schema.md section 6's invariant -- ``source_bytes[start_byte:end_byte]
    .decode('utf-8') == chunk.text`` -- on a file containing multibyte
    characters, where byte offsets and character offsets diverge. tree-sitter
    also reports offsets in bytes, so working in bytes means the AST rung needs
    no conversion at all.
    """

    __slots__ = ("_line_starts", "data", "text")

    def __init__(self, data: bytes) -> None:
        """Build the line table. ``data`` must be valid UTF-8."""
        self.data = data
        self.text = data.decode("utf-8")
        starts = [0]
        position = data.find(b"\n")
        while position != -1:
            starts.append(position + 1)
            position = data.find(b"\n", position + 1)
        self._line_starts = starts

    @classmethod
    def from_text(cls, text: str) -> SourceIndex:
        """Build from an in-memory string by encoding it to UTF-8 first."""
        return cls(text.encode("utf-8"))

    @property
    def line_count(self) -> int:
        """Number of lines, counting a trailing newline as ending the last line."""
        return len(self._line_starts)

    def line_of_byte(self, offset: int) -> int:
        """1-indexed line containing ``offset``."""
        return bisect_right(self._line_starts, offset)

    def line_start_byte(self, line: int) -> int:
        """First byte of a 1-indexed ``line``, clamped into range."""
        index = min(max(line, 1), len(self._line_starts)) - 1
        return self._line_starts[index]

    def line_end_byte(self, line: int) -> int:
        """Byte one past the last byte of a 1-indexed ``line``, newline excluded."""
        if line >= len(self._line_starts):
            end = len(self.data)
        else:
            end = self._line_starts[line] - 1
            if end > 0 and self.data[end - 1 : end] == b"\r":
                end -= 1
        return max(end, self.line_start_byte(line))

    def line_text(self, line: int) -> str:
        """Text of a 1-indexed ``line``, newline excluded."""
        return self.data[self.line_start_byte(line) : self.line_end_byte(line)].decode(
            "utf-8", errors="replace"
        )


def normalise_file_path(raw: str) -> str:
    """Coerce a path into the repo-relative POSIX form ``ChunkLocation`` demands.

    Rule 3 forbids raising on bad input, and an absolute or Windows-style path
    reaching the chunker is bad input, not a contract violation -- so it is
    repaired and logged rather than thrown. Without this a single caller passing
    ``/abs/path.js`` would abort the whole index inside a Pydantic validator.
    """
    cleaned = raw.replace("\\", "/")
    if len(cleaned) > 1 and cleaned[1] == ":":
        cleaned = cleaned[2:]
    was_absolute = cleaned.startswith("/")
    cleaned = cleaned.lstrip("/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = re.sub(r"/{2,}", "/", cleaned).strip()
    if was_absolute:
        _LOG.warning("file_path %r was absolute; stored as %r", raw, cleaned)
    return cleaned or "unknown"


def strip_comment_markers(raw: str) -> str | None:
    """Turn a JSDoc or ``//`` comment block into a ``ChunkMetadata.docstring``.

    Markers are stripped, each line is trimmed, blank lines are dropped, and the
    survivors are joined with newlines. Comments survive into the chunk text
    anyway; this field exists so the sparse and structural layers can address the
    prose separately from the code.
    """
    lines: list[str] = []
    for raw_line in raw.splitlines():
        line = _LINE_COMMENT_RE.sub("", raw_line.strip())
        line = _JSDOC_OPEN_RE.sub("", line)
        line = _JSDOC_CLOSE_RE.sub("", line.rstrip())
        line = _JSDOC_LINE_RE.sub("", line)
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines) or None


def extract_imports(text: str) -> list[str]:
    """File-level module specifiers, verbatim and quote-stripped, in source order.

    Covers ESM ``import``, ESM re-``export ... from``, and CommonJS ``require``,
    because a real JavaScript corpus mixes all three. Duplicates are dropped
    (first occurrence wins); unlike ``calls``, import order carries no query
    semantics, and a file that requires the same module twice should not say so
    twice.
    """
    seen: dict[str, None] = {}
    for pattern in (_IMPORT_RE, _REEXPORT_RE, _REQUIRE_RE):
        for match in pattern.finditer(text):
            seen.setdefault(match.group(1), None)
    return list(seen)


def extract_calls(text: str) -> list[str]:
    """Callee identifiers in source order, duplicates kept.

    The regex approximation of what the AST rung reads off ``call_expression``
    nodes. Order is the entire data source for "calls X before Y" (Schema.md
    section 5), so the two matchers are merged by match position and never
    sorted or de-duplicated.

    ``name(`` is ambiguous in a way the AST rung never is: it is a call in
    ``resolveTool(id)`` and a *declaration* in ``function resolveTool(id) {``.
    Counting the declaration would make every chunk appear to call itself, which
    is worse than a miss -- it would answer "what calls resolveTool" with
    resolveTool. :func:`_is_declaration_site` is the filter for that.
    """
    found: list[tuple[int, str]] = []
    for match in _CALL_RE.finditer(text):
        name = match.group(1)
        if name in _NON_CALL_KEYWORDS or _is_declaration_site(text, match.start(1), match.end()):
            continue
        found.append((match.start(1), name))
    for match in _MEMBER_CALL_RE.finditer(text):
        found.append((match.start(1), match.group(1)))
    found.sort(key=lambda item: item[0])
    return [name for _, name in found]


def _is_declaration_site(text: str, name_start: int, paren_start: int) -> bool:
    """True when ``name(`` is introducing a definition rather than calling one."""
    line_start = text.rfind("\n", 0, name_start) + 1
    before = text[line_start:name_start]
    if before.split()[-1:] and before.split()[-1] in _DECLARATION_KEYWORDS:
        return True
    if before.strip():
        return False
    # Nothing but indentation before the name: a shorthand method definition
    # such as `teardown() {`, which a call never looks like.
    return _METHOD_HEAD_RE.match(text, paren_start - 1) is not None


def make_chunk(
    src: SourceIndex,
    start_byte: int,
    end_byte: int,
    *,
    file_path: str,
    version_id: str,
    kind: ChunkKind,
    symbol: str | None = None,
    parent_symbol: str | None = None,
    is_exported: bool = False,
    imports: Sequence[str] = (),
    calls: Sequence[str] = (),
    docstring: str | None = None,
    commit_sha: str | None = None,
    last_modified: str | None = None,
) -> Chunk | None:
    """Mint one :class:`Chunk` from a byte span, or ``None`` if the span is empty.

    The single place in the package where a chunk is constructed, so the
    byte-equivalence invariant has exactly one implementation to be right about:
    ``text`` is *always* the decoded byte slice, never a string the caller
    assembled. Leading blank lines and trailing whitespace are trimmed off the
    span first (so a chunk never starts on an empty line), and because the trim
    moves the offsets rather than the text, the invariant survives it.

    Returns ``None`` instead of raising when the span is blank or a schema
    validator rejects the result -- Rule 3, one bad span must not abort a file.
    """
    start_byte, end_byte = _trim_span(src.data, start_byte, end_byte)
    if end_byte <= start_byte:
        return None
    try:
        text = src.data[start_byte:end_byte].decode("utf-8")
    except UnicodeDecodeError:
        _LOG.warning(
            "dropping chunk in %s: byte span %d-%d is not on a character boundary",
            file_path,
            start_byte,
            end_byte,
        )
        return None
    if not text.strip():
        return None
    try:
        location = ChunkLocation(
            file_path=file_path,
            start_line=src.line_of_byte(start_byte),
            end_line=src.line_of_byte(end_byte - 1),
            start_byte=start_byte,
            end_byte=end_byte,
        )
        metadata = ChunkMetadata(
            symbol=symbol,
            kind=kind,
            parent_symbol=parent_symbol,
            is_exported=is_exported,
            imports=list(imports),
            calls=list(calls),
            docstring=docstring,
            version_id=version_id,
            commit_sha=commit_sha,
            last_modified=last_modified,
        )
        return Chunk.create(text=text, location=location, metadata=metadata)
    except ValueError as exc:
        _LOG.warning("dropping malformed chunk in %s at byte %d: %s", file_path, start_byte, exc)
        return None


def _trim_span(data: bytes, start: int, end: int) -> tuple[int, int]:
    """Drop leading blank lines and trailing whitespace from a byte span."""
    start = max(start, 0)
    end = min(end, len(data))
    while end > start and data[end - 1] in b" \t\r\n":
        end -= 1
    while start < end:
        cursor = start
        while cursor < end and data[cursor] in b" \t":
            cursor += 1
        if cursor < end and data[cursor : cursor + 1] == b"\r":
            cursor += 1
        if cursor < end and data[cursor : cursor + 1] == b"\n":
            start = cursor + 1
        else:
            break
    return start, end


def as_source_index(source: str | bytes | Path | SourceIndex) -> SourceIndex | None:
    """Coerce any accepted source form into a :class:`SourceIndex`, or ``None``.

    ``None`` means "this is not indexable UTF-8 source": an unreadable file or a
    binary blob. Both are bad input, so both degrade to an empty chunk list
    rather than raising (Rule 3).
    """
    if isinstance(source, SourceIndex):
        return source
    try:
        if isinstance(source, Path):
            return SourceIndex(source.read_bytes())
        if isinstance(source, bytes):
            return SourceIndex(source)
        return SourceIndex.from_text(source)
    except (OSError, UnicodeDecodeError) as exc:
        label = str(source) if isinstance(source, Path) else "<in-memory source>"
        _LOG.warning("unreadable source %s: %s", label, exc)
        return None


def _resolve(settings: Settings | None) -> Settings:
    """Fall back to the ambient profile so callers may omit ``settings``."""
    return settings if settings is not None else get_settings()


def max_chunks_per_file(settings: Settings) -> int:
    """Per-file emission cap, TestPlan.md TC-023's ``settings.max_chunks_per_file``.

    Read through ``getattr`` because the frozen ``Settings`` in ``config.py`` has
    no such field yet while TC-023 names one: the lookup picks the real field up
    the moment it is added, and until then the documented 2000 applies. Same
    contract as a Settings default -- one named constant, no literal at a
    callsite (AP-07).
    """
    return int(getattr(settings, "max_chunks_per_file", DEFAULT_MAX_CHUNKS_PER_FILE))


#: TC-023's stated cap, standing in until ``Settings`` grows the field.
DEFAULT_MAX_CHUNKS_PER_FILE = 2000

#: TC-024's stated AST depth cap, standing in until ``Settings`` grows the field.
DEFAULT_MAX_AST_DEPTH = 64


def max_ast_depth(settings: Settings) -> int:
    """AST nesting cap, TestPlan.md TC-024's ``settings.max_ast_depth``.

    Same ``getattr`` bridge and same reason as :func:`max_chunks_per_file`.
    """
    return int(getattr(settings, "max_ast_depth", DEFAULT_MAX_AST_DEPTH))


def split_text(
    source: str | bytes | Path | SourceIndex,
    file_path: str,
    version_id: str,
    settings: Settings | None = None,
    *,
    start_byte: int = 0,
    end_byte: int | None = None,
    kind: ChunkKind = ChunkKind.BLOCK,
    symbol: str | None = None,
    parent_symbol: str | None = None,
    is_exported: bool = False,
    imports: Sequence[str] | None = None,
    docstring: str | None = None,
    commit_sha: str | None = None,
    last_modified: str | None = None,
    budget: int = 0,
) -> list[Chunk]:
    """Fixed-window line splitter with overlap -- the third ladder rung.

    Accumulates whole lines until the estimated token count would exceed the
    target, cuts, and restarts :data:`FALLBACK_OVERLAP_LINES` lines back so a
    construct straddling a cut still appears whole in one fragment.

    One line that is *itself* over budget cannot be split this way, and that is
    not hypothetical: a minified bundle is one line of 1.2 MB (TC-023). Such a
    line is cut at character level, preferring a position just after ``;`` or
    ``}`` so the fragment edge still lands near a statement boundary, and the
    character offsets are converted to byte offsets by encoding the prefix --
    never by assuming one character is one byte.

    Args:
        source: File path, raw bytes, source text, or a prepared index.
        file_path: Repo-relative POSIX path recorded on every chunk.
        version_id: Logical version every emitted chunk belongs to.
        settings: Tunables; the ambient profile when omitted.
        start_byte: First byte of the region to split. Defaults to the file start.
        end_byte: One past the last byte of the region. Defaults to the file end.
        kind: Kind stamped on every fragment. ``BLOCK`` for residual and
            fallback fragments, which is what they are -- spans whose boundaries
            are not a single AST node (Schema.md section 3.3).
        symbol: Symbol stamped on every fragment; a ``#<n>`` fragment index is
            appended when more than one fragment is produced.
        parent_symbol: Enclosing symbol, required when ``kind`` is ``METHOD``.
        is_exported: Export flag propagated from the enclosing declaration.
        imports: File-level import specifiers; computed from the region when
            omitted.
        docstring: Docstring propagated to every fragment.
        commit_sha: Provenance, passed through to metadata.
        last_modified: Provenance, passed through to metadata.
        budget: Token ceiling per fragment; ``settings.chunk_target_tokens``
            when zero or negative.

    Returns:
        Fragments in source order. Empty when the region holds no code.
    """
    resolved = _resolve(settings)
    src = as_source_index(source)
    if src is None:
        return []
    path = normalise_file_path(file_path)
    limit = budget if budget > 0 else resolved.chunk_target_tokens
    region_end = len(src.data) if end_byte is None else min(end_byte, len(src.data))
    if region_end <= start_byte:
        return []
    file_imports = list(imports) if imports is not None else extract_imports(src.text)
    cap = max_chunks_per_file(resolved)

    spans = _window_spans(src, start_byte, region_end, limit)
    chunks: list[Chunk] = []
    for index, (span_start, span_end) in enumerate(spans, start=1):
        if len(chunks) >= cap:
            _LOG.warning("chunk cap %d reached in %s; remaining text not chunked", cap, path)
            break
        fragment_symbol = symbol if len(spans) == 1 or symbol is None else f"{symbol}#{index}"
        chunk = make_chunk(
            src,
            span_start,
            span_end,
            file_path=path,
            version_id=version_id,
            kind=kind,
            symbol=fragment_symbol,
            parent_symbol=parent_symbol,
            is_exported=is_exported,
            imports=file_imports,
            calls=extract_calls(src.data[span_start:span_end].decode("utf-8", errors="replace")),
            docstring=docstring,
            commit_sha=commit_sha,
            last_modified=last_modified,
        )
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def _window_spans(
    src: SourceIndex, start_byte: int, end_byte: int, budget: int
) -> list[tuple[int, int]]:
    """Byte spans of the line windows covering ``[start_byte, end_byte)``.

    Whole lines are accumulated until one more would break the budget. A line
    that alone breaks the budget is handed to :func:`_character_spans` -- which
    is the only case where a fragment edge is not a line edge.
    """
    first_line = src.line_of_byte(start_byte)
    last_line = src.line_of_byte(max(end_byte - 1, start_byte))

    def bounds(line: int) -> tuple[int, int]:
        return max(src.line_start_byte(line), start_byte), min(src.line_end_byte(line), end_byte)

    def cost(line: int) -> int:
        # Measured *including* the line terminator: the window's text is the
        # lines joined by their newlines, and a newline costs a token. Charging
        # only the content would under-count by one token per line and let a
        # window overrun the budget by its own line count.
        low = max(src.line_start_byte(line), start_byte)
        high = min(max(src.line_start_byte(line + 1), src.line_end_byte(line)), end_byte)
        if high <= low:
            return 0
        return estimate_tokens(src.data[low:high].decode("utf-8", errors="replace"), limit=budget)

    spans: list[tuple[int, int]] = []
    line = first_line
    while line <= last_line:
        head_tokens = cost(line)
        if head_tokens > budget:
            low, high = bounds(line)
            spans.extend(_character_spans(src, low, high, budget))
            line += 1
            continue
        tokens = head_tokens
        cursor = line + 1
        while cursor <= last_line:
            next_tokens = cost(cursor)
            if tokens + next_tokens > budget:
                break
            tokens += next_tokens
            cursor += 1
        spans.append((bounds(line)[0], bounds(cursor - 1)[1]))
        if cursor > last_line:
            break
        line = max(cursor - FALLBACK_OVERLAP_LINES, line + 1)
    return spans


def _character_spans(
    src: SourceIndex, start_byte: int, end_byte: int, budget: int
) -> list[tuple[int, int]]:
    """Cut one over-long line into byte spans, preferring statement-ish edges.

    Byte offsets are derived by encoding each piece rather than by assuming one
    character is one byte, so a minified bundle containing a unicode string
    literal still slices exactly.
    """
    text = src.data[start_byte:end_byte].decode("utf-8", errors="replace")
    spans: list[tuple[int, int]] = []
    offset = start_byte
    cursor = 0
    while cursor < len(text):
        length = prefix_within_budget(text[cursor:], budget)
        stop = min(cursor + max(length, 1), len(text))
        if stop < len(text):
            best = -1
            for marker in _CUT_PREFERENCE:
                best = max(best, text.rfind(marker, cursor + 1, stop))
            if best > cursor:
                stop = best + 1
        piece_bytes = len(text[cursor:stop].encode("utf-8"))
        spans.append((offset, offset + piece_bytes))
        offset += piece_bytes
        cursor = stop
    return spans


def regex_chunks(
    source: str | bytes | Path | SourceIndex,
    file_path: str,
    version_id: str,
    settings: Settings | None = None,
    *,
    commit_sha: str | None = None,
    last_modified: str | None = None,
) -> list[Chunk]:
    """Identifier-based splitter -- the rung TC-021 lands a broken file on.

    Scans for definition anchors (``function f``, ``class C``, ``const f = () =>``,
    ``method(...) {``) line by line and cuts a chunk at every anchor. It has no
    idea where a construct *ends*, so a chunk runs to the next anchor; that is a
    worse boundary than an AST node and is exactly why this rung sits below the
    AST one.

    Every fragment is ``ChunkKind.BLOCK``: its boundaries are not a single AST
    node, which is precisely Schema.md section 3.3's definition of ``BLOCK``, and
    TC-021 asserts it. ``calls`` is still populated, by regex, because a broken
    file's callees are still real retrieval signal.

    Returns an empty list when no anchor is found, which is the caller's signal
    to drop to :func:`split_text`.
    """
    resolved = _resolve(settings)
    src = as_source_index(source)
    if src is None:
        return []
    path = normalise_file_path(file_path)
    anchors = _find_anchors(src)
    if not anchors:
        return []

    file_imports = extract_imports(src.text)
    # A definition's leading comment belongs with the definition, not with the
    # statements above it -- otherwise every JSDoc block lands in the previous
    # chunk and every function loses its docstring.
    boundaries: list[int] = []
    for line, _ in anchors:
        start = _comment_block_start(src, line)
        boundaries.append(max(start, boundaries[-1] + 1) if boundaries else start)
    segments: list[tuple[int, int, str | None]] = []
    if boundaries[0] > 1:
        segments.append((1, boundaries[0] - 1, None))
    for index, (_, name) in enumerate(anchors):
        end_line = boundaries[index + 1] - 1 if index + 1 < len(anchors) else src.line_count
        segments.append((boundaries[index], end_line, name))

    merged = _merge_subfloor_segments(src, segments, resolved.chunk_min_tokens)
    cap = max_chunks_per_file(resolved)
    chunks: list[Chunk] = []
    for start_line, end_line, name in merged:
        if len(chunks) >= cap:
            _LOG.warning("chunk cap %d reached in %s; remaining text not chunked", cap, path)
            break
        span_start = src.line_start_byte(start_line)
        span_end = src.line_end_byte(end_line)
        segment_text = src.data[span_start:span_end].decode("utf-8", errors="replace")
        if exceeds_budget(segment_text, resolved.chunk_target_tokens):
            chunks.extend(
                split_text(
                    src,
                    path,
                    version_id,
                    resolved,
                    start_byte=span_start,
                    end_byte=span_end,
                    symbol=name,
                    imports=file_imports,
                    commit_sha=commit_sha,
                    last_modified=last_modified,
                )
            )
            continue
        chunk = make_chunk(
            src,
            span_start,
            span_end,
            file_path=path,
            version_id=version_id,
            kind=ChunkKind.BLOCK,
            symbol=name,
            imports=file_imports,
            calls=extract_calls(segment_text),
            docstring=_leading_comment(segment_text),
            commit_sha=commit_sha,
            last_modified=last_modified,
        )
        if chunk is not None:
            chunks.append(chunk)
    return chunks[:cap]


def _find_anchors(src: SourceIndex) -> list[tuple[int, str | None]]:
    """Definition-looking lines, as ``(1-indexed line, symbol or None)``."""
    anchors: list[tuple[int, str | None]] = []
    for line_number in range(1, src.line_count + 1):
        line = src.line_text(line_number)
        if not line.strip():
            continue
        match = _ANCHOR_RE.match(line)
        if match is None:
            continue
        name = match.group("fn") or match.group("cls") or match.group("var") or match.group("prop")
        if name in _NON_CALL_KEYWORDS:
            continue
        anchors.append((line_number, name))
    return anchors


def _comment_block_start(src: SourceIndex, line: int) -> int:
    """First line of the comment block immediately above ``line``, else ``line``."""
    start = line
    cursor = line - 1
    while cursor >= 1:
        stripped = src.line_text(cursor).strip()
        if not stripped.startswith(("//", "*", "/*")):
            break
        start = cursor
        cursor -= 1
    return start


def _merge_subfloor_segments(
    src: SourceIndex, segments: Sequence[tuple[int, int, str | None]], floor: int
) -> list[tuple[int, int, str | None]]:
    """Absorb under-floor segments into the previous one (section 4.4.1 step 4)."""
    merged: list[tuple[int, int, str | None]] = []
    pending: tuple[int, int, str | None] | None = None
    for start_line, end_line, name in segments:
        if pending is not None:
            # A sub-floor *leading* segment has no previous chunk to fall into,
            # so it falls forward into the next one instead. Dropping it would
            # lose the file's import block, which is usually what it is.
            start_line, name = pending[0], name or pending[2]
            pending = None
        span = src.data[src.line_start_byte(start_line) : src.line_end_byte(end_line)]
        text = span.decode("utf-8", errors="replace")
        if not text.strip():
            continue
        if below_floor(text, floor):
            if merged:
                previous_start, _, previous_name = merged[-1]
                merged[-1] = (previous_start, end_line, previous_name)
            else:
                pending = (start_line, end_line, name)
            continue
        merged.append((start_line, end_line, name))
    if pending is not None:
        merged.append(pending)
    return merged


def module_chunk(
    source: str | bytes | Path | SourceIndex,
    file_path: str,
    version_id: str,
    settings: Settings | None = None,
    *,
    commit_sha: str | None = None,
    last_modified: str | None = None,
) -> list[Chunk]:
    """Whole file as one ``ChunkKind.MODULE`` chunk -- the terminal ladder rung.

    Also the *correct*, non-degraded answer for a file with no extractable
    top-level definitions (Schema.md section 3.3: config objects, barrel
    re-export files, pure side-effect scripts), which is why TC-022 expects
    exactly one ``MODULE`` chunk from a constants file and zero from an empty
    one. ``symbol`` stays ``None``: a module has no declared name.
    """
    src = as_source_index(source)
    if src is None or not src.text.strip():
        return []
    path = normalise_file_path(file_path)
    chunk = make_chunk(
        src,
        0,
        len(src.data),
        file_path=path,
        version_id=version_id,
        kind=ChunkKind.MODULE,
        imports=extract_imports(src.text),
        calls=extract_calls(src.text),
        docstring=_leading_comment(src.text),
        commit_sha=commit_sha,
        last_modified=last_modified,
    )
    return [chunk] if chunk is not None else []


def _leading_comment(text: str) -> str | None:
    """The file's leading comment block, if it opens with one."""
    stripped = text.lstrip()
    if stripped.startswith("/*"):
        end = stripped.find("*/")
        if end != -1:
            return strip_comment_markers(stripped[: end + 2])
        return None
    if stripped.startswith("//"):
        lines: list[str] = []
        for line in stripped.splitlines():
            if not line.lstrip().startswith("//"):
                break
            lines.append(line)
        return strip_comment_markers("\n".join(lines))
    return None


def chunk_file_fallback(
    source: str | bytes | Path | SourceIndex,
    file_path: str,
    version_id: str,
    settings: Settings | None = None,
    *,
    reason: str,
    commit_sha: str | None = None,
    last_modified: str | None = None,
) -> list[Chunk]:
    """Run the whole sub-AST ladder and return the first rung that produces chunks.

    ``regex_chunks`` -> ``split_text`` -> ``module_chunk``. Each descent is
    logged through :func:`axiom.core.logging.log_degradation`, because Rule 3
    requires a degrade to be loud -- a silent fallback is indistinguishable from
    a silent wrong answer.

    The terminal rung cannot fail on non-empty input, so this never raises
    ``DegradationExhaustedError``: an empty return means the file held no code,
    not that the ladder ran out.

    Args:
        reason: Why the AST rung was skipped or failed, quoted into the log.
    """
    resolved = _resolve(settings)
    src = as_source_index(source)
    if src is None or not src.text.strip():
        return []

    chunks = regex_chunks(
        src, file_path, version_id, resolved, commit_sha=commit_sha, last_modified=last_modified
    )
    if chunks:
        log_degradation(_LOG, f"chunker:{file_path}", reason, "regex identifier splitter")
        return chunks

    if not exceeds_budget(src.text, resolved.chunk_target_tokens):
        # No definition anchors and small enough to hold whole: this file *is* a
        # module (Schema.md section 3.3 -- a config object, a barrel re-export, a
        # side-effect script), and TC-022 wants exactly one MODULE chunk for it.
        # The line-window rung is skipped rather than demoted: windowing a file
        # that already fits would cut it for no reason.
        log_degradation(_LOG, f"chunker:{file_path}", reason, "whole-file MODULE chunk")
        return module_chunk(
            src,
            file_path,
            version_id,
            resolved,
            commit_sha=commit_sha,
            last_modified=last_modified,
        )

    chunks = split_text(
        src,
        file_path,
        version_id,
        resolved,
        commit_sha=commit_sha,
        last_modified=last_modified,
    )
    if chunks:
        log_degradation(_LOG, f"chunker:{file_path}", reason, "fixed-window line split")
        return chunks

    log_degradation(_LOG, f"chunker:{file_path}", reason, "whole-file MODULE chunk")
    return module_chunk(
        src, file_path, version_id, resolved, commit_sha=commit_sha, last_modified=last_modified
    )
