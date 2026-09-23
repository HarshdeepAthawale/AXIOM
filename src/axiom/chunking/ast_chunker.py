"""AST-boundary chunking -- the top rung of the chunker's degradation ladder.

Implements TechSpecifications.md section 4.4.1 against ``tree-sitter-javascript``:
walk the concrete syntax tree, cut chunks at node boundaries, split oversized
functions at statement boundaries with a one-statement overlap, absorb sub-floor
chunks into their parent, and mint both chunk identities from
:mod:`axiom.core.hashing`. This module is the *only* place in Axiom where a
``chunk_id`` is created (Rules.md Rule 1).

Two properties are load-bearing and are asserted by the smoke test below the
package as well as by TestPlan.md:

* **Byte equivalence.** ``source_bytes[start_byte:end_byte].decode("utf-8")``
  equals ``chunk.text``, on multibyte source too. Achieved by never building a
  chunk's text from anything but a byte slice -- see
  :func:`axiom.chunking.fallback.make_chunk`.
* **A file never aborts the index.** Every failure path -- ``tree-sitter`` not
  installed, a parse error, an unreadable file, a byte span a validator rejects
  -- degrades to a lower rung and logs, per Rules.md Rule 3 and NFR-07.

``tree_sitter`` and ``tree_sitter_javascript`` are optional extras and are
imported inside :func:`_load_language`, never at module scope. ``import
axiom.chunking.ast_chunker`` therefore succeeds on a bare install, which is what
keeps the demo alive on a machine where the ``structural`` extra never resolved.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axiom.config import Settings, get_settings
from axiom.core.logging import get_logger
from axiom.schema import Chunk, ChunkKind

from .fallback import (
    SourceIndex,
    as_source_index,
    chunk_file_fallback,
    make_chunk,
    max_ast_depth,
    max_chunks_per_file,
    module_chunk,
    normalise_file_path,
    split_text,
    strip_comment_markers,
)
from .tokens import below_floor, estimate_tokens, exceeds_budget

_LOG = get_logger("chunking.ast_chunker")

#: Extensions treated as JavaScript source. NG-04 limits the hackathon scope to
#: one grammar, so this list is also the definition of "indexable" for
#: :func:`chunk_repo`.
SOURCE_EXTENSIONS: tuple[str, ...] = (".js", ".mjs", ".cjs", ".jsx")

#: Directories never walked, from NG-26 (build output, vendored code, and our
#: own index). The ignore list belongs in the profile configs per NG-26, but no
#: profile declares one and ``Settings`` has no field for it, so this is the
#: standing default -- :func:`chunk_repo` takes an override argument for the day
#: the configs grow one.
IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".axiom",
        ".git",
        ".hg",
        ".next",
        ".nuxt",
        ".svn",
        ".venv",
        "__pycache__",
        "bower_components",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "out",
        "target",
        "vendor",
    }
)

#: Filename suffixes never chunked: build artefacts that dominate byte count and
#: carry near-zero retrieval value (NG-26).
IGNORED_SUFFIXES: tuple[str, ...] = (".min.js", ".min.mjs", ".min.cjs", ".bundle.js", ".map")

#: Overlap between consecutive oversize-split fragments, in statements.
#: TechSpecifications.md section 5.4 lists this as a "(chunker constant)" with no
#: env var, and marks it ``PLACEHOLDER`` under OQ-09 alongside the token bounds.
OVERSIZE_OVERLAP_STATEMENTS = 1

_FUNCTION_NODES = frozenset(
    {
        "arrow_function",
        "function",
        "function_declaration",
        "function_expression",
        "generator_function",
        "generator_function_declaration",
    }
)
_CLASS_NODES = frozenset({"class", "class_declaration", "class_expression"})
_DECLARATION_NODES = frozenset({"lexical_declaration", "variable_declaration"})
_CLASS_MEMBER_NODES = frozenset(
    {"field_definition", "method_definition", "public_field_definition"}
)

#: Cached ``tree_sitter.Language``. ``None`` means "tried and failed"; the key is
#: absent while untried. Cached because building the ``Language`` costs a dlopen
#: and a repo has thousands of files; ``Parser`` objects are *not* cached, since
#: tree-sitter parsers are not safe to share across threads.
_LANGUAGE_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class _Definition:
    """One syntactic definition found during the walk.

    ``host`` is the node whose byte span becomes the chunk -- the
    ``export_statement`` wrapper rather than the bare declaration when there is
    one, so the chunk text shows the reader that the symbol is exported.
    ``node`` is the declaration itself, which is what gets split when oversized.
    """

    host: Any
    node: Any
    kind: ChunkKind
    symbol: str | None
    is_exported: bool
    class_body: Any = None


def chunk_file(
    source: str | bytes | Path | SourceIndex,
    file_path: str,
    version_id: str,
    settings: Settings | None = None,
    *,
    commit_sha: str | None = None,
    last_modified: str | None = None,
) -> list[Chunk]:
    """Chunk one JavaScript file into :class:`~axiom.schema.Chunk` records.

    The canonical chunker entrypoint named throughout Appflow.md and Rules.md.
    Runs the ladder top-down: AST node boundaries, then statement-boundary
    splitting for oversized functions, then -- only if the tree is unusable --
    the regex, line-window, and whole-file rungs in
    :mod:`axiom.chunking.fallback`.

    Args:
        source: A :class:`~pathlib.Path` is read from disk; ``str`` and ``bytes``
            are taken as the source itself. The distinction is by type, not by
            inspection, so a one-line file and a path are never confused.
        file_path: Repo-relative POSIX path recorded on every chunk. Repaired
            rather than rejected if it arrives absolute or Windows-style.
        version_id: Logical version every emitted chunk belongs to.
        settings: Tunables; the ambient profile when omitted.
        commit_sha: Full 40-char sha of the commit being indexed, when known.
        last_modified: ISO-8601 date the file last changed, when known.

    Returns:
        Chunks in source order, deduplicated by ``chunk_id``. Empty for an empty
        or blank file (TC-022) and for a file that cannot be read or decoded --
        never an exception, whatever the input (Rules.md Rule 3).
    """
    resolved = settings if settings is not None else get_settings()
    src = as_source_index(source)
    if src is None or not src.text.strip():
        return []
    path = normalise_file_path(file_path)

    language, reason = _load_language()
    if language is None:
        return chunk_file_fallback(
            src,
            path,
            version_id,
            resolved,
            reason=reason,
            commit_sha=commit_sha,
            last_modified=last_modified,
        )

    tree = _parse(language, src)
    if tree is None:
        return chunk_file_fallback(
            src,
            path,
            version_id,
            resolved,
            reason="tree-sitter parse raised",
            commit_sha=commit_sha,
            last_modified=last_modified,
        )
    if tree.root_node.has_error:
        return chunk_file_fallback(
            src,
            path,
            version_id,
            resolved,
            reason="tree-sitter parse error, unsupported syntax",
            commit_sha=commit_sha,
            last_modified=last_modified,
        )

    walker = _Walker(
        src=src,
        file_path=path,
        version_id=version_id,
        settings=resolved,
        commit_sha=commit_sha,
        last_modified=last_modified,
    )
    return walker.run(tree.root_node)


def chunk_repo(
    root: str | Path,
    version_id: str,
    settings: Settings | None = None,
    *,
    commit_sha: str | None = None,
    last_modified: str | None = None,
    extensions: Sequence[str] | None = None,
    ignore_dirs: Iterable[str] | None = None,
) -> list[Chunk]:
    """Chunk every indexable source file under ``root``.

    Files are visited in sorted path order and each one's failure is contained:
    a file that cannot be read, decoded, or parsed contributes whatever its own
    ladder produces (often zero chunks) and the walk continues. That containment
    is the whole point -- NFR-07 says one bad file must not cost the index.

    Args:
        root: Repository root. Chunk paths are recorded relative to it.
        version_id: Logical version every emitted chunk belongs to.
        settings: Tunables; the ambient profile when omitted.
        commit_sha: Provenance stamped on every chunk.
        last_modified: Provenance stamped on every chunk.
        extensions: Override for :data:`SOURCE_EXTENSIONS`.
        ignore_dirs: Override for :data:`IGNORED_DIRECTORIES`.

    Returns:
        All chunks, grouped by file in sorted path order.
    """
    resolved = settings if settings is not None else get_settings()
    base = Path(root)
    chunks: list[Chunk] = []
    for path in discover_source_files(base, extensions=extensions, ignore_dirs=ignore_dirs):
        relative = path.relative_to(base).as_posix()
        chunks.extend(
            chunk_file(
                path,
                relative,
                version_id,
                resolved,
                commit_sha=commit_sha,
                last_modified=last_modified,
            )
        )
    _LOG.info(
        "chunked %d file(s) under %s into %d chunk(s)",
        len({chunk.location.file_path for chunk in chunks}),
        base,
        len(chunks),
    )
    return chunks


def discover_source_files(
    root: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    ignore_dirs: Iterable[str] | None = None,
) -> list[Path]:
    """List indexable source files under ``root``, in sorted order.

    Sorted because chunk order becomes FAISS row order becomes ``chunks.jsonl``
    line order; an index built twice from one tree must come out byte-identical
    (NFR-08), and ``os.walk`` order is filesystem-dependent.
    """
    base = Path(root)
    suffixes = tuple(extensions) if extensions is not None else SOURCE_EXTENSIONS
    skipped = frozenset(ignore_dirs) if ignore_dirs is not None else IGNORED_DIRECTORIES
    found: list[Path] = []
    for directory, subdirectories, filenames in os.walk(base):
        subdirectories[:] = sorted(d for d in subdirectories if d not in skipped)
        for filename in sorted(filenames):
            if not filename.endswith(suffixes):
                continue
            if filename.endswith(IGNORED_SUFFIXES):
                continue
            found.append(Path(directory) / filename)
    return found


def _load_language() -> tuple[Any | None, str]:
    """Load the JavaScript grammar lazily, returning ``(language, reason)``.

    ``language`` is ``None`` when the optional ``structural`` extra is not
    installed or the grammar will not bind; ``reason`` then carries the text that
    goes into the degradation log. Both outcomes are cached, so a repo of ten
    thousand files pays the import attempt once.
    """
    if "language" in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE["language"], _LANGUAGE_CACHE["reason"]
    try:
        import tree_sitter
        import tree_sitter_javascript
    except ImportError as exc:
        # The expected bare-install path: the `structural` extra never resolved.
        _LANGUAGE_CACHE["language"] = None
        _LANGUAGE_CACHE["reason"] = f"tree-sitter unavailable ({exc})"
        return None, _LANGUAGE_CACHE["reason"]
    try:
        language = tree_sitter.Language(tree_sitter_javascript.language())
    except (TypeError, ValueError) as exc:
        # tree-sitter < 0.22 wants a name alongside the pointer.
        try:
            language = tree_sitter.Language(tree_sitter_javascript.language(), "javascript")
        except Exception:  # any binding failure reads the same as "no grammar"
            _LANGUAGE_CACHE["language"] = None
            _LANGUAGE_CACHE["reason"] = f"tree-sitter grammar would not bind ({exc})"
            return None, _LANGUAGE_CACHE["reason"]
    _LANGUAGE_CACHE["language"] = language
    _LANGUAGE_CACHE["reason"] = ""
    return language, ""


def _parse(language: Any, src: SourceIndex) -> Any | None:
    """Parse bytes into a tree, or ``None`` if the parser itself fails.

    A fresh ``Parser`` per file: tree-sitter parsers hold mutable state and are
    not safe to share across the threads an index build may use, and
    constructing one is cheap next to the parse.
    """
    try:
        import tree_sitter

        try:
            parser = tree_sitter.Parser(language)
        except TypeError:  # older bindings: language is set, not injected
            parser = tree_sitter.Parser()
            try:
                parser.language = language
            except AttributeError:
                parser.set_language(language)
        return parser.parse(src.data)
    except Exception as exc:  # a parser crash degrades like any other bad input
        _LOG.warning("tree-sitter parse failed: %s", exc)
        return None


class _Walker:
    """Single-file AST walk. One instance per file; not reusable, not shared.

    Holds the per-file facts every chunk needs (imports, export table, caps) so
    the emission helpers stay readable instead of threading nine arguments each.
    """

    def __init__(
        self,
        *,
        src: SourceIndex,
        file_path: str,
        version_id: str,
        settings: Settings,
        commit_sha: str | None,
        last_modified: str | None,
    ) -> None:
        self.src = src
        self.file_path = file_path
        self.version_id = version_id
        self.settings = settings
        self.commit_sha = commit_sha
        self.last_modified = last_modified
        self.target = settings.chunk_target_tokens
        self.floor = settings.chunk_min_tokens
        self.cap = max_chunks_per_file(settings)
        self.max_depth = max_ast_depth(settings)
        self.imports: list[str] = []
        self.exported_names: set[str] = set()
        self.chunks: list[Chunk] = []
        self.definitions_seen = 0

    # -- driver ----------------------------------------------------------

    def run(self, root: Any) -> list[Chunk]:
        """Walk ``root`` and return the file's chunks in source order."""
        self.imports = self._collect_imports(root)
        self.exported_names = self._collect_exported_names(root)

        residual: list[Any] = []
        for node in root.children:
            definition = self._classify(node)
            if definition is None or self._is_sub_floor(definition.host):
                # Sub-floor declarations are not dropped: they fall into the
                # residual run so their text still reaches the index inside a
                # neighbouring BLOCK (section 4.4.1 step 4).
                residual.append(node)
                continue
            self._flush_residual(residual, trailing_comments_belong_to_next=True)
            residual = []
            self.definitions_seen += 1
            self._emit_definition(definition, depth=0, parent_symbol=None)
        self._flush_residual(residual, trailing_comments_belong_to_next=False)

        if self.definitions_seen == 0:
            # Schema.md section 3.3: a file with no extractable top-level
            # definitions -- a config object, a barrel re-export, a side-effect
            # script -- *is* a MODULE, not a degraded parse. TC-022 asserts
            # exactly one chunk for such a file.
            return self._emit_module()
        return self._finalise()

    def _finalise(self) -> list[Chunk]:
        """Sort, deduplicate, and cap. The only exit from :meth:`run`.

        Sorted by span, parent-before-child, then by ``chunk_id`` so the order is
        a total order with no dependence on dict or set iteration (Rules.md
        AP-05). Deduplicated because ``chunk_id`` is the primary key of every
        downstream index, and two identical spans would collide in
        ``dense.idmap.json``.
        """
        ordered = sorted(
            self.chunks,
            key=lambda c: (c.location.start_byte, -c.location.end_byte, c.chunk_id),
        )
        seen: set[str] = set()
        unique: list[Chunk] = []
        for chunk in ordered:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique.append(chunk)
        return unique

    def _emit_module(self) -> list[Chunk]:
        """Emit a file with no definitions as a MODULE chunk, or split it if huge."""
        if exceeds_budget(self.src.text, self.target):
            _LOG.debug(
                "%s has no definitions and exceeds the chunk target; windowing", self.file_path
            )
            return split_text(
                self.src,
                self.file_path,
                self.version_id,
                self.settings,
                imports=self.imports,
                commit_sha=self.commit_sha,
                last_modified=self.last_modified,
            )
        return module_chunk(
            self.src,
            self.file_path,
            self.version_id,
            self.settings,
            commit_sha=self.commit_sha,
            last_modified=self.last_modified,
        )

    # -- emission --------------------------------------------------------

    def _emit_definition(
        self, definition: _Definition, *, depth: int, parent_symbol: str | None
    ) -> None:
        """Emit one definition, splitting, recursing, or bailing out as required."""
        if len(self.chunks) >= self.cap:
            return
        if depth >= self.max_depth:
            # TC-024: beyond the nesting cap the subtree is one BLOCK, and the
            # walk stops descending -- which is also what keeps a pathological
            # file from costing unbounded time.
            _LOG.warning(
                "%s: AST nesting beyond %d levels at %s; emitting one BLOCK chunk",
                self.file_path,
                self.max_depth,
                definition.symbol or "<anonymous>",
            )
            self._add(
                definition.host.start_byte,
                definition.host.end_byte,
                kind=ChunkKind.BLOCK,
                symbol=definition.symbol,
                parent_symbol=parent_symbol,
                is_exported=definition.is_exported,
                calls=self._calls_in(definition.host),
                docstring=self._docstring_for(definition.host),
            )
            return

        docstring = self._docstring_for(definition.host)
        if definition.kind is ChunkKind.CLASS:
            self._emit_class(
                definition, depth=depth, parent_symbol=parent_symbol, docstring=docstring
            )
            return

        text = self._slice(definition.host)
        kind = definition.kind
        if kind is ChunkKind.METHOD and not parent_symbol:
            # ChunkMetadata refuses a parentless METHOD, and it is right to: a
            # method with no owning class is a classification bug, not input.
            kind = ChunkKind.FUNCTION
        if exceeds_budget(text, self.target):
            self._emit_split(definition, parent_symbol=parent_symbol, docstring=docstring)
        else:
            self._add(
                definition.host.start_byte,
                definition.host.end_byte,
                kind=kind,
                symbol=definition.symbol,
                parent_symbol=parent_symbol,
                is_exported=definition.is_exported,
                calls=self._calls_in(definition.node),
                docstring=docstring,
            )
        self._emit_nested(definition, depth=depth, parent_symbol=definition.symbol or parent_symbol)

    def _emit_class(
        self,
        definition: _Definition,
        *,
        depth: int,
        parent_symbol: str | None,
        docstring: str | None,
    ) -> None:
        """Emit the class chunk plus one chunk per method.

        Schema.md section 3.3 asks for a header+fields chunk *and*, when every
        method fits, a whole-class chunk. TC-013 counts exactly one ``CLASS``
        chunk for a four-method class whose methods all fit, so the two cannot
        both be emitted: the whole-class chunk *is* the header-and-fields chunk
        when the class fits, and only an oversized class falls back to a header
        span that stops at its first member. That reading satisfies both
        documents and avoids indexing the class body three times over.
        """
        host = definition.host
        members = self._class_members(definition.class_body)
        whole_text = self._slice(host)
        if exceeds_budget(whole_text, self.target) and members:
            shell_end = members[0].start_byte
        else:
            shell_end = host.end_byte
        self._add(
            host.start_byte,
            shell_end,
            kind=ChunkKind.CLASS,
            symbol=definition.symbol,
            parent_symbol=parent_symbol,
            is_exported=definition.is_exported,
            calls=self._calls_in_range(host, host.start_byte, shell_end),
            docstring=docstring,
        )
        for member in members:
            member_definition = self._classify_member(
                member, owner=definition.symbol, owner_exported=definition.is_exported
            )
            if member_definition is None or self._is_sub_floor(member):
                continue
            self._emit_definition(
                member_definition, depth=depth + 1, parent_symbol=definition.symbol
            )

    def _emit_split(
        self,
        definition: _Definition,
        *,
        parent_symbol: str | None,
        docstring: str | None,
    ) -> None:
        """Split an oversized function at statement boundaries (section 4.4.1 step 3).

        Fragment 1 starts at the declaration's own first byte and the last
        fragment ends at its last, so the union of the fragments covers the whole
        function with no gaps -- TC-019 checks exactly that. Consecutive
        fragments repeat one statement, which is the declared, bounded
        redundancy: never more than one statement, never spanning more than two
        adjacent fragments.
        """
        body = definition.node.child_by_field_name("body")
        statements = (
            list(body.named_children) if body is not None and body.type == "statement_block" else []
        )
        if len(statements) < 2:
            # Nothing to cut on: a concise arrow body, or one enormous
            # statement. Statement boundaries do not exist here, so the line
            # window is the honest fallback rather than a fake split.
            self.chunks.extend(
                split_text(
                    self.src,
                    self.file_path,
                    self.version_id,
                    self.settings,
                    start_byte=definition.host.start_byte,
                    end_byte=definition.host.end_byte,
                    symbol=definition.symbol,
                    parent_symbol=parent_symbol,
                    is_exported=definition.is_exported,
                    imports=self.imports,
                    docstring=docstring,
                    commit_sha=self.commit_sha,
                    last_modified=self.last_modified,
                )
            )
            return

        header_tokens = estimate_tokens(
            self._slice_bytes(definition.host.start_byte, statements[0].start_byte),
            limit=self.target,
        )
        fragments = self._group_statements(statements, header_tokens=header_tokens)
        symbol = definition.symbol
        for index, group in enumerate(fragments, start=1):
            first = index == 1
            last = index == len(fragments)
            start = definition.host.start_byte if first else group[0].start_byte
            end = definition.host.end_byte if last else group[-1].end_byte
            fragment_symbol = f"{symbol}#{index}" if symbol else None
            if len(group) == 1 and exceeds_budget(self._slice_bytes(start, end), self.target):
                self.chunks.extend(
                    split_text(
                        self.src,
                        self.file_path,
                        self.version_id,
                        self.settings,
                        start_byte=start,
                        end_byte=end,
                        symbol=fragment_symbol,
                        parent_symbol=symbol,
                        is_exported=definition.is_exported,
                        imports=self.imports,
                        docstring=docstring,
                        commit_sha=self.commit_sha,
                        last_modified=self.last_modified,
                    )
                )
                continue
            self._add(
                start,
                end,
                kind=ChunkKind.BLOCK,
                symbol=fragment_symbol,
                parent_symbol=symbol or parent_symbol,
                is_exported=definition.is_exported,
                calls=self._calls_in_nodes(group),
                docstring=docstring,
            )

    def _group_statements(
        self, statements: Sequence[Any], *, header_tokens: int = 0
    ) -> list[list[Any]]:
        """Pack statements into token-budgeted groups with a one-statement overlap.

        A statement's cost is measured from its first byte to the *next*
        statement's first byte, so the newlines and indentation that will end up
        inside the fragment are paid for. Measuring the bare statement span
        instead under-counts by about a token per line, which is enough to push
        every fragment of a long function over the target.

        ``header_tokens`` pre-charges the function signature, which fragment one
        carries in addition to its statements.
        """
        costs: list[int] = []
        for index, statement in enumerate(statements):
            stop = (
                statements[index + 1].start_byte
                if index + 1 < len(statements)
                else statement.end_byte
            )
            costs.append(
                estimate_tokens(self._slice_bytes(statement.start_byte, stop), limit=self.target)
            )

        groups: list[list[Any]] = []
        current: list[Any] = []
        tokens = header_tokens
        for index, statement in enumerate(statements):
            cost = costs[index]
            if current and tokens + cost > self.target:
                groups.append(current)
                overlap = list(range(max(index - OVERSIZE_OVERLAP_STATEMENTS, 0), index))
                current = [statements[position] for position in overlap]
                tokens = sum(costs[position] for position in overlap)
            current.append(statement)
            tokens += cost
        if current:
            groups.append(current)
        return groups

    def _emit_nested(
        self, definition: _Definition, *, depth: int, parent_symbol: str | None
    ) -> None:
        """Emit named functions lexically nested inside ``definition``.

        Section 4.4.1's table asks for "one chunk per top-level or nested
        standalone function". Only *named* nested functions qualify here: an
        anonymous inline callback has no symbol to retrieve by, and emitting one
        would duplicate its parent's text for no addressable gain. Sub-floor
        nested functions are skipped for the same reason (step 4) -- their text
        already lives inside the parent chunk.
        """
        for child in self._nested_definitions(definition.node):
            if len(self.chunks) >= self.cap:
                return
            if child.symbol is None or self._is_sub_floor(child.host):
                continue
            self._emit_definition(child, depth=depth + 1, parent_symbol=parent_symbol)

    def _flush_residual(
        self, nodes: Sequence[Any], *, trailing_comments_belong_to_next: bool
    ) -> None:
        """Emit a contiguous run of top-level non-definition statements as BLOCKs.

        Section 4.4.1 step 5. Trailing comments are handed to the definition that
        follows instead of being buried at the end of a residual run -- otherwise
        a function's JSDoc would land in the previous chunk and the function's
        own ``docstring`` would be empty.
        """
        run = list(nodes)
        if trailing_comments_belong_to_next:
            while run and run[-1].type == "comment":
                run.pop()
        if not run or len(self.chunks) >= self.cap:
            return
        start, end = run[0].start_byte, run[-1].end_byte
        text = self._slice_bytes(start, end)
        if not text.strip() or below_floor(text, self.floor):
            return
        if exceeds_budget(text, self.target):
            self.chunks.extend(
                split_text(
                    self.src,
                    self.file_path,
                    self.version_id,
                    self.settings,
                    start_byte=start,
                    end_byte=end,
                    imports=self.imports,
                    commit_sha=self.commit_sha,
                    last_modified=self.last_modified,
                )
            )
            return
        self._add(
            start,
            end,
            kind=ChunkKind.BLOCK,
            symbol=None,
            parent_symbol=None,
            is_exported=False,
            calls=self._calls_in_nodes(run),
            docstring=None,
        )

    def _add(
        self,
        start_byte: int,
        end_byte: int,
        *,
        kind: ChunkKind,
        symbol: str | None,
        parent_symbol: str | None,
        is_exported: bool,
        calls: Sequence[str],
        docstring: str | None,
    ) -> None:
        """Append one chunk, respecting the per-file cap."""
        if len(self.chunks) >= self.cap:
            return
        chunk = make_chunk(
            self.src,
            start_byte,
            end_byte,
            file_path=self.file_path,
            version_id=self.version_id,
            kind=kind,
            symbol=symbol,
            parent_symbol=parent_symbol,
            is_exported=is_exported,
            imports=self.imports,
            calls=calls,
            docstring=docstring,
            commit_sha=self.commit_sha,
            last_modified=self.last_modified,
        )
        if chunk is not None:
            self.chunks.append(chunk)
            if len(self.chunks) == self.cap:
                _LOG.warning(
                    "%s hit the %d-chunk cap; the rest of the file is not chunked",
                    self.file_path,
                    self.cap,
                )

    # -- classification --------------------------------------------------

    def _classify(self, node: Any) -> _Definition | None:
        """Map a statement node to a definition, or ``None`` if it is not one."""
        node_type = node.type
        if node_type == "export_statement":
            for child in node.named_children:
                inner = self._classify(child)
                if inner is not None:
                    return _Definition(
                        host=node,
                        node=inner.node,
                        kind=inner.kind,
                        symbol=inner.symbol,
                        is_exported=True,
                        class_body=inner.class_body,
                    )
            return None
        if node_type in _FUNCTION_NODES:
            symbol = self._field_text(node, "name")
            return _Definition(
                host=node,
                node=node,
                kind=ChunkKind.FUNCTION,
                symbol=symbol,
                is_exported=self._is_exported(symbol),
            )
        if node_type in _CLASS_NODES:
            symbol = self._field_text(node, "name")
            return _Definition(
                host=node,
                node=node,
                kind=ChunkKind.CLASS,
                symbol=symbol,
                is_exported=self._is_exported(symbol),
                class_body=node.child_by_field_name("body"),
            )
        if node_type in _DECLARATION_NODES:
            return self._classify_declaration(node)
        if node_type == "expression_statement":
            return self._classify_assignment(node)
        return None

    def _classify_declaration(self, node: Any) -> _Definition | None:
        """``const handler = () => {}`` and friends: a binding with a callable value."""
        for declarator in node.named_children:
            if declarator.type != "variable_declarator":
                continue
            value = declarator.child_by_field_name("value")
            if value is None:
                continue
            symbol = self._field_text(declarator, "name")
            if value.type in _FUNCTION_NODES:
                return _Definition(
                    host=node,
                    node=value,
                    kind=ChunkKind.FUNCTION,
                    symbol=symbol,
                    is_exported=self._is_exported(symbol),
                )
            if value.type in _CLASS_NODES:
                return _Definition(
                    host=node,
                    node=value,
                    kind=ChunkKind.CLASS,
                    symbol=symbol,
                    is_exported=self._is_exported(symbol),
                    class_body=value.child_by_field_name("body"),
                )
        return None

    def _classify_assignment(self, node: Any) -> _Definition | None:
        """``Agent.prototype.run = function () {}`` and ``exports.x = () => {}``."""
        for child in node.named_children:
            if child.type != "assignment_expression":
                continue
            value = child.child_by_field_name("right")
            target = child.child_by_field_name("left")
            if value is None or target is None:
                continue
            if value.type not in _FUNCTION_NODES and value.type not in _CLASS_NODES:
                continue
            symbol = (
                self._field_text(target, "property")
                if target.type == "member_expression"
                else self._slice(target)
            )
            is_class = value.type in _CLASS_NODES
            return _Definition(
                host=node,
                node=value,
                kind=ChunkKind.CLASS if is_class else ChunkKind.FUNCTION,
                symbol=symbol,
                is_exported=self._is_exported(symbol) or self._targets_commonjs_export(target),
                class_body=value.child_by_field_name("body") if is_class else None,
            )
        return None

    def _classify_member(
        self, member: Any, *, owner: str | None, owner_exported: bool
    ) -> _Definition | None:
        """Map a class-body member to a METHOD definition.

        A method inherits its class's export flag. It does not leave the module
        under its own name, but ``is_exported`` exists to drive "where is X
        used" resolution (Schema.md section 5), and a method of an exported
        class is reachable from outside the module exactly as the class is.
        """
        exported = owner_exported or self._is_exported(owner)
        if member.type == "method_definition":
            return _Definition(
                host=member,
                node=member,
                kind=ChunkKind.METHOD if owner else ChunkKind.FUNCTION,
                symbol=self._field_text(member, "name"),
                is_exported=exported,
            )
        value = member.child_by_field_name("value")
        if value is None or value.type not in _FUNCTION_NODES:
            return None
        return _Definition(
            host=member,
            node=value,
            kind=ChunkKind.METHOD if owner else ChunkKind.FUNCTION,
            symbol=self._field_text(member, "name"),
            is_exported=exported,
        )

    def _class_members(self, class_body: Any | None) -> list[Any]:
        """Method and field members of a class body, in source order."""
        if class_body is None:
            return []
        return [child for child in class_body.named_children if child.type in _CLASS_MEMBER_NODES]

    def _nested_definitions(self, node: Any) -> Iterator[_Definition]:
        """Yield definitions lexically nested inside ``node``, outermost first.

        Iterative, never recursive: TC-024 puts 200 levels of nested arrow
        functions through here and a recursive walk would hit Python's own stack
        limit long before the AST depth cap had a say. Past
        ``settings.max_ast_depth`` the branch is abandoned with a warning and
        whatever chunk already covers it stands as the record of that text --
        unbounded descent into a pathological tree buys nothing retrievable.
        """
        stack: list[tuple[Any, int]] = [(child, 1) for child in reversed(node.named_children)]
        warned = False
        while stack:
            current, depth = stack.pop()
            if depth > self.max_depth:
                if not warned:
                    _LOG.warning(
                        "%s: AST nesting beyond %d levels under %s; not descending further",
                        self.file_path,
                        self.max_depth,
                        self._field_text(node, "name") or "<anonymous>",
                    )
                    warned = True
                continue
            definition = self._classify(current)
            if definition is not None and definition.host is not node:
                yield definition
                continue  # its own subtree is walked when that definition is emitted
            stack.extend((child, depth + 1) for child in reversed(current.named_children))

    # -- file-level facts -------------------------------------------------

    def _collect_imports(self, root: Any) -> list[str]:
        """Module specifiers imported by the file, verbatim, in source order.

        Read off the tree rather than by regex so a specifier mentioned inside a
        string literal or a comment never becomes a phantom import edge. ESM
        ``import``, ESM re-``export ... from`` (a barrel file consists of nothing
        else), and CommonJS ``require`` all count.
        """
        found: dict[str, None] = {}
        for node in self._iter_nodes(root):
            if node.type in {"import_statement", "export_statement"}:
                source_node = node.child_by_field_name("source")
                if source_node is not None:
                    found.setdefault(self._string_value(source_node), None)
            elif node.type == "call_expression":
                callee = node.child_by_field_name("function")
                if callee is not None and self._slice(callee) == "require":
                    arguments = node.child_by_field_name("arguments")
                    literals = [
                        child
                        for child in (arguments.named_children if arguments is not None else [])
                        if child.type in {"string", "template_string"}
                    ]
                    if literals:
                        found.setdefault(self._string_value(literals[0]), None)
        found.pop("", None)
        return list(found)

    def _collect_exported_names(self, root: Any) -> set[str]:
        """Names leaving the module by ``export {}`` or by CommonJS assignment.

        Declarations wrapped in ``export`` are detected structurally instead; this
        set covers only the indirect forms, where the name and the export are in
        different statements.
        """
        names: set[str] = set()
        for node in root.children:
            if node.type == "export_statement":
                for child in node.named_children:
                    if child.type in {"export_clause", "named_exports"}:
                        for specifier in child.named_children:
                            local = self._field_text(specifier, "name") or self._slice(specifier)
                            if local:
                                names.add(local)
            elif node.type == "expression_statement":
                names.update(self._commonjs_exports(node))
        return names

    def _commonjs_exports(self, statement: Any) -> set[str]:
        """Names exported by ``module.exports = ...`` / ``exports.x = ...``."""
        names: set[str] = set()
        for child in statement.named_children:
            if child.type != "assignment_expression":
                continue
            target = child.child_by_field_name("left")
            value = child.child_by_field_name("right")
            if target is None or not self._targets_commonjs_export(target):
                continue
            property_name = self._field_text(target, "property")
            target_text = self._slice(target)
            if target_text not in {"module.exports", "exports"} and property_name:
                names.add(property_name)
                continue
            if value is None:
                continue
            if value.type == "object":
                for entry in value.named_children:
                    key = self._field_text(entry, "key")
                    names.add(key or self._slice(entry))
            elif value.type == "identifier":
                names.add(self._slice(value))
        return {name for name in names if name}

    def _targets_commonjs_export(self, target: Any) -> bool:
        """True when an assignment target is ``module.exports`` or ``exports.x``."""
        text = self._slice(target)
        return text in {"module.exports", "exports"} or text.startswith(
            ("module.exports.", "exports.")
        )

    def _is_exported(self, symbol: str | None) -> bool:
        """True when ``symbol`` appears in the file's indirect export table."""
        return bool(symbol) and symbol in self.exported_names

    # -- node utilities ---------------------------------------------------

    def _slice(self, node: Any) -> str:
        """Source text of a node, decoded from bytes."""
        return self._slice_bytes(node.start_byte, node.end_byte)

    def _slice_bytes(self, start: int, end: int) -> str:
        """Source text of a byte span, decoded leniently for measurement only."""
        return self.src.data[start:end].decode("utf-8", errors="replace")

    def _string_value(self, node: Any) -> str:
        """Contents of a string literal with its quotes removed."""
        return self._slice(node).strip("'\"`")

    def _field_text(self, node: Any, field: str) -> str | None:
        """Text of a named field, or ``None`` when the field is absent."""
        child = node.child_by_field_name(field)
        return self._slice(child) if child is not None else None

    def _is_sub_floor(self, node: Any) -> bool:
        """True when a node is too small to be worth its own chunk (step 4)."""
        return below_floor(self._slice(node), self.floor)

    def _iter_nodes(self, root: Any) -> Iterator[Any]:
        """Iterative pre-order traversal. Never recursive (TC-024)."""
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def _calls_in(self, node: Any) -> list[str]:
        """Callee identifiers inside a node, in source order, duplicates kept."""
        return self._calls_in_nodes([node])

    def _calls_in_nodes(self, nodes: Sequence[Any]) -> list[str]:
        """Callee identifiers across several sibling nodes, in source order.

        Order is the whole point: Schema.md section 5 says this list answers
        "calls X before Y", so it is never sorted by name and never
        deduplicated. "Source order" means the order the *callee names* appear
        in the text, not the order the call nodes nest: in
        ``resolveTool(id).invoke(args)`` the outer call node starts first but
        ``resolveTool`` is what the reader sees first, and Schema.md section 6's
        worked example lists it that way.

        A member call records the property (``this.n(q)`` -> ``n``), because
        that is the name a user searches for.
        """
        found: list[tuple[int, str]] = []
        for node in nodes:
            found.extend(self._calls_with_positions(node))
        found.sort(key=lambda item: item[0])
        return [name for _, name in found]

    def _calls_in_range(self, node: Any, start_byte: int, end_byte: int) -> list[str]:
        """Callee identifiers within a byte window of a node's subtree."""
        return [
            name
            for position, name in sorted(self._calls_with_positions(node))
            if start_byte <= position < end_byte
        ]

    def _calls_with_positions(self, node: Any) -> list[tuple[int, str]]:
        """``(callee name start byte, callee)`` pairs inside a subtree."""
        found: list[tuple[int, str]] = []
        for current in self._iter_nodes(node):
            if current.type not in {"call_expression", "new_expression"}:
                continue
            callee = current.child_by_field_name("function") or current.child_by_field_name(
                "constructor"
            )
            if callee is None:
                continue
            if callee.type == "identifier":
                found.append((callee.start_byte, self._slice(callee)))
            elif callee.type == "member_expression":
                name_node = callee.child_by_field_name("property")
                if name_node is not None:
                    found.append((name_node.start_byte, self._slice(name_node)))
        return found

    def _docstring_for(self, node: Any) -> str | None:
        """Leading JSDoc or ``//`` block attached to ``node``, markers stripped.

        A comment counts as attached only when nothing but a single newline
        separates it from the declaration; a comment with a blank line after it
        is a section header, not documentation of what follows.
        """
        blocks: list[str] = []
        current = node
        while True:
            previous = current.prev_sibling
            if previous is None or previous.type != "comment":
                break
            gap = self.src.data[previous.end_byte : current.start_byte]
            if gap.count(b"\n") > 1 or gap.strip():
                break
            blocks.append(self._slice(previous))
            current = previous
        if not blocks:
            return None
        return strip_comment_markers("\n".join(reversed(blocks)))
