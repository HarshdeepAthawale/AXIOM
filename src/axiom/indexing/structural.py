"""Structural index builder: the four-relation call/import/export graph in SQLite.

This is the on-disk half of Axiom's headline innovation claim (PRD.md section 8).
Dense and sparse retrieval both collapse a chunk to a bag of features and lose
the one thing query archetype Q2 -- *"which files call tool XYZ before tool
ABC?"* -- actually asks about: the **order** in which a function invokes other
functions. No embedding of any dimensionality encodes "call site 3 precedes call
site 7". A relational table with an ordinal column does, exactly, for free.

So the build's job is narrow and mechanical: project every ``Chunk`` and its
``ChunkMetadata`` into the four relations locked by Schema.md section 14.7 and
ADR-011, preserving ``ChunkMetadata.calls`` **positionally**. No parsing happens
here -- ``chunking/ast_chunker.py`` already walked the tree-sitter AST and the
ordered ``calls`` list is its output. Re-parsing on the index path would
duplicate the one expensive step and risk the two walks disagreeing.

Only ``sqlite3`` from the standard library is used, so this module has no
optional dependency and no degradation ladder of its own: it either writes the
file or it is skipped by profile (NFR-07).
"""

from __future__ import annotations

import os
import posixpath
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from axiom.config import Settings
from axiom.core.logging import get_logger
from axiom.schema.chunk import Chunk

_LOG = get_logger("indexing.structural")

#: Filename of the structural index inside ``.axiom/index/<version_id>/``.
INDEX_FILENAME = "structural.sqlite"

#: Value written to ``PRAGMA user_version``; Schema.md section 14.7 generation 1.
SCHEMA_USER_VERSION = 1

#: ``imports.import_kind`` written when the binding form is unknown.
#:
#: ``ChunkMetadata.imports`` carries only the verbatim module specifier -- the
#: chunker records *that the file imports './normalize'*, never whether it did so
#: as a default, named, or namespace binding. The DDL's CHECK constraint admits
#: no "unknown" member, and inventing a per-file AST pass here would violate the
#: single-parse rule above. ``esm_namespace`` is the least wrong of the five: it
#: is the form that binds a whole module without naming any export, which is
#: precisely the amount of information we have.
_UNKNOWN_IMPORT_KIND = "esm_namespace"

#: ``exports.export_kind`` written for an ``is_exported`` symbol.
#:
#: Same reasoning: ``ChunkMetadata.is_exported`` is a boolean, so default-vs-named
#: is not recoverable. ``named`` is chosen because the row carries a concrete
#: ``exported_name`` (the symbol), which a ``default`` row would not.
_UNKNOWN_EXPORT_KIND = "named"

#: Line recorded for an import whose statement line is unknown.
#:
#: ``import_line`` is ``NOT NULL CHECK (import_line >= 1)`` and the metadata has
#: no line for it. Line 1 is the honest choice for a file-level fact: ESM imports
#: are hoisted to the top of the module and a reader following the reference to
#: the top of the file will find the statement.
_UNKNOWN_IMPORT_LINE = 1

#: Extensions tried, in order, when resolving a relative specifier to a corpus file.
#: Node's own resolution order for the JS subset NG-04 scopes us to.
_JS_EXTENSIONS: tuple[str, ...] = (".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx")

#: Schema.md section 14.7, verbatim. Any drift here is a contract break.
_SCHEMA_SQL = """
PRAGMA user_version = 1;

CREATE TABLE symbols (
    symbol_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id      TEXT    NOT NULL,
    symbol        TEXT,
    kind          TEXT    NOT NULL
                  CHECK (kind IN ('function', 'method', 'class', 'module', 'block')),
    parent_symbol TEXT,
    file_path     TEXT    NOT NULL,
    start_line    INTEGER NOT NULL CHECK (start_line >= 1),
    end_line      INTEGER NOT NULL CHECK (end_line >= start_line),
    start_byte    INTEGER NOT NULL CHECK (start_byte >= 0),
    end_byte      INTEGER NOT NULL CHECK (end_byte > start_byte),
    is_exported   INTEGER NOT NULL DEFAULT 0 CHECK (is_exported IN (0, 1)),
    language      TEXT    NOT NULL DEFAULT 'javascript',
    version_id    TEXT    NOT NULL,
    UNIQUE (chunk_id)
);

CREATE INDEX idx_symbols_symbol      ON symbols (symbol);
CREATE INDEX idx_symbols_file        ON symbols (file_path, start_line);
CREATE INDEX idx_symbols_parent      ON symbols (parent_symbol);
CREATE INDEX idx_symbols_exported    ON symbols (is_exported) WHERE is_exported = 1;

CREATE TABLE calls (
    call_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    caller_id   INTEGER NOT NULL
                REFERENCES symbols (symbol_id) ON DELETE CASCADE,
    callee_name TEXT    NOT NULL,
    callee_id   INTEGER
                REFERENCES symbols (symbol_id) ON DELETE SET NULL,
    call_order  INTEGER NOT NULL CHECK (call_order >= 0),
    call_line   INTEGER NOT NULL CHECK (call_line >= 1),
    is_method   INTEGER NOT NULL DEFAULT 0 CHECK (is_method IN (0, 1)),
    receiver    TEXT,
    UNIQUE (caller_id, call_order)
);

CREATE INDEX idx_calls_callee_name ON calls (callee_name);
CREATE INDEX idx_calls_callee_id   ON calls (callee_id);
CREATE INDEX idx_calls_caller_ord  ON calls (caller_id, call_order);

CREATE TABLE imports (
    import_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT    NOT NULL,
    module_spec   TEXT    NOT NULL,
    resolved_path TEXT,
    imported_name TEXT,
    local_alias   TEXT,
    import_kind   TEXT    NOT NULL
                  CHECK (import_kind IN ('esm_default', 'esm_named', 'esm_namespace',
                                         'cjs_require', 'dynamic')),
    import_line   INTEGER NOT NULL CHECK (import_line >= 1),
    version_id    TEXT    NOT NULL
);

CREATE INDEX idx_imports_file     ON imports (file_path);
CREATE INDEX idx_imports_resolved ON imports (resolved_path);
CREATE INDEX idx_imports_name     ON imports (imported_name);

CREATE TABLE exports (
    export_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT    NOT NULL,
    exported_name TEXT    NOT NULL,
    local_name    TEXT,
    symbol_id     INTEGER
                  REFERENCES symbols (symbol_id) ON DELETE SET NULL,
    export_kind   TEXT    NOT NULL
                  CHECK (export_kind IN ('default', 'named', 'namespace',
                                         'cjs_module', 'cjs_property')),
    export_line   INTEGER NOT NULL CHECK (export_line >= 1),
    version_id    TEXT    NOT NULL,
    UNIQUE (file_path, exported_name)
);

CREATE INDEX idx_exports_name   ON exports (exported_name);
CREATE INDEX idx_exports_symbol ON exports (symbol_id);
"""


@dataclass(frozen=True)
class StructuralIndexStats:
    """What one :func:`build_structural_index` run actually wrote.

    Returned rather than logged-only so the indexing driver can fold the counts
    into its ``VersionManifest`` and its NFR-10 timing record without reopening
    the database.
    """

    path: Path
    skipped: bool = False
    symbol_count: int = 0
    call_count: int = 0
    resolved_call_count: int = 0
    import_count: int = 0
    export_count: int = 0
    duplicate_chunk_ids: int = 0

    @property
    def call_resolution_rate(self) -> float:
        """Fraction of call edges bound to a defining symbol.

        Far below 1.0 on real JavaScript and that is expected, not a defect:
        dynamic dispatch, computed member access, and callbacks make call
        resolution undecidable in general (Schema.md section 14.7). Unresolved
        edges still carry ``callee_name``, which is all a USAGE query needs.
        """
        return self.resolved_call_count / self.call_count if self.call_count else 0.0


def split_callee(raw: str) -> tuple[str, str | None]:
    """Split a recorded callee into ``(callee_name, receiver)``.

    The DDL documents ``callee_name`` as "identifier as written at the call site"
    and ``receiver`` as "'client' in ``client.send(x)``". So a dotted callee is
    stored split: the bare method name is what a user searches for and what joins
    to ``symbols.symbol``, while the qualifier is retained for explanation.
    Only the final segment is treated as the name, so ``a.b.c()`` yields
    ``("c", "a.b")``.
    """
    name = raw.strip()
    if "." not in name:
        return name, None
    receiver, _, bare = name.rpartition(".")
    if not bare or not receiver:
        return name, None
    return bare, receiver


def _resolve_module_spec(spec: str, importer_path: str, known_files: frozenset[str]) -> str | None:
    """Resolve a relative specifier to a repo-relative corpus path, or ``None``.

    Bare specifiers (``node:fs``, ``lodash``) are external by definition and stay
    unresolved. Relative ones are resolved the way Node does -- exact path, then
    each extension, then the directory's ``index`` file -- but only against paths
    the corpus actually contains, so a resolution can never invent a file.
    """
    if not spec.startswith("."):
        return None
    base = posixpath.normpath(posixpath.join(posixpath.dirname(importer_path), spec))
    if base.startswith(".."):
        return None
    candidates = [base]
    candidates.extend(f"{base}{ext}" for ext in _JS_EXTENSIONS)
    candidates.extend(f"{base}/index{ext}" for ext in _JS_EXTENSIONS)
    for candidate in candidates:
        if candidate in known_files:
            return candidate
    return None


def _ordered_chunks(chunks: Iterable[Chunk]) -> tuple[list[Chunk], int]:
    """Deduplicate by ``chunk_id`` and impose the canonical corpus order.

    ``(file_path, start_line, chunk_id)`` is the same ordering ``chunks.jsonl``
    uses (Schema.md section 14.3). Sorting here makes the integer surrogate keys
    a pure function of the chunk set, which is what lets TC-076 compare an
    incremental build against a full rebuild row for row.
    """
    seen: dict[str, Chunk] = {}
    duplicates = 0
    for chunk in chunks:
        if chunk.chunk_id in seen:
            duplicates += 1
            continue
        seen[chunk.chunk_id] = chunk
    ordered = sorted(
        seen.values(),
        key=lambda c: (c.location.file_path, c.location.start_line, c.chunk_id),
    )
    return ordered, duplicates


def _build_resolution_map(
    rows: Sequence[tuple[int, str | None, str]],
) -> Mapping[str, list[tuple[str, int]]]:
    """Index symbol name -> ``[(file_path, symbol_id), ...]`` for callee binding."""
    table: dict[str, list[tuple[str, int]]] = {}
    for symbol_id, symbol, file_path in rows:
        if not symbol:
            continue
        table.setdefault(symbol, []).append((file_path, symbol_id))
    return table


def _resolve_callee(
    name: str,
    caller_file: str,
    table: Mapping[str, list[tuple[str, int]]],
) -> int | None:
    """Bind a callee name to a defining ``symbol_id``, or ``None`` when ambiguous.

    Two rungs, in order: a definition in the caller's own file wins (JavaScript
    module scope makes a same-file name the overwhelmingly likely referent), then
    a corpus-wide *unique* definition. Anything genuinely ambiguous stays
    ``NULL`` -- a wrong edge in a graph that claims to prove call ordering is far
    more damaging than a missing one, and ``callee_name`` still carries the
    name-level signal.
    """
    candidates = table.get(name)
    if not candidates:
        return None
    same_file = [symbol_id for file_path, symbol_id in candidates if file_path == caller_file]
    if same_file:
        return min(same_file)
    if len(candidates) == 1:
        return candidates[0][1]
    return None


def build_structural_index(
    chunks: Iterable[Chunk],
    settings: Settings,
    out_dir: Path,
) -> StructuralIndexStats:
    """Write ``structural.sqlite`` for one version into ``out_dir``.

    Args:
        chunks: The complete chunk set for this version, in any order.
        settings: Active settings. ``structural_enabled=False`` (the ``eval``
            profile) skips the build entirely rather than writing an empty
            database -- Appflow.md step 5, PRD.md section 2.2: APPS documents
            have no cross-file structure worth indexing.
        out_dir: ``.axiom/index/<version_id>/``. Created if absent.

    Returns:
        Row counts for the manifest and the timing record.

    The build is a single transaction against a temporary file which is then
    ``os.replace``-d into position, so an interrupted run leaves either the old
    database or none -- never a half-populated one that the retriever would
    happily query and silently under-answer.
    """
    target = Path(out_dir) / INDEX_FILENAME
    if not settings.structural_enabled:
        _LOG.info(
            "structural index skipped by profile",
            extra={"axiom_extra": {"stage": "struct", "profile": settings.profile}},
        )
        return StructuralIndexStats(path=target, skipped=True)

    ordered, duplicates = _ordered_chunks(chunks)
    if duplicates:
        _LOG.warning(
            "dropped duplicate chunk_ids before structural build",
            extra={"axiom_extra": {"stage": "struct", "duplicates": duplicates}},
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    for stale in (tmp_path, Path(f"{tmp_path}-wal"), Path(f"{tmp_path}-shm")):
        stale.unlink(missing_ok=True)

    connection = sqlite3.connect(tmp_path)
    try:
        # WAL during the build for write throughput; checkpointed back to DELETE
        # below so the shipped artifact is one file (Schema.md section 14.7).
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA_SQL)
        stats = _populate(connection, ordered, duplicates, target)
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode = DELETE")
        # Deliberately no ANALYZE / PRAGMA optimize: it would add sqlite_stat1
        # and sqlite_stat4 to a file Schema.md section 14.7 declares to be
        # exactly four data tables, and the query shapes here are all covered by
        # a declared index, so the planner has nothing to learn.
    finally:
        connection.close()

    for residue in (Path(f"{tmp_path}-wal"), Path(f"{tmp_path}-shm")):
        residue.unlink(missing_ok=True)
    os.replace(tmp_path, target)

    _LOG.info(
        "structural index built",
        extra={
            "axiom_extra": {
                "stage": "struct",
                "path": str(target),
                "symbols": stats.symbol_count,
                "calls": stats.call_count,
                "resolved_calls": stats.resolved_call_count,
                "imports": stats.import_count,
                "exports": stats.export_count,
            }
        },
    )
    return stats


#: Public alias matching the stage name used in Appflow.md ("indexing.structural:build_index").
build_index = build_structural_index


def _populate(
    connection: sqlite3.Connection,
    ordered: Sequence[Chunk],
    duplicates: int,
    target: Path,
) -> StructuralIndexStats:
    """Fill the four relations from an already-ordered, deduplicated chunk list.

    Surrogate keys are assigned explicitly from the enumeration index rather than
    left to ``AUTOINCREMENT`` so that identical input produces a byte-identical
    database (NFR-08, TC-076).
    """
    symbol_rows: list[tuple[object, ...]] = []
    resolution_rows: list[tuple[int, str | None, str]] = []
    known_files = frozenset(chunk.location.file_path for chunk in ordered)

    for index, chunk in enumerate(ordered, start=1):
        location = chunk.location
        metadata = chunk.metadata
        symbol_rows.append(
            (
                index,
                chunk.chunk_id,
                metadata.symbol,
                str(metadata.kind),
                metadata.parent_symbol,
                location.file_path,
                location.start_line,
                location.end_line,
                location.start_byte,
                location.end_byte,
                int(metadata.is_exported),
                metadata.language,
                metadata.version_id,
            )
        )
        resolution_rows.append((index, metadata.symbol, location.file_path))

    connection.executemany(
        "INSERT INTO symbols (symbol_id, chunk_id, symbol, kind, parent_symbol, file_path,"
        " start_line, end_line, start_byte, end_byte, is_exported, language, version_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        symbol_rows,
    )

    table = _build_resolution_map(resolution_rows)

    call_rows: list[tuple[object, ...]] = []
    resolved = 0
    call_id = 0
    for symbol_id, chunk in enumerate(ordered, start=1):
        caller_file = chunk.location.file_path
        anchor_line = chunk.location.start_line
        for order, raw in enumerate(chunk.metadata.calls):
            # The ordinal is the *original* index, so a blank entry leaves a gap
            # rather than shifting every later call. UNIQUE (caller_id,
            # call_order) constrains uniqueness, not contiguity, and relative
            # order is the only property the ordered-pair query reads.
            callee_name, receiver = split_callee(raw)
            if not callee_name:
                continue
            callee_id = _resolve_callee(callee_name, caller_file, table)
            if callee_id is not None:
                resolved += 1
            call_id += 1
            call_rows.append(
                (
                    call_id,
                    symbol_id,
                    callee_name,
                    callee_id,
                    order,
                    anchor_line,
                    int(receiver is not None),
                    receiver,
                )
            )
    connection.executemany(
        "INSERT INTO calls (call_id, caller_id, callee_name, callee_id, call_order,"
        " call_line, is_method, receiver) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        call_rows,
    )

    # One row per (file, specifier): every chunk cut from a file repeats that
    # file's import list, and the relation's grain is the binding, not the chunk.
    seen_imports: set[tuple[str, str]] = set()
    import_rows: list[tuple[object, ...]] = []
    for chunk in ordered:
        file_path = chunk.location.file_path
        for spec in chunk.metadata.imports:
            spec = spec.strip()
            if not spec or (file_path, spec) in seen_imports:
                continue
            seen_imports.add((file_path, spec))
            import_rows.append(
                (
                    len(import_rows) + 1,
                    file_path,
                    spec,
                    _resolve_module_spec(spec, file_path, known_files),
                    None,
                    None,
                    _UNKNOWN_IMPORT_KIND,
                    _UNKNOWN_IMPORT_LINE,
                    chunk.metadata.version_id,
                )
            )
    connection.executemany(
        "INSERT INTO imports (import_id, file_path, module_spec, resolved_path,"
        " imported_name, local_alias, import_kind, import_line, version_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        import_rows,
    )

    # UNIQUE (file_path, exported_name): a file exporting one name twice is a
    # chunker artefact (e.g. a class and its re-exported alias), so the first
    # occurrence in canonical order wins and the rest are dropped silently --
    # dropping a duplicate row loses no information the first row lacks.
    seen_exports: set[tuple[str, str]] = set()
    export_rows: list[tuple[object, ...]] = []
    for symbol_id, chunk in enumerate(ordered, start=1):
        metadata = chunk.metadata
        if not metadata.is_exported or not metadata.symbol:
            continue
        key = (chunk.location.file_path, metadata.symbol)
        if key in seen_exports:
            continue
        seen_exports.add(key)
        export_rows.append(
            (
                len(export_rows) + 1,
                chunk.location.file_path,
                metadata.symbol,
                metadata.symbol,
                symbol_id,
                _UNKNOWN_EXPORT_KIND,
                chunk.location.start_line,
                metadata.version_id,
            )
        )
    connection.executemany(
        "INSERT INTO exports (export_id, file_path, exported_name, local_name, symbol_id,"
        " export_kind, export_line, version_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        export_rows,
    )

    return StructuralIndexStats(
        path=target,
        skipped=False,
        symbol_count=len(symbol_rows),
        call_count=len(call_rows),
        resolved_call_count=resolved,
        import_count=len(import_rows),
        export_count=len(export_rows),
        duplicate_chunk_ids=duplicates,
    )
