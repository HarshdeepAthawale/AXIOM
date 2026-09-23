"""Structural retrieval: the signal that answers ordering questions.

Dense retrieval answers *"what does this code mean?"* and sparse retrieval
answers *"where does this token appear?"*. Neither can answer *"which files call
tool XYZ **before** tool ABC?"* (PRD.md section 1.1, query archetype Q2), because
the fact being asked about -- the relative position of two call sites inside one
function body -- survives neither an embedding nor a bag of words. It does
survive a table with an ordinal column, and reading it back is an indexed
self-join, not graph analytics (ADR-011, NG-27).

This module is therefore deliberately small: five SQL query shapes over
``structural.sqlite``, a deterministic score derived from which shape matched,
and a hard rule that no failure escapes as an exception. Per FR-10 and Rules.md
Rule 3, an unresolvable identifier, a missing index, or a corrupt database all
produce an **empty list**; fusion then drops the structural weight and
renormalises the other two (TechSpecifications.md section 5.1.2), which is a
strictly better outcome for the user than a 500.

Only ``sqlite3`` from the standard library is used -- there is no optional
dependency on this path, so the structural signal is the one signal that cannot
be disabled by the evaluator's laptop lacking a wheel (NFR-07).
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from types import MappingProxyType, TracebackType

from axiom.config import Settings, get_settings
from axiom.core.logging import get_logger, log_degradation
from axiom.indexing.structural import INDEX_FILENAME, split_callee
from axiom.schema.enums import QueryType, SignalKind
from axiom.schema.plan import QueryPlan
from axiom.schema.retrieval import ScoredChunk

_LOG = get_logger("retrieval.structural")


class StructuralEvidence(StrEnum):
    """Why a chunk was surfaced by the structural signal.

    The member is both the explanation shown to the user and the input to the
    score: :data:`_EVIDENCE_PRIORITY` maps a ``QueryType`` to the order these
    count for, and position in that order *is* the score tier (see
    :func:`_evidence_score`).
    """

    ORDERED_PAIR = "ordered_pair"
    DEFINITION = "definition"
    CALLER = "caller"
    CALLEE = "callee"
    EXPORT = "export"
    IMPORT = "import"
    SUBSTRING = "substring"


@dataclass(frozen=True)
class GraphHit:
    """One chunk surfaced by one graph query, with the evidence that surfaced it."""

    chunk_id: str
    evidence: StructuralEvidence
    detail: str


#: Per-``QueryType`` evidence priority. Position 0 scores highest.
#:
#: This single table does double duty as the routing rule the spec asks for and
#: the scoring rule: a kind absent from a row is never *executed* for that query
#: type, and a kind's index in the row sets its score tier. Keeping routing and
#: scoring in one structure makes it impossible for them to disagree.
#:
#: The orderings are not arbitrary. A ``STRUCTURAL`` query ("which files call X
#: before Y") wants the *callers* -- the definition of X itself is the one chunk
#: the user already knows about -- so ordered pairs and caller edges outrank
#: definitions. A ``SEMANTIC`` query reached the structural signal only because
#: the planner lifted an identifier out of it, so the definition is the most
#: likely intent. A ``USAGE`` query ("where is X used") is all call sites,
#: exports and importers, and ordering is irrelevant, so the expensive
#: ordered-pair self-join is not run at all.
_EVIDENCE_PRIORITY: Mapping[QueryType, tuple[StructuralEvidence, ...]] = MappingProxyType(
    {
        QueryType.STRUCTURAL: (
            StructuralEvidence.ORDERED_PAIR,
            StructuralEvidence.CALLER,
            StructuralEvidence.DEFINITION,
            StructuralEvidence.CALLEE,
            StructuralEvidence.EXPORT,
            StructuralEvidence.IMPORT,
        ),
        QueryType.SEMANTIC: (
            StructuralEvidence.DEFINITION,
            StructuralEvidence.ORDERED_PAIR,
            StructuralEvidence.CALLEE,
            StructuralEvidence.CALLER,
            StructuralEvidence.EXPORT,
            StructuralEvidence.IMPORT,
        ),
        QueryType.USAGE: (
            StructuralEvidence.CALLER,
            StructuralEvidence.EXPORT,
            StructuralEvidence.IMPORT,
            StructuralEvidence.DEFINITION,
            StructuralEvidence.CALLEE,
        ),
        QueryType.HYBRID: (
            StructuralEvidence.ORDERED_PAIR,
            StructuralEvidence.DEFINITION,
            StructuralEvidence.CALLER,
            StructuralEvidence.CALLEE,
            StructuralEvidence.EXPORT,
            StructuralEvidence.IMPORT,
        ),
    }
)

#: Score of the top-priority evidence kind. Rule 4: structural scores live in (0, 1].
_TIER_TOP = 0.95

#: Score drop per priority position. Six tiers bottom out at 0.35, comfortably > 0.
_TIER_STEP = 0.12

#: Score for the degraded substring rung. Below every real graph tier on purpose:
#: a substring hit is a guess, and fusion should treat it as one.
_SUBSTRING_SCORE = 0.20

#: Added per *additional* distinct evidence kind on the same chunk.
#:
#: A chunk that both defines X and calls Y is better evidence than one that only
#: calls Y, but not better than a different chunk one whole tier up. The bonus is
#: capped at :data:`_MAX_CORROBORATION` steps = 0.05, strictly less than
#: :data:`_TIER_STEP`, so corroboration reorders *within* a tier and can never
#: promote across one.
_CORROBORATION_STEP = 0.01
_MAX_CORROBORATION = 5

#: Upper bound on identifiers fed to the quadratic ordered-pair expansion.
#:
#: A plan with 20 identifiers would otherwise issue 190 self-joins for one query.
#: Six covers every realistic phrasing of "calls A before B" (the archetype has
#: two) while keeping the worst case at 15 indexed lookups.
_MAX_PAIR_IDENTIFIERS = 6

#: Per-sub-query row cap, as a multiple of the requested ``k``.
#:
#: Each graph query is bounded so a pathologically popular symbol (``log``,
#: ``get``) cannot drag thousands of rows through Python. Oversampling by 4 leaves
#: room for corroboration to reorder the pool before the final truncation to k.
_CANDIDATE_OVERSAMPLE = 4

#: Identifier shapes mined from raw query text on the degraded rung: dotted paths,
#: camelCase, and snake_case. Prose words match none of the three, which is the
#: entire point -- "the", "before" and "files" must not become symbol lookups.
_CODE_TOKEN_RE = re.compile(
    r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+"
    r"|[a-z$_][A-Za-z0-9$]*[A-Z][A-Za-z0-9_$]*"
    r"|[A-Za-z$][A-Za-z0-9$]*_[A-Za-z0-9_$]+"
)

#: Maximal runs over the identifier alphabet -- the superset of every span
#: :data:`_CODE_TOKEN_RE` can match. A plain character-class star with nothing
#: after it, so it cannot backtrack and is linear in the query length.
_TOKEN_RUN_RE = re.compile(r"[A-Za-z0-9_$.]+")

#: Longest run :func:`mine_identifiers` will inspect.
#:
#: The bound is what keeps the shape regex's quadratic worst case off a long
#: input (TC-047): 128 characters of it costs microseconds. No symbol in a real
#: JavaScript symbol table is longer, so nothing minable is lost -- and this is
#: the degraded rung anyway, reached only when the planner extracted nothing.
#: Not a ``Settings`` field: it affects no score, only how much text a fallback
#: is willing to scan.
_MAX_MINED_TOKEN_CHARS = 128


#: Extensions stripped when reducing a module specifier to its stem, so that
#: './normalize', 'src/normalize.js' and 'normalize' all reach the same lookup.
_MODULE_EXTENSIONS: tuple[str, ...] = (".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx")


def _like_escape(value: str) -> str:
    """Neutralise LIKE wildcards in user-supplied text.

    An identifier containing ``_`` is ordinary in code and a single-character
    wildcard in SQL; without escaping, ``handle_tap`` would also match
    ``handleXtap``. ``!`` is the escape character, declared by every LIKE here.
    """
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _module_patterns(spec: str) -> tuple[str, str, str]:
    """Derive ``(stem, '%/stem', '%/stem.%')`` LIKE patterns from a specifier."""
    stem = spec.rsplit("/", 1)[-1]
    for extension in _MODULE_EXTENSIONS:
        stem = stem.removesuffix(extension)
    escaped = _like_escape(stem)
    return stem, f"%/{escaped}", f"%/{escaped}.%"


def _evidence_score(query_type: QueryType, evidence: StructuralEvidence) -> float:
    """Score tier for one evidence kind under one query type."""
    if evidence is StructuralEvidence.SUBSTRING:
        return _SUBSTRING_SCORE
    priority = _EVIDENCE_PRIORITY[query_type]
    position = priority.index(evidence)
    return _TIER_TOP - position * _TIER_STEP


def normalise_identifier(raw: str) -> str:
    """Strip the decoration a query puts around an identifier.

    Planners and users alike write ``resolveTool()``, ``"resolveTool"`` and
    `` `resolveTool` `` for the same symbol. Trailing call parens, surrounding
    quotes and punctuation are removed; the dotted form is preserved because
    module lookups need it and :func:`split_callee` reduces it later.
    """
    token = raw.strip().strip("`'\"")
    token = token.removesuffix("()")
    return token.strip().strip(",;:.()[]{}\"'`")


def mine_identifiers(query: str) -> list[str]:
    """Recover code-shaped tokens from raw query text, in order of appearance.

    Only reached when ``QueryPlan.extracted_identifiers`` is empty -- a degraded
    rung (Rules.md section 3: "symbol/call lookup -> identifier substring match
    -> empty list"), never the primary path.

    Applied naively, :data:`_CODE_TOKEN_RE` is quadratic in the query length
    (TestPlan.md TC-047). Its camelCase alternative has no anchor for its
    mandatory uppercase character, so on a long run of lowercase letters the
    engine scans to the end of the run from every start offset: measured at 5.6 /
    22 / 90 / 340 ms for 512 / 1024 / 2048 / 4096 characters. Bounded -- the
    planner truncates at 2048 -- but 90 ms *per (version, sub-query, pass)* is a
    fifth of the agent budget spent inside one regex.

    So the expensive pattern never sees the whole query. The text is first split
    into maximal identifier-alphabet runs by a pattern that cannot backtrack,
    and a run is handed on only if it is short enough to be a real symbol and
    contains at least one of the three characters the shapes actually require: a
    ``.`` (dotted), a ``_`` (snake) or an uppercase letter (camel). A run with
    none of them cannot match any alternative, so skipping it changes no result
    -- it just refuses to prove that fact character by character.
    """
    seen: dict[str, None] = {}
    for run in _TOKEN_RUN_RE.finditer(query):
        candidate = run.group(0)
        if len(candidate) > _MAX_MINED_TOKEN_CHARS:
            continue
        if "." not in candidate and "_" not in candidate and candidate.islower():
            continue
        for match in _CODE_TOKEN_RE.finditer(candidate):
            token = normalise_identifier(match.group(0))
            if token:
                seen.setdefault(token, None)
    return list(seen)


def _dedupe(values: Iterable[str]) -> list[str]:
    """Order-preserving deduplication. Order matters: identifier 0 is X, 1 is Y."""
    seen: dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(value, None)
    return list(seen)


class StructuralRetriever:
    """Read-only SQL traversal over one version's ``structural.sqlite``.

    Construction never touches the disk and never raises: the connection is
    opened lazily on the first query and a failure to open is a logged
    degradation that turns every subsequent query into an empty list. That
    ordering matters because the three signals run concurrently (Appflow.md flow
    2 step 3) -- a constructor that raised would take the whole query down with
    it, when the correct behaviour is for the other two signals to carry on.
    """

    def __init__(self, index_dir: Path | str, settings: Settings | None = None) -> None:
        self._path = Path(index_dir) / INDEX_FILENAME
        self._settings = settings or get_settings()
        self._connection: sqlite3.Connection | None = None
        self._unavailable = False
        # The retriever is created once per query but read from the thread that
        # runs the structural leg, which is not the thread that built it.
        # check_same_thread=False plus one lock is the cheapest correct answer;
        # SQLite serialises readers of a single connection anyway.
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection | None:
        """Open the database read-only, or record it as unavailable, once."""
        if self._connection is not None:
            return self._connection
        if self._unavailable:
            return None
        try:
            connection = sqlite3.connect(
                f"file:{self._path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            connection.execute("PRAGMA query_only = ON")
        except (sqlite3.Error, OSError) as exc:
            self._unavailable = True
            log_degradation(
                _LOG,
                "retrieval.structural",
                f"cannot open {self._path}: {type(exc).__name__}",
                "empty structural signal",
            )
            return None
        self._connection = connection
        return connection

    def close(self) -> None:
        """Release the connection. Idempotent."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> StructuralRetriever:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def available(self) -> bool:
        """Whether the index opened. False means every query returns ``[]``."""
        return self._connect() is not None

    def _query(self, sql: str, params: Sequence[object]) -> list[sqlite3.Row]:
        """Run one read query, degrading to no rows on any SQLite failure.

        The whole module's never-raise contract lives here: a corrupt page, a
        schema from a future generation, or a locked file all become an empty
        result and a WARNING, exactly as the ladder in Rules.md section 3 requires.
        """
        with self._lock:
            connection = self._connect()
            if connection is None:
                return []
            try:
                cursor = connection.execute(sql, tuple(params))
                return list(cursor.fetchall())
            except sqlite3.Error as exc:
                log_degradation(
                    _LOG,
                    "retrieval.structural",
                    f"query failed: {type(exc).__name__}: {exc}",
                    "empty result for this graph query",
                )
                return []

    # -- the five graph queries -------------------------------------------

    def callers_of(self, symbol: str, limit: int) -> list[GraphHit]:
        """Chunks containing a call to ``symbol``.

        Matched on ``callee_name`` rather than the resolved ``callee_id`` so that
        unresolved edges still count: JavaScript call resolution is undecidable
        in general and a name-level match is precisely what "where is X used"
        wants (Schema.md section 14.7).
        """
        name, _ = split_callee(normalise_identifier(symbol))
        if not name:
            return []
        rows = self._query(
            "SELECT DISTINCT s.chunk_id AS chunk_id, s.symbol AS caller"
            " FROM calls c JOIN symbols s ON s.symbol_id = c.caller_id"
            " WHERE c.callee_name = ?"
            " ORDER BY s.chunk_id LIMIT ?",
            (name, limit),
        )
        return [GraphHit(row[0], StructuralEvidence.CALLER, f"calls {name}") for row in rows]

    def callees_of(self, symbol: str, limit: int) -> list[GraphHit]:
        """Chunks defining a function that ``symbol``'s body invokes.

        Only resolved edges can answer this -- an unresolved callee has no chunk
        to point at -- so this query joins through ``calls.callee_id``.
        """
        name, _ = split_callee(normalise_identifier(symbol))
        if not name:
            return []
        rows = self._query(
            "SELECT DISTINCT t.chunk_id AS chunk_id, t.symbol AS callee"
            " FROM symbols s JOIN calls c ON c.caller_id = s.symbol_id"
            " JOIN symbols t ON t.symbol_id = c.callee_id"
            " WHERE s.symbol = ?"
            " ORDER BY t.chunk_id LIMIT ?",
            (name, limit),
        )
        return [GraphHit(row[0], StructuralEvidence.CALLEE, f"called by {name}") for row in rows]

    def definitions_of(self, symbol: str, limit: int) -> list[GraphHit]:
        """Chunks that *are* the declaration of ``symbol``."""
        name, _ = split_callee(normalise_identifier(symbol))
        if not name:
            return []
        rows = self._query(
            "SELECT chunk_id FROM symbols WHERE symbol = ? ORDER BY chunk_id LIMIT ?",
            (name, limit),
        )
        return [GraphHit(row[0], StructuralEvidence.DEFINITION, f"defines {name}") for row in rows]

    def imports_of(self, module: str, limit: int) -> list[GraphHit]:
        """Chunks in files that import ``module``.

        ``module`` may be written as the verbatim specifier (``./normalize``), a
        repo-relative path (``src/normalize.js``), or a bare module name
        (``normalize``); all three are matched, because a user asking "who
        imports normalize" does not know which spelling the source used.

        One chunk per importing file is returned -- the file's ``MODULE`` chunk
        when the chunker emitted one, otherwise its topmost chunk. Returning
        every chunk of every importing file would let a single popular utility
        module flood a K=50 candidate list with one file's contents. The
        correlated subquery picks that representative; it rides
        ``idx_symbols_file`` and is evaluated once per matching file.
        """
        spec = normalise_identifier(module)
        if not spec:
            return []
        stem, slash, slash_ext = _module_patterns(spec)
        rows = self._query(
            "SELECT s.chunk_id AS chunk_id, i.file_path AS file_path,"
            "       MIN(i.module_spec) AS spec"
            " FROM imports i JOIN symbols s ON s.symbol_id = ("
            "     SELECT s2.symbol_id FROM symbols s2 WHERE s2.file_path = i.file_path"
            "     ORDER BY (s2.kind <> 'module'), s2.start_line, s2.chunk_id LIMIT 1)"
            " WHERE i.module_spec = ?1 OR i.resolved_path = ?1 OR i.module_spec = ?2"
            "    OR i.module_spec LIKE ?3 ESCAPE '!' OR i.module_spec LIKE ?4 ESCAPE '!'"
            "    OR i.resolved_path LIKE ?3 ESCAPE '!' OR i.resolved_path LIKE ?4 ESCAPE '!'"
            " GROUP BY i.file_path"
            " ORDER BY i.file_path LIMIT ?5",
            (spec, stem, slash, slash_ext, limit),
        )
        return [
            GraphHit(row[0], StructuralEvidence.IMPORT, f"{row[1]} imports {row[2]}")
            for row in rows
        ]

    def exports_of(self, module: str, limit: int) -> list[GraphHit]:
        """Chunks exported by ``module``, or exporting a symbol named ``module``.

        Both readings are served because "exports-of" is asked both ways in
        practice: *what does ./normalize export* and *who exports normalize*.
        The union costs one extra indexed lookup and removes a guess about which
        one the planner meant.
        """
        spec = normalise_identifier(module)
        if not spec:
            return []
        stem, slash, slash_ext = _module_patterns(spec)
        rows = self._query(
            "SELECT s.chunk_id AS chunk_id, e.file_path AS file_path,"
            "       e.exported_name AS exported_name"
            " FROM exports e JOIN symbols s ON s.symbol_id = e.symbol_id"
            " WHERE e.file_path = ?1 OR e.exported_name = ?2"
            "    OR e.file_path LIKE ?3 ESCAPE '!' OR e.file_path LIKE ?4 ESCAPE '!'"
            " ORDER BY s.chunk_id LIMIT ?5",
            (spec, stem, slash, slash_ext, limit),
        )
        return [
            GraphHit(row[0], StructuralEvidence.EXPORT, f"{row[1]} exports {row[2]}")
            for row in rows
        ]

    def ordered_call_pair(self, first: str, second: str, limit: int) -> list[GraphHit]:
        """Chunks that call ``first`` before ``second``. Query archetype Q2.

        The predicate is existential: a chunk matches when **some** call to
        ``first`` precedes **some** call to ``second``. With duplicate call sites
        that is exactly ``MIN(ordinal of first) < MAX(ordinal of second)``, which
        is why the SQL aggregates rather than joining call rows pairwise --
        ``MIN``/``MAX`` are immune to the fan-out a pairwise join would create,
        and they express the existential quantifier in one indexed pass.

        The alternative readings were considered and rejected: *every* call to
        ``first`` before *every* call to ``second`` (``MAX(first) < MIN(second)``)
        rejects the extremely common retry/cleanup shape where a function calls
        ``connect(); send(); connect()``, and *first* before *first matching*
        pairs arbitrarily. The existential reading is also the one a human means
        by "calls X before Y".

        When ``first == second`` the predicate degenerates to "calls it at least
        twice", since ``MIN < MAX`` requires two distinct ordinals. That is the
        honest reading of "calls X before X" and it is left in deliberately.
        """
        x_name, _ = split_callee(normalise_identifier(first))
        y_name, _ = split_callee(normalise_identifier(second))
        if not x_name or not y_name:
            return []
        rows = self._query(
            "SELECT s.chunk_id AS chunk_id,"
            "       MIN(CASE WHEN c.callee_name = ?1 THEN c.call_order END) AS x_first,"
            "       MAX(CASE WHEN c.callee_name = ?2 THEN c.call_order END) AS y_last"
            " FROM calls c JOIN symbols s ON s.symbol_id = c.caller_id"
            " WHERE c.callee_name IN (?1, ?2)"
            " GROUP BY c.caller_id"
            " HAVING x_first IS NOT NULL AND y_last IS NOT NULL AND x_first < y_last"
            " ORDER BY s.chunk_id LIMIT ?3",
            (x_name, y_name, limit),
        )
        return [
            GraphHit(
                row[0],
                StructuralEvidence.ORDERED_PAIR,
                f"calls {x_name} (ordinal {row[1]}) before {y_name} (ordinal {row[2]})",
            )
            for row in rows
        ]

    def substring_match(self, fragment: str, limit: int) -> list[GraphHit]:
        """Last rung before empty: symbols whose name contains ``fragment``.

        ``LIKE '%x%'`` cannot use ``idx_symbols_symbol`` and is a table scan, so
        it runs only when every exact lookup came back empty -- the cost is paid
        exactly once, on a query that would otherwise have contributed nothing.
        """
        token = normalise_identifier(fragment)
        if len(token) < 3:
            return []
        escaped = _like_escape(token)
        rows = self._query(
            "SELECT chunk_id, symbol FROM symbols"
            " WHERE symbol LIKE ? ESCAPE '!' ORDER BY chunk_id LIMIT ?",
            (f"%{escaped}%", limit),
        )
        return [
            GraphHit(row[0], StructuralEvidence.SUBSTRING, f"symbol {row[1]} contains {token}")
            for row in rows
        ]

    # -- the retrieval entrypoint -----------------------------------------

    def search(self, query: str, plan: QueryPlan, k: int | None = None) -> list[ScoredChunk]:
        """Run the structural signal for one query plan.

        Args:
            query: Raw query text. Read only on the degraded rung, when the plan
                carries no identifiers and they must be mined from the text.
            plan: The agent's plan. ``query_type`` routes which graph queries run
                and how they score; ``extracted_identifiers`` are the lookups.
            k: Candidate width. Defaults to ``Settings.structural_top_k`` (50).

        Returns:
            At most ``k`` :class:`ScoredChunk`, ``signal=STRUCTURAL``, sorted
            descending by score with contiguous 1-indexed ranks. **Empty** when
            no identifier resolves or the index is missing -- FR-10 requires an
            empty list rather than an error, and fusion's empty-signal
            renormalisation (TechSpecifications.md section 5.1.2) handles it.
        """
        scored, _ = self.search_with_evidence(query, plan, k)
        return scored

    def search_with_evidence(
        self, query: str, plan: QueryPlan, k: int | None = None
    ) -> tuple[list[ScoredChunk], dict[str, str]]:
        """:meth:`search`, plus a ``chunk_id -> explanation`` map.

        The explanation is returned alongside rather than stashed on the
        instance because stages are pure (Rules.md Rule 2) and ``ScoredChunk`` is
        frozen with ``extra="forbid"``, so there is nowhere on the schema object
        to put it. The formatter uses this map to build
        ``RetrievalResult.match_reason`` -- "structural: calls preprocessInput
        before main" is only sayable if the evidence survives the return.
        """
        width = k if k is not None else self._settings.structural_top_k
        if width <= 0:
            return [], {}
        row_limit = width * _CANDIDATE_OVERSAMPLE

        identifiers = _dedupe(normalise_identifier(i) for i in plan.extracted_identifiers)
        if not identifiers:
            identifiers = mine_identifiers(query)
            if identifiers:
                log_degradation(
                    _LOG,
                    "retrieval.structural",
                    "plan carried no extracted_identifiers",
                    f"identifiers mined from query text: {identifiers}",
                )
        if not identifiers:
            log_degradation(
                _LOG,
                "retrieval.structural",
                "no identifier could be resolved from plan or query text",
                "empty structural signal (FR-10)",
            )
            return [], {}

        hits = self._collect(plan.query_type, identifiers, row_limit)
        if not hits:
            # Declared ladder rung 2: exact lookups found nothing, try substrings.
            hits = [
                hit
                for identifier in identifiers
                for hit in self.substring_match(identifier, row_limit)
            ]
            if hits:
                log_degradation(
                    _LOG,
                    "retrieval.structural",
                    f"no exact graph match for {identifiers}",
                    "identifier substring match",
                )
        if not hits:
            return [], {}

        return _rank(hits, plan.query_type, width)

    def _collect(
        self, query_type: QueryType, identifiers: Sequence[str], row_limit: int
    ) -> list[GraphHit]:
        """Run every graph query the routing table enables, in a fixed order.

        Iteration order is over the priority tuple and then over identifiers as
        the planner emitted them -- never over a set -- so the candidate pool is
        byte-identical between runs (NFR-08, Rules.md AP-05).
        """
        priority = _EVIDENCE_PRIORITY.get(query_type, _EVIDENCE_PRIORITY[QueryType.HYBRID])
        hits: list[GraphHit] = []
        for evidence in priority:
            if evidence is StructuralEvidence.ORDERED_PAIR:
                # Every ordered pair (i < j), not just consecutive ones: the
                # planner emits identifiers in query order, so "calls A, then B,
                # then C" asks about A-before-C as much as A-before-B. The
                # identifier list is capped first because this is the one
                # quadratic term in the module.
                for first, second in combinations(identifiers[:_MAX_PAIR_IDENTIFIERS], 2):
                    hits.extend(self.ordered_call_pair(first, second, row_limit))
                continue
            for identifier in identifiers:
                if evidence is StructuralEvidence.DEFINITION:
                    hits.extend(self.definitions_of(identifier, row_limit))
                elif evidence is StructuralEvidence.CALLER:
                    hits.extend(self.callers_of(identifier, row_limit))
                elif evidence is StructuralEvidence.CALLEE:
                    hits.extend(self.callees_of(identifier, row_limit))
                elif evidence is StructuralEvidence.EXPORT:
                    hits.extend(self.exports_of(identifier, row_limit))
                elif evidence is StructuralEvidence.IMPORT:
                    hits.extend(self.imports_of(identifier, row_limit))
        return hits


def _rank(
    hits: Sequence[GraphHit], query_type: QueryType, width: int
) -> tuple[list[ScoredChunk], dict[str, str]]:
    """Fold per-query hits into one ranked, deduplicated, truncated list.

    Score is ``best tier + corroboration``: the highest-priority evidence a chunk
    carries, plus a capped bonus for each *additional distinct* kind of evidence.
    Ties break on ascending ``chunk_id`` (Rules.md Rule 4), ranks are 1-indexed
    and contiguous, and the list is truncated to ``width`` only after ranking so
    the cut is by score, not by arrival order.
    """
    best: dict[str, StructuralEvidence] = {}
    kinds: dict[str, set[StructuralEvidence]] = {}
    details: dict[str, list[tuple[float, str]]] = {}

    for hit in hits:
        tier = _evidence_score(query_type, hit.evidence)
        current = best.get(hit.chunk_id)
        if current is None or tier > _evidence_score(query_type, current):
            best[hit.chunk_id] = hit.evidence
        kinds.setdefault(hit.chunk_id, set()).add(hit.evidence)
        bucket = details.setdefault(hit.chunk_id, [])
        if all(existing != hit.detail for _, existing in bucket):
            bucket.append((-tier, hit.detail))

    scores = {
        chunk_id: _evidence_score(query_type, evidence)
        + _CORROBORATION_STEP * min(len(kinds[chunk_id]) - 1, _MAX_CORROBORATION)
        for chunk_id, evidence in best.items()
    }
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:width]

    scored = [
        ScoredChunk(
            chunk_id=chunk_id,
            score=scores[chunk_id],
            rank=position,
            signal=SignalKind.STRUCTURAL,
        )
        for position, chunk_id in enumerate(ordered, start=1)
    ]
    evidence = {
        chunk_id: "; ".join(detail for _, detail in sorted(details[chunk_id]))
        for chunk_id in ordered
    }
    return scored, evidence


def search(
    query: str,
    plan: QueryPlan,
    index_dir: Path | str,
    settings: Settings | None = None,
    k: int | None = None,
) -> list[ScoredChunk]:
    """Module-level entrypoint named by the degradation ladder (Rules.md section 3).

    Opens, queries, and closes a :class:`StructuralRetriever`. Callers issuing
    more than one query against the same version should hold the retriever
    instead, so the connection and its page cache survive between passes of the
    agent loop.
    """
    with StructuralRetriever(index_dir, settings) as retriever:
        return retriever.search(query, plan, k)
