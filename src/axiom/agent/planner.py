"""Query planning: identifier extraction, expansion, decomposition, refinement.

Produces the :class:`~axiom.schema.plan.QueryPlan` that the rest of the pipeline
reads -- fusion takes ``strategy_weights``, the structural retriever takes
``extracted_identifiers``, the sparse retriever takes ``expansion_terms``, and
the fan-out takes ``effective_queries``. Every step is a pure function of
``(query, settings)`` so that a plan is reproducible from the logs (NFR-08).

The planner never raises on query text. An empty query, ten thousand characters
of noise, or a line of emoji all produce a well-formed plan; upstream
(``agent.loop:run``) decides whether an empty query is worth retrieving at all
(Rules.md section 3, "Query normalisation" row).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from axiom.agent.classifier import QUOTED_LITERAL_RE, classify
from axiom.agent.synonyms import expand_tokens
from axiom.config import Settings
from axiom.core.logging import get_logger, log_degradation
from axiom.schema.enums import QueryType
from axiom.schema.plan import QueryPlan
from axiom.schema.retrieval import FusedResult

_LOG = get_logger("agent.planner")

#: Hard cap on query length, in characters.
#:
#: TestPlan.md TC-010 names ``settings.max_query_chars`` with this value, but no
#: such field exists on the frozen :class:`~axiom.config.Settings` and config.py
#: is not ours to extend, so the constant lives here. It is a *safety* bound, not
#: a tunable: it exists so a 10,000-character adversarial query truncates instead
#: of blowing the LLM context window and the embedder's sequence limit.
MAX_QUERY_CHARS = 2048

#: Ceiling on ``sub_queries`` for a first-pass plan (FR-03: "<= 3 ordered
#: sub-queries"). Schema.md section 10 allows 4; FR-03's 3 is the tighter bound
#: and each sub-query costs a full three-signal fan-out, so 3 it is.
MAX_SUB_QUERIES = 3

#: Ceiling on first-pass ``expansion_terms``.
#:
#: Expansion terms are appended to the BM25 query, where every extra term
#: dilutes the IDF contribution of the terms the user actually typed. Twelve is
#: roughly four expandable words' worth -- enough to bridge vocabulary, short of
#: drowning the query.
MAX_EXPANSION_TERMS = 12

#: Ceiling on ``expansion_terms`` after a refinement pass. Refinement's whole
#: job is to broaden, so the cap doubles rather than staying put.
MAX_EXPANSION_TERMS_REFINED = 24

#: Stand-in ``original_query`` for a zero-length query.
#:
#: ``QueryPlan.original_query`` is ``min_length=1``, so "" is unrepresentable,
#: but Rules.md Rule 3 forbids raising on it. The loop's "Query normalisation"
#: ladder rung short-circuits an empty query before retrieval anyway; this value
#: exists so the plan object that gets logged is still well-formed.
EMPTY_QUERY_PLACEHOLDER = "(empty query)"


#: Minimum length of an unquoted identifier candidate. TestPlan.md TC-008:
#: "identifiers are never single-char unless quoted".
MIN_IDENTIFIER_CHARS = 2

#: English function words that identifier extraction must never emit (TC-008).
#:
#: Deliberately *not* a full stopword list: code vocabulary overlaps English
#: heavily ("call", "file", "handler", "main", "test" are all real symbols), so
#: only words that cannot plausibly be a symbol the user means are listed.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "during",
        "each",
        "either",
        "else",
        "every",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "however",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "more",
        "most",
        "must",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }
)

#: Ordering cues that mark a query as compound (FR-03).
#:
#: Only *ordering* words, not plain conjunctions: "before"/"after" carry a
#: source-order claim ("calls X before Y") that no single retrieval can express,
#: which is exactly the archetype Q2 shape. A bare "and" usually joins two
#: descriptions of one thing, and splitting on it costs a fan-out for nothing.
_ORDERING_CUES: tuple[str, ...] = ("before", "after", "then", "prior to", "followed by")

_ORDERING_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _ORDERING_CUES) + r")\b", re.IGNORECASE
)

# --- Identifier shapes, in priority order (TechSpecifications.md 3.3.2) ------
# Ordered longest/most-specific first: when two shapes match at the same offset,
# the one listed earlier wins, so "utils.normalize" is never shredded into
# "utils" plus a stray camel fragment.
_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("quoted", QUOTED_LITERAL_RE),
    ("dotted", re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")),
    ("kebab", re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")),
    ("camel", re.compile(r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b")),
    ("snake", re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")),
    ("const", re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*\b")),
    ("call", re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()")),
)

#: Splits an identifier into its word parts for token-level synonym lookup:
#: ``handleDeeplink`` -> ``handle``, ``deeplink``.
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

#: Dotted paths made of one-character segments ("e.g.", "i.e.") are prose, not
#: module paths.
_MIN_DOTTED_SEGMENT = 2


def normalise_query(query: str, *, max_chars: int = MAX_QUERY_CHARS) -> str:
    """Strip control characters, collapse whitespace, and truncate.

    The three things that actually break downstream stages are C0/C1 control
    bytes (which corrupt log lines and JSON), ragged whitespace (which changes
    BM25 tokenisation for no semantic reason), and unbounded length (TC-010).
    Unicode and emoji are left alone -- they tokenise fine and dropping them
    would silently change a French or Hindi query's meaning (TC-011).
    """
    cleaned = "".join(ch for ch in query if ch in "\t\n\r" or unicodedata.category(ch)[0] != "C")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_chars:
        log_degradation(
            _LOG,
            "agent.planner:normalise_query",
            f"query is {len(cleaned)} chars, over the {max_chars}-char bound",
            f"truncated to {max_chars} chars",
        )
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def split_identifier(identifier: str) -> list[str]:
    """Break an identifier into lowercase word parts.

    ``handleDeeplink`` -> ``["handle", "deeplink"]``; ``parse_input`` ->
    ``["parse", "input"]``; ``bluetooth-settings`` -> ``["bluetooth",
    "settings"]``. Used only to feed the synonym map, never to widen the
    identifier list itself -- the structural retriever wants exact symbols.
    """
    parts: list[str] = []
    for raw in re.split(r"[^A-Za-z0-9]+", identifier):
        if not raw:
            continue
        for piece in _CAMEL_BOUNDARY_RE.split(raw):
            if piece:
                parts.append(piece.lower())
    return parts


def _acceptable(candidate: str, shape: str) -> bool:
    """Reject prose that happens to match an identifier shape (TC-008)."""
    if shape == "quoted":
        return bool(candidate.strip())
    if len(candidate) < MIN_IDENTIFIER_CHARS:
        return False
    if candidate.lower() in STOPWORDS:
        return False
    if shape == "dotted":
        return all(len(seg) >= _MIN_DOTTED_SEGMENT for seg in candidate.split("."))
    return True


def extract_identifiers(query: str, *, include_quoted_fragments: bool = True) -> list[str]:
    """Lift code identifiers out of raw query text (FR-02).

    Five shapes are recognised -- quoted literal, dotted path, kebab token,
    camelCase, snake_case, CONST_CASE, and ``foo(`` call form
    (TechSpecifications.md section 3.3.2). Matches are collected with their
    offsets and accepted left to right, skipping anything that overlaps an
    already-accepted span, so ``utils.normalize`` is emitted whole instead of
    competing with its own fragments. Order is first occurrence in the query and
    duplicates are dropped, which is the ``QueryPlan`` invariant.

    Args:
        query: Raw or normalised query text; both work.
        include_quoted_fragments: When true, a quoted literal containing a
            non-word separator is emitted *and* split into its parts --
            ``"bluetooth-settings"`` also yields ``bluetooth`` and ``settings``,
            because the literal is what BM25 wants and the parts are what the
            symbol table holds. Turned off by the classifier, which counts
            identifiers to detect the "where is X used" shape and would be
            misled by derived fragments inflating the count.

    Returns:
        Deduplicated identifiers in first-occurrence order. Empty list when the
        query is prose -- a normal outcome, never an error.
    """
    if not query:
        return []

    matches: list[tuple[int, int, str, str]] = []
    for shape, pattern in _SHAPES:
        for match in pattern.finditer(query):
            # Group 1 exists only for the quoted shape, where it is the literal
            # without its delimiters.
            text = match.group(1) if match.lastindex else match.group(0)
            start = match.start(1) if match.lastindex else match.start(0)
            matches.append((start, -len(text), shape, text))

    matches.sort(key=lambda item: (item[0], item[1]))

    out: list[str] = []
    seen: set[str] = set()
    claimed: list[tuple[int, int]] = []
    for start, neg_len, shape, text in matches:
        end = start - neg_len
        if any(start < c_end and c_start < end for c_start, c_end in claimed):
            continue
        if not _acceptable(text, shape):
            continue
        claimed.append((start, end))
        if text not in seen:
            seen.add(text)
            out.append(text)
        if shape == "quoted" and include_quoted_fragments:
            for fragment in re.split(r"\W+", text):
                if (
                    len(fragment) >= MIN_IDENTIFIER_CHARS
                    and fragment.lower() not in STOPWORDS
                    and fragment != text
                    and fragment not in seen
                ):
                    seen.add(fragment)
                    out.append(fragment)
    return out


def query_tokens(query: str, identifiers: Sequence[str] = ()) -> list[str]:
    """Lowercase word stream for synonym lookup: prose words plus symbol parts."""
    tokens = [word.lower() for word in _WORD_RE.findall(query)]
    for identifier in identifiers:
        tokens.extend(split_identifier(identifier))
    return tokens


def build_expansion_terms(
    query: str,
    identifiers: Sequence[str],
    *,
    limit: int = MAX_EXPANSION_TERMS,
    extra_tokens: Iterable[str] = (),
) -> list[str]:
    """Look up expansion terms and enforce the ``QueryPlan`` invariants (FR-02).

    Three filters, all of which exist to stop the BM25 query from being diluted:
    a term already present in the query verbatim adds nothing; a term that
    collides with an extracted identifier would violate the schema's
    disjointness invariant (and double-count in scoring); and the list is capped.

    Comparison is case-insensitive throughout, because ``API`` as an identifier
    and ``api`` as a synonym are the same token to the code-aware tokeniser.
    """
    lowered_query_words = {word.lower() for word in _WORD_RE.findall(query)}
    reserved = {identifier.lower() for identifier in identifiers}
    for identifier in identifiers:
        reserved.update(split_identifier(identifier))

    tokens = [*query_tokens(query, identifiers), *extra_tokens]
    out: list[str] = []
    seen: set[str] = set()
    for term in expand_tokens(tokens):
        key = term.lower()
        if key in reserved or key in lowered_query_words or key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def decompose(
    query: str,
    identifiers: Sequence[str],
    *,
    limit: int = MAX_SUB_QUERIES,
) -> list[str]:
    """Split a compound query into ordered sub-queries (FR-03).

    Fires only on the shape FR-03 actually names: two or more identifiers joined
    by an ordering cue, i.e. archetype Q2 ("Which files call tool XYZ *before*
    tool ABC?"). That decomposes into one retrieval per identifier plus an
    ordering check, because a single retrieval cannot express "X precedes Y" --
    the fan-out has to look for each symbol and the fusion has to see both.

    Everything else returns ``[]``, which makes ``QueryPlan.effective_queries``
    fall back to ``[original_query]`` -- one retrieval, exactly as
    TechSpecifications.md section 3.3.4 specifies for a non-compound query.

    Note:
        This is the one place where the heuristic engine goes beyond what
        Appflow.md Flow 4 assumes (it treats decomposition as LLM-only). FR-03
        is a P0 requirement that never mentions the LLM, and archetype Q2 is
        unanswerable without it, so the conservative trigger above runs in both
        modes rather than leaving a P0 requirement unimplemented when the LLM is
        off.
    """
    if limit <= 0 or len(identifiers) < 2:
        return []
    ordering = _ORDERING_RE.search(query)
    if ordering is None:
        return []

    first, second = identifiers[0], identifiers[1]
    head_end = query.find(first)
    head = query[:head_end].strip(" ,;:?") if head_end > 0 else ""
    head = " ".join(head.split())

    per_identifier = [f"{head} {ident}".strip() for ident in (first, second)]
    ordering_check = f"{first} {ordering.group(1).lower()} {second}"

    out: list[str] = []
    for candidate in (*per_identifier, ordering_check):
        cleaned = candidate.strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out[:limit]


def build_plan(
    query: str,
    settings: Settings,
    *,
    query_type: QueryType | None = None,
) -> QueryPlan:
    """Build the first-pass :class:`QueryPlan` for a query (FR-01, FR-02, FR-03).

    Args:
        query: Raw user query. Normalised here; ``original_query`` records the
            normalised text, which is what every pass and every log line then
            agrees on.
        settings: Active configuration. Supplies the weight vector and gates the
            LLM.
        query_type: Forces the classification instead of running it, for the
            API's ``query_type`` override (API.md section 3).

    Returns:
        A valid ``QueryPlan``. Never raises on query text: a query that
        normalises to nothing still yields a plan, marked ``HYBRID`` with no
        identifiers, so the caller can short-circuit it with a well-formed empty
        result rather than an exception (Rules.md Rule 3).
    """
    normalised = normalise_query(query)
    if not normalised:
        log_degradation(
            _LOG,
            "agent.planner:build_plan",
            "query is empty after control-character stripping",
            "HYBRID plan over the raw text, no identifiers",
        )
        # ``original_query`` has min_length=1, so a zero-length query cannot be
        # represented. Falling back to the raw text keeps whatever the user
        # actually typed (whitespace included) visible in the logs; only the
        # genuinely empty string needs the placeholder.
        original = query or EMPTY_QUERY_PLACEHOLDER
        return QueryPlan(
            original_query=original,
            query_type=QueryType.HYBRID,
            strategy_weights=settings.strategy_weights_for(QueryType.HYBRID),
        )

    identifiers = extract_identifiers(normalised)
    primary = extract_identifiers(normalised, include_quoted_fragments=False)
    resolved_type = query_type or classify(normalised, settings, identifiers=primary)

    expansion_terms = build_expansion_terms(normalised, identifiers)
    sub_queries = decompose(normalised, identifiers)

    if settings.llm_enabled:
        expansion_terms, sub_queries = _llm_supplement(
            normalised, identifiers, expansion_terms, sub_queries, settings
        )

    return QueryPlan(
        original_query=normalised,
        query_type=resolved_type,
        sub_queries=sub_queries,
        extracted_identifiers=identifiers,
        expansion_terms=expansion_terms,
        strategy_weights=settings.strategy_weights_for(resolved_type),
    )


def _llm_supplement(
    query: str,
    identifiers: Sequence[str],
    expansion_terms: list[str],
    sub_queries: list[str],
    settings: Settings,
) -> tuple[list[str], list[str]]:
    """Let the LLM *add* to the heuristic plan, never replace it.

    Keeping the curated map's terms and appending the model's is what makes
    ``AXIOM_LLM_ENABLED=false`` behaviourally close to LLM-on rather than a
    degraded mode (TechSpecifications.md section 3.3.3). Any failure inside the
    adapter has already been degraded to an empty list, so there is nothing to
    catch here.
    """
    from axiom.agent.llm import get_llm

    llm = get_llm(settings)
    if llm is None:
        return expansion_terms, sub_queries

    reserved = {term.lower() for term in expansion_terms}
    reserved.update(identifier.lower() for identifier in identifiers)
    for suggestion in llm.expand(query, identifiers):
        if len(expansion_terms) >= MAX_EXPANSION_TERMS:
            break
        if suggestion.lower() not in reserved:
            reserved.add(suggestion.lower())
            expansion_terms.append(suggestion)

    if not sub_queries:
        proposed = llm.decompose(query, MAX_SUB_QUERIES)
        # A one-element decomposition is the LLM echoing the query back; that is
        # not a decomposition and would cost a redundant fan-out.
        if len(proposed) > 1:
            sub_queries = proposed[:MAX_SUB_QUERIES]
    return expansion_terms, sub_queries


def plans_differ(previous: QueryPlan, candidate: QueryPlan) -> bool:
    """Whether two plans would issue different retrievals.

    ``original_query`` is excluded by construction -- it is identical across all
    passes of one request -- so this compares exactly the fields refinement is
    allowed to touch. The agent loop uses it to stop with
    ``stop_reason="no_new_query"`` instead of paying for an identical pass
    (Appflow.md Flow 3, TestPlan.md TC-089).
    """
    return (
        previous.query_type != candidate.query_type
        or previous.sub_queries != candidate.sub_queries
        or previous.extracted_identifiers != candidate.extracted_identifiers
        or previous.expansion_terms != candidate.expansion_terms
        or previous.strategy_weights != candidate.strategy_weights
    )


def refine_plan(
    plan: QueryPlan,
    results: Sequence[FusedResult],
    settings: Settings,
) -> QueryPlan:
    """Rewrite a plan for the next agent pass (FR-13).

    Two modes, chosen from the evidence the sufficiency predicate already looked
    at -- never from the chunk text, which the agent is structurally forbidden to
    read (ADR-008):

    * **Broaden** when the last pass returned fewer than
      ``agent_sufficiency_min_results`` candidates at all. The query was too
      narrow, so second-order synonyms go in (synonyms of the synonyms) and each
      identifier gets its own retrieval.
    * **Sharpen** when there were plenty of candidates but none scored. The query
      was diffuse, so an identifier-led sub-query goes first and the original
      trails it, letting the fan-out spend its width on the exact symbols.

    ``original_query`` is never touched: Schema.md section 10 requires it to be
    identical across all passes of one request, and the returned plan is a new
    frozen object, so the caller's plan is not mutated either.

    Returns:
        A plan that issues different retrievals, or -- when there is genuinely
        nothing new to try -- a plan equal to ``plan`` on every refinable field.
        Callers detect that with :func:`plans_differ` and stop rather than
        re-running an identical pass.
    """
    identifiers = list(plan.extracted_identifiers)
    broaden = len(results) < settings.agent_sufficiency_min_results

    # Feeding the previous pass's expansion terms back in as tokens is what
    # makes this second-order: the synonyms of the synonyms, capped harder than
    # the first pass because breadth is the whole point of a refinement.
    expansion_terms = build_expansion_terms(
        plan.original_query,
        identifiers,
        limit=MAX_EXPANSION_TERMS_REFINED,
        extra_tokens=[term.lower() for term in plan.expansion_terms] if broaden else (),
    )

    sub_queries = _refined_sub_queries(plan, identifiers, expansion_terms, broaden=broaden)

    refined = QueryPlan(
        original_query=plan.original_query,
        query_type=plan.query_type,
        sub_queries=sub_queries,
        extracted_identifiers=identifiers,
        expansion_terms=expansion_terms,
        strategy_weights=dict(plan.strategy_weights),
    )
    if not plans_differ(plan, refined):
        _LOG.warning(
            "refinement produced no new query",
            extra={"axiom_extra": {"event": "no_new_query", "stage": "agent"}},
        )
    return refined


def _refined_sub_queries(
    plan: QueryPlan,
    identifiers: Sequence[str],
    expansion_terms: Sequence[str],
    *,
    broaden: bool,
) -> list[str]:
    """Build pass-N sub-queries that are guaranteed to differ from pass N-1's.

    Kept separate from :func:`refine_plan` because the ordering rules here are
    fiddly and each branch has to justify itself: a refinement that re-issues the
    previous pass's queries burns the loop's entire remaining budget for nothing
    (Appflow.md Flow 3's ``no_new_query`` guard exists because this is easy to
    get wrong).
    """
    original = plan.original_query
    candidates: list[str] = []

    if broaden:
        # One retrieval per symbol, widest first, then the original as a net.
        candidates.extend(f"{ident} {original}" for ident in identifiers[:2])
        if expansion_terms:
            candidates.append(" ".join([original, *expansion_terms[:4]]))
        candidates.append(original)
    else:
        if identifiers:
            candidates.append(" ".join([*identifiers[:3], *expansion_terms[:2]]))
        if expansion_terms:
            candidates.append(" ".join([original, *expansion_terms[:3]]))
        candidates.append(original)

    out: list[str] = []
    for candidate in candidates:
        cleaned = " ".join(candidate.split())
        if cleaned and cleaned not in out:
            out.append(cleaned)
        if len(out) >= MAX_SUB_QUERIES:
            break

    # A single sub-query equal to the original is the same retrieval the last
    # pass already ran; report "nothing new" honestly by keeping the field empty.
    if out == [original] and not plan.sub_queries:
        return []
    return out


#: Canonical name used by Appflow.md and Rules.md for the refinement entrypoint.
refine = refine_plan
