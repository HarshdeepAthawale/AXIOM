"""Query classification into :class:`~axiom.schema.enums.QueryType`.

The heuristic rule engine below is the **default** path, not a fallback rung
reached by failure (TechSpecifications.md section 3.3): with
``AXIOM_LLM_ENABLED=false`` -- the configuration a network-restricted evaluator's
laptop runs under -- every query is classified here, and all three PRD archetypes
must still land on the right ``QueryType``. The LLM, when enabled, runs *before*
this and only overrides it with a value that parses back to the same enum.

Ladder (Rules.md section 3): LLM classification -> heuristic rule engine ->
``QueryType.HYBRID``. ``HYBRID`` is also the answer to every ambiguity inside the
heuristic itself, because Schema.md section 3.1 is explicit that a wrong
confident classification costs more NDCG than a correct-but-flat weighting.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from axiom.config import Settings
from axiom.core.logging import get_logger, log_degradation
from axiom.schema.enums import QueryType

_LOG = get_logger("agent.classifier")

#: Call-graph, import/export, and file-relationship cues -> ``STRUCTURAL``.
#:
#: Note what is deliberately *absent*: bare ``before``/``after``. The reference
#: sketch in TechSpecifications.md section 3.3.1 lists them as standalone
#: alternatives, but the cue table one paragraph above it scopes them to
#: ``calls? before|after`` and ``invoke(s)? before``, and the standalone reading
#: misclassifies the document's own worked example -- archetype Q1 ("How is the
#: input preprocessed *before* going to the main function?") would fire both the
#: structural and semantic sets and collapse to HYBRID, contradicting the same
#: section's stated expectation of SEMANTIC. The cue table wins; ordering words
#: only count as structural when they qualify a call.
#:
#: ``referenced``/``used`` are absent for the mirror-image reason: they are the
#: natural phrasing of a USAGE query (archetype Q3), not a call-graph question.
_STRUCTURAL_RE = re.compile(
    r"\b(?:"
    r"calls?|called|calling|callers?|callees?|"
    r"imports?|imported|exports?|exported|"
    r"which\s+files?|what\s+files?|"
    r"invokes?|invoked|invoking|"
    r"depends?\s+on|dependenc(?:y|ies)|"
    r"subclass(?:es)?|extends|implements|inherits?|inherited|"
    r"call\s+(?:graph|order)|before\s+(?:calling|invoking)|after\s+(?:calling|invoking)"
    r")\b",
    re.IGNORECASE,
)

#: Intent-level, "explain the behaviour" cues -> ``SEMANTIC``.
#:
#: The ``\w*`` tails matter: a user writes "preprocessed", "validating",
#: "normalisation", and a ``\b``-anchored bare stem would miss all three.
_SEMANTIC_RE = re.compile(
    r"(?:"
    r"\bhow\s+(?:is|are|do|does|did|can|would|should)\b|"
    r"\bwhat\s+happens\b|\bwhy\s+(?:is|are|do|does)\b|"
    r"\bexplain\b|\bdescribe\b|\bwalk\s+me\s+through\b|"
    r"\bpurpose\s+of\b|\bresponsible\s+for\b|\bflow\s+of\b|"
    r"\bgoing\s+to\b|\bend[-\s]to[-\s]end\b|\bunder\s+the\s+hood\b|"
    r"\b(?:pre|post)process\w*|\bnormali[sz]\w*|\bsanitiz\w*|\btransform\w*|"
    r"\bvalidat\w*|\bhandl\w*|\bprocess\w*|\binitiali[sz]\w*|\bserializ\w*"
    r")",
    re.IGNORECASE,
)

#: A quoted literal anywhere in the query is the strongest USAGE tell there is:
#: the user has typed the exact token they expect to find in the corpus.
_QUOTED_RE = re.compile(r"['\"`]([^'\"`]+)['\"`]")

#: Public alias. ``agent.planner`` extracts the same literals as identifiers and
#: must use the identical pattern, or a query could be classified USAGE on a
#: literal the planner then fails to extract.
QUOTED_LITERAL_RE = _QUOTED_RE


def _mask_identifiers(query: str, identifiers: Sequence[str]) -> str:
    """Blank out extracted identifiers before cue matching.

    A cue word only means something when the *user* wrote it as prose. "What
    imports ``utils.normalize``?" carries the semantic stem ``normali[sz]``
    inside a symbol the user typed verbatim, and counting it fires two cue sets
    and drops an unambiguous structural query to HYBRID. Masking first keeps the
    cue sets reading intent rather than spelling.

    Longest identifier first, so masking ``bluetooth-settings`` does not leave a
    dangling ``-settings`` behind for a shorter identifier to half-match.
    """
    if not identifiers:
        return query
    masked = query
    for identifier in sorted(identifiers, key=len, reverse=True):
        masked = masked.replace(identifier, " ")
    return masked


def classify_heuristic(query: str, identifiers: Sequence[str]) -> QueryType:
    """Score the query against three disjoint cue sets and take the lone winner.

    Exactly one cue set firing is treated as a confident classification; zero or
    two or more is ambiguity, and ambiguity resolves to ``HYBRID`` by rule rather
    than by picking a favourite (Schema.md section 3.1).

    Args:
        query: Raw query text. Not normalised here -- the regexes are
            case-insensitive and whitespace-tolerant, so normalisation would only
            hide what the user actually typed.
        identifiers: Primary extracted identifiers (see
            :func:`axiom.agent.planner.extract_identifiers`). Only the *count*
            is read: one identifier and no other cue is the "where is X used"
            shape, several identifiers is a relationship question, none is prose.

    Returns:
        The winning :class:`QueryType`, or ``HYBRID``.
    """
    prose = _mask_identifiers(query, identifiers)
    structural_hit = bool(_STRUCTURAL_RE.search(prose))
    semantic_hit = bool(_SEMANTIC_RE.search(prose))
    # The lone-identifier branch is conditioned on the other two cue sets staying
    # silent, exactly as the cue table in TechSpecifications.md section 3.3.1
    # words it ("contains exactly one extracted identifier **and** no
    # structural/semantic cue fires"). The reference sketch below the table drops
    # that conjunction, which costs real classifications: "Which files import
    # tools.registry?" has one identifier and an unmistakable structural cue, and
    # the unconditional reading collapses it to HYBRID.
    usage_hit = bool(_QUOTED_RE.search(query)) or (
        len(identifiers) == 1 and not structural_hit and not semantic_hit
    )

    hits = (structural_hit, semantic_hit, usage_hit)
    if sum(hits) != 1:
        return QueryType.HYBRID
    if structural_hit:
        return QueryType.STRUCTURAL
    if usage_hit:
        return QueryType.USAGE
    return QueryType.SEMANTIC


def classify(
    query: str,
    settings: Settings,
    *,
    identifiers: Sequence[str] | None = None,
) -> QueryType:
    """Classify a query, walking the declared degradation ladder.

    Rungs, in order: the local GGUF LLM (only when ``settings.llm_enabled`` and
    ``llama_cpp`` is importable and the weights are on disk), the heuristic rule
    engine, then ``HYBRID``. Every rung below the first logs a degradation --
    except the LLM-disabled case, which is a *selected mode* rather than a
    failure (Appflow.md Flow 4) and so is logged at debug level only.

    Args:
        query: Raw user query. Empty or whitespace-only input is not an error
            here (Rules.md Rule 3); it simply has no cues and yields ``HYBRID``.
        settings: Active configuration. Supplies ``llm_enabled`` and, through the
            LLM adapter, the seed and token cap.
        identifiers: Pre-extracted identifiers, passed in by
            :func:`axiom.agent.planner.build_plan` so extraction runs once per
            plan instead of twice. Extracted here when omitted.

    Returns:
        The query's :class:`QueryType`. Never raises.
    """
    if identifiers is None:
        # Imported here, not at module scope: planner imports this module for the
        # cue regexes, and a top-level import in both directions is a cycle.
        from axiom.agent.planner import extract_identifiers

        identifiers = extract_identifiers(query, include_quoted_fragments=False)

    if settings.llm_enabled:
        from axiom.agent.llm import get_llm

        llm = get_llm(settings)
        if llm is not None:
            verdict = llm.classify(query)
            if verdict is not None:
                return verdict
            log_degradation(
                _LOG,
                "agent.classifier:classify",
                "LLM returned no parsable QueryType",
                "heuristic rule engine",
            )
    else:
        _LOG.debug("llm disabled; classifying with the heuristic rule engine")

    try:
        return classify_heuristic(query, identifiers)
    except re.error as exc:  # pragma: no cover - the patterns are literals
        log_degradation(
            _LOG, "agent.classifier:classify", f"heuristic engine failed: {exc}", "QueryType.HYBRID"
        )
        return QueryType.HYBRID
