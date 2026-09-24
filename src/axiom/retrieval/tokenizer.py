"""The code-aware tokenizer shared by the sparse index and the sparse query path.

This module exists so that exactly **one** function produces the token stream on
both sides of BM25. A tokeniser that diverges between index time and query time
does not fail loudly -- it quietly returns nothing, and the sparse signal looks
merely "weak" rather than broken. Schema.md section 14.6 records the tokeniser
configuration inside ``sparse.bm25s/prism_meta.json`` for exactly that reason:
bm25s does not persist its tokeniser, so PRISM persists ours.

The tokenisation is deliberately *expansive* rather than *reductive*. A single
source atom such as ``utils.handleDeeplink_v2`` emits the intact identifier as
well as every split fragment (TC-042), because the three query archetypes need
different granularities of the same text:

* Q1 ("what happens before the main function") wants the split words, so the
  dense-ish vocabulary of ``handle``/``deeplink`` is reachable from prose.
* Q2 ("which files call XYZ before ABC") wants the intact identifier, so an
  exact symbol name outranks a coincidental word match.
* Q3 ("where is the Bluetooth-settings deeplink used") wants the *literal*
  ``bluetooth-settings`` to survive kebab-case intact -- it is the highest
  precision signal that exists for that query, and splitting it away would
  leave only the two common words ``bluetooth`` and ``settings``.

Emitting all three costs index size and buys recall that cannot be recovered
later. Per-atom deduplication keeps that cost honest: a plain token like ``tool``
contributes term frequency 1, not 4, so compound identifiers do not silently
outweigh simple ones.

No stopword list, per TechSpecifications.md section 3.4.1 -- ``id``, ``in``, and
``on`` are real identifiers in code, and a prose stopword list would delete them.

Pure Python and dependency-free by construction: this is the one retrieval
component that must work on a machine with nothing optional installed (NFR-07).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from axiom.core.logging import get_logger, log_degradation

_LOG = get_logger("retrieval.tokenizer")

#: Identity recorded in ``prism_meta.json``. Schema.md section 14.6 names the
#: tokeniser by the ``axiom.indexing.sparse`` path, which stays importable
#: because ``indexing/sparse.py`` re-exports the symbols defined here.
TOKENIZER_NAME = "axiom.indexing.sparse:CodeTokenizer"

#: Bump when a change to :func:`CodeTokenizer.tokenize` alters the token stream.
#: A mismatch against a persisted index means query and corpus disagree, and the
#: index must be rebuilt rather than queried.
TOKENIZER_VERSION = 1

#: A maximal run of identifier-ish characters: word characters plus the three
#: separators that carry meaning in code (``.`` module paths, ``_`` snake_case,
#: ``-`` kebab-case and string literals).
#:
#: ReDoS-safe by inspection and required to stay so by TC-047: one literal atom
#: followed by one starred character class, with no nested quantifier, no
#: alternation, and no backreference. Matching is linear in input length.
_ATOM_RE = re.compile(r"\w[\w.\-]*")

#: Trimmed from both ends of an atom so that sentence punctuation ("the
#: deeplink.") and dangling hyphens do not become part of a token. ``_`` is
#: deliberately absent -- ``_private`` is a real identifier.
_ATOM_EDGE_CHARS = ".-"

#: Delimiters that split a dot-segment into words. Kebab-case is included
#: because string literals in JS overwhelmingly use it (``'bluetooth-settings'``)
#: while source identifiers cannot.
_SNAKE_DELIMITER = "_"
_KEBAB_DELIMITER = "-"


def _split_camel(part: str) -> list[str]:
    """Split a single word on case boundaries only.

    Two boundary rules, both required and both directional:

    1. ``lower|digit -> Upper`` splits *before* the upper character, which is
       ordinary camelCase (``handleDeeplink`` -> ``handle``, ``Deeplink``).
    2. ``UPPER-run -> Upper+lower`` splits *before the last* upper character,
       which is the acronym case (``XMLHttpRequest`` -> ``XML``, ``Http``,
       ``Request``). Without this rule an acronym swallows the word after it.

    Letter/digit boundaries are deliberately **not** split: ``v2`` is one token,
    not ``v`` plus ``2``. TC-042 asserts ``v2`` survives, and a bare ``2`` is
    pure noise in a code corpus.

    A hand-written scan rather than a regex: it is O(len(part)) with no
    backtracking at all, which is the cheapest possible way to satisfy TC-047's
    adversarial-input budget, and it behaves sanely on caseless scripts (a
    CJK or emoji-adjacent run simply yields itself).
    """
    if not part:
        return []
    out: list[str] = []
    start = 0
    for index in range(1, len(part)):
        previous = part[index - 1]
        current = part[index]
        if current.isupper() and not previous.isupper():
            out.append(part[start:index])
            start = index
        elif previous.isupper() and current.islower() and index - 1 > start:
            out.append(part[start : index - 1])
            start = index - 1
    out.append(part[start:])
    return out


def _split_delimited(segment: str, snake: bool, kebab: bool) -> list[str]:
    """Split a dot-segment on the enabled word delimiters, dropping empties."""
    parts = [segment]
    if snake:
        parts = [piece for part in parts for piece in part.split(_SNAKE_DELIMITER)]
    if kebab:
        parts = [piece for part in parts for piece in part.split(_KEBAB_DELIMITER)]
    return [part for part in parts if part]


@dataclass(frozen=True, slots=True)
class CodeTokenizer:
    """Deterministic code tokeniser, configured by the flags persisted on disk.

    Every field corresponds to a key in ``sparse.bm25s/prism_meta.json``
    (Schema.md section 14.6) and genuinely changes the output -- the record on
    disk is a description of behaviour, not decoration. The defaults are the
    documented ones and are what every PRISM build uses; the flags exist so that
    a future ablation can be recorded in the index it produced.

    Frozen because the same instance is shared between the index builder and the
    query path, and a mutated flag would desynchronise them mid-run.
    """

    split_camel_case: bool = True
    split_snake_case: bool = True
    split_kebab_case: bool = True
    split_dots: bool = True
    keep_intact_identifier: bool = True
    lowercase: bool = True

    def _case(self, token: str) -> str:
        """Apply the matching-case policy. Uniform across corpus and query."""
        return token.lower() if self.lowercase else token

    def _forms(self, atom: str) -> list[str]:
        """Every token one source atom contributes, in deterministic order.

        Order is intact-atom, then dot segments, then delimiter parts, then
        camel fragments -- coarsest to finest. Deduplication is *per atom*: the
        four passes over ``tool`` all produce ``tool``, and emitting it four
        times would quadruple its term frequency relative to a compound
        identifier that legitimately produced four distinct tokens. Across
        atoms there is no deduplication, because a term genuinely repeated in a
        chunk must raise that chunk's BM25 term frequency.
        """
        seen: set[str] = set()
        forms: list[str] = []

        def add(token: str) -> None:
            cased = self._case(token)
            if cased and cased not in seen:
                seen.add(cased)
                forms.append(cased)

        segments = [s for s in atom.split(".") if s] if self.split_dots else [atom]
        if not segments:
            return forms
        if self.keep_intact_identifier or len(segments) == 1:
            add(atom)

        for segment in segments:
            parts = _split_delimited(segment, self.split_snake_case, self.split_kebab_case)
            if not parts:
                continue
            if self.keep_intact_identifier or len(parts) == 1:
                add(segment)
            for part in parts:
                subs = _split_camel(part) if self.split_camel_case else [part]
                if self.keep_intact_identifier or len(subs) == 1:
                    add(part)
                for sub in subs:
                    add(sub)
        return forms

    def tokenize(self, text: str) -> list[str]:
        """Tokenise arbitrary text. Never raises; an empty result is legal.

        This is the code-aware rung of the sparse degradation ladder
        (Rules.md section 3). Callers that need the lower rungs should use
        :func:`tokenize_with_fallback` rather than reimplementing them.
        """
        if not text:
            return []
        tokens: list[str] = []
        for match in _ATOM_RE.finditer(text):
            atom = match.group(0).strip(_ATOM_EDGE_CHARS)
            if atom:
                tokens.extend(self._forms(atom))
        return tokens

    def config(self) -> dict[str, object]:
        """The tokeniser record written into ``prism_meta.json``.

        Schema.md section 14.6 fixes these keys; ``split_kebab_case`` is an
        additive key covering the literal-splitting behaviour Q3 depends on.
        ``stopwords`` is reported as the literal string ``"none"`` because the
        absence of a stopword list is a deliberate decision that must be visible
        in the artifact, not an omission inferred from a missing key.
        """
        return {
            "tokenizer": TOKENIZER_NAME,
            "tokenizer_version": TOKENIZER_VERSION,
            "split_camel_case": self.split_camel_case,
            "split_snake_case": self.split_snake_case,
            "split_kebab_case": self.split_kebab_case,
            "split_dots": self.split_dots,
            "keep_intact_identifier": self.keep_intact_identifier,
            "lowercase": self.lowercase,
            "stopwords": "none",
        }


#: The single shared instance. Index time and query time both route through it,
#: which is the whole point of this module -- see the module docstring.
DEFAULT_TOKENIZER = CodeTokenizer()


def tokenize(text: str) -> list[str]:
    """Tokenise with the default configuration. The one function both paths call."""
    return DEFAULT_TOKENIZER.tokenize(text)


def whitespace_tokenize(text: str) -> list[str]:
    """Second rung of the ladder: lowercased whitespace split, nothing else.

    Reached when the code-aware pass yields nothing, which in practice means the
    query contained no word characters at all. Splitting on whitespace cannot
    match a code corpus often, but it is strictly better than an empty query and
    it keeps the failure visible in the logs rather than in the scores.
    """
    return [piece for piece in text.lower().split() if piece]


def tokenize_with_fallback(text: str, component: str = "retrieval.sparse") -> list[str]:
    """Run the full sparse tokenisation ladder, logging each descent.

    Ladder, verbatim from Rules.md section 3: *code-aware tokenise -> whitespace
    tokenise -> empty list*. Bad input never raises here (Rules.md Rule 3,
    worked as AP-11) -- query 9,000 of a two-hour eval run being ``"   "`` must
    cost one warning line, not the run.

    Args:
        text: Raw query text, or any string to be matched against the index.
        component: Component name for the degradation record, so a degrade from
            the index builder is distinguishable from one on the query path.

    Returns:
        The token stream, possibly empty. An empty stream means the caller
        should report an absent signal, which fusion already renormalises
        around (TestPlan.md TC-055).
    """
    tokens = tokenize(text)
    if tokens:
        return tokens

    tokens = whitespace_tokenize(text)
    if tokens:
        log_degradation(
            _LOG,
            component,
            "code-aware tokeniser yielded zero tokens",
            "whitespace tokenise",
        )
        return tokens

    log_degradation(
        _LOG, component, "query yielded zero tokens after whitespace split", "empty list"
    )
    return []
