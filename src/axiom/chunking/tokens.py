"""Cheap, deterministic token estimation used to size chunks.

TechSpecifications.md section 4.4.1 step 2 says the token count that matters is
"the one the model will actually see" -- the embedder's own tokenizer. We do not
use it here, and the reason is NFR-07: ``transformers``/``tokenizers`` are
optional extras that may not be installed, and the chunker runs *before* any
model is loaded. A chunker that could only run with a 400 MB tokenizer on disk
would make the whole index path depend on an optional dependency, which is
exactly the failure mode the degradation ladder exists to prevent.

So this module approximates a BPE tokenizer over source code with a pure-stdlib
regex scan. The approximation, stated honestly:

* An identifier is split on camelCase / snake_case / digit boundaries -- which is
  what a code-trained BPE vocabulary does to an unseen identifier -- and each
  resulting word costs ``ceil(len / CHARS_PER_SUBTOKEN)`` tokens, minimum one.
* A newline costs one token (real tokenizers fold the newline and the following
  indentation into a single token); intra-line whitespace costs nothing, because
  BPE merges a leading space into the token that follows it.
* Every operator, bracket, or quote character costs one token. This
  *over*-counts multi-character operators (``===`` is usually one token, we
  charge three), which is the safe direction: it makes the estimate pessimistic,
  so chunks split slightly early and never overrun the reranker's window.
* Non-ASCII characters cost one token each, which is roughly right for CJK and
  pessimistic for accented Latin text.

Measured over the fixtures in this repo it lands at 2.4-2.7 characters per
estimated token on ordinary JavaScript and 1.9 on a minified bundle, against the
~3-4 characters per token a real code BPE vocabulary achieves. It therefore
*over*-estimates by roughly 25-30%, which is the direction that is safe: chunks
come out somewhat smaller than the 512-token ceiling rather than somewhat
larger, and the ceiling exists to stay inside the reranker's ~4096-character
window (TechSpecifications.md section 5.4). Calibrating the bias away is
``T-070``'s job, on the same pass that measures the bounds themselves -- the
bounds it feeds (``chunk_min_tokens``/``chunk_target_tokens``) are ``PLACEHOLDER``
under OQ-09, so tightening this estimator before those are measured would be
fitting one unmeasured number to another.

Determinism (NFR-08) is total: the same string always yields the same integer,
with no model, no download, and no locale dependence.
"""

from __future__ import annotations

import re
from functools import lru_cache

#: Characters per sub-token inside a single identifier word. A BPE vocabulary
#: holds whole common words (``resolve``, ``tool``) and splits rarer ones into
#: 4-6 character pieces; 5 is the midpoint.
CHARS_PER_SUBTOKEN = 5

#: Characters per token inside a numeric literal. Digit runs tokenise densely
#: (most vocabularies cap digit merges at three characters).
CHARS_PER_NUMBER_TOKEN = 3

#: Atom scanner. Order is significant: the identifier branch runs before the
#: numeric branch so ``e5`` reads as a name, and the catch-all single-character
#: branch is last so every byte of the input is accounted for exactly once.
_ATOM_RE = re.compile(
    r"[A-Za-z_$][A-Za-z0-9_$]*"  # identifier or keyword
    r"|\d[\d_.a-fA-FxXeE]*"  # numeric literal (deliberate superset)
    r"|\n"  # newline: one token, folded with the indent that follows
    r"|[ \t\r]+"  # intra-line whitespace: free, BPE merges it rightwards
    r"|.",  # operator, bracket, quote, or non-ASCII character
    re.DOTALL,
)

#: Sub-word splitter inside one identifier: ``parseHTTPResponse_v2`` ->
#: ``parse``, ``HTTP``, ``Response``, ``_``, ``v``, ``2``.
_SUBWORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+|[_$]+")


@lru_cache(maxsize=8192)
def _identifier_tokens(atom: str) -> int:
    """Token cost of one identifier. Cached because identifiers repeat heavily."""
    words = _SUBWORD_RE.findall(atom) or [atom]
    return sum(max(1, -(-len(word) // CHARS_PER_SUBTOKEN)) for word in words)


def _atom_cost(atom: str) -> int:
    """Token cost of one scanned atom."""
    head = atom[0]
    if head == "\n":
        return 1
    if head in " \t\r":
        return 0
    if head.isascii() and (head.isalpha() or head in "_$"):
        return _identifier_tokens(atom)
    if head.isdigit():
        return max(1, -(-len(atom) // CHARS_PER_NUMBER_TOKEN))
    return 1


def estimate_tokens(text: str, *, limit: int | None = None) -> int:
    """Estimate how many tokens an embedder would see for ``text``.

    Args:
        text: Source text, verbatim.
        limit: Optional early-exit ceiling. When the running total first exceeds
            ``limit`` the scan stops and that partial total is returned. This is
            what keeps a 1.2 MB minified bundle (TC-023) from being fully scanned
            once per candidate AST node: the answer the caller needs is
            "is this over budget?", and that is decidable after roughly
            ``limit`` tokens' worth of input. A value returned under early exit
            is a lower bound, valid only for comparison against ``limit``.

    Returns:
        Estimated token count, always ``>= 0``.
    """
    total = 0
    for match in _ATOM_RE.finditer(text):
        total += _atom_cost(match.group())
        if limit is not None and total > limit:
            return total
    return total


def prefix_within_budget(text: str, budget: int) -> int:
    """Length in characters of the longest prefix estimated at ``<= budget`` tokens.

    The character-level cut used when a single source line is larger than a whole
    chunk -- a minified bundle (TC-023). Cutting by a fixed character count would
    either overshoot the budget on dense punctuation or waste most of it on
    sparse text; cutting on the estimator's own atom boundaries keeps every
    fragment inside the budget while still filling it.

    The returned index always lands on an atom boundary, and atoms never straddle
    a character, so the caller can slice ``text[:n]`` and re-encode safely. At
    least one atom is always consumed, so a caller looping on this terminates.
    """
    if budget <= 0:
        return len(text)
    total = 0
    end = 0
    for match in _ATOM_RE.finditer(text):
        cost = _atom_cost(match.group())
        if end > 0 and total + cost > budget:
            return end
        total += cost
        end = match.end()
        if total >= budget:
            return end
    return len(text)


def exceeds_budget(text: str, budget: int) -> bool:
    """True when ``text`` is estimated to be larger than ``budget`` tokens.

    Bounded work: the scan stops as soon as the verdict is certain, so asking
    this about a megabyte-long node costs ``O(budget)``, not ``O(len(text))``.
    """
    if budget <= 0:
        return True
    return estimate_tokens(text, limit=budget) > budget


def below_floor(text: str, floor: int) -> bool:
    """True when ``text`` is estimated to be smaller than the merge floor.

    The sub-floor test of TechSpecifications.md section 4.4.1 step 4. Cheap for
    the same reason as :func:`exceeds_budget`: a chunk that is obviously large
    is rejected after ``floor`` tokens of scanning.
    """
    if floor <= 0:
        return False
    return estimate_tokens(text, limit=floor) < floor
