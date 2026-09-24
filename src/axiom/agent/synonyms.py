"""The curated code-synonym map behind ``QueryPlan.expansion_terms``.

Expansion exists because English query vocabulary and code identifier vocabulary
are different languages: a user asks about "preprocessing" and the repository
calls it ``normalize`` / ``sanitize`` / ``transform``. Dense retrieval already
covers that gap statistically, which is exactly why these terms are appended to
the *sparse* query only (TechSpecifications.md section 3.3.3) -- adding them to
the dense input would blur the query vector instead of sharpening it.

The map is a static literal, not a generated or learned artifact. That is a
determinism requirement (Rules.md Rule 2: stages are pure), and it is what makes
expansion identically available with ``AXIOM_LLM_ENABLED=false``: the LLM path is
permitted to *supplement* this table, never to replace it, so the heuristic mode
stays behaviourally close to the LLM mode rather than strictly worse (NFR-07).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

#: Query term -> related code vocabulary, all lowercase, order-stable.
#:
#: Curated, not exhaustive. Entries earn their place by being a word a developer
#: plausibly *says* whose code spelling is plausibly different; a synonym that is
#: merely a longer way of saying the same token adds BM25 noise for no recall.
#: The first seven rows are quoted verbatim from TechSpecifications.md section
#: 3.3.3; the rest cover the demo corpus's subsystems (navigation, Bluetooth,
#: lifecycle, transport) and the generic CRUD/transform verbs that dominate
#: real queries.
CODE_SYNONYMS: Mapping[str, tuple[str, ...]] = {
    # --- Quoted verbatim from TechSpecifications.md section 3.3.3 -----------
    "preprocess": ("normalize", "sanitize", "transform", "clean"),
    "validate": ("verify", "check", "assert"),
    "auth": ("authenticate", "authorize", "token"),
    "fetch": ("request", "load", "retrieve"),
    "dispatch": ("route", "invoke", "handle"),
    "parse": ("decode", "deserialize", "extract"),
    "error": ("exception", "fail", "reject"),
    # --- Transform / data-shaping -----------------------------------------
    "normalize": ("normalise", "canonicalize", "standardize"),
    "sanitize": ("escape", "strip", "clean"),
    "transform": ("map", "convert", "translate"),
    "serialize": ("stringify", "encode", "marshal"),
    "deserialize": ("parse", "decode", "unmarshal"),
    "encode": ("serialize", "stringify", "escape"),
    "decode": ("parse", "deserialize", "unescape"),
    "format": ("template", "stringify", "render"),
    "filter": ("select", "match", "predicate"),
    "sort": ("order", "rank", "compare"),
    "merge": ("combine", "concat", "assign"),
    "split": ("chunk", "partition", "slice"),
    # --- Lifecycle ---------------------------------------------------------
    "init": ("initialize", "setup", "bootstrap"),
    "initialize": ("init", "setup", "bootstrap"),
    "start": ("launch", "run", "boot"),
    "stop": ("shutdown", "teardown", "dispose"),
    "main": ("entrypoint", "bootstrap", "run"),
    "config": ("settings", "options", "env"),
    "settings": ("preferences", "config", "options"),
    # --- Control flow / eventing ------------------------------------------
    "handle": ("process", "manage", "dispatch"),
    "handler": ("callback", "listener", "hook"),
    "listen": ("subscribe", "on", "observe"),
    "emit": ("publish", "trigger", "fire"),
    "register": ("subscribe", "attach", "add"),
    "resolve": ("lookup", "find", "locate"),
    "retry": ("backoff", "reattempt", "resend"),
    "queue": ("buffer", "backlog", "pending"),
    "schedule": ("timer", "interval", "defer"),
    "cache": ("memoize", "store", "lru"),
    "log": ("logger", "trace", "debug"),
    # --- Transport / IO ----------------------------------------------------
    "request": ("fetch", "call", "http"),
    "response": ("reply", "result", "payload"),
    "send": ("post", "emit", "publish"),
    "receive": ("consume", "subscribe", "read"),
    "connect": ("open", "bind", "attach"),
    "disconnect": ("close", "teardown", "release"),
    "upload": ("put", "post", "multipart"),
    "download": ("get", "fetch", "stream"),
    # --- CRUD --------------------------------------------------------------
    "create": ("make", "build", "construct"),
    "update": ("patch", "modify", "mutate"),
    "delete": ("remove", "destroy", "drop"),
    "read": ("get", "load", "fetch"),
    # --- Security ----------------------------------------------------------
    "login": ("signin", "authenticate", "session"),
    "logout": ("signout", "revoke", "session"),
    "token": ("jwt", "bearer", "credential"),
    "encrypt": ("cipher", "sign", "seal"),
    "decrypt": ("decipher", "verify", "unseal"),
    "permission": ("grant", "scope", "acl"),
    # --- Demo-corpus domain (mobile / navigation / Bluetooth) --------------
    "deeplink": ("uri", "scheme", "intent"),
    "link": ("url", "href", "uri"),
    "navigate": ("route", "redirect", "push"),
    "screen": ("page", "view", "activity"),
    "bluetooth": ("ble", "gatt", "pairing"),
    "pair": ("bond", "connect", "discover"),
    "scan": ("discover", "probe", "enumerate"),
    "notification": ("alert", "toast", "push"),
    "permissioncheck": ("grant", "scope", "acl"),
    # --- Testing -----------------------------------------------------------
    "test": ("spec", "fixture", "mock"),
    "mock": ("stub", "fake", "spy"),
    # --- Nominalisations the suffix table cannot reach ----------------------
    # "validation" -> strip "tion" -> "valida", which is not a key. Rather than
    # widen the stemmer (and start matching "tokenizer" to "token"), the handful
    # of -ation nouns that actually appear in queries are listed outright.
    "configuration": ("settings", "options", "env"),
    "authentication": ("authenticate", "authorize", "token"),
    "authorization": ("authorize", "permission", "scope"),
    "initialization": ("init", "setup", "bootstrap"),
    "validation": ("verify", "check", "assert"),
    "normalization": ("normalise", "canonicalize", "standardize"),
    "serialization": ("stringify", "encode", "marshal"),
    "navigation": ("route", "redirect", "push"),
    "connection": ("socket", "session", "channel"),
}

#: Suffixes stripped when looking a query word up in :data:`CODE_SYNONYMS`.
#:
#: "preprocessed" and "preprocessing" must both reach the ``preprocess`` row, and
#: a real stemmer (Porter, Snowball) is both a dependency and a source of
#: surprises on identifier-shaped input. Longest-first so "ation" wins over "ion"
#: and the result is independent of dict iteration order (NFR-08).
_SUFFIXES: tuple[str, ...] = (
    "ization",
    "isation",
    "ation",
    "tion",
    "ment",
    "ing",
    "ers",
    "ed",
    "es",
    "er",
    "s",
)

#: Shortest stem a suffix strip may leave behind. Guards against "es" turning
#: "res" into "r" and matching nothing useful while costing a dict probe.
_MIN_STEM = 3


def canonical_key(token: str) -> str | None:
    """Map one query word onto its :data:`CODE_SYNONYMS` key, if it has one.

    Two probes in a fixed order, so the same token always resolves the same way:
    exact match, then a longest-suffix strip. Deliberately *not* a prefix match
    -- ``token``/``tokenizer`` and ``log``/``logic`` share prefixes but not
    meaning, and a wrong expansion is worse than none because it is appended to
    the BM25 query where it competes for exact-match precision.

    Returns ``None`` when the token is not in the map -- an ordinary outcome, not
    a failure, since most words in a query are not expandable.
    """
    word = token.strip().lower()
    if not word:
        return None
    if word in CODE_SYNONYMS:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            stem = word[: -len(suffix)]
            if stem in CODE_SYNONYMS:
                return stem
    return None


def synonyms_for(token: str) -> tuple[str, ...]:
    """Expansion terms for a single query word; empty tuple when unmapped."""
    key = canonical_key(token)
    return CODE_SYNONYMS[key] if key is not None else ()


def expand_tokens(tokens: Iterable[str]) -> list[str]:
    """Expand a token stream, deduped, in first-occurrence order.

    Order is first-occurrence rather than sorted so that the terms most likely to
    matter (the ones from the head of the query) survive the caller's cap on list
    length, and so two runs over the same query always produce the same list.
    """
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for term in synonyms_for(token):
            if term not in seen:
                seen.add(term)
                out.append(term)
    return out
