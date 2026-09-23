"""Structured logging.

Logs are the audit trail behind every performance and degradation claim: which
profile ran, which model actually loaded, which rung of a degradation ladder was
taken. Rule 3 requires a degrade to be *loud* -- silent fallbacks are forbidden.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_CONFIGURED = False
_LOGGER_NAME = "axiom"


class _JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "axiom_extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


#: Level applied when nobody has asked for one. Named so the "did the caller
#: choose this, or is it just the default?" distinction is expressible.
DEFAULT_LOG_LEVEL = "INFO"


def configure_logging(level: str | None = None, json_output: bool = False) -> None:
    """Install Axiom's handler on the root ``axiom`` logger. Idempotent.

    ``level=None`` means "do not change the level", which is what
    :func:`get_logger` passes. That distinction is load-bearing: Axiom imports
    most of itself lazily *during* the first query, so every one of those
    imports calls :func:`get_logger`, and when this function unconditionally set
    the level those calls silently reset it back to ``INFO`` mid-run --
    ``axiom --log-level ERROR`` still printed twenty INFO lines. The level is
    now only written when a caller names one, or once to seed the default.
    """
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if level is not None:
        logger.setLevel(level.upper())
    elif not _CONFIGURED:
        logger.setLevel(DEFAULT_LOG_LEVEL)
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s  %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(module: str) -> logging.Logger:
    """Return the logger for a module, e.g. ``get_logger("retrieval.dense")``."""
    configure_logging()
    return logging.getLogger(f"{_LOGGER_NAME}.{module}")


def log_degradation(logger: logging.Logger, component: str, reason: str, fell_back_to: str) -> None:
    """Record that a degradation ladder rung was taken.

    Always WARNING: a degrade that nobody notices is indistinguishable from a
    silent wrong answer, and NFR-07 requires the log to state which fallback ran.
    """
    logger.warning(
        "degraded %s: %s -> %s",
        component,
        reason,
        fell_back_to,
        extra={
            "axiom_extra": {
                "event": "degradation",
                "component": component,
                "reason": reason,
                "fell_back_to": fell_back_to,
            }
        },
    )


#: Marker written by :func:`log_degradation` and read back by
#: :func:`capture_degradations`.
DEGRADATION_EVENT = "degradation"


#: Distinct degradation entries a single capture will report before summarising.
#:
#: Some ladders degrade *per file* -- the chunker names its component
#: ``chunker:<path>``, so a tree-sitter-less build of a 10k-file repository
#: produces 10k distinct strings. That belongs in the log, not in an API
#: response body, so the capture keeps the first few dozen and counts the rest.
#: Not a tunable: it affects no score and no ranking, only report size.
MAX_CAPTURED_DEGRADATIONS = 40


class _DegradationCollector(logging.Handler):
    """Handler that keeps :func:`log_degradation` records and prints nothing."""

    def __init__(self, sink: list[str], limit: int) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink
        self._limit = limit
        self._seen: set[str] = set()
        #: Distinct entries dropped because the cap was already reached.
        self.overflow = 0

    def emit(self, record: logging.LogRecord) -> None:
        extra = getattr(record, "axiom_extra", None)
        if not isinstance(extra, dict) or extra.get("event") != DEGRADATION_EVENT:
            return
        entry = f"{extra.get('component')}: {extra.get('reason')} -> {extra.get('fell_back_to')}"
        if entry in self._seen:
            return
        self._seen.add(entry)
        if len(self._sink) >= self._limit:
            self.overflow += 1
            return
        self._sink.append(entry)


@contextmanager
def capture_degradations() -> Iterator[list[str]]:
    """Collect every degradation logged anywhere under ``axiom`` while this is open.

    NFR-07 asks the system to *report* which fallback was active, not merely to
    log it. Each ladder lives in the module that owns it -- the chunker's
    tree-sitter rung, the embedder's three model rungs, dense's faiss rung,
    sparse's bm25s rung -- so an orchestrator has no way to enumerate them by
    asking. It can only listen, which is what this does: one temporary handler
    on the root ``axiom`` logger, filtered to the structured marker that
    :func:`log_degradation` writes.

    The logger level is floored at ``WARNING`` for the duration, because a run
    at ``--log-level ERROR`` would otherwise drop the records before any handler
    sees them and report "no degradations" on a fully degraded build. Console
    output is unchanged: the existing stream handler is raised to the old level
    for exactly as long as the floor is in place, so what the user sees is
    identical either way.

    Yields:
        A list of ``"component: reason -> fell_back_to"`` strings, deduplicated,
        in the order the rungs were taken, capped at
        :data:`MAX_CAPTURED_DEGRADATIONS` with a trailing count of what was
        elided. It fills as the block runs.
    """
    collected: list[str] = []
    logger = logging.getLogger(_LOGGER_NAME)
    handler = _DegradationCollector(collected, MAX_CAPTURED_DEGRADATIONS)

    previous_level = logger.level
    floored = previous_level > logging.WARNING
    muted: list[tuple[logging.Handler, int]] = []
    if floored:
        for existing in logger.handlers:
            muted.append((existing, existing.level))
            existing.setLevel(previous_level)
        logger.setLevel(logging.WARNING)

    logger.addHandler(handler)
    try:
        yield collected
    finally:
        logger.removeHandler(handler)
        if handler.overflow:
            collected.append(f"... and {handler.overflow} more degradation(s); see the log")
        if floored:
            logger.setLevel(previous_level)
            for existing, level in muted:
                existing.setLevel(level)
