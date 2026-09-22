"""Structured logging.

Logs are the audit trail behind every performance and degradation claim: which
profile ran, which model actually loaded, which rung of a degradation ladder was
taken. Rule 3 requires a degrade to be *loud* -- silent fallbacks are forbidden.
"""

from __future__ import annotations

import json
import logging
import sys
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


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Install Axiom's handler on the root ``axiom`` logger. Idempotent."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level.upper())
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


def log_degradation(
    logger: logging.Logger, component: str, reason: str, fell_back_to: str
) -> None:
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
