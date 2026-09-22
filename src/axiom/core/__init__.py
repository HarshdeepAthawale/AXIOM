"""Cross-cutting primitives: hashing, logging, timing, errors."""

from __future__ import annotations

from axiom.core.errors import (
    AxiomContractError,
    AxiomError,
    DegradationExhaustedError,
    IndexNotFoundError,
)
from axiom.core.hashing import (
    blake2b_128,
    compute_chunk_id,
    compute_content_hash,
    compute_family_id,
    compute_file_hash,
    normalise_for_hash,
)
from axiom.core.logging import configure_logging, get_logger, log_degradation
from axiom.core.timing import Deadline, StageTiming, TimingLedger

__all__ = [
    "AxiomContractError",
    "AxiomError",
    "Deadline",
    "DegradationExhaustedError",
    "IndexNotFoundError",
    "StageTiming",
    "TimingLedger",
    "blake2b_128",
    "compute_chunk_id",
    "compute_content_hash",
    "compute_family_id",
    "compute_file_hash",
    "configure_logging",
    "get_logger",
    "log_degradation",
    "normalise_for_hash",
]
