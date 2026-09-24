"""Stage timing records.

NFR-10 requires every pipeline stage to emit a structured timing record, and
every ``--json`` response to carry a ``timings`` block, so that latency claims in
the submission are auditable rather than asserted.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageTiming:
    """One stage's wall-clock cost, in milliseconds."""

    stage: str
    elapsed_ms: float
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class TimingLedger:
    """Ordered collection of stage timings for a single query or index run."""

    stages: list[StageTiming] = field(default_factory=list)

    @contextmanager
    def measure(self, stage: str, **detail: object) -> Iterator[dict[str, object]]:
        """Time a block and append a :class:`StageTiming`.

        The yielded dict is merged into the record's ``detail``, so a stage can
        report what it actually did::

            with ledger.measure("dense") as d:
                hits = search(...)
                d["candidates"] = len(hits)
        """
        extra: dict[str, object] = {}
        start = time.perf_counter()
        try:
            yield extra
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            merged: dict[str, object] = {**detail, **extra}
            self.stages.append(StageTiming(stage=stage, elapsed_ms=elapsed_ms, detail=merged))

    @property
    def total_ms(self) -> float:
        """Sum of all recorded stages. Concurrent stages therefore over-count."""
        return sum(s.elapsed_ms for s in self.stages)

    def as_dict(self) -> dict[str, object]:
        """The full nested record: total, every stage occurrence, and its detail.

        This is the logging shape. The ``timings`` field of a ``--json`` body is
        typed ``dict[str, float]`` (API.md sections 3.1 and 8), so serialise with
        :meth:`as_flat_dict` for anything that goes on the wire.
        """
        return {
            "total_ms": round(self.total_ms, 3),
            "stages": [
                {"stage": s.stage, "elapsed_ms": round(s.elapsed_ms, 3), **s.detail}
                for s in self.stages
            ],
        }

    def as_flat_dict(self) -> dict[str, float]:
        """Collapse to ``stage -> total ms``, the documented wire shape.

        A stage that ran more than once -- the agent loop's second pass re-runs
        ``dense``, an incremental build re-runs ``chunk`` per file group -- is
        summed rather than last-write-wins, because the question the field
        answers is where the wall clock went, not which occurrence was last.
        """
        totals: dict[str, float] = {}
        for stage in self.stages:
            totals[stage.stage] = round(totals.get(stage.stage, 0.0) + stage.elapsed_ms, 3)
        return totals


class Deadline:
    """A monotonic wall-clock budget.

    The agent loop's 5 s cap (FR-13, NFR-04) is enforced by checking this before
    each pass, never by hoping a pass returns in time.
    """

    def __init__(self, budget_ms: float) -> None:
        self._budget_s = budget_ms / 1000.0
        self._start = time.monotonic()

    @property
    def expired(self) -> bool:
        return time.monotonic() - self._start >= self._budget_s

    @property
    def remaining_ms(self) -> float:
        left = self._budget_s - (time.monotonic() - self._start)
        return max(0.0, left * 1000.0)

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0
