"""Primal-integral tracking for minimization runs.

The trace records the incumbent (best-found) objective against a supplied optimal or
best-known reference.  If callers have no external reference, they may use a
run-local feasible incumbent but must label it accordingly in the returned metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class PrimalIntegralTracker:
    """Accumulate the normalized minimization primal integral over a run."""

    reference_objective: float
    reference_source: str
    start_time: float = field(default_factory=perf_counter)
    _best_found: float | None = field(default=None, init=False)
    _last_elapsed: float = field(default=0.0, init=False)
    _last_gap: float = field(default=0.0, init=False)
    _integral: float = field(default=0.0, init=False)
    _samples: list[dict[str, float]] = field(default_factory=list, init=False)

    def record(self, objective: float, elapsed_seconds: float | None = None) -> None:
        """Record an observed feasible objective and update the incumbent trace."""
        if objective < 0:
            raise ValueError("objective must be non-negative for normalized primal tracking")

        elapsed = perf_counter() - self.start_time if elapsed_seconds is None else elapsed_seconds
        if elapsed < self._last_elapsed:
            raise ValueError("elapsed_seconds must be monotonic")

        self._best_found = objective if self._best_found is None else min(self._best_found, objective)
        denominator = max(abs(self.reference_objective), 1e-12)
        relative_gap = max(0.0, (self._best_found - self.reference_objective) / denominator)
        self._integral += self._last_gap * (elapsed - self._last_elapsed)
        self._last_elapsed = elapsed
        self._last_gap = relative_gap
        self._samples.append(
            {
                "elapsed_seconds": float(elapsed),
                "objective": float(objective),
                "best_found_objective": float(self._best_found),
                "relative_gap": float(relative_gap),
                "primal_integral": float(self._integral),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata and samples suitable for plotting."""
        return {
            "sense": "minimization",
            "reference_objective": float(self.reference_objective),
            "reference_source": self.reference_source,
            "samples": self._samples,
            "final_primal_integral": float(self._integral),
        }
