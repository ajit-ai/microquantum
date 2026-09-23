"""Adaptive parameter sweeps with grid refinement.

:class:`AdaptiveSweep` wraps a :class:`ParameterSweep` grid and narrows
it around the best observed point: each refinement round rebuilds a
small local grid from axis neighbors (or a fractional span), so coarse
exploration followed by focused exploitation needs no extra
infrastructure.
"""

from __future__ import annotations

from typing import Any, Sequence

from .sweep import ParameterSweep

__all__ = [
    "AdaptiveSweep",
]


class AdaptiveSweep:
    """Refining sweep built on :class:`ParameterSweep` grids.

    Args:
        sweep: Base grid defining parameter axes.
        refine_fraction: Span fraction for refined axes (0 exclusive,
            1 exclusive).
        max_rounds: Maximum refinement rounds via :meth:`refine`.
    """

    def __init__(
        self,
        sweep: ParameterSweep,
        refine_fraction: float = 0.25,
        max_rounds: int = 2,
    ) -> None:
        if not 0.0 < refine_fraction < 1.0:
            raise ValueError("refine_fraction must be in (0, 1)")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self._sweep = sweep
        self._refine_fraction = refine_fraction
        self._max_rounds = max_rounds
        self._rounds_completed = 0

    @property
    def sweep(self) -> ParameterSweep:
        """Current grid."""
        return self._sweep

    @property
    def rounds_completed(self) -> int:
        """Refinement rounds performed so far."""
        return self._rounds_completed

    @property
    def max_rounds(self) -> int:
        """Maximum refinement rounds."""
        return self._max_rounds

    def _axes(self) -> dict[str, list[float]]:
        """Current axes reconstructed from the sweep payload."""
        payload = self._sweep.to_dict()
        values = payload.get("values", {})
        if not isinstance(values, dict) or not values:
            raise ValueError("Base sweep carries no parameter axes")
        return {str(name): [float(v) for v in axis] for name, axis in values.items()}

    def refine(
        self,
        results: Sequence[tuple[dict[str, float], float]],
        *,
        higher_is_better: bool = False,
    ) -> ParameterSweep:
        """Build a narrowed grid around the best of *results*.

        Args:
            results: ``(bindings, value)`` observations from the current
                grid (bindings outside the grid are ignored for the
                best-point search but do not error).
            higher_is_better: Maximize instead of minimize.

        Raises:
            ValueError: If no results are given or rounds are exhausted.
        """
        if not results:
            raise ValueError("refine() requires at least one observation")
        if self._rounds_completed >= self._max_rounds:
            raise ValueError(f"max_rounds={self._max_rounds} already exhausted")
        if higher_is_better:
            best_bindings = max(results, key=lambda item: item[1])
        else:
            best_bindings = min(results, key=lambda item: item[1])
        best, _ = best_bindings
        axes = self._axes()
        refined: dict[str, list[float]] = {}
        for name, axis in axes.items():
            ordered = sorted(axis)
            center = float(best.get(name, ordered[len(ordered) // 2]))
            span = (max(ordered) - min(ordered)) * self._refine_fraction
            if span <= 0:
                refined[name] = [center]
                continue
            low, high = center - span / 2.0, center + span / 2.0
            refined[name] = [low, center, high]
        self._rounds_completed += 1
        self._sweep = ParameterSweep(
            refined, name=f"{self._sweep_name()}-r{self._rounds_completed}"
        )
        return self._sweep

    def _sweep_name(self) -> str:
        """Base sweep name for refined generations."""
        name = getattr(self._sweep, "name", "sweep")
        return str(name).rsplit("-r", 1)[0] if "-r" in str(name) else str(name)

    def combinations(self) -> list[dict[str, float]]:
        """Current grid combinations."""
        return self._sweep.combinations()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the adaptive state."""
        return {
            "sweep": self._sweep.to_dict(),
            "refine_fraction": self._refine_fraction,
            "max_rounds": self._max_rounds,
            "rounds_completed": self._rounds_completed,
        }
