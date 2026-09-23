"""Clifford data regression (CDR) for error mitigation.

Learns an affine noise map ``exact ≈ slope * noisy + intercept`` from
classically simulable Clifford training circuits, then applies it to
mitigate observables measured on nearby non-Clifford circuits of
interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "CDRTrainingPoint",
    "CliffordDataRegression",
]


@dataclass(frozen=True)
class CDRTrainingPoint:
    """One Clifford training observation."""

    noisy: float
    exact: float


class CliffordDataRegression:
    """Affine CDR model fitted by least squares.

    Attributes:
        slope: Fitted slope (exact ≈ slope * noisy + intercept).
        intercept: Fitted intercept.
    """

    def __init__(self) -> None:
        self._slope: float | None = None
        self._intercept: float | None = None
        self._num_points = 0

    @property
    def is_trained(self) -> bool:
        """True after a successful :meth:`train`."""
        return self._slope is not None

    @property
    def slope(self) -> float:
        """Fitted slope (raises when untrained)."""
        if self._slope is None:
            raise RuntimeError("CliffordDataRegression is not trained; call train()")
        return self._slope

    @property
    def intercept(self) -> float:
        """Fitted intercept (raises when untrained)."""
        if self._intercept is None:
            raise RuntimeError("CliffordDataRegression is not trained; call train()")
        return self._intercept

    def train(
        self, noisy: Sequence[float], exact: Sequence[float]
    ) -> dict[str, Any]:
        """Fit the affine map on paired Clifford observations.

        Returns:
            Training summary with slope, intercept and R².
        """
        noisy_list = [float(v) for v in noisy]
        exact_list = [float(v) for v in exact]
        if len(noisy_list) != len(exact_list):
            raise ValueError("noisy and exact must have equal length")
        if len(noisy_list) < 2:
            raise ValueError("CDR training needs at least two points")
        if not all(np.isfinite(noisy_list)) or not all(np.isfinite(exact_list)):
            raise ValueError("Training values must be finite")
        design = np.column_stack([np.asarray(noisy_list), np.ones(len(noisy_list))])
        target = np.asarray(exact_list)
        (slope, intercept), _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        self._slope = float(slope)
        self._intercept = float(intercept)
        self._num_points = len(noisy_list)
        fitted = self._slope * np.asarray(noisy_list) + self._intercept
        ss_res = float(np.sum((target - fitted) ** 2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return {
            "slope": self._slope,
            "intercept": self._intercept,
            "r_squared": r_squared,
            "num_points": self._num_points,
        }

    def mitigate(self, noisy_value: float) -> float:
        """Apply the learned map to a noisy observable."""
        if not np.isfinite(noisy_value):
            raise ValueError("noisy_value must be finite")
        return self.slope * float(noisy_value) + self.intercept

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fitted model."""
        return {
            "slope": self._slope,
            "intercept": self._intercept,
            "num_points": self._num_points,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CliffordDataRegression:
        """Rebuild a fitted model from :meth:`to_dict` output."""
        model = cls()
        if data.get("slope") is not None:
            model._slope = float(data["slope"])
            model._intercept = float(data.get("intercept", 0.0))
            model._num_points = int(data.get("num_points", 0))
        return model

    def __repr__(self) -> str:
        if self.is_trained:
            return f"CliffordDataRegression(slope={self._slope:.4f}, intercept={self._intercept:.4f})"
        return "CliffordDataRegression(untrained)"
