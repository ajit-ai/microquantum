"""Inferential statistics for experiment analysis.

:class:`HypothesisTest` compares two outcome distributions with a
chi-square statistic and a permutation p-value (exact, seedable, no
special functions required); :func:`bootstrap_ci` builds percentile
confidence intervals for any statistic via NumPy resampling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = [
    "TestOutcome",
    "HypothesisTest",
    "float_mean",
    "bootstrap_ci",
]


@dataclass(frozen=True)
class TestOutcome:
    """Result of a two-sample comparison."""

    statistic: float
    p_value: float
    significant: bool
    degrees_of_freedom: int
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "degrees_of_freedom": self.degrees_of_freedom,
            "metadata": dict(self.metadata or {}),
        }


def _chi_square(counts_a: Mapping[str, int], counts_b: Mapping[str, int]) -> tuple[float, int]:
    """Chi-square homogeneity statistic and degrees of freedom."""
    keys = sorted(set(counts_a) | set(counts_b))
    if len(keys) < 2:
        raise ValueError("Need at least two distinct outcomes to compare")
    table = np.array(
        [[float(counts_a.get(k, 0)), float(counts_b.get(k, 0))] for k in keys]
    )
    total = table.sum()
    if total == 0:
        raise ValueError("Both distributions are empty")
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / np.where(expected > 0, expected, 1.0), 0.0)
    return float(terms.sum()), (len(keys) - 1) * (table.shape[1] - 1)


class HypothesisTest:
    """Two-sample comparison with permutation p-values.

    Args:
        alpha: Significance level.
        permutations: Permutation resamples for the p-value.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, alpha: float = 0.05, permutations: int = 1000, seed: Optional[int] = None) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if permutations < 1:
            raise ValueError("permutations must be >= 1")
        self._alpha = alpha
        self._permutations = permutations
        self._seed = seed

    @property
    def alpha(self) -> float:
        """Significance level."""
        return self._alpha

    def compare(self, counts_a: Mapping[str, int], counts_b: Mapping[str, int]) -> TestOutcome:
        """Compare two outcome distributions."""
        statistic, dof = _chi_square(counts_a, counts_b)
        keys = sorted(set(counts_a) | set(counts_b))
        pooled = [key for key in keys for _ in range(int(counts_a.get(key, 0) + counts_b.get(key, 0)))]
        size_a = sum(int(v) for v in counts_a.values())
        rng = np.random.default_rng(self._seed)
        extreme = 0
        for _ in range(self._permutations):
            rng.shuffle(pooled)
            perm_a: dict[str, int] = {}
            perm_b: dict[str, int] = {}
            for position, key in enumerate(pooled):
                target = perm_a if position < size_a else perm_b
                target[key] = target.get(key, 0) + 1
            perm_stat, _ = _chi_square(perm_a, perm_b)
            if perm_stat >= statistic:
                extreme += 1
        p_value = (extreme + 1) / (self._permutations + 1)
        return TestOutcome(
            statistic=statistic,
            p_value=float(p_value),
            significant=bool(p_value < self._alpha),
            degrees_of_freedom=dof,
            metadata={"permutations": self._permutations},
        )


def float_mean(values: Sequence[float]) -> float:
    """Mean of a sequence (default bootstrap statistic)."""
    items = [float(v) for v in values]
    if not items:
        raise ValueError("values must be non-empty")
    return float(sum(items) / len(items))


def bootstrap_ci(
    samples: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = float_mean,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for *statistic*."""
    data = [float(v) for v in samples]
    if not data:
        raise ValueError("samples must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    rng = np.random.default_rng(seed)
    estimates = [
        float(statistic(list(rng.choice(data, size=len(data), replace=True))))
        for _ in range(resamples)
    ]
    lower_pct = (1.0 - confidence) / 2.0 * 100.0
    upper_pct = (1.0 + confidence) / 2.0 * 100.0
    return float(np.percentile(estimates, lower_pct)), float(np.percentile(estimates, upper_pct))
