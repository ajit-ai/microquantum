"""Statistical utilities for repeated measurements (MQ-07).

Small, dependency-light helpers (pure Python + NumPy scalar coercion) with a
clear **population vs sample** distinction where meaningful:

* :func:`mean` — arithmetic mean.
* :func:`variance` — ``ddof=0`` population variance, ``ddof=1`` sample
  variance.
* :func:`standard_deviation` — square root of :func:`variance`.
* :func:`standard_error` — *sample* standard error of the mean
  (``sample std / sqrt(n)``).
* :func:`confidence_interval` — normal-approximation confidence interval.
* :func:`minimum` / :func:`maximum` / :func:`count`.

These utilities accept any iterable of real numbers (lists, tuples, NumPy
arrays).  Non-numeric or non-finite inputs raise :class:`ValueError`.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def _coerce(values: Any, name: str = "values") -> list[float]:
    """Coerce an iterable to a finite-float list (rejecting bad input)."""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ValueError(f"{name} must be an iterable of real numbers")
    out: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, np.generic):
            value = value.item()
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name}[{index}] is not a real number: {value!r}"
            ) from exc
        if isinstance(value, complex):
            raise ValueError(f"{name}[{index}] must be real, got complex {value!r}")
        if not math.isfinite(result):
            raise ValueError(f"{name}[{index}] is not finite: {value!r}")
        out.append(result)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def count(values: Any) -> int:
    """Number of (valid) samples."""
    return len(_coerce(values))


def mean(values: Any) -> float:
    """Arithmetic mean of the values."""
    data = _coerce(values)
    return sum(data) / len(data)


def variance(values: Any, *, ddof: int = 0) -> float:
    """Variance of the values.

    Args:
        values: Iterable of real numbers.
        ddof: Delta degrees of freedom.  ``0`` (default) gives the
            **population variance**; ``1`` gives the **sample variance**
            (unbiased estimator).

    Raises:
        ValueError: If ``ddof`` is not 0/1, or not strictly below the sample
            size.
    """
    data = _coerce(values)
    if ddof not in (0, 1):
        raise ValueError("variance ddof must be 0 (population) or 1 (sample)")
    if len(data) <= ddof:
        raise ValueError(
            f"cannot compute variance with {ddof} degrees of freedom from "
            f"{len(data)} sample(s)"
        )
    center = mean(data)
    total = sum((x - center) ** 2 for x in data)
    return total / (len(data) - ddof)


def standard_deviation(values: Any, *, ddof: int = 0) -> float:
    """Standard deviation (population unless ``ddof=1`` for sample)."""
    return math.sqrt(variance(values, ddof=ddof))


def standard_error(values: Any) -> float:
    """Standard error of the mean using the *sample* standard deviation.

    ``se = sd_sample / sqrt(n)``.  This is the standard error of the mean for
    repeated independent measurements.
    """
    data = _coerce(values)
    if len(data) < 2:
        raise ValueError("standard_error requires at least 2 samples")
    return standard_deviation(data, ddof=1) / math.sqrt(len(data))


#: Normal quantiles for the most common confidence levels.
_Z_SCORES = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


def confidence_interval(
    values: Any,
    confidence: float = 0.95,
    *,
    z: Optional[float] = None,
) -> tuple[float, float]:
    """Normal-approximation confidence interval for the mean.

    Args:
        values: Iterable of real numbers.
        confidence: Confidence level in ``(0, 1)``; supported preset levels
            are 0.90, 0.95 and 0.99 for the default ``z``.
        z: Explicit normal quantile (overrides ``confidence``).  Pass this
            when a non-preset confidence level is needed.

    Returns:
        ``(lower, upper)`` bounds of the interval.

    Raises:
        ValueError: If the confidence preset is unsupported and ``z`` is not
            provided, or fewer than 2 samples are available.
    """
    data = _coerce(values)
    if len(data) < 2:
        raise ValueError("confidence_interval requires at least 2 samples")
    if z is None:
        if confidence not in _Z_SCORES:
            raise ValueError(
                f"no preset z-score for confidence {confidence}; "
                f"supported presets are {sorted(_Z_SCORES)} — pass z= explicitly"
            )
        z = _Z_SCORES[confidence]
    if not isinstance(z, (int, float)) or math.isnan(float(z)):
        raise ValueError(f"invalid z-score: {z!r}")
    width = float(z) * standard_error(data)
    center = mean(data)
    return center - width, center + width


def minimum(values: Any) -> float:
    """Minimum of the values."""
    return min(_coerce(values))


def maximum(values: Any) -> float:
    """Maximum of the values."""
    return max(_coerce(values))


__all__ = [
    "mean",
    "variance",
    "standard_deviation",
    "standard_error",
    "confidence_interval",
    "minimum",
    "maximum",
    "count",
]