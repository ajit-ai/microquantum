"""Expectation-value analysis (MQ-07).

:class:`ExpectationAnalysis` aggregates labeled expectation values produced by
backends — exactly the ``BackendResult.expectations`` representation (a
``{label: float}`` mapping) — across repeated executions: mean, variance,
standard deviation and standard error per label, plus a
parameter-to-expectation mapping for sweeps.

The analyzer works with any collection of:

* :class:`~microquantum.backends.base.BackendResult` objects,
* :class:`~microquantum.experiments.record.ExecutionRecord` objects,
* :class:`~microquantum.experiments.experiment.ExperimentResult` objects,
* serialized ``to_dict()`` dicts carrying an ``expectations`` mapping.

No new expectation format is invented — the existing
``BackendResult.expectations`` contract is consumed directly.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from .._json import JSONSerializable, json_safe, json_string
from . import statistics as _stats


def _item_expectations(item: Any) -> Optional[dict[str, float]]:
    """Extract an ``{label: float}`` mapping from a supported item."""
    if isinstance(item, dict):
        if "expectations" in item and isinstance(item["expectations"], dict):
            return dict(item["expectations"])
        return None
    mapping = getattr(item, "expectations", None)
    if isinstance(mapping, dict):
        return dict(mapping)
    return None


def _coerce_items(source: Any) -> Sequence[Any]:
    """Normalize a source into a sequence of expectation-carrying items."""
    if source is None:
        raise TypeError("ExpectationAnalysis requires a result source")
    if hasattr(source, "records"):  # ExperimentResult
        return list(source.records)
    if isinstance(source, dict):
        return [source]
    if hasattr(source, "expectations") or hasattr(source, "records"):
        return [source]
    if isinstance(source, (list, tuple)):
        items = list(source)
        if not items:
            raise ValueError("ExpectationAnalysis requires at least one result")
        return items
    raise TypeError(
        "ExpectationAnalysis needs a result source or a sequence of them; "
        f"got {type(source).__name__}"
    )


class ExpectationAnalysis(JSONSerializable):
    """Aggregation/statistics over labeled expectation-value results.

    Args:
        source: A single result (BackendResult / ExecutionRecord /
            ExperimentResult / serialized dict) or a sequence of them.
    """

    def __init__(self, source: Any) -> None:
        self._items: list[Any] = list(_coerce_items(source))
        self._expectations: list[dict[str, float]] = []
        for item in self._items:
            mapping = _item_expectations(item)
            if mapping is not None:
                self._expectations.append(mapping)

    # -- introspection --------------------------------------------------------

    @property
    def result_count(self) -> int:
        """Number of expectation-carrying results consumed."""
        return len(self._expectations)

    @property
    def keys(self) -> tuple[str, ...]:
        """Expectation labels present across all results (sorted)."""
        known: set[str] = set()
        for mapping in self._expectations:
            known.update(mapping)
        return tuple(sorted(known))

    def has(self, key: str) -> bool:
        """True if any result carries the given expectation label."""
        return any(key in mapping for mapping in self._expectations)

    def values(self, key: str) -> list[float]:
        """Raw expectation values for a label (over the results carrying it)."""
        values = [
            float(mapping[key])
            for mapping in self._expectations
            if key in mapping and mapping[key] is not None
        ]
        if not values:
            raise KeyError(
                f"no expectation values for label {key!r}; "
                f"known labels: {list(self.keys)}"
            )
        return values

    def count(self, key: str) -> int:
        """Number of results that produced the label (row count)."""
        return sum(1 for mapping in self._expectations if key in mapping)

    # -- per-label statistics -------------------------------------------------

    def mean(self, key: str) -> float:
        """Mean expectation value for a label."""
        return _stats.mean(self.values(key))

    def variance(self, key: str, *, ddof: int = 0) -> float:
        """Variance of a label's expectations (``ddof=0`` population)."""
        return _stats.variance(self.values(key), ddof=ddof)

    def standard_deviation(self, key: str, *, ddof: int = 0) -> float:
        """Standard deviation of a label's expectations."""
        return _stats.standard_deviation(self.values(key), ddof=ddof)

    def standard_error(self, key: str) -> float:
        """Standard error of the mean expectation for a label.

        Needs at least 2 repeated results carrying the label.
        """
        return _stats.standard_error(self.values(key))

    def minimum(self, key: str) -> float:
        """Minimum expectation value for a label."""
        return _stats.minimum(self.values(key))

    def maximum(self, key: str) -> float:
        """Maximum expectation value for a label."""
        return _stats.maximum(self.values(key))

    # -- parameter mapping ----------------------------------------------------

    def parameter_points(
        self, parameter: str, key: str, *, include_error: bool = False
    ) -> list[Tuple[float, float, Optional[float]]]:
        """Ordered ``(parameter_value, mean_expectation[, std_error])`` points.

        Groups results by their binding of ``parameter``, averaging each
        group's expectation for ``key``.  Points are ordered by ascending
        parameter value (deterministic).
        """
        groups: dict[float, list[float]] = {}
        for item in self._items:
            bindings = getattr(item, "parameter_bindings", None)
            if not bindings or parameter not in bindings:
                continue
            mapping = _item_expectations(item)
            if mapping is None or key not in mapping:
                continue
            try:
                value = float(bindings[parameter])
            except (TypeError, ValueError):
                continue
            groups.setdefault(value, []).append(float(mapping[key]))
        if not groups:
            raise KeyError(
                f"no results bound parameter {parameter!r} with label {key!r}"
            )
        points: list[Tuple[float, float, Optional[float]]] = []
        for value in sorted(groups):
            group = groups[value]
            center = _stats.mean(group)
            error = (
                _stats.standard_error(group)
                if include_error and len(group) >= 2
                else None
            )
            points.append((value, center, error))
        return points

    def parameter_to_expectation(
        self, parameter: str, key: str
    ) -> dict[float, float]:
        """``{parameter_value: mean_expectation}`` for a label.

        Convenience over :meth:`parameter_points` (identical grouping rules).
        """
        return {
            value: center
            for value, center, _ in self.parameter_points(parameter, key)
        }

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize per-label summaries to a JSON-safe dictionary."""
        summaries: dict[str, Any] = {}
        for key in self.keys:
            values = self.values(key)
            try:
                error = _stats.standard_error(values)
            except ValueError:
                error = None
            summaries[key] = {
                "mean": _stats.mean(values),
                "variance": _stats.variance(values),
                "standard_deviation": _stats.standard_deviation(values),
                "standard_error": error,
                "minimum": _stats.minimum(values),
                "maximum": _stats.maximum(values),
                "count": self.count(key),
            }
        return {
            "type": "expectation",
            "result_count": self.result_count,
            "keys": list(self.keys),
            "summaries": json_safe(summaries),
        }

    def to_json(self) -> str:
        """Serialize the analysis snapshot to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"ExpectationAnalysis(results={self.result_count}, "
            f"labels={list(self.keys)})"
        )


__all__ = ["ExpectationAnalysis"]