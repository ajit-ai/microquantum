"""Result aggregation across repeated executions (MQ-07).

:class:`ResultAggregator` groups execution results by parameter bindings,
backend, status, metadata keys or any user-defined accessor — while **always
preserving the raw results**.  Groups reference the original
:class:`ExecutionRecord` / result objects; nothing is replaced by a summary.

Raw results may come from:

* :class:`~microquantum.experiments.experiment.ExperimentResult` (its records),
* sequences of :class:`~microquantum.experiments.record.ExecutionRecord` or
  :class:`~microquantum.backends.base.BackendResult` objects.

The flow is ``Raw Results -> Aggregation -> Derived Analysis`` — never
``Raw Results -> replace with summary``.
"""

from __future__ import annotations

import enum as _enum
from typing import Any, Callable, Dict, Union

import numpy as np

from .._json import JSONSerializable, json_safe, json_string
from . import statistics as _stats
from .expectation import _item_expectations

Accessor = Union[str, Callable[[Any], Any]]


def _as_records(results: Any) -> list[Any]:
    """Normalize results to a list of record-like items."""
    if results is None:
        raise TypeError("ResultAggregator requires a results source")
    if hasattr(results, "records"):  # ExperimentResult
        items = list(results.records)
    elif isinstance(results, (list, tuple)):
        items = list(results)
    else:
        items = [results]
    if not items:
        raise ValueError("ResultAggregator requires at least one result")
    return items


def _resolve(obj: Any, key: str) -> Any:
    """Attribute or mapping lookup."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _group_key(value: Any) -> Any:
    """Coerce a group value to a stable, JSON-safe key."""
    if isinstance(value, _enum.Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return None
    return str(value)


class ResultAggregator(JSONSerializable):
    """Group results and produce derived summaries without losing raw data.

    Args:
        results: An :class:`ExperimentResult`, or a sequence of records /
            results to group over.
    """

    def __init__(self, results: Any) -> None:
        self._items = _as_records(results)

    @property
    def records(self) -> list[Any]:
        """The raw results (read-only view, original objects preserved)."""
        return list(self._items)

    @property
    def record_count(self) -> int:
        """Number of raw results aggregated over."""
        return len(self._items)

    # ------------------------------------------------------------------
    # grouping
    # ------------------------------------------------------------------

    def group_by(self, accessor: Accessor) -> Dict[Any, list[Any]]:
        """Group the raw results by an accessor.

        Args:
            accessor: A callable ``record -> value``, or a dotted path string
                resolved against each record: an attribute (``backend``,
                ``status``, ``plan_name``, ``shots``, ``seed``) or a nested
                field (``parameter_bindings.<name>``, ``metadata.<key>``,
                ``timing.<key>``).

        Returns:
            ``{group_key: [records...]}`` preserving the original record
            objects; group keys appear in first-seen order.
        """
        groups: Dict[Any, list[Any]] = {}
        for item in self._items:
            value = self._value_of(item, accessor)
            key = _group_key(value)
            groups.setdefault(key, []).append(item)
        return groups

    def _value_of(self, item: Any, accessor: Accessor) -> Any:
        if callable(accessor):
            return accessor(item)
        if isinstance(accessor, str):
            if not accessor:
                raise ValueError("group accessor must not be empty")
            current: Any = item
            for segment in accessor.split("."):
                if current is None:
                    return None
                current = _resolve(current, segment)
            return current
        raise TypeError(
            "group accessor must be a dotted-path string or a callable, "
            f"got {type(accessor).__name__}"
        )

    # -- convenience groupings ------------------------------------------------

    def group_by_parameter(self, parameter: str) -> Dict[Any, list[Any]]:
        """Group by the binding value of ``parameter``."""
        return self.group_by(f"parameter_bindings.{parameter}")

    def group_by_backend(self) -> Dict[Any, list[Any]]:
        """Group by backend name."""
        return self.group_by("backend")

    def group_by_status(self) -> Dict[Any, list[Any]]:
        """Group by execution status (``completed`` / ``failed``)."""
        return self.group_by("status")

    def group_counts(self, groups: Dict[Any, list[Any]]) -> Dict[Any, int]:
        """Map each group key to its number of raw results."""
        return {key: len(members) for key, members in groups.items()}

    # ------------------------------------------------------------------
    # derived summaries (never mutate raw results)
    # ------------------------------------------------------------------

    def mean_expectation(
        self, groups: Dict[Any, list[Any]], expectations_key: str
    ) -> Dict[Any, float]:
        """Mean of an expectation label per group.

        Results without the label are skipped per group (a group with no
        matching expectations is omitted from the output).  Raw records are
        untouched.
        """
        summary: Dict[Any, float] = {}
        for key, members in groups.items():
            values: list[float] = []
            for member in members:
                mapping = _item_expectations(member)
                if mapping is None or expectations_key not in mapping:
                    continue
                values.append(float(mapping[expectations_key]))
            if not values:
                continue
            summary[key] = _stats.mean(values)
        return summary

    def parameter_expectations(
        self, parameter: str, expectations_key: str
    ) -> Dict[Any, float]:
        """``{parameter_value: mean_expectation}`` across all raw results.

        Shorthand for ``mean_expectation(group_by_parameter(parameter), key)``.
        """
        return self.mean_expectation(
            self.group_by_parameter(parameter), expectations_key
        )

    def expectation_keys(self) -> tuple[str, ...]:
        """Labels present across the aggregated results (sorted)."""
        known: set[str] = set()
        for item in self._items:
            mapping = _item_expectations(item)
            if mapping:
                known.update(mapping)
        return tuple(sorted(known))

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self, accessor: Accessor = "backend") -> dict[str, Any]:
        """Serialize groupings/derived summaries (raw results excluded).

        Raw results are not embedsable without loss across an SDKS boundary —
        they remain available via :attr:`records`; the serialized form carries
        group sizes and per-group expectation means.
        """
        groups = self.group_by(accessor)
        payload: Dict[Any, Any] = {}
        for key, members in groups.items():
            entry: Dict[str, Any] = {"count": len(members)}
            for label in self.expectation_keys():
                values: list[float] = []
                for member in members:
                    mapping = _item_expectations(member)
                    if mapping is None or label not in mapping:
                        continue
                    values.append(float(mapping[label]))
                if values:
                    entry[label] = {
                        "mean": _stats.mean(values),
                        "standard_error": (
                            _stats.standard_error(values)
                            if len(values) >= 2
                            else None
                        ),
                    }
            payload[str(key)] = entry
        return {
            "type": "aggregation",
            "accessor": accessor if isinstance(accessor, str) else "custom",
            "record_count": self.record_count,
            "groups": json_safe(payload),
        }

    def to_json(self, accessor: Accessor = "backend") -> str:
        """Serialize the aggregation snapshot to a JSON string."""
        return json_string(self.to_dict(accessor=accessor))

    def __repr__(self) -> str:
        return f"ResultAggregator(records={self.record_count})"


__all__ = ["ResultAggregator"]