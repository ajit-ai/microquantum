"""Parameter sweeps (MQ-07).

A :class:`ParameterSweep` describes a deterministic grid of parameter
combinations: one or many parameters, explicit values or generated numeric
ranges, combined via the Cartesian product in a stable order.

The sweep is pure *configuration* — it never executes anything.  It is meant
to be attached to an :class:`~microquantum.experiments.experiment.Experiment`
(or consumed directly via ``combinations()``) and validated against the
parameters of a base circuit/plan before execution.

Only pure Python / NumPy is used (no pandas or heavyweight dependencies).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping

import numpy as np

from .._json import JSONSerializable, json_safe, json_string

#: A single parameter definition: explicit values (list/tuple of numbers) or
#: a generated range specification (dict).
ParameterDefinition = Any

_RANGE_KEYS = {"start", "stop"}
_LINSPACE_KEYS = {"start", "stop", "num_points"}


def _as_scalar(value: Any, parameter: str) -> float:
    """Coerce a candidate value to a finite float (rejecting bad input)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str) or value is None:
        raise ValueError(
            f"sweep parameter '{parameter}' has invalid value {value!r}; "
            f"values must be numeric"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sweep parameter '{parameter}' has non-numeric value {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(
            f"sweep parameter '{parameter}' received a non-finite value {value!r}"
        )
    return result


def _expand_definition(parameter: str, definition: ParameterDefinition) -> list[float]:
    """Expand one parameter definition into an ordered, validated list of floats."""
    if isinstance(definition, Mapping):
        if "values" in definition:
            values = definition["values"]
            if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
                raise ValueError(
                    f"sweep parameter '{parameter}' 'values' must be an iterable "
                    f"of numbers"
                )
            return [_as_scalar(v, parameter) for v in list(values)]
        if "range" in definition:
            spec = definition["range"]
            try:
                start, stop, step = spec[0], spec[1], spec[2]
            except (IndexError, TypeError) as exc:
                raise ValueError(
                    f"sweep parameter '{parameter}' range must be "
                    f"(start, stop, step)"
                ) from exc
            start = _as_scalar(start, parameter)
            stop = _as_scalar(stop, parameter)
            step = _as_scalar(step, parameter)
            if step <= 0:
                raise ValueError(
                    f"sweep parameter '{parameter}' range step must be > 0, "
                    f"got {step}"
                )
            return [float(v) for v in np.arange(start, stop, step)]
        if _LINSPACE_KEYS.issubset(definition.keys()) or _RANGE_KEYS.issubset(
            definition.keys()
        ):
            start = _as_scalar(definition.get("start"), parameter)
            stop = _as_scalar(definition.get("stop"), parameter)
            num_points = definition.get("num_points")
            if num_points is not None:
                num_points = int(num_points)
                if num_points < 2:
                    raise ValueError(
                        f"sweep parameter '{parameter}' num_points must be >= 2, "
                        f"got {num_points}"
                    )
                return [float(v) for v in np.linspace(start, stop, num_points)]
            step = _as_scalar(definition.get("step", 0.1), parameter)
            if step <= 0:
                raise ValueError(
                    f"sweep parameter '{parameter}' step must be > 0, got {step}"
                )
            return [float(v) for v in np.arange(start, stop + step, step)]
        raise ValueError(
            f"sweep parameter '{parameter}' definition {definition!r} is not a "
            f"recognized values/range specification"
        )
    if isinstance(definition, (str, bytes)):
        raise ValueError(
            f"sweep parameter '{parameter}' must be a list of values or a range "
            f"specification, got {definition!r}"
        )
    try:
        values = list(definition)
    except TypeError as exc:
        raise ValueError(
            f"sweep parameter '{parameter}' definition {definition!r} is not "
            f"iterable"
        ) from exc
    if not values:
        raise ValueError(f"sweep parameter '{parameter}' has no values")
    return [_as_scalar(v, parameter) for v in values]


class ParameterSweep(JSONSerializable):
    """A deterministic grid of parameter combinations.

    Args:
        definitions: Mapping of ``{parameter_name: definition}`` where a
            definition is one of:

            * a list/tuple of explicit numeric values, e.g. ``[0.0, 0.5, 1.0]``;
            * ``{"values": [...]}`` — explicit values;
            * ``{"range": (start, stop, step)}`` — NumPy ``arange``-style;
            * ``{"start": .., "stop": .., "num_points": n}`` — ``linspace``;
            * ``{"start": .., "stop": .., "step": ..}`` — ``arange``-style.

        name: Optional sweep identifier (used in experiment metadata).

    Raises:
        ValueError: If definitions are empty, parameter names are invalid,
            values are non-numeric/non-finite, or ranges are malformed.

    Example:
        ``sweep = ParameterSweep({"theta": [0.0, 0.5, 1.0], "phi": [0.0, 1.0]})``
        produces 6 deterministic combinations (``theta`` outer, ``phi`` inner).
    """

    def __init__(
        self,
        definitions: Mapping[str, ParameterDefinition],
        *,
        name: str = "sweep",
    ) -> None:
        if not isinstance(definitions, Mapping):
            raise ValueError("ParameterSweep requires a mapping of definitions")
        if not definitions:
            raise ValueError("ParameterSweep requires at least one parameter")
        self._name = str(name)
        self._definitions: list[tuple[str, list[float]]] = []
        for parameter, definition in definitions.items():
            if not isinstance(parameter, str) or not parameter.strip():
                raise ValueError(
                    f"sweep parameter names must be non-empty strings, "
                    f"got {parameter!r}"
                )
            expanded = _expand_definition(parameter, definition)
            assert expanded, "expansion must not be empty"
            self._definitions.append((str(parameter), expanded))

    # -- identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        """Sweep identifier."""
        return self._name

    @property
    def parameters(self) -> tuple[str, ...]:
        """Parameter names in definition (thus combination) order."""
        return tuple(name for name, _ in self._definitions)

    @property
    def values(self) -> dict[str, list[float]]:
        """Expanded per-parameter value lists (deterministic order)."""
        return {name: list(values) for name, values in self._definitions}

    @property
    def num_combinations(self) -> int:
        """Number of Cartesian-product combinations."""
        if not self._definitions:
            return 0
        total = 1
        for _, values in self._definitions:
            total *= len(values)
        return total

    def __len__(self) -> int:
        return self.num_combinations

    def combinations(self) -> list[dict[str, float]]:
        """All parameter combinations in deterministic order.

        The order is the Cartesian product over parameters in *definition*
        order; within each parameter, values appear in their given order.
        """
        grids = [values for _, values in self._definitions]
        names = self.parameters
        return [
            dict(zip(names, combo, strict=True))
            for combo in itertools.product(*grids)
        ]

    # -- validation ---------------------------------------------------------

    def validate(self, parameter_names: Any) -> list[str]:
        """Return problems binding this sweep to a set of available parameters.

        Args:
            parameter_names: Iterable of parameter names available on the
                base circuit/plan.

        Returns:
            A list of human-readable problems (empty means this sweep is fully
            bindable).
        """
        available = set(parameter_names)
        problems: list[str] = []
        for parameter in self.parameters:
            if parameter not in available:
                problems.append(
                    f"sweep parameter '{parameter}' has no matching parameter "
                    f"on the base work"
                )
        return problems

    def verify(self, parameter_names: Any) -> None:
        """Raise :class:`ValueError` if the sweep cannot bind to the work.

        Args:
            parameter_names: Iterable of available parameter names.

        Raises:
            ValueError: If any swept parameter does not exist on the base work
                (``incompatible bindings``).
        """
        problems = self.validate(parameter_names)
        if problems:
            raise ValueError(
                "ParameterSweep is incompatible with the base work:\n  - "
                + "\n  - ".join(problems)
            )

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the sweep definition to a JSON-safe dictionary.

        Generated ranges are expanded to their concrete float values so the
        serialization is fully deterministic and round-trip stable.
        """
        return {
            "name": self._name,
            "parameters": list(self.parameters),
            "values": json_safe({name: list(values) for name, values in self._definitions}),
        }

    def to_json(self) -> str:
        """Serialize the sweep to a JSON string."""
        return json_string(self.to_dict())

    def __iter__(self):
        return iter(self.combinations())

    def __repr__(self) -> str:
        return (
            f"ParameterSweep('{self._name}', "
            f"parameters={list(self.parameters)}, "
            f"combinations={self.num_combinations})"
        )


__all__ = ["ParameterSweep"]