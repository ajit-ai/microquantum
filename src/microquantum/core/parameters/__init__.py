"""Parameter system for variational quantum computing.

Re-exports the canonical :class:`Parameter` / :class:`ParameterExpression`
and adds :class:`ParameterVector`, :class:`ParameterBinding`, trigonometric
expression nodes and deterministic ordering/discovery helpers.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, cast

from microquantum.core.parameter import Parameter, ParameterExpression

__all__ = [
    "Parameter",
    "ParameterExpression",
    "ParameterVector",
    "ParameterBinding",
    "Sin",
    "Cos",
    "discover_parameters",
    "ordered_parameters",
    "bind_all",
]


class ParameterVector:
    """An ordered vector of related parameters (e.g. QAOA angles)."""

    def __init__(self, name: str, size: int) -> None:
        if not name:
            raise ValueError("ParameterVector name must be non-empty")
        if size < 1:
            raise ValueError("ParameterVector size must be >= 1")
        self._name = name
        self._params = [Parameter(f"{name}[{i}]") for i in range(size)]

    @property
    def name(self) -> str:
        """Vector name."""
        return self._name

    @property
    def size(self) -> int:
        """Number of parameters."""
        return len(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __getitem__(self, index: int) -> Parameter:
        """Return the parameter at ``index``."""
        return self._params[index]

    def __iter__(self) -> Any:
        return iter(self._params)

    def __repr__(self) -> str:
        return f"ParameterVector('{self._name}', size={len(self._params)})"


class ParameterBinding:
    """A validated mapping from parameter names to numeric values."""

    def __init__(self, values: Mapping[str, complex | float]) -> None:
        cleaned: dict[str, complex] = {}
        for key, value in values.items():
            if not key:
                raise ValueError("Parameter names must be non-empty")
            cleaned[str(key)] = complex(value)
        self._values = cleaned

    @property
    def values(self) -> dict[str, complex]:
        """Bound values (copy)."""
        return dict(self._values)

    @property
    def names(self) -> tuple[str, ...]:
        """Bound parameter names in sorted order."""
        return tuple(sorted(self._values))

    def __getitem__(self, name: str) -> complex:
        """Return the value bound to ``name``."""
        if name not in self._values:
            raise KeyError(f"Parameter '{name}' is not bound")
        return self._values[name]

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def merge(self, other: ParameterBinding) -> ParameterBinding:
        """Merge two bindings (``other`` wins on conflicts)."""
        merged = dict(self._values)
        merged.update(other._values)
        return ParameterBinding(merged)

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize to a JSON-safe dictionary."""
        return {k: [v.real, v.imag] for k, v in self._values.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterBinding:
        """Deserialize from :meth:`to_dict` output."""
        return cls({k: complex(v[0], v[1]) for k, v in data.items()})


class _TrigNode:
    """Base class for sin/cos expression nodes."""

    _kind: str = "trig"

    def __init__(self, operand: Any) -> None:
        self._operand = operand

    @property
    def operand(self) -> Any:
        """Inner expression."""
        return self._operand

    def parameters(self) -> tuple[Parameter, ...]:
        """Parameters used by this node."""
        op = self._operand
        if isinstance(op, Parameter):
            return (op,)
        single = getattr(op, "parameter", None)
        if isinstance(single, Parameter):
            return (single,)
        get = getattr(op, "parameters", None)
        if callable(get):
            result = get()
            return tuple(result)
        return ()

    def evaluate(self, values: Mapping[str, complex | float]) -> complex:
        """Evaluate the node given parameter values."""
        op = self._operand
        if isinstance(op, Parameter):
            if op.name not in values:
                raise ValueError(f"Unbound parameter '{op.name}'")
            base = complex(values[op.name])
        elif isinstance(op, ParameterExpression):
            real_values = cast("dict[str | Parameter, float]", _real_map(values))
            base = complex(op.evaluate(real_values))
        elif hasattr(op, "evaluate"):
            evaluate = op.evaluate
            if not callable(evaluate):
                raise TypeError(f"Cannot evaluate operand of type {type(op).__name__}")
            real_values = cast("dict[str | Parameter, float]", _real_map(values))
            base = complex(evaluate(real_values))
        else:
            base = complex(op)
        fn = math.sin if self._kind == "sin" else math.cos
        # Support complex arguments via cmath-compatible path.
        import cmath as _cmath

        resolved = _cmath.sin(base) if self._kind == "sin" else _cmath.cos(base)
        _ = fn  # keep real fast-path reference for readability
        return resolved

    def __repr__(self) -> str:
        return f"{self._kind}({self._operand!r})"


class Sin(_TrigNode):
    """Sine of a parameter expression."""

    _kind = "sin"


class Cos(_TrigNode):
    """Cosine of a parameter expression."""

    _kind = "cos"


def _real_map(values: Mapping[str, complex | float]) -> dict[str, float]:
    """Convert a binding to real values for expression evaluation."""
    return {str(k): float(complex(v).real) for k, v in values.items()}


def _expr_params(expr: Any) -> tuple[Parameter, ...]:
    if isinstance(expr, Parameter):
        return (expr,)
    single = getattr(expr, "parameter", None)
    if isinstance(single, Parameter):
        return (single,)
    params = getattr(expr, "parameters", None)
    if params is None:
        return ()
    result = params() if callable(params) else params
    return tuple(result)


def discover_parameters(expressions: Sequence[Any]) -> tuple[Parameter, ...]:
    """Collect unique parameters across expressions in first-seen order."""
    seen: dict[str, Parameter] = {}
    for expr in expressions:
        for p in _expr_params(expr):
            if p.name not in seen:
                seen[p.name] = p
    return tuple(seen.values())


def ordered_parameters(expressions: Sequence[Any]) -> tuple[Parameter, ...]:
    """Deterministic (name-sorted) parameter ordering."""
    return tuple(sorted(discover_parameters(expressions), key=lambda p: p.name))


def bind_all(
    expression: Any, binding: Mapping[str, complex | float] | ParameterBinding
) -> complex:
    """Evaluate *expression* under *binding*."""
    values = binding.values if isinstance(binding, ParameterBinding) else dict(binding)
    if isinstance(expression, Parameter):
        if expression.name not in values:
            raise ValueError(f"Unbound parameter '{expression.name}'")
        return complex(values[expression.name])
    evaluate = getattr(expression, "evaluate", None)
    if callable(evaluate):
        return complex(evaluate(dict(values)))
    return complex(expression)
