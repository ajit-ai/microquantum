"""Coherent gradient framework: methods, results and engine.

Supports parameter-shift, finite-difference and analytic gradients over
expectation values, dispatching to the canonical Core gradient
implementation while allowing future methods without circuit API
changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

import numpy as np

from microquantum.core.gradient import gradient, parameter_shift_gradient

if TYPE_CHECKING:
    from microquantum.core.parameter import Parameter

__all__ = [
    "gradient",
    "parameter_shift_gradient",
    "GradientResult",
    "GradientMethod",
    "ParameterShiftGradient",
    "FiniteDifferenceGradient",
    "AnalyticGradient",
    "GradientEngine",
]


@dataclass(frozen=True)
class GradientResult:
    """Gradient of an expectation value w.r.t. circuit parameters."""

    values: tuple[float, ...]
    parameter_names: tuple[str, ...]
    method: str = "parameter-shift"

    def __post_init__(self) -> None:
        if len(self.values) != len(self.parameter_names):
            raise ValueError("values and parameter_names must have equal length")

    def as_dict(self) -> dict[str, float]:
        """Return gradients keyed by parameter name."""
        return dict(zip(self.parameter_names, self.values, strict=False))

    def __getitem__(self, name: str) -> float:
        """Return the gradient for parameter ``name``."""
        return self.as_dict()[name]


class GradientMethod(ABC):
    """Abstract gradient method."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Method identifier."""

    @abstractmethod
    def compute(
        self,
        circuit: Any,
        observable: Any,
        param_values: Mapping[str, float],
        params: Sequence[Any] | None = None,
    ) -> GradientResult:
        """Compute gradients of ⟨observable⟩ w.r.t. circuit parameters."""


def _param_names(circuit: Any, params: Sequence[Any] | None) -> list[str]:
    if params is not None:
        return [p.name if hasattr(p, "name") else str(p) for p in params]
    discovered = getattr(circuit, "parameters", ())
    items = discovered() if callable(discovered) else discovered
    return [p.name if hasattr(p, "name") else str(p) for p in items]


class ParameterShiftGradient(GradientMethod):
    """Parameter-shift gradients (exact for Pauli rotations)."""

    def __init__(self, shift: float = float(np.pi / 2)) -> None:
        if abs(float(np.sin(shift))) < 1e-12:
            raise ValueError("shift must not be a multiple of pi")
        self._shift = float(shift)

    @property
    def name(self) -> str:
        return "parameter-shift"

    def compute(
        self,
        circuit: Any,
        observable: Any,
        param_values: Mapping[str, float],
        params: Sequence[Any] | None = None,
    ) -> GradientResult:
        names = _param_names(circuit, params)
        bound = cast("Mapping[str | Parameter, float]", dict(param_values))
        grads = gradient(circuit, observable, bound, shift=self._shift)
        by_name = {getattr(p, "name", str(p)): float(g) for p, g in grads.items()}
        try:
            values = tuple(by_name[n] for n in names)
        except KeyError as exc:
            raise KeyError(f"Gradient missing parameter {exc}") from exc
        return GradientResult(values, tuple(names), method=self.name)


def _simulate_amplitudes(circuit: Any, values: Mapping[str, float]) -> Any:
    """Bind *values* and simulate the circuit to a state amplitude vector."""
    import numpy as _np  # noqa: PLC0415

    bound = circuit
    if dict(values) and hasattr(circuit, "bind_parameters"):
        bound = circuit.bind_parameters(dict(values))
    get_unitary = getattr(bound, "get_unitary", None)
    if not callable(get_unitary):
        raise TypeError(f"Cannot simulate circuit of type {type(circuit).__name__}")
    mat = _np.asarray(get_unitary().matrix, dtype=_np.complex128)
    dim = mat.shape[0]
    psi0 = _np.zeros(dim, dtype=_np.complex128)
    psi0[0] = 1.0
    return mat @ psi0


def _expectation_from_amplitudes(amplitudes: Any, observable: Any) -> float:
    """Evaluate ⟨observable⟩ for simulated amplitudes."""
    import numpy as _np  # noqa: PLC0415

    from microquantum.core.state import StateVector  # noqa: PLC0415

    vec = _np.asarray(amplitudes, dtype=_np.complex128)
    n = int(_np.log2(vec.shape[0]))
    state = StateVector(n, amplitudes=vec)
    expectation = getattr(observable, "expectation", None)
    if callable(expectation):
        return float(expectation(state))
    to_matrix = getattr(observable, "to_matrix", None)
    if callable(to_matrix):
        mat = _np.asarray(to_matrix(), dtype=_np.complex128)
        return float(_np.real(_np.vdot(vec, mat @ vec)))
    dense = getattr(observable, "matrix", None)
    if isinstance(dense, _np.ndarray):
        mat = _np.asarray(dense, dtype=_np.complex128)
        if mat.shape != (vec.shape[0], vec.shape[0]):
            raise ValueError("Observable dimension does not match state")
        return float(_np.real(_np.vdot(vec, mat @ vec)))
    raise TypeError(f"Unsupported observable type {type(observable).__name__}")


class FiniteDifferenceGradient(GradientMethod):
    """Central finite-difference gradients for any expectation function."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._epsilon = float(epsilon)

    @property
    def name(self) -> str:
        return "finite-difference"

    def compute(
        self,
        circuit: Any,
        observable: Any,
        param_values: Mapping[str, float],
        params: Sequence[Any] | None = None,
    ) -> GradientResult:
        names = _param_names(circuit, params)
        base = dict(param_values)
        for target in names:
            if target not in base:
                raise KeyError(f"No value provided for parameter '{target}'")
        values: list[float] = []
        for target in names:
            plus = dict(base)
            minus = dict(base)
            plus[target] = base[target] + self._epsilon
            minus[target] = base[target] - self._epsilon
            fp = _expectation_from_amplitudes(_simulate_amplitudes(circuit, plus), observable)
            fm = _expectation_from_amplitudes(_simulate_amplitudes(circuit, minus), observable)
            values.append((fp - fm) / (2 * self._epsilon))
        return GradientResult(tuple(values), tuple(names), method=self.name)


class AnalyticGradient(GradientMethod):
    """Analytic gradients where available; delegates to parameter-shift."""

    def __init__(self) -> None:
        self._inner = ParameterShiftGradient()

    @property
    def name(self) -> str:
        return "analytic"

    def compute(
        self,
        circuit: Any,
        observable: Any,
        param_values: Mapping[str, float],
        params: Sequence[Any] | None = None,
    ) -> GradientResult:
        inner = self._inner.compute(circuit, observable, param_values, params)
        return GradientResult(inner.values, inner.parameter_names, method=self.name)


class GradientEngine:
    """Dispatching engine over registered :class:`GradientMethod` objects."""

    def __init__(self, method: GradientMethod | str = "parameter-shift") -> None:
        self._methods: dict[str, GradientMethod] = {
            "parameter-shift": ParameterShiftGradient(),
            "finite-difference": FiniteDifferenceGradient(),
            "analytic": AnalyticGradient(),
        }
        if isinstance(method, GradientMethod):
            self._methods[method.name] = method
            self._default = method.name
        else:
            if method not in self._methods:
                raise ValueError(f"Unknown gradient method '{method}'")
            self._default = method

    @property
    def default_method(self) -> str:
        """Default method name."""
        return self._default

    @property
    def available_methods(self) -> tuple[str, ...]:
        """Registered method names."""
        return tuple(sorted(self._methods))

    def register(self, method: GradientMethod) -> None:
        """Register a custom gradient method."""
        self._methods[method.name] = method

    def compute(
        self,
        circuit: Any,
        observable: Any,
        param_values: Mapping[str, float],
        params: Sequence[Any] | None = None,
        method: str | None = None,
    ) -> GradientResult:
        """Compute gradients with ``method`` (default: engine default)."""
        key = method or self._default
        if key not in self._methods:
            raise ValueError(f"Unknown gradient method '{key}'")
        return self._methods[key].compute(circuit, observable, param_values, params)
