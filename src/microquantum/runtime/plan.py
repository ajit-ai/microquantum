"""Execution plans: the declarative description of what to run.

An :class:`ExecutionPlan` is the canonical input for the execution runtime:

* exactly **one** of ``circuit`` / ``ir`` / ``compiled`` describes the work,
* ``target`` / ``backend`` describe where it runs,
* ``shots`` / ``parameter_bindings`` / ``initial_state`` / ``seed`` describe
  how it runs,
* ``optimization_level`` / ``options`` describe how it is compiled.

Plans are plain data: they are validated on demand (see :meth:`validate`)
but never execute anything themselves.  The runtime (:mod:`runtime.runtime`)
turns a plan into a :class:`~microquantum.backends.base.Job` and a
:class:`~microquantum.backends.base.BackendResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union, cast

import numpy as np

from .._json import JSONSerializable, json_safe, json_string
from ..core.parameter import Parameter
from ..core.state import StateVector
from ..ir import IRCircuit, from_ir

if TYPE_CHECKING:
    from ..backends.base import Backend
    from ..core.circuit import QuantumCircuit
    from ..core.device import Target

#: Explicit backend selection: a Backend instance or a registered name.
BackendRef = Union["Backend", str]

ParameterBinding = Mapping[Union[str, "Parameter"], float]


def _validate_shots(shots: int) -> int:
    if not isinstance(shots, int):
        raise ValueError(f"shots must be an integer, got {type(shots).__name__}")
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    return shots


@dataclass
class ExecutionPlan(JSONSerializable):
    """Declarative description of a single execution.

    Attributes:
        name: Human-readable plan identifier used in metadata and traces.
        circuit: The :class:`~microquantum.core.circuit.QuantumCircuit` to run.
        ir: An :class:`~microquantum.ir.IRCircuit` to run instead of a circuit.
        compiled: A :class:`~microquantum.ir.CompilationResult` to reuse
            instead of recompiling.
        target: Optional :class:`~microquantum.core.device.Target` the work is
            compiled for and validated against.
        backend: Optional :class:`~microquantum.backends.base.Backend` to run
            on, or the *name* of a registered backend (resolved through the
            runtime's :class:`~microquantum.backends.registry.BackendRegistry`).
            The runtime falls back to its default backend when omitted.
        shots: Number of measurement shots.
        parameter_bindings: Optional mapping of parameter names (or
            :class:`~microquantum.core.parameter.Parameter` objects) to
            numeric values to bind before execution.
        initial_state: Optional starting :class:`StateVector`.
        seed: RNG seed for reproducible execution.
        optimization_level: Compiler optimization level (0-2).
        options: Free-form runtime/compiler options.
        metadata: Free-form user metadata merged into the result.
    """

    name: str = "main"
    circuit: Optional[Any] = None
    ir: Optional[IRCircuit] = None
    compiled: Optional[Any] = None
    target: Optional[Any] = None
    backend: Optional[Any] = None
    shots: int = 1024
    parameter_bindings: Optional[ParameterBinding] = None
    initial_state: Optional[StateVector] = None
    seed: Optional[int] = None
    optimization_level: int = 0
    options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shots = _validate_shots(self.shots)
        if self.ir is not None:
            if self.circuit is not None or self.compiled is not None:
                raise ValueError("ExecutionPlan must specify exactly one of circuit/ir/compiled")
        elif self.compiled is not None:
            if self.circuit is not None:
                raise ValueError("ExecutionPlan must specify exactly one of circuit/ir/compiled")
        elif self.circuit is None:
            raise ValueError("ExecutionPlan must specify one of circuit/ir/compiled")
        if self.optimization_level < 0 or self.optimization_level > 2:
            raise ValueError(
                f"optimization_level must be in [0, 2], got {self.optimization_level}"
            )

    @classmethod
    def from_circuit(
        cls,
        circuit: QuantumCircuit,
        *,
        name: str = "main",
        target: Optional[Target] = None,
        backend: Optional[BackendRef] = None,
        shots: int = 1024,
        parameter_bindings: Optional[ParameterBinding] = None,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
        optimization_level: int = 0,
        options: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Build a plan directly from a circuit with convenient defaults."""
        return cls(
            name=name,
            circuit=circuit,
            target=target,
            backend=backend,
            shots=shots,
            parameter_bindings=parameter_bindings,
            initial_state=initial_state,
            seed=seed,
            optimization_level=optimization_level,
            options=dict(options or {}),
            metadata=dict(metadata or {}),
        )

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Parameter set of the described work (empty if none)."""
        if self.circuit is not None:
            return tuple(
                sorted(self.circuit.parameters, key=lambda p: p.name)
            )
        if self.ir is not None:
            return tuple(sorted(self.ir.parameters, key=lambda p: p.name))
        return ()

    @property
    def num_qubits(self) -> int:
        """Qubit count of the described work."""
        if self.circuit is not None:
            return cast(int, self.circuit.num_qubits)
        if self.ir is not None:
            return self.ir.num_qubits
        if self.compiled is not None:
            return cast(int, self.compiled.result.num_qubits)
        return 0

    def required_parameter_names(self) -> set[str]:
        """Names of the parameters that must be bound before execution."""
        params = self.parameters
        provided = {self._key_name(k) for k in (self.parameter_bindings or {})}
        needed = {p.name for p in params}
        remaining = needed - provided
        return remaining

    def _key_name(self, key: Union[str, Parameter]) -> str:
        return key if isinstance(key, str) else key.name

    def missing_bindings(self) -> list[str]:
        """Names of parameters referenced by bindings that do not exist."""
        params = self.parameters
        known = {p.name for p in params}
        bound_names = [self._key_name(k) for k in (self.parameter_bindings or {})]
        return sorted(n for n in bound_names if n not in known)

    def validate(self) -> list[str]:
        """Run static validation, returning a list of human-readable problems.

        A plan with no problems is executable.  Value-level problems (such
        as unknown bindings) are also checked here so callers can fail fast
        before a submission is made.
        """
        problems: list[str] = []
        remaining = self.required_parameter_names()
        if remaining:
            problems.append(
                f"plan is parameterized but bindings incomplete; "
                f"unbound parameters: {sorted(remaining)}"
            )
        unknown = self.missing_bindings()
        if unknown:
            problems.append(
                f"bindings reference parameters that are not in the circuit: {unknown}"
            )
        for key, value in (self.parameter_bindings or {}).items():
            if not isinstance(value, (int, float, np.generic)):
                problems.append(
                    f"binding for '{self._key_name(key)}' must be numeric, "
                    f"got {type(value).__name__}"
                )
        return problems

    def bound(self) -> QuantumCircuit:
        """Return the work as a parameter-bound static circuit.

        Raises:
            ValueError: If the plan is parameterized without full bindings, or
                if the work is described as compiled dynamic IR.
        """
        if self.circuit is not None:
            base: QuantumCircuit = self.circuit
        elif self.ir is not None:
            base = from_ir(self.ir)
        elif self.compiled is not None:
            base = self.compiled.circuit()
        else:  # pragma: no cover - guarded by __post_init__
            raise ValueError("ExecutionPlan has no circuit/ir/compiled work")

        bindings = self.parameter_bindings or {}
        if bindings:
            base = base.bind_parameters(dict(bindings))
        if base.is_parameterized:
            remaining = sorted(p.name for p in base.parameters)
            raise ValueError(
                f"plan remains parameterized; unbound parameters: {remaining}"
            )
        return base

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a JSON-safe dictionary."""
        backend_value: Optional[str] = None
        if self.backend is not None:
            backend_value = (
                self.backend if isinstance(self.backend, str) else self.backend.name
            )
        return {
            "name": self.name,
            "circuit": json_safe(self.circuit) if self.circuit is not None else None,
            "ir": self.ir.to_dict() if self.ir is not None else None,
            "compiled": json_safe(self.compiled) if self.compiled is not None else None,
            "target": self.target.to_dict() if self.target is not None else None,
            "backend": backend_value,
            "shots": self.shots,
            "parameter_bindings": json_safe(dict(self.parameter_bindings or {})),
            "initial_state": json_safe(self.initial_state)
            if self.initial_state is not None
            else None,
            "seed": self.seed,
            "optimization_level": self.optimization_level,
            "options": json_safe(dict(self.options)),
            "metadata": json_safe(dict(self.metadata)),
        }

    def to_json(self) -> str:
        """Serialize the plan to a pretty-printed JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        work = (
            f"circuit({self.num_qubits}q)"
            if self.circuit is not None
            else "ir"
            if self.ir is not None
            else "compiled"
        )
        return (
            f"ExecutionPlan('{self.name}', {work}, shots={self.shots}, "
            f"optimization_level={self.optimization_level})"
        )