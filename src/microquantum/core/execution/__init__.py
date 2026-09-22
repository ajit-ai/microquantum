"""Backend-independent execution model.

Separates *what* a quantum program is from *how* it executes:
:class:`ExecutionRequest` → :class:`Executor` → :class:`ExecutionResult`,
with :class:`ExecutionOptions` and :class:`ExecutionContext` carrying
shots, seeds and metadata.  A minimal :class:`StateVectorExecutor`
reference implementation is included; state-vector, density-matrix,
native and hardware providers all implement :class:`Executor`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = [
    "ExecutionOptions",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionContext",
    "Executor",
    "StateVectorExecutor",
]


@dataclass(frozen=True)
class ExecutionOptions:
    """Execution configuration (shots, seed, result kinds)."""

    shots: int = 1024
    seed: int | None = None
    want_state: bool = False
    want_counts: bool = True

    def __post_init__(self) -> None:
        if self.shots < 1:
            raise ValueError("shots must be >= 1")


@dataclass(frozen=True)
class ExecutionRequest:
    """A backend-independent execution request."""

    circuit: Any
    options: ExecutionOptions = field(default_factory=ExecutionOptions)
    parameters: Mapping[str, complex | float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def bound_circuit(self) -> Any:
        """Return the circuit with parameters bound (if parameterized)."""
        circuit: Any = self.circuit
        binder = getattr(circuit, "bind_parameters", None)
        if self.parameters and callable(binder):
            try:
                return binder(dict(self.parameters))
            except Exception:
                return circuit
        return circuit


@dataclass
class ExecutionResult:
    """Backend-independent execution result."""

    counts: dict[str, int] | None = None
    state: Any = None
    expectation: float | None = None
    shots: int = 0
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_counts(self) -> bool:
        """True when shot counts are present."""
        return self.counts is not None

    @property
    def has_state(self) -> bool:
        """True when a final state is present."""
        return self.state is not None

    def get_counts(self) -> dict[str, int]:
        """Return shot counts (raises when absent)."""
        if self.counts is None:
            raise ValueError("No counts in this ExecutionResult")
        return dict(self.counts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        from microquantum._json import json_safe  # noqa: PLC0415

        state_payload: Any = None
        if self.state is not None:
            arr = getattr(self.state, "amplitudes", self.state)
            arr_np = np.asarray(arr)
            state_payload = {"real": arr_np.real.ravel().tolist(), "imag": arr_np.imag.ravel().tolist()}
        return {
            "version": 1,
            "counts": dict(self.counts) if self.counts else None,
            "state": state_payload,
            "expectation": self.expectation,
            "shots": self.shots,
            "seed": self.seed,
            "metadata": json_safe(dict(self.metadata)),
        }


@dataclass
class ExecutionContext:
    """Ambient execution context (backend name, extra config)."""

    backend_name: str = "statevector"
    config: dict[str, Any] = field(default_factory=dict)


class Executor(ABC):
    """Abstract executor: runs :class:`ExecutionRequest` objects."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Executor name."""

    @abstractmethod
    def run(self, request: ExecutionRequest, context: ExecutionContext | None = None) -> ExecutionResult:
        """Execute *request* and return an :class:`ExecutionResult`."""

    def run_circuit(self, circuit: Any, options: ExecutionOptions | None = None) -> ExecutionResult:
        """Convenience wrapper around :meth:`run`."""
        return self.run(ExecutionRequest(circuit=circuit, options=options or ExecutionOptions()))


def _simulate_circuit(circuit: Any) -> Any:
    """Simulate a bound :class:`QuantumCircuit` to amplitudes.

    Uses the Core unitary model (``get_unitary`` applied to ``|0...0⟩``)
    so execution stays independent of any particular simulator backend.
    """
    get_unitary = getattr(circuit, "get_unitary", None)
    if not callable(get_unitary):
        raise TypeError(f"Cannot simulate circuit of type {type(circuit).__name__}")
    mat = np.asarray(get_unitary().matrix, dtype=np.complex128)
    dim = mat.shape[0]
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0
    return mat @ psi0


class StateVectorExecutor(Executor):
    """Reference state-vector executor using the Core circuit simulator."""

    @property
    def name(self) -> str:
        return "statevector"

    def run(self, request: ExecutionRequest, context: ExecutionContext | None = None) -> ExecutionResult:
        circuit = request.bound_circuit()
        options = request.options
        amplitudes = _simulate_circuit(circuit)
        result = ExecutionResult(shots=options.shots, seed=options.seed)
        if options.want_state:
            result.state = amplitudes.copy()
        if options.want_counts:
            probs = (np.abs(amplitudes) ** 2).astype(float)
            probs = probs / probs.sum()
            n = int(np.log2(amplitudes.shape[0]))
            rng = np.random.default_rng(options.seed)
            draws = rng.choice(amplitudes.shape[0], size=options.shots, p=probs)
            counts: dict[str, int] = {}
            for d in draws:
                key = format(int(d), f"0{n}b")
                counts[key] = counts.get(key, 0) + 1
            result.counts = counts
        result.metadata = {"executor": self.name, **dict(request.metadata)}
        return result
