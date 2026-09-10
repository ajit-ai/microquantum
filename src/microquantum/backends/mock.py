"""Deterministic mock backend for tests and demos (MQ-06).

The mock backend implements the *real* :class:`Backend` contract (it does
not bypass it): it accepts bound circuits, validates plans through the same
``validate`` / ``execute`` path every backend uses, and returns a
deterministic :class:`BackendResult` whose counts are a pure function of the
circuit gate labels.  No simulation engine is used — this is a stub for
verifying registration, routing, validation and result plumbing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.device import Device, DeviceType, Target
from ..core.state import StateVector
from .base import Backend, BackendResult
from .capabilities import (
    EXECUTION_SAMPLING,
    EXECUTION_SHOTS,
    EXECUTION_STATEVECTOR,
    FEATURE_MEASUREMENT,
    FEATURE_PARAMETERIZED_CIRCUITS,
    BackendCapabilities,
    TargetClass,
)


class MockBackend(Backend):
    """Deterministic plan-contract backend.

    Args:
        name: Backend name (default ``"mock"``).
        max_qubits: Advertised qubit capacity.  ``None`` = unbounded.
        supports_shots: Whether shot-based execution is advertised.
        options: Free-form capability metadata.
    """

    def __init__(
        self,
        name: str = "mock",
        max_qubits: Optional[int] = None,
        supports_shots: bool = True,
        options: Optional[dict] = None,
    ) -> None:
        self._name = name
        self._max_qubits = max_qubits
        self._supports_shots = supports_shots
        self._options = dict(options or {})

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_qubits(self) -> Optional[int]:
        return self._max_qubits

    @property
    def device(self) -> Device:
        return Device(
            name=self.name,
            device_type=DeviceType.SIMULATOR,
            max_qubits=self._max_qubits,
            metadata=self._options,
        )

    @property
    def target(self) -> Target:
        return Target.universal(
            name=f"{self.name}_target", num_qubits=self._max_qubits
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        execution = {EXECUTION_STATEVECTOR, EXECUTION_SAMPLING}
        if self._supports_shots:
            execution.add(EXECUTION_SHOTS)
        return BackendCapabilities(
            target_class=TargetClass.SIMULATOR,
            execution=frozenset(execution),
            circuit_features=frozenset(
                {FEATURE_PARAMETERIZED_CIRCUITS, FEATURE_MEASUREMENT}
            ),
            max_qubits=self._max_qubits,
            metadata=self._options,
        )

    def run_circuit(
        self,
        num_qubits: int,
        gates: list[tuple[NDArray[np.complex128], list[int]]],
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Execute deterministically without running a simulator.

        Counts are derived from the number of 2-qubit gates and the seed so
        repeat executions with the same seed produce identical results.
        """
        rng = np.random.default_rng(0 if seed is None else seed)
        two_qubit = sum(1 for _matrix, targets in gates if len(targets) >= 2)
        n_states = 2**num_qubits
        # Bias toward bitstring ``1`` repeated, keyed by two-qubit depth.
        weight = np.zeros(n_states, dtype=float)
        weight[1] = 0.5 + 0.1 * two_qubit
        weight[0] = 0.5 - 0.1 * two_qubit
        weight = np.clip(weight, 1e-3, None)
        weight /= weight.sum()
        indices = rng.choice(n_states, size=shots, p=weight)
        counts: dict[str, int] = {}
        for index in indices:
            key = format(int(index), f"0{num_qubits}b")
            counts[key] = counts.get(key, 0) + 1
        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            counts=counts,
            samples=[int(i) for i in indices],
            shots=shots,
            seed=seed,
            metadata={"mock": True, "legal": True},
        )


__all__ = ["MockBackend"]