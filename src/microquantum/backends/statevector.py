"""Statevector simulation backend."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.device import Target
from ..core.engine import apply_gate
from ..core.measurement import sample_state
from ..core.state import StateVector
from .base import Backend, BackendResult


class StatevectorBackend(Backend):
    """Backend that simulates circuits using state vectors.

    Applies gates via tensor contraction (einsum) for memory efficiency.
    Supports measurement sampling via Born rule probabilities.
    """

    @property
    def name(self) -> str:
        return "statevector"

    @property
    def target(self) -> Target:
        """Advertises a universal simulator gate set."""
        return Target.universal(name=f"{self.name}_simulator")

    def run_circuit(
        self,
        num_qubits: int,
        gates: list[tuple[NDArray[np.complex128], list[int]]],
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Execute circuit via state vector simulation.

        Args:
            num_qubits: Number of qubits.
            gates: List of (gate_matrix, target_qubits) pairs.
            shots: Number of measurement shots.
            initial_state: Starting state. Defaults to |0...0>.
            seed: RNG seed for reproducibility.

        Returns:
            BackendResult with state vector and measurement counts.
        """
        if initial_state is None:
            state = StateVector(num_qubits)
        else:
            if initial_state.num_qubits != num_qubits:
                raise ValueError(
                    f"Initial state has {initial_state.num_qubits} qubits "
                    f"but circuit has {num_qubits}"
                )
            state = initial_state.copy()

        for gate_matrix, targets in gates:
            state = apply_gate(state, gate_matrix, targets)

        measurement = sample_state(state, shots=shots, seed=seed)

        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            statevector=state.amplitudes.copy(),
            counts=measurement.counts,
            metadata={"shots": shots},
        )
