"""Statevector simulation backend."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.device import Target
from ..core.engine import apply_gate
from ..core.measurement import sample_state
from ..core.state import StateVector
from .base import Backend, BackendResult, _normalize_shots
from .capabilities import BackendCapabilities, simulator_capabilities


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

    @property
    def capabilities(self) -> BackendCapabilities:
        """Simulator capability set with the statevector engine metadata."""
        caps = simulator_capabilities(max_qubits=self.num_qubits, statevector=True)
        caps.metadata.update(
            {
                "engine": "statevector",
                "device_type": "cpu",
                "numerics": "numpy",
            }
        )
        return caps

    def run_circuit(
        self,
        num_qubits: int,
        gates: list[tuple[NDArray[np.complex128], list[int]]],
        shots: Optional[int] = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Execute circuit via state vector simulation.

        Args:
            num_qubits: Number of qubits.
            gates: List of (gate_matrix, target_qubits) pairs.
            shots: Number of measurement shots.  ``None`` requests a
                deterministic (un-sampled) run: no counts are produced and
                the exact final state is returned.
            initial_state: Starting state. Defaults to |0...0>.
            seed: RNG seed for reproducibility.

        Returns:
            BackendResult with state vector and measurement counts.
        """
        shots = _normalize_shots(shots)

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

        if shots is None:
            counts: dict[str, int] = {}
            samples: Optional[list[int]] = None
            result_shots: Optional[int] = None
        else:
            measurement = sample_state(state, shots=shots, seed=seed)
            counts = measurement.counts
            samples = measurement.samples
            result_shots = shots

        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            statevector=state.amplitudes.copy(),
            counts=counts,
            samples=samples,
            shots=result_shots,
            seed=seed,
            target_name=self.target.name,
            metadata={"shots": result_shots},
        )
