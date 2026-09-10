"""Density matrix simulation backend."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.density_matrix import DensityMatrix
from ..core.state import StateVector
from .base import Backend, BackendResult


class DensityMatrixBackend(Backend):
    """Backend that simulates circuits using density matrices.

    Supports both pure and mixed state simulation, and is the
    foundation for noise modeling via Kraus operators.
    """

    @property
    def name(self) -> str:
        return "density_matrix"

    def run_circuit(
        self,
        num_qubits: int,
        gates: list[tuple[NDArray[np.complex128], list[int]]],
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
        noise_model: Optional[object] = None,
    ) -> BackendResult:
        """Execute circuit via density matrix simulation.

        Args:
            num_qubits: Number of qubits.
            gates: List of (gate_matrix, target_qubits) pairs.
            shots: Number of measurement shots.
            initial_state: Starting state vector. Defaults to |0...0>.
            seed: RNG seed.
            noise_model: Optional NoiseModel to apply after each gate.

        Returns:
            BackendResult with density matrix and measurement counts.
        """
        if initial_state is not None:
            if initial_state.num_qubits != num_qubits:
                raise ValueError(
                    f"Initial state has {initial_state.num_qubits} qubits "
                    f"but circuit has {num_qubits}"
                )
            rho = DensityMatrix.from_statevector(initial_state)
        else:
            rho = DensityMatrix(num_qubits)

        for gate_matrix, targets in gates:
            full_gate = self._expand_gate(gate_matrix, targets, num_qubits)
            rho = rho.apply_unitary(full_gate)

            if noise_model is not None:
                from .noise import NoiseModel
                if isinstance(noise_model, NoiseModel):
                    rho = noise_model.apply(rho)

        counts = self._sample_density_matrix(rho, shots, seed)

        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            density_matrix=rho.matrix.copy(),
            counts=counts,
            metadata={"shots": shots, "purity": rho.is_pure},
        )

    @staticmethod
    def _expand_gate(
        gate_matrix: NDArray[np.complex128],
        targets: list[int],
        num_qubits: int,
    ) -> NDArray[np.complex128]:
        """Expand a gate to act on the full Hilbert space.

        Uses Kronecker products to embed a k-qubit gate into the
        full 2^N dimensional space.
        """
        from ..core.operators import Operator
        from ..core.tensor import expand_operator

        op = Operator(np.asarray(gate_matrix, dtype=np.complex128))
        expanded = expand_operator(op, targets, num_qubits)
        return expanded.matrix

    @staticmethod
    def _sample_density_matrix(
        rho: DensityMatrix,
        shots: int,
        seed: Optional[int],
    ) -> dict[str, int]:
        """Sample measurement outcomes from a density matrix.

        Uses the diagonal elements (probabilities) to sample.
        """
        probs = np.real(np.diag(rho.matrix))
        probs = np.clip(probs, 0, None)
        total = probs.sum()
        if total > 0:
            probs = probs / total

        rng = np.random.default_rng(seed)
        outcomes = rng.choice(rho.dim, size=shots, p=probs)

        counts: dict[str, int] = {}
        for outcome in outcomes:
            bitstring = format(outcome, f"0{rho.num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
