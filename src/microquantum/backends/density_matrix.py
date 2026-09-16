"""Density matrix simulation backend."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.density_matrix import DensityMatrix
from ..core.device import Target
from ..core.state import StateVector
from .base import Backend, BackendResult, _normalize_shots
from .capabilities import BackendCapabilities, simulator_capabilities


class DensityMatrixBackend(Backend):
    """Backend that simulates circuits using density matrices.

    Supports both pure and mixed state simulation, and is the
    foundation for noise modeling via Kraus operators.
    """

    @property
    def name(self) -> str:
        return "density_matrix"

    @property
    def target(self) -> Target:
        """Advertises a universal simulator gate set."""
        return Target.universal(name=f"{self.name}_simulator")

    @property
    def capabilities(self) -> BackendCapabilities:
        """Simulator capability set including density-matrix execution."""
        caps = simulator_capabilities(
            max_qubits=self.num_qubits, density_matrix=True
        )
        caps.metadata.update(
            {
                "engine": "density_matrix",
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
        noise_model: Optional[object] = None,
    ) -> BackendResult:
        """Execute circuit via density matrix simulation.

        Args:
            num_qubits: Number of qubits.
            gates: List of (gate_matrix, target_qubits) pairs.
            shots: Number of measurement shots.  ``None`` requests a
                deterministic (un-sampled) run: no counts are produced and
                the exact final density matrix is returned.
            initial_state: Starting state vector. Defaults to |0...0>.
            seed: RNG seed.
            noise_model: Optional :class:`~microquantum.NoiseModel` applied
                after each gate.  Any other value raises ``TypeError`` — it
                is never silently ignored.

        Returns:
            BackendResult with density matrix and measurement counts.

        Raises:
            TypeError: If ``noise_model`` is not ``None`` or a
                :class:`~microquantum.NoiseModel`.
        """
        shots = _normalize_shots(shots)

        if noise_model is not None:
            from .noise import NoiseModel

            if not isinstance(noise_model, NoiseModel):
                raise TypeError(
                    "noise_model must be a NoiseModel or None, "
                    f"got {type(noise_model).__name__}"
                )

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
                rho = noise_model.apply(rho)

        if shots is None:
            counts: dict[str, int] = {}
            result_shots: Optional[int] = None
        else:
            counts = self._sample_density_matrix(rho, shots, seed)
            result_shots = shots

        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            density_matrix=rho.matrix.copy(),
            counts=counts,
            shots=result_shots,
            seed=seed,
            target_name=self.target.name,
            metadata={"shots": result_shots, "purity": rho.is_pure},
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
        shots: Optional[int],
        seed: Optional[int],
    ) -> dict[str, int]:
        """Sample measurement outcomes from a density matrix.

        Uses the diagonal elements (probabilities) to sample.
        """
        shots = _normalize_shots(shots)
        if shots is None:
            return {}

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
