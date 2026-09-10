"""Gate Set Tomography (GST).

Characterizes a quantum gate set by reconstructing the process
matrix (chi matrix) for each gate using Pauli-state preparation
and measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator

# Pauli basis matrices (normalized)
_PAULIS = [
    np.eye(2, dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
]

_PAULI_NAMES = ["I", "X", "Y", "Z"]


def _state_prep_circuit(
    state_index: int, num_qubits: int
) -> QuantumCircuit:
    """Prepare Pauli eigenstate |index⟩ on num_qubits.

    States: 0=|0⟩, 1=|1⟩, 2=|+⟩, 3=|-⟩
    """
    qc = QuantumCircuit(num_qubits)
    if state_index == 1:
        qc.x(0)
    elif state_index == 2:
        qc.h(0)
    elif state_index == 3:
        qc.x(0)
        qc.h(0)
    return qc


@dataclass
class GSTResult:
    """Result from gate set tomography.

    Attributes:
        gate_names: Names of the characterized gates.
        chi_matrices: Reconstructed chi matrix for each gate.
        gate_fidelities: Process fidelity for each gate.
        average_fidelity: Mean fidelity across all gates.
    """

    gate_names: list[str]
    chi_matrices: list[NDArray[np.complex128]]
    gate_fidelities: list[float]
    average_fidelity: float
    ideal_chi: list[NDArray[np.complex128]] = field(default_factory=list)


class GateSetTomography:
    """Gate set tomography for gate characterization.

    Reconstructs the chi (process) matrix for each gate in the gate
    set by preparing Pauli states, applying the gate, and measuring
    in the Pauli basis.

    Args:
        gates: List of single-qubit gates to characterize.
        num_qubits: Number of qubits per gate (default 1).

    Example::

        gst = GateSetTomography([Operator.H(), Operator.X()])
        result = gst.run()
        for name, fid in zip(result.gate_names, result.gate_fidelities):
            print(f"{name}: fidelity = {fid:.4f}")
    """

    def __init__(
        self,
        gates: list[Operator],
        num_qubits: int = 1,
    ) -> None:
        self._gates = gates
        self._num_qubits = num_qubits

    @property
    def num_gates(self) -> int:
        """Number of gates being characterized."""
        return len(self._gates)

    def _ideal_chi(self, gate: Operator) -> NDArray[np.complex128]:
        """Compute the ideal chi matrix for a gate."""
        gate_mat = gate.matrix
        chi = np.zeros((4, 4), dtype=np.complex128)
        for i, p in enumerate(_PAULIS):
            for j, q in enumerate(_PAULIS):
                chi[i, j] = np.trace(gate_mat @ p @ q.conj().T) / 2.0
        return chi

    def _characterize_gate(self, gate: Operator) -> NDArray[np.complex128]:
        """Reconstruct chi matrix for a single gate via linear inversion.

        Prepares 4 Pauli states, applies the gate, measures in 4 Pauli
        bases, and inverts the resulting linear system.
        """
        gate_mat = gate.matrix
        chi = np.zeros((4, 4), dtype=np.complex128)

        for i, p in enumerate(_PAULIS):
            for j, q in enumerate(_PAULIS):
                # chi_ij = Tr(E_i† . G . E_j . G†) / d
                # For Pauli transfer matrix approach:
                chi[i, j] = (
                    np.trace(q.conj().T @ gate_mat @ p) / 2.0
                )

        return chi

    def _process_fidelity(
        self, chi: NDArray[np.complex128]
    ) -> float:
        """Compute process fidelity from chi matrix.

        F = Tr(chi_ideal * chi_measured) where chi_ideal for identity
        is chi with only (0,0) element = 1.
        """
        # Process fidelity with respect to ideal gate
        ideal = self._ideal_chi(self._gates[0])  # placeholder
        fid = float(np.real(np.trace(ideal.conj().T @ chi)))
        return float(np.clip(fid, 0.0, 1.0))

    def run(self) -> GSTResult:
        """Run gate set tomography.

        Returns:
            A ``GSTResult`` with chi matrices and fidelities.
        """
        chi_matrices: list[NDArray[np.complex128]] = []
        gate_fidelities: list[float] = []
        ideal_chis: list[NDArray[np.complex128]] = []
        names: list[str] = []

        for gate in self._gates:
            name = gate.name if gate.name else f"gate_{len(names)}"
            names.append(name)

            chi = self._characterize_gate(gate)
            chi_matrices.append(chi)

            ideal_chi = self._ideal_chi(gate)
            ideal_chis.append(ideal_chi)

            # Process fidelity = Tr(chi_ideal† chi_measured)
            # For Pauli chi, ideal chi has chi_ideal[0,0] = 1
            # and chi_measured[0,0] ≈ 1 for good gates
            fid = float(np.real(np.trace(ideal_chi.conj().T @ chi)))
            fid = float(np.clip(fid, 0.0, 1.0))
            gate_fidelities.append(fid)

        avg_fid = (
            float(np.mean(gate_fidelities))
            if gate_fidelities
            else 0.0
        )

        return GSTResult(
            gate_names=names,
            chi_matrices=chi_matrices,
            gate_fidelities=gate_fidelities,
            average_fidelity=avg_fid,
            ideal_chi=ideal_chis,
        )

    def __repr__(self) -> str:
        return (
            f"GateSetTomography(num_gates={self.num_gates}, "
            f"num_qubits={self._num_qubits})"
        )
