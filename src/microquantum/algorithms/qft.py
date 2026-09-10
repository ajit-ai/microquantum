"""Quantum Fourier Transform.

Implements the QFT and inverse QFT using the standard textbook
decomposition with Hadamard and controlled-Rz gates, optionally
followed by SWAP gates to reverse qubit ordering.
"""

from __future__ import annotations

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.state import StateVector


def _crz(circuit: QuantumCircuit, control: int, target: int, theta: float) -> None:
    """Apply CRz(theta) via Rz-CX-Rz-CX decomposition."""
    circuit.rz(theta / 2, target)
    circuit.cx(control, target)
    circuit.rz(-theta / 2, target)
    circuit.cx(control, target)


def qft_circuit(num_qubits: int, do_swaps: bool = True) -> QuantumCircuit:
    """Build the Quantum Fourier Transform circuit.

    Args:
        num_qubits: Number of qubits.
        do_swaps: If True, append SWAP gates to reverse qubit order.

    Returns:
        QuantumCircuit implementing the QFT.
    """
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        for j in range(i + 1, num_qubits):
            angle = 2.0 * np.pi / (2 ** (j - i + 1))
            _crz(qc, j, i, angle)
    if do_swaps:
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - 1 - i)
    return qc


def inverse_qft_circuit(num_qubits: int, do_swaps: bool = True) -> QuantumCircuit:
    """Build the inverse Quantum Fourier Transform circuit.

    Args:
        num_qubits: Number of qubits.
        do_swaps: If True, assumes the forward QFT included SWAP gates.

    Returns:
        QuantumCircuit implementing the inverse QFT.
    """
    return qft_circuit(num_qubits, do_swaps=do_swaps).inverse()


class QFT:
    """Quantum Fourier Transform.

    Args:
        num_qubits: Number of qubits.
        do_swaps: If True, include SWAP gates to reverse qubit order.
    """

    def __init__(self, num_qubits: int, do_swaps: bool = True) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self._num_qubits = num_qubits
        self._do_swaps = do_swaps

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def build_circuit(self) -> QuantumCircuit:
        return qft_circuit(self._num_qubits, do_swaps=self._do_swaps)

    def build_inverse_circuit(self) -> QuantumCircuit:
        return inverse_qft_circuit(self._num_qubits, do_swaps=self._do_swaps)

    def run_forward(self, state: StateVector) -> StateVector:
        """Apply QFT to a state vector."""
        if state.num_qubits != self._num_qubits:
            raise ValueError(
                f"State has {state.num_qubits} qubits but QFT expects {self._num_qubits}"
            )
        return self.build_circuit().run(state)

    def run_inverse(self, state: StateVector) -> StateVector:
        """Apply inverse QFT to a state vector."""
        if state.num_qubits != self._num_qubits:
            raise ValueError(
                f"State has {state.num_qubits} qubits but QFT expects {self._num_qubits}"
            )
        return self.build_inverse_circuit().run(state)

    def __repr__(self) -> str:
        return (
            f"QFT(num_qubits={self._num_qubits}, do_swaps={self._do_swaps})"
        )
