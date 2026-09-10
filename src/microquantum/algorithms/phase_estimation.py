"""Quantum Phase Estimation.

Estimates the phase θ of a unitary operator U such that
U|ψ⟩ = e^(2πiθ)|ψ⟩, using controlled-U powers and the inverse QFT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..core.state import StateVector
from ..core.tensor import expand_operator
from .qft import inverse_qft_circuit


def _u_power(unitary: Operator, power: int) -> Operator:
    """Compute U^power via repeated squaring."""
    if power == 0:
        dim = 2 ** unitary.num_qubits
        return Operator(np.eye(dim, dtype=np.complex128), name="i")
    if power == 1:
        return unitary
    mat = unitary.matrix.copy()
    current_mat = unitary.matrix.copy()
    remaining = power - 1
    while remaining > 0:
        if remaining & 1:
            mat = mat @ current_mat
        current_mat = current_mat @ current_mat
        remaining >>= 1
    return Operator(np.asarray(mat, dtype=np.complex128), name="u_power")


def _controlled_u(unitary: Operator, total_qubits: int, control: int, targets: list[int]) -> Operator:
    """Build a controlled-U operator embedded into total_qubits system."""
    n_u = unitary.num_qubits
    I_target = np.eye(2**n_u, dtype=np.complex128)
    Z_mat = Operator.Z().matrix
    proj0 = (np.eye(2, dtype=np.complex128) + Z_mat) / 2
    proj1 = (np.eye(2, dtype=np.complex128) - Z_mat) / 2
    cu_mat = np.kron(proj0, I_target) + np.kron(proj1, unitary.matrix)
    cu_op = Operator(np.asarray(cu_mat, dtype=np.complex128), name="cu")
    return expand_operator(cu_op, [control] + targets, total_qubits)


@dataclass
class PhaseEstimationResult:
    """Result from Quantum Phase Estimation.

    Attributes:
        phase: Estimated phase (0 to 1).
        phase_radians: Estimated phase in radians (0 to 2*pi).
        eigenvalue: exp(2*pi*i*phase).
        num_counting_qubits: Number of counting qubits used.
        success_probability: Probability of measuring the correct phase.
    """

    phase: float
    phase_radians: float
    eigenvalue: complex
    num_counting_qubits: int
    success_probability: float


class PhaseEstimation:
    """Quantum Phase Estimation algorithm.

    Estimates the phase θ of a unitary operator U where U|ψ⟩ = e^(2πiθ)|ψ⟩.

    Args:
        unitary: The unitary operator whose eigenphase is to be estimated.
        num_counting_qubits: Number of qubits in the counting register.
    """

    def __init__(
        self, unitary: Operator, num_counting_qubits: int = 4
    ) -> None:
        if num_counting_qubits < 1:
            raise ValueError(
                f"num_counting_qubits must be >= 1, got {num_counting_qubits}"
            )
        if not unitary.is_unitary:
            raise ValueError("Operator must be unitary")
        self._unitary = unitary
        self._num_counting_qubits = num_counting_qubits

    @property
    def unitary(self) -> Operator:
        return self._unitary

    @property
    def num_counting_qubits(self) -> int:
        return self._num_counting_qubits

    def build_circuit(self) -> QuantumCircuit:
        """Build the QPE circuit (without state preparation).

        Qubits 0..num_counting_qubits-1 are the counting register.
        Qubits num_counting_qubits..num_counting_qubits+n-1 hold the eigenstate.
        """
        n_u = self._unitary.num_qubits
        n_c = self._num_counting_qubits
        total = n_c + n_u
        qc = QuantumCircuit(total)

        for i in range(n_c):
            qc.h(i)

        for k in range(n_c):
            power = 2**k
            u_pow = _u_power(self._unitary, power)
            controlled = _controlled_u(
                u_pow, total, control=k, targets=list(range(n_c, total))
            )
            qc.append(controlled, list(range(total)))

        iqft = inverse_qft_circuit(n_c, do_swaps=False)
        for op, targets in iqft.gates:
            qc.append(op, targets)

        return qc

    def estimate(self) -> PhaseEstimationResult:
        """Run QPE assuming the eigenstate |1> of the target register."""
        n_u = self._unitary.num_qubits
        eigenstate = StateVector(n_u)
        eigenstate._amplitudes = np.zeros(2**n_u, dtype=np.complex128)
        if 1 < 2**n_u:
            eigenstate._amplitudes[1] = 1.0
        else:
            eigenstate._amplitudes[0] = 1.0
        return self.estimate_from_state(eigenstate)

    def estimate_from_state(
        self, eigenstate: StateVector
    ) -> PhaseEstimationResult:
        """Run QPE with a provided eigenstate."""
        n_u = eigenstate.num_qubits
        n_c = self._num_counting_qubits
        total = n_c + n_u

        if n_u != self._unitary.num_qubits:
            raise ValueError(
                f"Eigenstate has {n_u} qubits but unitary acts on {self._unitary.num_qubits}"
            )

        dim = 2**total
        amps = np.zeros(dim, dtype=np.complex128)
        eigen_amps = eigenstate.amplitudes
        for j in range(2**n_u):
            amps[j] = eigen_amps[j]

        initial = StateVector(total, amplitudes=amps)
        qc = self.build_circuit()
        final_state = qc.run(initial)

        probs = np.abs(final_state.amplitudes) ** 2

        best_idx = int(np.argmax(probs))
        phase_val = 0.0
        for k in range(n_c):
            idx_bit = total - 1 - k
            if best_idx & (1 << idx_bit):
                phase_val += 1.0 / (2 ** (k + 1))

        phase_radians = phase_val * 2 * np.pi
        eigenvalue = np.exp(2j * np.pi * phase_val)
        success_prob = float(probs[best_idx])

        return PhaseEstimationResult(
            phase=phase_val,
            phase_radians=phase_radians,
            eigenvalue=eigenvalue,
            num_counting_qubits=n_c,
            success_probability=success_prob,
        )

    def __repr__(self) -> str:
        return (
            f"PhaseEstimation(num_counting_qubits={self._num_counting_qubits}, "
            f"unitary_qubits={self._unitary.num_qubits})"
        )
