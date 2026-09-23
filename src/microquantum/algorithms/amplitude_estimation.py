"""Quantum Amplitude Estimation algorithm.

Estimates the amplitude 'a' of a state |psi> = sqrt(a)|good> + sqrt(1-a)|bad>
using quantum phase estimation on Grover's operator.

Applications:
- Quantum Monte Carlo integration
- Estimation of expectation values / probabilities
- Machine learning kernel methods

Reference: Brassard et al., "Quantum Amplitude Amplification and Estimation" (2000)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit, _narrow_concrete
from ..core.operators import Operator
from ..core.tensor import expand_operator
from .qft import inverse_qft_circuit


@dataclass
class AmplitudeEstimationResult(JSONSerializable):
    """Result container for amplitude estimation.

    Attributes:
        estimated_amplitude: Estimated amplitude a.
        confidence_interval: Half-width of the confidence interval.
        num_evaluations: Number of times the Grover operator was applied.
        phases: All estimated phases (for debugging).
    """

    estimated_amplitude: float = 0.0
    confidence_interval: float = 0.0
    num_evaluations: int = 0
    phases: list[float] = field(default_factory=list)


class AmplitudeEstimation:
    """Quantum Amplitude Estimation.

    Uses Quantum Phase Estimation (QPE) on the Grover operator Q = -A S_0 A^-1 S_chi
    to estimate the amplitude of a marked state.

    The algorithm works as follows:
    1. Construct the Grover operator Q from the oracle and state preparation
    2. Apply QPE to estimate eigenvalues of Q
    3. Convert eigenvalues to amplitude estimates

    Args:
        num_evaluation_qubits: Number of qubits for phase estimation (precision).
        state_preparation: Circuit preparing the state A|0>.
        oracle: Circuit that marks the "good" states (S_chi).
    """

    def __init__(
        self,
        num_evaluation_qubits: int = 3,
        state_preparation: Optional[QuantumCircuit] = None,
        oracle: Optional[QuantumCircuit] = None,
    ) -> None:
        self._num_eval_qubits = num_evaluation_qubits
        self._state_prep = state_preparation
        self._oracle = oracle

    @property
    def num_evaluation_qubits(self) -> int:
        return self._num_eval_qubits

    def build_grover_operator(
        self,
        num_qubits: int,
        state_prep: Optional[QuantumCircuit] = None,
        oracle: Optional[QuantumCircuit] = None,
    ) -> QuantumCircuit:
        """Build the Grover operator Q.

        Q = -A S_0 A^-1 S_chi

        Where:
        - A: state preparation circuit
        - A^-1: inverse of state preparation
        - S_chi: oracle (phase flip on marked states)
        - S_0: phase flip on |0> state

        Args:
            num_qubits: Number of data qubits.
            state_prep: State preparation circuit (overrides constructor arg).
            oracle: Oracle circuit (overrides constructor arg).

        Returns:
            Circuit implementing the Grover operator.
        """
        sp = state_prep or self._state_prep
        orc = oracle or self._oracle

        if sp is None or orc is None:
            raise ValueError(
                "state_preparation and oracle are required "
                "(pass to constructor or build_grover_operator)"
            )

        qc = QuantumCircuit(num_qubits)

        # -A S_0 A^-1 S_chi
        # S_chi (oracle)
        qc = qc + orc

        # A^-1
        qc = qc + sp.inverse()

        # S_0: phase flip on |0> = (2|0><0| - I)
        dim = 2**num_qubits
        s0_matrix = np.eye(dim, dtype=np.complex128)
        s0_matrix[0, 0] = -1.0
        qc.append(Operator(s0_matrix), list(range(num_qubits)))

        # A
        qc = qc + sp

        # Global phase -1 (negate all amplitudes)
        qc.append(
            Operator(-np.eye(dim, dtype=np.complex128)),
            list(range(num_qubits)),
        )

        return qc

    def estimate(
        self,
        num_qubits: int,
        state_prep: Optional[QuantumCircuit] = None,
        oracle: Optional[QuantumCircuit] = None,
        num_shots: int = 1024,
    ) -> AmplitudeEstimationResult:
        """Run amplitude estimation.

        Args:
            num_qubits: Number of data qubits.
            state_prep: State preparation circuit.
            oracle: Oracle circuit.
            num_shots: Number of measurement shots.

        Returns:
            AmplitudeEstimationResult with the estimated amplitude.
        """
        grover = self.build_grover_operator(num_qubits, state_prep, oracle)
        sp = state_prep or self._state_prep
        assert sp is not None

        # Build full QPE circuit
        m = self._num_eval_qubits
        n = num_qubits
        total = m + n

        qc = QuantumCircuit(total)

        # Prepare target state on data qubits
        for gate_instr in sp._gate_instructions:
            if QuantumCircuit._is_parameterized_gate(gate_instr):
                continue
            op, targets = _narrow_concrete(gate_instr)
            expanded = expand_operator(op, targets, n)
            qc.append(expanded, list(range(m, total)))

        # Apply Hadamards to evaluation qubits
        for i in range(m):
            qc.h(i)

        # Controlled Grover operator powers
        grover_op = Operator(grover.get_unitary().matrix)
        for i in range(m):
            power = 2**i
            powered = self._matrix_power(grover_op.matrix, power)
            # Control on qubit i, target on data qubits
            controlled = self._controlled_unitary(powered, i, list(range(m, total)), total)
            qc.append(Operator(controlled), list(range(total)))

        # Inverse QFT on evaluation qubits (canonical no-swap form,
        # mirroring PhaseEstimation so bit order matches the readout).
        iqft = inverse_qft_circuit(m, do_swaps=False)
        for iqft_op, iqft_targets in iqft.gates:
            qc.append(iqft_op, list(iqft_targets))

        # Run and get phase estimate
        state = qc.run()
        probs = abs(state.amplitudes) ** 2

        # Find the most probable measurement outcome
        measurement = max(range(len(probs)), key=lambda i: probs[i])

        # Extract phase from evaluation qubits
        phase_bits = (measurement >> (n)) & ((1 << m) - 1)
        phase = phase_bits / (2**m)

        # Convert phase to amplitude: theta = 2*arcsin(sqrt(a))
        # a = sin^2(pi * phase)
        a = math.sin(math.pi * phase) ** 2

        # Confidence interval ~ 1/sqrt(num_evaluations)
        num_evals = 2**m
        ci = math.pi / (2 * math.sqrt(num_evals))

        return AmplitudeEstimationResult(
            estimated_amplitude=a,
            confidence_interval=ci,
            num_evaluations=num_evals,
            phases=[phase],
        )

    @staticmethod
    def _matrix_power(matrix: np.ndarray, power: int) -> np.ndarray:
        """Compute matrix^power using repeated squaring."""
        if power == 0:
            return np.eye(matrix.shape[0], dtype=np.complex128)
        if power == 1:
            return matrix.copy()

        result = np.eye(matrix.shape[0], dtype=np.complex128)
        base = matrix.copy()
        while power > 0:
            if power % 2 == 1:
                result = result @ base
            base = base @ base
            power //= 2
        return result

    @staticmethod
    def _controlled_unitary(
        unitary: np.ndarray,
        control_qubit: int,
        target_qubits: list[int],
        total_qubits: int,
    ) -> np.ndarray:
        """Create a controlled unitary gate.

        Applies unitary to target_qubits when control_qubit is |1>
        (big-endian qubit ordering: qubit 0 is the most significant
        bit), identity otherwise.
        """
        matrix = np.asarray(unitary, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("unitary must be a square matrix")
        if matrix.shape[0] != 2 ** len(target_qubits):
            raise ValueError("unitary dimension must match the target qubit count")
        if not 0 <= control_qubit < total_qubits:
            raise ValueError("control_qubit out of range")
        if any(not 0 <= target < total_qubits for target in target_qubits):
            raise ValueError("target qubit out of range")
        if control_qubit in target_qubits:
            raise ValueError("control qubit must not be a target")
        dim = 2**total_qubits
        n_targets = len(target_qubits)
        result = np.eye(dim, dtype=np.complex128)
        for row in range(dim):
            if not (row >> (total_qubits - 1 - control_qubit)) & 1:
                continue  # Control=0: identity
            row_targets = [(row >> (total_qubits - 1 - t)) & 1 for t in target_qubits]
            row_other = row
            for t in target_qubits:
                row_other &= ~(1 << (total_qubits - 1 - t))
            for col in range(dim):
                if not (col >> (total_qubits - 1 - control_qubit)) & 1:
                    continue  # Control=0 column
                col_other = col
                for t in target_qubits:
                    col_other &= ~(1 << (total_qubits - 1 - t))
                if row_other != col_other:
                    continue
                col_targets = [(col >> (total_qubits - 1 - t)) & 1 for t in target_qubits]
                row_t_idx = 0
                col_t_idx = 0
                for j in range(n_targets):
                    row_t_idx |= row_targets[j] << (n_targets - 1 - j)
                    col_t_idx |= col_targets[j] << (n_targets - 1 - j)
                result[row, col] = matrix[row_t_idx, col_t_idx]
        return result

    def __repr__(self) -> str:
        return (
            f"AmplitudeEstimation(eval_qubits={self._num_eval_qubits})"
        )
