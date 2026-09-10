"""Quantum Amplitude Estimation algorithm.

Estimates the amplitude 'a' of a state |psi> = sqrt(a)|good> + sqrt(1-a)|bad>
using quantum phase estimation on Grover's operator.

Applications:
- Quantum Monte Carlo integration
- Risk analysis in quantum finance
- Machine learning kernel methods

Reference: Brassard et al., "Quantum Amplitude Amplification and Estimation" (2000)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..core.state import StateVector
from ..core.tensor import expand_operator


@dataclass
class AmplitudeEstimationResult:
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
            op: Operator = gate_instr[0]  # type: ignore[assignment]
            targets: list[int] = gate_instr[1]  # type: ignore[assignment]
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

        # Inverse QFT on evaluation qubits (embedded in total qubit space)
        iqft = self._inverse_qft(m)
        for instr in iqft._gate_instructions:
            if QuantumCircuit._is_parameterized_gate(instr):
                continue
            iqft_op: Operator = instr[0]  # type: ignore[assignment]
            iqft_targets: list[int] = instr[1]  # type: ignore[assignment]
            # Shift target qubits to evaluation qubit positions
            shifted = [t for t in iqft_targets]
            expanded = expand_operator(iqft_op, shifted, total)
            qc.append(expanded, list(range(total)))

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

        Applies unitary to target_qubits when control_qubit is |1>.
        """
        dim = 2**total_qubits
        controlled = np.eye(dim, dtype=np.complex128)

        # Extract the sub-matrix for the control=|1> block
        ctrl_dim = 2**total_qubits
        n_targets = len(target_qubits)

        for i in range(dim):
            if not (i >> (total_qubits - 1 - control_qubit)) & 1:
                continue  # Control is 0, identity
            # Apply unitary to target qubits
            target_indices = []
            for t in target_qubits:
                bit = (i >> (total_qubits - 1 - t)) & 1
                target_indices.append(bit)

            # Build target state index
            new_target = 0
            for j, bit in enumerate(target_indices):
                new_target |= bit << (n_targets - 1 - j)

            # Map full state
            new_i = i
            for j, t in enumerate(target_qubits):
                old_bit = (i >> (total_qubits - 1 - t)) & 1
                new_bit = (new_target >> (n_targets - 1 - j)) & 1
                if old_bit != new_bit:
                    new_i ^= 1 << (total_qubits - 1 - t)

            # Only set non-diagonal elements if not identity
            if new_i != i:
                # Build the column for this row
                pass

        # Simpler approach: construct full matrix element by element
        result = np.eye(dim, dtype=np.complex128)
        for row in range(dim):
            if not (row >> (total_qubits - 1 - control_qubit)) & 1:
                continue  # Control=0: identity
            for col in range(dim):
                if col == row:
                    continue
                if not (col >> (total_qubits - 1 - control_qubit)) & 1:
                    continue  # Control=0 column

                # Check if target qubits match the transformation
                row_targets = []
                col_targets = []
                for t in target_qubits:
                    row_targets.append((row >> (total_qubits - 1 - t)) & 1)
                    col_targets.append((col >> (total_qubits - 1 - t)) & 1)

                # Check non-target qubits match
                row_other = row
                col_other = col
                for t in target_qubits:
                    mask = 1 << (total_qubits - 1 - t)
                    row_other &= ~mask
                    col_other &= ~mask

                if row_other != col_other:
                    continue

                # Compute unitary matrix element
                row_t_idx = 0
                col_t_idx = 0
                for j, t in enumerate(target_qubits):
                    row_t_idx |= row_targets[j] << (n_targets - 1 - j)
                    col_t_idx |= col_targets[j] << (n_targets - 1 - j)

                val = unitary[row_t_idx, col_t_idx]
                if val != 0:
                    result[row, col] = val

        return result

    @staticmethod
    def _inverse_qft(num_qubits: int) -> QuantumCircuit:
        """Build inverse QFT circuit."""
        qc = QuantumCircuit(num_qubits)

        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - 1 - i)

        for i in range(num_qubits):
            for j in range(i):
                # Controlled phase gate: Rz(-pi / 2^(i-j))
                angle = -math.pi / (2 ** (i - j))
                # Apply controlled-Rz using decomposition
                qc.cx(j, i)
                qc.rz(angle / 2, i)
                qc.cx(j, i)
                qc.rz(-angle / 2, j)
                qc.rz(angle / 2, i)
            qc.h(i)

        return qc

    def __repr__(self) -> str:
        return (
            f"AmplitudeEstimation(eval_qubits={self._num_eval_qubits})"
        )
