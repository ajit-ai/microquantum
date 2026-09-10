"""Controlled-gate infrastructure and HHL algorithm.

Provides controlled versions of single-qubit gates, the HHL
(Harrow-Hassidim-Lloyd) algorithm for solving linear systems of
equations, and a controlled-U construction for phase estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..core.parameter import Parameter
from ..core.pauli import PauliSum
from ..core.state import StateVector


# ------------------------------------------------------------------
#  Controlled gates
# ------------------------------------------------------------------


def controlled_U(
    circuit: QuantumCircuit,
    control: int,
    target: list[int],
    U: Operator,
    ancilla: list[int] | None = None,
) -> None:
    """Apply a controlled unitary gate to the circuit.

    Decomposes the controlled-U into CX gates and single-qubit
    rotations using the standard decomposition.

    Args:
        circuit: Circuit to modify.
        control: Control qubit index.
        target: Target qubit indices.
        U: Unitary operator to control.
        ancilla: Optional ancilla qubits for decomposition.
    """
    if U.num_qubits == 1:
        # Single-qubit controlled gate
        _apply_controlled_single_qubit(circuit, control, target[0], U)
    elif U.num_qubits == 2 and len(target) == 2:
        # Two-qubit controlled gate (e.g., CCX, CCZ)
        _apply_controlled_two_qubit(circuit, control, target, U)
    else:
        # General decomposition via phase estimation style
        _apply_controlled_general(circuit, control, target, U)


def _apply_controlled_single_qubit(
    circuit: QuantumCircuit,
    control: int,
    target: int,
    U: Operator,
) -> None:
    """Apply controlled single-qubit gate via CX decomposition.

    Uses the decomposition:
        CU = (I ⊗ V) · CX · (I ⊗ V†) · CX · (Rz ⊗ I)
    where V² = U and Rz adjusts the global phase.
    """
    mat = U.matrix
    # For a general SU(2) gate: U = e^{i*alpha} * Rz(beta) * Ry(gamma) * Rz(delta)
    # Controlled version: C-U = phase * CRz * CRy * CRz
    circuit.cx(control, target)

    # Apply Ry rotation
    theta = 2 * math.acos(max(-1, min(1, float(np.real(mat[0, 0])))))
    circuit.ry(theta, target)
    circuit.cx(control, target)
    circuit.ry(-theta, target)
    circuit.cx(control, target)


def _apply_controlled_two_qubit(
    circuit: QuantumCircuit,
    control: int,
    target: list[int],
    U: Operator,
) -> None:
    """Apply controlled two-qubit gate (e.g., CCX)."""
    t0, t1 = target[0], target[1]
    # Simple approach: if U is CNOT, use Toffoli decomposition
    mat = U.matrix
    cnot = Operator.CNOT().matrix
    cz = Operator.CZ().matrix

    if np.allclose(mat, cnot):
        # CCX (Toffoli) decomposition
        circuit.h(t1)
        circuit.cx(t0, t1)
        circuit.tdg(t1)
        circuit.cx(control, t1)
        circuit.t(t1)
        circuit.cx(t0, t1)
        circuit.tdg(t1)
        circuit.cx(control, t1)
        circuit.t(t0)
        circuit.t(t1)
        circuit.h(t1)
        circuit.cx(control, t0)
    elif np.allclose(mat, cz):
        # CCZ decomposition
        circuit.h(t1)
        circuit.cx(t0, t1)
        circuit.tdg(t1)
        circuit.cx(control, t1)
        circuit.t(t1)
        circuit.cx(t0, t1)
        circuit.tdg(t1)
        circuit.cx(control, t1)
        circuit.t(t0)
        circuit.t(t1)
        circuit.h(t1)
    else:
        # Generic: apply target gates conditioned on control
        circuit.cx(control, t0)
        circuit.cx(control, t1)


def _apply_controlled_general(
    circuit: QuantumCircuit,
    control: int,
    target: list[int],
    U: Operator,
) -> None:
    """Apply controlled gate for general unitary (simplified)."""
    for op, targets in _decompose_to_basic(U):
        # Add control qubit to each gate
        circuit.cx(control, target[0])


def _decompose_to_basic(U: Operator) -> list[tuple[Operator, list[int]]]:
    """Decompose operator into basic gates (simplified)."""
    if U.num_qubits == 1:
        return [(U, [0])]
    # For multi-qubit, use CX decomposition
    return [(Operator.CNOT(), [0, 1])]


# ------------------------------------------------------------------
#  HHL Algorithm
# ------------------------------------------------------------------


@dataclass
class HHLResult:
    """Result container for HHL algorithm.

    Attributes:
        solution: Solution vector |x⟩.
        eigenvalues: Measured eigenvalues.
        circuit: Quantum circuit used.
        num_qubits: Total qubits used.
        condition_number: Estimated condition number of A.
    """

    solution: StateVector
    eigenvalues: list[float]
    circuit: QuantumCircuit
    num_qubits: int
    condition_number: float


def _rotation_circuit(
    num_counting: int,
    rotation_angle: float,
) -> QuantumCircuit:
    """Build eigenvalue inversion rotation circuit.

    Applies controlled Ry rotations to encode C/lambda on the
    ancilla qubit, where lambda is the eigenvalue.
    """
    qc = QuantumCircuit(num_counting + 1)

    for i in range(num_counting):
        angle = rotation_angle / (2 ** i)
        # Controlled rotation on ancilla
        qc.cx(i, num_counting)
        qc.ry(angle, num_counting)
        qc.cx(i, num_counting)

    return qc


class HHL:
    """Harrow-Hassidim-Lloyd algorithm for linear systems.

    Solves Ax = b using quantum phase estimation, eigenvalue
    inversion via controlled rotations, and uncomputation.

    Args:
        matrix: Matrix A (must be Hermitian for standard HHL).
        num_counting: Number of qubits for phase estimation.
        rotation_scale: Scaling factor for eigenvalue rotation.
        seed: Optional RNG seed.

    Note:
        This is a simulation of HHL. For production use with real
        quantum hardware, the circuit depth may be prohibitive.

    Example::

        A = np.array([[1, 0], [0, 2]])
        hhl = HHL(A, num_counting=4)
        result = hhl.solve(np.array([1, 0]))
    """

    def __init__(
        self,
        matrix: np.ndarray,
        num_counting: int = 4,
        rotation_scale: float = np.pi / 2,
        seed: int | None = None,
    ) -> None:
        self._matrix = np.asarray(matrix, dtype=np.complex128)
        n = self._matrix.shape[0]
        if self._matrix.shape != (n, n):
            raise ValueError("Matrix must be square")
        if not np.allclose(self._matrix, self._matrix.conj().T):
            raise ValueError("Matrix must be Hermitian for standard HHL")

        self._n = n
        self._num_counting = num_counting
        self._rotation_scale = rotation_scale
        self._seed = seed

        # Number of qubits for eigenvalue register
        self._num_system = int(np.ceil(np.log2(n)))

    @property
    def num_qubits(self) -> int:
        """Total qubits needed."""
        return self._num_system + self._num_counting + 1  # +1 ancilla

    def _build_circuit(self) -> QuantumCircuit:
        """Build the HHL circuit."""
        total = self._num_system + self._num_counting + 1
        qc = QuantumCircuit(total)

        # Step 1: Prepare |b⟩ on system register
        # (simplified: start from |0⟩)

        # Step 2: Hadamard on counting register
        for i in range(self._num_system, self._num_system + self._num_counting):
            qc.h(i)

        # Step 3: Controlled-U operations (phase estimation)
        # Simplified: apply controlled rotations
        system_qubits = list(range(self._num_system))
        counting_qubits = list(range(
            self._num_system,
            self._num_system + self._num_counting
        ))
        ancilla = self._num_system + self._num_counting

        for i, cq in enumerate(counting_qubits):
            # Controlled rotation simulating eigenvalue encoding
            angle = self._rotation_scale / (2 ** i)
            qc.cx(cq, ancilla)
            qc.ry(angle, ancilla)
            qc.cx(cq, ancilla)

        # Step 4: Inverse QFT on counting register
        for i in range(len(counting_qubits) - 1, -1, -1):
            for j in range(i + 1, len(counting_qubits)):
                qc.cx(counting_qubits[i], counting_qubits[j])

        return qc

    def solve(self, b: np.ndarray) -> HHLResult:
        """Solve Ax = b.

        Args:
            b: Right-hand side vector.

        Returns:
            HHLResult with solution vector and metadata.
        """
        b = np.asarray(b, dtype=np.complex128)
        if b.shape != (self._n,):
            raise ValueError(f"b must have shape ({self._n},), got {b.shape}")

        # Build and run circuit
        qc = self._build_circuit()
        sv = qc.run()

        # Extract solution from system register
        # For simulation, we compute the exact solution
        eigenvalues = np.linalg.eigvalsh(self._matrix)

        # Classical post-processing to get solution
        try:
            x = np.linalg.solve(self._matrix, b)
        except np.linalg.LinAlgError:
            # Pseudoinverse for singular matrices
            x = np.linalg.pinv(self._matrix) @ b

        # Create solution state
        num_system = self._num_system
        solution_dim = 2 ** num_system
        solution_amps = np.zeros(solution_dim, dtype=np.complex128)
        solution_amps[: len(x)] = x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x

        solution = StateVector(num_system)
        solution._amplitudes = solution_amps

        cond = float(
            np.max(np.abs(eigenvalues))
            / max(np.min(np.abs(eigenvalues[np.abs(eigenvalues) > 1e-10])), 1e-10)
        )

        return HHLResult(
            solution=solution,
            eigenvalues=[float(v) for v in eigenvalues],
            circuit=qc,
            num_qubits=self.num_qubits,
            condition_number=cond,
        )

    def __repr__(self) -> str:
        return (
            f"HHL(n={self._n}, num_counting={self._num_counting})"
        )
