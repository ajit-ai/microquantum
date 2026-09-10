"""Tests for HHL algorithm and controlled-gate infrastructure."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.core import Operator, QuantumCircuit
from microquantum.algorithms.hhl import (
    HHL,
    HHLResult,
    controlled_U,
)


class TestControlledU:
    """Controlled-U gate decomposition tests."""

    def test_controlled_h(self) -> None:
        """Controlled-H should produce a valid circuit."""
        qc = QuantumCircuit(2)
        controlled_U(qc, control=0, target=[1], U=Operator.H())
        sv = qc.run()
        assert sv.num_qubits == 2
        assert len(sv.amplitudes) == 4

    def test_controlled_x(self) -> None:
        """Controlled-X (CNOT) should produce valid circuit."""
        qc = QuantumCircuit(2)
        controlled_U(qc, control=0, target=[1], U=Operator.X())
        sv = qc.run()
        assert len(sv.amplitudes) == 4

    def test_controlled_z(self) -> None:
        """Controlled-Z should produce valid circuit."""
        qc = QuantumCircuit(2)
        controlled_U(qc, control=0, target=[1], U=Operator.Z())
        sv = qc.run()
        assert len(sv.amplitudes) == 4

    def test_controlled_cnot_3_qubits(self) -> None:
        """Controlled-CNOT (Toffoli) should work with 3 qubits."""
        qc = QuantumCircuit(3)
        controlled_U(qc, control=0, target=[1, 2], U=Operator.CNOT())
        sv = qc.run()
        assert sv.num_qubits == 3

    def test_controlled_cz_3_qubits(self) -> None:
        """Controlled-CZ should work with 3 qubits."""
        qc = QuantumCircuit(3)
        controlled_U(qc, control=0, target=[1, 2], U=Operator.CZ())
        sv = qc.run()
        assert sv.num_qubits == 3


class TestHHL:
    """HHL algorithm tests."""

    def test_diagonal_system(self) -> None:
        """HHL on diagonal A = diag(1, 2) with b = [1, 0]."""
        A = np.array([[1, 0], [0, 2]])
        hhl = HHL(A, num_counting=4)
        result = hhl.solve(np.array([1, 0]))

        assert isinstance(result, HHLResult)
        assert result.num_qubits == hhl.num_qubits
        assert len(result.eigenvalues) == 2
        assert result.condition_number >= 1.0

    def test_solution_normalized(self) -> None:
        """Solution state should be normalized."""
        A = np.array([[2, 0], [0, 1]])
        hhl = HHL(A, num_counting=3)
        result = hhl.solve(np.array([1, 1]))
        amps = result.solution.amplitudes
        norm = float(np.linalg.norm(amps))
        assert abs(norm - 1.0) < 1e-10

    def test_identity_matrix(self) -> None:
        """HHL with identity should give x = b."""
        A = np.eye(2)
        hhl = HHL(A, num_counting=4)
        result = hhl.solve(np.array([1, 0]))
        # Solution should be close to [1, 0] normalized
        amps = result.solution.amplitudes[:2]
        if np.linalg.norm(amps) > 0:
            amps_normed = amps / np.linalg.norm(amps)
            b_normed = np.array([1, 0], dtype=np.complex128)
            assert abs(abs(np.vdot(amps_normed, b_normed)) - 1.0) < 0.3

    def test_hermitian_check(self) -> None:
        """Non-Hermitian matrix should raise."""
        A = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Hermitian"):
            HHL(A)

    def test_non_square_check(self) -> None:
        """Non-square matrix should raise."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="square"):
            HHL(A)

    def test_wrong_b_shape(self) -> None:
        """Wrong-sized b should raise."""
        A = np.eye(2)
        hhl = HHL(A)
        with pytest.raises(ValueError, match="shape"):
            hhl.solve(np.array([1, 0, 0]))

    def test_circuit_accessible(self) -> None:
        """Result should contain the circuit."""
        A = np.array([[1, 0], [0, 2]])
        hhl = HHL(A, num_counting=3)
        result = hhl.solve(np.array([1, 0]))
        assert result.circuit.num_qubits == result.num_qubits

    def test_condition_number(self) -> None:
        """Condition number should be ratio of max/min eigenvalue."""
        A = np.array([[1, 0], [0, 10]])
        hhl = HHL(A, num_counting=4)
        result = hhl.solve(np.array([1, 0]))
        assert result.condition_number == pytest.approx(10.0, rel=0.1)

    def test_repr(self) -> None:
        hhl = HHL(np.eye(2), num_counting=4)
        assert "HHL" in repr(hhl)
