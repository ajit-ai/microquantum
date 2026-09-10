"""Tests for DensityMatrix class."""

import numpy as np
import pytest

from microquantum.core.density_matrix import DensityMatrix
from microquantum.core.state import StateVector


class TestDensityMatrixCreation:
    def test_default_ground_state(self) -> None:
        rho = DensityMatrix(1)
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        np.testing.assert_allclose(rho.matrix, expected)
        assert rho.trace == 1.0

    def test_two_qubits_ground(self) -> None:
        rho = DensityMatrix(2)
        assert rho.dim == 4
        assert rho.matrix[0, 0] == 1.0
        assert rho.trace == 1.0

    def test_custom_matrix(self) -> None:
        mat = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        rho = DensityMatrix(1, matrix=mat)
        np.testing.assert_allclose(rho.matrix, mat)

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            DensityMatrix(1, matrix=np.eye(4))

    def test_zero_qubits_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            DensityMatrix(0)


class TestDensityMatrixFromStatevector:
    def test_from_ket0(self) -> None:
        sv = StateVector(1)
        rho = DensityMatrix.from_statevector(sv)
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        np.testing.assert_allclose(rho.matrix, expected, atol=1e-12)

    def test_from_superposition(self) -> None:
        sv = StateVector(1, amplitudes=np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        rho = DensityMatrix.from_statevector(sv)
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        np.testing.assert_allclose(rho.matrix, expected, atol=1e-12)

    def test_from_bell_state(self) -> None:
        amps = np.zeros(4, dtype=np.complex128)
        amps[0] = 1 / np.sqrt(2)
        amps[3] = 1 / np.sqrt(2)
        sv = StateVector(2, amplitudes=amps)
        rho = DensityMatrix.from_statevector(sv)
        assert rho.trace == pytest.approx(1.0)
        assert rho.is_pure


class TestDensityMatrixFromLabel:
    def test_from_label_0(self) -> None:
        rho = DensityMatrix.from_label("0")
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        np.testing.assert_allclose(rho.matrix, expected)

    def test_from_label_1(self) -> None:
        rho = DensityMatrix.from_label("1")
        expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        np.testing.assert_allclose(rho.matrix, expected)

    def test_from_label_10(self) -> None:
        rho = DensityMatrix.from_label("10")
        assert rho.num_qubits == 2
        assert rho.matrix[2, 2] == 1.0


class TestDensityMatrixProperties:
    def test_is_pure_from_statevector(self) -> None:
        sv = StateVector(1)
        rho = DensityMatrix.from_statevector(sv)
        assert rho.is_pure

    def test_mixed_state_not_pure(self) -> None:
        mat = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
        rho = DensityMatrix(1, matrix=mat)
        assert not rho.is_pure

    def test_trace(self) -> None:
        rho = DensityMatrix(2)
        assert rho.trace == 1.0


class TestDensityMatrixOperations:
    def test_apply_unitary_x(self) -> None:
        rho = DensityMatrix.from_label("0")
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        rho2 = rho.apply_unitary(X)
        assert rho2.matrix[1, 1] == pytest.approx(1.0)
        assert rho2.matrix[0, 0] == pytest.approx(0.0)

    def test_apply_unitary_h(self) -> None:
        rho = DensityMatrix.from_label("0")
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        rho2 = rho.apply_unitary(H)
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        np.testing.assert_allclose(rho2.matrix, expected, atol=1e-12)

    def test_unitary_preserves_trace(self) -> None:
        sv = StateVector(1, amplitudes=np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        rho = DensityMatrix.from_statevector(sv)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        rho2 = rho.apply_unitary(X)
        assert rho2.trace == pytest.approx(1.0)

    def test_expectation_z_ket0(self) -> None:
        rho = DensityMatrix.from_label("0")
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        exp_val = rho.expectation(Z)
        assert exp_val == pytest.approx(1.0)

    def test_expectation_z_ket1(self) -> None:
        rho = DensityMatrix.from_label("1")
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        exp_val = rho.expectation(Z)
        assert exp_val == pytest.approx(-1.0)


class TestDensityMatrixPartialTrace:
    def test_partial_trace_single_qubit(self) -> None:
        rho_full = DensityMatrix.from_label("00")
        rho_reduced = rho_full.partial_trace([1])
        assert rho_reduced.num_qubits == 1
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        np.testing.assert_allclose(rho_reduced.matrix, expected, atol=1e-12)

    def test_partial_trace_preserves_trace(self) -> None:
        sv = StateVector(2)
        sv._amplitudes = np.array([1, 0, 0, 0], dtype=np.complex128)
        rho = DensityMatrix.from_statevector(sv)
        rho_reduced = rho.partial_trace([0])
        assert rho_reduced.trace == pytest.approx(1.0)


class TestDensityMatrixKraus:
    def test_identity_kraus(self) -> None:
        rho = DensityMatrix.from_label("0")
        I = np.eye(2, dtype=np.complex128)
        rho2 = rho.apply_kraus([I])
        np.testing.assert_allclose(rho2.matrix, rho.matrix, atol=1e-12)

    def test_bit_flip_channel(self) -> None:
        rho = DensityMatrix.from_label("0")
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        I = np.eye(2, dtype=np.complex128)
        # p=0: no flip
        e0 = I
        e1 = np.zeros((2, 2), dtype=np.complex128)
        rho2 = rho.apply_kraus([e0, e1])
        assert rho2.matrix[0, 0] == pytest.approx(1.0)


class TestDensityMatrixCopy:
    def test_copy_independence(self) -> None:
        rho = DensityMatrix.from_label("0")
        rho2 = rho.copy()
        rho2._matrix[0, 0] = 0.5
        assert rho.matrix[0, 0] == 1.0

    def test_copy_preserves_state(self) -> None:
        rho = DensityMatrix.from_label("1")
        rho2 = rho.copy()
        np.testing.assert_allclose(rho2.matrix, rho.matrix)


class TestDensityMatrixRepr:
    def test_repr_pure(self) -> None:
        rho = DensityMatrix.from_label("0")
        r = repr(rho)
        assert "pure" in r

    def test_repr_mixed(self) -> None:
        mat = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
        rho = DensityMatrix(1, matrix=mat)
        r = repr(rho)
        assert "mixed" in r

    def test_str(self) -> None:
        rho = DensityMatrix(1)
        s = str(rho)
        assert "1 qubits" in s
