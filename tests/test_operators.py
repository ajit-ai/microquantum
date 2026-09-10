"""Unit tests for microquantum.core.operators.Operator."""


import numpy as np
import pytest

from microquantum.core import Operator, StateVector

# ---------------------------------------------------------------------------
# Unitarity tests for all standard gates
# ---------------------------------------------------------------------------

class TestUnitarity:
    """Verify U^dag @ U = I for all standard gates."""

    @pytest.mark.parametrize(
        "gate",
        [
            Operator.I(),
            Operator.X(),
            Operator.Y(),
            Operator.Z(),
            Operator.H(),
            Operator.S(),
            Operator.T(),
            Operator.Rx(0.5),
            Operator.Ry(1.2),
            Operator.Rz(0.3),
            Operator.CNOT(),
            Operator.CZ(),
            Operator.SWAP(),
        ],
        ids=["I", "X", "Y", "Z", "H", "S", "T", "Rx", "Ry", "Rz",
             "CNOT", "CZ", "SWAP"],
    )
    def test_unitarity(self, gate: Operator) -> None:
        product = gate.dag @ gate
        expected = Operator(np.eye(gate.matrix.shape[0], dtype=np.complex128))
        np.testing.assert_array_almost_equal(
            product.matrix, expected.matrix, decimal=7
        )


# ---------------------------------------------------------------------------
# Identity algebra: H^2 = I, X^2 = I, Z^2 = I
# ---------------------------------------------------------------------------

class TestIdentityAlgebra:
    """Verify H^2 = I, X^2 = I, Z^2 = I."""

    def test_h_squared_is_identity(self) -> None:
        H2 = Operator.H() @ Operator.H()
        expected = Operator.I()
        np.testing.assert_array_almost_equal(
            H2.matrix, expected.matrix, decimal=7
        )

    def test_x_squared_is_identity(self) -> None:
        X2 = Operator.X() @ Operator.X()
        expected = Operator.I()
        np.testing.assert_array_almost_equal(
            X2.matrix, expected.matrix, decimal=7
        )

    def test_z_squared_is_identity(self) -> None:
        Z2 = Operator.Z() @ Operator.Z()
        expected = Operator.I()
        np.testing.assert_array_almost_equal(
            Z2.matrix, expected.matrix, decimal=7
        )

    def test_y_squared_is_identity(self) -> None:
        Y2 = Operator.Y() @ Operator.Y()
        expected = Operator.I()
        np.testing.assert_array_almost_equal(
            Y2.matrix, expected.matrix, decimal=7
        )

    def test_s_squared_is_z(self) -> None:
        S2 = Operator.S() @ Operator.S()
        expected = Operator.Z()
        np.testing.assert_array_almost_equal(
            S2.matrix, expected.matrix, decimal=7
        )

    def test_cnot_squared_is_identity(self) -> None:
        CNOT2 = Operator.CNOT() @ Operator.CNOT()
        expected = Operator(np.eye(4, dtype=np.complex128))
        np.testing.assert_array_almost_equal(
            CNOT2.matrix, expected.matrix, decimal=7
        )


# ---------------------------------------------------------------------------
# Operator arithmetic dunder tests
# ---------------------------------------------------------------------------

class TestOperatorArithmetic:
    """Test dunder methods: @, *, +."""

    def test_hxh_equals_z(self) -> None:
        """HXH = Z (Hadamard basis change identity)."""
        hxh = Operator.H() @ Operator.X() @ Operator.H()
        expected = Operator.Z()
        np.testing.assert_array_almost_equal(
            hxh.matrix, expected.matrix, decimal=7
        )

    def test_scalar_mul(self) -> None:
        """2 * X produces correct matrix."""
        result = 2 * Operator.X()
        expected_matrix = np.array([[0, 2], [2, 0]], dtype=np.complex128)
        np.testing.assert_array_almost_equal(
            result.matrix, expected_matrix, decimal=7
        )

    def test_mul_scalar(self) -> None:
        """X * 2 produces correct matrix."""
        result = Operator.X() * 2
        expected_matrix = np.array([[0, 2], [2, 0]], dtype=np.complex128)
        np.testing.assert_array_almost_equal(
            result.matrix, expected_matrix, decimal=7
        )

    def test_add(self) -> None:
        """X + Z produces correct matrix."""
        result = Operator.X() + Operator.Z()
        expected_matrix = np.array(
            [[1, 1], [1, -1]], dtype=np.complex128
        )
        np.testing.assert_array_almost_equal(
            result.matrix, expected_matrix, decimal=7
        )

    def test_matmul_with_state_vector(self) -> None:
        """Operator @ StateVector returns transformed state."""
        ket0 = StateVector(1)
        result = Operator.X() @ ket0
        expected = StateVector(1, amplitudes=np.array([0, 1], dtype=np.complex128))
        np.testing.assert_array_almost_equal(
            result.amplitudes, expected.amplitudes, decimal=7
        )

    def test_matmul_operator_returns_operator(self) -> None:
        """Operator @ Operator returns Operator."""
        result = Operator.H() @ Operator.X()
        assert isinstance(result, Operator)

    def test_matmul_state_returns_state(self) -> None:
        """Operator @ StateVector returns StateVector."""
        result = Operator.H() @ StateVector(1)
        assert isinstance(result, StateVector)


# ---------------------------------------------------------------------------
# Init validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    """Test invalid matrix dimension errors."""

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            Operator(np.ones((2, 3), dtype=np.complex128))

    def test_non_power_of_two_raises(self) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            Operator(np.ones((3, 3), dtype=np.complex128))

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            Operator(np.ones((2, 2, 2), dtype=np.complex128))

    def test_dimension_mismatch_matmul(self) -> None:
        op_1q = Operator.X()
        op_2q = Operator.CNOT()
        with pytest.raises(ValueError, match="Dimension mismatch"):
            op_1q @ op_2q

    def test_dimension_mismatch_add(self) -> None:
        op_1q = Operator.X()
        op_2q = Operator.CNOT()
        with pytest.raises(ValueError, match="Dimension mismatch"):
            op_1q + op_2q

    def test_dimension_mismatch_state_vector(self) -> None:
        op = Operator.CNOT()
        ket0 = StateVector(1)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            op @ ket0


# ---------------------------------------------------------------------------
# Properties and dunder string tests
# ---------------------------------------------------------------------------

class TestProperties:
    """Test num_qubits, is_unitary, dag, repr, str."""

    def test_num_qubits(self) -> None:
        assert Operator.X().num_qubits == 1
        assert Operator.CNOT().num_qubits == 2

    def test_dag_is_hermitian_for_pauli(self) -> None:
        """Pauli matrices are Hermitian: X^dag = X."""
        assert np.allclose(Operator.X().dag.matrix, Operator.X().matrix)
        assert np.allclose(Operator.Y().dag.matrix, Operator.Y().matrix)
        assert np.allclose(Operator.Z().dag.matrix, Operator.Z().matrix)

    def test_dag_for_s_gate(self) -> None:
        """S^dag = S^dagger, which is the inverse of S."""
        s_dag = Operator.S().dag
        product = Operator.S() @ s_dag
        np.testing.assert_array_almost_equal(
            product.matrix, np.eye(2, dtype=np.complex128), decimal=7
        )

    def test_repr_contains_info(self) -> None:
        r = repr(Operator.X())
        assert "Operator" in r
        assert "2x2" in r
        assert "is_unitary=True" in r

    def test_str_contains_info(self) -> None:
        s = str(Operator.H())
        assert "Operator" in s
        assert "Unitary:" in s

    def test_is_unitary_false(self) -> None:
        """Non-unitary matrix should report is_unitary=False."""
        non_unitary = np.array([[1, 1], [0, 1]], dtype=np.complex128)
        op = Operator(non_unitary)
        assert op.is_unitary is False


# ---------------------------------------------------------------------------
# Rotation gate tests
# ---------------------------------------------------------------------------

class TestRotationGates:
    """Test rotation gates with known identities."""

    def test_rx_pi_is_x(self) -> None:
        """Rx(pi) = -iX (global phase)."""
        rx_pi = Operator.Rx(np.pi)
        x = Operator.X()
        # Rx(pi) = -i * X, so Rx(pi) @ X should be -i * I
        product = rx_pi @ x
        expected = -1j * Operator.I()
        np.testing.assert_array_almost_equal(
            product.matrix, expected.matrix, decimal=7
        )

    def test_ry_pi_is_y(self) -> None:
        """Ry(pi) applied to |0> gives |1>."""
        ry_pi = Operator.Ry(np.pi)
        ket0 = StateVector(1)
        result = ry_pi @ ket0
        expected = StateVector(1, amplitudes=np.array([0, 1], dtype=np.complex128))
        np.testing.assert_array_almost_equal(
            result.amplitudes, expected.amplitudes, decimal=7
        )

    def test_rz_zero_is_identity(self) -> None:
        """Rz(0) = I."""
        rz0 = Operator.Rz(0)
        expected = Operator.I()
        np.testing.assert_array_almost_equal(
            rz0.matrix, expected.matrix, decimal=7
        )


# ---------------------------------------------------------------------------
# is_hermitian and inverse tests
# ---------------------------------------------------------------------------

class TestIsHermitian:
    """Test is_hermitian property."""

    def test_pauli_x_hermitian(self) -> None:
        """Pauli X is Hermitian."""
        assert Operator.X().is_hermitian is True

    def test_pauli_y_hermitian(self) -> None:
        """Pauli Y is Hermitian."""
        assert Operator.Y().is_hermitian is True

    def test_pauli_z_hermitian(self) -> None:
        """Pauli Z is Hermitian."""
        assert Operator.Z().is_hermitian is True

    def test_identity_hermitian(self) -> None:
        """Identity is Hermitian."""
        assert Operator.I().is_hermitian is True

    def test_hadamard_hermitian(self) -> None:
        """Hadamard is Hermitian."""
        assert Operator.H().is_hermitian is True

    def test_s_gate_not_hermitian(self) -> None:
        """S gate is not Hermitian."""
        assert Operator.S().is_hermitian is False

    def test_t_gate_not_hermitian(self) -> None:
        """T gate is not Hermitian."""
        assert Operator.T().is_hermitian is False

    def test_rx_not_hermitian(self) -> None:
        """Rx(pi/4) is not Hermitian."""
        assert Operator.Rx(np.pi / 4).is_hermitian is False

    def test_rx_pi_hermitian(self) -> None:
        """Rx(pi) = -iX is not Hermitian (complex phase)."""
        assert Operator.Rx(np.pi).is_hermitian is False

    def test_cnot_hermitian(self) -> None:
        """CNOT is Hermitian."""
        assert Operator.CNOT().is_hermitian is True

    def test_cz_hermitian(self) -> None:
        """CZ is Hermitian."""
        assert Operator.CZ().is_hermitian is True

    def test_swap_hermitian(self) -> None:
        """SWAP is Hermitian."""
        assert Operator.SWAP().is_hermitian is True


class TestInverse:
    """Test inverse() method."""

    def test_h_inverse_is_h(self) -> None:
        """H is self-inverse."""
        h_inv = Operator.H().inverse()
        np.testing.assert_array_almost_equal(
            h_inv.matrix, Operator.H().matrix, decimal=7
        )

    def test_x_inverse_is_x(self) -> None:
        """X is self-inverse."""
        x_inv = Operator.X().inverse()
        np.testing.assert_array_almost_equal(
            x_inv.matrix, Operator.X().matrix, decimal=7
        )

    def test_s_inverse_is_s_dag(self) -> None:
        """S inverse is S^dagger."""
        s_inv = Operator.S().inverse()
        np.testing.assert_array_almost_equal(
            s_inv.matrix, Operator.S().dag.matrix, decimal=7
        )

    def test_inverse_composes_to_identity(self) -> None:
        """U @ U^-1 = I for unitary operators."""
        for gate in [Operator.H(), Operator.X(), Operator.S(), Operator.CNOT()]:
            product = gate @ gate.inverse()
            np.testing.assert_array_almost_equal(
                product.matrix,
                np.eye(gate.matrix.shape[0], dtype=np.complex128),
                decimal=7,
            )

    def test_rx_inverse(self) -> None:
        """Rx(theta) inverse is Rx(-theta)."""
        theta = 0.5
        rx = Operator.Rx(theta)
        rx_inv = rx.inverse()
        expected = Operator.Rx(-theta)
        np.testing.assert_array_almost_equal(
            rx_inv.matrix, expected.matrix, decimal=7
        )

    def test_ry_inverse(self) -> None:
        """Ry(theta) inverse is Ry(-theta)."""
        theta = 0.7
        ry = Operator.Ry(theta)
        ry_inv = ry.inverse()
        expected = Operator.Ry(-theta)
        np.testing.assert_array_almost_equal(
            ry_inv.matrix, expected.matrix, decimal=7
        )

    def test_rz_inverse(self) -> None:
        """Rz(theta) inverse is Rz(-theta)."""
        theta = 1.2
        rz = Operator.Rz(theta)
        rz_inv = rz.inverse()
        expected = Operator.Rz(-theta)
        np.testing.assert_array_almost_equal(
            rz_inv.matrix, expected.matrix, decimal=7
        )

    def test_inverse_returns_operator(self) -> None:
        """inverse() returns an Operator."""
        assert isinstance(Operator.H().inverse(), Operator)

    def test_t_inverse(self) -> None:
        """T inverse is T^dagger."""
        t_inv = Operator.T().inverse()
        np.testing.assert_array_almost_equal(
            t_inv.matrix, Operator.T().dag.matrix, decimal=7
        )
