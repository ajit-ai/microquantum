"""Tests for PauliString and PauliSum."""

import numpy as np
import pytest

from microquantum.core import (
    Operator,
    PauliString,
    PauliSum,
    StateVector,
)


class TestPauliStringCreation:
    """Test PauliString initialization and properties."""

    def test_single_qubit_x(self):
        """Create single-qubit X Pauli string."""
        ps = PauliString("X")
        assert ps.num_qubits == 1
        assert ps.label == "X"
        assert ps.coefficient == 1.0

    def test_two_qubit_xy(self):
        """Create two-qubit XY Pauli string."""
        ps = PauliString("XY")
        assert ps.num_qubits == 2
        assert ps.label == "XY"

    def test_with_coefficient(self):
        """Create Pauli string with coefficient."""
        ps = PauliString("Z", 2.0)
        assert ps.coefficient == 2.0

    def test_complex_coefficient(self):
        """Create Pauli string with complex coefficient."""
        ps = PauliString("X", 1.0 + 1.0j)
        assert ps.coefficient == 1.0 + 1.0j

    def test_lowercase_label_converted(self):
        """Lowercase labels are converted to uppercase."""
        ps = PauliString("xyz")
        assert ps.label == "XYZ"

    def test_empty_label_raises(self):
        """Empty label raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            PauliString("")

    def test_invalid_label_raises(self):
        """Invalid characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            PauliString("ABC")


class TestPauliStringProperties:
    """Test PauliString properties."""

    def test_is_hermitian_real_coefficient(self):
        """Real coefficient → Hermitian."""
        ps = PauliString("X", 2.0)
        assert ps.is_hermitian is True

    def test_is_hermitian_complex_coefficient(self):
        """Complex coefficient → not Hermitian."""
        ps = PauliString("X", 1.0 + 1.0j)
        assert ps.is_hermitian is False

    def test_is_hermitian_identity(self):
        """Identity with real coefficient → Hermitian."""
        ps = PauliString("II", 1.0)
        assert ps.is_hermitian is True


class TestPauliStringToOperator:
    """Test conversion to full Operator."""

    def test_x_to_operator(self):
        """X Pauli string converts to Pauli-X matrix."""
        ps = PauliString("X")
        op = ps.to_operator()
        np.testing.assert_array_almost_equal(
            op.matrix, Operator.X().matrix
        )

    def test_y_to_operator(self):
        """Y Pauli string converts to Pauli-Y matrix."""
        ps = PauliString("Y")
        op = ps.to_operator()
        np.testing.assert_array_almost_equal(
            op.matrix, Operator.Y().matrix
        )

    def test_z_to_operator(self):
        """Z Pauli string converts to Pauli-Z matrix."""
        ps = PauliString("Z")
        op = ps.to_operator()
        np.testing.assert_array_almost_equal(
            op.matrix, Operator.Z().matrix
        )

    def test_identity_to_operator(self):
        """Identity Pauli string converts to identity matrix."""
        ps = PauliString("I")
        op = ps.to_operator()
        np.testing.assert_array_almost_equal(
            op.matrix, np.eye(2, dtype=np.complex128)
        )

    def test_two_qubit_xz_to_operator(self):
        """Two-qubit XZ converts to X⊗Z."""
        ps = PauliString("XZ")
        op = ps.to_operator()
        expected = np.kron(
            Operator.X().matrix, Operator.Z().matrix
        )
        np.testing.assert_array_almost_equal(op.matrix, expected)

    def test_with_coefficient(self):
        """Coefficient multiplies the matrix."""
        ps = PauliString("X", 2.0)
        op = ps.to_operator()
        np.testing.assert_array_almost_equal(
            op.matrix, 2.0 * Operator.X().matrix
        )


class TestPauliStringExpectation:
    """Test expectation value computation."""

    def test_z_on_ket0(self):
        """<0|Z|0> = 1."""
        ps = PauliString("Z")
        state = StateVector(1)  # |0>
        assert np.isclose(ps.expectation(state), 1.0)

    def test_z_on_ket1(self):
        """<1|Z|1> = -1."""
        ps = PauliString("Z")
        state = StateVector(1, amplitudes=np.array([0, 1], dtype=np.complex128))
        assert np.isclose(ps.expectation(state), -1.0)

    def test_x_on_plus_state(self):
        """<+|X|+> = 1."""
        ps = PauliString("X")
        plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        state = StateVector(1, amplitudes=plus)
        assert np.isclose(ps.expectation(state), 1.0)

    def test_identity_expectation(self):
        """<psi|I|psi> = 1 for normalized state."""
        ps = PauliString("I")
        state = StateVector(1)
        assert np.isclose(ps.expectation(state), 1.0)

    def test_two_qubit_zz_on_bell(self):
        """<Bell|ZZ|Bell> = 1."""
        ps = PauliString("ZZ")
        bell = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        state = StateVector(2, amplitudes=bell)
        assert np.isclose(ps.expectation(state), 1.0)

    def test_two_qubit_zz_on_product(self):
        """<01|ZZ|01> = -1."""
        ps = PauliString("ZZ")
        state = StateVector(2, amplitudes=np.array([0, 1, 0, 0], dtype=np.complex128))
        assert np.isclose(ps.expectation(state), -1.0)

    def test_matches_matrix_computation(self):
        """Expectation matches matrix computation for 3-qubit system."""
        ps = PauliString("XYZ")
        op = ps.to_operator()

        # Random 3-qubit state
        np.random.seed(42)
        amps = np.random.randn(8) + 1j * np.random.randn(8)
        amps /= np.linalg.norm(amps)
        state = StateVector(3, amplitudes=amps)

        # PauliString expectation
        ps_exp = ps.expectation(state)

        # Matrix expectation
        matrix_exp = float(np.real(np.conj(state.amplitudes) @ op.matrix @ state.amplitudes))

        assert np.isclose(ps_exp, matrix_exp, atol=1e-10)


class TestPauliStringArithmetic:
    """Test PauliString arithmetic operations."""

    def test_tensor(self):
        """Tensor product of two Pauli strings."""
        ps1 = PauliString("X")
        ps2 = PauliString("Z")
        result = ps1.tensor(ps2)
        assert result.label == "XZ"
        assert result.coefficient == 1.0

    def test_tensor_with_coefficients(self):
        """Tensor product multiplies coefficients."""
        ps1 = PauliString("X", 2.0)
        ps2 = PauliString("Z", 3.0)
        result = ps1.tensor(ps2)
        assert result.coefficient == 6.0

    def test_add(self):
        """Add two Pauli strings produces PauliSum."""
        ps1 = PauliString("X")
        ps2 = PauliString("Z")
        result = ps1 + ps2
        assert isinstance(result, PauliSum)
        assert len(result) == 2

    def test_scalar_mul(self):
        """Scalar multiplication."""
        ps = PauliString("X")
        result = 3.0 * ps
        assert result.coefficient == 3.0

    def test_mul_scalar(self):
        """Scalar multiplication on right."""
        ps = PauliString("X")
        result = ps * 2.0
        assert result.coefficient == 2.0

    def test_equality(self):
        """Equal Pauli strings are equal."""
        ps1 = PauliString("X", 2.0)
        ps2 = PauliString("X", 2.0)
        assert ps1 == ps2

    def test_inequality(self):
        """Different Pauli strings are not equal."""
        ps1 = PauliString("X")
        ps2 = PauliString("Y")
        assert ps1 != ps2

    def test_hash(self):
        """Equal Pauli strings have same hash."""
        ps1 = PauliString("X", 2.0)
        ps2 = PauliString("X", 2.0)
        assert hash(ps1) == hash(ps2)


class TestPauliSumCreation:
    """Test PauliSum initialization."""

    def test_empty_sum(self):
        """Empty PauliSum has zero terms."""
        ps = PauliSum()
        assert ps.num_terms == 0
        assert ps.num_qubits == 0

    def test_from_label(self):
        """Create from single label."""
        ps = PauliSum.from_label("XYZ")
        assert ps.num_terms == 1
        assert ps.num_qubits == 3

    def test_from_terms(self):
        """Create from list of terms."""
        terms = [PauliString("X"), PauliString("Z")]
        ps = PauliSum(terms)
        assert ps.num_terms == 2
        assert ps.num_qubits == 1


class TestPauliSumProperties:
    """Test PauliSum properties."""

    def test_num_qubits_max(self):
        """num_qubits is max of all terms."""
        terms = [PauliString("X"), PauliString("YZ")]
        ps = PauliSum(terms)
        assert ps.num_qubits == 2

    def test_is_hermitian_all_real(self):
        """All real coefficients → Hermitian."""
        terms = [PauliString("X", 2.0), PauliString("Z", 3.0)]
        ps = PauliSum(terms)
        assert ps.is_hermitian is True

    def test_is_hermitian_complex(self):
        """Complex coefficient → not Hermitian."""
        terms = [PauliString("X", 1.0 + 1.0j)]
        ps = PauliSum(terms)
        assert ps.is_hermitian is False


class TestPauliSumExpectation:
    """Test PauliSum expectation value."""

    def test_single_term(self):
        """Single term matches PauliString expectation."""
        ps = PauliSum([PauliString("Z")])
        state = StateVector(1)
        assert np.isclose(ps.expectation(state), 1.0)

    def test_sum_of_terms(self):
        """Sum of terms gives correct expectation."""
        # H = 0.5 * (Z ⊗ I + I ⊗ Z) for 2-qubit system
        terms = [
            PauliString("ZI", 0.5),
            PauliString("IZ", 0.5),
        ]
        ps = PauliSum(terms)
        state = StateVector(2)  # |00>
        # <00|0.5*ZI + 0.5*IZ|00> = 0.5*1 + 0.5*1 = 1.0
        assert np.isclose(ps.expectation(state), 1.0)


class TestPauliSumToOperator:
    """Test PauliSum conversion to Operator."""

    def test_single_term(self):
        """Single term converts correctly."""
        ps = PauliSum([PauliString("X")])
        op = ps.to_operator()
        np.testing.assert_array_almost_equal(
            op.matrix, Operator.X().matrix
        )

    def test_sum_terms(self):
        """Sum of terms converts correctly."""
        ps = PauliSum([
            PauliString("ZI", 0.5),
            PauliString("IZ", 0.5),
        ])
        op = ps.to_operator()
        # Should be 0.5 * (Z⊗I + I⊗Z)
        expected = 0.5 * (
            np.kron(Operator.Z().matrix, np.eye(2, dtype=np.complex128))
            + np.kron(np.eye(2, dtype=np.complex128), Operator.Z().matrix)
        )
        np.testing.assert_array_almost_equal(op.matrix, expected)


class TestPauliSumArithmetic:
    """Test PauliSum arithmetic."""

    def test_add_pauli_sum(self):
        """Add two PauliSums."""
        ps1 = PauliSum([PauliString("X")])
        ps2 = PauliSum([PauliString("Z")])
        result = ps1 + ps2
        assert result.num_terms == 2

    def test_add_pauli_string(self):
        """Add PauliString to PauliSum."""
        ps1 = PauliSum([PauliString("X")])
        ps2 = PauliString("Z")
        result = ps1 + ps2
        assert result.num_terms == 2

    def test_subtract(self):
        """Subtract PauliSums."""
        ps1 = PauliSum([PauliString("X")])
        ps2 = PauliSum([PauliString("Z")])
        result = ps1 - ps2
        assert result.num_terms == 2

    def test_scalar_mul(self):
        """Scalar multiplication."""
        ps = PauliSum([PauliString("X"), PauliString("Z")])
        result = ps * 2.0
        assert result.num_terms == 2
        assert all(t.coefficient == 2.0 for t in result)

    def test_simplify(self):
        """Simplify combines like terms."""
        ps = PauliSum([
            PauliString("X", 1.0),
            PauliString("X", 2.0),
            PauliString("Z"),
        ])
        simplified = ps.simplify()
        assert simplified.num_terms == 2  # X + Z

    def test_simplify_removes_zeros(self):
        """Simplify removes zero terms."""
        ps = PauliSum([
            PauliString("X", 1.0),
            PauliString("X", -1.0),
        ])
        simplified = ps.simplify()
        assert simplified.num_terms == 0


class TestPauliSumString:
    """Test PauliSum string representations."""

    def test_repr(self):
        """repr contains PauliSum."""
        ps = PauliSum([PauliString("X")])
        assert "PauliSum" in repr(ps)

    def test_str(self):
        """str contains term string."""
        ps = PauliSum([PauliString("X")])
        assert "X" in str(ps)

    def test_empty_str(self):
        """Empty PauliSum string is '0'."""
        ps = PauliSum()
        assert str(ps) == "0"
