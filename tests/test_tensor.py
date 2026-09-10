"""Unit tests for microquantum.core.tensor (tensor & expand_operator)."""

import numpy as np
import pytest

from microquantum.core import Operator, StateVector, expand_operator, tensor


# ---------------------------------------------------------------------------
# tensor() with StateVector
# ---------------------------------------------------------------------------

class TestTensorStateVectors:
    """Tests for tensor product of state vectors."""

    def test_ket0_tensor_ket1(self) -> None:
        """|0> ⊗ |1> = |01>."""
        ket0 = StateVector(1)
        ket1 = StateVector(1, amplitudes=np.array([0, 1], dtype=np.complex128))
        result = tensor(ket0, ket1)
        expected = StateVector(
            2, amplitudes=np.array([0, 1, 0, 0], dtype=np.complex128)
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_ket1_tensor_ket0(self) -> None:
        """|1> ⊗ |0> = |10>."""
        ket1 = StateVector(1, amplitudes=np.array([0, 1], dtype=np.complex128))
        ket0 = StateVector(1)
        result = tensor(ket1, ket0)
        expected = StateVector(
            2, amplitudes=np.array([0, 0, 1, 0], dtype=np.complex128)
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_three_qubit_tensor(self) -> None:
        """|0> ⊗ |0> ⊗ |1> = |001>."""
        ket0 = StateVector(1)
        ket1 = StateVector(1, amplitudes=np.array([0, 1], dtype=np.complex128))
        result = tensor(ket0, ket0, ket1)
        assert result.num_qubits == 3
        expected = np.zeros(8, dtype=np.complex128)
        expected[0b001] = 1.0
        np.testing.assert_array_almost_equal(result.amplitudes, expected)

    def test_result_dimension(self) -> None:
        """Result dimension is product of input dimensions."""
        s1 = StateVector(2)
        s2 = StateVector(3)
        result = tensor(s1, s2)
        assert result.num_qubits == 5
        assert result.dim == 32

    def test_single_item_passthrough(self) -> None:
        """tensor(s) returns equivalent state."""
        s = StateVector(2)
        result = tensor(s)
        np.testing.assert_array_almost_equal(result.amplitudes, s.amplitudes)

    def test_superposition_tensor(self) -> None:
        """(a|0> + b|1>) ⊗ |0> = a|00> + b|10>."""
        plus = 1 / np.sqrt(2)
        ket_plus = StateVector(
            1, amplitudes=np.array([plus, plus], dtype=np.complex128)
        )
        ket0 = StateVector(1)
        result = tensor(ket_plus, ket0)
        expected = np.array([plus, 0, plus, 0], dtype=np.complex128)
        np.testing.assert_array_almost_equal(result.amplitudes, expected)


# ---------------------------------------------------------------------------
# tensor() with Operator
# ---------------------------------------------------------------------------

class TestTensorOperators:
    """Tests for tensor product of operators."""

    def test_x_tensor_z(self) -> None:
        """X ⊗ Z produces correct 4x4 matrix."""
        result = tensor(Operator.X(), Operator.Z())
        expected_matrix = np.kron(
            np.array([[0, 1], [1, 0]], dtype=np.complex128),
            np.array([[1, 0], [0, -1]], dtype=np.complex128),
        )
        np.testing.assert_array_almost_equal(result.matrix, expected_matrix)

    def test_unitarity_preserved(self) -> None:
        """U1 ⊗ U2 remains unitary when U1, U2 are unitary."""
        result = tensor(Operator.H(), Operator.X())
        assert result.is_unitary

    def test_result_dimension(self) -> None:
        """Result dimension is product of input dimensions."""
        result = tensor(Operator.X(), Operator.Z())
        assert result.num_qubits == 2
        assert result.matrix.shape == (4, 4)

    def test_identity_tensor_identity(self) -> None:
        """I ⊗ I = I (4x4)."""
        result = tensor(Operator.I(), Operator.I())
        expected = Operator(np.eye(4, dtype=np.complex128))
        np.testing.assert_array_almost_equal(result.matrix, expected.matrix)

    def test_h_tensor_i(self) -> None:
        """H ⊗ I produces correct 4x4 matrix."""
        result = tensor(Operator.H(), Operator.I())
        expected_matrix = np.kron(
            Operator.H().matrix, np.eye(2, dtype=np.complex128)
        )
        np.testing.assert_array_almost_equal(result.matrix, expected_matrix)


# ---------------------------------------------------------------------------
# expand_operator() tests
# ---------------------------------------------------------------------------

class TestExpandOperator:
    """Tests for embedding operators into full Hilbert space."""

    def test_x_on_qubit1_of_2(self) -> None:
        """Expanding X on qubit 1 of 2 gives I ⊗ X."""
        result = expand_operator(Operator.X(), [1], 2)
        expected = tensor(Operator.I(), Operator.X())
        np.testing.assert_array_almost_equal(result.matrix, expected.matrix)

    def test_h_on_qubit0_of_2(self) -> None:
        """Expanding H on qubit 0 of 2 gives H ⊗ I."""
        result = expand_operator(Operator.H(), [0], 2)
        expected = tensor(Operator.H(), Operator.I())
        np.testing.assert_array_almost_equal(result.matrix, expected.matrix)

    def test_expanded_operator_unitary(self) -> None:
        """Expanded operator remains unitary."""
        result = expand_operator(Operator.X(), [2], 4)
        assert result.is_unitary

    def test_apply_expanded_h_to_00(self) -> None:
        """(H ⊗ I)(|00>) = (1/√2)(|00> + |10>)."""
        expanded = expand_operator(Operator.H(), [0], 2)
        ket00 = StateVector(2)
        result = expanded @ ket00
        inv_sqrt2 = 1 / np.sqrt(2)
        expected = StateVector(
            2,
            amplitudes=np.array([inv_sqrt2, 0, inv_sqrt2, 0], dtype=np.complex128),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_apply_expanded_x_to_00_gives_10(self) -> None:
        """(I ⊗ X)(|00>) = |01> (X on LSB)."""
        expanded = expand_operator(Operator.X(), [1], 2)
        ket00 = StateVector(2)
        result = expanded @ ket00
        expected = StateVector(
            2, amplitudes=np.array([0, 1, 0, 0], dtype=np.complex128)
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_expand_on_qubit0_of_3(self) -> None:
        """Expanding X on qubit 0 of 3 gives X ⊗ I ⊗ I."""
        result = expand_operator(Operator.X(), [0], 3)
        expected = tensor(Operator.X(), Operator.I(), Operator.I())
        np.testing.assert_array_almost_equal(result.matrix, expected.matrix)

    def test_expand_on_qubit2_of_3(self) -> None:
        """Expanding X on qubit 2 of 3 gives I ⊗ I ⊗ X."""
        result = expand_operator(Operator.X(), [2], 3)
        expected = tensor(Operator.I(), Operator.I(), Operator.X())
        np.testing.assert_array_almost_equal(result.matrix, expected.matrix)

    def test_expand_cnot_on_2(self) -> None:
        """Expanding CNOT on [0,1] of 2 gives CNOT itself."""
        result = expand_operator(Operator.CNOT(), [0, 1], 2)
        np.testing.assert_array_almost_equal(result.matrix, Operator.CNOT().matrix)

    def test_expand_cnot_on_3_targets_01(self) -> None:
        """Expanding CNOT on [0,1] of 3 gives CNOT ⊗ I."""
        result = expand_operator(Operator.CNOT(), [0, 1], 3)
        expected = tensor(Operator.CNOT(), Operator.I())
        np.testing.assert_array_almost_equal(result.matrix, expected.matrix)

    def test_expand_swap_non_contiguous(self) -> None:
        """Expanding SWAP on non-contiguous targets [0, 2] of 3 qubits."""
        result = expand_operator(Operator.SWAP(), [0, 2], 3)
        assert result.matrix.shape == (8, 8)
        assert result.is_unitary

    def test_non_contiguous_vs_manual(self) -> None:
        """Expanding X on [1] of 3 equals manual I⊗X⊗I."""
        result = expand_operator(Operator.X(), [1], 3)
        expected = tensor(Operator.I(), Operator.X(), Operator.I())
        np.testing.assert_array_almost_equal(result.matrix, expected.matrix)

    def test_preserves_dimension(self) -> None:
        """Expanded operator has correct dimensions."""
        result = expand_operator(Operator.H(), [0], 4)
        assert result.matrix.shape == (16, 16)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------

class TestValidationErrors:
    """Tests for validation error handling."""

    def test_tensor_mixed_types(self) -> None:
        """Mixing StateVector and Operator raises TypeError."""
        with pytest.raises(TypeError, match="same type"):
            tensor(StateVector(1), Operator.X())

    def test_tensor_empty(self) -> None:
        """Empty tensor() raises TypeError."""
        with pytest.raises(TypeError, match="at least one"):
            tensor()

    def test_tensor_invalid_type(self) -> None:
        """Non-quantum object raises TypeError."""
        with pytest.raises(TypeError, match="StateVector or Operator"):
            tensor("hello")  # type: ignore

    def test_expand_target_out_of_range(self) -> None:
        """Out-of-range target qubit raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            expand_operator(Operator.X(), [2], 2)

    def test_expand_target_negative(self) -> None:
        """Negative target qubit raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            expand_operator(Operator.X(), [-1], 2)

    def test_expand_target_count_mismatch(self) -> None:
        """Wrong number of targets raises ValueError."""
        with pytest.raises(ValueError, match="target"):
            expand_operator(Operator.X(), [0, 1], 2)

    def test_expand_duplicate_targets(self) -> None:
        """Duplicate targets raises ValueError."""
        with pytest.raises(ValueError, match="distinct"):
            expand_operator(Operator.CNOT(), [0, 0], 2)
