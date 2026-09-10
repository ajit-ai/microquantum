"""Unit tests for microquantum.core.engine.apply_gate."""

import numpy as np
import pytest

from microquantum.core import StateVector, apply_gate


# ---------------------------------------------------------------------------
# Standard gate matrices
# ---------------------------------------------------------------------------

X_GATE = np.array([[0, 1], [1, 0]], dtype=np.complex128)
H_GATE = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I_GATE = np.array([[1, 0], [0, 1]], dtype=np.complex128)

CNOT_GATE = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
    dtype=np.complex128,
)


# ---------------------------------------------------------------------------
# Single-qubit gate tests
# ---------------------------------------------------------------------------

class TestSingleQubitGates:
    """Tests for single-qubit gate application."""

    def test_x_on_zero(self) -> None:
        """X|0> = |1>"""
        ket0 = StateVector(1)
        result = apply_gate(ket0, X_GATE, [0])
        expected = StateVector(1, amplitudes=np.array([0.0, 1.0], dtype=np.complex128))
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_x_on_one(self) -> None:
        """X|1> = |0>"""
        ket1 = StateVector(1, amplitudes=np.array([0.0, 1.0], dtype=np.complex128))
        result = apply_gate(ket1, X_GATE, [0])
        expected = StateVector(1)
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_h_on_zero(self) -> None:
        """H|0> = (|0> + |1>)/sqrt(2)"""
        ket0 = StateVector(1)
        result = apply_gate(ket0, H_GATE, [0])
        inv_sqrt2 = 1 / np.sqrt(2)
        expected = StateVector(
            1,
            amplitudes=np.array([inv_sqrt2, inv_sqrt2], dtype=np.complex128),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_h_on_one(self) -> None:
        """H|1> = (|0> - |1>)/sqrt(2)"""
        ket1 = StateVector(1, amplitudes=np.array([0.0, 1.0], dtype=np.complex128))
        result = apply_gate(ket1, H_GATE, [0])
        inv_sqrt2 = 1 / np.sqrt(2)
        expected = StateVector(
            1,
            amplitudes=np.array([inv_sqrt2, -inv_sqrt2], dtype=np.complex128),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_h_squared_is_identity(self) -> None:
        """H^2 = I"""
        ket0 = StateVector(1)
        after_h1 = apply_gate(ket0, H_GATE, [0])
        after_h2 = apply_gate(after_h1, H_GATE, [0])
        np.testing.assert_array_almost_equal(after_h2.amplitudes, ket0.amplitudes)

    def test_z_on_zero(self) -> None:
        """Z|0> = |0>"""
        ket0 = StateVector(1)
        result = apply_gate(ket0, Z_GATE, [0])
        np.testing.assert_array_almost_equal(result.amplitudes, ket0.amplitudes)

    def test_z_on_one(self) -> None:
        """Z|1> = -|1>"""
        ket1 = StateVector(1, amplitudes=np.array([0.0, 1.0], dtype=np.complex128))
        result = apply_gate(ket1, Z_GATE, [0])
        expected = StateVector(
            1, amplitudes=np.array([0.0, -1.0], dtype=np.complex128)
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_identity_on_qubit(self) -> None:
        """I|psi> = |psi>"""
        ket0 = StateVector(1)
        result = apply_gate(ket0, I_GATE, [0])
        np.testing.assert_array_almost_equal(result.amplitudes, ket0.amplitudes)


# ---------------------------------------------------------------------------
# Multi-qubit single-gate tests
# ---------------------------------------------------------------------------

class TestSingleGateOnMultiQubitState:
    """Tests for single-qubit gate applied to multi-qubit states."""

    def test_x_on_qubit0_of_two_qubit(self) -> None:
        """X on qubit 0 (MSB) of |00> = |10> (index 2)."""
        state = StateVector(2)
        result = apply_gate(state, X_GATE, [0])
        expected = StateVector(
            2,
            amplitudes=np.array([0, 0, 1, 0], dtype=np.complex128),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_x_on_qubit1_of_two_qubit(self) -> None:
        """X on qubit 1 (LSB) of |00> = |01> (index 1)."""
        state = StateVector(2)
        result = apply_gate(state, X_GATE, [1])
        expected = StateVector(
            2,
            amplitudes=np.array([0, 1, 0, 0], dtype=np.complex128),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_h_on_qubit1_of_three_qubit(self) -> None:
        """H on qubit 1 of |000> should only transform qubit 1."""
        state = StateVector(3)
        result = apply_gate(state, H_GATE, [1])
        inv_sqrt2 = 1 / np.sqrt(2)

        expected = np.zeros(8, dtype=np.complex128)
        expected[0b000] = inv_sqrt2
        expected[0b010] = inv_sqrt2

        np.testing.assert_array_almost_equal(result.amplitudes, expected)

    def test_num_qubits_preserved(self) -> None:
        """Applying a gate preserves the qubit count."""
        state = StateVector(3)
        result = apply_gate(state, H_GATE, [2])
        assert result.num_qubits == 3

    def test_normalized_output(self) -> None:
        """Output state should be normalized for unitary gates."""
        state = StateVector(3)
        result = apply_gate(state, H_GATE, [0])
        assert result.is_normalized


# ---------------------------------------------------------------------------
# Two-qubit gate tests
# ---------------------------------------------------------------------------

class TestTwoQubitGates:
    """Tests for two-qubit gate application."""

    def test_cnot_10_to_11(self) -> None:
        """CNOT(ctl=0, tgt=1) |10> = |11>"""
        amplitudes = np.array([0, 0, 1, 0], dtype=np.complex128)
        state = StateVector(2, amplitudes=amplitudes)
        result = apply_gate(state, CNOT_GATE, [0, 1])
        expected = StateVector(
            2,
            amplitudes=np.array([0, 0, 0, 1], dtype=np.complex128),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_cnot_00_to_00(self) -> None:
        """CNOT(ctl=0, tgt=1) |00> = |00>"""
        state = StateVector(2)
        result = apply_gate(state, CNOT_GATE, [0, 1])
        np.testing.assert_array_almost_equal(result.amplitudes, state.amplitudes)

    def test_cnot_01_to_01(self) -> None:
        """CNOT(ctl=0, tgt=1) |01> = |01> (control is 0)"""
        amplitudes = np.array([0, 1, 0, 0], dtype=np.complex128)
        state = StateVector(2, amplitudes=amplitudes)
        result = apply_gate(state, CNOT_GATE, [0, 1])
        np.testing.assert_array_almost_equal(result.amplitudes, state.amplitudes)

    def test_cnot_11_to_10(self) -> None:
        """CNOT(ctl=0, tgt=1) |11> = |10>"""
        amplitudes = np.array([0, 0, 0, 1], dtype=np.complex128)
        state = StateVector(2, amplitudes=amplitudes)
        result = apply_gate(state, CNOT_GATE, [0, 1])
        expected = StateVector(
            2,
            amplitudes=np.array([0, 0, 1, 0], dtype=np.complex128),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)


# ---------------------------------------------------------------------------
# Bell state creation
# ---------------------------------------------------------------------------

class TestBellState:
    """Tests for Bell state creation via H + CNOT."""

    def test_bell_state_fidelity(self) -> None:
        """H on qubit 0 of |00>, then CNOT(0,1) -> (|00>+|11>)/sqrt(2)."""
        state = StateVector(2)
        state = apply_gate(state, H_GATE, [0])
        bell = apply_gate(state, CNOT_GATE, [0, 1])

        inv_sqrt2 = 1 / np.sqrt(2)
        expected_bell = StateVector(
            2,
            amplitudes=np.array(
                [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
            ),
        )

        fidelity = bell.fidelity(expected_bell)
        assert fidelity == pytest.approx(1.0)

    def test_bell_state_amplitudes(self) -> None:
        """Verify exact amplitudes of Bell state."""
        state = StateVector(2)
        state = apply_gate(state, H_GATE, [0])
        bell = apply_gate(state, CNOT_GATE, [0, 1])

        inv_sqrt2 = 1 / np.sqrt(2)
        assert bell.amplitudes[0b00] == pytest.approx(inv_sqrt2)
        assert bell.amplitudes[0b01] == pytest.approx(0.0)
        assert bell.amplitudes[0b10] == pytest.approx(0.0)
        assert bell.amplitudes[0b11] == pytest.approx(inv_sqrt2)

    def test_bell_state_is_normalized(self) -> None:
        """Bell state should be normalized."""
        state = StateVector(2)
        state = apply_gate(state, H_GATE, [0])
        bell = apply_gate(state, CNOT_GATE, [0, 1])
        assert bell.is_normalized


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------

class TestValidationErrors:
    """Tests for input validation and error handling."""

    def test_target_qubit_out_of_range_high(self) -> None:
        state = StateVector(2)
        with pytest.raises(ValueError, match="out of range"):
            apply_gate(state, H_GATE, [2])

    def test_target_qubit_negative(self) -> None:
        state = StateVector(2)
        with pytest.raises(ValueError, match="out of range"):
            apply_gate(state, H_GATE, [-1])

    def test_duplicate_target_qubits(self) -> None:
        state = StateVector(2)
        with pytest.raises(ValueError, match="distinct"):
            apply_gate(state, H_GATE, [0, 0])

    def test_empty_target_qubits(self) -> None:
        state = StateVector(2)
        with pytest.raises(ValueError, match="must not be empty"):
            apply_gate(state, H_GATE, [])

    def test_gate_wrong_dimension_for_one_qubit(self) -> None:
        state = StateVector(1)
        bad_gate = np.eye(4, dtype=np.complex128)
        with pytest.raises(ValueError, match="must have shape"):
            apply_gate(state, bad_gate, [0])

    def test_gate_wrong_dimension_for_two_qubit(self) -> None:
        state = StateVector(2)
        with pytest.raises(ValueError, match="must have shape"):
            apply_gate(state, H_GATE, [0, 1])

    def test_gate_rectangular_matrix(self) -> None:
        state = StateVector(1)
        bad_gate = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.complex128)
        with pytest.raises(ValueError, match="must have shape"):
            apply_gate(state, bad_gate, [0])


# ---------------------------------------------------------------------------
# StateVector immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    """Verify apply_gate returns a new state, not mutated original."""

    def test_original_state_unchanged(self) -> None:
        state = StateVector(1)
        original_amps = state.amplitudes.copy()
        _ = apply_gate(state, H_GATE, [0])
        np.testing.assert_array_equal(state.amplitudes, original_amps)

    def test_returns_new_instance(self) -> None:
        state = StateVector(1)
        result = apply_gate(state, H_GATE, [0])
        assert result is not state
