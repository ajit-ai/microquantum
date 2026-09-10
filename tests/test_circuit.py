"""Unit tests for microquantum.core.circuit.QuantumCircuit."""

import numpy as np
import pytest

from microquantum.core import Operator, QuantumCircuit, StateVector


# ---------------------------------------------------------------------------
# Bell State Generation
# ---------------------------------------------------------------------------

class TestBellState:
    """H(0) -> CNOT(0,1) on |00> yields (|00>+|11>)/sqrt(2)."""

    def test_bell_state_amplitudes(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1)
        result = qc.run()
        inv_sqrt2 = 1 / np.sqrt(2)
        expected = StateVector(
            2,
            amplitudes=np.array(
                [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
            ),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_bell_state_normalized(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        assert qc.run().is_normalized

    def test_bell_fidelity(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        bell = qc.run()
        inv_sqrt2 = 1 / np.sqrt(2)
        expected = StateVector(
            2,
            amplitudes=np.array(
                [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
            ),
        )
        assert bell.fidelity(expected) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GHZ State Generation
# ---------------------------------------------------------------------------

class TestGHZState:
    """H(0) -> CNOT(0,1) -> CNOT(0,2) on |000> yields (|000>+|111>)/sqrt(2)."""

    def test_ghz_amplitudes(self) -> None:
        qc = QuantumCircuit(3)
        qc.h(0).cnot(0, 1).cnot(0, 2)
        result = qc.run()
        inv_sqrt2 = 1 / np.sqrt(2)
        expected = np.zeros(8, dtype=np.complex128)
        expected[0b000] = inv_sqrt2
        expected[0b111] = inv_sqrt2
        np.testing.assert_array_almost_equal(result.amplitudes, expected)

    def test_ghz_normalized(self) -> None:
        qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(0, 2)
        assert qc.run().is_normalized


# ---------------------------------------------------------------------------
# Unitary Equivalence
# ---------------------------------------------------------------------------

class TestUnitaryEquivalence:
    """Compare circuit.run() against circuit.get_unitary() @ initial_state."""

    def test_bell_unitary_vs_run(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        run_result = qc.run()
        unitary_result = qc.get_unitary() @ StateVector(2)
        np.testing.assert_array_almost_equal(
            run_result.amplitudes, unitary_result.amplitudes, decimal=7
        )

    def test_unitary_is_unitary(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        U = qc.get_unitary()
        assert U.is_unitary

    def test_unitary_on_custom_state(self) -> None:
        qc = QuantumCircuit(2).x(0)
        custom = StateVector(
            2, amplitudes=np.array([0, 0, 1, 0], dtype=np.complex128)
        )
        run_result = qc.run(custom)
        unitary_result = qc.get_unitary() @ custom
        np.testing.assert_array_almost_equal(
            run_result.amplitudes, unitary_result.amplitudes, decimal=7
        )

    def test_3qubit_unitary_vs_run(self) -> None:
        qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(0, 2)
        run_result = qc.run()
        unitary_result = qc.get_unitary() @ StateVector(3)
        np.testing.assert_array_almost_equal(
            run_result.amplitudes, unitary_result.amplitudes, decimal=7
        )


# ---------------------------------------------------------------------------
# Method Chaining
# ---------------------------------------------------------------------------

class TestMethodChaining:
    """Test fluent circuit construction."""

    def test_basic_chaining(self) -> None:
        result = QuantumCircuit(2).h(0).cx(0, 1).run()
        inv_sqrt2 = 1 / np.sqrt(2)
        expected = StateVector(
            2,
            amplitudes=np.array(
                [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
            ),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_chaining_returns_circuit(self) -> None:
        qc = QuantumCircuit(2)
        assert qc.h(0) is qc
        assert qc.x(1) is qc
        assert qc.cx(0, 1) is qc
        assert qc.cz(0, 1) is qc
        assert qc.swap(0, 1) is qc
        assert qc.y(0) is qc
        assert qc.z(1) is qc
        assert qc.s(0) is qc
        assert qc.t(1) is qc

    def test_rotation_chaining(self) -> None:
        qc = QuantumCircuit(1).rx(0.5, 0).ry(1.0, 0).rz(0.3, 0)
        result = qc.run()
        assert result.is_normalized


# ---------------------------------------------------------------------------
# Circuit Composition (__add__)
# ---------------------------------------------------------------------------

class TestCircuitComposition:
    """Test qc1 + qc2 concatenation."""

    def test_add_concatenates_gates(self) -> None:
        qc1 = QuantumCircuit(2).h(0)
        qc2 = QuantumCircuit(2).cx(0, 1)
        qc = qc1 + qc2
        assert qc.num_gates == 2
        assert qc.num_qubits == 2

    def test_add_produces_correct_state(self) -> None:
        qc1 = QuantumCircuit(2).h(0)
        qc2 = QuantumCircuit(2).cx(0, 1)
        result = (qc1 + qc2).run()
        inv_sqrt2 = 1 / np.sqrt(2)
        expected = StateVector(
            2,
            amplitudes=np.array(
                [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
            ),
        )
        np.testing.assert_array_almost_equal(result.amplitudes, expected.amplitudes)

    def test_add_does_not_mutate_originals(self) -> None:
        qc1 = QuantumCircuit(2).h(0)
        qc2 = QuantumCircuit(2).cx(0, 1)
        _ = qc1 + qc2
        assert qc1.num_gates == 1
        assert qc2.num_gates == 1

    def test_add_mismatched_qubits_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot concatenate"):
            QuantumCircuit(2).h(0) + QuantumCircuit(3).h(0)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    """Test num_qubits, depth, num_gates."""

    def test_empty_circuit(self) -> None:
        qc = QuantumCircuit(3)
        assert qc.num_qubits == 3
        assert qc.num_gates == 0
        assert qc.depth == 0

    def test_depth_single_qubit(self) -> None:
        qc = QuantumCircuit(1).h(0).x(0).z(0)
        assert qc.depth == 3

    def test_depth_parallel_gates(self) -> None:
        """Parallel gates on different qubits don't increase depth."""
        qc = QuantumCircuit(3).h(0).h(1).h(2)
        assert qc.depth == 1

    def test_depth_mixed(self) -> None:
        """Mixed sequential and parallel gates."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        assert qc.depth == 3


# ---------------------------------------------------------------------------
# Validation Errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    """Test error handling."""

    def test_qubit_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            QuantumCircuit(2).h(2)

    def test_qubit_negative(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            QuantumCircuit(2).h(-1)

    def test_target_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="target"):
            QuantumCircuit(2).append(Operator.CNOT(), [0])

    def test_init_zero_qubits(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            QuantumCircuit(0)

    def test_run_wrong_size_state(self) -> None:
        with pytest.raises(ValueError, match="qubits"):
            QuantumCircuit(2).run(StateVector(3))


# ---------------------------------------------------------------------------
# Repr and Str
# ---------------------------------------------------------------------------

class TestStringRepresentations:
    """Test __repr__ and __str__."""

    def test_repr(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        r = repr(qc)
        assert "QuantumCircuit" in r
        assert "num_qubits=2" in r
        assert "num_gates=2" in r

    def test_str(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        s = str(qc)
        assert "2 qubits" in s
        assert "2 gates" in s


# ---------------------------------------------------------------------------
# New Circuit Properties and Methods
# ---------------------------------------------------------------------------

class TestQubitsProperty:
    """Test qubits property."""

    def test_qubits_returns_list(self) -> None:
        """qubits returns list of qubit indices."""
        qc = QuantumCircuit(3)
        assert qc.qubits == [0, 1, 2]

    def test_qubits_single_qubit(self) -> None:
        """Single qubit circuit."""
        qc = QuantumCircuit(1)
        assert qc.qubits == [0]


class TestGateCount:
    """Test gate_count() method."""

    def test_count_all_gates(self) -> None:
        """Count all gates."""
        qc = QuantumCircuit(2).h(0).cx(0, 1).x(1)
        assert qc.gate_count() == 3

    def test_count_by_type(self) -> None:
        """Count gates by type."""
        qc = QuantumCircuit(2).h(0).h(1).x(0)
        assert qc.gate_count("h") == 2
        assert qc.gate_count("x") == 1
        assert qc.gate_count("z") == 0

    def test_count_parameterized_gates(self) -> None:
        """Count parameterized gates."""
        from microquantum.core import Parameter
        qc = QuantumCircuit(1)
        qc.rx(Parameter("theta"), 0)
        qc.ry(Parameter("phi"), 0)
        assert qc.gate_count("rx") == 1
        assert qc.gate_count("ry") == 1
        assert qc.gate_count() == 2

    def test_count_case_insensitive(self) -> None:
        """Gate count is case-insensitive."""
        qc = QuantumCircuit(1).h(0)
        assert qc.gate_count("H") == 1
        assert qc.gate_count("h") == 1


class TestContainsGate:
    """Test contains_gate() method."""

    def test_contains_h(self) -> None:
        """Circuit with H gate contains H."""
        qc = QuantumCircuit(1).h(0)
        assert qc.contains_gate("h") is True

    def test_not_contains_z(self) -> None:
        """Circuit without Z gate doesn't contain Z."""
        qc = QuantumCircuit(1).h(0)
        assert qc.contains_gate("z") is False

    def test_empty_circuit(self) -> None:
        """Empty circuit contains no gates."""
        qc = QuantumCircuit(2)
        assert qc.contains_gate("h") is False


class TestCircuitInverse:
    """Test inverse() method."""

    def test_inverse_returns_circuit(self) -> None:
        """inverse() returns a QuantumCircuit."""
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        inv = qc.inverse()
        assert isinstance(inv, QuantumCircuit)

    def test_inverse_same_num_qubits(self) -> None:
        """inverse() preserves qubit count."""
        qc = QuantumCircuit(3).h(0).cx(0, 1)
        inv = qc.inverse()
        assert inv.num_qubits == 3

    def test_inverse_same_gate_count(self) -> None:
        """inverse() has same number of gates."""
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        inv = qc.inverse()
        assert inv.num_gates == qc.num_gates

    def test_inverse_reverses_gates(self) -> None:
        """inverse() reverses gate order."""
        qc = QuantumCircuit(2).h(0).cx(0, 1).x(1)
        inv = qc.inverse()
        # inv should have gates in reverse order with inverse applied
        assert inv.num_gates == 3

    def test_inverse_produces_identity(self) -> None:
        """circuit + circuit.inverse() produces identity."""
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        combined = qc + qc.inverse()
        state = combined.run()
        expected_amps = np.array([1, 0, 0, 0], dtype=np.complex128)
        np.testing.assert_array_almost_equal(
            state.amplitudes, expected_amps, decimal=10
        )

    def test_inverse_parameterized_gates(self) -> None:
        """inverse() with bound parameterized gates works."""
        from microquantum.core import Parameter
        qc = QuantumCircuit(1)
        qc.rx(Parameter("theta"), 0)
        bound = qc.bind_parameters({"theta": 0.5})
        inv = bound.inverse()
        assert inv.num_gates == 1

    def test_inverse_unbound_parameterized_raises(self) -> None:
        """inverse() with unbound parameters raises ValueError."""
        from microquantum.core import Parameter
        qc = QuantumCircuit(1)
        qc.rx(Parameter("theta"), 0)
        with pytest.raises(ValueError, match="unbound parameter"):
            qc.inverse()

    def test_inverse_single_gate(self) -> None:
        """inverse() of single gate works."""
        qc = QuantumCircuit(1).h(0)
        inv = qc.inverse()
        # H is self-inverse, so inverse should also be H
        combined = qc + inv
        state = combined.run()
        expected_amps = np.array([1, 0], dtype=np.complex128)
        np.testing.assert_array_almost_equal(
            state.amplitudes, expected_amps, decimal=10
        )
