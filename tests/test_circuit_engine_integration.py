"""Tests for QuantumCircuit.run() using efficient engine.apply_gate()."""

import numpy as np
import pytest

from microquantum.core import (
    Operator,
    QuantumCircuit,
    StateVector,
    expand_operator,
)


class TestCircuitEngineIntegration:
    """Verify circuit.run() produces correct results using einsum contraction."""

    def test_single_gate_matches_manual(self):
        """Single H gate via circuit matches manual expansion."""
        circuit = QuantumCircuit(1)
        circuit.h(0)
        state = circuit.run()

        manual = expand_operator(Operator.H(), [0], 1)
        expected = manual @ StateVector(1)

        np.testing.assert_array_almost_equal(
            state.amplitudes, expected.amplitudes, decimal=10
        )

    def test_two_qubit_circuit(self):
        """2-qubit circuit with H and CNOT produces Bell state."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        state = circuit.run()

        expected_amps = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        np.testing.assert_array_almost_equal(
            state.amplitudes, expected_amps, decimal=10
        )

    def test_three_qubit_ghz(self):
        """3-qubit GHZ state via circuit matches manual construction."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.cnot(0, 2)
        state = circuit.run()

        expected_amps = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        np.testing.assert_array_almost_equal(
            state.amplitudes, expected_amps, decimal=10
        )

    def test_circuit_run_matches_get_unitary(self):
        """circuit.run() produces same result as get_unitary() @ |0>."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)

        state_run = circuit.run()
        unitary = circuit.get_unitary()
        state_unitary = unitary @ StateVector(2)

        np.testing.assert_array_almost_equal(
            state_run.amplitudes, state_unitary.amplitudes, decimal=10
        )

    def test_circuit_run_with_initial_state(self):
        """circuit.run() with custom initial state works."""
        circuit = QuantumCircuit(2)
        circuit.x(0)

        # |10⟩ = amplitude at index 2 (qubit 0 = MSB = 1, qubit 1 = 0)
        initial = StateVector(2, amplitudes=np.array([0, 0, 1, 0], dtype=np.complex128))
        state = circuit.run(initial_state=initial)

        # X on qubit 0 flips |10⟩ → |00⟩
        expected_amps = np.array([1, 0, 0, 0], dtype=np.complex128)
        np.testing.assert_array_almost_equal(
            state.amplitudes, expected_amps, decimal=10
        )

    def test_inverse_circuit_produces_identity(self):
        """circuit + circuit.inverse() produces identity."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.x(1)

        combined = circuit + circuit.inverse()
        state = combined.run()

        expected_amps = np.array([1, 0, 0, 0], dtype=np.complex128)
        np.testing.assert_array_almost_equal(
            state.amplitudes, expected_amps, decimal=10
        )

    def test_inverse_reverses_state(self):
        """circuit.inverse().run() reverses circuit.run()."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)

        final_state = circuit.run()
        original_state = circuit.inverse().run()

        # Apply final_state to get back to |00>
        recovered = final_state  # This is the state after circuit
        # Apply inverse to get back to initial
        inverse_circuit = circuit.inverse()
        recovered_state = inverse_circuit.run(final_state)

        expected_amps = np.array([1, 0, 0, 0], dtype=np.complex128)
        np.testing.assert_array_almost_equal(
            recovered_state.amplitudes, expected_amps, decimal=10
        )


class TestLongCircuit:
    """Verify circuit works with many gates (performance regression)."""

    def test_many_single_qubit_gates(self):
        """100 single-qubit gates on a 5-qubit circuit."""
        circuit = QuantumCircuit(5)
        for _ in range(100):
            circuit.h(0)
            circuit.x(1)
            circuit.z(2)
        state = circuit.run()

        assert state.num_qubits == 5
        assert np.isclose(np.sum(np.abs(state.amplitudes) ** 2), 1.0)

    def test_many_two_qubit_gates(self):
        """50 CNOT gates on a 3-qubit circuit."""
        circuit = QuantumCircuit(3)
        for _ in range(50):
            circuit.cnot(0, 1)
            circuit.cnot(1, 2)
        state = circuit.run()

        assert state.num_qubits == 3
        assert np.isclose(np.sum(np.abs(state.amplitudes) ** 2), 1.0)
