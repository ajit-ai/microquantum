"""Tests for StatevectorBackend."""

import numpy as np
import pytest

from microquantum.backends.statevector import StatevectorBackend
from microquantum.core import Operator, QuantumCircuit, StateVector


class TestStatevectorBackendBasic:
    def setup_method(self) -> None:
        self.backend = StatevectorBackend()

    def test_name(self) -> None:
        assert self.backend.name == "statevector"

    def test_empty_circuit(self) -> None:
        result = self.backend.run_circuit(
            num_qubits=1, gates=[], shots=1000, seed=42
        )
        assert result.num_qubits == 1
        assert result.backend_name == "statevector"
        assert "0" in result.counts
        assert result.counts["0"] == 1000
        assert result.statevector is not None

    def test_x_gate(self) -> None:
        x_matrix = Operator.X().matrix
        result = self.backend.run_circuit(
            num_qubits=1, gates=[(x_matrix, [0])], shots=1000, seed=42
        )
        assert result.counts.get("1", 0) == 1000

    def test_hadamard_superposition(self) -> None:
        h_matrix = Operator.H().matrix
        result = self.backend.run_circuit(
            num_qubits=1, gates=[(h_matrix, [0])], shots=10000, seed=42
        )
        assert "0" in result.counts
        assert "1" in result.counts
        ratio = result.counts["0"] / result.counts["1"]
        assert 0.8 < ratio < 1.2

    def test_bell_state(self) -> None:
        h_matrix = Operator.H().matrix
        cnot_matrix = Operator.CNOT().matrix
        result = self.backend.run_circuit(
            num_qubits=2,
            gates=[(h_matrix, [0]), (cnot_matrix, [0, 1])],
            shots=10000,
            seed=42,
        )
        assert "00" in result.counts
        assert "11" in result.counts
        assert result.counts.get("01", 0) < 200
        assert result.counts.get("10", 0) < 200

    def test_statevector_preserved(self) -> None:
        h_matrix = Operator.H().matrix
        result = self.backend.run_circuit(
            num_qubits=1, gates=[(h_matrix, [0])], shots=100, seed=42
        )
        amp = result.statevector
        assert amp is not None
        expected = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        np.testing.assert_allclose(np.abs(amp), np.abs(expected), atol=1e-10)

    def test_initial_state(self) -> None:
        state = StateVector(1, amplitudes=np.array([0, 1], dtype=np.complex128))
        result = self.backend.run_circuit(
            num_qubits=1, gates=[], shots=100, initial_state=state, seed=42
        )
        assert result.counts.get("1", 0) == 100

    def test_wrong_size_initial_state_raises(self) -> None:
        state = StateVector(2)
        with pytest.raises(ValueError, match="Initial state"):
            self.backend.run_circuit(
                num_qubits=1, gates=[], initial_state=state
            )


class TestStatevectorBackendWithCircuit:
    def setup_method(self) -> None:
        self.backend = StatevectorBackend()

    def test_run_quantum_circuit(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        gates = [(op.matrix, targets) for op, targets in qc.gates]
        result = self.backend.run_circuit(
            num_qubits=2, gates=gates, shots=10000, seed=42
        )
        assert "00" in result.counts
        assert "11" in result.counts

    def test_metadata(self) -> None:
        result = self.backend.run_circuit(
            num_qubits=1, gates=[], shots=500, seed=42
        )
        assert result.metadata["shots"] == 500
