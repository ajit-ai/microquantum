"""Tests for DensityMatrixBackend."""

import numpy as np
import pytest

from microquantum.backends.density_matrix import DensityMatrixBackend
from microquantum.core import Operator, StateVector


class TestDensityMatrixBackend:
    def setup_method(self) -> None:
        self.backend = DensityMatrixBackend()

    def test_name(self) -> None:
        assert self.backend.name == "density_matrix"

    def test_empty_circuit(self) -> None:
        result = self.backend.run_circuit(
            num_qubits=1, gates=[], shots=1000, seed=42
        )
        assert result.num_qubits == 1
        assert result.density_matrix is not None
        assert result.counts["0"] == 1000

    def test_x_gate(self) -> None:
        x_matrix = Operator.X().matrix
        result = self.backend.run_circuit(
            num_qubits=1, gates=[(x_matrix, [0])], shots=1000, seed=42
        )
        assert result.counts.get("1", 0) == 1000

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

    def test_initial_state(self) -> None:
        state = StateVector(1, amplitudes=np.array([0, 1], dtype=np.complex128))
        result = self.backend.run_circuit(
            num_qubits=1, gates=[], shots=100, initial_state=state, seed=42
        )
        assert result.counts.get("1", 0) == 100

    def test_wrong_size_initial_state_raises(self) -> None:
        state = StateVector(2)
        with pytest.raises(ValueError, match="Initial state"):
            self.backend.run_circuit(num_qubits=1, gates=[], initial_state=state)

    def test_metadata_purity(self) -> None:
        result = self.backend.run_circuit(
            num_qubits=1, gates=[], shots=100, seed=42
        )
        assert result.metadata["purity"]

    def test_density_matrix_trace(self) -> None:
        result = self.backend.run_circuit(
            num_qubits=2, gates=[], shots=100, seed=42
        )
        assert result.density_matrix is not None
        trace = np.trace(result.density_matrix)
        assert trace == pytest.approx(1.0)

    def test_with_noise_model(self) -> None:
        from microquantum.backends.noise import NoiseModel

        noise = NoiseModel().depolarizing(0.1)
        h_matrix = Operator.H().matrix
        result = self.backend.run_circuit(
            num_qubits=1,
            gates=[(h_matrix, [0])],
            shots=10000,
            seed=42,
            noise_model=noise,
        )
        # With noise, superposition is partially depolarized
        assert "0" in result.counts
        assert "1" in result.counts
        assert not result.metadata["purity"]


class TestDensityMatrixBackendExpandGate:
    def test_expand_single_qubit(self) -> None:
        from microquantum.core.operators import Operator
        x = Operator.X().matrix
        full = DensityMatrixBackend._expand_gate(x, [0], 2)
        assert full.shape == (4, 4)
        # X on qubit 0 should be X ⊗ I
        expected = np.kron(x, np.eye(2))
        np.testing.assert_allclose(full, expected)

    def test_expand_two_qubit(self) -> None:
        from microquantum.core.operators import Operator
        cnot = Operator.CNOT().matrix
        full = DensityMatrixBackend._expand_gate(cnot, [0, 1], 3)
        assert full.shape == (8, 8)
