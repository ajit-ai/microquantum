"""Phase 121 W6: foundations (stdlib + core micro-adds)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from microquantum.core.gates import ControlledUnitary, H
from microquantum.core.information import concurrence, entanglement_entropy
from microquantum.stdlib import dicke_state, graph_state, gray_code


class TestStdlibAdditions:
    def test_gray_code(self) -> None:
        assert gray_code(1) == ["0", "1"]
        assert gray_code(2) == ["00", "01", "11", "10"]
        words = gray_code(3)
        assert len(words) == 8 and words[0] == "000"
        for first, second in zip(words, words[1:], strict=False):
            assert sum(a != b for a, b in zip(first, second, strict=True)) == 1
        with pytest.raises(ValueError):
            gray_code(0)
        with pytest.raises(TypeError):
            gray_code(True)  # type: ignore[arg-type]

    def test_dicke_state(self) -> None:
        state = dicke_state(3, 1)
        assert state.is_normalized
        assert state.amplitudes[0] == pytest.approx(0.0)
        assert abs(state.amplitudes[1]) == pytest.approx(1.0 / math.sqrt(3))
        assert dicke_state(2, 2).amplitudes[3] == pytest.approx(1.0)
        with pytest.raises(ValueError):
            dicke_state(2, 3)
        with pytest.raises(ValueError):
            dicke_state(0, 0)

    def test_graph_state_matches_circuit(self) -> None:
        from microquantum.core.circuit import QuantumCircuit

        edges = [(0, 1), (1, 2)]
        factory = graph_state(edges, 3)
        assert factory.is_normalized
        circuit = QuantumCircuit(3)
        for qubit in range(3):
            circuit.h(qubit)
        for first, second in edges:
            circuit.cz(first, second)
        from microquantum.core.state import StateVector

        unitary = circuit.get_unitary().matrix
        zero = np.zeros(8, dtype=complex)
        zero[0] = 1.0
        expected = StateVector(3, amplitudes=unitary @ zero)
        assert np.allclose(factory.amplitudes, expected.amplitudes, atol=1e-12)
        with pytest.raises(ValueError):
            graph_state([(0, 0)], 2)
        with pytest.raises(ValueError):
            graph_state([(0, 5)], 2)


class TestCoreMicroAdds:
    def test_controlled_unitary(self) -> None:
        gate = ControlledUnitary(H().to_matrix(), num_controls=1)
        assert gate.num_qubits == 2
        assert gate.num_controls == 1
        matrix = gate.to_matrix()
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix[2:, 2:], H().to_matrix(), atol=1e-12)
        double = ControlledUnitary(np.eye(2), num_controls=2)
        assert double.num_qubits == 3
        with pytest.raises(ValueError):
            ControlledUnitary(np.eye(2), num_controls=0)
        with pytest.raises(ValueError):
            ControlledUnitary(np.eye(3))

    def test_entanglement_measures(self) -> None:
        bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        assert entanglement_entropy(bell, [0]) == pytest.approx(1.0)
        assert entanglement_entropy(np.array([1, 0, 0, 0], dtype=complex), [0]) == pytest.approx(0.0)
        assert concurrence(bell) == pytest.approx(1.0)
        assert concurrence(np.eye(4, dtype=complex) / 4) == pytest.approx(0.0)
        with pytest.raises(ValueError):
            concurrence(np.eye(2, dtype=complex))
        with pytest.raises(ValueError):
            entanglement_entropy(bell, [9])
