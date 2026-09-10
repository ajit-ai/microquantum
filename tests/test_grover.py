"""Tests for Grover's search algorithm."""

import math

import numpy as np
import pytest

from microquantum.algorithms.grover import GroverResult, GroverSearch


class TestGroverSearchCreation:
    """Test GroverSearch initialization."""

    def test_single_target(self):
        gs = GroverSearch(num_qubits=3, target=5)
        assert gs.num_qubits == 3
        assert gs.targets == [5]

    def test_target_list(self):
        gs = GroverSearch(num_qubits=3, target=[1, 5])
        assert gs.targets == [1, 5]

    def test_custom_iterations(self):
        gs = GroverSearch(num_qubits=3, target=0, num_iterations=2)
        assert gs.num_iterations == 2

    def test_optimal_iterations(self):
        gs = GroverSearch(num_qubits=3, target=0)
        N = 8
        expected = max(1, int(math.floor(math.pi / 4 * math.sqrt(N))))
        assert gs.num_iterations == expected

    def test_repr(self):
        gs = GroverSearch(num_qubits=3, target=5)
        r = repr(gs)
        assert "GroverSearch" in r
        assert "3" in r

    def test_invalid_qubits(self):
        with pytest.raises(ValueError, match="Need >= 1 qubit"):
            GroverSearch(num_qubits=0, target=0)

    def test_target_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            GroverSearch(num_qubits=2, target=4)

    def test_list_target_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            GroverSearch(num_qubits=2, target=[0, 5])

    def test_invalid_iterations(self):
        with pytest.raises(ValueError, match="Need >= 1 iteration"):
            GroverSearch(num_qubits=3, target=0, num_iterations=0)


class TestGroverSearchCircuit:
    """Test circuit construction."""

    def test_circuit_qubits(self):
        gs = GroverSearch(num_qubits=3, target=5)
        qc = gs.build_circuit()
        assert qc.num_qubits == 3

    def test_circuit_has_gates(self):
        gs = GroverSearch(num_qubits=3, target=5)
        qc = gs.build_circuit()
        assert qc.num_gates > 0

    def test_circuit_num_gates_scales(self):
        gs1 = GroverSearch(num_qubits=2, target=1)
        gs2 = GroverSearch(num_qubits=4, target=1)
        qc1 = gs1.build_circuit()
        qc2 = gs2.build_circuit()
        assert qc2.num_gates > qc1.num_gates


class TestGroverSearchRun:
    """Test running the algorithm."""

    def test_single_target_success(self):
        gs = GroverSearch(num_qubits=3, target=5)
        result = gs.run()
        assert isinstance(result, GroverResult)
        assert result.num_qubits == 3
        assert result.target == 5
        assert result.success_probability > 0.5

    def test_multiple_targets_success(self):
        gs = GroverSearch(num_qubits=3, target=[1, 5])
        result = gs.run()
        assert result.success_probability > 0.5

    def test_result_probabilities_sum_to_one(self):
        gs = GroverSearch(num_qubits=3, target=0)
        result = gs.run()
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_result_num_probabilities(self):
        gs = GroverSearch(num_qubits=3, target=0)
        result = gs.run()
        assert len(result.probabilities) == 8

    def test_result_most_probable_is_target(self):
        gs = GroverSearch(num_qubits=3, target=5)
        result = gs.run()
        assert result.most_probable == 5

    def test_custom_iterations(self):
        gs = GroverSearch(num_qubits=3, target=5, num_iterations=2)
        result = gs.run()
        assert result.num_iterations == 2

    def test_zero_iterations_gives_uniform(self):
        gs = GroverSearch(num_qubits=3, target=0, num_iterations=1)
        result = gs.run()
        # With only 1 iteration, target should still get boosted
        assert result.success_probability > 0.1

    def test_two_qubit_search(self):
        gs = GroverSearch(num_qubits=2, target=3)
        result = gs.run()
        assert result.most_probable == 3
        assert result.success_probability > 0.5


class TestGroverOracle:
    """Test oracle construction."""

    def test_custom_oracle_fn(self):
        def my_oracle(n: int):
            from microquantum.core import QuantumCircuit
            from microquantum.core.operators import Operator
            dim = 2**n
            matrix = np.eye(dim, dtype=np.complex128)
            matrix[5, 5] = -1.0
            qc = QuantumCircuit(n)
            qc.append(Operator(matrix), list(range(n)))
            return qc

        gs = GroverSearch(num_qubits=3, target=my_oracle)
        result = gs.run()
        assert result.most_probable == 5
        assert result.probabilities["101"] > 0.5
