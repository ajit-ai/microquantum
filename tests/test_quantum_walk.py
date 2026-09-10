"""Tests for Quantum Walk algorithms."""

import math

import numpy as np
import pytest

from microquantum.algorithms.quantum_walk import (
    ContinuousQuantumWalk,
    DiscreteQuantumWalk,
    QuantumWalkResult,
)


def _cycle_graph(n: int) -> list[list[int]]:
    return [[(i - 1) % n, (i + 1) % n] for i in range(n)]


def _complete_graph(n: int) -> list[list[int]]:
    return [[j for j in range(n) if j != i] for i in range(n)]


class TestDiscreteQuantumWalk:
    def test_cycle_4_nodes(self):
        adj = _cycle_graph(4)
        walk = DiscreteQuantumWalk(adj, num_steps=2)
        result = walk.run()
        assert isinstance(result, QuantumWalkResult)
        assert result.num_nodes == 4

    def test_result_has_probabilities(self):
        adj = _cycle_graph(4)
        walk = DiscreteQuantumWalk(adj, num_steps=2)
        result = walk.run()
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_default_steps(self):
        adj = _cycle_graph(6)
        walk = DiscreteQuantumWalk(adj)
        assert walk.num_steps >= 1

    def test_complete_graph_3(self):
        adj = _complete_graph(3)
        walk = DiscreteQuantumWalk(adj, num_steps=3)
        result = walk.run()
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=0.1)

    def test_build_circuit(self):
        adj = _cycle_graph(4)
        walk = DiscreteQuantumWalk(adj, num_steps=2)
        qc = walk.build_circuit()
        assert qc.num_qubits > 0
        assert qc.num_gates > 0

    def test_validation_empty(self):
        with pytest.raises(ValueError):
            DiscreteQuantumWalk([])

    def test_validation_bad_steps(self):
        with pytest.raises(ValueError):
            DiscreteQuantumWalk(_cycle_graph(4), num_steps=0)

    def test_repr(self):
        walk = DiscreteQuantumWalk(_cycle_graph(4), num_steps=3)
        assert "DiscreteQuantumWalk" in repr(walk)

    def test_properties(self):
        adj = _cycle_graph(5)
        walk = DiscreteQuantumWalk(adj, num_steps=4)
        assert walk.num_nodes == 5
        assert walk.num_steps == 4


class TestContinuousQuantumWalk:
    def test_cycle_4(self):
        adj = _cycle_graph(4)
        walk = ContinuousQuantumWalk(adj, evolution_time=1.0)
        result = walk.run()
        assert isinstance(result, QuantumWalkResult)
        assert result.num_nodes == 4

    def test_result_probabilities(self):
        adj = _cycle_graph(4)
        walk = ContinuousQuantumWalk(adj, evolution_time=0.5)
        result = walk.run()
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=0.1)

    def test_default_time(self):
        adj = _cycle_graph(4)
        walk = ContinuousQuantumWalk(adj)
        assert walk.evolution_time > 0

    def test_build_circuit(self):
        adj = _cycle_graph(4)
        walk = ContinuousQuantumWalk(adj, evolution_time=0.5)
        qc = walk.build_circuit()
        assert qc.num_qubits > 0

    def test_validation(self):
        with pytest.raises(ValueError):
            ContinuousQuantumWalk([])

    def test_repr(self):
        walk = ContinuousQuantumWalk(_cycle_graph(4))
        assert "ContinuousQuantumWalk" in repr(walk)
