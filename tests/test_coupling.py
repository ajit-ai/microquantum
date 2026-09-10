"""Tests for CouplingMap and routing passes."""

import pytest

from microquantum.core import QuantumCircuit
from microquantum.core.coupling import CouplingMap
from microquantum.core.transpiler import (
    NoiseAwarePlacementPass,
    PassManager,
    RoutingPass,
)


class TestCouplingMap:
    def test_linear_4(self):
        cmap = CouplingMap.linear(4)
        assert cmap.num_qubits == 4
        assert cmap.size == 3
        assert cmap.are_connected(0, 1)
        assert cmap.are_connected(2, 3)
        assert not cmap.are_connected(0, 3)

    def test_grid_2x3(self):
        cmap = CouplingMap.grid(2, 3)
        assert cmap.num_qubits == 6
        assert cmap.are_connected(0, 1)
        assert cmap.are_connected(0, 3)
        assert not cmap.are_connected(0, 5)

    def test_all_to_all(self):
        cmap = CouplingMap.all_to_all(4)
        assert cmap.num_qubits == 4
        assert cmap.size == 6
        for i in range(4):
            for j in range(i + 1, 4):
                assert cmap.are_connected(i, j)

    def test_shortest_path(self):
        cmap = CouplingMap.linear(5)
        assert cmap.shortest_path(0, 3) == [0, 1, 2, 3]
        assert cmap.shortest_path(0, 0) == [0]

    def test_distance(self):
        cmap = CouplingMap.linear(5)
        assert cmap.distance(0, 1) == 0
        assert cmap.distance(0, 2) == 1
        assert cmap.distance(0, 4) == 3

    def test_degree(self):
        cmap = CouplingMap.linear(4)
        assert cmap.degree(0) == 1
        assert cmap.degree(1) == 2
        assert cmap.degree(3) == 1

    def test_neighbors(self):
        cmap = CouplingMap.linear(4)
        assert cmap.neighbors(1) == {0, 2}

    def test_is_connected_graph(self):
        assert CouplingMap.linear(5).is_connected_graph()
        assert CouplingMap.all_to_all(3).is_connected_graph()

    def test_to_adjacency_list(self):
        cmap = CouplingMap.linear(3)
        adj = cmap.to_adjacency_list()
        assert adj == [[1], [0, 2], [1]]

    def test_from_adjacency_list(self):
        adj = [[1], [0, 2], [1]]
        cmap = CouplingMap.from_adjacency_list(adj)
        assert cmap.num_qubits == 3
        assert cmap.are_connected(0, 1)
        assert cmap.are_connected(1, 2)
        assert not cmap.are_connected(0, 2)

    def test_validation_bad_qubit(self):
        with pytest.raises(ValueError):
            CouplingMap([(0, 5)], num_qubits=3)

    def test_repr(self):
        cmap = CouplingMap.linear(4)
        assert "CouplingMap" in repr(cmap)

    def test_equality(self):
        a = CouplingMap.linear(4)
        b = CouplingMap.linear(4)
        assert a == b

    def test_heavy_hex(self):
        cmap = CouplingMap.heavy_hex(4, 4)
        assert cmap.num_qubits == 16
        assert cmap.size > 0


class TestRoutingPass:
    def test_no_routing_needed(self):
        cmap = CouplingMap.linear(4)
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        pm = PassManager()
        pm.append_pass(RoutingPass(cmap))
        result = pm.run(qc)
        assert result.num_gates > 0

    def test_routing_inserts_swaps(self):
        cmap = CouplingMap.linear(4)
        qc = QuantumCircuit(4)
        qc.cx(0, 3)  # not adjacent in linear topology
        pm = PassManager()
        pm.append_pass(RoutingPass(cmap))
        result = pm.run(qc)
        # Should have SWAPs + original CX
        assert result.num_gates >= 2

    def test_routing_preserves_single_qubit_gates(self):
        cmap = CouplingMap.linear(3)
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 2)
        pm = PassManager()
        pm.append_pass(RoutingPass(cmap))
        result = pm.run(qc)
        # H + SWAPs + CX
        assert result.num_gates >= 3

    def test_routing_too_many_qubits(self):
        cmap = CouplingMap.linear(2)
        qc = QuantumCircuit(4)
        pm = PassManager()
        pm.append_pass(RoutingPass(cmap))
        with pytest.raises(ValueError, match="qubits"):
            pm.run(qc)

    def test_integration_with_pass_manager(self):
        cmap = CouplingMap.linear(4)
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 3)
        qc.h(3)
        pm = PassManager.from_optimization_level(2, coupling_map=cmap)
        result = pm.run(qc)
        assert result.num_qubits == 4
        assert result.num_gates > 0


class TestNoiseAwarePlacementPass:
    def test_reorders_by_noise(self):
        cmap = CouplingMap.linear(4)
        noise = {(0, 1): 0.01, (1, 2): 0.5, (2, 3): 0.01}
        qc = QuantumCircuit(4)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.cx(2, 3)
        pm = PassManager()
        pm.append_pass(NoiseAwarePlacementPass(cmap, noise))
        result = pm.run(qc)
        assert result.num_gates == 3

    def test_parameterized_skip(self):
        from microquantum.core import Parameter
        cmap = CouplingMap.linear(3)
        noise = {(0, 1): 0.01}
        qc = QuantumCircuit(3)
        theta = Parameter("theta")
        qc.rx(theta, 0)
        qc.cx(0, 1)
        pm = PassManager()
        pm.append_pass(NoiseAwarePlacementPass(cmap, noise))
        result = pm.run(qc)
        assert result.is_parameterized
