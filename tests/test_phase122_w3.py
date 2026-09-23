"""Phase 122 W3: execution at scale."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.backends import (
    MPSBackend,
    StatevectorBackend,
    TreeTensorNetworkBackend,
    available_backends,
    get_array_backend,
    gpu_available,
    is_gpu,
    set_array_backend,
)
from microquantum.backends.array_backend import (
    asarray,
    einsum,
    eye,
    matmul,
    tensordot,
    to_numpy,
    zeros,
)
from microquantum.core.circuit import QuantumCircuit
from microquantum.runtime import Budget, DAGScheduler, ExecutionPlan
from microquantum.runtime.errors import PlanningError


class TestArrayBackend:
    def test_cpu_ops(self) -> None:
        assert np.allclose(to_numpy(eye(2)), np.eye(2))
        assert to_numpy(zeros((2, 2))).shape == (2, 2)
        assert np.allclose(
            to_numpy(matmul(asarray([[0, 1], [1, 0]]), asarray([[1, 0], [0, -1]]))),
            [[0, -1], [1, 0]],
        )
        assert np.allclose(
            to_numpy(einsum("ij,jk->ik", asarray(np.eye(2)), asarray(np.eye(2)))),
            np.eye(2),
        )
        assert np.allclose(
            to_numpy(tensordot(asarray([1, 0]), asarray([0, 1]), axes=0)),
            [[0, 1], [0, 0]],
        )

    def test_backend_selection(self) -> None:
        previous = get_array_backend()
        try:
            set_array_backend("numpy")
            assert get_array_backend() == "numpy"
            assert not is_gpu()
            assert "numpy" in available_backends()
        finally:
            set_array_backend(previous)
        with pytest.raises(ValueError):
            set_array_backend("not-a-backend")

    @pytest.mark.skipif(not gpu_available(), reason="requires a GPU array backend")
    def test_gpu_ops(self) -> None:
        set_array_backend("cupy")
        try:
            assert is_gpu()
            assert np.allclose(to_numpy(eye(2)), np.eye(2))
        finally:
            set_array_backend("numpy")


class TestScheduledBatchExecution:
    def test_scheduler_orders_execution(self) -> None:
        from microquantum.runtime import default_runtime

        circuits = [QuantumCircuit(1) for _ in range(3)]
        for circuit in circuits:
            circuit.x(0)
        scheduler = DAGScheduler(max_parallel=1)
        results = default_runtime.execute_batch(
            circuits, shots=8, seed=0, scheduler=scheduler
        )
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, str)
            assert result.get_counts() == {"1": 8}

    def test_scheduler_dependency_levels(self) -> None:
        from microquantum.runtime import default_runtime

        circuits = [QuantumCircuit(1) for _ in range(3)]
        scheduler = DAGScheduler(max_parallel=3)
        results = default_runtime.execute_batch(
            circuits, shots=4, seed=1, scheduler=scheduler
        )
        assert len(results) == 3

    def test_scheduler_rejects_wrong_type(self) -> None:
        from microquantum.runtime import default_runtime

        with pytest.raises(TypeError):
            default_runtime.execute_batch(
                [QuantumCircuit(1)], shots=4, scheduler=object()  # type: ignore[arg-type]
            )

    def test_module_level_forwarding(self) -> None:
        from microquantum.runtime import execute_batch

        results = execute_batch(
            [QuantumCircuit(1)], shots=4, seed=0, scheduler=DAGScheduler(max_parallel=2)
        )
        assert len(results) == 1


class TestBudgetDispatch:
    def test_over_budget_plan_rejected(self) -> None:
        from microquantum.runtime import default_runtime

        plan = ExecutionPlan.from_circuit(QuantumCircuit(1), shots=1000)
        plan = __import__("dataclasses").replace(plan, budget=Budget(max_shots=10))
        with pytest.raises(PlanningError):
            default_runtime.execute(plan)

    def test_within_budget_executes(self) -> None:
        from microquantum.runtime import default_runtime

        plan = ExecutionPlan.from_circuit(
            QuantumCircuit(1), shots=8, budget=Budget(max_shots=10)
        )
        result = default_runtime.execute(plan)
        assert sum(result.get_counts().values()) == 8


class TestTensorNetworkDepth:
    def _ladder(self, num_qubits: int, depth: int) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits)
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.h(qubit)
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        return circuit

    def test_mps_depth_scaling(self) -> None:
        backend = MPSBackend()
        for depth in (1, 2, 3, 4):
            circuit = self._ladder(3, depth)
            gates = [
                (np.asarray(op.matrix, dtype=np.complex128), list(targets))
                for op, targets in circuit.gates
            ]
            result = backend.run_circuit(3, gates, shots=64, seed=0)
            total = sum(result.get_counts().values())
            assert total == 64

    def test_ttn_depth_scaling(self) -> None:
        backend = TreeTensorNetworkBackend()
        for depth in (1, 2, 3):
            circuit = self._ladder(3, depth)
            gates = [
                (np.asarray(op.matrix, dtype=np.complex128), list(targets))
                for op, targets in circuit.gates
            ]
            result = backend.run_circuit(3, gates, shots=64, seed=0)
            assert sum(result.get_counts().values()) == 64

    def test_statevector_reference(self) -> None:
        backend = StatevectorBackend()
        circuit = self._ladder(2, 2)
        gates = [
            (np.asarray(op.matrix, dtype=np.complex128), list(targets))
            for op, targets in circuit.gates
        ]
        result = backend.run_circuit(2, gates, shots=16, seed=0)
        assert sum(result.get_counts().values()) == 16
