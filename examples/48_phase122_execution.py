"""Phase 122 W3 tour: execution at scale."""

from __future__ import annotations

import numpy as np

from microquantum.backends import MPSBackend, StatevectorBackend
from microquantum.backends.array_backend import asarray, eye, matmul, to_numpy
from microquantum.core.circuit import QuantumCircuit
from microquantum.runtime import Budget, DAGScheduler, ExecutionPlan, execute_batch


def main() -> None:
    print("matmul:", to_numpy(matmul(asarray([[0, 1], [1, 0]]), eye(2))).tolist())

    plans = [ExecutionPlan.from_circuit(QuantumCircuit(1), shots=8) for _ in range(3)]
    results = execute_batch(plans, seed=0, scheduler=DAGScheduler(max_parallel=2))
    print("scheduled results:", len(results))

    plan = ExecutionPlan.from_circuit(QuantumCircuit(1), shots=8, budget=Budget(max_shots=10))
    from microquantum.runtime import default_runtime

    print("budgeted shots:", sum(default_runtime.execute(plan).get_counts().values()))

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    gates = [(np.asarray(op.matrix, dtype=np.complex128), list(t)) for op, t in circuit.gates]
    result = MPSBackend().run_circuit(2, gates, shots=32, seed=0)
    print("mps counts:", result.get_counts())
    print("reference:", StatevectorBackend().run_circuit(2, gates, shots=32, seed=0).get_counts())


if __name__ == "__main__":
    main()
