"""Phase 121 W2 tour: compilation and execution extensions."""

from __future__ import annotations

from microquantum.backends import AsyncJob, JobStatus, RetryPolicy, with_retry
from microquantum.core.circuit import QuantumCircuit
from microquantum.ir import AliasAnalysis, Compiler, CostModel, Gate, IRCircuit, Loop, to_ir
from microquantum.runtime import DAGScheduler, ExecutionPlan, ResultCache


def main() -> None:
    circuit = IRCircuit(num_qubits=2)
    circuit.add(Loop(body=(Gate(name="h", qubits=(0,)),), trip_count=3))
    print("loop unrolled gates:", len(circuit[0].unrolled()))  # type: ignore[union-attr]

    bell = QuantumCircuit(2)
    bell.h(0).cx(0, 1)
    model = CostModel(gate_costs={"h": 2.0, "cnot": 5.0})
    compiled = Compiler(cost_model=model).compile(bell)
    print("estimated cost:", compiled.metadata["estimated_cost"])

    analysis = AliasAnalysis()
    analysis.run(to_ir(bell))
    print("interacting pairs:", analysis.interacting_pairs())

    job = AsyncJob(poll_interval_s=0.0)
    print("poll:", job.poll(lambda: JobStatus.COMPLETED).value)

    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionError("down")
        return "up"

    print("retry:", with_retry(RetryPolicy(max_attempts=2, backoff_s=0.0), flaky))

    plans = [ExecutionPlan.from_circuit(QuantumCircuit(1), shots=10) for _ in range(4)]
    batches = DAGScheduler(max_parallel=2).schedule(plans, dependencies={3: {0, 1}})
    print("batches:", [(b.level, b.indices) for b in batches])

    cache = ResultCache(max_entries=8)
    cache.put(plans[0], {"value": 1})
    print("cached:", cache.get(plans[0]))


if __name__ == "__main__":
    main()
