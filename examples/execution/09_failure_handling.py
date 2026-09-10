"""Execution example 9: failure handling.

The runtime surfaces failures deterministically: invalid plans, incompatible
targets and backend job failures all raise ``ValueError``.  Inspect jobs to
see the failure inside the lifecycle envelope, and use
``raise_on_error=False`` to collect partially-failed batches.
"""

from microquantum import (
    ExecutionPlan,
    Parameter,
    JobStatus,
    QuantumCircuit,
    StateVector,
    submit,
    execute_batch,
)


def main():
    print("=== 1. Plan validation failure ===")
    theta = Parameter("theta")
    unbound = ExecutionPlan.from_circuit(
        QuantumCircuit(1).rx(theta, 0)
    )
    try:
        from microquantum import execute

        execute(unbound)
    except ValueError as exc:
        print(f"ValueError: {exc}")

    print("\n=== 2. Backend job failure (mismatched initial state) ===")
    qc = QuantumCircuit(2)
    qc.x(0)
    bad = ExecutionPlan(
        circuit=qc, shots=8, initial_state=StateVector(1), options={}
    )
    job = submit(bad)
    print(f"job.status: {job.status.value}")
    print(f"job.error : {job.error}")

    print("\n=== 3. Batch continues past a bad item ===")
    good = ExecutionPlan.from_circuit(QuantumCircuit(1), shots=8)
    results = execute_batch([good, bad], raise_on_error=False, shots=8)
    for item in results:
        kind = "BackendResult" if "job_id" in getattr(item, "metadata", {}) else "error"
        print(f"  item type: {kind}")


if __name__ == "__main__":
    main()