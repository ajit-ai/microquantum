"""Execution example 4: batch execution.

:func:`microquantum.execute_batch` runs many plans in one go — useful for
sampling the same circuit under different seeds or configurations. With
``raise_on_error=False`` a single bad item is reported in place instead of
discarding the whole batch.
"""

from microquantum import BackendResult, QuantumCircuit, execute_batch


def make(name_head: int, seed: int) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(float(name_head) * 0.1, 1)
    return qc


def main():
    circuits = [make(i, seed=i) for i in range(4)]

    print("=== Batch (4 executions) ===")
    results = execute_batch(circuits, shots=512, seed=9)
    for index, result in enumerate(results):
        batch = result.metadata["batch_id"]
        print(
            f"item {index}: batch={batch} "
            f"counts_top={sorted(result.counts.items(), key=lambda kv: -kv[1])[0]}"
        )

    print("\n=== Batch that tolerates a failing item ===")
    circuits.append(QuantumCircuit(1))  # fine
    circuits.append(circuits[0])  # fine
    results = execute_batch(circuits, shots=64, raise_on_error=False)
    good = sum(isinstance(r, BackendResult) for r in results)
    print(f"completed {good}/{len(results)} items")


if __name__ == "__main__":
    main()