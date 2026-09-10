"""Execution example 1: basic execution.

The one-line way to run a circuit: :func:`microquantum.execute` handles
planning, backend selection (a local state-vector simulator) and collection,
returning an enriched :class:`BackendResult`.
"""

from microquantum import QuantumCircuit, execute


def main():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)

    result = execute(qc, shots=512, seed=7)

    print("=== Execution ===")
    print(f"backend   : {result.metadata['backend']}")
    print(f"strategy  : {result.metadata['strategy']}")
    print(f"job id    : {result.metadata['job_id']}")
    print(f"counts    : {result.counts}")
    print(f"prob('000'): {result.probabilities.get('000', 0.0):.3f}")
    print(f"elapsed   : {result.metadata['elapsed_seconds'] * 1000:.2f} ms")
    print("states with nonzero amplitude:", result.probabilities)


if __name__ == "__main__":
    main()