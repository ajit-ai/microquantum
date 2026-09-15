"""Execution example 7: seeded, reproducible sampling (MQ-11).

Re-running the same circuit with the same ``seed`` produces identical sampled
counts on the same backend; a different seed produces a fresh sample.
"""

from microquantum import QuantumCircuit, StatevectorBackend


def main():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    backend = StatevectorBackend()

    r1 = backend.run(qc, shots=1000, seed=42)
    r2 = backend.run(qc, shots=1000, seed=42)
    r3 = backend.run(qc, shots=1000, seed=7)

    print("=== Seeded execution ===")
    print(f"run(seed=42) #1 : {r1.counts}")
    print(f"run(seed=42) #2 : {r2.counts}")
    print(f"run(seed=7)     : {r3.counts}")

    assert r1.counts == r2.counts
    assert r1.samples == r2.samples
    assert r1.counts != r3.counts

    print(f"same seed reproducible : {r1.counts == r2.counts}")
    print(f"different seed differs : {r1.counts != r3.counts}")
    print(f"seed recorded on result: {r1.seed}")


if __name__ == "__main__":
    main()