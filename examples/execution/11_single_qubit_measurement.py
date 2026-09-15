"""Execution example 5: single-qubit measurement (MQ-11).

Prepare |1>, measure it, and inspect the resulting counts.  This shows the
MQ-11 execution core: circuit measurement annotations (``measure_all``) are
explicit circuit data, and the backend performs the real shot-based sampling.
"""

from microquantum import QuantumCircuit, StatevectorBackend


def main():
    qc = QuantumCircuit(1)
    qc.x(0)             # prepare |1>
    qc.measure_all()    # annotate measurement (does not execute anything)

    print("=== Single-qubit measurement ===")
    print(f"circuit      : {qc!r}")
    print(f"measured     : {qc.measurements}")

    result = StatevectorBackend().run(qc, shots=1000, seed=42)

    print(f"counts       : {result.counts}")
    print(f"probabilities: {result.probabilities}")
    print(f"most frequent: {result.most_frequent()}")
    print(f"state        : {result.state}")
    print(f"shots        : {result.shots} (sum of counts = "
          f"{sum(result.counts.values())})")


if __name__ == "__main__":
    main()