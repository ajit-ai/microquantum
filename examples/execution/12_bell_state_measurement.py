"""Execution example 6: Bell-state measurement and sampling (MQ-11).

Build a Bell state H|0> then CX, measure all qubits and sample 1000 shots.
The sampled distribution should be dominated by ``00`` and ``11``.
"""

from microquantum import QuantumCircuit, StatevectorBackend


def main():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    print("=== Bell state sampling (1000 shots) ===")
    result = StatevectorBackend().run(qc, shots=1000, seed=42)
    for bitstring in sorted(result.counts):
        bar = "#" * (result.counts[bitstring] // 20)
        print(f"  |{bitstring}>: {result.counts[bitstring]:4d} {bar}")

    total = sum(result.counts.values())
    assert total == 1000
    assert set(result.counts.keys()) <= {"00", "11"}
    print(f"total shots : {total}")
    print(f"only '00'/'11' observed: "
          f"{set(result.counts.keys()) <= {'00', '11'}}")
    print(f"state vector: {result.state}")


if __name__ == "__main__":
    main()