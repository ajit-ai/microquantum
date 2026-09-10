"""IR example 1: circuit -> IR.

Demonstrates converting a static QuantumCircuit into MicroQuantum IR
(the internal, hardware-neutral representation) and printing it.
"""

from microquantum import QuantumCircuit, assert_valid


def main():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.rz(0.25, 1)
    qc.to_ir()

    print("=== Constructed circuit ===")
    print(qc.qasm(header=False))

    print("=== IR (with terminal measurements) ===")
    measured = qc.to_ir(include_terminal_measurements=True)
    print(measured)

    assert_valid(measured)
    print("\nIR validates cleanly.")


if __name__ == "__main__":
    main()