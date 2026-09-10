"""Example 1: Basic circuits, simulation, and I/O.

Demonstrates:
- Building quantum circuits with method chaining
- Running circuits on the statevector simulator
- Drawing ASCII circuit diagrams
- Exporting/importing QASM 2.0
- Saving/loading circuits as JSON
"""

from microquantum.core import QuantumCircuit
from microquantum.core.operators import Operator


def bell_state():
    """Create a Bell state |00> + |11>."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    state = qc.run()
    probs = abs(state.amplitudes) ** 2
    print("=== Bell State ===")
    print(f"|00>: {probs[0]:.4f}")
    print(f"|11>: {probs[3]:.4f}")
    return qc


def ghz_state():
    """Create a 4-qubit GHZ state."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    state = qc.run()
    probs = abs(state.amplitudes) ** 2
    print("\n=== 4-Qubit GHZ State ===")
    for i, p in enumerate(probs):
        if p > 0.01:
            print(f"|{i:04b}>: {p:.4f}")
    return qc


def method_chaining():
    """Demonstrate fluent circuit construction."""
    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).cx(1, 2)  # Chain multiple gates

    print("\n=== Method Chaining ===")
    print(f"Circuit: {qc.num_qubits} qubits, {qc.num_gates} gates, depth {qc.depth}")
    print(qc.draw())
    return qc


def circuit_io():
    """QASM export/import and JSON save/load."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.5, 1)

    # QASM
    qasm = qc.qasm()
    print("\n=== QASM 2.0 ===")
    print(qasm)
    qc2 = QuantumCircuit.from_qasm(qasm)
    print(f"Imported back: {qc2.num_qubits} qubits, {qc2.num_gates} gates")

    # JSON
    qc.save("my_circuit.json")
    loaded = QuantumCircuit.load("my_circuit.json")
    print(f"Loaded from JSON: {loaded.num_qubits} qubits, {loaded.num_gates} gates")


def multi_qubit_gates():
    """Demonstrate various gate types."""
    qc = QuantumCircuit(3)
    qc.h(0)                          # Hadamard
    qc.x(1)                          # Pauli-X (NOT)
    qc.y(2)                          # Pauli-Y
    qc.append(Operator.Z(), [0])     # Pauli-Z
    qc.append(Operator.S(), [1])     # S gate
    qc.append(Operator.T(), [2])     # T gate
    qc.cx(0, 1)                      # CNOT
    qc.append(Operator.CZ(), [1, 2]) # Controlled-Z
    qc.swap(0, 2)                    # SWAP
    qc.ry(0.785, 0)                  # Ry rotation
    qc.rz(1.2, 1)                    # Rz rotation

    print("\n=== Multi-Qubit Gates ===")
    print(qc.draw())
    print(f"Gates: {qc.num_gates}, Depth: {qc.depth}")


if __name__ == "__main__":
    bell_state()
    ghz_state()
    method_chaining()
    multi_qubit_gates()
    circuit_io()

    # Cleanup
    import os
    for f in ["my_circuit.json"]:
        if os.path.exists(f):
            os.remove(f)

    print("\nAll examples completed!")
