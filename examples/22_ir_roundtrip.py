"""Example 22: Circuit -> IR -> Circuit round trip.

Demonstrates:
- Converting a ``QuantumCircuit`` into MicroQuantum IR (``to_ir``)
- Inspecting the IR: qubit count, gate-by-type counts, depth
- Rebuilding a circuit from the IR (``from_ir``) and verifying the
  behavior is preserved (state vector fidelity == 1)
- Optional terminal measurements appended to the IR
"""

from __future__ import annotations

import numpy as np

from microquantum import (
    Measurement,
    QuantumCircuit,
    StatevectorBackend,
    from_ir,
)

NUM_QUBITS = 3


def amplitudes(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(
        StatevectorBackend().run(circuit, shots=1, seed=7).statevector,
        dtype=np.complex128,
    )


def main() -> None:
    print("=== 22 Circuit -> IR -> Circuit round trip ===\n")

    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.7, 2)

    ir = qc.to_ir()
    print(f"circuit      : {NUM_QUBITS} qubits, {qc.gate_count()} gates")
    print(f"ir           : {ir.num_qubits} qubits, {ir.num_gates} gates")
    print(f"gate counts  : {ir.gate_names()}")
    print(f"ir depth     : {ir.depth}")

    rebuilt = from_ir(ir)
    a, b = amplitudes(qc), amplitudes(rebuilt)
    fidelity = float(abs(np.vdot(a, b)) ** 2)
    print(f"round-trip fidelity: {fidelity:.12f}")
    assert fidelity > 1.0 - 1e-9, "round trip changed the state"

    measured = qc.to_ir(include_terminal_measurements=True)
    terminals = [op for op in measured.operations if isinstance(op, Measurement)]
    print(f"terminal measurements with flag: {[m.qubit for m in terminals]}")
    assert [m.qubit for m in terminals] == [0, 1, 2]

    print("\nExample 22 completed!")


if __name__ == "__main__":
    main()