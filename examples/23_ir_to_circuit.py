"""Example 23: Building IR and converting it back to a circuit.

Demonstrates:
- Constructing an ``IRCircuit`` by hand with ``Gate`` nodes
- Structural validation (``validate`` / ``assert_valid``)
- Rebuilding an executable ``QuantumCircuit`` with ``from_ir``
- Loud, explicit failures for malformed IR (never silent repair)
"""

from __future__ import annotations

import numpy as np

from microquantum import (
    Gate,
    IRCircuit,
    QuantumCircuit,
    StatevectorBackend,
    assert_valid,
    from_ir,
    validate,
)


def amplitudes(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(
        StatevectorBackend().run(circuit, shots=1, seed=7).statevector,
        dtype=np.complex128,
    )


def main() -> None:
    print("=== 23 IR -> Circuit ===\n")

    ir = IRCircuit(
        num_qubits=2,
        name="hand-built",
        operations=[
            Gate(name="h", qubits=(0,)),
            Gate(name="cnot", qubits=(0, 1)),
            Gate(name="rz", qubits=(1,), params=(0.5,)),
        ],
    )
    print("hand-built IR:")
    print(f"  {ir.name}: qubits={ir.num_qubits} gates={ir.num_gates} "
          f"depth={ir.depth}")
    print(f"  gate counts: {ir.gate_names()}")

    errors = validate(ir)
    print(f"validation errors: {errors}")
    assert errors == [], "hand-built IR should validate cleanly"
    assert_valid(ir)

    rebuilt = from_ir(ir)
    expected = QuantumCircuit(2).h(0).cx(0, 1).rz(0.5, 1)
    a, b = amplitudes(rebuilt), amplitudes(expected)
    fidelity = float(abs(np.vdot(a, b)) ** 2)
    print(f"rebuilt vs equivalent circuit fidelity: {fidelity:.12f}")
    assert fidelity > 1.0 - 1e-9

    print("\nMalformed IR is rejected explicitly:")
    bad_cases = [
        IRCircuit(1, operations=[Gate(name="mystery", qubits=(0,))]),
        IRCircuit(1, operations=[Gate(name="h", qubits=(7,))]),
        IRCircuit(1, operations=[Gate(name="rx", qubits=(0,), params=("nope",))]),
    ]
    for bad in bad_cases:
        try:
            from_ir(bad)
        except ValueError as exc:
            print(f"  rejected: {str(exc).splitlines()[-1]}")
        else:  # pragma: no cover - the whole point of the example
            raise AssertionError("malformed IR was accepted")

    print("\nExample 23 completed!")


if __name__ == "__main__":
    main()