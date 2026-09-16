"""Example 25: Optimization before / after.

Demonstrates:
- Building a circuit with a mix of optimizable and essential gates
- Comparing gate counts, depth and per-gate-type metrics before and
  after compilation at level 1
- Proving the optimization preserves the state vector exactly
"""

from __future__ import annotations

import numpy as np

from microquantum import Compiler, QuantumCircuit, StatevectorBackend


def amplitudes(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(
        StatevectorBackend().run(circuit, shots=1, seed=7).statevector,
        dtype=np.complex128,
    )


def main() -> None:
    print("=== 25 Optimization before / after ===\n")

    qc = QuantumCircuit(3)
    qc.s(0)
    qc.sdg(0)          # inverse pair -> identity
    qc.h(0)
    qc.rz(1.1, 1)
    qc.rz(-1.1, 1)     # opposite rotations -> identity
    qc.cx(0, 1)
    qc.z(2)
    qc.z(2)            # self-inverse pair -> identity
    qc.ry(0.6, 2)

    print("before:")
    print(f"  gates: {qc.gate_count():2d}  depth: {qc.depth():2d}  "
          f"types: {qc.to_ir().gate_names()}")

    result = Compiler(optimization_level=1).compile(qc)
    compiled = result.circuit()
    meta = result.metadata

    print("after:")
    print(f"  gates: {compiled.gate_count():2d}  depth: {compiled.depth():2d}  "
          f"types: {compiled.to_ir().gate_names()}")

    removed = meta["source_gates"] - meta["compiled_gates"]
    print(f"removed gates               : {removed}")
    print(f"depth reduction             : "
          f"{meta['source_depth'] - meta['compiled_depth']}")

    a, b = amplitudes(qc), amplitudes(compiled)
    fidelity = float(abs(np.vdot(a, b)) ** 2)
    print(f"state fidelity after        : {fidelity:.12f}")

    assert removed > 0, "expected the optimizer to remove at least one gate"
    assert meta["compiled_gates"] == compiled.gate_count()
    assert fidelity > 1.0 - 1e-9, "optimization changed the state"

    print("\nExample 25 completed!")


if __name__ == "__main__":
    main()