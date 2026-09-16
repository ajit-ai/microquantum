"""Example 24: Basic compilation pipeline.

Demonstrates:
- The single public compiler entry point (``Compiler``)
- Optimization levels: 0 = validation only, 1 = identity removal +
  inverse cancellation, 2 = also fuse same-axis rotations
- ``CompilationResult``: applied passes, diagnostics, metadata
- Rebuilding an executable circuit from the compiled IR
"""

from __future__ import annotations

import numpy as np

from microquantum import Compiler, ExecutionPlan, QuantumCircuit, StatevectorBackend


def amplitudes(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(
        StatevectorBackend().run(circuit, shots=1, seed=7).statevector,
        dtype=np.complex128,
    )


def main() -> None:
    print("=== 24 Basic compilation ===\n")

    target_plan = ExecutionPlan.from_circuit(QuantumCircuit(2), shots=16)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(0)          # cancels: identity pair
    qc.cx(0, 1)
    qc.rz(0.0, 1)    # zero rotation: structural identity

    print(f"input circuit : {qc.gate_count()} gates, depth {qc.depth()}")
    print(f"plan level is {target_plan.optimization_level} by default\n")

    for level in (0, 1, 2):
        result = Compiler(optimization_level=level).compile(qc)
        print(f"level {level}")
        print(f"  passes      : {result.passes_applied}")
        print(f"  gates       : {result.source.num_gates} -> {result.result.num_gates}")
        print(f"  gate counts : {result.result.gate_names()}")
        print(f"  compatible  : {result.is_compatible}")
        print(f"  depth       : {result.metadata['source_depth']} -> "
              f"{result.metadata['compiled_depth']}")

    result = Compiler(optimization_level=1).compile(qc)
    a, b = amplitudes(qc), amplitudes(result.circuit())
    fidelity = float(abs(np.vdot(a, b)) ** 2)
    print(f"\nsemantics preserved: fidelity = {fidelity:.12f}")
    assert result.result.num_gates < qc.gate_count(), "level 1 should simplify"
    assert fidelity > 1.0 - 1e-9, "compilation must preserve the state"

    print("\nExample 24 completed!")


if __name__ == "__main__":
    main()