"""Example 27: Measurement-preserving optimization.

Demonstrates:
- Terminal measurements annotated on a ``QuantumCircuit`` survive
  compilation (exact subset AND order)
- An unmeasured circuit gets full terminal measurements when compiled
- Optimizations never remove a measurement (measurements act as
  separators for gate cancellation)
"""

from __future__ import annotations

from microquantum import Compiler, Measurement, QuantumCircuit


def main() -> None:
    print("=== 27 Measurement-preserving optimization ===\n")

    qc = QuantumCircuit(4)
    qc.x(1)
    qc.measure(3)
    qc.measure(1)
    qc.measure(2)
    print(f"measurement order in source: {qc.measurements}")

    compiled = Compiler(optimization_level=1).compile(qc).circuit()
    print(f"measurement order compiled : {compiled.measurements}")
    assert compiled.measurements == [3, 1, 2], "order must be preserved"

    trivial = QuantumCircuit(3).h(0)
    result = Compiler(optimization_level=1).compile(trivial)
    terminals = [op for op in result.result.operations if isinstance(op, Measurement)]
    print(f"unmeasured circuit gets full terminal measurements: "
          f"{[m.qubit for m in terminals]}")
    assert [m.qubit for m in terminals] == [0, 1, 2]

    cancellable = QuantumCircuit(3)
    cancellable.h(0)
    cancellable.measure(0)
    cancellable.h(1)
    cancellable.measure(1)
    out = Compiler(optimization_level=1).compile(cancellable).circuit()
    print(f"measurements act as barriers to cancellation: "
          f"gates={out.gate_count()} measurements={out.measurements}")
    assert out.measurements == [0, 1]
    assert out.gate_count() == 2, "the two H gates must survive"

    print("\nExample 27 completed!")


if __name__ == "__main__":
    main()