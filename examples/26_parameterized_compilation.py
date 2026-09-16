"""Example 26: Parameterized compilation.

Demonstrates:
- Carrying symbolic ``Parameter`` objects through the compiler
- Binding a compiled circuit to numeric values and executing it
- ``compile(bind(circuit))`` == ``bind(compile(circuit))``
- Safety: symbolic rotations are NEVER fused or cancelled, because an
  unbound RX(theta) RX(theta) is not RX(something)
"""

from __future__ import annotations

import numpy as np

from microquantum import (
    Compiler,
    Gate,
    IRCircuit,
    Parameter,
    QuantumCircuit,
    StatevectorBackend,
)


def amplitudes(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(
        StatevectorBackend().run(circuit, shots=1, seed=7).statevector,
        dtype=np.complex128,
    )


def main() -> None:
    print("=== 26 Parameterized compilation ===\n")

    theta = Parameter("theta")
    qc = QuantumCircuit(2).ry(theta, 0).cx(0, 1)
    result = Compiler(optimization_level=2).compile(qc)

    compiled = result.circuit()
    names = {p.name for p in compiled.parameters}
    print(f"compiled parameters: {sorted(names)}")
    print(f"passes applied     : {result.passes_applied}")
    assert names == {"theta"}, "theta must survive compilation"

    bound_pre = compiled.bind_parameters({"theta": 0.8})
    bound_post = Compiler(optimization_level=2).compile(
        qc.bind_parameters({"theta": 0.8})
    ).circuit()

    a, b = amplitudes(bound_pre), amplitudes(bound_post)
    print(f"compile(bind) == bind(compile): "
          f"fidelity {abs(np.vdot(a, b)) ** 2:.12f}")
    assert abs(np.vdot(a, b)) ** 2 > 1.0 - 1e-9

    print("\nSymbolic rotations are never simplified:")
    sym = IRCircuit(
        num_qubits=1,
        operations=[
            Gate(name="rx", qubits=(0,), params=(theta,)),
            Gate(name="rx", qubits=(0,), params=(theta,)),
        ],
    )
    for level in (1, 2):
        out = Compiler(optimization_level=level).compile(sym)
        print(f"  level {level}: {out.result.gate_names()}")
        assert out.result.gate_names() == {"rx": 2}

    print("\nExample 26 completed!")


if __name__ == "__main__":
    main()