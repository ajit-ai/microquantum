"""Example 32: Parameterized compile-then-execute pipeline.

Demonstrates the parameter pipeline through compilation and the runtime:

- compile a *symbolic* circuit (parameters survive all passes),
- rebuild the executable circuit and bind values afterward,
- the runtime resolves ``parameter_bindings`` on a compiled plan,
- fidelity between original-bound and compiled-bound states is 1.00.

``compile``/``bind`` commute: parameters are carried through the IR and the
handed-off :class:`CompilationResult` can be reused, then bound per run.
"""

from __future__ import annotations

import numpy as np

from microquantum import (
    Compiler,
    Parameter,
    QuantumCircuit,
    StatevectorBackend,
    execute,
)


def parameterized() -> QuantumCircuit:
    theta, phi = Parameter("theta"), Parameter("phi")
    qc = QuantumCircuit(2)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.rz(phi, 1)
    return qc


def statevector(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(
        StatevectorBackend().run(circuit, shots=None).statevector,
        dtype=np.complex128,
    )


def fidelity(a: QuantumCircuit, b: QuantumCircuit) -> float:
    return float(abs(np.vdot(statevector(a), statevector(b))) ** 2)


def main() -> None:
    print("=== 32 Parameterized compile-then-execute ===\n")

    theta, phi = Parameter("theta"), Parameter("phi")
    bindings = {"theta": 0.6, "phi": -0.9}

    compiled = Compiler(optimization_level=2).compile(parameterized())
    rebuilt = compiled.circuit()
    print(f"compiled parameters: {[p.name for p in rebuilt.parameters]}")
    assert rebuilt.parameters == (phi, theta)   # deterministic name order

    bound = rebuilt.bind_parameters(bindings)
    f = fidelity(parameterized().bind_parameters(bindings), bound)
    print(f"bind-after-compile fidelity: {f:.2f}")
    assert f > 1 - 1e-9

    # Parameter bindings resolved by the runtime on a compiled plan
    plan = execute(
        parameterized(),
        backend=StatevectorBackend(),
        shots=2048,
        seed=5,
        optimization_level=2,
        parameter_bindings=bindings,
    )
    print(f"parameterized runtime execution: {len(plan.counts)} bases, "
          f"sum-counts={sum(plan.counts.values())}")
    assert plan.metadata["strategy"] == "compiled"
    assert sum(plan.counts.values()) == 2048

    print("\nExample 32 completed!")


if __name__ == "__main__":
    main()