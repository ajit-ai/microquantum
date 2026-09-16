"""Example 28: Target and capability validation.

Demonstrates:
- Compiling toward a ``Target`` with a native gate basis
- Clean compilation when every gate is supported
- Explicit diagnostics (never silent drops) for unsupported gates
- Automatic lowering of ``cz`` into ``h, cnot, h`` in a limited basis
- The ``cx`` / ``cnot`` naming equivalence (Target vs IR)
"""

from __future__ import annotations

from microquantum import Compiler, QuantumCircuit, Target

FULL = Target(
    name="universal-2q",
    num_qubits=2,
    native_gates=("h", "cnot"),
    supports_measurement=True,
)


def main() -> None:
    print("=== 28 Target and capability validation ===\n")

    supported = Compiler().compile(
        QuantumCircuit(2).h(0).cx(0, 1), target=FULL
    )
    print(f"supported gates -> compatible   : {supported.is_compatible}")
    print(f"  diagnostics : {supported.diagnostics}")
    assert supported.is_compatible

    limited = Target(
        name="limited-cx",
        num_qubits=2,
        native_gates=("h", "cx"),  # note: 'cx' spelling on the Target
        supports_measurement=True,
    )
    lowered = Compiler().compile(QuantumCircuit(2).cz(0, 1), target=limited)
    print("cz lowered toward 'h, cx' basis")
    print(f"  compiled gates : {lowered.circuit().to_ir().gate_names()}")
    print(f"  decomp. pass   : {'gate-decomposition' in lowered.passes_applied}")
    assert lowered.circuit().to_ir().gate_names() == {"h": 2, "cnot": 1}
    assert lowered.is_compatible

    unsupported = Compiler().compile(
        QuantumCircuit(1).t(0).h(0), target=FULL
    )
    print("\nunsupported 't' gate is reported, not dropped")
    print(f"  compatible   : {unsupported.is_compatible}")
    print(f"  gates kept   : {unsupported.result.gate_names()}")
    print(f"  diagnostics  : {unsupported.diagnostics}")
    assert not unsupported.is_compatible
    assert "t" in unsupported.result.gate_names(), "T must not be dropped"

    print("\nExample 28 completed!")


if __name__ == "__main__":
    main()