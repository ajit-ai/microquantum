"""IR example 6: target-aware compilation.

Demonstrates compiling a circuit toward an MQ-02 Target: gate
decomposition into the target's native basis, compatibility
diagnostics and result serialization.
"""

import json

from microquantum import Compiler, QuantumCircuit, Target


def main():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cz(0, 1)
    qc.swap(1, 2)

    target = Target(
        name="generic-2q",
        num_qubits=3,
        native_gates=("h", "cnot", "rx", "rz"),
        supports_measurement=True,
    )

    print("=== Compiling toward target ===")
    print(f"target: {target.name} native gates: {sorted(target.native_gates)}")
    print("source gates:", qc.to_ir().gate_names())

    result = Compiler(optimization_level=2).compile(qc, target=target)
    print("passes applied:", result.passes_applied)
    print("compiled gates:", result.result.gate_names())
    print("diagnostics:", result.diagnostics)
    print("compatible:", result.is_compatible)

    print("\n=== Compiled IR ===")
    print(result.result)

    payload = json.loads(result.to_json())
    print("\n=== Serialized compilation result ===")
    print("target name    :", payload["target"]["name"])
    print("compiled ops   :", len(payload["result"]["operations"]))

    rebuilt = result.circuit()
    print("rebuild vqc gates:", rebuilt.gate_count())


if __name__ == "__main__":
    main()