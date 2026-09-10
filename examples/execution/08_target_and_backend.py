"""Execution example 8: target + backend coordination.

A plan can name a :class:`Target` (compilation + capability checks) while a
:class:`Backend` supplies the actual execution.  Here a small custom target
drops the circuit into the two-qubit ``{h, cnot}`` basis and the runtime
rejects a circuit that exceeds the target's qubit budget before it runs.
"""

from microquantum import (
    ExecutionPlan,
    QuantumCircuit,
    StatevectorBackend,
    Target,
    execute,
)


def main():
    backend = StatevectorBackend()
    target = Target(
        name="two_qubit_basis",
        num_qubits=2,
        native_gates=("h", "cnot"),
        supports_measurement=True,
        max_shots=100_000,
    )

    ok = QuantumCircuit(2)
    ok.h(0)
    ok.cx(0, 1)

    plan = ExecutionPlan.from_circuit(
        ok, target=target, backend=backend, shots=256, seed=1
    )
    result = execute(plan)

    print("=== Target-aware execution ===")
    print(f"strategy           : {result.metadata['strategy']}")
    print(f"target             : {result.metadata['target']['name']}")
    print(f"backend            : {result.metadata['backend']}")
    print(f"counts             : {result.counts}")

    too_big = QuantumCircuit(3)
    too_big.h(0)
    too_big.cx(0, 1)
    too_big.cx(0, 2)
    bad_plan = ExecutionPlan.from_circuit(too_big, target=target, backend=backend)

    print("\n=== Capability rejection ===")
    try:
        execute(bad_plan)
        print("unexpectedly executed")
    except ValueError as exc:
        print(f"ValueError raised as expected: {exc}")


if __name__ == "__main__":
    main()