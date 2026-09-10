"""Backend example 4: backend selection precedence.

MQ-06 resolves which backend runs a plan with a single precedence rule:

1. plan.backend (a Backend instance or a registered name)
2. the runtime's explicit ``backend=`` default
3. the attached registry default
4. the runtime's own ``default_backend`` (lazy ``statevector`` simulator)

This example switches between each level and shows the win.
"""

from microquantum import (
    BackendRegistry,
    ExecutionPlan,
    ExecutionRuntime,
    MockBackend,
    QuantumCircuit,
    StatevectorBackend,
)


def make_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def main():
    plan = ExecutionPlan.from_circuit(make_circuit(), shots=256, seed=0)

    # (a) default backend, no registry: the lazy statevector simulator
    print("a) bare runtime          ->", ExecutionRuntime().resolve_backend(None).name)

    # (b) explicit runtime backend wins over the default
    rt = ExecutionRuntime(backend=StatevectorBackend())
    print("b) explicit runtime      ->", rt.default_backend.name)

    # (c) registry default is used when no plan/backend names one
    registry = BackendRegistry()
    registry.register(MockBackend(name="registered_default", max_qubits=4))
    registry.set_default("registered_default")
    rt2 = ExecutionRuntime(registry=registry)
    result2 = rt2.execute(plan)
    print(f"c) registry default      -> {result2.backend_name} ({result2.metadata['backend']})")

    # (d) a plan naming a backend wins over everything else
    plan_named = ExecutionPlan.from_circuit(
        make_circuit(), backend=MockBackend(name="plan_pick", max_qubits=2), shots=256
    )
    result3 = rt2.execute(plan_named)
    print(f"d) plan-pinned backend   -> {result3.backend_name}")

    # (e) a plan naming a backend *by string* resolves through the registry
    plan_string = ExecutionPlan.from_circuit(
        make_circuit(), backend="registered_default", shots=256
    )
    result4 = rt2.execute(plan_string)
    print(f"e) plan string name      -> {result4.backend_name}")


if __name__ == "__main__":
    main()