"""Backend example 3: registering and routing through a BackendRegistry.

A :class:`BackendRegistry` is the single place to register, discover and
inspect backends.  The ExecutionRuntime resolves backend *names* through it,
so plans can travel as JSON and still name their target backend.
"""

from microquantum import (
    BackendRegistry,
    ExecutionPlan,
    ExecutionRuntime,
    MockBackend,
    QuantumCircuit,
)


def main():
    registry = BackendRegistry()
    registry.register(MockBackend(name="fast_mock", max_qubits=4))
    registry.register(MockBackend(name="precise_mock", max_qubits=2))
    registry.set_default("fast_mock")

    print("=== Registry contents ===")
    print(f"names   : {registry.names()}")
    print(f"default : {registry.default.name}")
    print(f"json    : {registry.to_json()}")

    runtime = ExecutionRuntime(registry=registry)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # plan just carries a name; the runtime resolves it
    plan = ExecutionPlan.from_circuit(qc, backend="precise_mock", shots=64, seed=1)
    result = runtime.execute(plan)
    print(f"resolved plan backend -> {result.backend_name}")

    # unbacked plans fall back to the registry default
    result = runtime.execute(ExecutionPlan.from_circuit(qc, shots=64, seed=2))
    print(f"default backend       -> {result.backend_name}")

    # duplicate registration is rejected with a helpful error
    try:
        registry.register(MockBackend(name="precise_mock"))
    except ValueError as exc:
        print(f"duplicate rejected    -> {exc}")

    # unknown names are rejected with the full listing
    try:
        registry.get("no_such_backend")
    except KeyError as exc:
        print(f"unknown name          -> {exc}")


if __name__ == "__main__":
    main()