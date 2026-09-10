"""Backend example 5: runtime orchestration over a chosen backend.

This joins the MQ-06 backend layer with the MQ-04 runtime: plan preparation,
strategy dispatch (direct/compiled), jobs, trace, history and the enriched
:class:`BackendResult` (shots/seed/target payload) all flow through a backend
selected by name.
"""

from microquantum import (
    BackendRegistry,
    ExecutionPlan,
    ExecutionRuntime,
    Target,
)


def main():
    registry = BackendRegistry()
    runtime = ExecutionRuntime(registry=registry)

    from microquantum import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    target = Target(name="custom_target", num_qubits=2, max_shots=50000)
    plan = ExecutionPlan.from_circuit(
        qc,
        name="target-aware",
        backend="local_simulator",  # a plain string name
        target=target,
        shots=1024,
        seed=7,
    )

    result = runtime.execute(plan)

    print("=== Result payload ===")
    print(f"backend      : {result.backend_name}")
    print(f"shots        : {result.shots}")
    print(f"seed         : {result.seed}")
    print(f"target_name  : {result.target_name}")
    print(f"counts       : {dict(result.counts)}")
    print("=== Enriched metadata ===")
    meta = result.metadata
    print(f"strategy     : {meta['strategy']}")
    print(f"target dict  : {meta['target']['name']}")
    print(f"trace events : {len(meta['trace']['events'])}")
    print("=== Runtime history ===")
    print(f"history      : {[(h['plan'], h['backend']) for h in runtime.history]}")

    # JSON-safe exports survive a round trip
    import json

    payload = json.loads(result.to_json())
    print(
        "json roundtrip: "
        f"samples={'present' if payload['samples'] is not None else 'n/a'} "
        f"shots={payload['shots']}"
    )


if __name__ == "__main__":
    main()