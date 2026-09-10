"""Execution example 2: an explicit execution plan.

An :class:`ExecutionPlan` captures everything about a run — the work, where
it runs, how many shots, parameter bindings, and compilation intent — as
plain, serializable data.
"""

import json

from microquantum import (
    ExecutionPlan,
    Parameter,
    QuantumCircuit,
    execute,
)


def main():
    theta = Parameter("theta")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(theta, 1)

    plan = ExecutionPlan.from_circuit(
        qc,
        name="bell-angle",
        shots=1024,
        seed=3,
        parameter_bindings={"theta": 1.25},
        target=None,
        optimization_level=1,
        metadata={"purpose": "demo"},
    )

    print("=== Plan (JSON) ===")
    print(json.dumps(plan.to_dict(), indent=2)[:600], "...")

    result = execute(plan)
    print("\n=== Executed plan ===")
    print(f"strategy : {result.metadata['strategy']}")
    print(f"bindings : {result.metadata['parameter_bindings']}")
    print(f"counts   : {result.counts}")


if __name__ == "__main__":
    main()