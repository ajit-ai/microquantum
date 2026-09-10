"""Execution example 10: end-to-end pipeline.

Ties the pieces together: define a parameterized circuit, express it as
explicit IR, plan it against a named target, execute on a backend, inspect
the enriched result and export the plan/result to JSON.
"""

import json

from microquantum import (
    ExecutionPlan,
    Parameter,
    QuantumCircuit,
    StatevectorBackend,
    Target,
    execute,
    to_ir,
)


def main():
    theta = Parameter("theta")
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.rz(theta, 2)

    print("=== Work as IR ===")
    ir = to_ir(qc)
    print(ir.num_qubits, "qubits,", ir.num_gates, "gates,",
          "depth", ir.depth)

    print("\n=== Execution plan ===")
    target = Target(
        name="mq04_target",
        num_qubits=3,
        native_gates=("h", "cnot", "rz"),
        supports_measurement=True,
    )
    plan = ExecutionPlan.from_circuit(
        qc,
        name="ghz-scan",
        target=target,
        backend=StatevectorBackend(),
        shots=8192,
        seed=17,
        parameter_bindings={"theta": 1.2},
        optimization_level=1,
        metadata={"run_tag": "final-demo"},
    )
    print(f"plan: {plan!r}")

    print("\n=== Execute ===")
    result = execute(plan)
    print(f"strategy   : {result.metadata['strategy']}")
    print(f"job id     : {result.metadata['job_id']}")
    print(f"counts     : {result.counts}")
    print(f"correlation: P(000) + P(111) = "
          f"{result.probabilities.get('000', 0.0) + result.probabilities.get('111', 0.0):.3f}")

    print("\n=== JSON exports ===")
    plan_json = json.loads(plan.to_json())
    result_json = json.loads(result.to_json())
    print(f"plan   meta: {plan_json['metadata']}")
    print(f"result meta keys: {sorted(result_json['metadata'].keys())}")
    print(f"trace events    : {[e['name'] for e in result_json['metadata']['trace']['events']]}")


if __name__ == "__main__":
    main()