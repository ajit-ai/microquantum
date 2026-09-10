"""Backend example 1: the canonical backend execution path.

MQ-06 gives every backend a unified plan-level contract:
``validate(plan) -> supports(plan) -> execute(plan)``.  A plan names the
work (circuit, shots, seed, target); the backend binds parameters and runs.
No runtime is required for the single-call path — this example uses both
the direct backend execution and the runtime to show they agree.
"""

from microquantum import (
    ExecutionPlan,
    ExecutionRuntime,
    LocalSimulatorBackend,
    QuantumCircuit,
)


def main():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    plan = ExecutionPlan.from_circuit(qc, shots=2048, seed=0)

    backend = LocalSimulatorBackend()
    print("=== Backend identity ===")
    print(f"name       : {backend.name}")
    print(f"is_simulator: {backend.capabilities.is_simulator}")
    print(f"supports_statevector: {backend.capabilities.supports_statevector}")

    # 1) validation is explicit and informative
    problems = backend.validate(plan)
    print(f"validation : {problems or 'ok'}")

    # 2) direct single-call execution
    direct = backend.execute(plan)
    print("=== Direct backend.execute(plan) ===")
    print(f"backend  : {direct.backend_name}")
    print(f"shots    : {direct.shots}")
    print(f"counts   : {dict(direct.counts)}")

    # 3) the runtime rides the same plan/backend contract
    runtime_result = ExecutionRuntime().execute(plan)
    assert direct.counts == runtime_result.counts
    print("=== Runtime agrees ===")
    print(f"runtime backend: {runtime_result.metadata['backend']}"
          f" / strategy: {runtime_result.metadata['strategy']}")


if __name__ == "__main__":
    main()