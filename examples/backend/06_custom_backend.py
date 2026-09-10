"""Backend example 6: building a custom backend with the MQ-06 contract.

A custom backend subclasses :class:`Backend`, advertises
:class:`BackendCapabilities`, obeys ``validate``/``supports``/``execute``,
and returns a :class:`BackendResult` with the new payload fields
(samples, expectations, shots, seed, native).  The runtime treats it like
any other backend.
"""

from microquantum import (
    Backend,
    BackendCapabilities,
    BackendResult,
    ExecutionPlan,
    ExecutionRuntime,
    QuantumCircuit,
    TargetClass,
)
from microquantum.backends.capabilities import (
    EXECUTION_EXPECTATION_VALUES,
    EXECUTION_SAMPLING,
    EXECUTION_SHOTS,
    FEATURE_MEASUREMENT,
    FEATURE_PARAMETERIZED_CIRCUITS,
)


class BatteryBackend(Backend):
    """Toy backend: "discharges" each shot deterministically.

    Demonstrates the full plan contract — capabilities, validation and a
    result payload — without any simulation machinery.
    """

    @property
    def name(self) -> str:
        return "battery"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            target_class=TargetClass.CUSTOM,
            execution=frozenset(
                {EXECUTION_SAMPLING, EXECUTION_SHOTS, EXECUTION_EXPECTATION_VALUES}
            ),
            circuit_features=frozenset(
                {FEATURE_PARAMETERIZED_CIRCUITS, FEATURE_MEASUREMENT}
            ),
            max_qubits=4,
            metadata={"discharge_rate": 0.1},
        )

    def run_circuit(self, num_qubits, gates, shots=1024, initial_state=None, seed=None):
        import numpy as np

        rng = np.random.default_rng(seed)
        drain = rng.binomial(shots, 0.3)
        counts = {
            "0" * num_qubits: shots - drain,
            "1" * num_qubits: drain,
        }
        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            counts=counts,
            samples=[1] * drain + [0] * (shots - drain),
            expectations={"drained": drain / shots},
            shots=shots,
            seed=seed,
            native={"model": "battery", "real": False},
        )


def main():
    # metadata() is JSON-safe and advertises capabilities automatically
    backend = BatteryBackend()
    print("identity:", backend.metadata()["name"])

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    plan = ExecutionPlan.from_circuit(qc, shots=1000, seed=3)

    print("validate:", backend.validate(plan) or "ok")
    result = backend.execute(plan)

    print("=== BatteryBackend result ===")
    print(f"counts      : {dict(result.counts)}")
    print(f"samples     : {len(result.samples)} shots")
    print(f"expectations: {result.expectations}")
    print(f"shots/seed  : {result.shots}/{result.seed}")

    # and the runtime handles it like any other backend
    runtime_result = ExecutionRuntime().execute(qc, backend=backend, shots=1000, seed=3)
    print(f"via runtime : {runtime_result.metadata['backend']} "
          f"(trace events: {len(runtime_result.metadata['trace']['events'])})")


if __name__ == "__main__":
    main()