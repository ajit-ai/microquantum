"""Execution example 7: plugging in a custom backend.

Any object exposing the :class:`Backend` contract can run under the runtime.
Here a deliberately naive "fair coin" backend samples a uniform distribution
to show that execution planning, job handling and result collection are
backend-agnostic.
"""

import numpy as np

from microquantum import (
    Backend,
    BackendResult,
    ExecutionRuntime,
    QuantumCircuit,
)
from microquantum.core.device import Device, DeviceType, Target


class FairCoinBackend(Backend):
    """Toy backend that returns uniform measurement counts."""

    @property
    def name(self) -> str:
        return "faircoin"

    @property
    def num_qubits(self) -> int | None:
        return 4

    @property
    def device(self) -> Device:
        return Device(name=self.name, device_type=DeviceType.SIMULATOR, max_qubits=4)

    @property
    def target(self) -> Target:
        return Target(name=f"{self.name}_target", num_qubits=4)

    def run_circuit(self, num_qubits, gates, shots=1024, initial_state=None, seed=None):
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, 1 << num_qubits, size=shots)
        counts: dict[str, int] = {}
        for value in samples:
            key = format(int(value), f"0{num_qubits}b")
            counts[key] = counts.get(key, 0) + 1
        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            counts=counts,
        )


def main():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    rt = ExecutionRuntime(backend=FairCoinBackend())
    result = rt.execute(qc, shots=1000, seed=1)

    print("=== Custom backend execution ===")
    print(f"backend : {result.metadata['backend']}")
    print(f"strategy: {result.metadata['strategy']}")
    print(f"samples : {len(set(result.counts))} distinct bitstrings")
    print(f"sum     : {sum(result.counts.values())}")
    print(f"history : {[h['backend'] for h in rt.history]}")


if __name__ == "__main__":
    main()