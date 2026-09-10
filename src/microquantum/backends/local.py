"""Default local simulator backend (MQ-06).

``LocalSimulatorBackend`` is the reference backend every execution target
can be compared against.  It reuses the existing state-vector simulation
engine (:class:`~microquantum.backends.statevector.StatevectorBackend`)
verbatim — MQ-06 does not invent a second simulator.
"""

from __future__ import annotations

from ..core.device import Device, DeviceType, Target
from .capabilities import BackendCapabilities, simulator_capabilities
from .statevector import StatevectorBackend


class LocalSimulatorBackend(StatevectorBackend):
    """Local NumPy state-vector simulator backend.

    A thin, named variant of :class:`StatevectorBackend` that carries the
    MQ-06 capability model and is the default backend of the
    :class:`~microquantum.backends.registry.BackendRegistry` and
    :class:`~microquantum.backends.provider.LocalProvider`.
    """

    @property
    def name(self) -> str:
        return "local_simulator"

    @property
    def device(self) -> Device:
        return Device(
            name=self.name,
            device_type=DeviceType.CPU,
            max_qubits=self.num_qubits,
            metadata={"engine": "statevector", "numerics": "numpy"},
        )

    @property
    def target(self) -> Target:
        return Target.universal(name=f"{self.name}_target")

    @property
    def capabilities(self) -> BackendCapabilities:
        caps = simulator_capabilities(max_qubits=self.num_qubits, statevector=True)
        if self.target.supports_dynamic_circuits:
            features = set(caps.circuit_features) | {
                "mid_circuit_measurement"
            }
            caps.circuit_features = frozenset(features)
        caps.metadata.update(
            {
                "device_type": "cpu",
                "engine": "statevector",
                "default": True,
            }
        )
        return caps


__all__ = ["LocalSimulatorBackend"]