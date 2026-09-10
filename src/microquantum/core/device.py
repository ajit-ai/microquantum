"""Public compute-device and execution-target abstractions.

These capability-oriented descriptors describe *where* and *how* a
circuit may be executed, independently of the numerical implementation
(NumPy today; CPU/GPU/NPU accelerators and real quantum hardware in
the future).  They carry only plain data so hardware and accelerator
implementations are not forced into a specific numerical layer.

Programming model::

    Device/Target --describe--> Backend --executes--> Job / Result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .._json import json_safe, json_string


class DeviceType(Enum):
    """Broad class of a compute or hardware device."""

    SIMULATOR = "simulator"
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    ACCELERATOR = "accelerator"
    QPU = "qpu"


@dataclass
class Device:
    """A compute or hardware device capable of executing quantum work.

    A minimal identity/capability descriptor extensible to CPU/GPU/NPU
    simulators as well as real quantum hardware.

    Attributes:
        name: Stable device identifier.
        device_type: Broad :class:`DeviceType` of the device.
        max_qubits: Maximum qubits supported (``None`` = unbounded).
        available: Whether the device is currently available.
        metadata: Free-form device metadata (vendor, version, location...).
    """

    name: str
    device_type: DeviceType = DeviceType.SIMULATOR
    max_qubits: Optional[int] = None
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "device_type": self.device_type.value,
            "max_qubits": self.max_qubits,
            "available": self.available,
            "metadata": json_safe(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def __str__(self) -> str:
        return (
            f"Device(name='{self.name}', type={self.device_type.value}, "
            f"max_qubits={self.max_qubits if self.max_qubits is not None else 'unbounded'})"
        )


_UNIVERSAL_GATES: tuple[str, ...] = (
    "h",
    "x",
    "y",
    "z",
    "s",
    "t",
    "rx",
    "ry",
    "rz",
    "cx",
    "cy",
    "cz",
    "swap",
    "ccx",
    "measure",
)


@dataclass(frozen=True)
class Target:
    """Execution constraints a backend satisfies.

    A stable target contract for future transpilation and hardware
    backends: supported operations/gates, qubit count, connectivity,
    measurement and dynamic-circuit capability.

    Attributes:
        name: Target (gate-set) identifier.
        num_qubits: Maximum qubits supported (``None`` = unbounded).
        native_gates: Supported gate/operation names.
        connectivity: Optional qubit coupling pairs (topology).
        supports_measurement: Whether measurement is supported.
        max_shots: Maximum shots per execution (``None`` = unbounded).
        supports_dynamic_circuits: Mid-circuit measurement support.
        metadata: Free-form extra execution constraints.
    """

    name: str
    num_qubits: Optional[int] = None
    native_gates: tuple[str, ...] = field(default_factory=tuple)
    connectivity: Optional[tuple[tuple[int, int], ...]] = None
    supports_measurement: bool = True
    max_shots: Optional[int] = None
    supports_dynamic_circuits: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports_gate(self, gate_name: str) -> bool:
        """Return True if *gate_name* is in the native gate set."""
        return gate_name in self.native_gates

    @classmethod
    def universal(
        cls,
        name: str = "universal",
        num_qubits: Optional[int] = None,
    ) -> Target:
        """Target describing a universal simulator (full gate set)."""
        return cls(
            name=name,
            num_qubits=num_qubits,
            native_gates=_UNIVERSAL_GATES,
            supports_measurement=True,
            supports_dynamic_circuits=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "native_gates": list(self.native_gates),
            "connectivity": (
                [list(pair) for pair in self.connectivity]
                if self.connectivity is not None
                else None
            ),
            "supports_measurement": self.supports_measurement,
            "max_shots": self.max_shots,
            "supports_dynamic_circuits": self.supports_dynamic_circuits,
            "metadata": json_safe(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def __str__(self) -> str:
        return (
            f"Target(name='{self.name}', num_qubits="
            f"{self.num_qubits if self.num_qubits is not None else 'unbounded'}, "
            f"gates={len(self.native_gates)})"
        )
