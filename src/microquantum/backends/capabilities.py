"""Backend capabilities: a structured, JSON-safe description of what a
backend can and cannot do.

Capabilities are *descriptions*, never execution logic.  They let callers
and the runtime inspect a backend before submitting work:

* **target class** — broad category (:class:`TargetClass`): simulators,
  CPU/GPU machines, real quantum hardware, remote services, custom.
* **execution modes** — statevector, sampling, shots, expectation values,
  unitary, density matrices.
* **circuit features** — parameterized circuits, measurement, mid-circuit
  measurement, reset, controlled operations, custom gates.
* **hardware characteristics** — qubit capacity, connectivity, native gate
  set, precision.

A backend advertises which capabilities it has; it is *not* required to
implement every capability.  Unknown capability tokens are stored as plain
strings so custom backends can tag their own modes without forking the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .._json import JSONSerializable, json_safe


class TargetClass(Enum):
    """Broad category of an execution target.

    Members:
        CPU: Local CPU execution (usually a simulator).
        GPU: GPU-accelerated execution.
        SIMULATOR: Any simulator (CPU/GPU/NPU).
        QUANTUM_HARDWARE: Real quantum processing unit.
        REMOTE: Remote service / cloud API.
        CUSTOM: Private or custom enterprise backend.
    """

    CPU = "cpu"
    GPU = "gpu"
    SIMULATOR = "simulator"
    QUANTUM_HARDWARE = "quantum_hardware"
    REMOTE = "remote"
    CUSTOM = "custom"


# --- execution-mode capability tokens -----------------------------------

EXECUTION_STATEVECTOR = "statevector"
EXECUTION_SAMPLING = "sampling"
EXECUTION_SHOTS = "shots"
EXECUTION_EXPECTATION_VALUES = "expectation_values"
EXECUTION_UNITARY = "unitary"
EXECUTION_DENSITY_MATRIX = "density_matrix"

EXECUTION_CAPABILITIES: tuple[str, ...] = (
    EXECUTION_STATEVECTOR,
    EXECUTION_SAMPLING,
    EXECUTION_SHOTS,
    EXECUTION_EXPECTATION_VALUES,
    EXECUTION_UNITARY,
    EXECUTION_DENSITY_MATRIX,
)

# --- circuit-feature capability tokens ------------------------------------

FEATURE_PARAMETERIZED_CIRCUITS = "parameterized_circuits"
FEATURE_MEASUREMENT = "measurement"
FEATURE_MID_CIRCUIT_MEASUREMENT = "mid_circuit_measurement"
FEATURE_RESET = "reset"
FEATURE_CONTROLLED_OPERATIONS = "controlled_operations"
FEATURE_CUSTOM_GATES = "custom_gates"

CIRCUIT_FEATURES: tuple[str, ...] = (
    FEATURE_PARAMETERIZED_CIRCUITS,
    FEATURE_MEASUREMENT,
    FEATURE_MID_CIRCUIT_MEASUREMENT,
    FEATURE_RESET,
    FEATURE_CONTROLLED_OPERATIONS,
    FEATURE_CUSTOM_GATES,
)


@dataclass
class CalibrationData(JSONSerializable):
    """Backend-reported calibration snapshot.

    Attributes:
        gate_errors: Per-gate error rates keyed by gate name.
        readout_errors: Per-qubit readout error rates keyed by qubit
            label (e.g. ``"q0"``).
        t1_us: Per-qubit T1 times in microseconds.
        t2_us: Per-qubit T2 times in microseconds.
        timestamp: ISO-8601 capture time (empty = unknown).
    """

    gate_errors: dict[str, float] = field(default_factory=dict)
    readout_errors: dict[str, float] = field(default_factory=dict)
    t1_us: dict[str, float] = field(default_factory=dict)
    t2_us: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        for table_name in ("gate_errors", "readout_errors"):
            for key, value in getattr(self, table_name).items():
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{table_name}[{key!r}] must be in [0, 1]")
        for table_name in ("t1_us", "t2_us"):
            for key, value in getattr(self, table_name).items():
                if value < 0:
                    raise ValueError(f"{table_name}[{key!r}] must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "gate_errors": dict(self.gate_errors),
            "readout_errors": dict(self.readout_errors),
            "t1_us": dict(self.t1_us),
            "t2_us": dict(self.t2_us),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationData:
        """Rebuild calibration data from :meth:`to_dict` output."""
        return cls(
            gate_errors={str(k): float(v) for k, v in data.get("gate_errors", {}).items()},
            readout_errors={
                str(k): float(v) for k, v in data.get("readout_errors", {}).items()
            },
            t1_us={str(k): float(v) for k, v in data.get("t1_us", {}).items()},
            t2_us={str(k): float(v) for k, v in data.get("t2_us", {}).items()},
            timestamp=str(data.get("timestamp", "")),
        )


@dataclass
class BackendCapabilities(JSONSerializable):
    """Structured capability description of a backend.

    Attributes:
        target_class: Broad :class:`TargetClass` of the execution target.
        execution: Execution-mode capability tokens (e.g.
            ``"statevector"``, ``"sampling"``, ``"shots"``).
        circuit_features: Circuit-feature capability tokens (e.g.
            ``"parameterized_circuits"``, ``"measurement"``).
        max_qubits: Maximum qubits supported (``None`` = unbounded).
        connectivity: Optional coupling-map of supported qubit pairs.
        native_gates: Tuple of supported gate/operation names (empty = any
            gate acceptable).
        precision: Optional execution precision descriptor, e.g. ``16``
            (bits) or ``"double"``.
        metadata: Free-form extra capability metadata (JSON-safe).
        calibration: Optional backend calibration snapshot.
        max_circuit_depth: Maximum circuit depth supported
            (``None`` = unbounded).
    """

    target_class: TargetClass = TargetClass.SIMULATOR
    execution: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                EXECUTION_STATEVECTOR,
                EXECUTION_SAMPLING,
                EXECUTION_SHOTS,
            }
        )
    )
    circuit_features: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                FEATURE_PARAMETERIZED_CIRCUITS,
                FEATURE_MEASUREMENT,
                FEATURE_CONTROLLED_OPERATIONS,
                FEATURE_CUSTOM_GATES,
            }
        )
    )
    max_qubits: Optional[int] = None
    connectivity: Optional[tuple[tuple[int, int], ...]] = None
    native_gates: tuple[str, ...] = field(default_factory=tuple)
    precision: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    calibration: Optional[CalibrationData] = None
    max_circuit_depth: Optional[int] = None

    # -- helpers ---------------------------------------------------------

    def supports_execution(self, mode: str) -> bool:
        """Return True if *mode* (e.g. ``"shots"``) is advertised."""
        return mode in self.execution

    def supports_feature(self, feature: str) -> bool:
        """Return True if *feature* (e.g. ``"reset"``) is advertised."""
        return feature in self.circuit_features

    @property
    def supports_shots(self) -> bool:
        """Whether the backend can consume a shot count."""
        return self.supports_execution(EXECUTION_SHOTS)

    @property
    def supports_statevector(self) -> bool:
        """Whether the backend exposes an exact state vector."""
        return self.supports_execution(EXECUTION_STATEVECTOR)

    @property
    def supports_density_matrix(self) -> bool:
        """Whether the backend exposes a density matrix."""
        return self.supports_execution(EXECUTION_DENSITY_MATRIX)

    @property
    def is_simulator(self) -> bool:
        """Whether this is a simulator-class target."""
        return self.target_class in (
            TargetClass.SIMULATOR,
            TargetClass.CPU,
            TargetClass.GPU,
        )

    @property
    def is_hardware(self) -> bool:
        """Whether this is a real-hardware target."""
        return self.target_class is TargetClass.QUANTUM_HARDWARE

    @property
    def is_remote(self) -> bool:
        """Whether this talks to a remote service."""
        return self.target_class is TargetClass.REMOTE

    def merge(self, other: "BackendCapabilities") -> "BackendCapabilities":
        """Return a new capability set combining both descriptions.

        The intersection of execution/feature tokens and the tighter
        capacity bound is kept so the result describes work both backends
        can handle.
        """
        return BackendCapabilities(
            target_class=(
                self.target_class
                if self.target_class is other.target_class
                else TargetClass.CUSTOM
            ),
            execution=self.execution & other.execution,
            circuit_features=self.circuit_features & other.circuit_features,
            max_qubits=_min_optional(self.max_qubits, other.max_qubits),
            connectivity=self.connectivity or other.connectivity,
            native_gates=self.native_gates or other.native_gates,
            precision=self.precision or other.precision,
            metadata={**self.metadata, **other.metadata},
            calibration=self.calibration or other.calibration,
            max_circuit_depth=_min_optional(
                self.max_circuit_depth, other.max_circuit_depth
            ),
        )

    # -- construction from a serialized description -------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackendCapabilities":
        """Reconstruct capabilities from a :meth:`to_dict` mapping."""
        return cls(
            target_class=TargetClass(data["target_class"]),
            execution=frozenset(data.get("execution", ())),
            circuit_features=frozenset(data.get("circuit_features", ())),
            max_qubits=data.get("max_qubits"),
            connectivity=(
                tuple(tuple(pair) for pair in data["connectivity"])
                if data.get("connectivity") is not None
                else None
            ),
            native_gates=tuple(data.get("native_gates", ())),
            precision=data.get("precision"),
            metadata=dict(data.get("metadata", {})),
            calibration=(
                CalibrationData.from_dict(data["calibration"])
                if data.get("calibration") is not None
                else None
            ),
            max_circuit_depth=data.get("max_circuit_depth"),
        )

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "target_class": self.target_class.value,
            "execution": sorted(self.execution),
            "circuit_features": sorted(self.circuit_features),
            "max_qubits": self.max_qubits,
            "connectivity": (
                [list(pair) for pair in self.connectivity]
                if self.connectivity is not None
                else None
            ),
            "native_gates": list(self.native_gates),
            "precision": json_safe(self.precision),
            "metadata": json_safe(self.metadata),
            "calibration": self.calibration.to_dict() if self.calibration is not None else None,
            "max_circuit_depth": self.max_circuit_depth,
        }


def _min_optional(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def simulator_capabilities(
    *,
    max_qubits: Optional[int] = None,
    statevector: bool = True,
    density_matrix: bool = False,
) -> BackendCapabilities:
    """Build a typical NumPy-simulator capability set."""
    execution = {
        EXECUTION_SAMPLING,
        EXECUTION_SHOTS,
        EXECUTION_EXPECTATION_VALUES,
        EXECUTION_UNITARY,
    }
    if statevector:
        execution.add(EXECUTION_STATEVECTOR)
    if density_matrix:
        execution.add(EXECUTION_DENSITY_MATRIX)
    return BackendCapabilities(
        target_class=TargetClass.SIMULATOR,
        execution=frozenset(execution),
        circuit_features=frozenset(
            {
                FEATURE_PARAMETERIZED_CIRCUITS,
                FEATURE_MEASUREMENT,
                FEATURE_CONTROLLED_OPERATIONS,
                FEATURE_CUSTOM_GATES,
            }
        ),
        max_qubits=max_qubits,
        native_gates=(),
    )


__all__ = [
    "BackendCapabilities",
    "CalibrationData",
    "CIRCUIT_FEATURES",
    "EXECUTION_CAPABILITIES",
    "EXECUTION_DENSITY_MATRIX",
    "EXECUTION_EXPECTATION_VALUES",
    "EXECUTION_SAMPLING",
    "EXECUTION_SHOTS",
    "EXECUTION_STATEVECTOR",
    "EXECUTION_UNITARY",
    "FEATURE_CONTROLLED_OPERATIONS",
    "FEATURE_CUSTOM_GATES",
    "FEATURE_MEASUREMENT",
    "FEATURE_MID_CIRCUIT_MEASUREMENT",
    "FEATURE_PARAMETERIZED_CIRCUITS",
    "FEATURE_RESET",
    "TargetClass",
    "simulator_capabilities",
]