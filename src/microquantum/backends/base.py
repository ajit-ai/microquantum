"""Abstract base class for execution backends."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from .._json import json_safe, json_string
from ..core.device import Device, DeviceType, Target
from ..core.state import StateVector

if TYPE_CHECKING:
    from ..core.circuit import QuantumCircuit
    from ..runtime.plan import ExecutionPlan

from .capabilities import BackendCapabilities, simulator_capabilities


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _array_from_dict(mapping: dict[str, Any], key: str) -> Optional[NDArray[np.complex128]]:
    """Rebuild a complex128 array from its serialized ``to_dict()`` value.

    Complex values were encoded element-wise as ``{"real": .., "imag": ..}``
    by :func:`~microquantum._json.json_safe`; this reverses that encoding.
    """
    value = mapping.get(key)
    if value is None:
        return None
    return np.asarray(_decode_complex(value), dtype=np.complex128)


def _decode_complex(value: Any) -> Any:
    """Recursively restore ``{"real": .., "imag": ..}`` encoded values."""
    if isinstance(value, dict):
        if set(value) == {"real", "imag"} and all(
            isinstance(v, (int, float)) for v in value.values()
        ):
            return complex(float(value["real"]), float(value["imag"]))
        return {k: _decode_complex(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_complex(v) for v in value]
    return value


class JobStatus(Enum):
    """Status of a quantum job.

    The lifecycle flows ``CREATED -> QUEUED -> RUNNING -> COMPLETED`` and
    may terminate early as ``FAILED`` or ``CANCELLED``.  Local backends
    move through the non-terminal states synchronously.
    """

    CREATED = "created"
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def final(self) -> bool:
        """Whether this status is a terminal lifecycle state."""
        return self in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )


@dataclass
class BackendResult:
    """Result from a backend execution.

    Attributes:
        statevector: Final state vector (if simulated).
        density_matrix: Final density matrix (if applicable).
        counts: Measurement bitstring counts.
        samples: Raw per-shot outcome indices (if the backend provides them).
        expectations: Labeled expectation values (e.g. Hamiltonian terms).
        eigenvalues: Computed eigenvalues (where applicable).
        native: Raw/vendor-native result payload (JSON-safe where possible).
        shots: Number of shots requested for this execution.
        seed: RNG seed used (if deterministic).
        target_name: Target the result was produced on (if known).
        num_qubits: Number of qubits in the system.
        backend_name: Name of the backend used.
        metadata: Additional execution metadata.
    """

    num_qubits: int
    backend_name: str
    statevector: Optional[NDArray[np.complex128]] = None
    density_matrix: Optional[NDArray[np.complex128]] = None
    counts: dict[str, int] = field(default_factory=dict)
    samples: Optional[list[int]] = None
    expectations: dict[str, float] = field(default_factory=dict)
    eigenvalues: Optional[list[float]] = None
    native: dict[str, Any] = field(default_factory=dict)
    shots: Optional[int] = None
    seed: Optional[int] = None
    target_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def probabilities(self) -> dict[str, float]:
        """Compute measurement probabilities from counts."""
        if not self.counts:
            return {}
        total = sum(self.counts.values())
        return {k: v / total for k, v in self.counts.items()}

    def most_frequent(self) -> str:
        """Return the most frequently measured bitstring."""
        if not self.counts:
            raise ValueError("No measurement counts available")
        return max(self.counts, key=lambda k: self.counts[k])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary.

        Complex-valued arrays (``statevector`` / ``density_matrix``) are
        encoded element-wise as ``{"real": ..., "imag": ...}`` pairs.
        """
        return {
            "num_qubits": self.num_qubits,
            "backend_name": self.backend_name,
            "statevector": json_safe(self.statevector),
            "density_matrix": json_safe(self.density_matrix),
            "counts": dict(self.counts),
            "probabilities": dict(self.probabilities),
            "samples": list(self.samples) if self.samples is not None else None,
            "expectations": dict(self.expectations),
            "eigenvalues": (
                list(self.eigenvalues) if self.eigenvalues is not None else None
            ),
            "native": json_safe(dict(self.native)),
            "shots": self.shots,
            "seed": self.seed,
            "target_name": self.target_name,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackendResult":
        """Rebuild a result from its ``to_dict()`` mapping.

        Complex-valued arrays are decoded back to ``complex128`` arrays so
        statevector / density-matrix data survive a serialization round trip.
        """
        statevector = _array_from_dict(data, "statevector")
        density_matrix = _array_from_dict(data, "density_matrix")
        samples = data.get("samples")
        eigenvalues = data.get("eigenvalues")
        return cls(
            num_qubits=int(data["num_qubits"]),
            backend_name=str(data["backend_name"]),
            statevector=statevector,
            density_matrix=density_matrix,
            counts=dict(data.get("counts") or {}),
            samples=list(samples) if samples is not None else None,
            expectations=dict(data.get("expectations") or {}),
            eigenvalues=list(eigenvalues) if eigenvalues is not None else None,
            native=dict(data.get("native") or {}),
            shots=data.get("shots"),
            seed=data.get("seed"),
            target_name=data.get("target_name"),
            metadata=dict(data.get("metadata") or {}),
        )

    def __repr__(self) -> str:
        return (
            f"BackendResult(num_qubits={self.num_qubits}, "
            f"backend='{self.backend_name}', count_distinct={len(self.counts)}, "
            f"samples={len(self.samples) if self.samples else 0})"
        )

    def __str__(self) -> str:
        lines = [
            f"BackendResult ({self.num_qubits} qubits, backend={self.backend_name})",
        ]
        if self.counts:
            lines.append("Measurement counts:")
            for bitstring in sorted(self.counts.keys()):
                lines.append(f"  |{bitstring}>: {self.counts[bitstring]}")
        return "\n".join(lines)


@dataclass
class Job:
    """A single circuit-execution job.

    Local (simulator) backends return jobs that are already completed: the
    lifecycle moves ``CREATED -> QUEUED -> RUNNING -> COMPLETED`` (or
    ``FAILED`` / and ``CANCELLED`` via :meth:`cancel`).  Hardware backends
    return jobs that finish asynchronously once the provider responds (see
    :class:`~microquantum.providers.HardwareJob`).

    Attributes:
        job_id: Unique identifier for the job.
        status: Current :class:`JobStatus`.
        result: Execution result (if completed).
        error: Error message (if failed).
        backend_name: Backend that accepted the job (if set).
        target_name: Target the job runs on (if set).
        created_at: ISO-8601 timestamp when the job was created.
        started_at: ISO-8601 timestamp when execution started.
        finished_at: ISO-8601 timestamp when execution finished.
        info: Arbitrary job lifecycle metadata (plan details, trace hints).
    """

    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.PENDING
    result: Optional[BackendResult] = None
    error: Optional[str] = None
    backend_name: Optional[str] = None
    target_name: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)

    def cancel(self) -> None:
        """Mark the job as cancelled.

        A no-op once the job already completed, failed or was cancelled.
        """
        if self.status.final:
            return
        self.status = JobStatus.CANCELLED

    def metadata(self) -> dict[str, Any]:
        """Expose JSON-safe job lifecycle metadata."""
        data: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status.value,
        }
        if self.backend_name is not None:
            data["backend_name"] = self.backend_name
        if self.target_name is not None:
            data["target_name"] = self.target_name
        if self.created_at is not None:
            data["created_at"] = self.created_at
        if self.started_at is not None:
            data["started_at"] = self.started_at
        if self.finished_at is not None:
            data["finished_at"] = self.finished_at
        if self.error is not None:
            data["error"] = self.error
        if self.info:
            data["info"] = self.info
        return data

    def __repr__(self) -> str:
        return f"Job(id={self.job_id}, status={self.status.value})"


class Backend(ABC):
    """Abstract base class for quantum execution backends.

    Every backend exposes:

    * **Identity**: :attr:`name`, :attr:`num_qubits`.
    * **Capabilities**: :attr:`device` and :attr:`target` describe the
      compute device and the execution constraints the backend satisfies;
      :meth:`metadata` returns them as a JSON-safe dictionary.
    * **Execution**: :meth:`run_circuit` (low-level gate matrices),
      :meth:`run` (high-level :class:`QuantumCircuit`) and
      :meth:`submit_circuit` / :meth:`submit` which wrap execution in a
      :class:`Job`.
    * **Results**: :class:`BackendResult` carries state vectors / density
      matrices, measurement counts/probabilities and metadata.

    Backends may use different simulation strategies (state vector, density
    matrix, ...) or connect to real quantum hardware.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""

    @property
    def num_qubits(self) -> Optional[int]:
        """Maximum number of qubits supported. None = unlimited."""
        return None

    @property
    def device(self) -> Device:
        """The compute device this backend runs on.

        Simulators report the CPU/GPU executing the simulation; hardware
        backends report the target QPU.  Override for accurate reporting.
        """
        return Device(
            name=self.name,
            device_type=DeviceType.SIMULATOR,
            max_qubits=self.num_qubits,
        )

    @property
    def target(self) -> Target:
        """Execution constraints this backend satisfies.

        Override to advertise native gates, connectivity, measurement
        capabilities or dynamic-circuit support.
        """
        return Target(name=self.name, num_qubits=self.num_qubits)

    @property
    def capabilities(self) -> BackendCapabilities:
        """Structured :class:`BackendCapabilities` of this backend.

        The default derives a simulator capability set from :attr:`device`
        and :attr:`target`.  Override to advertise richer/different modes.
        """
        caps = simulator_capabilities(max_qubits=self.num_qubits)
        target = self.target
        features = set(caps.circuit_features)
        if target.supports_dynamic_circuits:
            features.add("mid_circuit_measurement")
        return BackendCapabilities(
            target_class=caps.target_class,
            execution=caps.execution,
            circuit_features=frozenset(features),
            max_qubits=self.num_qubits,
            connectivity=target.connectivity,
            native_gates=target.native_gates,
            metadata={
                "device_type": self.device.device_type.value,
            },
        )

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe backend identity and capability metadata."""
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "device": self.device.to_dict(),
            "target": self.target.to_dict(),
            "capabilities": self.capabilities.to_dict(),
        }

    # ------------------------------------------------------------------
    # plan-level contract (MQ-06)
    # ------------------------------------------------------------------

    def validate(self, plan: "ExecutionPlan") -> list[str]:
        """Validate a plan against this backend, returning descriptive problems.

        An empty list means the plan is executable on this backend.  The
        default checks qubit capacity, the shot mode and the requested
        target; subclasses override to enforce richer constraints (native
        gates, connectivity, dynamic-circuit features, ...).
        """
        problems: list[str] = []
        caps = self.capabilities

        num_qubits = plan.num_qubits
        if caps.max_qubits is not None and num_qubits > caps.max_qubits:
            problems.append(
                f"plan uses {num_qubits} qubits but backend '{self.name}' "
                f"supports at most {caps.max_qubits}"
            )
        if plan.shots is not None and not caps.supports_shots:
            problems.append(
                f"plan requests {plan.shots} shots but backend '{self.name}' "
                f"does not support shot-based execution"
            )
        target = plan.target
        if (
            target is not None
            and target.num_qubits is not None
            and num_qubits > target.num_qubits
        ):
            problems.append(
                f"plan uses {num_qubits} qubits but target '{target.name}' "
                f"supports at most {target.num_qubits}"
            )
        return problems

    def supports(self, plan: "ExecutionPlan") -> bool:
        """Return True if the plan validates against this backend.

        Shorthand for ``self.validate(plan) == []``.
        """
        return not self.validate(plan)

    def execute(self, plan: "ExecutionPlan") -> BackendResult:
        """Execute a validated plan on this backend and return a result.

        This is the canonical single-call backend entry point: it validates
        the plan first, binds parameters and runs the described work.  The
        execution runtime (see :mod:`microquantum.runtime`) orchestrates a
        richer pipeline (trace, metadata, jobs) on top of this contract.

        Args:
            plan: Diagnostic :class:`ExecutionPlan` to run.

        Returns:
            :class:`BackendResult` from executing the plan.

        Raises:
            ValueError: If the plan does not validate against this backend.
        """
        problems = self.validate(plan)
        if problems:
            raise ValueError(
                f"backend '{self.name}' rejected the plan: {'; '.join(problems)}"
            )
        circuit = plan.bound()
        result = self.run(
            circuit,
            shots=plan.shots,
            initial_state=plan.initial_state,
            seed=plan.seed,
        )
        target = plan.target if plan.target is not None else self.target
        return replace(
            result,
            shots=plan.shots if result.shots is None else result.shots,
            seed=plan.seed if result.seed is None else result.seed,
            target_name=result.target_name
            if result.target_name is not None
            else target.name,
        )

    @abstractmethod
    def run_circuit(
        self,
        num_qubits: int,
        gates: list[tuple[NDArray[np.complex128], list[int]]],
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Execute a quantum circuit.

        Args:
            num_qubits: Number of qubits in the circuit.
            gates: List of (gate_matrix, target_qubits) pairs.
            shots: Number of measurement shots.
            initial_state: Optional initial state vector.
            seed: Optional RNG seed.

        Returns:
            BackendResult with simulation output.
        """

    def submit(
        self,
        num_qubits: int,
        gates: list[tuple[NDArray[np.complex128], list[int]]],
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> Job:
        """Submit a circuit for execution (synchronous wrapper).

        Args:
            num_qubits: Number of qubits.
            gates: List of (gate_matrix, target_qubits) pairs.
            shots: Number of measurement shots.
            initial_state: Optional initial state.
            seed: Optional RNG seed.

        Returns:
            Completed Job with result.
        """
        job = Job(
            backend_name=self.name,
            target_name=self.target.name,
            created_at=_now_iso(),
        )
        job.status = JobStatus.QUEUED
        job.status = JobStatus.RUNNING
        job.started_at = _now_iso()
        try:
            result = self.run_circuit(
                num_qubits,
                gates,
                shots=shots,
                initial_state=initial_state,
                seed=seed,
            )
            job.result = result
            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
        job.finished_at = _now_iso()
        return job

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Execute a bound circuit via the high-level API.

        The preferred way to run a :class:`QuantumCircuit` on any backend —
        both simulators and hardware. Translates the circuit's gate
        instructions into the low-level ``run_circuit`` contract.

        Args:
            circuit: The quantum circuit to execute. Unbound parameters
                raise ``ValueError``.
            shots: Number of measurement shots.
            initial_state: Optional initial state vector.
            seed: RNG seed for reproducibility.

        Returns:
            BackendResult with measurement counts and simulation state.
        """
        circuit._ensure_bound()
        gates = [(op.matrix, targets) for op, targets in circuit.gates]
        return self.run_circuit(
            num_qubits=circuit.num_qubits,
            gates=gates,
            shots=shots,
            initial_state=initial_state,
            seed=seed,
        )

    def submit_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> Job:
        """Submit a high-level circuit for execution and return a Job.

        ``Circuit -> Backend.submit_circuit() -> Job``.  Simulator backends
        return a completed :class:`Job` synchronously; hardware subclasses
        may keep the job running until the provider responds.

        Args:
            circuit: The bound quantum circuit to execute.
            shots: Number of measurement shots.
            initial_state: Optional initial state vector.
            seed: RNG seed for reproducibility.

        Returns:
            A :class:`Job` holding the execution result on completion.
        """
        job = Job(
            backend_name=self.name,
            target_name=self.target.name,
            created_at=_now_iso(),
        )
        job.status = JobStatus.QUEUED
        job.status = JobStatus.RUNNING
        job.started_at = _now_iso()
        try:
            result = self.run(
                circuit,
                shots=shots,
                initial_state=initial_state,
                seed=seed,
            )
            job.result = result
            job.status = JobStatus.COMPLETED
        except Exception as exc:
            job.error = str(exc)
            job.status = JobStatus.FAILED
        job.finished_at = _now_iso()
        return job

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
