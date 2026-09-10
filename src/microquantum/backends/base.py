"""Abstract base class for execution backends."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from .._json import json_safe, json_string
from ..core.state import StateVector

if TYPE_CHECKING:
    from ..core.circuit import QuantumCircuit


class JobStatus(Enum):
    """Status of a quantum job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackendResult:
    """Result from a backend execution.

    Attributes:
        statevector: Final state vector (if simulated).
        density_matrix: Final density matrix (if applicable).
        counts: Measurement bitstring counts.
        num_qubits: Number of qubits in the system.
        backend_name: Name of the backend used.
        metadata: Additional execution metadata.
    """

    num_qubits: int
    backend_name: str
    statevector: Optional[NDArray[np.complex128]] = None
    density_matrix: Optional[NDArray[np.complex128]] = None
    counts: dict[str, int] = field(default_factory=dict)
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
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"BackendResult(num_qubits={self.num_qubits}, "
            f"backend='{self.backend_name}', "
            f"measurements={len(self.counts)})"
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
    """A quantum execution job.

    Tracks the lifecycle of a circuit execution submitted to a backend.

    Attributes:
        job_id: Unique identifier for the job.
        status: Current job status.
        result: Execution result (if completed).
        error: Error message (if failed).
    """

    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.PENDING
    result: Optional[BackendResult] = None
    error: Optional[str] = None

    def __repr__(self) -> str:
        return f"Job(id={self.job_id}, status={self.status.value})"


class Backend(ABC):
    """Abstract base class for quantum execution backends.

    All backends must implement the run() method which takes a quantum circuit
    and produces a BackendResult. Backends may use different simulation
    strategies (state vector, density matrix, etc.) or connect to real
    quantum hardware.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""

    @property
    def num_qubits(self) -> Optional[int]:
        """Maximum number of qubits supported. None = unlimited."""
        return None

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
        job = Job()
        job.status = JobStatus.RUNNING
        try:
            result = self.run_circuit(
                num_qubits, gates, shots=shots,
                initial_state=initial_state, seed=seed,
            )
            job.result = result
            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
