"""Base classes for the domain adapter framework.

The adapter pattern sits between domain applications and the quantum SDK:

    Domain Application → DomainAdapter → MicroQuantum SDK → Execution Backend
                                        ↑
                              Domain-specific encoding/decoding
                              Physics validation
                              Result caching
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from .._json import json_safe, json_string
from ..backends.base import Backend, BackendResult
from ..core.circuit import QuantumCircuit


class ProblemStatus(Enum):
    """Status of a quantum problem."""

    CREATED = "created"
    VALIDATED = "validated"
    ENCODED = "encoded"
    EXECUTED = "executed"
    DECODED = "decoded"
    FAILED = "failed"


@dataclass
class QuantumProblem:
    """A problem formulated for quantum execution.

    Encapsulates the parameters, constraints, and metadata needed to
    encode a problem into a quantum circuit.

    Attributes:
        name: Problem identifier (e.g., "binary_optimization").
        domain: Problem domain (e.g., "optimization").
        parameters: Problem-specific parameters.
        constraints: Constraints for validation.
        num_qubits: Requested number of qubits (may be adjusted).
        metadata: Additional problem metadata.
    """

    name: str
    domain: str
    parameters: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    num_qubits: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: ProblemStatus = ProblemStatus.CREATED
    created_at: float = field(default_factory=time.time)

    @property
    def problem_id(self) -> str:
        """Unique deterministic ID based on problem content."""
        content = json.dumps(
            {
                "name": self.name,
                "domain": self.domain,
                "parameters": _serialize_dict(self.parameters),
                "constraints": _serialize_dict(self.constraints),
                "num_qubits": self.num_qubits,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        return (
            f"QuantumProblem(name='{self.name}', domain='{self.domain}', "
            f"status={self.status.value})"
        )

    def __str__(self) -> str:
        lines = [
            f"QuantumProblem: {self.name}",
            f"  Domain: {self.domain}",
            f"  Status: {self.status.value}",
            f"  Parameters: {list(self.parameters.keys())}",
        ]
        if self.num_qubits:
            lines.append(f"  Qubits: {self.num_qubits}")
        return "\n".join(lines)


@dataclass
class QuantumResult:
    """Result from executing a quantum problem through an adapter.

    Contains both the raw quantum execution output and the decoded
    interpretation.

    Attributes:
        problem: The original problem.
        backend_result: Raw result from the quantum backend.
        decoded: Decoded result dictionary.
        fidelity: Solution fidelity (0-1).
        execution_time: Time spent in quantum execution (seconds).
        metadata: Additional result metadata.
    """

    problem: QuantumProblem
    backend_result: Optional[BackendResult] = None
    decoded: dict[str, Any] = field(default_factory=dict)
    fidelity: float = 0.0
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def most_frequent_state(self) -> str:
        """Most frequently measured quantum state."""
        if self.backend_result and self.backend_result.counts:
            return self.backend_result.most_frequent()
        return ""

    @property
    def probabilities(self) -> dict[str, float]:
        """Measurement probability distribution."""
        if self.backend_result:
            return self.backend_result.probabilities
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "problem": {
                "name": self.problem.name,
                "domain": self.problem.domain,
                "num_qubits": self.problem.num_qubits,
                "parameters": json_safe(self.problem.parameters),
                "constraints": json_safe(self.problem.constraints),
                "metadata": json_safe(self.problem.metadata),
                "status": self.problem.status.value,
            },
            "backend_result": (
                self.backend_result.to_dict() if self.backend_result is not None else None
            ),
            "decoded": json_safe(self.decoded),
            "fidelity": float(self.fidelity),
            "execution_time": float(self.execution_time),
            "metadata": json_safe(self.metadata),
            "most_frequent_state": self.most_frequent_state,
            "probabilities": dict(self.probabilities),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return f"QuantumResult(problem='{self.problem.name}', fidelity={self.fidelity:.4f})"

    def __str__(self) -> str:
        lines = [
            f"QuantumResult: {self.problem.name}",
            f"  Fidelity: {self.fidelity:.4f}",
            f"  Execution time: {self.execution_time:.4f}s",
        ]
        if self.decoded:
            lines.append("  Decoded results:")
            for k, v in self.decoded.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)


class DomainAdapter(ABC):
    """Abstract base class for domain-specific quantum adapters.

    Each adapter bridges a physics domain to the quantum SDK by:
    1. Validating that a problem has physically realizable parameters
    2. Encoding the problem into a parameterized quantum circuit
    3. Executing the circuit on a backend
    4. Decoding the quantum measurement results back to domain language

    Subclasses must implement all abstract methods.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Domain identifier (e.g., 'optimization', 'signal_processing')."""

    @property
    @abstractmethod
    def supported_problems(self) -> list[str]:
        """List of problem types this adapter handles."""

    @abstractmethod
    def validate(self, problem: QuantumProblem) -> list[str]:
        """Validate a problem's physics before encoding.

        Args:
            problem: The problem to validate.

        Returns:
            List of validation error messages. Empty list = valid.
        """

    @abstractmethod
    def encode(self, problem: QuantumProblem) -> QuantumCircuit:
        """Encode a validated problem into a quantum circuit.

        Args:
            problem: The validated problem.

        Returns:
            Parameterized quantum circuit representing the problem.
        """

    @abstractmethod
    def decode(
        self,
        problem: QuantumProblem,
        result: BackendResult,
    ) -> dict[str, Any]:
        """Decode quantum measurement results into domain results.

        Args:
            problem: The original problem.
            result: Raw backend execution result.

        Returns:
            Dictionary of domain-specific decoded results.
        """

    def solve(
        self,
        problem: QuantumProblem,
        backend: Backend,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> QuantumResult:
        """Full solve pipeline: validate → encode → execute → decode.

        Args:
            problem: The problem to solve.
            backend: Quantum backend to execute on.
            shots: Number of measurement shots.
            seed: Optional RNG seed.

        Returns:
            QuantumResult with decoded domain results.

        Raises:
            ValueError: If problem fails validation.
        """
        import time as _time

        # Validate
        errors = self.validate(problem)
        if errors:
            problem.status = ProblemStatus.FAILED
            raise ValueError("Problem validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        problem.status = ProblemStatus.VALIDATED

        # Encode
        circuit = self.encode(problem)
        problem.status = ProblemStatus.ENCODED

        # Execute via the canonical high-level Backend.run contract
        start = _time.time()
        backend_result = backend.run(
            circuit,
            shots=shots,
            seed=seed,
        )
        exec_time = _time.time() - start
        problem.status = ProblemStatus.EXECUTED

        # Decode
        decoded = self.decode(problem, backend_result)
        problem.status = ProblemStatus.DECODED

        # Compute fidelity from most frequent state
        fidelity = 0.0
        if backend_result.counts:
            total = sum(backend_result.counts.values())
            most_freq = max(backend_result.counts.values())
            fidelity = most_freq / total

        return QuantumResult(
            problem=problem,
            backend_result=backend_result,
            decoded=decoded,
            fidelity=fidelity,
            execution_time=exec_time,
            metadata={"adapter": self.domain_name, "shots": shots},
        )

    def can_handle(self, problem: QuantumProblem) -> bool:
        """Check if this adapter can handle a given problem."""
        return problem.domain == self.domain_name and problem.name in self.supported_problems

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(domain='{self.domain_name}', "
            f"problems={self.supported_problems})"
        )


def _serialize_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Serialize a dict for deterministic hashing."""
    result = {}
    for k, v in sorted(d.items()):
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, dict):
            result[k] = _serialize_dict(v)
        else:
            result[k] = v
    return result
