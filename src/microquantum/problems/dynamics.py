"""Dynamics and excited-state problem contracts.

Provides input contracts for Hamiltonian-simulation algorithms
(:class:`TimeEvolutionProblem`) and deflation-based excited-state
methods such as VQD (:class:`ExcitedStateProblem`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .base import Problem
from .eigenvalue import EigenvalueProblem, Hamiltonian, hamiltonian_from_dict, hamiltonian_to_dict

__all__ = [
    "TimeEvolutionProblem",
    "ExcitedStateProblem",
]


@dataclass
class TimeEvolutionProblem(Problem):
    """Evolve an initial state under a Hamiltonian for a given time.

    Attributes:
        hamiltonian: The Hamiltonian (``Operator`` or ``PauliSum``).
        time: Evolution time (must be >= 0).
        num_steps: Trotter/QDrift discretization steps (must be >= 1).
    """

    hamiltonian: Optional[Hamiltonian] = None
    time: float = 0.0
    num_steps: int = 1

    def validate(self) -> list[str]:
        """Return a list of problem defects (empty when valid)."""
        problems = super().validate()
        if self.hamiltonian is None:
            problems.append("hamiltonian is required")
        if self.time < 0:
            problems.append(f"time must be >= 0, got {self.time}")
        if self.num_steps < 1:
            problems.append(f"num_steps must be >= 1, got {self.num_steps}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        data = super().to_dict()
        data["hamiltonian"] = (
            hamiltonian_to_dict(self.hamiltonian) if self.hamiltonian is not None else None
        )
        data["time"] = self.time
        data["num_steps"] = self.num_steps
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeEvolutionProblem:
        """Deserialize from :meth:`to_dict` output."""
        raw = data.get("hamiltonian")
        return cls(
            name=str(data.get("name", "problem")),
            num_qubits=data.get("num_qubits"),
            metadata=dict(data.get("metadata", {})),
            hamiltonian=hamiltonian_from_dict(raw) if raw is not None else None,
            time=float(data.get("time", 0.0)),
            num_steps=int(data.get("num_steps", 1)),
        )


class ExcitedStateProblem(EigenvalueProblem):
    """Seek several lowest eigenvalues via deflation (VQD-style).

    Attributes:
        num_states: Number of low-lying states to resolve (``k`` indexes
            into them, so ``k <= num_states`` is required).
    """

    def __init__(
        self,
        hamiltonian: Optional[Hamiltonian] = None,
        *,
        num_states: int = 2,
        k: int = 1,
        num_qubits: Optional[int] = None,
        name: str = "problem",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if num_states < 1:
            raise ValueError(f"num_states must be >= 1, got {num_states}")
        self.num_states = num_states
        super().__init__(
            hamiltonian,
            k=k,
            num_qubits=num_qubits,
            name=name,
            metadata=metadata,
        )

    def validate(self) -> list[str]:
        """Return a list of problem defects (empty when valid)."""
        problems = super().validate()
        if self.k > self.num_states:
            problems.append(f"k={self.k} exceeds num_states={self.num_states}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Serialize, including ``num_states``."""
        data = super().to_dict()
        data["num_states"] = self.num_states
        return data
