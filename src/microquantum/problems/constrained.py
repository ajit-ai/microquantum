"""Constrained combinatorial optimization problems.

Extends :class:`OptimizationProblem` with linear constraints handled by
penalty weights, so constrained problems flow through the same
algorithm interfaces (QAOA, VQE, brute-force) as unconstrained ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .optimization import OptimizationProblem

__all__ = [
    "LinearConstraint",
    "ConstrainedOptimizationProblem",
]


@dataclass(frozen=True)
class LinearConstraint:
    """A single linear constraint over binary variables.

    Attributes:
        indices: Variable indices participating in the constraint.
        sense: One of ``"<="``, ``"=="`` or ``">="``.
        rhs: Right-hand-side value.
        penalty: Penalty weight applied per unit of squared violation.
    """

    indices: tuple[int, ...]
    sense: str
    rhs: float
    penalty: float = 10.0

    def __post_init__(self) -> None:
        if self.sense not in ("<=", "==", ">="):
            raise ValueError(f"sense must be '<=', '==' or '>=', got {self.sense!r}")
        if not self.indices:
            raise ValueError("LinearConstraint requires at least one index")
        if any(i < 0 for i in self.indices):
            raise ValueError("Variable indices must be >= 0")
        if self.penalty < 0:
            raise ValueError("penalty must be >= 0")

    def violation(self, bits: np.ndarray) -> float:
        """Non-negative constraint violation for a binary vector."""
        total = float(sum(int(bits[i]) for i in self.indices))
        if self.sense == "<=":
            return max(0.0, total - self.rhs)
        if self.sense == ">=":
            return max(0.0, self.rhs - total)
        return abs(total - self.rhs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "indices": list(self.indices),
            "sense": self.sense,
            "rhs": self.rhs,
            "penalty": self.penalty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinearConstraint:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            indices=tuple(int(i) for i in data["indices"]),
            sense=str(data["sense"]),
            rhs=float(data["rhs"]),
            penalty=float(data.get("penalty", 10.0)),
        )


@dataclass
class ConstrainedOptimizationProblem(OptimizationProblem):
    """An optimization problem with linear constraints.

    The penalized energy ``energy(bits) + penalty(bits)`` is what
    algorithms minimize; :meth:`penalty` is zero exactly when all
    constraints hold.
    """

    constraints: list[LinearConstraint] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Return a list of problem defects (empty when valid)."""
        problems = super().validate()
        for position, constraint in enumerate(self.constraints):
            if not isinstance(constraint, LinearConstraint):
                problems.append(f"constraints[{position}] is not a LinearConstraint")
                continue
            for index in constraint.indices:
                if index >= self.num_variables:
                    problems.append(
                        f"constraints[{position}] references variable {index} "
                        f"but num_variables={self.num_variables}"
                    )
        return problems

    def penalty(self, bits: np.ndarray) -> float:
        """Total penalty for constraint violations."""
        return float(sum(c.penalty * c.violation(bits) ** 2 for c in self.constraints))

    def penalized_energy(self, bits: np.ndarray) -> float:
        """Objective energy plus constraint penalty."""
        return float(self.energy(bits) + self.penalty(bits))

    def is_feasible(self, bits: np.ndarray) -> bool:
        """True when all constraints hold (zero violation)."""
        return all(c.violation(bits) <= 0.0 for c in self.constraints)

    def to_dict(self) -> dict[str, Any]:
        """Serialize, including constraints."""
        data = super().to_dict()
        data["constraints"] = [c.to_dict() for c in self.constraints]
        return data
