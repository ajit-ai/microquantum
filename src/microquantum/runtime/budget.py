"""Execution budgets: pre-flight cost guards for plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .errors import PlanningError

__all__ = [
    "Budget",
]


@dataclass(frozen=True)
class Budget:
    """Upper bounds on execution cost.

    Attributes:
        max_shots: Maximum total shots (``None`` = unbounded).
        max_circuits: Maximum number of circuits (``None`` = unbounded).
        max_wall_s: Maximum wall-clock seconds (``None`` = unbounded,
            enforced by the runtime before dispatch).
    """

    max_shots: Optional[int] = None
    max_circuits: Optional[int] = None
    max_wall_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_shots is not None and self.max_shots < 1:
            raise ValueError("max_shots must be >= 1")
        if self.max_circuits is not None and self.max_circuits < 1:
            raise ValueError("max_circuits must be >= 1")
        if self.max_wall_s is not None and self.max_wall_s <= 0:
            raise ValueError("max_wall_s must be positive")

    def check(self, *, shots: Optional[int] = None, circuits: int = 1) -> None:
        """Raise :class:`PlanningError` when the request exceeds budget."""
        if (
            shots is not None
            and self.max_shots is not None
            and shots > self.max_shots
        ):
            raise PlanningError(
                f"Requested {shots} shots exceeds budget of {self.max_shots}"
            )
        if self.max_circuits is not None and circuits > self.max_circuits:
            raise PlanningError(
                f"Requested {circuits} circuits exceeds budget of {self.max_circuits}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "max_shots": self.max_shots,
            "max_circuits": self.max_circuits,
            "max_wall_s": self.max_wall_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Budget:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            max_shots=data.get("max_shots"),
            max_circuits=data.get("max_circuits"),
            max_wall_s=data.get("max_wall_s"),
        )
