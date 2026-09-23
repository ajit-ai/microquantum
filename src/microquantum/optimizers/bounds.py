"""Box constraints shared by iterative optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ..core.parameter import Parameter

__all__ = [
    "Bounds",
]


@dataclass(frozen=True)
class Bounds:
    """Scalar box constraints applied elementwise to parameters.

    Attributes:
        lower: Lower bound (``None`` = unbounded below).
        upper: Upper bound (``None`` = unbounded above).
    """

    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self) -> None:
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"lower ({self.lower}) exceeds upper ({self.upper})")

    def project_value(self, value: float) -> float:
        """Clip a single value into the box."""
        if self.lower is not None:
            value = max(self.lower, value)
        if self.upper is not None:
            value = min(self.upper, value)
        return value

    def project(self, params: Mapping[Parameter, float]) -> dict[Parameter, float]:
        """Clip every parameter value into the box."""
        return {param: self.project_value(float(value)) for param, value in params.items()}

    def to_dict(self) -> dict[str, Optional[float]]:
        """Serialize to a JSON-safe dictionary."""
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_dict(cls, data: dict[str, Optional[float]]) -> Bounds:
        """Deserialize from :meth:`to_dict` output."""
        return cls(lower=data.get("lower"), upper=data.get("upper"))
