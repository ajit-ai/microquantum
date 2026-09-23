"""Higher-order (PUBO/HUBO) binary optimization problems.

Mirrors the :mod:`optimization.qubo` style (plain data + builder with
``None``-returning ``add_*`` methods): terms map variable-index tuples
to coefficients, covering linear, quadratic and higher-order terms in
one structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "PUBOProblem",
    "PUBOBuilder",
]


@dataclass
class PUBOProblem:
    """Polynomial unconstrained binary optimization problem.

    Attributes:
        terms: Mapping of variable-index tuples to coefficients.
            ``(i,)`` is linear, ``(i, j)`` quadratic, longer tuples
            higher-order.
        offset: Constant energy offset.
        num_variables: Number of binary variables.
        name: Problem identifier.
        metadata: Free-form metadata.
    """

    terms: dict[tuple[int, ...], float] = field(default_factory=dict)
    offset: float = 0.0
    num_variables: int = 0
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_variables < 0:
            raise ValueError("num_variables must be >= 0")
        for indices in self.terms:
            if not indices:
                raise ValueError("PUBO terms must reference at least one variable")
            if any(i < 0 or i >= self.num_variables for i in indices):
                raise ValueError(f"Term {indices} out of range for {self.num_variables} variables")

    def energy(self, x: np.ndarray) -> float:
        """Evaluate the polynomial at binary vector *x*."""
        bits = np.asarray(x).ravel()
        if len(bits) != self.num_variables:
            raise ValueError(
                f"Expected {self.num_variables} bits, got {len(bits)}"
            )
        total = float(self.offset)
        for indices, coefficient in self.terms.items():
            factor = 1.0
            for index in indices:
                factor *= float(bits[index])
            total += float(coefficient) * factor
        return total

    def max_order(self) -> int:
        """Highest term order (0 when there are no terms)."""
        return max((len(indices) for indices in self.terms), default=0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "terms": {",".join(str(i) for i in indices): coef for indices, coef in self.terms.items()},
            "offset": self.offset,
            "num_variables": self.num_variables,
            "name": self.name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PUBOProblem:
        """Deserialize from :meth:`to_dict` output."""
        terms = {
            tuple(int(i) for i in key.split(",") if i != ""): float(value)
            for key, value in data.get("terms", {}).items()
        }
        return cls(
            terms=terms,
            offset=float(data.get("offset", 0.0)),
            num_variables=int(data.get("num_variables", 0)),
            name=str(data.get("name", "")),
            metadata=dict(data.get("metadata", {})),
        )


class PUBOBuilder:
    """Incremental builder for :class:`PUBOProblem` (QUBOBuilder style)."""

    def __init__(self, num_variables: int) -> None:
        if num_variables < 1:
            raise ValueError(f"num_variables must be >= 1, got {num_variables}")
        self._num_variables = num_variables
        self._terms: dict[tuple[int, ...], float] = {}
        self._offset = 0.0

    @property
    def num_variables(self) -> int:
        """Number of binary variables."""
        return self._num_variables

    def add_term(self, indices: tuple[int, ...], coefficient: float) -> None:
        """Add (accumulating) a higher-order term."""
        if not indices:
            raise ValueError("Term must reference at least one variable")
        if any(i < 0 or i >= self._num_variables for i in indices):
            raise ValueError(f"Term {indices} out of range")
        key = tuple(sorted(indices))
        self._terms[key] = self._terms.get(key, 0.0) + float(coefficient)

    def add_linear(self, i: int, coefficient: float) -> None:
        """Add a linear term."""
        self.add_term((i,), coefficient)

    def add_quadratic(self, i: int, j: int, coefficient: float) -> None:
        """Add a quadratic term."""
        self.add_term((i, j), coefficient)

    def add_constant(self, value: float) -> None:
        """Add a constant offset."""
        self._offset += float(value)

    def build(self, name: str = "") -> PUBOProblem:
        """Build the :class:`PUBOProblem`."""
        return PUBOProblem(
            terms=dict(self._terms),
            offset=self._offset,
            num_variables=self._num_variables,
            name=name,
        )
