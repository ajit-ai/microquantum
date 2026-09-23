"""Bridges from optimization formulations to Pauli operators.

:class:`IsingToPauli` validates diagonal (I/Z-only) Pauli sums and
normalizes them; :func:`qubo_to_pauli_sum` converts a
:class:`QUBOProblem` all the way to a :class:`PauliSum` via the
existing Ising converter — the one-direction bridge from
``optimization`` into ``core.pauli``.
"""

from __future__ import annotations

from .hubo import PUBOProblem
from .qubo import IsingConverter, QUBOProblem

__all__ = [
    "IsingToPauli",
    "qubo_to_pauli_sum",
    "pubo_to_qubo_projection",
]


class IsingToPauli:
    """Validate and normalize diagonal Pauli sums from Ising models."""

    @staticmethod
    def to_pauli_sum(ising: object) -> object:
        """Return the simplified :class:`PauliSum` for a diagonal Ising sum.

        Raises:
            TypeError: If *ising* is not a Pauli sum.
            ValueError: If any term contains X or Y operators.
        """
        from ..core.pauli import PauliString, PauliSum  # noqa: PLC0415

        if not isinstance(ising, PauliSum):
            raise TypeError(f"Expected a PauliSum, got {type(ising).__name__}")
        for term in ising.terms:
            if not isinstance(term, PauliString):
                raise TypeError("Ising terms must be PauliString instances")
            if any(char in ("X", "Y") for char in term.label):
                raise ValueError(
                    f"Ising term '{term.label}' is not diagonal (I/Z only)"
                )
        return ising.simplify()


def qubo_to_pauli_sum(qubo: QUBOProblem) -> object:
    """Convert a :class:`QUBOProblem` to a :class:`PauliSum`."""
    ising = IsingConverter.qubo_to_ising(qubo)
    return IsingToPauli.to_pauli_sum(ising)


def pubo_to_qubo_projection(pubo: PUBOProblem) -> QUBOProblem:
    """Project a :class:`PUBOProblem` onto its linear/quadratic part.

    Higher-order terms are dropped; the dropped coefficient mass is
    reported in the problem metadata under ``"dropped_higher_order"``.
    Use when a quadratic-only solver must handle higher-order input.
    """
    from .qubo import QUBOBuilder  # noqa: PLC0415

    builder = QUBOBuilder(pubo.num_variables)
    dropped = 0.0
    for indices, coefficient in pubo.terms.items():
        if len(indices) == 1:
            builder.add_linear(indices[0], coefficient)
        elif len(indices) == 2:
            builder.add_quadratic(indices[0], indices[1], coefficient)
        else:
            dropped += abs(float(coefficient))
    builder.add_constant(pubo.offset)
    problem = builder.build(name=pubo.name)
    problem.metadata["dropped_higher_order"] = dropped
    problem.metadata["source"] = "pubo-projection"
    return problem
