"""Hamiltonian and eigenvalue problem abstractions.

Operators (:class:`~microquantum.core.operators.Operator`) and Pauli sums
(:class:`~microquantum.core.pauli.PauliSum`) are the raw mathematical
objects; the problem classes here wrap them as *problems to be solved*.

``HamiltonianProblem`` is the generic statement "minimize/spectrum of this
Hermitian Hamiltonian"; ``EigenvalueProblem`` additionally records how many
lowest eigenvalues are sought.  Both stay fully domain-neutral — quantum
chemistry can later build on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from ..core.operators import Operator
from ..core.pauli import PauliSum
from .base import Problem

if TYPE_CHECKING:
    pass

Hamiltonian = Union[Operator, PauliSum]


def hamiltonian_expectation(hamiltonian: Any, state: Any) -> float:
    """Evaluate ``<state|hamiltonian|state>``.

    Accepts either a full-matrix :class:`Operator` or a
    :class:`PauliSum`.  Returns the real expectation value, raising
    ``ValueError`` for non-Hermitian operators.
    """
    if isinstance(hamiltonian, PauliSum):
        return hamiltonian.expectation(state)
    if isinstance(hamiltonian, Operator):
        psi = np.asarray(state.amplitudes)
        value = complex(np.vdot(psi, hamiltonian.matrix @ psi))
        if abs(value.imag) > 1e-7:
            raise ValueError(
                f"Expectation value has non-negligible imaginary part "
                f"({value.imag:.2e}). Hamiltonian may not be Hermitian."
            )
        return float(value.real)
    raise TypeError(
        f"unsupported Hamiltonian type {type(hamiltonian).__name__}; "
        f"expected Operator or PauliSum"
    )


def _hamiltonian_qubits(hamiltonian: Any) -> int:
    if isinstance(hamiltonian, (Operator, PauliSum)):
        return int(hamiltonian.num_qubits)
    raise TypeError(
        f"unsupported Hamiltonian type {type(hamiltonian).__name__}; "
        f"expected Operator or PauliSum"
    )


def hamiltonian_to_dict(hamiltonian: Any) -> dict[str, Any]:
    """Serialize an Operator/PauliSum Hamiltonian into a JSON-safe dict."""
    if isinstance(hamiltonian, PauliSum):
        return {
            "type": "PauliSum",
            "num_qubits": hamiltonian.num_qubits,
            "terms": [
                {
                    "label": str(t.label),
                    "coefficient": [
                        float(t.coefficient.real),
                        float(t.coefficient.imag),
                    ],
                }
                for t in hamiltonian.terms
            ],
        }
    if isinstance(hamiltonian, Operator):
        return {
            "type": "Operator",
            "num_qubits": hamiltonian.num_qubits,
            "matrix": np.asarray(hamiltonian.matrix, dtype=complex).tolist(),
        }
    return {"type": type(hamiltonian).__name__, "value": str(hamiltonian)}


@dataclass
class HamiltonianProblem(Problem):
    """A Hermitian Hamiltonian whose spectrum/eigenstates are of interest.

    Attributes:
        hamiltonian: :class:`Operator` or :class:`PauliSum`.
        num_qubits: Qubit count; inferred from the Hamiltonian when unset.
    """

    hamiltonian: Optional[Hamiltonian] = None

    def __init__(
        self,
        hamiltonian: Optional[Hamiltonian] = None,
        *,
        num_qubits: Optional[int] = None,
        name: str = "problem",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.num_qubits = num_qubits
        self.metadata = dict(metadata or {})
        self.hamiltonian = hamiltonian
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.hamiltonian is not None:
            n = _hamiltonian_qubits(self.hamiltonian)
            if self.num_qubits is None:
                self.num_qubits = n
            elif self.num_qubits != n:
                raise ValueError(
                    f"hamiltonian acts on {n} qubits but problem declares {self.num_qubits}"
                )
        super().__post_init__()

    def validate(self) -> list[str]:
        problems = super().validate()
        if self.hamiltonian is None:
            problems.append("hamiltonian is required")
        elif (
            isinstance(self.hamiltonian, Operator)
            and self.hamiltonian.num_qubits != self.num_qubits
        ):
            problems.append("hamiltonian.num_qubits does not match problem.num_qubits")
        return problems

    def expectation(self, state: Any) -> float:
        """Compute ``<psi|H|psi>`` for the wrapped Hamiltonian."""
        if self.hamiltonian is None:
            raise ValueError("hamiltonian is not set")
        return hamiltonian_expectation(self.hamiltonian, state)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["hamiltonian"] = (
            hamiltonian_to_dict(self.hamiltonian) if self.hamiltonian else None
        )
        return data


@dataclass
class EigenvalueProblem(HamiltonianProblem):
    """Request the ``k`` lowest eigenvalues of a Hamiltonian.

    Attributes:
        k: Number of lowest eigenvalues sought (>= 1).
    """

    k: int = 1

    def __init__(
        self,
        hamiltonian: Optional[Hamiltonian] = None,
        *,
        k: int = 1,
        num_qubits: Optional[int] = None,
        name: str = "problem",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.num_qubits = num_qubits
        self.metadata = dict(metadata or {})
        self.hamiltonian = hamiltonian
        self.k = k
        self.__post_init__()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")

    def validate(self) -> list[str]:
        problems = super().validate()
        if self.k < 1:
            problems.append("k must be >= 1")
        return problems

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["k"] = self.k
        return data


__all__ = [
    "HamiltonianProblem",
    "EigenvalueProblem",
    "hamiltonian_expectation",
    "hamiltonian_to_dict",
    "Hamiltonian",
]