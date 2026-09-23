"""Integral sources, active spaces and fermionic transforms for chemistry.

:class:`IntegralSource` is the interface future PySCF/FCIDUMP loaders
implement (no dependency added here); :class:`ActiveSpace` selects
core/active orbital windows with validation; :class:`FermionicOp` plus
:func:`jordan_wigner` implement the real Jordan-Wigner transform into
:mod:`core.pauli`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

__all__ = [
    "IntegralSource",
    "ActiveSpace",
    "ActiveSpaceSelection",
    "FermionicOp",
    "jordan_wigner",
]


class IntegralSource(Protocol):
    """Interface for molecular integral providers."""

    @property
    def num_orbitals(self) -> int:
        """Number of spatial orbitals."""
        ...  # pragma: no cover - protocol stub

    def one_body(self) -> Sequence[Sequence[float]]:
        """One-body integrals ``h[p, q]``."""
        ...  # pragma: no cover - protocol stub

    def two_body(self) -> Sequence[Sequence[Sequence[Sequence[float]]]]:
        """Two-body integrals ``g[p, q, r, s]`` in physicist notation."""
        ...  # pragma: no cover - protocol stub

    def nuclear_repulsion(self) -> float:
        """Nuclear repulsion energy."""
        ...  # pragma: no cover - protocol stub


@dataclass(frozen=True)
class ActiveSpaceSelection:
    """Resolved core/active orbital partition."""

    num_core_orbitals: int
    num_active_orbitals: int
    num_active_electrons: int
    frozen_core_energy: float = 0.0

    @property
    def num_qubits(self) -> int:
        """Qubits needed (one per spin-orbital)."""
        return 2 * self.num_active_orbitals

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "num_core_orbitals": self.num_core_orbitals,
            "num_active_orbitals": self.num_active_orbitals,
            "num_active_electrons": self.num_active_electrons,
            "frozen_core_energy": self.frozen_core_energy,
        }


@dataclass(frozen=True)
class ActiveSpace:
    """Core/active orbital window selector.

    Attributes:
        num_core_orbitals: Frozen doubly-occupied orbitals below the window.
        num_active_orbitals: Correlated orbitals in the window.
    """

    num_core_orbitals: int = 0
    num_active_orbitals: int = 0

    def __post_init__(self) -> None:
        if self.num_core_orbitals < 0:
            raise ValueError("num_core_orbitals must be >= 0")
        if self.num_active_orbitals < 1:
            raise ValueError("num_active_orbitals must be >= 1")

    def select(
        self,
        num_electrons: int,
        num_orbitals: int,
        frozen_core_energy: float = 0.0,
    ) -> ActiveSpaceSelection:
        """Resolve the partition for a system with *num_electrons* in
        *num_orbitals* spatial orbitals.

        Raises:
            ValueError: If the window does not fit the system.
        """
        if num_electrons < 0:
            raise ValueError("num_electrons must be >= 0")
        if self.num_core_orbitals + self.num_active_orbitals > num_orbitals:
            raise ValueError(
                f"Active space ({self.num_core_orbitals} core + "
                f"{self.num_active_orbitals} active) exceeds {num_orbitals} orbitals"
            )
        frozen_electrons = 2 * self.num_core_orbitals
        if frozen_electrons > num_electrons:
            raise ValueError("Frozen core holds more electrons than the system has")
        return ActiveSpaceSelection(
            num_core_orbitals=self.num_core_orbitals,
            num_active_orbitals=self.num_active_orbitals,
            num_active_electrons=num_electrons - frozen_electrons,
            frozen_core_energy=float(frozen_core_energy),
        )


@dataclass(frozen=True)
class FermionicOp:
    """A fermionic operator term: coefficient times ladder operators.

    Operators are ``(action, index)`` pairs with ``action`` in
    ``{"+", "-"}`` (create/annihilate) applied right-to-left, e.g.
    ``((( "+", 0), ("-", 1)),)`` is ``a†₀ a₁``.
    """

    terms: Mapping[tuple[tuple[str, int], ...], complex] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for operators in self.terms:
            for action, index in operators:
                if action not in ("+", "-"):
                    raise ValueError(f"Action must be '+' or '-', got {action!r}")
                if index < 0:
                    raise ValueError("Orbital indices must be >= 0")

    @property
    def num_orbitals(self) -> int:
        """Number of spin-orbitals touched (0 when empty)."""
        highest = -1
        for operators in self.terms:
            for _, index in operators:
                highest = max(highest, index)
        return highest + 1


def _jordan_wigner_ladder(action: str, index: int, num_qubits: int) -> list[tuple[str, complex]]:
    """Single ladder operator as ``(pauli_label, coefficient)`` terms."""
    if not 0 <= index < num_qubits:
        raise ValueError("Orbital index out of range")
    prefix = "Z" * index
    suffix = "I" * (num_qubits - index - 1)
    if action == "+":
        return [
            (prefix + "X" + suffix, 0.5),
            (prefix + "Y" + suffix, -0.5j),
        ]
    return [
        (prefix + "X" + suffix, 0.5),
        (prefix + "Y" + suffix, 0.5j),
    ]


def jordan_wigner(operator: FermionicOp, num_qubits: int | None = None) -> Any:
    """Jordan-Wigner transform of a fermionic operator to a Pauli sum.

    Args:
        operator: Fermionic operator to transform.
        num_qubits: Qubit count (defaults to the operator span).

    Returns:
        A simplified :class:`PauliSum`.
    """
    from ..core.pauli import PauliString, PauliSum  # noqa: PLC0415

    nq = num_qubits if num_qubits is not None else operator.num_orbitals
    if nq < 1:
        raise ValueError("Need at least one qubit")
    if operator.num_orbitals > nq:
        raise ValueError("Operator spans more orbitals than num_qubits")
    result = PauliSum([])
    for operators, coefficient in operator.terms.items():
        factor_lists = [
            _jordan_wigner_ladder(action, index, nq) for action, index in operators
        ]
        combos: list[tuple[list[str], complex]] = [([], 1.0)]
        for choices in factor_lists:
            combos = [
                (labels + [next_label], phase * next_phase)
                for labels, phase in combos
                for next_label, next_phase in choices
            ]
        for labels, phase in combos:
            product: PauliString = PauliString("I" * nq, 1.0)
            for label in labels:
                step = product * PauliString(label, 1.0)
                assert isinstance(step, PauliString)
                product = step
            result = result + PauliSum(
                [PauliString(product.label, coefficient * phase * product.coefficient)]
            )
    return result.simplify()
