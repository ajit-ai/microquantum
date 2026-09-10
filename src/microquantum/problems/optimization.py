"""Generic combinatorial optimization problem abstraction.

An :class:`OptimizationProblem` describes an objective over binary
variables *without* encoding how it is optimized.  It supports two
equivalent, interchangeable views of the same problem:

* an objective function ``objective(bits) -> float`` over binary vectors,
* an Ising Hamiltonian (spin variables) ``cost_hamiltonian()``.

Classmethods :meth:`OptimizationProblem.from_qubo` and
:meth:`OptimizationProblem.from_ising` construct problems from the QUBO /
Ising formulations used by combinatorial-optimization tooling, so the same
problem object stays usable by algorithms (QAOA) as well as classical
brute-force evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from .base import Problem

if TYPE_CHECKING:
    pass


@dataclass
class OptimizationProblem(Problem):
    """Minimize an objective over ``num_variables`` binary variables.

    Attributes:
        num_variables: Number of binary variables.
        objective: Optional ``objective(bits) -> float`` over a binary vector.
            Required unless ``ising`` is provided.
        ising: Optional spin-representation Hamiltonian (:class:`PauliSum`).
        sampler: Optional ``sampler(num_variables) -> bits`` used by
            algorithms to sample candidate solutions classically.
    """

    num_variables: int = 1
    objective: Optional[Callable[[np.ndarray], float]] = None
    ising: Optional[Any] = None
    sampler: Optional[Callable[[int], np.ndarray]] = None

    def __post_init__(self) -> None:
        if self.num_qubits is not None:
            raise ValueError(
                "OptimizationProblem expresses size through num_variables, "
                "not num_qubits"
            )
        if self.num_variables < 1:
            raise ValueError(f"num_variables must be >= 1, got {self.num_variables}")
        if isinstance(self.ising, (np.ndarray, list, tuple, float, int)):
            raise TypeError(
                "ising must be a PauliSum (spin Hamiltonian), not a raw matrix"
            )

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        problems = super().validate()
        if self.num_variables < 1:
            problems.append("num_variables must be >= 1")
        if self.objective is None and self.ising is None:
            problems.append("objective or ising Hamiltonian is required")
        return problems

    # ------------------------------------------------------------------
    # view conversions
    # ------------------------------------------------------------------

    @classmethod
    def from_ising(
        cls,
        pauli_sum: Any,
        *,
        name: str = "optimization",
    ) -> "OptimizationProblem":
        """Build a problem whose Ising cost Hamiltonian is ``pauli_sum``."""
        from ..core.pauli import PauliSum

        if not isinstance(pauli_sum, PauliSum):
            raise TypeError(f"expected a PauliSum, got {type(pauli_sum).__name__}")
        return cls(num_variables=pauli_sum.num_qubits, ising=pauli_sum, name=name)

    @classmethod
    def from_qubo(
        cls,
        qubo: Any,
        *,
        name: str = "optimization",
    ) -> "OptimizationProblem":
        """Build a problem from a ``QUBOProblem``.

        The objective is the QUBO energy and the Ising view is derived via
        the standard QUBO -> Ising mapping.
        """
        from ..optimization.qubo import IsingConverter, QUBOProblem

        if not isinstance(qubo, QUBOProblem):
            raise TypeError(f"expected a QUBOProblem, got {type(qubo).__name__}")

        ising = IsingConverter.qubo_to_ising(qubo)
        return cls(
            num_variables=qubo.num_variables,
            objective=lambda bits: float(qubo.energy(np.asarray(bits, dtype=float))),
            ising=ising,
            name=name,
        )

    def cost_hamiltonian(self) -> Any:
        """Return the Ising cost Hamiltonian (:class:`PauliSum`).

        Raises:
            ValueError: If the problem was defined purely by a binary
                objective and carries no Ising view.
        """
        if self.ising is None:
            raise ValueError(
                "OptimizationProblem has no Ising cost Hamiltonian; "
                "construct it with from_qubo()/from_ising() or pass ising="
            )
        return self.ising

    def to_qubo(self) -> Any:
        """Convert the Ising view back to a ``QUBOProblem`` when available."""
        from ..optimization.qubo import IsingConverter

        if self.ising is not None:
            return IsingConverter.ising_to_qubo(self.ising)
        raise ValueError("no Ising view to convert; supply ising=")

    # ------------------------------------------------------------------
    # evaluation helpers
    # ------------------------------------------------------------------

    def energy(self, bits: np.ndarray) -> float:
        """Evaluate the objective (or the Ising Hamiltonian) on a bit vector.

        Args:
            bits: Binary vector of shape (num_variables,).

        Raises:
            ValueError: If the vector has the wrong length.
        """
        bits = np.asarray(bits, dtype=float)
        if bits.shape != (self.num_variables,):
            raise ValueError(
                f"expected a bit vector of length {self.num_variables}, "
                f"got shape {bits.shape}"
            )
        if self.objective is not None:
            return float(self.objective(bits))
        return self._ising_spin_energy(self.encode_spins(bits))

    def encode_spins(self, bits: np.ndarray) -> np.ndarray:
        """Map a binary vector to spin values ``s = 1 - 2 * x``."""
        bits = np.asarray(bits, dtype=float)
        return 1.0 - 2.0 * bits

    def sample(self, seed: Optional[int] = None) -> np.ndarray:
        """Sample a candidate bit vector classically.

        Uses the problem's ``sampler`` when provided; otherwise a uniform
        random bit vector (``seed`` applied).
        """
        if self.sampler is not None:
            return np.asarray(self.sampler(self.num_variables), dtype=float)
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=self.num_variables).astype(float)

    def _ising_spin_energy(self, spin: np.ndarray) -> float:
        if self.ising is None:
            raise ValueError("no objective or Ising Hamiltonian to evaluate")
        total = 0.0
        for term in self.ising.terms:
            coeff = float(term.coefficient.real)
            label = term.label
            value = 1.0
            for k, ch in enumerate(label):
                if ch == "Z":
                    value *= spin[k]
            total += coeff * value
        return total

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    @property
    def qubits_needed(self) -> int:
        """Number of qubits an exact solver must allocate for this problem."""
        return self.num_variables

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["num_variables"] = self.num_variables
        data["num_qubits"] = self.qubits_needed
        data["objective"] = None
        data["sampler"] = None
        if self.ising is None:
            data["ising"] = None
        else:
            from .eigenvalue import hamiltonian_to_dict

            data["ising"] = hamiltonian_to_dict(self.ising)
        return data

    def __str__(self) -> str:
        return (
            f"OptimizationProblem(name='{self.name}', "
            f"num_variables={self.num_variables})"
        )


def standard_binary_encoding(value: int, num_bits: int) -> np.ndarray:
    """Encode a non-negative integer as a binary bit vector (MSB first).

    Args:
        value: Integer in ``[0, 2**num_bits)`` to encode.
        num_bits: Number of bits (variables) to use.

    Returns:
        Float bit vector of length ``num_bits``.
    """
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if value >= (1 << num_bits):
        raise ValueError(
            f"value {value} does not fit in {num_bits} bits "
            f"(max {2 ** num_bits - 1})"
        )
    bits = np.zeros(num_bits, dtype=float)
    for k in range(num_bits):
        bits[num_bits - 1 - k] = float((value >> k) & 1)
    return bits


__all__ = ["OptimizationProblem", "standard_binary_encoding"]