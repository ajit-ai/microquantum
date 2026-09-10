"""Pauli string and Pauli sum for efficient observable representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
from numpy.typing import NDArray

from .state import StateVector

if TYPE_CHECKING:
    from .operators import Operator


class PauliString:
    """A single Pauli term: coefficient * (P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1}).

    Attributes:
        label: Pauli label string, e.g. "XYZI" (qubit 0 = leftmost).
        coefficient: Complex coefficient.
        num_qubits: Number of qubits this acts on.
    """

    IDENTITY = "I"
    PAULI_X = "X"
    PAULI_Y = "Y"
    PAULI_Z = "Z"

    _VALID_GATES = frozenset({IDENTITY, PAULI_X, PAULI_Y, PAULI_Z})

    def __init__(self, label: str, coefficient: complex = 1.0) -> None:
        """Initialize a PauliString.

        Args:
            label: Pauli label string, e.g. "XYZ" (qubit 0 = leftmost).
            coefficient: Complex coefficient.

        Raises:
            ValueError: If label contains invalid characters.
        """
        label = label.upper()
        if not label:
            raise ValueError("Label must not be empty")
        if not all(g in self._VALID_GATES for g in label):
            raise ValueError(
                f"Invalid Pauli label '{label}'. "
                f"Valid characters: {self._VALID_GATES}"
            )
        self._label = label
        self._coefficient = complex(coefficient)
        self._num_qubits = len(label)

    @property
    def label(self) -> str:
        """Pauli label string."""
        return self._label

    @property
    def coefficient(self) -> complex:
        """Complex coefficient."""
        return self._coefficient

    @property
    def num_qubits(self) -> int:
        """Number of qubits this acts on."""
        return self._num_qubits

    @property
    def is_hermitian(self) -> bool:
        """Check if PauliString is Hermitian.

        A PauliString is Hermitian if its coefficient is real and
        the Pauli operators are all Hermitian (X, Y, Z, I are all Hermitian).
        """
        return bool(np.isreal(self._coefficient))

    def to_operator(self) -> Operator:
        """Convert to full matrix Operator (expensive for large n).

        Returns:
            Operator representing the full 2^n × 2^n matrix.
        """
        from .operators import Operator

        pauli_matrices = {
            "I": np.eye(2, dtype=np.complex128),
            "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }

        result = np.array([1.0], dtype=np.complex128)
        for pauli in self._label:
            result = np.asarray(np.kron(result, pauli_matrices[pauli]), dtype=np.complex128)
        return Operator(self._coefficient * np.asarray(result, dtype=np.complex128))

    def expectation(self, state: StateVector) -> float:
        """Compute <psi|PauliString|psi>.

        Uses matrix computation for correctness across all Pauli combinations.

        Args:
            state: Quantum state vector.

        Returns:
            Expectation value <psi|PauliString|psi>.

        Raises:
            ValueError: If state dimension doesn't match PauliString size.
        """
        if state.num_qubits != self._num_qubits:
            raise ValueError(
                f"State has {state.num_qubits} qubits but "
                f"PauliString has {self._num_qubits}"
            )

        op = self.to_operator()
        amps = state.amplitudes
        result = float(np.real(np.conj(amps) @ op.matrix @ amps))
        return result

    def tensor(self, other: PauliString) -> PauliString:
        """Tensor product of two Pauli strings.

        Args:
            other: Another PauliString.

        Returns:
            New PauliString with label = self.label + other.label.
        """
        return PauliString(
            self._label + other._label,
            self._coefficient * other._coefficient,
        )

    def __mul__(self, other: PauliString | complex | float | int) -> PauliString | "PauliSum":
        """Product with another PauliString or scalar multiplication."""
        if isinstance(other, PauliString):
            if self._num_qubits != other._num_qubits:
                raise ValueError(
                    f"PauliString sizes differ: {self._num_qubits} vs {other._num_qubits}"
                )

            product_map = {
                ("I", "I"): ("I", 1),
                ("I", "X"): ("X", 1), ("X", "I"): ("X", 1),
                ("I", "Y"): ("Y", 1), ("Y", "I"): ("Y", 1),
                ("I", "Z"): ("Z", 1), ("Z", "I"): ("Z", 1),
                ("X", "X"): ("I", 1), ("Y", "Y"): ("I", 1), ("Z", "Z"): ("I", 1),
                ("X", "Y"): ("Z", 1j), ("Y", "X"): ("Z", -1j),
                ("Y", "Z"): ("X", 1j), ("Z", "Y"): ("X", -1j),
                ("Z", "X"): ("Y", 1j), ("X", "Z"): ("Y", -1j),
            }

            new_label = []
            total_phase: complex = 1.0
            for p1, p2 in zip(self._label, other._label):
                result_pauli, phase = product_map[(p1, p2)]
                new_label.append(result_pauli)
                total_phase *= phase

            return PauliString(
                "".join(new_label),
                self._coefficient * other._coefficient * total_phase,
            )
        return PauliString(self._label, self._coefficient * complex(other))

    def __rmul__(self, scalar: complex | float | int) -> PauliString:
        """Scalar multiplication: scalar * pauli_string."""
        return PauliString(self._label, complex(scalar) * self._coefficient)

    def __add__(self, other: PauliString) -> "PauliSum":
        """Add two Pauli strings to form a PauliSum."""
        from .pauli import PauliSum
        if isinstance(other, PauliString):
            return PauliSum([self, other])
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Check equality of two PauliStrings."""
        if not isinstance(other, PauliString):
            return False
        return bool(
            self._label == other._label
            and np.isclose(self._coefficient, other._coefficient)
        )

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash((self._label, self._coefficient))

    def __repr__(self) -> str:
        """String representation."""
        if self._coefficient == 1.0:
            return f"PauliString('{self._label}')"
        return f"PauliString('{self._label}', {self._coefficient})"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self._coefficient} * {self._label}"


class PauliSum:
    """Weighted sum of Pauli strings: H = Σ c_i * P_i.

    This is the efficient representation for Hamiltonians.
    Avoids constructing full 2^N matrices.
    """

    def __init__(self, terms: list[PauliString] | None = None) -> None:
        """Initialize a PauliSum.

        Args:
            terms: List of PauliStrings to sum.
        """
        self._terms: list[PauliString] = list(terms) if terms else []

    @classmethod
    def from_label(cls, label: str, coefficient: complex = 1.0) -> PauliSum:
        """Create from a single label like 'XYZ'.

        Args:
            label: Pauli label string.
            coefficient: Complex coefficient.

        Returns:
            PauliSum containing a single PauliString term.
        """
        return cls([PauliString(label, coefficient)])

    @property
    def num_qubits(self) -> int:
        """Number of qubits (from the longest term)."""
        if not self._terms:
            return 0
        return max(t.num_qubits for t in self._terms)

    @property
    def num_terms(self) -> int:
        """Number of terms in the sum."""
        return len(self._terms)

    @property
    def is_hermitian(self) -> bool:
        """Check if all terms are Hermitian."""
        return all(t.is_hermitian for t in self._terms)

    @property
    def terms(self) -> list[PauliString]:
        """List of terms."""
        return list(self._terms)

    def expectation(self, state: StateVector) -> float:
        """Compute <psi|H|psi> using Pauli decomposition (no matrix).

        Args:
            state: Quantum state vector.

        Returns:
            Expectation value.
        """
        return sum(t.expectation(state) for t in self._terms)

    def to_operator(self) -> Operator:
        """Convert to full matrix (expensive).

        Returns:
            Operator representing the full 2^n × 2^n matrix.
        """
        from .operators import Operator

        if not self._terms:
            raise ValueError("Cannot convert empty PauliSum to Operator")

        n = self.num_qubits
        dim = 2**n
        result = np.zeros((dim, dim), dtype=np.complex128)

        for term in self._terms:
            # Pad label with I's if needed
            padded_label = term.label.rjust(n, "I")
            padded_term = PauliString(padded_label, term.coefficient)
            result += padded_term.to_operator().matrix

        return Operator(result)

    def simplify(self) -> PauliSum:
        """Combine like terms and remove zeros.

        Returns:
            New PauliSum with combined like terms.
        """
        from collections import defaultdict

        combined: dict[str, complex] = defaultdict(lambda: 0.0)
        for term in self._terms:
            combined[term.label] += term.coefficient

        new_terms = []
        for label, coeff in combined.items():
            if not np.isclose(coeff, 0.0):
                new_terms.append(PauliString(label, coeff))

        return PauliSum(new_terms)

    def __add__(self, other: PauliSum | PauliString) -> PauliSum:
        """Add PauliSum or PauliString."""
        if isinstance(other, PauliString):
            return PauliSum(self._terms + [other])
        if isinstance(other, PauliSum):
            return PauliSum(self._terms + other._terms)
        return NotImplemented

    def __radd__(self, other: PauliString) -> PauliSum:
        """Right addition: PauliString + PauliSum."""
        if isinstance(other, PauliString):
            return PauliSum([other] + self._terms)
        return NotImplemented

    def __sub__(self, other: PauliSum | PauliString) -> PauliSum:
        """Subtract PauliSum or PauliString."""
        if isinstance(other, PauliString):
            return PauliSum(self._terms + [PauliString(other.label, -other.coefficient)])
        if isinstance(other, PauliSum):
            neg_terms = [PauliString(t.label, -t.coefficient) for t in other._terms]
            return PauliSum(self._terms + neg_terms)
        return NotImplemented

    def __mul__(self, scalar: complex) -> PauliSum:
        """Scalar multiplication: pauli_sum * scalar."""
        return PauliSum([PauliString(t.label, t.coefficient * scalar) for t in self._terms])

    def __rmul__(self, scalar: complex) -> PauliSum:
        """Scalar multiplication: scalar * pauli_sum."""
        return self.__mul__(scalar)

    def __len__(self) -> int:
        """Number of terms."""
        return len(self._terms)

    def __iter__(self) -> Iterator[PauliString]:
        """Iterate over terms."""
        return iter(self._terms)

    def __getitem__(self, index: int) -> PauliString:
        """Get term by index."""
        return self._terms[index]

    def __repr__(self) -> str:
        """String representation."""
        if not self._terms:
            return "PauliSum([])"
        terms_str = ", ".join(repr(t) for t in self._terms)
        return f"PauliSum([{terms_str}])"

    def __str__(self) -> str:
        """Human-readable string."""
        if not self._terms:
            return "0"
        return " + ".join(str(t) for t in self._terms)
