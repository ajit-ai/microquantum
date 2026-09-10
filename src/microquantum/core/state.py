"""State vector representation for quantum states."""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class StateVector:
    """A quantum state vector in the computational basis.

    Represents an N-qubit quantum state as a complex vector of dimension 2^N,
    where each component corresponds to the amplitude of a computational basis
    state.

    Attributes:
        num_qubits: Number of qubits in the system.
        dim: Dimension of the state vector (2^num_qubits).
        amplitudes: Complex array of state amplitudes.
        is_normalized: Whether the state vector is normalized (unit norm).
    """

    def __init__(
        self,
        num_qubits: int,
        amplitudes: Optional[NDArray[np.complex128]] = None,
    ) -> None:
        """Initialize a quantum state vector.

        Args:
            num_qubits: Number of qubits. Must be a positive integer.
            amplitudes: Optional array of complex amplitudes. If provided,
                must have length 2^num_qubits and complex dtype. If None,
                defaults to the computational basis state |0...0>.

        Raises:
            ValueError: If num_qubits is not positive, or amplitudes have
                incorrect length or non-complex dtype.
        """
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")

        self._num_qubits = num_qubits
        self._dim = 2**num_qubits

        if amplitudes is not None:
            if amplitudes.shape != (self._dim,):
                raise ValueError(
                    f"amplitudes must have length {self._dim}, "
                    f"got {amplitudes.shape}"
                )
            self._amplitudes = np.array(amplitudes, dtype=np.complex128)
        else:
            self._amplitudes = np.zeros(self._dim, dtype=np.complex128)
            self._amplitudes[0] = 1.0

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the system."""
        return self._num_qubits

    @property
    def dim(self) -> int:
        """Dimension of the state vector (2^num_qubits)."""
        return int(self._dim)

    @property
    def amplitudes(self) -> NDArray[np.complex128]:
        """Complex array of state amplitudes."""
        return self._amplitudes

    @property
    def is_normalized(self) -> bool:
        """Check if the state vector is normalized to within 1e-9 tolerance."""
        norm_sq = float(np.sum(np.abs(self._amplitudes) ** 2))
        return abs(norm_sq - 1.0) < 1e-9

    def normalize(self) -> StateVector:
        """Normalize the state vector in-place.

        Scales amplitudes so that the sum of squared magnitudes equals 1.

        Returns:
            self, for method chaining.
        """
        norm = np.linalg.norm(self._amplitudes)
        if norm > 0:
            self._amplitudes = self._amplitudes / norm
        return self

    def inner_product(self, other: StateVector) -> complex:
        """Compute the inner product <other | self>.

        Uses the standard quantum mechanical convention where the bra vector
        (other) is conjugate-transposed.

        Args:
            other: The other state vector (ket) to compute <other|self>.

        Returns:
            Complex inner product <other | self>.

        Raises:
            ValueError: If the other state vector has different dimension.
        """
        if other.dim != self._dim:
            raise ValueError(
                f"Dimension mismatch: self.dim={self._dim}, "
                f"other.dim={other.dim}"
            )
        return complex(np.vdot(other._amplitudes, self._amplitudes))

    def fidelity(self, other: StateVector) -> float:
        """Compute the fidelity |<other | self>|^2.

        Fidelity measures the overlap between two quantum states.
        Returns 1.0 for identical states and 0.0 for orthogonal states.

        Args:
            other: The other state vector to compute fidelity with.

        Returns:
            Fidelity value in [0, 1].

        Raises:
            ValueError: If the other state vector has different dimension.
        """
        ip = self.inner_product(other)
        return float(abs(ip) ** 2)

    def copy(self) -> StateVector:
        """Return a deep copy of this state vector.

        Returns:
            A new StateVector instance with identical amplitudes.
        """
        return StateVector(
            num_qubits=self._num_qubits,
            amplitudes=copy.deepcopy(self._amplitudes),
        )

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"StateVector(num_qubits={self._num_qubits}, "
            f"amplitudes={self._amplitudes!r})"
        )

    def __str__(self) -> str:
        """Mathematical string representation of the quantum state.

        Formats as a sum of basis states with their amplitudes,
        e.g., "(0.707+0j)|00> + (0.707+0j)|11>".
        """
        parts: list[str] = []
        for idx in range(self._dim):
            amp = self._amplitudes[idx]
            if abs(amp) < 1e-12:
                continue
            label = format(idx, f"0{self._num_qubits}b")
            parts.append(f"({amp})|{label}>")

        return " + ".join(parts) if parts else "0"
