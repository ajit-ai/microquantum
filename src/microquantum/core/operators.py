"""Quantum operator (gate matrix) algebra and standard gate library."""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from .state import StateVector


class Operator:
    """A quantum operator represented by a unitary matrix.

    Encapsulates a 2^k x 2^k complex matrix and provides algebraic
    operations (composition, scalar multiplication, addition) as well as
    a library of standard single- and multi-qubit gate constructors.

    Attributes:
        num_qubits: Number of qubits this operator acts on.
        matrix: The unitary matrix in complex128 dtype.
        is_unitary: Whether U^dag @ U approx I within tolerance.
        dag: Hermitian adjoint U^dagger.
    """

    _TOLERANCE: float = 1e-9

    def __init__(self, matrix: NDArray[np.complex128], name: str = "custom") -> None:
        """Initialize an Operator from a square matrix.

        Args:
            matrix: A 2D square numpy array of dimension 2^k x 2^k.
            name: Gate name identifier (e.g., "h", "x", "cnot").

        Raises:
            ValueError: If matrix is not 2D, not square, or dimension
                is not a power of two.
        """
        arr = np.asarray(matrix, dtype=np.complex128)
        if arr.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got ndim={arr.ndim}")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"Matrix must be square, got shape {arr.shape}"
            )
        dim = arr.shape[0]
        if dim == 0 or (dim & (dim - 1)) != 0:
            raise ValueError(
                f"Matrix dimension must be a power of 2, got {dim}"
            )
        self._matrix = arr
        self._dim = dim
        self._num_qubits = int(math.log2(dim))
        self._name = name.lower()

    @property
    def num_qubits(self) -> int:
        """Number of qubits this operator acts on."""
        return self._num_qubits

    @property
    def shape(self) -> tuple[int, int]:
        """The matrix shape ``(2^n, 2^n)`` of this operator."""
        return self._matrix.shape

    @property
    def name(self) -> str:
        """Gate name identifier."""
        return self._name

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """The operator matrix."""
        return self._matrix

    @property
    def is_unitary(self) -> bool:
        """Check if the operator is unitary (U^dag @ U ≈ I)."""
        product = self._matrix.conj().T @ self._matrix
        return bool(np.allclose(product, np.eye(self._dim), atol=self._TOLERANCE))

    @property
    def dag(self) -> Operator:
        """Return the Hermitian adjoint U^dagger."""
        return Operator(self._matrix.conj().T)

    @property
    def is_hermitian(self) -> bool:
        """Check if the operator is Hermitian (U = U^dagger)."""
        return bool(np.allclose(self._matrix, self._matrix.conj().T, atol=self._TOLERANCE))

    def inverse(self) -> Operator:
        """Return the inverse of this operator (U^-1 = U^dagger for unitary).

        For unitary operators, inverse() returns the Hermitian adjoint.
        For non-unitary operators, it computes the matrix inverse.
        """
        if self.is_unitary:
            return self.dag
        return Operator(np.linalg.inv(self._matrix))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Operator:
        """Reconstruct an :class:`Operator` from its serialized dictionary.

        Accepts the payload produced by :func:`~microquantum.problems.eigenvalue.hamiltonian_to_dict`
        (``{"type": "Operator", "matrix": [...], ...}``) or a plain
        ``{"matrix": [...]}`` mapping.

        Args:
            data: The serialized operator dictionary.

        Returns:
            The reconstructed operator.
        """
        matrix = data.get("matrix")
        if matrix is None:
            raise ValueError(
                f"Operator.from_dict requires a 'matrix' entry, got keys {sorted(data)}"
            )
        return cls(np.asarray(matrix, dtype=np.complex128), name=data.get("name", "custom"))

    # ------------------------------------------------------------------
    # Algebraic dunder methods
    # ------------------------------------------------------------------

    def __matmul__(
        self, other: Union[Operator, StateVector]
    ) -> Union[Operator, StateVector]:
        """Compose two operators or apply an operator to a state vector.

        Args:
            other: Another Operator or a StateVector.

        Returns:
            Operator (for Operator @ Operator) or StateVector
            (for Operator @ StateVector).
        """
        if isinstance(other, Operator):
            if self._dim != other._dim:
                raise ValueError(
                    f"Dimension mismatch: {self._dim} vs {other._dim}"
                )
            return Operator(self._matrix @ other._matrix)
        if isinstance(other, StateVector):
            if self._dim != other.dim:
                raise ValueError(
                    f"Dimension mismatch: operator dim={self._dim}, "
                    f"state dim={other.dim}"
                )
            new_amps = (self._matrix @ other._amplitudes).astype(np.complex128)
            return StateVector(
                num_qubits=other.num_qubits, amplitudes=new_amps
            )
        return NotImplemented

    def __rmul__(self, scalar: complex) -> Operator:
        """Scalar multiplication: scalar * operator."""
        return Operator(np.array(scalar, dtype=np.complex128) * self._matrix)

    def __mul__(self, scalar: complex) -> Operator:
        """Scalar multiplication: operator * scalar."""
        return Operator(self._matrix * np.array(scalar, dtype=np.complex128))

    def __add__(self, other: Operator) -> Operator:
        """Matrix addition of two operators."""
        if not isinstance(other, Operator):
            return NotImplemented
        if self._dim != other._dim:
            raise ValueError(
                f"Dimension mismatch: {self._dim} vs {other._dim}"
            )
        return Operator(self._matrix + other._matrix)

    def __sub__(self, other: Operator) -> Operator:
        """Matrix subtraction of two operators."""
        if not isinstance(other, Operator):
            return NotImplemented
        if self._dim != other._dim:
            raise ValueError(
                f"Dimension mismatch: {self._dim} vs {other._dim}"
            )
        return Operator(self._matrix - other._matrix)

    def __rsub__(self, other: Operator) -> Operator:
        """Right subtraction: scalar - operator (not supported) or operator - operator."""
        if not isinstance(other, Operator):
            return NotImplemented
        return other.__sub__(self)

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Operator(dim={self._dim}x{self._dim}, "
            f"num_qubits={self._num_qubits}, "
            f"is_unitary={self.is_unitary})"
        )

    def __str__(self) -> str:
        """Human-readable matrix display."""
        lines = [
            f"Operator {self._dim}x{self._dim} "
            f"({self._num_qubits} qubit{'s' if self._num_qubits != 1 else ''})",
            f"Unitary: {self.is_unitary}",
            str(self._matrix),
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Static factory methods: single-qubit gates
    # ------------------------------------------------------------------

    @staticmethod
    def I() -> Operator:  # noqa: E743  (Pauli-I is a conventional quantum gate name)
        """Pauli-I (identity) gate."""
        return Operator(np.eye(2, dtype=np.complex128), name="i")

    @staticmethod
    def X() -> Operator:
        """Pauli-X (NOT) gate."""
        return Operator(np.array([[0, 1], [1, 0]], dtype=np.complex128), name="x")

    @staticmethod
    def Y() -> Operator:
        """Pauli-Y gate."""
        return Operator(
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128), name="y"
        )

    @staticmethod
    def Z() -> Operator:
        """Pauli-Z gate."""
        return Operator(np.array([[1, 0], [0, -1]], dtype=np.complex128), name="z")

    @staticmethod
    def H() -> Operator:
        """Hadamard gate."""
        return Operator(
            np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2), name="h"
        )

    @staticmethod
    def S() -> Operator:
        """Phase gate (S gate, sqrt of Z)."""
        return Operator(np.array([[1, 0], [0, 1j]], dtype=np.complex128), name="s")

    @staticmethod
    def Sdg() -> Operator:
        """S-dagger (S†) gate, inverse of the S gate."""
        return Operator(
            np.array([[1, 0], [0, -1j]], dtype=np.complex128), name="sdg"
        )

    @staticmethod
    def T() -> Operator:
        """T gate (pi/8 gate, sqrt of S)."""
        return Operator(
            np.array(
                [[1, 0], [0, np.exp(1j * np.pi / 4)]],
                dtype=np.complex128,
            ), name="t"
        )

    @staticmethod
    def Tdg() -> Operator:
        """T-dagger (T†) gate, inverse of the T gate."""
        return Operator(
            np.array(
                [[1, 0], [0, np.exp(-1j * np.pi / 4)]],
                dtype=np.complex128,
            ), name="tdg"
        )

    @staticmethod
    def Rx(theta: float) -> Operator:
        """Rotation gate around X-axis by angle theta (radians)."""
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return Operator(
            np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128), name="rx"
        )

    @staticmethod
    def Ry(theta: float) -> Operator:
        """Rotation gate around Y-axis by angle theta (radians)."""
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return Operator(
            np.array([[c, -s], [s, c]], dtype=np.complex128), name="ry"
        )

    @staticmethod
    def Rz(theta: float) -> Operator:
        """Rotation gate around Z-axis by angle theta (radians)."""
        return Operator(
            np.array(
                [[np.exp(-1j * theta / 2), 0],
                 [0, np.exp(1j * theta / 2)]],
                dtype=np.complex128,
            ), name="rz"
        )

    # ------------------------------------------------------------------
    # Static factory methods: multi-qubit gates
    # ------------------------------------------------------------------

    @staticmethod
    def CNOT() -> Operator:
        """Controlled-NOT (CX) gate on 2 qubits."""
        return Operator(
            np.array(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]],
                dtype=np.complex128,
            ), name="cnot"
        )

    @staticmethod
    def CZ() -> Operator:
        """Controlled-Z gate on 2 qubits."""
        return Operator(
            np.array(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, -1]],
                dtype=np.complex128,
            ), name="cz"
        )

    @staticmethod
    def SWAP() -> Operator:
        """SWAP gate on 2 qubits."""
        return Operator(
            np.array(
                [[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]],
                dtype=np.complex128,
            ), name="swap"
        )
