"""General quantum operators: linear, unitary, Hermitian and projectors.

The canonical dense :class:`Operator` lives in
``microquantum.core.operators``.  This package re-exports it and adds
the distinct Core abstractions: :class:`LinearOperator`,
:class:`UnitaryOperator`, :class:`HermitianOperator` and
:class:`Projector`, with composition, tensor products, adjoints and
dimension validation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._model import Operator

__all__ = [
    "Operator",
    "LinearOperator",
    "UnitaryOperator",
    "HermitianOperator",
    "Projector",
    "compose",
    "tensor_product",
]


def _check_square_dim(matrix: NDArray[np.complex128]) -> int:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Operator matrix must be square 2D, got shape {arr.shape}")
    dim: int = arr.shape[0]
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Operator dimension must be a power of 2, got {dim}")
    return dim


class LinearOperator:
    """A general (not necessarily unitary) linear operator."""

    def __init__(self, matrix: Any) -> None:
        arr = np.asarray(matrix, dtype=np.complex128)
        _check_square_dim(arr)
        self._matrix = arr

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """Dense matrix representation (copy)."""
        return self._matrix.copy()

    @property
    def dim(self) -> int:
        """Hilbert-space dimension."""
        size: int = self._matrix.shape[0]
        return size

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return int(math.log2(self.dim))

    def adjoint(self) -> LinearOperator:
        """Conjugate transpose."""
        return LinearOperator(self._matrix.conj().T)

    def compose(self, other: LinearOperator) -> LinearOperator:
        """Matrix composition ``self @ other`` with dimension check."""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return LinearOperator(self._matrix @ other._matrix)

    def tensor(self, other: LinearOperator) -> LinearOperator:
        """Kronecker product ``self ⊗ other``."""
        return LinearOperator(np.kron(self._matrix, other._matrix))

    def __matmul__(self, other: LinearOperator) -> LinearOperator:
        return self.compose(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LinearOperator):
            return bool(np.allclose(self._matrix, other._matrix))
        return NotImplemented

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"


class UnitaryOperator(LinearOperator):
    """A unitary operator (``U†U = I``) with an inverse."""

    _TOL = 1e-9

    def __init__(self, matrix: Any) -> None:
        super().__init__(matrix)
        ident = np.eye(self.dim, dtype=np.complex128)
        if not np.allclose(self._matrix.conj().T @ self._matrix, ident, atol=self._TOL):
            raise ValueError("Matrix is not unitary within tolerance 1e-9")

    @property
    def is_unitary(self) -> bool:
        """Always True (validated at construction)."""
        return True

    def inverse(self) -> UnitaryOperator:
        """Adjoint (= inverse) unitary."""
        return UnitaryOperator(self._matrix.conj().T)

    def controlled(self) -> UnitaryOperator:
        """Single-control extension of this unitary."""
        dim = self.dim * 2
        mat = np.eye(dim, dtype=np.complex128)
        mat[dim - self.dim :, dim - self.dim :] = self._matrix
        return UnitaryOperator(mat)

    def to_operator(self) -> Operator:
        """Convert to the canonical dense :class:`Operator`."""
        return Operator(self._matrix, name="unitary")


class HermitianOperator(LinearOperator):
    """A Hermitian operator (``H† = H``), e.g. an observable matrix."""

    _TOL = 1e-9

    def __init__(self, matrix: Any) -> None:
        super().__init__(matrix)
        if not np.allclose(self._matrix, self._matrix.conj().T, atol=self._TOL):
            raise ValueError("Matrix is not Hermitian within tolerance 1e-9")

    @property
    def is_hermitian(self) -> bool:
        """Always True (validated at construction)."""
        return True

    def eigenvalues(self) -> NDArray[np.float64]:
        """Real eigenvalues in ascending order."""
        return np.asarray(np.linalg.eigvalsh(self._matrix), dtype=np.float64)


class Projector(HermitianOperator):
    """An orthogonal projector (``P² = P``, ``P† = P``)."""

    _TOL = 1e-9

    def __init__(self, matrix: Any) -> None:
        super().__init__(matrix)
        if not np.allclose(self._matrix @ self._matrix, self._matrix, atol=self._TOL):
            raise ValueError("Matrix is not idempotent (P² ≠ P)")
        self._rank = int(round(float(np.trace(self._matrix).real)))

    @property
    def rank(self) -> int:
        """Rank (= trace) of the projector."""
        return self._rank

    @classmethod
    def zero_state(cls, num_qubits: int) -> Projector:
        """Projector onto ``|0...0⟩``."""
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        dim = 2**num_qubits
        mat = np.zeros((dim, dim), dtype=np.complex128)
        mat[0, 0] = 1.0
        return cls(mat)

    @classmethod
    def computational_basis(cls, index: int, num_qubits: int) -> Projector:
        """Projector onto computational basis state ``|index⟩``."""
        dim = 2**num_qubits
        if not 0 <= index < dim:
            raise ValueError(f"index {index} out of range for {num_qubits} qubits")
        mat = np.zeros((dim, dim), dtype=np.complex128)
        mat[index, index] = 1.0
        return cls(mat)


def compose(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Compose two operators (``a @ b``)."""
    return a.compose(b)


def tensor_product(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Kronecker product of two operators."""
    return a.tensor(b)
