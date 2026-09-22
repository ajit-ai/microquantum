"""First-class observable abstraction.

Observables are Hermitian quantities with expectation/variance
semantics.  Common cases (single Pauli, Pauli sums, dense Hermitian
matrices) avoid unnecessary dense-matrix construction: Pauli paths
evaluate against state amplitudes directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Observable",
    "PauliObservable",
    "MatrixObservable",
    "SumObservable",
    "observable_from_pauli",
    "observable_from_matrix",
]

_StateLike = Any


class Observable(ABC):
    """Abstract Hermitian observable."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits the observable acts on."""

    @abstractmethod
    def to_matrix(self) -> NDArray[np.complex128]:
        """Dense Hermitian matrix."""

    @abstractmethod
    def expectation(self, state: _StateLike) -> float:
        """Expectation value ``⟨ψ|O|ψ⟩`` (or ``Tr[ρO]``)."""

    def variance(self, state: _StateLike) -> float:
        """Variance ``⟨O²⟩ − ⟨O⟩²`` (default via dense matrices)."""
        mat = self.to_matrix()
        exp = self.expectation(state)
        vec = _as_state_vector(state)
        if vec is not None:
            o2 = float(np.real(np.vdot(vec, mat @ mat @ vec)))
            return max(0.0, o2 - exp * exp)
        rho = _as_density_matrix(state)
        if rho is not None:
            o2 = float(np.real(np.trace(rho @ mat @ mat)))
            return max(0.0, o2 - exp * exp)
        raise TypeError(f"Unsupported state type {type(state).__name__}")

    @abstractmethod
    def commutes_with(self, other: Observable) -> bool:
        """Return True when two observables commute."""

    def __add__(self, other: Observable) -> SumObservable:
        if not isinstance(other, Observable):
            raise TypeError("Can only add Observable to Observable")
        return SumObservable([self, other], [1.0, 1.0])

    def __mul__(self, scalar: complex) -> SumObservable:
        return SumObservable([self], [complex(scalar)])

    def __rmul__(self, scalar: complex) -> SumObservable:
        return SumObservable([self], [complex(scalar)])


def _as_state_vector(state: _StateLike) -> NDArray[np.complex128] | None:
    if isinstance(state, np.ndarray) and state.ndim == 1:
        return np.asarray(state, dtype=np.complex128)
    amps = getattr(state, "amplitudes", None)
    if amps is not None:
        return np.asarray(amps, dtype=np.complex128)
    vec = getattr(state, "state_vector", None)
    if isinstance(vec, np.ndarray):
        return np.asarray(vec, dtype=np.complex128)
    return None


def _as_density_matrix(state: _StateLike) -> NDArray[np.complex128] | None:
    mat = getattr(state, "density_matrix", None)
    if isinstance(mat, np.ndarray) and mat.ndim == 2:
        return np.asarray(mat, dtype=np.complex128)
    if isinstance(state, np.ndarray) and state.ndim == 2:
        return np.asarray(state, dtype=np.complex128)
    rho = getattr(state, "rho", None)
    if isinstance(rho, np.ndarray):
        return np.asarray(rho, dtype=np.complex128)
    return None


def _pauli_expectation(label: str, amplitudes: NDArray[np.complex128]) -> complex:
    """Evaluate ⟨ψ|P|ψ⟩ for a Pauli label without dense matrices."""
    n = len(label)
    dim = 2**n
    if amplitudes.shape != (dim,):
        raise ValueError(f"State dim {amplitudes.shape} incompatible with {n} qubits")
    total = complex(0)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - k)) & 1 for k in range(n)]
        jdx = 0
        phase = complex(1)
        for k, ch in enumerate(label):
            b = bits[k]
            if ch == "I":
                j = b
            elif ch == "X":
                j = 1 - b
            elif ch == "Z":
                j = b
                if b == 1:
                    phase *= -1
            elif ch == "Y":
                j = 1 - b
                phase *= 1j if b == 0 else -1j
            else:
                raise ValueError(f"Invalid Pauli char {ch!r}")
            jdx = (jdx << 1) | j
        total += np.conj(amplitudes[idx]) * phase * amplitudes[jdx]
    return total


class PauliObservable(Observable):
    """Observable wrapping a single Pauli string (plus weight)."""

    def __init__(self, label: str, coefficient: complex = 1.0) -> None:
        from microquantum.core.pauli import PauliString  # noqa: PLC0415

        self._pauli = PauliString(label, coefficient=1.0)
        if not np.isreal(complex(coefficient)):
            raise ValueError("PauliObservable coefficient must be real (Hermiticity)")
        self._coefficient = float(complex(coefficient).real)

    @property
    def num_qubits(self) -> int:
        return self._pauli.num_qubits

    @property
    def label(self) -> str:
        """Pauli label."""
        return self._pauli.label

    @property
    def coefficient(self) -> float:
        """Real weight."""
        return self._coefficient

    def to_matrix(self) -> NDArray[np.complex128]:
        return (self._coefficient * self._pauli.to_operator().matrix).astype(np.complex128)

    def expectation(self, state: _StateLike) -> float:
        vec = _as_state_vector(state)
        if vec is not None:
            val = _pauli_expectation(self._pauli.label, vec)
            return float(np.real(self._coefficient * val))
        rho = _as_density_matrix(state)
        if rho is not None:
            return float(np.real(np.trace(rho @ self.to_matrix())))
        raise TypeError(f"Unsupported state type {type(state).__name__}")

    def commutes_with(self, other: Observable) -> bool:
        from microquantum.core.pauli import commutes as _commutes  # noqa: PLC0415

        if isinstance(other, PauliObservable):
            from microquantum.core.pauli import PauliString  # noqa: PLC0415

            return _commutes(PauliString(self.label), PauliString(other.label))
        return bool(np.allclose(self.to_matrix() @ other.to_matrix(), other.to_matrix() @ self.to_matrix()))


class MatrixObservable(Observable):
    """Observable wrapping an explicit Hermitian matrix."""

    def __init__(self, matrix: Any) -> None:
        from microquantum.core.operators import HermitianOperator  # noqa: PLC0415

        self._op = HermitianOperator(matrix)

    @property
    def num_qubits(self) -> int:
        return self._op.num_qubits

    def to_matrix(self) -> NDArray[np.complex128]:
        return self._op.matrix

    def expectation(self, state: _StateLike) -> float:
        mat = self._op.matrix
        vec = _as_state_vector(state)
        if vec is not None:
            if vec.shape != (mat.shape[0],):
                raise ValueError("State dimension does not match observable")
            return float(np.real(np.vdot(vec, mat @ vec)))
        rho = _as_density_matrix(state)
        if rho is not None:
            if rho.shape != mat.shape:
                raise ValueError("Density-matrix dimension does not match observable")
            return float(np.real(np.trace(rho @ mat)))
        raise TypeError(f"Unsupported state type {type(state).__name__}")

    def commutes_with(self, other: Observable) -> bool:
        a = self.to_matrix()
        b = other.to_matrix()
        if a.shape != b.shape:
            raise ValueError("Observables act on different dimensions")
        return bool(np.allclose(a @ b, b @ a))


class SumObservable(Observable):
    """Weighted sum of observables."""

    def __init__(
        self, terms: Sequence[Observable], weights: Sequence[complex | float] | None = None
    ) -> None:
        if not terms:
            raise ValueError("SumObservable requires at least one term")
        n = terms[0].num_qubits
        for t in terms:
            if not isinstance(t, Observable):
                raise TypeError("SumObservable terms must be Observable instances")
            if t.num_qubits != n:
                raise ValueError("All SumObservable terms must act on the same qubit count")
        w = [1.0] * len(terms) if weights is None else [complex(x) for x in weights]
        if len(w) != len(terms):
            raise ValueError("weights length must match terms length")
        if any(not np.isreal(x) for x in w):
            raise ValueError("SumObservable weights must be real (Hermiticity)")
        self._terms = list(terms)
        self._weights = [float(x.real) for x in w]

    @property
    def num_qubits(self) -> int:
        return self._terms[0].num_qubits

    @property
    def terms(self) -> list[Observable]:
        """Observable terms (copies of the list)."""
        return list(self._terms)

    @property
    def weights(self) -> list[float]:
        """Real weights."""
        return list(self._weights)

    def to_matrix(self) -> NDArray[np.complex128]:
        acc = self._weights[0] * self._terms[0].to_matrix()
        for w, t in zip(self._weights[1:], self._terms[1:], strict=False):
            acc = acc + w * t.to_matrix()
        return np.asarray(acc, dtype=np.complex128)

    def expectation(self, state: _StateLike) -> float:
        return float(sum(w * t.expectation(state) for w, t in zip(self._weights, self._terms, strict=False)))

    def commutes_with(self, other: Observable) -> bool:
        a = self.to_matrix()
        b = other.to_matrix()
        if a.shape != b.shape:
            raise ValueError("Observables act on different dimensions")
        return bool(np.allclose(a @ b, b @ a))


def observable_from_pauli(label: str, coefficient: complex = 1.0) -> PauliObservable:
    """Build a :class:`PauliObservable` from a label."""
    return PauliObservable(label, coefficient)


def observable_from_matrix(matrix: Any) -> MatrixObservable:
    """Build a :class:`MatrixObservable` from a Hermitian matrix."""
    return MatrixObservable(matrix)


def observable_from_pauli_sum(pauli_sum: Any) -> SumObservable:
    """Build a :class:`SumObservable` from a :class:`PauliSum`.

    Accepts the canonical ``microquantum.core.pauli.PauliSum`` (duck-typed
    via ``terms``/``__iter__``) without importing domain packages.
    """
    terms: list[Observable] = []
    weights: list[complex] = []
    iterable = getattr(pauli_sum, "terms", pauli_sum)
    if callable(iterable):
        iterable = iterable()
    if iterable is None:
        raise TypeError("PauliSum exposes neither .terms nor iteration")
    for term in iterable:
        label = getattr(term, "label", None)
        coef = complex(getattr(term, "coefficient", 1.0))
        if label is None:
            raise TypeError("PauliSum terms must expose .label and .coefficient")
        if abs(coef.imag) > 1e-12:
            raise ValueError("PauliSum coefficients must be real for observables")
        terms.append(PauliObservable(str(label)))
        weights.append(float(coef.real))
    if not terms:
        raise ValueError("PauliSum has no terms")
    return SumObservable(terms, weights)
