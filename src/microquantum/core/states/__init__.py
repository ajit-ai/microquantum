"""Quantum states: State base, StateVector and DensityMatrix workflows.

The canonical dense implementations live in ``microquantum.core.state``
(:class:`StateVector`) and ``microquantum.core.density_matrix``
(:class:`DensityMatrix`).  This package unifies them behind the
:class:`State` interface and adds tensor products, subsystem extraction,
partial trace, probabilities, fidelity, purity and measurement helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from microquantum.core.density_matrix import DensityMatrix
from microquantum.core.state import StateVector

__all__ = [
    "State",
    "StateVector",
    "DensityMatrix",
    "tensor_states",
    "partial_trace",
    "state_fidelity",
    "state_purity",
    "probabilities",
    "is_normalized",
]


class State(ABC):
    """Abstract quantum state (pure or mixed)."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Hilbert-space dimension."""

    @abstractmethod
    def to_statevector(self) -> StateVector:
        """Convert to a :class:`StateVector` (pure states only)."""

    @abstractmethod
    def to_density_matrix(self) -> DensityMatrix:
        """Convert to a :class:`DensityMatrix`."""

    @abstractmethod
    def is_pure(self) -> bool:
        """True for pure states."""

    def purity(self) -> float:
        """Purity ``Tr[ρ²]`` (1.0 for pure states)."""
        rho = self.to_density_matrix()
        mat = np.asarray(getattr(rho, "matrix", getattr(rho, "rho", None)), dtype=np.complex128)
        return float(np.real(np.trace(mat @ mat)))

    def probabilities(self) -> NDArray[np.float64]:
        """Computational-basis outcome probabilities."""
        rho = self.to_density_matrix()
        mat = np.asarray(getattr(rho, "matrix", getattr(rho, "rho", None)), dtype=np.complex128)
        return np.real(np.diag(mat)).astype(np.float64)


def _vector_amplitudes(state: Any) -> NDArray[np.complex128]:
    if isinstance(state, StateVector):
        return np.asarray(state.amplitudes, dtype=np.complex128)
    if isinstance(state, np.ndarray) and state.ndim == 1:
        return np.asarray(state, dtype=np.complex128)
    amps = getattr(state, "amplitudes", None)
    if amps is not None:
        return np.asarray(amps, dtype=np.complex128)
    raise TypeError(f"Cannot interpret {type(state).__name__} as a state vector")


def _density_data(state: Any) -> tuple[NDArray[np.complex128], int]:
    if isinstance(state, DensityMatrix):
        mat = getattr(state, "matrix", None)
        if mat is None:
            mat = getattr(state, "rho", None)
        if mat is None:
            # Fall back to constructing from the object itself.
            raise TypeError("DensityMatrix exposes neither .matrix nor .rho")
        mat = np.asarray(mat, dtype=np.complex128)
        return mat, int(np.log2(mat.shape[0]))
    if isinstance(state, StateVector):
        vec = np.asarray(state.amplitudes, dtype=np.complex128)
        return np.asarray(np.outer(vec, vec.conj()), dtype=np.complex128), state.num_qubits
    if isinstance(state, np.ndarray):
        arr = np.asarray(state, dtype=np.complex128)
        if arr.ndim == 1:
            return np.asarray(np.outer(arr, arr.conj()), dtype=np.complex128), int(np.log2(arr.shape[0]))
        if arr.ndim == 2:
            return arr, int(np.log2(arr.shape[0]))
    amps = getattr(state, "amplitudes", None)
    if amps is not None:
        vec = np.asarray(amps, dtype=np.complex128)
        return np.asarray(np.outer(vec, vec.conj()), dtype=np.complex128), int(np.log2(vec.shape[0]))
    for attr in ("matrix", "rho"):
        mat = getattr(state, attr, None)
        if isinstance(mat, np.ndarray):
            dense = np.asarray(mat, dtype=np.complex128)
            return dense, int(np.log2(dense.shape[0]))
    raise TypeError(f"Cannot interpret {type(state).__name__} as a quantum state")


def tensor_states(a: Any, b: Any) -> NDArray[np.complex128]:
    """Tensor product of two state vectors (``a ⊗ b``)."""
    va = _vector_amplitudes(a)
    vb = _vector_amplitudes(b)
    return np.asarray(np.kron(va, vb), dtype=np.complex128)


def partial_trace(state: Any, keep: Sequence[int]) -> NDArray[np.complex128]:
    """Partial trace of a state, keeping the listed qubit indices.

    Qubit 0 is the leftmost (big-endian) qubit.  Returns the reduced
    density matrix of ``len(keep)`` qubits.
    """
    rho, n = _density_data(state)
    keep_list = list(keep)
    if any(not 0 <= q < n for q in keep_list):
        raise ValueError(f"keep indices {keep_list} out of range for {n} qubits")
    if len(set(keep_list)) != len(keep_list):
        raise ValueError("keep indices must be unique")
    if not keep_list:
        raise ValueError("keep must list at least one qubit")
    traced = [q for q in range(n) if q not in keep_list]
    tensor = rho.reshape([2] * (2 * n))
    # Move kept row/col axes first: rows keep, rows traced, cols keep, cols traced.
    row_axes = keep_list + traced
    col_axes = [n + q for q in keep_list] + [n + q for q in traced]
    perm = row_axes + col_axes
    tensor = np.transpose(tensor, perm)
    d_keep = 2 ** len(keep_list)
    d_trace = 2 ** len(traced)
    tensor = tensor.reshape(d_keep, d_trace, d_keep, d_trace)
    reduced = np.einsum("ikjk->ij", tensor)
    return np.asarray(reduced, dtype=np.complex128)


def state_fidelity(a: Any, b: Any) -> float:
    """Fidelity between two states (pure or mixed).

    Pure–pure: ``|⟨a|b⟩|²``.  General: ``(Tr|√ρ√σ|)²``.
    """
    try:
        va = _vector_amplitudes(a)
        vb = _vector_amplitudes(b)
        if va.shape != vb.shape:
            raise ValueError("State dimensions do not match")
        return float(abs(np.vdot(va, vb)) ** 2)
    except TypeError:
        pass
    rho, n1 = _density_data(a)
    sigma, n2 = _density_data(b)
    if rho.shape != sigma.shape:
        raise ValueError("State dimensions do not match")
    # Fuchs–van de Graaf: use eigenvalue formula for Hermitian PSD inputs.
    sqrt_rho = _matrix_sqrt(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    eigs = np.linalg.eigvalsh(inner)
    eigs = np.clip(eigs, 0.0, None)
    return float(float(np.sum(np.sqrt(eigs))) ** 2)


def _matrix_sqrt(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
    vals, vecs = np.linalg.eigh((mat + mat.conj().T) / 2)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def state_purity(state: Any) -> float:
    """Purity ``Tr[ρ²]`` in ``[1/d, 1]``."""
    rho, _ = _density_data(state)
    return float(np.real(np.trace(rho @ rho)))


def probabilities(state: Any) -> NDArray[np.float64]:
    """Computational-basis probabilities for a state."""
    try:
        vec = _vector_amplitudes(state)
        return (np.abs(vec) ** 2).astype(np.float64)
    except TypeError:
        pass
    rho, _ = _density_data(state)
    return np.real(np.diag(rho)).astype(np.float64)


def is_normalized(state: Any, atol: float = 1e-9) -> bool:
    """Check normalization (unit vector or unit-trace density matrix)."""
    try:
        vec = _vector_amplitudes(state)
        return bool(abs(float(np.vdot(vec, vec).real) - 1.0) <= atol)
    except TypeError:
        pass
    rho, _ = _density_data(state)
    return bool(abs(float(np.trace(rho).real) - 1.0) <= atol)
