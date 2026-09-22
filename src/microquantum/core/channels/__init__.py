"""Quantum channels as first-class mathematical objects.

Supports :class:`QuantumChannel` / :class:`KrausChannel` with validation,
application to density matrices, composition, tensor products and
trace-preservation checks, plus standard channels: bit flip, phase flip,
bit-phase flip, depolarizing, amplitude damping and phase damping.
Channel mathematics lives in Core; higher-level noise models remain in
``backends``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "QuantumChannel",
    "KrausChannel",
    "bit_flip",
    "phase_flip",
    "bit_phase_flip",
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
]


def _as_matrix(data: Any) -> NDArray[np.complex128]:
    return np.asarray(data, dtype=np.complex128)


class QuantumChannel(ABC):
    """Abstract quantum channel (CPTP map)."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits the channel acts on."""

    @property
    @abstractmethod
    def kraus_operators(self) -> list[NDArray[np.complex128]]:
        """Kraus operators ``{Kᵢ}`` with ``Σ Kᵢ†Kᵢ = I``."""

    def is_trace_preserving(self, atol: float = 1e-9) -> bool:
        """Check the trace-preservation condition."""
        ops = self.kraus_operators
        dim = ops[0].shape[0]
        total = np.zeros((dim, dim), dtype=np.complex128)
        for k in ops:
            total = total + k.conj().T @ k
        return bool(np.allclose(total, np.eye(dim), atol=atol))

    def validate(self) -> None:
        """Raise ValueError when the channel is not trace-preserving."""
        if not self.is_trace_preserving():
            raise ValueError("Channel is not trace-preserving (ΣK†K ≠ I)")

    def apply(self, rho: Any) -> NDArray[np.complex128]:
        """Apply the channel to a density matrix."""
        mat = _as_matrix(rho)
        dim = 2**self.num_qubits
        if mat.shape != (dim, dim):
            raise ValueError(f"State shape {mat.shape} incompatible with {self.num_qubits} qubits")
        self.validate()
        out = np.zeros_like(mat)
        for k in self.kraus_operators:
            out = out + k @ mat @ k.conj().T
        return out

    def compose(self, other: QuantumChannel) -> KrausChannel:
        """Compose channels: ``self ∘ other`` (apply other first)."""
        if self.num_qubits != other.num_qubits:
            raise ValueError("Channels act on different qubit counts")
        ops = [a @ b for a in self.kraus_operators for b in other.kraus_operators]
        return KrausChannel(ops, num_qubits=self.num_qubits)

    def tensor(self, other: QuantumChannel) -> KrausChannel:
        """Tensor product of two channels."""
        ops = [np.kron(a, b) for a in self.kraus_operators for b in other.kraus_operators]
        return KrausChannel(ops, num_qubits=self.num_qubits + other.num_qubits)

    def __call__(self, rho: Any) -> NDArray[np.complex128]:
        return self.apply(rho)


class KrausChannel(QuantumChannel):
    """Channel defined by an explicit Kraus set."""

    def __init__(self, kraus: Sequence[Any], num_qubits: int | None = None) -> None:
        ops = [_as_matrix(k) for k in kraus]
        if not ops:
            raise ValueError("KrausChannel requires at least one Kraus operator")
        dim = ops[0].shape[0]
        for k in ops:
            if k.shape != (dim, dim):
                raise ValueError("All Kraus operators must be square with equal dimension")
        if dim == 0 or (dim & (dim - 1)) != 0:
            raise ValueError(f"Kraus dimension must be a power of 2, got {dim}")
        inferred = int(math.log2(dim))
        if num_qubits is not None and num_qubits != inferred:
            raise ValueError(f"num_qubits {num_qubits} inconsistent with Kraus dim {dim}")
        self._ops = ops
        self._num_qubits = inferred
        self.validate()

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def kraus_operators(self) -> list[NDArray[np.complex128]]:
        return [k.copy() for k in self._ops]

    @property
    def num_kraus(self) -> int:
        """Number of Kraus operators."""
        return len(self._ops)


def _single(pairs: Sequence[tuple[Any, float]]) -> KrausChannel:
    ops = [np.sqrt(p) * _as_matrix(m) for m, p in pairs]
    return KrausChannel(ops, num_qubits=1)


def bit_flip(p: float) -> KrausChannel:
    """Bit-flip channel with probability ``p``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    return _single([(np.eye(2), 1 - p), (x, p)])


def phase_flip(p: float) -> KrausChannel:
    """Phase-flip channel with probability ``p``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return _single([(np.eye(2), 1 - p), (z, p)])


def bit_phase_flip(p: float) -> KrausChannel:
    """Bit-phase (Y) flip channel with probability ``p``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    return _single([(np.eye(2), 1 - p), (y, p)])


def depolarizing(p: float) -> KrausChannel:
    """Single-qubit depolarizing channel with strength ``p``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return _single([(np.eye(2), 1 - 3 * p / 4), (x, p / 4), (y, p / 4), (z, p / 4)])


def amplitude_damping(gamma: float) -> KrausChannel:
    """Amplitude-damping channel with decay probability ``gamma``."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    k0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return KrausChannel([k0, k1], num_qubits=1)


def phase_damping(gamma: float) -> KrausChannel:
    """Phase-damping channel with scattering probability ``gamma``."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    k0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0, 0], [0, math.sqrt(gamma)]], dtype=np.complex128)
    return KrausChannel([k0, k1], num_qubits=1)
