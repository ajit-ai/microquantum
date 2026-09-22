"""Tensor and subsystem utilities: Kronecker products, reshaping,
qubit permutation, subsystem extraction and partial trace.

This layer keeps multi-qubit mathematics explicit while isolating raw
NumPy details from higher-level Core APIs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ._model import expand_operator, tensor

__all__ = [
    "tensor",
    "expand_operator",
    "kron",
    "kron_all",
    "reshape_state",
    "permute_qubits",
    "partial_trace_matrix",
    "subsystem_probabilities",
]


def kron(a: Any, b: Any) -> NDArray[np.complex128]:
    """Kronecker product ``a ⊗ b``."""
    left = np.asarray(a, dtype=np.complex128)
    right = np.asarray(b, dtype=np.complex128)
    return np.asarray(np.kron(left, right), dtype=np.complex128)


def kron_all(operators: Sequence[Any]) -> NDArray[np.complex128]:
    """Kronecker product of a sequence (left-to-right)."""
    if not operators:
        raise ValueError("kron_all requires at least one operator")
    acc = np.asarray(operators[0], dtype=np.complex128)
    for op in operators[1:]:
        acc = np.kron(acc, np.asarray(op, dtype=np.complex128))
    return acc.astype(np.complex128)


def reshape_state(
    amplitudes: Any, num_qubits: int
) -> NDArray[np.complex128]:
    """Reshape a flat amplitude vector to per-qubit tensor form."""
    vec = np.asarray(amplitudes, dtype=np.complex128)
    if vec.shape != (2**num_qubits,):
        raise ValueError(f"Expected shape ({2**num_qubits},), got {vec.shape}")
    return vec.reshape((2,) * num_qubits)


def permute_qubits(
    amplitudes: Any, permutation: Sequence[int]
) -> NDArray[np.complex128]:
    """Permute qubits of a state vector.

    ``permutation[new_position] = old_position`` with qubit 0 leftmost.
    """
    vec = np.asarray(amplitudes, dtype=np.complex128)
    n = len(permutation)
    if vec.shape != (2**n,):
        raise ValueError(f"State dim {vec.shape} incompatible with permutation of {n} qubits")
    if sorted(permutation) != list(range(n)):
        raise ValueError(f"Invalid permutation {list(permutation)}")
    tensor = vec.reshape((2,) * n)
    # np.transpose permutes axes; axis i of the result comes from permutation[i].
    tensor = np.transpose(tensor, list(permutation))
    return tensor.reshape(-1).astype(np.complex128)


def partial_trace_matrix(
    rho: Any, keep: Sequence[int], num_qubits: int
) -> NDArray[np.complex128]:
    """Partial trace of an explicit density matrix.

    Args:
        rho: ``2^n × 2^n`` density matrix.
        keep: Qubit indices to keep (0 = leftmost).
        num_qubits: Total qubit count.
    """
    mat = np.asarray(rho, dtype=np.complex128)
    dim = 2**num_qubits
    if mat.shape != (dim, dim):
        raise ValueError(f"Expected shape ({dim}, {dim}), got {mat.shape}")
    keep_list = list(keep)
    if any(not 0 <= q < num_qubits for q in keep_list):
        raise ValueError("keep indices out of range")
    if len(set(keep_list)) != len(keep_list):
        raise ValueError("keep indices must be unique")
    if not keep_list:
        raise ValueError("keep must be non-empty")
    traced = [q for q in range(num_qubits) if q not in keep_list]
    tensor = mat.reshape([2] * (2 * num_qubits))
    perm = keep_list + traced + [num_qubits + q for q in keep_list] + [num_qubits + q for q in traced]
    tensor = np.transpose(tensor, perm)
    d_keep = 2 ** len(keep_list)
    d_trace = 2 ** len(traced)
    tensor = tensor.reshape(d_keep, d_trace, d_keep, d_trace)
    reduced: NDArray[np.complex128] = np.asarray(np.einsum("ikjk->ij", tensor), dtype=np.complex128)
    return reduced


def subsystem_probabilities(amplitudes: Any, qubits: Sequence[int], num_qubits: int) -> dict[str, float]:
    """Marginal probabilities for a subset of qubits."""
    vec = np.asarray(amplitudes, dtype=np.complex128)
    if vec.shape != (2**num_qubits,):
        raise ValueError("State dimension does not match num_qubits")
    qlist = list(qubits)
    if any(not 0 <= q < num_qubits for q in qlist):
        raise ValueError("qubit indices out of range")
    probs = np.abs(vec) ** 2
    marginal: dict[str, float] = {}
    for idx, p in enumerate(probs):
        bits = format(idx, f"0{num_qubits}b")
        key = "".join(bits[q] for q in qlist)
        marginal[key] = marginal.get(key, 0.0) + float(p)
    return marginal
