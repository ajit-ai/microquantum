"""Quantum-information utility layer: fidelity, distances, entropies.

All helpers validate dimensions and mathematical domains.  Inputs may be
state vectors (1D), density matrices (2D), classical distributions (1D
real), or the canonical ``StateVector``/``DensityMatrix`` objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "fidelity",
    "trace_distance",
    "purity",
    "von_neumann_entropy",
    "shannon_entropy",
    "relative_entropy",
    "mutual_information",
    "state_overlap",
    "entanglement_entropy",
    "concurrence",
]


def _to_density(state: Any) -> NDArray[np.complex128]:
    if isinstance(state, np.ndarray):
        arr = np.asarray(state, dtype=np.complex128)
        if arr.ndim == 1:
            return np.outer(arr, arr.conj())
        if arr.ndim == 2:
            return arr
        raise ValueError("State array must be 1D or 2D")
    for attr in ("matrix", "rho", "density_matrix"):
        mat = getattr(state, attr, None)
        if isinstance(mat, np.ndarray) and mat.ndim == 2:
            return np.asarray(mat, dtype=np.complex128)
    amps = getattr(state, "amplitudes", None)
    if amps is not None:
        vec = np.asarray(amps, dtype=np.complex128)
        return np.outer(vec, vec.conj())
    raise TypeError(f"Cannot interpret {type(state).__name__} as a quantum state")


def _to_vector(state: Any) -> NDArray[np.complex128] | None:
    if isinstance(state, np.ndarray) and state.ndim == 1:
        return np.asarray(state, dtype=np.complex128)
    amps = getattr(state, "amplitudes", None)
    if amps is not None:
        return np.asarray(amps, dtype=np.complex128)
    return None


def fidelity(a: Any, b: Any) -> float:
    """State fidelity in ``[0, 1]``."""
    va, vb = _to_vector(a), _to_vector(b)
    if va is not None and vb is not None:
        if va.shape != vb.shape:
            raise ValueError("State dimensions do not match")
        return float(abs(np.vdot(va, vb)) ** 2)
    rho, sigma = _to_density(a), _to_density(b)
    if rho.shape != sigma.shape:
        raise ValueError("State dimensions do not match")
    vals, vecs = np.linalg.eigh((rho + rho.conj().T) / 2)
    vals = np.clip(vals, 0.0, None)
    sqrt_rho = (vecs * np.sqrt(vals)) @ vecs.conj().T
    eigs = np.linalg.eigvalsh(sqrt_rho @ sigma @ sqrt_rho)
    return float(float(np.sum(np.sqrt(np.clip(eigs, 0.0, None)))) ** 2)


def trace_distance(a: Any, b: Any) -> float:
    """Trace distance ``½‖ρ − σ‖₁`` in ``[0, 1]``."""
    rho, sigma = _to_density(a), _to_density(b)
    if rho.shape != sigma.shape:
        raise ValueError("State dimensions do not match")
    eigs = np.linalg.eigvalsh((rho - sigma + (rho - sigma).conj().T) / 2)
    return float(0.5 * np.sum(np.abs(eigs)))


def purity(state: Any) -> float:
    """Purity ``Tr[ρ²]``."""
    rho = _to_density(state)
    return float(np.real(np.trace(rho @ rho)))


def _eig_probs(rho: NDArray[np.complex128]) -> NDArray[np.float64]:
    eigs = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    eigs = np.clip(eigs, 0.0, None)
    total = eigs.sum()
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Density matrix trace must be 1, got {total}")
    return eigs.astype(np.float64)


def von_neumann_entropy(state: Any, base: float = 2.0) -> float:
    """Von Neumann entropy ``S(ρ) = −Tr[ρ log ρ]`` (bits by default)."""
    if base <= 0 or base == 1.0:
        raise ValueError("log base must be positive and != 1")
    probs = _eig_probs(_to_density(state))
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log(nz) / np.log(base)))


def shannon_entropy(distribution: Any, base: float = 2.0) -> float:
    """Shannon entropy of a classical probability distribution."""
    if base <= 0 or base == 1.0:
        raise ValueError("log base must be positive and != 1")
    p = np.asarray(distribution, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("distribution must be a non-empty 1D array")
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative")
    total = p.sum()
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1, got {total}")
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz) / np.log(base)))


def relative_entropy(a: Any, b: Any, base: float = 2.0) -> float:
    """Quantum relative entropy ``S(ρ‖σ)`` (bits by default)."""
    if base <= 0 or base == 1.0:
        raise ValueError("log base must be positive and != 1")
    rho, sigma = _to_density(a), _to_density(b)
    if rho.shape != sigma.shape:
        raise ValueError("State dimensions do not match")
    r_vals, r_vecs = np.linalg.eigh((rho + rho.conj().T) / 2)
    s_vals, s_vecs = np.linalg.eigh((sigma + sigma.conj().T) / 2)
    log_base = float(np.log(base))
    log_r = np.zeros_like(r_vals, dtype=float)
    positive = r_vals > 0
    log_r[positive] = np.log(r_vals[positive]) / log_base
    # Support of rho must lie within support of sigma.
    for i, rv in enumerate(r_vals):
        if rv > 1e-12:
            overlap = float(np.sum(np.abs(s_vecs.conj().T @ r_vecs[:, i]) ** 2 * (s_vals <= 1e-12)))
            if overlap > 1e-9:
                return float("inf")
    inv_log_s = np.zeros_like(s_vals, dtype=float)
    s_positive = s_vals > 0
    inv_log_s[s_positive] = np.log(s_vals[s_positive]) / log_base
    # S(ρ‖σ) = Tr[ρ log ρ] − Tr[ρ log σ].
    term1 = float(np.sum(r_vals * log_r))
    # Tr[ρ log σ] via eigenbasis change.
    cross = r_vecs.conj().T @ s_vecs
    term2 = 0.0
    for i, rv in enumerate(r_vals):
        if rv > 0:
            w = np.abs(cross[i, :]) ** 2
            term2 += float(rv * np.sum(w * inv_log_s))
    return float(term1 - term2)


def mutual_information(rho_ab: Any, keep_a: list[int], num_qubits: int, base: float = 2.0) -> float:
    """Mutual information ``I(A:B) = S(A) + S(B) − S(AB)``."""
    from microquantum.core.tensor import partial_trace_matrix  # noqa: PLC0415

    mat = np.asarray(rho_ab, dtype=np.complex128)
    set_a = list(keep_a)
    set_b = [q for q in range(num_qubits) if q not in set_a]
    if not set_a or not set_b:
        raise ValueError("Both subsystems must be non-empty")
    rho_a = partial_trace_matrix(mat, set_a, num_qubits)
    rho_b = partial_trace_matrix(mat, set_b, num_qubits)
    return float(
        von_neumann_entropy(rho_a, base) + von_neumann_entropy(rho_b, base) - von_neumann_entropy(mat, base)
    )


def state_overlap(a: Any, b: Any) -> float:
    """Overlap ``|⟨a|b⟩|²`` for pure states (``Tr[ρσ]`` generally)."""
    va, vb = _to_vector(a), _to_vector(b)
    if va is not None and vb is not None:
        if va.shape != vb.shape:
            raise ValueError("State dimensions do not match")
        return float(abs(np.vdot(va, vb)) ** 2)
    rho, sigma = _to_density(a), _to_density(b)
    if rho.shape != sigma.shape:
        raise ValueError("State dimensions do not match")
    return float(np.real(np.trace(rho @ sigma)))


def entanglement_entropy(state: Any, keep: list[int], base: float = 2.0) -> float:
    """EXPERIMENTAL: entanglement entropy across a bipartition.

    Von Neumann entropy of the reduced state keeping the qubits in
    *keep* (pure overall states give the bipartite entanglement).
    The API is experimental and may gain Rényi-order support.
    """
    from microquantum.core.tensor import partial_trace_matrix  # noqa: PLC0415

    rho = _to_density(state)
    dim = rho.shape[0]
    num_qubits = int(np.log2(dim))
    if rho.shape != (dim, dim) or dim & (dim - 1) != 0:
        raise ValueError("State dimension must be a power of 2")
    reduced = partial_trace_matrix(rho, keep, num_qubits)
    return von_neumann_entropy(reduced, base)


def concurrence(state: Any) -> float:
    """EXPERIMENTAL: Wootters concurrence of a two-qubit state.

    ``max(0, √λ₁ − √λ₂ − √λ₃ − √λ₄)`` with ``λᵢ`` the descending
    eigenvalues of ``ρ(Y⊗Y)ρ*(Y⊗Y)``.  The API is experimental and
    limited to two qubits.
    """
    rho = _to_density(state)
    if rho.shape != (4, 4):
        raise ValueError(f"Concurrence needs a 2-qubit state, got shape {rho.shape}")
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    spin_flip = np.kron(pauli_y, pauli_y)
    matrix = rho @ spin_flip @ rho.conj() @ spin_flip
    eigenvalues = np.real(np.linalg.eigvals(matrix))
    ordered = sorted((max(value, 0.0) for value in eigenvalues), reverse=True)
    roots = [float(np.sqrt(value)) for value in ordered]
    return max(0.0, roots[0] - roots[1] - roots[2] - roots[3])
