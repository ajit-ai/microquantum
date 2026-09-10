"""Matrix Product State (MPS) simulation.

An MPS represents a pure state of ``n`` qubits as ``n`` tensors
``A[t]`` of shape ``(chi_left, 2, chi_right)``, keeping the memory
footprint polynomial in the bond dimension ``chi`` instead of
exponential in the qubit count.  Gates are applied locally; two-qubit
gates use SVD truncation to cap the bond dimension, trading a small,
tracked amount of fidelity for massive state compression.

Supported:

* exact evolution (no truncation) on any circuit,
* bond-dimension truncation with an accumulating truncation error,
* left/right canonical-gauges (optimal truncation cuts),
* sequential (peeling) Born-rule sampling,
* reconstruction to a dense state vector for verification.

All tensor operations are dispatched through
:mod:`microquantum.backends.array_backend` so the same code runs on
NumPy or CuPy.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.state import StateVector
from .array_backend import (
    abs as ab_abs,
    asarray,
    get_array_backend,
    maximum as ab_maximum,
    sqrt as ab_sqrt,
    sum as ab_sum,
    svd,
    tensordot,
    to_numpy,
    transpose,
    zeros,
)
from .base import Backend, BackendResult

_COMPLEX = complex

# State vectors are only materialized when this small; beyond it the
# MPS is evaluated purely through sampling / local measurements.
_MAX_SV_QUBITS = 18


def _swap_matrix() -> Any:
    """4x4 SWAP gate in (q0, q1) combined-index order."""
    m = np.zeros((4, 4), dtype=np.complex128)
    m[0, 0] = 1.0
    m[1, 2] = 1.0
    m[2, 1] = 1.0
    m[3, 3] = 1.0
    return asarray(m, dtype=_COMPLEX)


class MatrixProductState:
    """Tensor-chain representation of a quantum state.

    Attributes:
        num_qubits: Number of qubits.
        tensors: List of site tensors ``A[t]`` (chi_left, 2, chi_right).
        truncation_error: Accumulated squared truncation error (norm loss).
    """

    def __init__(
        self,
        tensors: list[Any],
        truncation_error: float = 0.0,
    ) -> None:
        if not tensors:
            raise ValueError("MPS requires at least one tensor")
        self.tensors = list(tensors)
        self.num_qubits = len(self.tensors)
        self.truncation_error = truncation_error

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_zeros(cls, num_qubits: int) -> "MatrixProductState":
        """Zero state |0..0> as an MPS with all bonds equal to 1."""
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        tensors: list[Any] = []
        for _ in range(num_qubits):
            a = zeros((1, 2, 1), dtype=_COMPLEX)
            a[0, 0, 0] = complex(1.0, 0.0)
            tensors.append(a)
        return cls(tensors)

    @classmethod
    def from_statevector(
        cls,
        amplitudes: Any,
        num_qubits: int,
    ) -> "MatrixProductState":
        """Build an exact MPS from a dense state vector (big-endian)."""
        vec = asarray(amplitudes, dtype=_COMPLEX)
        if vec.size != (1 << num_qubits):
            raise ValueError(
                f"Expected {1 << num_qubits} amplitudes for {num_qubits} "
                f"qubits, got {vec.size}"
            )
        vec = vec.reshape(2**num_qubits)
        norm = ab_sum(ab_abs(vec) ** 2)
        if norm > 0:
            vec = vec / ab_sqrt(norm)

        cur: Any = vec.reshape([2] * num_qubits)
        tensors: list[Any] = []
        chi_l = 1
        for i in range(num_qubits - 1):
            mat = cur.reshape(chi_l * 2, -1)
            u, s, vh = svd(mat, full_matrices=False)
            k = max(1, int(s.shape[0]))
            tensors.append(u[:, :k].reshape(chi_l, 2, k))
            chi_l = k
            cur = s[:k, None] * vh[:k]
        tensors.append(cur.reshape(chi_l, 2, 1))
        return cls(tensors)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bond_dims(self) -> list[int]:
        """List of bond dimensions (one shorter than the site count)."""
        return [int(self.tensors[t].shape[2]) for t in range(self.num_qubits - 1)]

    @property
    def max_bond_dim(self) -> int:
        """Largest bond dimension in the chain."""
        return max(self.bond_dims) if self.num_qubits > 1 else 1

    # ------------------------------------------------------------------
    # Gauge (canonicalization)
    # ------------------------------------------------------------------

    def _left_canonicalize(self, stop: int) -> None:
        """Make sites ``0..stop-1`` left-canonical via SVD sweeps.

        After this, contracting each site's (left, physical) indices
        yields an identity on its right index.
        """
        n = self.num_qubits
        for t in range(min(stop, n)):
            a = self.tensors[t]
            chi_l, _, chi_r = a.shape
            mat = a.reshape(chi_l * 2, chi_r)
            u, s, vh = svd(mat, full_matrices=False)
            k = max(1, int(s.shape[0]))
            if t + 1 < n:
                self.tensors[t] = u[:, :k].reshape(chi_l, 2, k)
                bond = s[:k, None] * vh[:k]
                self.tensors[t + 1] = tensordot(
                    bond, self.tensors[t + 1], axes=([1], [0])
                )
            else:
                # Rightmost site: fold the (scalar) singular weight back in.
                self.tensors[t] = (u[:, :k] * s[:k]).reshape(chi_l, 2, chi_r)

    def _right_canonicalize(self, start: int) -> None:
        """Make sites ``start..n-1`` right-canonical via SVD sweeps.

        After this, contracting each site's (physical, right) indices
        yields an identity on its left index.  The leftmost site is left
        untouched (it carries the global boundary weight).
        """
        n = self.num_qubits
        for t in range(n - 1, max(start, 0) - 1, -1):
            if t == 0:
                continue
            a = self.tensors[t]
            chi_l, _, chi_r = a.shape
            mat = a.reshape(chi_l, 2 * chi_r)
            u, s, vh = svd(mat, full_matrices=False)
            k = max(1, int(s.shape[0]))
            self.tensors[t] = vh[:k].reshape(k, 2, chi_r)
            bond = u[:, :k] * s[:k]
            self.tensors[t - 1] = tensordot(
                self.tensors[t - 1], bond, axes=([-1], [0])
            )

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def apply_single(self, gate: Any, target: int) -> None:
        """Apply a local 2x2 unitary to a single qubit (exact)."""
        if not (0 <= target < self.num_qubits):
            raise ValueError(f"target qubit {target} out of range")
        gate = asarray(gate, dtype=_COMPLEX)
        a = self.tensors[target]
        out = tensordot(gate, a, axes=([1], [1]))
        self.tensors[target] = transpose(out, (1, 0, 2))

    def apply_two(
        self,
        gate: Any,
        targets: list[int],
        max_bond_dim: Optional[int] = None,
        truncation_threshold: float = 1e-12,
    ) -> None:
        """Apply a 4x4 two-qubit gate, truncating the shared bonds.

        The gate matrix acts with the *first* target as the leading
        (most significant) physical index, matching ``apply_gate``.
        Non-adjacent qubits are brought together by exact SWAP gates and
        restored afterwards.
        """
        if len(targets) != 2:
            raise ValueError("two-qubit gate requires exactly 2 targets")
        t, u = int(targets[0]), int(targets[1])
        n = self.num_qubits
        if not (0 <= t < n and 0 <= u < n):
            raise ValueError(f"targets {targets} out of range")

        gate = asarray(gate, dtype=_COMPLEX)
        if gate.shape != (4, 4):
            raise ValueError("two-qubit gate must be a 4x4 matrix")

        lo = min(t, u)

        # Bring the two targets together so the first target ends up on
        # the left: it must be the leading physical index of the gate.
        if t < u:
            for p in range(u - 1, t, -1):
                self._apply_two_adjacent(_swap_matrix(), p, None, 0.0)
        else:
            for p in range(t - 1, u - 1, -1):
                self._apply_two_adjacent(_swap_matrix(), p, None, 0.0)

        # Center the gauge so the SVD cut is globally optimal.
        if lo > 0:
            self._left_canonicalize(lo)
        if lo + 1 < n - 1:
            self._right_canonicalize(lo + 2)

        self._apply_two_adjacent(
            gate, lo, max_bond_dim, truncation_threshold
        )

        # Restore the original ordering with exact SWAPs.
        if t < u:
            for p in range(t + 1, u):
                self._apply_two_adjacent(_swap_matrix(), p, None, 0.0)
        else:
            for p in range(u, t):
                self._apply_two_adjacent(_swap_matrix(), p, None, 0.0)

    def _apply_two_adjacent(
        self,
        gate: Any,
        pos: int,
        max_bond_dim: Optional[int],
        truncation_threshold: float,
    ) -> None:
        """Apply a two-qubit gate on adjacent sites ``pos, pos+1``."""
        n = self.num_qubits
        a = self.tensors[pos]
        b = self.tensors[pos + 1]
        chi_l = a.shape[0]
        chi_r = b.shape[2]

        two = tensordot(a, b, axes=([2], [0]))          # (chi_l, 2, 2, chi_r)
        four = two.reshape(chi_l, 4, chi_r)
        out = tensordot(gate, four, axes=([1], [1]))    # (4, chi_l, chi_r)
        out = transpose(out, (1, 0, 2))                 # (chi_l, 4, chi_r)
        out = out.reshape(chi_l, 2, 2, chi_r)
        mat = out.reshape(chi_l * 2, 2 * chi_r)

        u, s, vh = svd(mat, full_matrices=False)
        s = asarray(s)
        s_np = to_numpy(s)

        if s_np.size == 0:
            k = 1
        else:
            cutoff = truncation_threshold * float(s_np.max())
            keep = s_np > max(cutoff, 0.0)
            if max_bond_dim is not None:
                keep[min(int(keep.sum()), max_bond_dim):] = False  # type: ignore[index]
            k = max(1, int(keep.sum()))
            discarded = float(s_np[~keep].sum() ** 2) if (~keep).any() else 0.0
            self.truncation_error += discarded

        u = u[:, :k]
        s = s[:k]
        vh = vh[:k]
        self.tensors[pos] = u.reshape(chi_l, 2, k)
        bond = s[:, None] * vh
        self.tensors[pos + 1] = bond.reshape(k, 2, chi_r)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def to_statevector(self) -> Any:
        """Contract the full chain to a dense state vector (big-endian)."""
        psi = self.tensors[0]
        for t in range(1, self.num_qubits):
            psi = tensordot(psi, self.tensors[t], axes=([-1], [0]))
        vec = psi.reshape(1 << self.num_qubits)
        return to_numpy(vec)

    def amplitude(self, bitstring: str) -> complex:
        """Return the complex amplitude of a computational basis state."""
        vec = to_numpy(self.to_statevector())
        idx = 0
        for i, ch in enumerate(bitstring):
            idx |= (int(ch) << (self.num_qubits - 1 - i))
        return complex(vec[idx])

    def probabilities(self) -> dict[str, float]:
        """Return the full Born-rule distribution (dense, for small n)."""
        vec = to_numpy(self.to_statevector())
        probs = np.abs(vec) ** 2
        out: dict[str, float] = {}
        for idx, p in enumerate(probs):
            if p > 0:
                out[format(idx, f"0{self.num_qubits}b")] = float(p)
        return out

    def valid_statevector(self) -> bool:
        """True when dense reconstruction is affordable for this system."""
        return self.num_qubits <= _MAX_SV_QUBITS

    def norm_sq(self) -> float:
        """Square norm of the state (== 1 if no truncation error)."""
        vec = to_numpy(self.to_statevector())
        return float(np.sum(np.abs(vec) ** 2))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, shots: int, seed: Optional[int] = None) -> dict[str, int]:
        """Sample bitstrings via sequential (peeling) Born-rule sampling.

        The state is right-canonicalized first so each conditional
        distribution is exact.
        """
        if shots < 1:
            raise ValueError("shots must be >= 1")
        self._right_canonicalize(0)

        rng = np.random.default_rng(seed)
        counts: dict[str, int] = {}
        n = self.num_qubits
        for _ in range(shots):
            boundary = asarray([complex(1.0, 0.0)], dtype=_COMPLEX)
            bits: list[str] = []
            for t in range(n):
                a = self.tensors[t]
                env = tensordot(boundary, a, axes=([0], [0]))  # (2, chi_r)
                probs = ab_abs(env) ** 2
                probs = ab_sum(probs, axis=1)
                probs = probs / ab_sum(probs)
                p_np = to_numpy(probs)
                p_np = np.maximum(p_np, 1e-15)
                p_np = p_np / p_np.sum()
                b = int(rng.choice(2, p=p_np))
                bits.append(str(b))
                boundary = env[b] / ab_maximum(ab_sqrt(probs[b]), 1e-15)
            bs = "".join(bits)
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Overlap
    # ------------------------------------------------------------------

    def fidelity_to(self, reference: Any) -> float:
        """Fidelity with a reference dense state vector: |<psi|phi>|^2."""
        vec = to_numpy(self.to_statevector())
        ref = np.asarray(to_numpy(reference), dtype=np.complex128).reshape(-1)
        if vec.size != ref.size:
            raise ValueError("reference state dimension mismatch")
        return float(abs(np.vdot(vec, ref)) ** 2)

    def __repr__(self) -> str:
        return (
            f"MatrixProductState(num_qubits={self.num_qubits}, "
            f"bonds={self.bond_dims})"
        )


class MPSBackend(Backend):
    """Backend that simulates circuits via matrix product states.

    Args:
        max_bond_dim: Optional cap on the bond dimension for two-qubit
            gates (None = exact, grows as needed).
        truncation_threshold: Relative singular-value cutoff.
        seed: Default RNG seed.
    """

    def __init__(
        self,
        max_bond_dim: Optional[int] = None,
        truncation_threshold: float = 1e-12,
        seed: Optional[int] = None,
    ) -> None:
        self.max_bond_dim = max_bond_dim
        self.truncation_threshold = truncation_threshold
        self.seed = seed

    @property
    def name(self) -> str:
        return "mps"

    def run_circuit(
        self,
        num_qubits: int,
        gates: list[tuple[Any, list[int]]],
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Execute a circuit via matrix product state simulation."""
        effective_seed = seed if seed is not None else self.seed
        if initial_state is None:
            mps = MatrixProductState.from_zeros(num_qubits)
        else:
            if initial_state.num_qubits != num_qubits:
                raise ValueError(
                    f"Initial state has {initial_state.num_qubits} qubits "
                    f"but circuit has {num_qubits}"
                )
            mps = MatrixProductState.from_statevector(
                initial_state.amplitudes, num_qubits
            )

        for gate_matrix, targets in gates:
            if len(targets) == 1:
                mps.apply_single(gate_matrix, targets[0])
            else:
                mps.apply_two(
                    gate_matrix,
                    targets,
                    max_bond_dim=self.max_bond_dim,
                    truncation_threshold=self.truncation_threshold,
                )

        counts = mps.sample(shots, seed=effective_seed)
        statevector = None
        if mps.valid_statevector():
            statevector = np.asarray(to_numpy(mps.to_statevector()), dtype=np.complex128)

        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            statevector=statevector,
            counts=counts,
            metadata={
                "shots": shots,
                "seed": effective_seed,
                "max_bond_dim": mps.max_bond_dim,
                "bond_dims": mps.bond_dims,
                "truncation_error": mps.truncation_error,
                "truncation_threshold": self.truncation_threshold,
                "array_backend": get_array_backend(),
                "statevector_available": mps.valid_statevector(),
            },
        )

    def __repr__(self) -> str:
        trunc = f", max_bond_dim={self.max_bond_dim}" if self.max_bond_dim else ""
        return f"MPSBackend(name='mps'{trunc})"