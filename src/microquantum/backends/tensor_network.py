"""Tree tensor network (TTN) simulation.

A TTN represents a state as a balanced binary tree of three-leg tensors.
The root tensor opens onto the whole system; every internal node tensor
has shape ``(up_bond, left_bond, right_bond)`` and leaves open onto the
physical index of a single qubit.

Two-site gate application contracts the subtree of the gate's *least
common ancestor*, applies the gate, and re-splits the block into fresh
tree tensors via two sequential SVDs (exact when every bond is retained,
optionally truncating to a maximum bond dimension).  All tensor
operations are dispatched through :mod:`microquantum.backends.array_backend`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.state import StateVector
from .array_backend import (
    asarray,
    get_array_backend,
    svd,
    tensordot,
    to_numpy,
    transpose,
    zeros,
)
from .base import Backend, BackendResult

_COMPLEX = complex
_MAX_SV_QUBITS = 18


class _Node:
    """A single tensor in the tree network.

    Attributes:
        is_leaf: True for physical qubit tensors.
        leaf_index: Qubit index (leaves only).
        left / right: Child nodes (internal nodes only).
        tensor: (up, left_bond, right_bond) for internal nodes,
            (up, 2) for leaves.
        num_leaves: Number of qubits in this subtree.
    """

    def __init__(
        self,
        is_leaf: bool,
        leaf_index: Optional[int] = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
    ) -> None:
        self.is_leaf = is_leaf
        self.leaf_index = leaf_index
        self.left = left
        self.right = right
        self.tensor: Any = None
        self.num_leaves: int = 1 if is_leaf else 0
        if left is not None and right is not None:
            self.num_leaves = left.num_leaves + right.num_leaves

    @property
    def up(self) -> int:
        """Size of the bond to the parent."""
        return int(self.tensor.shape[0]) if self.tensor is not None else 1


class TreeTensorNetwork:
    """Balanced binary tree tensor network over ``n`` qubits.

    Attributes:
        num_qubits: Number of qubits.
        root: Root ``_Node``.
        truncation_error: Accumulated squared truncation error (norm loss).
    """

    def __init__(
        self,
        num_qubits: int,
        root: "_Node",
        truncation_error: float = 0.0,
    ) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.root = root
        self.truncation_error = truncation_error
        self._leaves: dict[int, _Node] = {}
        self._collect_leaves(root)

    def _collect_leaves(self, node: _Node) -> None:
        if node.is_leaf:
            if node.leaf_index is not None:
                self._leaves[node.leaf_index] = node
            return
        if node.left is not None:
            self._collect_leaves(node.left)
        if node.right is not None:
            self._collect_leaves(node.right)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build(lo: int, hi: int) -> _Node:
        """Balanced binary tree over qubit indices ``[lo, hi)``."""
        if hi - lo == 1:
            return _Node(is_leaf=True, leaf_index=lo)
        mid = (lo + hi) // 2
        left = TreeTensorNetwork._build(lo, mid)
        right = TreeTensorNetwork._build(mid, hi)
        return _Node(is_leaf=False, left=left, right=right)

    @classmethod
    def from_zeros(cls, num_qubits: int) -> "TreeTensorNetwork":
        """Product state |0..0> as a tree network (all bonds size 1)."""
        root = cls._build(0, num_qubits)

        def init(node: _Node) -> None:
            if node.is_leaf:
                t = zeros((1, 2), dtype=_COMPLEX)
                t[0, 0] = complex(1.0, 0.0)
                node.tensor = t
            else:
                t = zeros((1, 1, 1), dtype=_COMPLEX)
                t[0, 0, 0] = complex(1.0, 0.0)
                node.tensor = t
                if node.left is not None:
                    init(node.left)
                if node.right is not None:
                    init(node.right)

        init(root)
        return cls(num_qubits, root)

    @classmethod
    def from_statevector(
        cls, amplitudes: Any, num_qubits: int
    ) -> "TreeTensorNetwork":
        """Exact tree network reconstructing a dense state vector."""
        ttn = cls.from_zeros(num_qubits)
        block = asarray(amplitudes, dtype=_COMPLEX).reshape(
            1, *((2,) * num_qubits)
        )
        ttn._resplit(ttn.root, block, None, 0.0)
        return ttn

    # ------------------------------------------------------------------
    # LCA lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _lca(root: _Node, a: int, b: int) -> _Node:
        """Lowest common ancestor of qubits ``a`` and ``b`` (a != b)."""
        current = root
        while not current.is_leaf:
            left = current.left
            right = current.right
            a_in_l = left is not None and a in _leaf_set(left)
            b_in_l = left is not None and b in _leaf_set(left)
            if a_in_l == b_in_l:
                current = left if a_in_l else right  # type: ignore[assignment]
            else:
                return current
        return current

    def lowest_common_ancestor(self, a: int, b: int) -> _Node:
        """Tree node that is an ancestor of both qubits ``a`` and ``b``."""
        return self._lca(self.root, a, b)

    # ------------------------------------------------------------------
    # Contraction
    # ------------------------------------------------------------------

    def _contract(self, node: _Node) -> Any:
        """Contract a subtree to a dense block (up, 2^leaves)."""
        if node.is_leaf:
            return node.tensor
        bl = self._contract(node.left)  # type: ignore[arg-type]
        br = self._contract(node.right)  # type: ignore[arg-type]
        t = node.tensor
        # t: (d_up, d_l, d_r); bl: (d_l, 2^kl); br: (d_r, 2^kr)
        tmp = tensordot(t, bl, axes=([1], [0]))          # (d_up, d_r, 2^kl)
        return tensordot(tmp, br, axes=([1], [0]))       # (d_up, leaves)

    def to_statevector(self) -> Any:
        """Contract the whole tree to a dense state vector (big-endian)."""
        block = self._contract(self.root)
        return to_numpy(block.reshape(1 << self.num_qubits))

    def valid_statevector(self) -> bool:
        """True when dense reconstruction is affordable."""
        return self.num_qubits <= _MAX_SV_QUBITS

    def probabilities(self) -> dict[str, float]:
        """Full Born-rule distribution (dense, for small n)."""
        vec = to_numpy(self.to_statevector())
        return {
            format(i, f"0{self.num_qubits}b"): float(p)
            for i, p in enumerate(np.abs(vec) ** 2)
            if p > 0
        }

    # ------------------------------------------------------------------
    # Splitting / re-split
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(s: Any, max_bond: Optional[int], threshold: float) -> int:
        """Number of singular values to keep."""
        s_np = to_numpy(s)
        if s_np.size == 0:
            return 1
        if max_bond is None and threshold <= 0.0:
            return int(s_np.size)
        cutoff = threshold * float(np.abs(s_np).max())
        keep = np.abs(s_np) > max(cutoff, 0.0)
        if max_bond is not None:
            limit = min(int(keep.sum()), max_bond)
            keep[limit:] = False
        return max(1, int(keep.sum()))

    def _resplit(
        self,
        node: _Node,
        block: Any,
        max_bond: Optional[int],
        threshold: float,
    ) -> None:
        """Decompose a dense subtree block into fresh tree tensors.

        Uses two sequential SVDs per node: one across the right child
        cut and one across the left child cut.  Exact when every bond is
        retained.
        """
        if node.is_leaf:
            node.tensor = block
            return

        left = node.left
        right = node.right
        assert left is not None and right is not None, "internal node children"
        kl = left.num_leaves
        kr = right.num_leaves
        d_up = block.shape[0]

        # Cut A: (up + left leaves) x right leaves
        mat = block.reshape(d_up * (1 << kl), (1 << kr))
        u, s, vh = svd(mat, full_matrices=False)
        s = asarray(s)
        k1 = self._truncate(s, max_bond, threshold)
        s_arr = to_numpy(s)
        if k1 < s_arr.size:
            self.truncation_error += float(np.sum(np.abs(s_arr[k1:]) ** 2))
        left_part = u[:, :k1]                       # (d_up 2^kl, k1)
        right_block = (s[:k1, None] * vh[:k1]).reshape(k1, (1 << kr))

        # Cut B: left leaves x (up + right bond)
        # left_part rows are fused (up, left leaves) in up-major order;
        # transpose so rows run over the leaves before the SVD.
        mat2 = transpose(left_part.reshape(d_up, (1 << kl), k1), (1, 0, 2))
        mat2 = mat2.reshape((1 << kl), d_up * k1)
        u2, s2, vh2 = svd(mat2, full_matrices=False)
        s2 = asarray(s2)
        k2 = self._truncate(s2, max_bond, threshold)
        s2_arr = to_numpy(s2)
        if k2 < s2_arr.size:
            self.truncation_error += float(np.sum(np.abs(s2_arr[k2:]) ** 2))
        left_block = transpose(u2[:, :k2]).reshape(k2, (1 << kl))
        node_tensor = (s2[:k2, None] * vh2[:k2]).reshape(
            k2, d_up, k1
        )
        node.tensor = transpose(node_tensor, (1, 0, 2))  # (d_up, k2, k1)

        self._resplit(left, left_block, max_bond, threshold)
        self._resplit(right, right_block, max_bond, threshold)

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def apply_single(self, gate: Any, target: int) -> None:
        """Apply a local 2x2 unitary to a single qubit (exact)."""
        if target not in self._leaves:
            raise ValueError(f"target qubit {target} out of range")
        gate = asarray(gate, dtype=_COMPLEX)
        leaf = self._leaves[target]
        out = tensordot(gate, leaf.tensor, axes=([1], [1]))
        leaf.tensor = transpose(out, (1, 0))

    def apply_two(
        self,
        gate: Any,
        targets: list[int],
        max_bond_dim: Optional[int] = None,
        truncation_threshold: float = 1e-12,
    ) -> None:
        """Apply a 4x4 two-qubit gate via LCA subtree recontraction."""
        if len(targets) != 2:
            raise ValueError("two-qubit gate requires exactly 2 targets")
        a, b = int(targets[0]), int(targets[1])
        if a not in self._leaves or b not in self._leaves:
            raise ValueError(f"targets {targets} out of range")
        gate = asarray(gate, dtype=_COMPLEX)
        if gate.shape != (4, 4):
            raise ValueError("two-qubit gate must be a 4x4 matrix")

        node = self.lowest_common_ancestor(a, b)
        block = self._contract(node)                      # (d_up, 2^leaves)

        # Apply gate on the physical axes of the (a, b) pair.
        leaves: list[int] = []
        self._leaf_order(node, leaves)
        ia = leaves.index(a)
        ib = leaves.index(b)
        block = self._apply_gate_to_block(block, ia, ib, gate)

        self._resplit(node, block, max_bond_dim, truncation_threshold)

    def _leaf_order(self, node: _Node, out: list[int]) -> None:
        if node.is_leaf:
            if node.leaf_index is not None:
                out.append(node.leaf_index)
        else:
            self._leaf_order(node.left, out)  # type: ignore[arg-type]
            self._leaf_order(node.right, out)  # type: ignore[arg-type]

    @staticmethod
    def _apply_gate_to_block(block: Any, ia: int, ib: int, gate: Any) -> Any:
        """Act with a 4x4 gate on leaf axes ``ia``/``ib`` (1-based after up)."""
        # block: (d_up, 2, 2, ..., 2) with leaf axes #1..#k.
        d_up = block.shape[0]
        ax_a = ia + 1
        ax_b = ib + 1
        other = [k for k in range(1, block.ndim) if k not in (ax_a, ax_b)]
        perm = [0, ax_a, ax_b] + other
        b2 = transpose(block, perm)
        rest = 1
        for d in b2.shape[3:]:
            rest *= d
        b3 = b2.reshape(d_up, 2, 2, rest)
        # gate flat index = a*2 + b over targets (a, b); C-order reshape
        # yields (in_a, in_b, out_a, out_b).
        g4 = gate.reshape(2, 2, 2, 2)
        out = tensordot(g4, b3, axes=([0, 1], [1, 2]))  # (out_a, out_b, d_up, rest)
        out = transpose(out, (2, 0, 1, 3))             # (d_up, out_a, out_b, rest)
        dims = list(b2.shape)
        dims[1] = 2
        dims[2] = 2
        out = out.reshape(*dims)
        # Transpose back to the original axis layout.
        inv = [0] * block.ndim
        for i, p in enumerate(perm):
            inv[p] = i
        return transpose(out, inv)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, shots: int, seed: Optional[int] = None) -> dict[str, int]:
        """Sample bitstrings from the Born distribution."""
        if shots < 1:
            raise ValueError("shots must be >= 1")
        vec = to_numpy(self.to_statevector())
        probs = np.abs(vec) ** 2
        probs = probs / probs.sum()
        rng = np.random.default_rng(seed)
        outcomes = rng.choice(1 << self.num_qubits, size=shots, p=probs)
        counts: dict[str, int] = {}
        for out in outcomes:
            bs = format(int(out), f"0{self.num_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def fidelity_to(self, reference: Any) -> float:
        """Fidelity |<psi|phi>|^2 with a dense reference state."""
        vec = to_numpy(self.to_statevector())
        ref = np.asarray(to_numpy(reference), dtype=np.complex128).reshape(-1)
        if vec.size != ref.size:
            raise ValueError("reference state dimension mismatch")
        return float(abs(np.vdot(vec, ref)) ** 2)

    def __repr__(self) -> str:
        return f"TreeTensorNetwork(num_qubits={self.num_qubits})"


def _leaf_set(node: _Node) -> set[int]:
    """Set of qubit indices in a subtree (small-n utility)."""
    result: set[int] = set()
    _walk(node, result)
    return result


def _walk(node: _Node, out: set[int]) -> None:
    if node.is_leaf:
        if node.leaf_index is not None:
            out.add(node.leaf_index)
        return
    if node.left is not None:
        _walk(node.left, out)
    if node.right is not None:
        _walk(node.right, out)


class TreeTensorNetworkBackend(Backend):
    """Backend that simulates circuits via a balanced tree network.

    Args:
        max_bond_dim: Optional cap on bond dimensions during two-qubit
            gate re-splitting (None = exact).
        truncation_threshold: Relative singular-value cutoff (0 = exact).
        seed: Default RNG seed.
    """

    def __init__(
        self,
        max_bond_dim: Optional[int] = None,
        truncation_threshold: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.max_bond_dim = max_bond_dim
        self.truncation_threshold = truncation_threshold
        self.seed = seed

    @property
    def name(self) -> str:
        return "ttn"

    def run_circuit(
        self,
        num_qubits: int,
        gates: list[tuple[Any, list[int]]],
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Execute a circuit via tree tensor network simulation."""
        effective_seed = seed if seed is not None else self.seed
        ttn = TreeTensorNetwork.from_zeros(num_qubits)
        if initial_state is not None:
            if initial_state.num_qubits != num_qubits:
                raise ValueError(
                    f"Initial state has {initial_state.num_qubits} qubits "
                    f"but circuit has {num_qubits}"
                )
            ttn = TreeTensorNetwork.from_statevector(
                initial_state.amplitudes, num_qubits
            )

        for gate_matrix, targets in gates:
            if len(targets) == 1:
                ttn.apply_single(gate_matrix, targets[0])
            else:
                ttn.apply_two(
                    gate_matrix,
                    targets,
                    max_bond_dim=self.max_bond_dim,
                    truncation_threshold=self.truncation_threshold,
                )

        counts = ttn.sample(shots, seed=effective_seed)
        statevector = None
        if ttn.valid_statevector():
            statevector = np.asarray(to_numpy(ttn.to_statevector()), dtype=np.complex128)

        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            statevector=statevector,
            counts=counts,
            metadata={
                "shots": shots,
                "seed": effective_seed,
                "truncation_error": ttn.truncation_error,
                "truncation_threshold": self.truncation_threshold,
                "max_bond_dim": self.max_bond_dim,
                "array_backend": get_array_backend(),
                "statevector_available": ttn.valid_statevector(),
            },
        )

    def __repr__(self) -> str:
        trunc = f", max_bond_dim={self.max_bond_dim}" if self.max_bond_dim else ""
        return f"TreeTensorNetworkBackend(name='ttn'{trunc})"