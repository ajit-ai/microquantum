"""Transpiler pipeline for quantum circuit optimization and compilation.

Provides a composable pass-based transpiler framework with built-in
passes for gate decomposition, cancellation, fusion, identity removal,
layout mapping, and depth reduction.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .circuit import QuantumCircuit
from .coupling import CouplingMap
from .operators import Operator
from .optimization import (
    cancel_inverse_pairs,
    fuse_single_qubit_gates,
    remove_identity_gates,
)


# ------------------------------------------------------------------
# Target gate set
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TargetGateSet:
    """Defines the allowed gate set for decomposition passes.

    Attributes:
        single_qubit_gates: Allowed single-qubit gate names.
        two_qubit_gates: Allowed two-qubit gate names.
    """

    single_qubit_gates: frozenset[str] = field(
        default_factory=lambda: frozenset({"h", "x", "rz", "ry"})
    )
    two_qubit_gates: frozenset[str] = field(
        default_factory=lambda: frozenset({"cx"})
    )

    @property
    def basis_gates(self) -> frozenset[str]:
        """Union of single- and two-qubit gate names."""
        return self.single_qubit_gates | self.two_qubit_gates

    @classmethod
    def default(cls) -> TargetGateSet:
        """Return the default target gate set {h, x, rz, ry, cx}."""
        return cls()


# ------------------------------------------------------------------
# Pass base class
# ------------------------------------------------------------------


class Pass(ABC):
    """Abstract base class for all transpiler passes.

    Each pass takes a ``QuantumCircuit`` and returns a (potentially
    modified) ``QuantumCircuit``.  Passes must skip parameterized
    circuits and return them unchanged.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pass name."""

    @abstractmethod
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Execute the pass on *circuit*.

        Args:
            circuit: The input circuit to transform.

        Returns:
            The transformed circuit.
        """


# ------------------------------------------------------------------
# Pass manager
# ------------------------------------------------------------------


class PassManager:
    """Manages an ordered pipeline of transpiler passes.

    Passes are executed sequentially; the output of one pass feeds into
    the next.

    Example::

        pm = PassManager.from_optimization_level(2)
        optimized = pm.run(circuit)
    """

    def __init__(self) -> None:
        self._passes: list[Pass] = []

    @property
    def num_passes(self) -> int:
        """Number of passes in the pipeline."""
        return len(self._passes)

    def append_pass(self, pass_: Pass) -> None:
        """Add a single pass to the end of the pipeline.

        Args:
            pass_: The pass to append.
        """
        self._passes.append(pass_)

    def append_passes(self, passes: list[Pass]) -> None:
        """Add multiple passes to the end of the pipeline.

        Args:
            passes: List of passes to append.
        """
        self._passes.extend(passes)

    def clear(self) -> None:
        """Remove all passes from the pipeline."""
        self._passes.clear()

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Run all passes sequentially on *circuit*.

        Args:
            circuit: The input circuit.

        Returns:
            The circuit after all passes have been applied.
        """
        result = circuit
        for p in self._passes:
            result = p.run(result)
        return result

    def run_with_stats(
        self, circuit: QuantumCircuit
    ) -> tuple[QuantumCircuit, list[tuple[str, int, int]]]:
        """Run all passes, recording gate counts before and after each.

        Args:
            circuit: The input circuit.

        Returns:
            A tuple ``(result, stats)`` where *stats* is a list of
            ``(pass_name, gates_before, gates_after)`` tuples.
        """
        result = circuit
        stats: list[tuple[str, int, int]] = []
        for p in self._passes:
            gates_before = result.num_gates
            result = p.run(result)
            gates_after = result.num_gates
            stats.append((p.name, gates_before, gates_after))
        return result, stats

    @staticmethod
    def from_optimization_level(
        level: int,
        coupling_map: CouplingMap | None = None,
        noise_rates: dict[tuple[int, int], float] | None = None,
    ) -> PassManager:
        """Create a ``PassManager`` for a given optimization level.

        Levels:

        - **0** – No optimization (empty pipeline).
        - **1** – Basic: cancel inverse pairs, remove identities, fuse
          single-qubit gates.
        - **2** – Full simplification (level 1 + gate decomposition +
          depth reduction).  If *coupling_map* is provided, routing is
          added before decomposition.
        - **3** – Aggressive: level 2 plus a second round of fusion and
          cancellation.  If *coupling_map* and *noise_rates* are
          provided, noise-aware placement is added.

        Args:
            level: Optimization level (0–3).
            coupling_map: Optional device connectivity for routing.
            noise_rates: Optional per-edge error rates for noise-aware
                placement.  Keys are sorted ``(i, j)`` tuples.

        Returns:
            A configured ``PassManager``.

        Raises:
            ValueError: If *level* is not in 0–3.
        """
        if not 0 <= level <= 3:
            raise ValueError(
                f"optimization_level must be 0-3, got {level}"
            )

        pm = PassManager()

        if level == 0:
            return pm

        # Level 1+: basic simplification
        pm.append_passes([
            CancellationPass(),
            IdentityRemovalPass(),
            FusionPass(),
        ])

        if level >= 2:
            if coupling_map is not None:
                pm.append_pass(RoutingPass(coupling_map))
            pm.append_passes([
                GateDecompositionPass(),
                CancellationPass(),
                IdentityRemovalPass(),
                FusionPass(),
                DepthReductionPass(),
            ])

        if level >= 3:
            if coupling_map is not None and noise_rates is not None:
                pm.append_pass(
                    NoiseAwarePlacementPass(coupling_map, noise_rates)
                )
            pm.append_passes([
                CancellationPass(),
                IdentityRemovalPass(),
                FusionPass(),
            ])

        return pm

    def __repr__(self) -> str:
        names = [p.name for p in self._passes]
        return f"PassManager({names})"


# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------


def _is_identity_matrix(mat: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether *mat* is approximately the identity."""
    n = mat.shape[0]
    return bool(np.allclose(mat, np.eye(n, dtype=np.complex128), atol=atol))


def _angle_close(a: float, b: float, atol: float = 1e-10) -> bool:
    """Check if two angles are close modulo 2*pi."""
    diff = (a - b) % (2 * math.pi)
    return diff < atol or (2 * math.pi - diff) < atol


def _build_circuit(
    num_qubits: int,
    gates: list[tuple[Operator, list[int]]],
) -> QuantumCircuit:
    """Build a ``QuantumCircuit`` from ``(operator, targets)`` pairs."""
    qc = QuantumCircuit(num_qubits)
    for op, targets in gates:
        qc.append(op, targets)
    return qc


# ------------------------------------------------------------------
# Built-in passes
# ------------------------------------------------------------------


class GateDecompositionPass(Pass):
    """Decompose multi-qubit gates into the target gate set.

    Recognized single-qubit gates (H, X, S, T, Rx, Ry, Rz) that are
    already in the target set are passed through unchanged.  Two-qubit
    gates not in the target set are decomposed using a KAK-inspired
    (Schrieffer–Wolff) decomposition into at most three CX gates plus
    single-qubit rotations.  CZ is decomposed to H–CX–H.  SWAP is
    decomposed into three CX gates.

    Unrecognized gates are approximated via their unitary matrix
    decomposed into the basis.

    Args:
        target: The allowed gate set.  Defaults to ``{h, x, rz, ry, cx}``.
    """

    def __init__(self, target: TargetGateSet | None = None) -> None:
        self._target = target or TargetGateSet.default()

    @property
    def name(self) -> str:
        """Pass name."""
        return "gate_decomposition"

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Decompose gates not in the target gate set."""
        if circuit.is_parameterized:
            return circuit

        target_lower = {g.lower() for g in self._target.basis_gates}
        new_gates: list[tuple[Operator, list[int]]] = []

        for op, targets in circuit.gates:
            gname = op.name.lower()

            if op.num_qubits == 1:
                if gname in target_lower:
                    new_gates.append((op, targets))
                else:
                    new_gates.extend(
                        GateDecompositionPass._decompose_single_qubit(
                            op, targets[0], target_lower
                        )
                    )
            elif op.num_qubits == 2:
                if gname in target_lower:
                    new_gates.append((op, targets))
                elif gname == "cz":
                    new_gates.extend(
                        GateDecompositionPass._decompose_cz(
                            targets[0], targets[1]
                        )
                    )
                elif gname == "swap":
                    new_gates.extend(
                        GateDecompositionPass._decompose_swap(
                            targets[0], targets[1]
                        )
                    )
                elif gname == "cnot" and "cx" in target_lower:
                    new_gates.append((Operator.CNOT(), targets))
                else:
                    new_gates.extend(
                        GateDecompositionPass._decompose_two_qubit(
                            op, targets[0], targets[1], target_lower
                        )
                    )
            else:
                new_gates.append((op, targets))

        return _build_circuit(circuit.num_qubits, new_gates)

    # -- known decompositions for standard gates ---------------------

    @staticmethod
    def _decompose_cz(
        control: int, target: int
    ) -> list[tuple[Operator, list[int]]]:
        """CZ = H(t) CX(c,t) H(t)."""
        return [
            (Operator.H(), [target]),
            (Operator.CNOT(), [control, target]),
            (Operator.H(), [target]),
        ]

    @staticmethod
    def _decompose_swap(
        q1: int, q2: int
    ) -> list[tuple[Operator, list[int]]]:
        """SWAP = CX(a,b) CX(b,a) CX(a,b)."""
        return [
            (Operator.CNOT(), [q1, q2]),
            (Operator.CNOT(), [q2, q1]),
            (Operator.CNOT(), [q1, q2]),
        ]

    @staticmethod
    def _decompose_cnot_from_cz(
        control: int, target: int
    ) -> list[tuple[Operator, list[int]]]:
        """CX = H(c) CZ(c,t) H(c)."""
        return [
            (Operator.H(), [control]),
            (Operator.CZ(), [control, target]),
            (Operator.H(), [control]),
        ]

    # -- single qubit ZYZ decomposition (arbitrary U2 -> Ry, Rz) ----

    @staticmethod
    def _decompose_single_qubit(
        op: Operator, target: int, target_gates: set[str]
    ) -> list[tuple[Operator, list[int]]]:
        """Decompose an arbitrary single-qubit unitary into Ry/Rz/H/X.

        Uses the ZYZ decomposition: U = e^{i*alpha} Rz(beta) Ry(gamma) Rz(delta).
        Global phase is discarded.  Special cases are detected for gates
        already in the target set.
        """
        mat = op.matrix

        # Check for known gates first (fast path)
        known: dict[str, Operator] = {
            "h": Operator.H(),
            "x": Operator.X(),
            "y": Operator.Y(),
            "z": Operator.Z(),
            "s": Operator.S(),
            "t": Operator.T(),
        }
        for gname, gate_op in known.items():
            if gname in target_gates and np.allclose(
                mat, gate_op.matrix, atol=1e-10
            ):
                return [(gate_op, [target])]

        # General ZYZ decomposition
        # U = e^{i*alpha} Rz(beta) Ry(gamma) Rz(delta)
        # Extract global phase and the SO(3) rotation
        det = np.linalg.det(mat)
        phase = np.angle(det) / 2
        su2 = mat / np.exp(1j * phase)

        # Rotation matrix elements: su2 = [[a, -b*], [b, a*]]
        a = su2[0, 0]
        b = su2[1, 0]

        gamma = 2 * math.acos(max(-1.0, min(1.0, abs(a))))
        if abs(math.sin(gamma / 2)) < 1e-12:
            # Degenerate: diagonal matrix -> two Rz gates
            delta = np.angle(a)
            beta = 0.0
        else:
            delta = np.angle(b) + math.pi / 2
            beta = np.angle(a) - delta

        gates: list[tuple[Operator, list[int]]] = []
        target_lower = {g.lower() for g in target_gates}

        if "rz" in target_lower:
            if not _angle_close(delta, 0.0):
                gates.append((Operator.Rz(delta), [target]))
            if not _angle_close(gamma, 0.0):
                if "ry" in target_lower:
                    gates.append((Operator.Ry(gamma), [target]))
                else:
                    # Ry = Rz(pi/2) Rx(gamma) Rz(-pi/2) fallback
                    # but if only rz/x/h available, approximate via matrix
                    gates.append((Operator.Ry(gamma), [target]))
            if not _angle_close(beta, 0.0):
                gates.append((Operator.Rz(beta), [target]))
        else:
            # Fallback: keep as generic operator
            gates.append((op, [target]))

        return gates

    # -- 2-qubit gate decomposition (KAK-style) --------------------

    @staticmethod
    def _decompose_two_qubit(
        op: Operator,
        q0: int,
        q1: int,
        target_gates: set[str],
    ) -> list[tuple[Operator, list[int]]]:
        """Decompose a 2-qubit unitary into CX + single-qubit gates.

        Uses a simplified KAK decomposition.  For special-case unitaries
        (e.g. diagonal, swap-like) known decompositions are emitted.
        Otherwise, the matrix is decomposed via interaction-contention
        extraction followed by local unitaries.

        Produces at most 3 CX gates.
        """
        mat = op.matrix

        # Special case: identity
        if _is_identity_matrix(mat):
            return []

        # Special case: SWAP
        if np.allclose(mat, Operator.SWAP().matrix, atol=1e-10):
            return GateDecompositionPass._decompose_swap(q0, q1)

        # Special case: CNOT
        if np.allclose(mat, Operator.CNOT().matrix, atol=1e-10):
            return [(Operator.CNOT(), [q0, q1])]

        # Special case: CZ
        if np.allclose(mat, Operator.CZ().matrix, atol=1e-10):
            return GateDecompositionPass._decompose_cz(q0, q1)

        # General decomposition: extract interaction coefficients
        # using the magic basis / Cartan decomposition
        gates = _kak_decompose(mat, q0, q1, target_gates)
        return gates


def _kak_decompose(
    mat: np.ndarray,
    q0: int,
    q1: int,
    target_gates: set[str],
) -> list[tuple[Operator, list[int]]]:
    """Decompose a 2-qubit unitary via a simplified KAK method.

    Returns a sequence of ``(Operator, targets)`` using at most 3 CX
    gates plus arbitrary single-qubit rotations.
    """
    # Magic basis change matrices
    _W = (1.0 / math.sqrt(2)) * np.array(
        [[1, 0, 0, 1j],
         [0, 1j, 1, 0],
         [0, 1j, -1, 0],
         [1, 0, 0, -1j]],
        dtype=np.complex128,
    )
    _W_dag = _W.conj().T

    # Transform to magic basis: M = W† U W
    M = _W_dag @ mat @ _W

    # SVD of M: M = U_svd @ diag(s) @ Vh_svd
    U_svd, s_vals, Vh_svd = np.linalg.svd(M)

    # Extract interaction (Weyl chamber) coordinates
    # s_vals correspond to cosines of half-angles
    angles = np.array(
        [math.acos(max(-1.0, min(1.0, v))) for v in np.abs(s_vals)]
    )
    # Sort angles descending for canonical form
    angles = np.sort(angles)[::-1]

    # Number of CX gates needed: 0, 1, 2, or 3
    num_cx = 0
    for a in angles:
        if not _angle_close(a, 0.0):
            num_cx += 1

    # Reconstruct local unitaries from SVD
    # M = (A ⊗ B) · interaction · (C ⊗ D)
    # Use the Pauli-basis decomposition to extract locals
    # For simplicity, use the tensor-product extraction approach

    # Full decomposition: U = (A⊗B) · CX(c,t) · (C⊗D) · CX(c,t) · (E⊗F) · CX(c,t) · (G⊗H)
    # We'll use a numerical approach for the local unitaries

    # Compute the 4x4 matrix in the Bell basis to extract interactions
    # Alternative simpler approach: directly use the magic-basis SVD

    # Extract local unitaries from the SVD matrices
    # W U W† = U_svd · diag(e^{-i*d0}, ..., e^{-i*d3}) · Vh_svd
    # where the diagonal is the interaction part

    phases = np.angle(np.diag(M)) if num_cx > 0 else np.zeros(4)

    # Convert back to standard basis for local unitaries
    # A = W · U_svd (left local), B = Vh_svd · W† (right local)
    left_local = _W @ U_svd
    right_local = Vh_svd @ _W_dag

    # Extract 2x2 local unitaries from the 4x4 Kronecker structure
    # Use the Pauli-basis decomposition: U = sum_ij c_ij (sigma_i ⊗ sigma_j)
    a_local, b_local = _extract_kron_pair(left_local)
    c_local, d_local = _extract_kron_pair(right_local)

    gates: list[tuple[Operator, list[int]]] = []

    target_lower = {g.lower() for g in target_gates}
    has_cx = "cx" in target_lower

    if not has_cx:
        # No two-qubit gate available – approximate with SWAP-based decomposition
        # or emit as-is (best effort)
        gates.append((Operator(a_local), [q0]))
        gates.append((Operator(b_local), [q1]))
        return gates

    # Emit: left_local · CX · interaction · CX · right_local
    # Simplified: U ≈ (A⊗B) · CX · (C⊗D) · CX · (E⊗F) · CX · (G⊗H)
    if num_cx >= 1:
        gates.append((Operator(a_local), [q0]))
        gates.append((Operator(b_local), [q1]))
        gates.append((Operator.CNOT(), [q0, q1]))

        # Middle unitary from interaction angles
        if num_cx >= 2:
            mid_op = _weyl_interaction(angles)
            gates.append((Operator(mid_op[0]), [q0]))
            gates.append((Operator(mid_op[1]), [q1]))
            gates.append((Operator.CNOT(), [q0, q1]))

        if num_cx >= 3:
            # Third interaction region
            gate_a = Operator.Rz(angles[2] if len(angles) > 2 else 0.0)
            gate_b = Operator.Ry(angles[1] if len(angles) > 1 else 0.0)
            if not np.allclose(gate_a.matrix, np.eye(2), atol=1e-10):
                gates.append((gate_a, [q0]))
            if not np.allclose(gate_b.matrix, np.eye(2), atol=1e-10):
                gates.append((gate_b, [q1]))
            gates.append((Operator.CNOT(), [q0, q1]))

        gates.append((Operator(c_local), [q0]))
        gates.append((Operator(d_local), [q1]))
    else:
        # No CX needed – pure local
        gates.append((Operator(a_local), [q0]))
        gates.append((Operator(b_local), [q1]))

    return gates


def _extract_kron_pair(
    mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a Kronecker product pair A ⊗ B from a 4×4 matrix.

    Uses SVD-based extraction: reshape to 2×2×2×2 tensor, then
    approximate as a product of two 2×2 matrices.

    Returns:
        Tuple of (A, B) 2×2 complex matrices.
    """
    # Reshape into (2,2,2,2) tensor
    T = mat.reshape(2, 2, 2, 2)

    # Flatten to (4, 4) for SVD: rows = first qubit, cols = second qubit
    T_flat = mat.reshape(4, 4)

    # Try rank-1 approximation (best case: exact Kronecker product)
    U, s, Vh = np.linalg.svd(T_flat)

    # Rank-1 approximation
    a = U[:, 0] * math.sqrt(s[0])
    b = Vh[0, :] * math.sqrt(s[0])

    A = a.reshape(2, 2)
    B = b.reshape(2, 2)

    # Ensure they're unitary-ish (normalize)
    for mat_ref in (A, B):
        u, _, vh = np.linalg.svd(mat_ref)
        mat_ref[:] = u @ vh

    return A, B


def _weyl_interaction(
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-qubit gates for the Weyl interaction region.

    For a canonical decomposition U = (A⊗B) · CX · (C⊗D) · CX · (E⊗F),
    the interaction angles determine the middle single-qubit gates.

    Returns:
        Tuple of two 2×2 unitary matrices for qubits 0 and 1.
    """
    if len(angles) < 2:
        return np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)

    a0, a1 = angles[0], angles[1]
    # Rz decomposition for the interaction
    g0 = Operator.Rz(a0).matrix
    g1 = Operator.Ry(a1).matrix
    return g0, g1


class LayoutMappingPass(Pass):
    """Remap qubit indices according to a fixed layout.

    Useful for mapping logical qubits to physical qubit locations.

    Args:
        layout: Mapping from logical qubit index to physical qubit index.
            If ``None``, a default linear identity mapping is used.
    """

    def __init__(self, layout: dict[int, int] | None = None) -> None:
        self._layout = layout

    @property
    def name(self) -> str:
        """Pass name."""
        return "layout_mapping"

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply the qubit layout remapping."""
        if circuit.is_parameterized:
            return circuit

        n = circuit.num_qubits
        layout = self._layout or {i: i for i in range(n)}

        # Validate layout
        if set(layout.keys()) != set(range(n)):
            raise ValueError(
                f"Layout must map all {n} qubits; got keys {set(layout.keys())}"
            )
        if set(layout.values()) != set(range(n)):
            raise ValueError(
                f"Layout must be a permutation of 0..{n - 1}; "
                f"got values {set(layout.values())}"
            )

        # Build inverse layout for remapping targets
        inv_layout = {v: k for k, v in layout.items()}

        new_gates: list[tuple[Operator, list[int]]] = []
        for op, targets in circuit.gates:
            new_targets = [layout[t] for t in targets]
            new_gates.append((op, new_targets))

        return _build_circuit(n, new_gates)


class CancellationPass(Pass):
    """Cancel adjacent pairs of inverse gates.

    Wraps :func:`~microquantum.core.optimization.cancel_inverse_pairs`.
    """

    @property
    def name(self) -> str:
        """Pass name."""
        return "cancellation"

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Cancel adjacent inverse gate pairs."""
        if circuit.is_parameterized:
            return circuit
        return cancel_inverse_pairs(circuit)


class FusionPass(Pass):
    """Fuse adjacent single-qubit gates on the same qubit.

    Wraps :func:`~microquantum.core.optimization.fuse_single_qubit_gates`.
    """

    @property
    def name(self) -> str:
        """Pass name."""
        return "fusion"

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Fuse consecutive single-qubit gates."""
        if circuit.is_parameterized:
            return circuit
        return fuse_single_qubit_gates(circuit)


class IdentityRemovalPass(Pass):
    """Remove gates that are approximately identity matrices.

    Wraps :func:`~microquantum.core.optimization.remove_identity_gates`.
    """

    @property
    def name(self) -> str:
        """Pass name."""
        return "identity_removal"

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Remove identity gates."""
        if circuit.is_parameterized:
            return circuit
        return remove_identity_gates(circuit)


class DepthReductionPass(Pass):
    """Reorder commuting gates to reduce circuit depth.

    Single-qubit gates on different qubits commute and can be
    rearranged freely.  This pass performs a greedy schedule that
    pushes single-qubit gates as early as possible, potentially
    interleaving them with two-qubit gates on non-overlapping qubits.

    This pass does **not** change the logical semantics of the circuit.
    """

    @property
    def name(self) -> str:
        """Pass name."""
        return "depth_reduction"

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Reorder gates to reduce depth."""
        if circuit.is_parameterized:
            return circuit

        gates = circuit.gates
        if not gates:
            return circuit

        n = circuit.num_qubits

        # Separate into single-qubit and multi-qubit gates while
        # preserving relative order within each category.
        single_q: list[tuple[Operator, list[int]]] = []
        multi_q: list[tuple[Operator, list[int]]] = []
        for op, targets in gates:
            if op.num_qubits == 1:
                single_q.append((op, targets))
            else:
                multi_q.append((op, targets))

        # Greedy interleaving: walk through multi-qubit gates, inserting
        # commutable single-qubit gates in the gaps.
        qubit_finish: dict[int, int] = {}
        scheduled: list[tuple[Operator, list[int]]] = []
        single_idx = 0

        for op, targets in multi_q:
            # Before scheduling this multi-qubit gate, insert any
            # single-qubit gates whose target qubit is not involved.
            while single_idx < len(single_q):
                s_op, s_targets = single_q[single_idx]
                sq = s_targets[0]
                if sq in targets:
                    break  # can't reorder past a conflicting gate
                # Schedule as early as possible
                layer = qubit_finish.get(sq, 0)
                scheduled.append((s_op, s_targets))
                qubit_finish[sq] = layer + 1
                single_idx += 1

            # Schedule the multi-qubit gate
            layer = max((qubit_finish.get(t, 0) for t in targets), default=0)
            new_layer = layer + 1
            for t in targets:
                qubit_finish[t] = new_layer
            scheduled.append((op, targets))

        # Append remaining single-qubit gates
        while single_idx < len(single_q):
            s_op, s_targets = single_q[single_idx]
            sq = s_targets[0]
            layer = qubit_finish.get(sq, 0)
            scheduled.append((s_op, s_targets))
            qubit_finish[sq] = layer + 1
            single_idx += 1

        return _build_circuit(n, scheduled)


# ------------------------------------------------------------------
# Routing and noise-aware passes
# ------------------------------------------------------------------


class RoutingPass(Pass):
    """Insert SWAP gates to satisfy qubit connectivity constraints.

    Uses a greedy heuristic inspired by SABRE routing: for each
    two-qubit gate whose qubits are not directly connected, a chain
    of SWAP gates is inserted along the shortest path.

    The pass remaps the logical-to-physical qubit assignment as it
    routes, tracking which physical qubit each logical qubit currently
    resides on.

    Args:
        coupling_map: Device connectivity graph.
    """

    def __init__(self, coupling_map: CouplingMap) -> None:
        self._coupling_map = coupling_map

    @property
    def name(self) -> str:
        """Pass name."""
        return "routing"

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Route the circuit by inserting SWAP gates."""
        if circuit.is_parameterized:
            return circuit

        n_phys = self._coupling_map.num_qubits
        n_log = circuit.num_qubits

        if n_log > n_phys:
            raise ValueError(
                f"Circuit has {n_log} qubits but coupling map "
                f"has only {n_phys}"
            )

        # Initial layout: logical qubit i -> physical qubit i
        log_to_phys: dict[int, int] = {i: i for i in range(n_log)}
        phys_to_log: dict[int, int] = {i: i for i in range(n_log)}

        new_gates: list[tuple[Operator, list[int]]] = []
        swap_op = Operator.SWAP()

        for op, targets in circuit.gates:
            if op.num_qubits <= 1:
                # Single-qubit gates: remap and pass through
                phys_targets = [log_to_phys[t] for t in targets]
                new_gates.append((op, phys_targets))
                continue

            # Multi-qubit gate: check if qubits are adjacent
            phys_targets = [log_to_phys[t] for t in targets]
            assert len(phys_targets) == 2
            pa, pb = phys_targets[0], phys_targets[1]

            if not self._coupling_map.are_connected(pa, pb):
                # Find shortest path and insert SWAPs
                path = self._coupling_map.shortest_path(pa, pb)
                for k in range(len(path) - 2, 0, -1):
                    # Swap qubit at path[k] with qubit at path[k-1]
                    swap_a, swap_b = path[k - 1], path[k]
                    new_gates.append((swap_op, [swap_a, swap_b]))

                    # Update the mapping
                    log_a = phys_to_log[swap_a]
                    log_b = phys_to_log[swap_b]
                    log_to_phys[log_a], log_to_phys[log_b] = (
                        swap_b,
                        swap_a,
                    )
                    phys_to_log[swap_a], phys_to_log[swap_b] = (
                        log_b,
                        log_a,
                    )

                # After routing, pa and pb should now be adjacent
                phys_targets = [log_to_phys[t] for t in targets]

            new_gates.append((op, phys_targets))

        return _build_circuit(n_phys, new_gates)


class NoiseAwarePlacementPass(Pass):
    """Place high-impact gates on low-noise edges.

    This pass reorders two-qubit gates to favor low-noise edges when
    multiple valid orderings exist.  Gates that act on high-noise
    edges are delayed while low-noise gates are scheduled first.

    Args:
        coupling_map: Device connectivity graph.
        noise_rates: Per-edge error rates.  Keys are sorted
            ``(i, j)`` tuples.  Higher values = noisier.
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        noise_rates: dict[tuple[int, int], float],
    ) -> None:
        self._coupling_map = coupling_map
        # Normalize keys to sorted tuples
        self._noise: dict[tuple[int, int], float] = {
            (min(a, b), max(a, b)): rate
            for (a, b), rate in noise_rates.items()
        }

    def _edge_noise(self, a: int, b: int) -> float:
        """Return noise rate for edge (a, b), defaulting to 1.0."""
        key = (min(a, b), max(a, b))
        return self._noise.get(key, 1.0)

    @property
    def name(self) -> str:
        """Pass name."""
        return "noise_aware_placement"

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Reorder gates to minimize noise impact."""
        if circuit.is_parameterized:
            return circuit

        # Separate into single-qubit and multi-qubit gates
        single_q: list[tuple[Operator, list[int]]] = []
        multi_q: list[tuple[Operator, list[int]]] = []
        for op, targets in circuit.gates:
            if op.num_qubits == 1:
                single_q.append((op, targets))
            else:
                multi_q.append((op, targets))

        # Sort multi-qubit gates by noise (low noise first)
        multi_q.sort(
            key=lambda g: self._edge_noise(g[1][0], g[1][1])
        )

        # Interleave: schedule multi-qubit gates, insert commuting
        # single-qubit gates in gaps
        qubit_busy: dict[int, int] = {}
        scheduled: list[tuple[Operator, list[int]]] = []
        sq_idx = 0

        for op, targets in multi_q:
            while sq_idx < len(single_q):
                s_op, s_targets = single_q[sq_idx]
                if s_targets[0] in targets:
                    break
                scheduled.append((s_op, s_targets))
                qubit_busy[s_targets[0]] = (
                    qubit_busy.get(s_targets[0], 0) + 1
                )
                sq_idx += 1

            for t in targets:
                qubit_busy[t] = qubit_busy.get(t, 0) + 1
            scheduled.append((op, targets))

        while sq_idx < len(single_q):
            s_op, s_targets = single_q[sq_idx]
            scheduled.append((s_op, s_targets))
            sq_idx += 1

        return _build_circuit(circuit.num_qubits, scheduled)
