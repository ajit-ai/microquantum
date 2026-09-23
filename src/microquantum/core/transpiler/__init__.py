"""Proper quantum compilation framework: passes, context and manager.

Architecture-aware but provider-independent::

    QuantumCircuit → Validation → Analysis → Layout → Routing
        → Basis decomposition → Optimization → Scheduling → executable

The canonical pass implementations live in
``microquantum.core.transpiler._model`` (moved from the stable 1.0.0
flat module) and are re-exported unchanged.  This package adds the
``AnalysisPass`` / ``TransformationPass`` / ``PassContext`` model plus
new measurable passes (validation, basis translation, inverse
cancellation, gate fusion, layout, scheduling) composed by
:func:`default_pipeline` and :func:`transpile_with`.  Every pass has a
measurable, testable responsibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ._model import (
    CancellationPass,
    DepthReductionPass,
    FusionPass,
    GateDecompositionPass,
    IdentityRemovalPass,
    LayoutMappingPass,
    NoiseAwarePlacementPass,
    Pass,
    PassManager,
    RoutingPass,
    TargetGateSet,
)

__all__ = [
    "Pass",
    "PassManager",
    "TargetGateSet",
    "GateDecompositionPass",
    "LayoutMappingPass",
    "CancellationPass",
    "FusionPass",
    "IdentityRemovalPass",
    "DepthReductionPass",
    "RoutingPass",
    "NoiseAwarePlacementPass",
    "AnalysisPass",
    "TransformationPass",
    "PassContext",
    "ValidationPass",
    "BasisTranslationPass",
    "InverseCancellationPass",
    "AdjacentCancellationPass",
    "GateFusionPass",
    "LayoutPass",
    "SchedulingPass",
    "SwapRoutingPass",
    "NoiseAwareLayout",
    "CommutationAwareCancellation",
    "default_pipeline",
    "transpile_with",
]


@dataclass
class PassContext:
    """Mutable context shared across passes in one pipeline run."""

    architecture: Any = None
    target_gates: TargetGateSet | None = None
    layout: dict[int, int] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def record(self, key: str, value: Any) -> None:
        """Record analysis data under ``key``."""
        self.analysis[key] = value


class AnalysisPass(Pass, ABC):
    """A pass that only reads the circuit and records analysis."""

    @abstractmethod
    def analyze(self, circuit: Any, context: PassContext) -> Any:
        """Analyze *circuit*, returning it unmodified."""

    def run(self, circuit: Any) -> Any:
        """Run analysis with a fresh context."""
        self.analyze(circuit, PassContext())
        return circuit


class TransformationPass(Pass, ABC):
    """A pass that returns a transformed circuit."""

    @abstractmethod
    def transform(self, circuit: Any, context: PassContext) -> Any:
        """Transform *circuit*."""

    def run(self, circuit: Any) -> Any:
        """Run the transformation with a fresh context."""
        return self.transform(circuit, PassContext())


class ValidationPass(AnalysisPass):
    """Validate qubit indices, gate arities and measurement targets."""

    @property
    def name(self) -> str:
        return "validation"

    def analyze(self, circuit: Any, context: PassContext) -> Any:
        # Import here: the sibling ``circuit`` package is fully loaded by
        # the time any pass executes.
        from microquantum.core.circuit import (  # noqa: PLC0415
            QuantumCircuit,
            _narrow_concrete,
        )

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(f"Expected QuantumCircuit, got {type(circuit).__name__}")
        n = circuit.num_qubits
        for instr in circuit._gate_instructions:  # noqa: SLF001
            if circuit._is_parameterized_gate(instr):  # noqa: SLF001
                from microquantum.core.circuit import _narrow_parameterized  # noqa: PLC0415

                _, _, target = _narrow_parameterized(instr)
                targets = [int(target)]
            else:
                _, concrete_targets = _narrow_concrete(instr)
                targets = [int(t) for t in concrete_targets]
            for t in targets:
                if not 0 <= t < n:
                    raise ValueError(f"Qubit index {t} out of range for {n} qubits")
        for m in circuit._measurements:  # noqa: SLF001
            if not 0 <= int(m) < n:
                raise ValueError(f"Measurement qubit {m} out of range for {n} qubits")
        context.record("valid", True)
        context.record("num_qubits", n)
        context.record("num_gates", circuit.num_gates)
        return circuit


class BasisTranslationPass(TransformationPass):
    """Translate single-qubit X/Y/Z/S/T gates into RZ rotations.

    Measures the number of translated gates in the pass context under
    ``"translated"``.  Two-qubit gates and parameterized rotations pass
    through unchanged.
    """

    @property
    def name(self) -> str:
        return "basis-translation"

    def transform(self, circuit: Any, context: PassContext) -> Any:
        import math as _math  # noqa: PLC0415

        from microquantum.core.circuit import (  # noqa: PLC0415
            QuantumCircuit,
            _narrow_concrete,
        )

        _angles = {
            "x": _math.pi,
            "y": _math.pi,
            "z": _math.pi,
            "s": _math.pi / 2,
            "sdg": -_math.pi / 2,
            "t": _math.pi / 4,
            "tdg": -_math.pi / 4,
        }
        out = QuantumCircuit(circuit.num_qubits)
        translated = 0
        for instr in circuit._gate_instructions:  # noqa: SLF001
            if circuit._is_parameterized_gate(instr):  # noqa: SLF001
                from microquantum.core.circuit import _narrow_parameterized  # noqa: PLC0415

                gname, param, target = _narrow_parameterized(instr)
                getattr(out, str(gname))(param, int(target))
                continue
            op, targets = _narrow_concrete(instr)
            lname = op.name.lower()
            if lname in _angles:
                out.rz(_angles[lname], int(targets[0]))
                translated += 1
            elif lname == "i":
                translated += 1  # identity dropped
            elif hasattr(out, lname):
                getattr(out, lname)(*[int(t) for t in targets])
            else:
                out.append(op, [int(t) for t in targets])
        for m in circuit._measurements:  # noqa: SLF001
            out.measure(int(m))
        context.record("translated", translated)
        return out


class InverseCancellationPass(TransformationPass):
    """Cancel adjacent self-inverse gate pairs on identical targets.

    Records the cancelled-pair count in the context under
    ``"cancelled_pairs"``.
    """

    _SELF_INVERSE = frozenset({"h", "x", "y", "z", "cx", "cnot", "cy", "cz", "swap"})

    @property
    def name(self) -> str:
        return "inverse-cancellation"

    def transform(self, circuit: Any, context: PassContext) -> Any:
        from microquantum.core.circuit import (  # noqa: PLC0415
            QuantumCircuit,
            _narrow_concrete,
        )

        out = QuantumCircuit(circuit.num_qubits)
        pending: list[tuple[Any, list[int]]] = []
        cancelled = 0
        for instr in circuit._gate_instructions:  # noqa: SLF001
            if circuit._is_parameterized_gate(instr):  # noqa: SLF001
                for op, tg in pending:
                    out.append(op, tg)
                pending.clear()
                from microquantum.core.circuit import _narrow_parameterized  # noqa: PLC0415

                gname, param, target = _narrow_parameterized(instr)
                getattr(out, str(gname))(param, int(target))
                continue
            op, targets = _narrow_concrete(instr)
            key = (op.name.lower(), [int(t) for t in targets])
            if (
                pending
                and pending[-1][0].name.lower() == key[0]
                and pending[-1][1] == key[1]
                and key[0] in self._SELF_INVERSE
            ):
                pending.pop()
                cancelled += 1
            else:
                pending.append((op, key[1]))
        for op, tg in pending:
            out.append(op, tg)
        for m in circuit._measurements:  # noqa: SLF001
            out.measure(int(m))
        context.record("cancelled_pairs", cancelled)
        return out


AdjacentCancellationPass = InverseCancellationPass


class GateFusionPass(TransformationPass):
    """Fuse consecutive single-qubit gates on the same qubit.

    Records the fused-group count in the context under
    ``"fused_groups"``.
    """

    @property
    def name(self) -> str:
        return "gate-fusion"

    def transform(self, circuit: Any, context: PassContext) -> Any:
        from microquantum.core.circuit import (  # noqa: PLC0415
            QuantumCircuit,
            _narrow_concrete,
        )

        out = QuantumCircuit(circuit.num_qubits)
        pending: dict[int, Any] = {}
        groups = 0

        def flush(qubit: int) -> None:
            if qubit in pending:
                out.append(pending.pop(qubit), [qubit])

        for instr in circuit._gate_instructions:  # noqa: SLF001
            if circuit._is_parameterized_gate(instr):  # noqa: SLF001
                for q in list(pending):
                    flush(q)
                from microquantum.core.circuit import _narrow_parameterized  # noqa: PLC0415

                gname, param, target = _narrow_parameterized(instr)
                getattr(out, str(gname))(param, int(target))
                continue
            op, targets = _narrow_concrete(instr)
            if len(targets) == 1 and op.name.lower() not in ("measure", "barrier", "reset"):
                q = int(targets[0])
                pending[q] = op @ pending[q] if q in pending else op
            else:
                for q in list(pending):
                    flush(q)
                out.append(op, [int(t) for t in targets])
        for q in list(pending):
            flush(q)
            groups += 1
        for m in circuit._measurements:  # noqa: SLF001
            out.measure(int(m))
        context.record("fused_groups", groups)
        return out


class LayoutPass(TransformationPass):
    """Identity layout: record virtual→physical mapping in the context."""

    @property
    def name(self) -> str:
        return "layout"

    def transform(self, circuit: Any, context: PassContext) -> Any:
        arch = context.architecture
        if arch is not None and hasattr(arch, "validate_circuit_qubits"):
            arch.validate_circuit_qubits(circuit.num_qubits)
        context.layout = {i: i for i in range(circuit.num_qubits)}
        context.record("layout", dict(context.layout))
        return circuit


class SchedulingPass(AnalysisPass):
    """Scheduling analysis: record depth and gate count."""

    @property
    def name(self) -> str:
        return "scheduling"

    def analyze(self, circuit: Any, context: PassContext) -> Any:
        context.record("depth", circuit.depth())
        context.record("num_gates", circuit.num_gates)
        return circuit


def _bfs_path(
    adjacency: dict[int, set[int]], source: int, target: int
) -> list[int]:
    """Shortest path (inclusive) or empty list when disconnected."""
    if source == target:
        return [source]
    visited: dict[int, int | None] = {source: None}
    queue = [source]
    while queue:
        node = queue.pop(0)
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in visited:
                visited[neighbor] = node
                if neighbor == target:
                    path = [target]
                    while path[-1] != source:
                        previous = visited[path[-1]]
                        assert previous is not None
                        path.append(previous)
                    return list(reversed(path))
                queue.append(neighbor)
    return []


class SwapRoutingPass(TransformationPass):
    """Route two-qubit gates onto device connectivity with SWAPs.

    Greedily walks each non-adjacent pair together along a shortest
    path, inserting logical SWAPs while tracking the logical→slot
    permutation.  Output satisfies ``U_routed = M · U_orig`` with ``M``
    the logical-to-slot basis permutation from ``final_layout``
    (initial mapping is identity, so readout remaps by ``final_layout``;
    measurement indices are rewritten to physical slots accordingly).
    The inserted-SWAP count is recorded as ``"swaps_added"`` and the
    final mapping as ``"final_layout"`` (``{logical: slot}``).

    Args:
        architecture: Hardware-independent architecture providing a
            ``topology`` with ``edges`` (or a ``to_coupling_map()``).
        coupling: Explicit edge list alternative to *architecture*.
    """

    def __init__(
        self, architecture: Any = None, coupling: list[tuple[int, int]] | None = None
    ) -> None:
        self._architecture = architecture
        self._coupling = list(coupling) if coupling is not None else None

    @property
    def name(self) -> str:
        return "swap-routing"

    def _edges(self) -> list[tuple[int, int]]:
        """Resolve the connectivity edge list."""
        if self._coupling is not None:
            return [(int(a), int(b)) for a, b in self._coupling]
        arch = self._architecture
        if arch is None:
            return []
        topo = getattr(arch, "topology", None)
        raw = getattr(topo, "edges", None) if topo is not None else None
        if raw:
            return [(int(a), int(b)) for a, b in raw]
        return []

    def transform(self, circuit: Any, context: PassContext) -> Any:
        from microquantum.core.circuit import (  # noqa: PLC0415
            QuantumCircuit,
            _narrow_concrete,
        )

        edges = self._edges()
        if not edges:
            context.record("swaps_added", 0)
            return circuit
        if self._architecture is not None and hasattr(
            self._architecture, "validate_circuit_qubits"
        ):
            self._architecture.validate_circuit_qubits(circuit.num_qubits)
        adjacency: dict[int, set[int]] = {}
        for first, second in edges:
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        out = QuantumCircuit(circuit.num_qubits)
        at = list(range(circuit.num_qubits))
        loc = list(range(circuit.num_qubits))
        swaps = 0

        def emit_swap(slot_a: int, slot_b: int) -> None:
            """Emit SWAP on slots and update the permutation."""
            nonlocal swaps
            out.swap(slot_a, slot_b)
            logical_a, logical_b = at[slot_a], at[slot_b]
            at[slot_a], at[slot_b] = logical_b, logical_a
            loc[logical_a], loc[logical_b] = slot_b, slot_a
            swaps += 1

        def emit_concrete(op: Any, slots: list[int]) -> None:
            """Emit a concrete gate on physical slots."""
            name = op.name.lower()
            if hasattr(out, name):
                getattr(out, name)(*[int(s) for s in slots])
            else:
                out.append(op, [int(s) for s in slots])

        for instr in circuit._gate_instructions:  # noqa: SLF001
            if circuit._is_parameterized_gate(instr):  # noqa: SLF001
                from microquantum.core.circuit import _narrow_parameterized  # noqa: PLC0415

                gate_name, param, target = _narrow_parameterized(instr)
                getattr(out, str(gate_name))(param, int(loc[int(target)]))
                continue
            op, targets = _narrow_concrete(instr)
            logical = [int(t) for t in targets]
            if len(logical) == 1:
                emit_concrete(op, [loc[logical[0]]])
            elif len(logical) == 2:
                slot_a, slot_b = loc[logical[0]], loc[logical[1]]
                if slot_b in adjacency.get(slot_a, ()):
                    emit_concrete(op, [slot_a, slot_b])
                else:
                    path = _bfs_path(adjacency, slot_a, slot_b)
                    if len(path) < 2:
                        raise ValueError(
                            f"No routing path between qubits {slot_a} and {slot_b}"
                        )
                    # Walk `a` adjacent to `b` without crossing it: all
                    # steps except the last, so `b` keeps its slot.
                    for step in range(len(path) - 2):
                        emit_swap(path[step], path[step + 1])
                    emit_concrete(op, [loc[logical[0]], loc[logical[1]]])
            else:
                emit_concrete(op, [loc[t] for t in logical])
        for m in circuit._measurements:  # noqa: SLF001
            out.measure(int(loc[int(m)]))
        context.record("swaps_added", swaps)
        context.record("final_layout", {logical: slot for logical, slot in enumerate(loc)})
        return out


class NoiseAwareLayout(TransformationPass):
    """Assign logical qubits to physical qubits by error rates.

    Greedily places the most-measured logical qubits onto the physical
    qubits with the lowest readout error (ties broken by gate-error
    average, then index).  The circuit passes through unchanged; the
    decision — the measurable product — is recorded as ``"layout"``
    (``{logical: physical}``) plus ``"estimated_readout_error"``.

    Args:
        architecture: Architecture with ``num_qubits`` (topology edges
            are not required for placement).
        calibration: Readout/gate error mapping or a
            ``CalibrationData`` object (``readout_errors`` / 
            ``gate_errors`` attributes).  Missing entries default to 0.
    """

    def __init__(self, architecture: Any, calibration: Any = None) -> None:
        if architecture is None:
            raise ValueError("NoiseAwareLayout requires an architecture")
        self._architecture = architecture
        self._calibration = calibration

    @property
    def name(self) -> str:
        return "noise-aware-layout"

    def _readout_error(self, physical: int) -> float:
        """Readout error for a physical qubit (0 when unknown)."""
        calibration = self._calibration
        if calibration is None:
            return 0.0
        table = getattr(calibration, "readout_errors", None)
        if isinstance(table, dict):
            for key in (f"q{physical}", str(physical), physical):
                if key in table:
                    value = table[key]
                    return float(value) if isinstance(value, (int, float)) else 0.0
            return 0.0
        if isinstance(calibration, dict):
            raw = calibration.get(f"q{physical}", calibration.get(physical, 0.0))
            return float(raw) if isinstance(raw, (int, float)) else 0.0
        return 0.0

    def transform(self, circuit: Any, context: PassContext) -> Any:
        if hasattr(self._architecture, "validate_circuit_qubits"):
            self._architecture.validate_circuit_qubits(circuit.num_qubits)
        num_physical = int(self._architecture.num_qubits)
        measurements = list(getattr(circuit, "_measurements", []) or [])
        pressure = [0] * circuit.num_qubits
        for m in measurements:
            pressure[int(m)] += 10
        for instr in circuit._gate_instructions:  # noqa: SLF001
            for target in circuit._get_targets(instr):  # noqa: SLF001
                pressure[int(target)] += 1
        logical_order = sorted(range(circuit.num_qubits), key=lambda q: (-pressure[q], q))
        physical_order = sorted(
            range(num_physical), key=lambda p: (self._readout_error(p), p)
        )
        layout = {logical: physical_order[i] for i, logical in enumerate(logical_order)}
        context.layout = dict(layout)
        context.record("layout", dict(layout))
        context.record(
            "estimated_readout_error",
            sum(self._readout_error(layout[q]) for q in range(circuit.num_qubits) if q in measurements),
        )
        return circuit


# (two-qubit gate, single-qubit gate, allowed position) commutation facts.
# Position is "control"/"target" for asymmetric gates, "either" for symmetric.
_COMMUTING_SINGLES: set[tuple[str, str, str]] = {
    ("cx", "x", "target"),
    ("cnot", "x", "target"),
    ("cx", "z", "control"),
    ("cnot", "z", "control"),
    ("cz", "z", "either"),
}


class CommutationAwareCancellation(TransformationPass):
    """Cancel inverses separated only by commuting gates.

    Slides commuting single-qubit gates (X past a CX target, Z past a
    CX control or either CZ side) until self-inverse pairs become
    adjacent, then cancels them.  Records ``"slid"`` (slides performed)
    and ``"cancelled_pairs"``.  Parameterized and >2-qubit gates block
    sliding conservatively.
    """

    _SELF_INVERSE = frozenset({"h", "x", "y", "z", "cx", "cnot", "cy", "cz", "swap"})

    @property
    def name(self) -> str:
        return "commutation-aware-cancellation"

    @staticmethod
    def _key(
        instr: Any, narrow_concrete: Any, narrow_parameterized: Any, is_parameterized: Any
    ) -> tuple[str, tuple[int, ...], bool] | None:
        """Normalize an instruction to (kind, targets, is_single_concrete)."""
        if is_parameterized(instr):
            name, _, target = narrow_parameterized(instr)
            return (f"p:{name}", (int(target),), False)
        op, targets = narrow_concrete(instr)
        return (op.name.lower(), tuple(int(t) for t in targets), len(targets) == 1)

    @classmethod
    def _commutes(cls, single: str, gate: str, targets: tuple[int, ...], qubit: int) -> bool:
        """True when single-qubit *single* on *qubit* commutes with 2q *gate*."""
        if len(targets) != 2:
            return False
        if (gate, single, "either") in _COMMUTING_SINGLES:
            return True
        position = "control" if targets[0] == qubit else "target" if targets[1] == qubit else ""
        return bool(position) and (gate, single, position) in _COMMUTING_SINGLES

    def transform(self, circuit: Any, context: PassContext) -> Any:
        from microquantum.core.circuit import (  # noqa: PLC0415
            QuantumCircuit,
            _narrow_concrete,
            _narrow_parameterized,
        )

        items: list[Any] = list(circuit._gate_instructions)  # noqa: SLF001
        slid = 0
        is_param = circuit._is_parameterized_gate  # noqa: SLF001
        for _ in range(len(items) + 1):
            moved = False
            index = 0
            while index < len(items) - 1:
                first = self._key(items[index], _narrow_concrete, _narrow_parameterized, is_param)
                second = self._key(items[index + 1], _narrow_concrete, _narrow_parameterized, is_param)
                if (
                    first is not None
                    and second is not None
                    and first[2]
                    and not second[0].startswith("p:")
                    and len(second[1]) == 2
                    and first[1][0] in second[1]
                    and self._commutes(first[0], second[0], second[1], first[1][0])
                ):
                    items[index], items[index + 1] = items[index + 1], items[index]
                    slid += 1
                    moved = True
                index += 1
            if not moved:
                break
        out = QuantumCircuit(circuit.num_qubits)
        pending: list[tuple[Any, list[int]]] = []
        cancelled = 0
        for instr in items:
            if is_param(instr):
                for op, tg in pending:
                    out.append(op, tg)
                pending.clear()
                gate_name, param, target = _narrow_parameterized(instr)
                getattr(out, str(gate_name))(param, int(target))
                continue
            op, targets = _narrow_concrete(instr)
            key = (op.name.lower(), [int(t) for t in targets])
            if (
                pending
                and pending[-1][0].name.lower() == key[0]
                and pending[-1][1] == key[1]
                and key[0] in self._SELF_INVERSE
            ):
                pending.pop()
                cancelled += 1
            else:
                pending.append((op, key[1]))
        for op, tg in pending:
            out.append(op, tg)
        for m in circuit._measurements:  # noqa: SLF001
            out.measure(int(m))
        context.record("slid", slid)
        context.record("cancelled_pairs", cancelled)
        return out


def default_pipeline(target: TargetGateSet | None = None) -> PassManager:
    """Build the default compilation pipeline.

    Validation → identity removal → inverse cancellation → gate
    decomposition → basis translation → scheduling analysis.
    """
    manager = PassManager()
    manager.append_pass(ValidationPass())
    manager.append_pass(IdentityRemovalPass())
    manager.append_pass(CancellationPass())
    manager.append_pass(GateDecompositionPass(target))
    manager.append_pass(BasisTranslationPass())
    manager.append_pass(SchedulingPass())
    return manager


def transpile_with(circuit: Any, target: TargetGateSet | None = None, architecture: Any = None) -> Any:
    """Transpile *circuit* through validation, optimization, layout,
    architecture-aware routing (when *architecture* provides edges) and
    basis decomposition, returning the executable circuit.
    """
    current = ValidationPass().run(circuit)
    current = IdentityRemovalPass().run(current)
    current = CancellationPass().run(current)
    context = PassContext(architecture=architecture, target_gates=target or TargetGateSet.default())
    current = LayoutPass().transform(current, context)
    if architecture is not None:
        edges: list[tuple[int, int]] = []
        topo = getattr(architecture, "topology", None)
        raw_edges: Any = getattr(topo, "edges", None) if topo is not None else None
        if raw_edges:
            edges = [(int(a), int(b)) for a, b in raw_edges]
        elif hasattr(architecture, "to_coupling_map"):
            cmap = architecture.to_coupling_map() if hasattr(topo, "to_coupling_map") else None
            if cmap is not None:
                edges = [(int(a), int(b)) for a, b in getattr(cmap, "edges", [])]
        if edges:
            from microquantum.core.coupling import CouplingMap  # noqa: PLC0415

            current = RoutingPass(CouplingMap(edges)).run(current)
            context.record("routed", True)
        else:
            context.record("routed", False)
    current = GateDecompositionPass(target).run(current)
    current = BasisTranslationPass().run(current)
    SchedulingPass().run(current)
    return current
