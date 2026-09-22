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
