"""Target-aware compilation foundation.

Connects MicroQuantum IR to the MQ-02 :class:`~microquantum.core.device.Target`
contract::

    Circuit / IR
        └── Compiler
              ├── validation
              ├── optimization passes
              ├── gate decomposition toward the target basis
              └── compatibility diagnostics
        └── CompilationResult (compiled IR + target + applied passes)

The foundation deliberately stops before hardware: no vendor APIs, no QPU
control, no cloud orchestration.  Future phases plug routing, scheduling
and platform execution downstream of :class:`Compiler`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .._json import JSONSerializable, json_safe, json_string
from ..core.device import Target
from .builder import from_ir, to_ir
from .circuit_ir import IRCircuit
from .cost import CostModel
from .nodes import (
    ConditionalBlock,
    Gate,
    Measurement,
    Reset,
)
from .passes import (
    CancelAdjacentInverse,
    CombineRotations,
    GateDecomposition,
    IRPass,
    IRPassManager,
    RemoveIdentityGates,
)
from .validation import assert_valid

_OPTIMIZATION_PASSES = [
    RemoveIdentityGates,
    CancelAdjacentInverse,
    CombineRotations,
]


def _normalize_basis(basis: set[str]) -> set[str]:
    """Normalize a native-gate name set to MicroQuantum IR names.

    The SDK's public :class:`~microquantum.core.device.Target`
    ``native_gates`` may spell the controlled-NOT as ``"cx"`` while the IR
    and passes use ``"cnot"``; treat them as the same gate so target-aware
    decomposition and diagnostics behave consistently.
    """
    return {("cnot" if g.lower() == "cx" else g.lower()) for g in basis}


def _compat_diagnostics(ir: IRCircuit, target: Target) -> list[str]:
    """Return a list of target-compatibility diagnostics for *ir*."""
    problems: list[str] = []
    basis = _normalize_basis(set(target.native_gates))
    if target.num_qubits is not None and ir.num_qubits > target.num_qubits:
        problems.append(
            f"circuit uses {ir.num_qubits} qubits but target '{target.name}' "
            f"supports at most {target.num_qubits}"
        )
    for op in ir.walk():
        if isinstance(op, Gate):
            if basis and op.name not in basis:
                problems.append(
                    f"gate '{op.name}' on qubits {tuple(op.qubits)} is not in "
                    f"target basis {sorted(basis)}"
                )
        elif isinstance(op, Measurement) and not target.supports_measurement:
            problems.append("circuit measures but target does not support measurement")
        elif isinstance(op, Reset) and not target.supports_dynamic_circuits:
            problems.append("circuit resets qubits but target does not support dynamic circuits")
        elif isinstance(op, ConditionalBlock) and not target.supports_dynamic_circuits:
            problems.append("circuit is classically conditioned but target does not support dynamic circuits")
    return problems


@dataclass
class CompilationResult(JSONSerializable):
    """Outcome of compiling an IR circuit against a target.

    Integrates with the SDK's result serialization contract
    (``to_dict()`` / ``to_json()``).

    Attributes:
        source: The input IR (pre-compilation).
        result: The compiled IR.
        target: The target the IR was compiled for (or None).
        passes_applied: Names of the passes executed, in order.
        diagnostics: Target-compatibility warnings/errors.
        mapping: Optional initial-layout qubit mapping
            (logical -> physical) when one is applied.
        metadata: Free-form compilation metadata.
    """

    source: IRCircuit
    result: IRCircuit
    target: Optional[Target] = None
    passes_applied: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    mapping: Optional[dict[int, int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_compatible(self) -> bool:
        """True when the compiled IR produced no target diagnostics."""
        return not self.diagnostics

    def circuit(self) -> Any:
        """Rebuild an executable :class:`QuantumCircuit` from the compiled IR.

        Raises:
            ValueError: If the compiled IR contains reset or conditional
                nodes (not representable in a static circuit).
        """
        return from_ir(self.result)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "source": self.source.to_dict(),
            "result": self.result.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
            "passes_applied": list(self.passes_applied),
            "diagnostics": list(self.diagnostics),
            "mapping": json_safe(self.mapping),
            "metadata": json_safe(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        status = "compatible" if self.is_compatible else f"{len(self.diagnostics)} diagnostics"
        return (
            f"CompilationResult(passes={self.passes_applied}, "
            f"status={status})"
        )


class Compiler:
    """Compile circuits/IR toward a :class:`Target`.

    The pipeline is: validate -> optimize -> (decompose toward the target
    basis if a target is supplied) -> compatibility diagnostics.

    Args:
        optimization_level: 0 = validation only (no passes), 1 = identity
            removal + inverse cancellation, 2 = also fuse same-axis
            rotations.  Levels outside 0..2 are rejected.
        cost_model: Optional :class:`CostModel` used to record an
            ``estimated_cost`` entry in the result metadata.
    """

    def __init__(
        self, optimization_level: int = 1, cost_model: Optional[CostModel] = None
    ) -> None:
        if not 0 <= optimization_level <= 2:
            raise ValueError(
                f"optimization_level must be in 0..2, got {optimization_level}"
            )
        self._optimization_level = optimization_level
        self._cost_model = cost_model

    @property
    def optimization_level(self) -> int:
        """Optimization level configured for this compiler."""
        return self._optimization_level

    @property
    def cost_model(self) -> Optional[CostModel]:
        """Cost model configured for this compiler (if any)."""
        return self._cost_model

    def compile(
        self,
        input_: Union[IRCircuit, Any],
        target: Optional[Target] = None,
        passes: Optional[list[IRPass]] = None,
        mapping: Optional[dict[int, int]] = None,
    ) -> CompilationResult:
        """Compile a circuit or IR to optimized IR for *target*.

        Args:
            input_: A :class:`QuantumCircuit` or :class:`IRCircuit` to compile.
            target: Optional execution target; enables basis decomposition
                and compatibility diagnostics.
            passes: Optional extra passes run after the standard pipeline.
            mapping: Optional initial-layout qubit mapping to record.

        Returns:
            A :class:`CompilationResult` holding source IR, compiled IR,
            applied passes and diagnostics.  ``result.circuit()`` rebuilds an
            executable :class:`QuantumCircuit`; terminal measurements present
            in the source are preserved through compilation and rebuilding.
        """
        if isinstance(input_, IRCircuit):
            source = input_
        else:
            source = to_ir(input_, include_terminal_measurements=True)

        assert_valid(source)

        pipeline: list[IRPass] = []
        if self._optimization_level >= 1:
            pipeline.append(RemoveIdentityGates())
        if self._optimization_level >= 1:
            pipeline.append(CancelAdjacentInverse())
        if self._optimization_level >= 2:
            pipeline.append(CombineRotations())
        if target is not None:
            pipeline.append(GateDecomposition(target.native_gates))
        if passes:
            pipeline.extend(passes)

        manager = IRPassManager(pipeline)
        compiled = manager.run(source)

        diagnostics: list[str] = []
        if target is not None:
            diagnostics = _compat_diagnostics(compiled, target)

        metadata = _compilation_metadata(source, compiled, self._optimization_level)
        if self._cost_model is not None:
            metadata["estimated_cost"] = self._cost_model.total(compiled)
            metadata["cost_breakdown"] = self._cost_model.breakdown(compiled)

        return CompilationResult(
            source=source,
            result=compiled,
            target=target,
            passes_applied=[p.name for p in pipeline],
            diagnostics=diagnostics,
            mapping=mapping,
            metadata=metadata,
        )


def _compilation_metadata(
    source: IRCircuit,
    compiled: IRCircuit,
    optimization_level: int,
) -> dict[str, Any]:
    """Build the JSON-safe compilation metadata payload."""
    def gate_counts(ir: IRCircuit) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in ir.walk():
            if isinstance(op, Gate):
                counts[op.name] = counts.get(op.name, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "optimization_level": optimization_level,
        "source_qubits": source.num_qubits,
        "source_gates": source.num_gates,
        "source_depth": source.depth,
        "compiled_qubits": compiled.num_qubits,
        "compiled_gates": compiled.num_gates,
        "compiled_depth": compiled.depth,
        "source_gates_by_type": gate_counts(source),
        "compiled_gates_by_type": gate_counts(compiled),
    }


__all__ = ["Compiler", "CompilationResult"]