"""MicroQuantum IR containers: IRCircuit and IRModule.

An :class:`IRCircuit` is an ordered list of :class:`IRNode` operations plus
the metadata needed to inspect it (qubits, classical bits, parameters,
measurements, depth).  An :class:`IRModule` groups one or more circuits —
the top-level container for multi-block workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .._json import json_safe, json_string
from ..core.parameter import Parameter
from .nodes import (
    Barrier,
    Condition,
    ConditionalBlock,
    Gate,
    IRNode,
    IRParam,
    Measurement,
    Reset,
)


@dataclass
class IRCircuit:
    """An ordered sequence of IR operations.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        num_classical_bits: Number of classical bits (for measurements
            and conditions).  Defaults to ``num_qubits``.
        name: Circuit identifier.
        operations: Ordered IR operations.
        metadata: Free-form circuit metadata (kept JSON-safe).
    """

    num_qubits: int
    num_classical_bits: Optional[int] = None
    name: str = "main"
    operations: list[IRNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        if self.num_classical_bits is None:
            self.num_classical_bits = self.num_qubits
        if self.num_classical_bits < 0:
            raise ValueError(
                f"num_classical_bits must be >= 0, got {self.num_classical_bits}"
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add(self, node: IRNode) -> "IRCircuit":
        """Append an IR node and return self for chaining."""
        if not isinstance(node, IRNode):
            raise TypeError(f"Expected IRNode, got {type(node).__name__}")
        self.operations.append(node)
        return self

    # ------------------------------------------------------------------
    # Traversal / inspection
    # ------------------------------------------------------------------

    def walk(self) -> Iterator[IRNode]:
        """Iterate operations in order, descending into conditional blocks."""
        for op in self.operations:
            yield op
            if isinstance(op, ConditionalBlock):
                for nested in op.operations:
                    yield nested

    @property
    def qubits(self) -> set[int]:
        """Set of qubit indices referenced by the circuit."""
        used: set[int] = set()
        for op in self.walk():
            used.update(getattr(op, "qubits", ()))
        return used

    @property
    def cbits(self) -> set[int]:
        """Set of classical bit indices referenced by the circuit."""
        used: set[int] = set()
        for op in self.walk():
            used.update(op.cbits)
        return used

    @property
    def parameters(self) -> set[Parameter]:
        """Set of unbound symbolic parameters in the circuit."""
        params: set[Parameter] = set()
        for op in self.walk():
            if not isinstance(op, Gate):
                continue
            for p in op.params:
                if isinstance(p, Parameter):
                    params.add(p)
                elif hasattr(p, "parameter"):
                    params.add(p.parameter)
        return params

    @property
    def is_parameterized(self) -> bool:
        """Whether the circuit contains symbolic parameters."""
        return len(self.parameters) > 0

    @property
    def num_gates(self) -> int:
        """Number of :class:`Gate` nodes (including inside blocks)."""
        return sum(1 for op in self.walk() if isinstance(op, Gate))

    @property
    def depth(self) -> int:
        """Circuit depth via longest-path scheduling over Gate nodes."""
        qubit_finish: dict[int, int] = {}
        for op in self.walk():
            if isinstance(op, Gate):
                layer = max((qubit_finish.get(q, 0) for q in op.qubits), default=0)
                new_layer = layer + 1
                for q in op.qubits:
                    qubit_finish[q] = new_layer
        return max(qubit_finish.values()) if qubit_finish else 0

    def gate_names(self) -> dict[str, int]:
        """Count of each gate name (including inside blocks)."""
        counts: dict[str, int] = {}
        for op in self.walk():
            if isinstance(op, Gate):
                counts[op.name] = counts.get(op.name, 0) + 1
        return counts

    def measurements(self) -> list[Measurement]:
        """Ordered list of :class:`Measurement` nodes (including inside blocks)."""
        return [op for op in self.walk() if isinstance(op, Measurement)]

    def has_reset(self) -> bool:
        """Whether the circuit contains a :class:`Reset` node."""
        return any(isinstance(op, Reset) for op in self.walk())

    def has_barrier(self) -> bool:
        """Whether the circuit contains a :class:`Barrier` node."""
        return any(isinstance(op, Barrier) for op in self.walk())

    def has_conditions(self) -> bool:
        """Whether the circuit contains classically-conditioned nodes."""
        for op in self.walk():
            if isinstance(op, Gate) and op.condition is not None:
                return True
            if isinstance(op, ConditionalBlock):
                return True
        return False

    # ------------------------------------------------------------------
    # Serialization / display
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_classical_bits": self.num_classical_bits,
            "operations": [op.to_dict() for op in self.operations],
            "metadata": json_safe(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self) -> Iterator[IRNode]:
        return iter(self.operations)

    def __getitem__(self, index: int) -> IRNode:
        return self.operations[index]

    def __repr__(self) -> str:
        return (
            f"IRCircuit(name='{self.name}', num_qubits={self.num_qubits}, "
            f"ops={len(self.operations)}, gates={self.num_gates}, "
            f"depth={self.depth}"
            + (f", parameters={len(self.parameters)}" if self.is_parameterized else "")
            + ")"
        )

    def __str__(self) -> str:
        lines = [
            f"IRCircuit '{self.name}' ({self.num_qubits} qubits, "
            f"{self.num_classical_bits} classical bits, "
            f"{self.num_gates} gates, depth {self.depth})",
        ]
        for op in self.operations:
            lines.append(f"  {op}")
        return "\n".join(lines)


@dataclass
class IRModule:
    """Top-level container holding one or more :class:`IRCircuit` blocks.

    Attributes:
        name: Module identifier.
        circuits: Ordered circuits making up the module.
        metadata: Free-form module metadata.
    """

    name: str = "main"
    circuits: list[IRCircuit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_circuit(self, circuit: IRCircuit) -> "IRModule":
        """Append a circuit and return self for chaining."""
        self.circuits.append(circuit)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "circuits": [c.to_dict() for c in self.circuits],
            "metadata": json_safe(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return f"IRModule(name='{self.name}', circuits={len(self.circuits)})"

    def __str__(self) -> str:
        return "\n\n".join(str(c) for c in self.circuits)


__all__ = ["IRCircuit", "IRModule", "Condition", "Gate", "Measurement", "Reset",
           "Barrier", "ConditionalBlock", "IRNode", "IRParam"]