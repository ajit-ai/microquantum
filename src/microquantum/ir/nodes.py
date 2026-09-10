"""MicroQuantum IR node types.

Nodes are immutable (frozen dataclasses) so IR passes can share
structure freely and memoize comparisons.  Each node understands the
qubits, classical bits and symbolic parameters it touches — the minimal
data required for inspection, validation and transformation.

The IR is intentionally NumPy-free: gate identity is structural
(gate name + qubits + parameters), not matrix-based.

Node hierarchy::

    IRNode
    ├── Gate
    ├── Measurement
    ├── Reset
    ├── Barrier
    └── ConditionalBlock   (contains a Condition + nested IRNode list)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .._json import json_string
from ..core.parameter import Parameter, ParameterExpression

IRParam = Union[float, int, complex, Parameter, ParameterExpression]


def param_to_dict(param: IRParam) -> dict[str, Any]:
    """Serialize an IR parameter to a JSON-safe dictionary."""
    if isinstance(param, Parameter):
        return {"symbol": param.name}
    if isinstance(param, ParameterExpression):
        return {
            "symbol": param.parameter.name,
            "coefficient": {"real": param.coefficient.real, "imag": param.coefficient.imag},
            "constant": {"real": param.constant.real, "imag": param.constant.imag},
        }
    if isinstance(param, complex):
        return {"real": param.real, "imag": param.imag}
    return {"value": float(param)}


def param_to_str(param: IRParam) -> str:
    """Render a parameter for human-readable IR strings."""
    if isinstance(param, Parameter):
        return param.name
    if isinstance(param, ParameterExpression):
        return str(param)
    return str(param)


@dataclass(frozen=True)
class Condition:
    """A classical condition guarding an operation or block.

    Attributes:
        bit: Classical bit index tested.
        value: Value the bit must equal (0 or 1).
        register: Optional classical register name.
    """

    bit: int
    value: int = 1
    register: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "bit": self.bit,
            "value": self.value,
            "register": self.register,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def __str__(self) -> str:
        return f"c{self.bit}=={self.value}"


class IRNode(ABC):
    """Base class for all MicroQuantum IR nodes.

    Every concrete node must provide ``qubits`` — either as a dataclass
    field (Gate, Barrier) or as a property (Measurement, Reset,
    ConditionalBlock).  The ``cbits`` / ``parameters`` result properties
    declared here give conservative defaults.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Node kind identifier (e.g. 'gate', 'measure', 'reset', ...)."""

    @property
    def cbits(self) -> tuple[int, ...]:
        """Classical bits this node touches."""
        return ()

    @property
    def parameters(self) -> tuple[IRParam, ...]:
        """Symbolic/numeric parameters this node carries."""
        return ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {"kind": self.kind, "qubits": list(getattr(self, "qubits", ()))}

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())


@dataclass(frozen=True)
class Gate(IRNode):
    """A quantum gate operation in the IR.

    Attributes:
        name: Gate name from the SDK gate library (h, x, y, z, s, sdg,
            t, tdg, rx, ry, rz, cnot, cz, swap) or a custom operator name.
        qubits: Target qubit indices (deterministic order).
        params: Gate parameters.  Numeric rotation angles are stored as
            plain floats; symbolic angles as Parameter/ParameterExpression.
        condition: Optional :class:`Condition` guarding the gate.
        source: Optional source/debug metadata (e.g. originating circuit
            gate index) attached by the builder.
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[IRParam, ...] = ()
    condition: Optional[Condition] = None
    source: Optional[dict[str, Any]] = None

    @property
    def kind(self) -> str:
        return "gate"

    @property
    def parameters(self) -> tuple[IRParam, ...]:
        return self.params

    @property
    def cbits(self) -> tuple[int, ...]:
        return (self.condition.bit,) if self.condition is not None else ()

    def display_name(self) -> str:
        """Upper-case gate name for human-readable output."""
        return self.name.upper()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "qubits": list(self.qubits),
            "params": [param_to_dict(p) for p in self.params],
            "condition": self.condition.to_dict() if self.condition is not None else None,
            "source": self.source,
        }

    def __str__(self) -> str:
        name = self.display_name()
        if self.params:
            prefix = f"{name}({', '.join(param_to_str(p) for p in self.params)}, "
            return prefix + ",".join(str(q) for q in self.qubits) + ")"
        return f"{name}({','.join(str(q) for q in self.qubits)})"


@dataclass(frozen=True)
class Measurement(IRNode):
    """A (mid-circuit or terminal) measurement of one qubit.

    Attributes:
        qubit: Qubit index measured.
        classical: Classical bit index the outcome is stored into, or
            None when the outcome is only observed.
    """

    qubit: int
    classical: Optional[int] = None

    @property
    def kind(self) -> str:
        return "measure"

    @property
    def qubits(self) -> tuple[int, ...]:
        return (self.qubit,)

    @property
    def cbits(self) -> tuple[int, ...]:
        return (self.classical,) if self.classical is not None else ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "qubit": self.qubit,
            "classical": self.classical,
        }

    def __str__(self) -> str:
        if self.classical is None:
            return f"M(q{self.qubit})"
        return f"M(q{self.qubit}->c{self.classical})"


@dataclass(frozen=True)
class Reset(IRNode):
    """Reset a qubit to the |0> computational basis state.

    Prepares the IR for dynamic-circuit and QEC workflows.
    """

    qubit: int

    @property
    def kind(self) -> str:
        return "reset"

    @property
    def qubits(self) -> tuple[int, ...]:
        return (self.qubit,)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "qubit": self.qubit}

    def __str__(self) -> str:
        return f"reset q{self.qubit}"


@dataclass(frozen=True)
class Barrier(IRNode):
    """A synchronization barrier across a set of qubits."""

    qubits: tuple[int, ...]

    @property
    def kind(self) -> str:
        return "barrier"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "qubits": list(self.qubits)}

    def __str__(self) -> str:
        return "barrier " + ",".join(f"q{q}" for q in self.qubits)


@dataclass(frozen=True)
class ConditionalBlock(IRNode):
    """A classically-controlled block of IR operations.

    Attributes:
        condition: The classical condition that gates the block.
        operations: Ordered operations executed when the condition holds.
    """

    condition: Condition
    operations: tuple["IRNode", ...] = field(default_factory=tuple)

    # A conditional block has no intrinsic qubit list (it is a container);
    # its qubits are the union of nested operations.
    @property
    def kind(self) -> str:
        return "conditional"

    @property
    def qubits(self) -> tuple[int, ...]:
        seen: list[int] = []
        for op in self.operations:
            for q in getattr(op, "qubits", ()):
                if q not in seen:
                    seen.append(q)
        return tuple(seen)

    @property
    def cbits(self) -> tuple[int, ...]:
        bits = [self.condition.bit]
        for op in self.operations:
            for b in op.cbits:
                if b not in bits:
                    bits.append(b)
        return tuple(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "condition": self.condition.to_dict(),
            "operations": [op.to_dict() for op in self.operations],
        }

    def __str__(self) -> str:
        inner = ", ".join(str(op) for op in self.operations)
        return f"IF({self.condition}) {{ {inner} }}"


IRNodeSubtype = Union[Gate, Measurement, Reset, Barrier, ConditionalBlock]