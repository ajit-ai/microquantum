"""Conversion between public circuits and MicroQuantum IR.

``to_ir(``:class:`~microquantum.core.circuit.QuantumCircuit```)`` produces
the internal representation of a user circuit.  ``from_ir`` rebuilds a
``QuantumCircuit`` from compiled IR so transformed programs can still be
executed on the existing backends.

Rotation angles of concrete gate operators (:class:`Operator.Rx`, ``Ry``,
``Rz``) are recovered from their matrices at the circuit->IR boundary; the
IR itself carries plain floats and stays NumPy-free.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union, cast

import numpy as np

from ..core.operators import Operator
from ..core.parameter import Parameter, ParameterExpression
from .circuit_ir import IRCircuit
from .nodes import Barrier, Condition, ConditionalBlock, Gate, IRParam, Measurement, Reset

if TYPE_CHECKING:
    from ..core.circuit import QuantumCircuit
    from ..core.dynamic import DynamicCircuit

_ROTATION_GATES = {"rx", "ry", "rz"}

_CircuitInstruction = Union[
    tuple[Operator, list[int]],
    tuple[str, Union[Parameter, ParameterExpression], int],
]

_OP_FACTORIES: dict[str, object] = {
    "h": Operator.H,
    "x": Operator.X,
    "y": Operator.Y,
    "z": Operator.Z,
    "s": Operator.S,
    "sdg": Operator.Sdg,
    "t": Operator.T,
    "tdg": Operator.Tdg,
    "rx": lambda th: Operator.Rx(float(th)),
    "ry": lambda th: Operator.Ry(float(th)),
    "rz": lambda th: Operator.Rz(float(th)),
    "cnot": Operator.CNOT,
    "cz": Operator.CZ,
    "swap": Operator.SWAP,
}


def _rotation_angle(op: Operator) -> float:
    """Recover the rotation angle theta from Rx/Ry/Rz matrix."""
    m = np.asarray(op.matrix, dtype=np.complex128)
    if op.name in ("rx", "ry"):
        c = float(m[0, 0].real)
        return 2.0 * math.acos(max(-1.0, min(1.0, c)))
    if op.name == "rz":
        # M = [[e^{-i*theta/2}, 0], [0, e^{i*theta/2}]]
        return -2.0 * math.atan2(float(m[0, 0].imag), float(m[0, 0].real))
    raise ValueError(f"{op.name} is not a rotation gate")


def _to_ir_instruction(instr: _CircuitInstruction, index: int) -> Gate:
    """Convert one circuit instruction tuple into an IR Gate node."""
    # Parameterized descriptor: (gate_type, Parameter, target)
    if len(instr) == 3 and isinstance(instr[0], str):
        gate_type, param, target = instr
        return Gate(
            name=gate_type,
            qubits=(int(target),),
            params=(param,),
            source={"circuit_index": index},
        )
    op, targets = instr  # (Operator, list[int])
    qubits = tuple(int(t) for t in targets)
    name = op.name
    if name in _ROTATION_GATES:
        return Gate(
            name=name,
            qubits=qubits,
            params=(float(_rotation_angle(op)),),
            source={"circuit_index": index},
        )
    return Gate(name=name, qubits=qubits, source={"circuit_index": index})


def to_ir(
    circuit: "QuantumCircuit",
    include_terminal_measurements: bool = False,
    name: str = "main",
) -> IRCircuit:
    """Convert a :class:`QuantumCircuit` into MicroQuantum IR.

    Args:
        circuit: The circuit to convert.
        include_terminal_measurements: If True, append a ``Measurement(q, q)``
            for every qubit so the IR describes a full sampling program.
        name: Name for the resulting :class:`IRCircuit`.

    Returns:
        The IR representation of the circuit.
    """
    ir = IRCircuit(
        num_qubits=circuit.num_qubits,
        name=name,
        metadata={"source": "QuantumCircuit"},
    )
    for index, instr in enumerate(circuit._gate_instructions):
        ir.add(_to_ir_instruction(instr, index))
    if include_terminal_measurements:
        for q in range(circuit.num_qubits):
            ir.add(Measurement(qubit=q, classical=q))
    return ir


# ----------------------------------------------------------------------
# DynamicCircuit conversion
# ----------------------------------------------------------------------

def _dynamic_ops(dc: "DynamicCircuit") -> IRCircuit:
    """Convert a DynamicCircuit's _ops sequence into an IRCircuit."""
    from ..core.dynamic import DynamicCircuit

    ir = IRCircuit(
        num_qubits=dc.num_qubits,
        num_classical_bits=dc.num_classical_bits,
    )
    for op in dc._ops:
        kind = op[0]
        if kind == "gate":
            ir.add(Gate(name=op[1].name, qubits=tuple(int(t) for t in op[2])))
        elif kind == "measure":
            ir.add(Measurement(qubit=int(op[1]), classical=int(op[2])))
        elif kind == "reset":
            ir.add(Reset(qubit=int(op[1])))
        elif kind == "classical_if":
            classical_bit, gate_fn = op[1], op[2]
            temp = DynamicCircuit(dc.num_qubits, dc.num_classical_bits)
            gate_fn(temp)
            block = ConditionalBlock(
                condition=Condition(bit=int(classical_bit), value=1),
                operations=tuple(_dynamic_ops(temp).operations),
            )
            ir.add(block)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported DynamicCircuit op kind: {kind!r}")
    return ir


def to_ir_dynamic(dc: "DynamicCircuit", name: str = "main") -> IRCircuit:
    """Convert a :class:`DynamicCircuit` into MicroQuantum IR.

    Preserves mid-circuit measurements, resets and classically-conditioned
    blocks as first-class IR nodes.
    """
    ir = _dynamic_ops(dc)
    ir.name = name
    ir.metadata = {"source": "DynamicCircuit"}
    return ir


# ----------------------------------------------------------------------
# IR -> QuantumCircuit
# ----------------------------------------------------------------------

def _gate_to_instruction(gate: Gate) -> _CircuitInstruction:
    """Rebuild a circuit instruction tuple from an IR Gate."""
    if gate.condition is not None:
        raise ValueError(
            "IR Gate with a classical condition cannot round-trip to "
            "QuantumCircuit; condition nodes need a DynamicCircuit target."
        )
    if gate.name in _ROTATION_GATES:
        if len(gate.qubits) != 1:
            raise ValueError(f"{gate.name} must act on exactly one qubit")
        if len(gate.params) != 1:
            raise ValueError(f"{gate.name} requires exactly one parameter")
        param: IRParam = gate.params[0]
        if isinstance(param, (int, float, complex)):
            op = _OP_FACTORIES[gate.name](complex(param).real)  # type: ignore[misc,operator]
            return (op, list(gate.qubits))
        return (gate.name, param, gate.qubits[0])
    if gate.params:
        raise ValueError(f"Gate '{gate.name}' does not accept parameters")
    factory = _OP_FACTORIES.get(gate.name)
    if factory is None:
        raise ValueError(f"Unknown gate name for IR->circuit: {gate.name!r}")
    return (factory(), list(gate.qubits))  # type: ignore[misc,operator]


def from_ir(ir: IRCircuit) -> "QuantumCircuit":
    """Rebuild a :class:`QuantumCircuit` from gate-level IR.

    Measurement and Barrier nodes are dropped (sampling happens on the
    backend / is a synchronization hint).  Reset and conditional nodes are
    not representable in a static :class:`QuantumCircuit` and raise
    ``ValueError`` — use the dynamic-circuit path for those.

    Args:
        ir: The IR to convert back.

    Returns:
        A QuantumCircuit executing the IR's gate list.
    """
    from ..core.circuit import QuantumCircuit

    circuit = QuantumCircuit(ir.num_qubits)
    for op in ir.operations:
        if isinstance(op, Gate):
            instr = _gate_to_instruction(op)
            if len(instr) == 2:
                circuit.append(instr[0], instr[1])  # type: ignore[arg-type]
            else:
                circuit.append_parameterized(
                    instr[0],
                    cast("Parameter", instr[1]),
                    instr[2],
                )
        elif isinstance(op, (Measurement, Barrier)):
            continue
        elif isinstance(op, Reset):
            raise ValueError(
                "reset nodes are not representable in a QuantumCircuit; "
                "use a DynamicCircuit for reset support"
            )
        elif isinstance(op, ConditionalBlock):
            raise ValueError(
                "conditional blocks are not representable in a QuantumCircuit; "
                "use a DynamicCircuit for classical control"
            )
    return circuit