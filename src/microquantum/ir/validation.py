"""Structural validation of MicroQuantum IR.

Uses the SDK's existing conventions: ``validate`` returns a list of human
readable error strings (an empty list meaning valid) while
:func:`assert_valid` raises ``ValueError`` with all problems joined — the
same exception type the public circuit layer uses for invalid inputs.
"""

from __future__ import annotations

from .circuit_ir import IRCircuit
from .nodes import (
    Barrier,
    Condition,
    ConditionalBlock,
    Gate,
    IRNode,
    Measurement,
    Reset,
)

_ROTATION_GATES = frozenset({"rx", "ry", "rz"})

_GATE_ARITY: dict[str, int] = {
    "h": 1, "x": 1, "y": 1, "z": 1,
    "s": 1, "sdg": 1, "t": 1, "tdg": 1,
    "rx": 1, "ry": 1, "rz": 1,
    "cnot": 2, "cz": 2, "swap": 2,
}


def validate_operation(
    op: IRNode,
    num_qubits: int,
    num_classical_bits: int,
    errors: list[str],
    where: str = "",
) -> None:
    """Validate a single IR node, appending problems to *errors*."""
    loc = f"{where}: " if where else ""

    if isinstance(op, Gate):
        if not op.qubits:
            errors.append(f"{loc}gate '{op.name}' has no target qubits")
        for q in op.qubits:
            if not (0 <= q < num_qubits):
                errors.append(f"{loc}gate '{op.name}' references invalid qubit {q}")
        arity = _GATE_ARITY.get(op.name)
        if arity is None:
            errors.append(f"{loc}gate '{op.name}' is not a known gate")
        elif arity != len(op.qubits):
            errors.append(
                f"{loc}gate '{op.name}' expects {arity} qubit(s), "
                f"got {len(op.qubits)}"
            )
        if op.name in _ROTATION_GATES:
            if len(op.params) != 1:
                errors.append(
                    f"{loc}rotation gate '{op.name}' requires exactly one parameter"
                )
        elif op.params:
            errors.append(f"{loc}gate '{op.name}' does not accept parameters")
        if op.condition is not None:
            validate_condition(op.condition, num_classical_bits, errors, loc)

    elif isinstance(op, Measurement):
        if not (0 <= op.qubit < num_qubits):
            errors.append(f"{loc}measurement references invalid qubit {op.qubit}")
        if op.classical is not None and not (0 <= op.classical < num_classical_bits):
            errors.append(
                f"{loc}measurement references invalid classical bit {op.classical}"
            )

    elif isinstance(op, Reset):
        if not (0 <= op.qubit < num_qubits):
            errors.append(f"{loc}reset references invalid qubit {op.qubit}")

    elif isinstance(op, Barrier):
        if not op.qubits:
            errors.append(f"{loc}barrier has no qubits")
        for q in op.qubits:
            if not (0 <= q < num_qubits):
                errors.append(f"{loc}barrier references invalid qubit {q}")

    elif isinstance(op, ConditionalBlock):
        validate_condition(op.condition, num_classical_bits, errors, loc)
        for nested in op.operations:
            validate_operation(
                nested, num_qubits, num_classical_bits, errors, loc
            )

    else:  # pragma: no cover - guarded by type system
        errors.append(f"{loc}unknown IR node type {type(op).__name__}")


def validate_condition(
    condition: Condition,
    num_classical_bits: int,
    errors: list[str],
    loc: str = "",
) -> None:
    """Validate a :class:`Condition` against the circuit's classical bits."""
    if not (0 <= condition.bit < num_classical_bits):
        errors.append(f"{loc}condition references invalid classical bit {condition.bit}")
    if condition.value not in (0, 1):
        errors.append(f"{loc}condition value must be 0 or 1, got {condition.value}")


def validate(ir: IRCircuit) -> list[str]:
    """Return a list of structural validation errors for *ir*.

    An empty list means the IR is structurally valid.

    Args:
        ir: The IR circuit to validate.
    """
    errors: list[str] = []
    num_classical_bits = (
        ir.num_classical_bits if ir.num_classical_bits is not None else ir.num_qubits
    )
    if ir.num_qubits < 1:
        errors.append(f"IRCircuit must have at least one qubit, got {ir.num_qubits}")
    if num_classical_bits < 0:  # pragma: no cover - guarded in constructor
        errors.append("num_classical_bits must be >= 0")
    for op in ir.operations:
        validate_operation(op, ir.num_qubits, num_classical_bits, errors)
    return errors


def assert_valid(ir: IRCircuit) -> None:
    """Raise ``ValueError`` if *ir* is not structurally valid.

    Args:
        ir: The IR circuit to validate.

    Raises:
        ValueError: With all detected problems joined into one message.
    """
    errors = validate(ir)
    if errors:
        raise ValueError(
            "Invalid MicroQuantum IR:\n" + "\n".join(f"  - {e}" for e in errors)
        )