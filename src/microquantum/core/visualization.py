"""Text-based quantum circuit visualization.

Generates ASCII circuit diagrams for debugging and display.

Example output for a Bell state circuit:

    q[0]: |0> ── H ── ● ──
    q[1]: |0> ──────── ⊕ ──
"""

from __future__ import annotations

from typing import Optional

from .circuit import QuantumCircuit
from .operators import Operator

# Gate display widths (character count for the gate label)
_GATE_WIDTHS: dict[str, int] = {
    "i": 1, "x": 1, "y": 1, "z": 1,
    "h": 1, "s": 1, "t": 1,
    "cx": 1, "cz": 1, "swap": 1,
    "rx": 0, "ry": 0, "rz": 0,
}

# Standard symbols
_WIRE = "-"
_CONTROL = "*"
_TARGET = "X"
_TOP_T = "|"
_BOT_T = "|"
_CROSS = "+"
_MEASURE = "M"


def draw(circuit: QuantumCircuit, title: Optional[str] = None) -> str:
    """Generate an ASCII circuit diagram.

    Args:
        circuit: The quantum circuit to visualize.
        title: Optional title for the diagram.

    Returns:
        Multi-line string with the ASCII circuit diagram.
    """
    if circuit.num_qubits == 0:
        return "(empty circuit)"

    n = circuit.num_qubits
    gates = circuit.gates

    if not gates:
        return _empty_circuit(n)

    columns = _build_columns(n, gates)
    return _render(n, columns, title)


def text(circuit: QuantumCircuit) -> str:
    """Alias for draw() - generate text circuit representation."""
    return draw(circuit)


def _empty_circuit(n: int) -> str:
    """Render an empty circuit."""
    lines: list[str] = []
    for i in range(n):
        lines.append(f"    q[{i}]: |0> ──")
    return "\n".join(lines)


def _build_columns(
    n: int,
    gates: list[tuple[Operator, list[int]]],
) -> list[list[Optional[tuple[Operator, list[int]]]]]:
    """Assign gates to time-step columns, handling multi-qubit gates."""
    columns: list[list[Optional[tuple[Operator, list[int]]]]] = []
    qubit_occupied: list[int] = [0] * n  # step when qubit becomes free

    for op, targets in gates:
        needed_step = max(qubit_occupied[t] for t in targets)
        while len(columns) <= needed_step:
            columns.append([None] * n)

        step = needed_step
        columns[step][targets[0]] = (op, targets)
        next_step = step + 1
        for t in targets:
            qubit_occupied[t] = next_step

    return columns


def _render(
    n: int,
    columns: list[list[Optional[tuple[Operator, list[int]]]]],
    title: Optional[str] = None,
) -> str:
    """Render columns into ASCII lines."""
    parts: list[str] = []

    if title:
        parts.append(f"  {title}")
        parts.append("")

    # Qubit labels
    max(len(f"    q[{i}]: ") for i in range(n))

    wire_lines: list[list[str]] = [[] for _ in range(n)]

    for i in range(n):
        prefix = f"    q[{i}]: |0> "
        wire_lines[i].append(prefix)

    for _col_idx, column in enumerate(columns):
        # Determine which qubits are involved in this step
        involved: set[int] = set()
        gate_at: dict[int, tuple[Operator, list[int]]] = {}
        for q in range(n):
            if column[q] is not None:
                op, targets = column[q]  # type: ignore[misc]
                gate_at[q] = (op, targets)
                involved.update(targets)

        if not involved:
            # Empty time step
            for i in range(n):
                wire_lines[i].append(_WIRE * 4 + "  ")
            continue

        # Single-qubit gates or multi-qubit gates
        min_q = min(involved)
        max_q = max(involved)

        if len(involved) == 1:
            # Simple single-qubit gate
            q = list(involved)[0]
            op, _ = gate_at[q]
            label = _gate_label(op)
            width = max(len(label), 3)

            for i in range(n):
                if i == q:
                    wire_lines[i].append(f" {label} ")
                else:
                    wire_lines[i].append(_WIRE * (width + 2))
        else:
            # Multi-qubit gate
            op = gate_at[min_q][0]
            is_swap = op.name == "swap"
            is_cnot = op.name in ("cx", "cnot")

            if is_cnot:
                label = _WIRE * 4
                for i in range(n):
                    if i in involved and i != min_q:
                        # Target
                        wire_lines[i].append(f" {_TARGET} ")
                    elif i == min_q:
                        # Control
                        wire_lines[i].append(f" {_CONTROL} ")
                    elif min_q < i < max_q:
                        # Vertical connector
                        wire_lines[i].append(f" {_WIRE}  ")
                    else:
                        wire_lines[i].append(_WIRE * 4)
            elif is_swap:
                for i in range(n):
                    if i in involved:
                        wire_lines[i].append(" x ")
                    elif min_q < i < max_q:
                        wire_lines[i].append(f" {_WIRE}  ")
                    else:
                        wire_lines[i].append(_WIRE * 4)
            else:
                # Generic multi-qubit
                for i in range(n):
                    if i in involved:
                        wire_lines[i].append(f" {_CONTROL} ")
                    elif min_q < i < max_q:
                        wire_lines[i].append(f" {_WIRE}  ")
                    else:
                        wire_lines[i].append(_WIRE * 4)

    # Join everything
    lines: list[str] = []
    if title:
        lines.append(f"  {title}")
        lines.append("")
    for i in range(n):
        lines.append("".join(wire_lines[i]))

    return "\n".join(lines)


def _gate_label(op: Operator) -> str:
    """Get a display label for a gate."""
    name = op.name.lower()

    if name in ("rx", "ry", "rz"):
        angle = _quick_angle(op)
        return f"{name}({angle:.2f})"

    labels = {
        "i": "I", "x": "X", "y": "Y", "z": "Z",
        "h": "H", "s": "S", "t": "T",
        "s_dag": "S†", "t_dag": "T†",
    }
    return labels.get(name, name.upper())


def _quick_angle(op: Operator) -> float:
    """Quick angle extraction for display purposes."""
    import math

    import numpy as np

    name = op.name.lower()
    m = op.matrix

    if name == "rx" or name == "ry":
        c = float(np.real(m[0, 0]))
        return 2.0 * math.acos(max(-1.0, min(1.0, c)))
    elif name == "rz":
        return float(np.angle(m[1, 1])) * 2.0
    return 0.0
