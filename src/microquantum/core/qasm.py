"""OpenQASM 2.0 export and import for microquantum circuits.

Implements the OpenQASM 2.0 specification natively within the microquantum
SDK. QASM 2.0 is a public open standard for representing quantum circuits
as text-based programs (Spec: https://arxiv.org/abs/1707.03429).

This module provides microquantum-native serialization and deserialization
of circuits using the QASM 2.0 text format.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .circuit import QuantumCircuit
from .operators import Operator

# Map microquantum gate names to QASM gate names
_TO_QASM: dict[str, str] = {
    "i": "id",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "t": "t",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "cnot": "cx",
    "cx": "cx",
    "cz": "cz",
    "swap": "swap",
}

# QASM gate names to microquantum factory methods
_FROM_QASM: dict[str, tuple[str, int]] = {
    "id": ("i", 1),
    "x": ("x", 1),
    "y": ("y", 1),
    "z": ("z", 1),
    "h": ("h", 1),
    "s": ("s", 1),
    "sdg": ("s_dag", 1),
    "t": ("t", 1),
    "tdg": ("t_dag", 1),
    "rx": ("rx", 1),
    "ry": ("ry", 1),
    "rz": ("rz", 1),
    "cx": ("cx", 2),
    "cz": ("cz", 2),
    "swap": ("swap", 2),
}


def to_qasm(circuit: QuantumCircuit, header: bool = True) -> str:
    """Export a quantum circuit to OpenQASM 2.0 format.

    Args:
        circuit: The quantum circuit to export.
        header: Whether to include the QASM header and qreg/creg declarations.

    Returns:
        OpenQASM 2.0 string representation.

    Raises:
        ValueError: If the circuit contains gates not representable in QASM 2.0.
    """
    lines: list[str] = []

    if header:
        lines.append('OPENQASM 2.0;')
        lines.append('include "qelib1.inc";')
        lines.append(f'qreg q[{circuit.num_qubits}];')
        lines.append(f'creg c[{circuit.num_qubits}];')
        lines.append('')

    for op, targets in circuit.gates:
        qasm_gate = _TO_QASM.get(op.name)
        if qasm_gate is None:
            raise ValueError(
                f"Gate '{op.name}' has no QASM 2.0 equivalent. "
                f"Supported: {list(_TO_QASM.keys())}"
            )

        qubit_args = ", ".join(f"q[{t}]" for t in targets)

        if qasm_gate in ("rx", "ry", "rz"):
            angle = _extract_rotation_angle(op, qasm_gate)
            lines.append(f'{qasm_gate}({angle:.6f}) {qubit_args};')
        else:
            lines.append(f'{qasm_gate} {qubit_args};')

    return "\n".join(lines) + "\n"


def from_qasm(qasm_str: str) -> QuantumCircuit:
    """Import a quantum circuit from OpenQASM 2.0 format.

    Parses the QASM string and creates a microquantum QuantumCircuit.
    Supports standard gates from qelib1.inc.

    Args:
        qasm_str: OpenQASM 2.0 formatted string.

    Returns:
        QuantumCircuit constructed from the QASM.

    Raises:
        ValueError: If the QASM is invalid or contains unsupported gates.
    """
    lines = _parse_qasm_lines(qasm_str)

    num_qubits = 0
    gate_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("qreg"):
            num_qubits = _parse_qreg(stripped)
        elif stripped.startswith("creg"):
            continue
        elif stripped and not stripped.startswith(("OPENQASM", "include", "//")):
            gate_lines.append(stripped)

    if num_qubits == 0:
        raise ValueError("No qreg declaration found in QASM")

    qc = QuantumCircuit(num_qubits)

    for line in gate_lines:
        _apply_qasm_gate(qc, line)

    return qc


def _parse_qasm_lines(qasm_str: str) -> list[str]:
    """Parse QASM string into cleaned lines."""
    result: list[str] = []
    for line in qasm_str.strip().split("\n"):
        cleaned = line.split("//")[0].strip()
        if cleaned:
            result.append(cleaned)
    return result


def _parse_qreg(line: str) -> int:
    """Extract qubit count from qreg declaration."""
    parts = line.replace(";", "").split("[")
    if len(parts) != 2:
        raise ValueError(f"Invalid qreg: {line}")
    return int(parts[1].replace("]", ""))


def _apply_qasm_gate(qc: QuantumCircuit, line: str) -> None:
    """Apply a single QASM gate line to a circuit."""
    line = line.rstrip(";").strip()
    if "(" in line:
        gate_part, qubit_part = line.split("(", 1)
        gate_name = gate_part.strip()
        angle_str, qubit_part = qubit_part.split(")", 1)
        angle = float(angle_str)
    else:
        parts = line.split(" ", 1)
        gate_name = parts[0].strip()
        qubit_part = parts[1].strip() if len(parts) > 1 else ""
        angle = 0.0

    qubits = _parse_qubit_args(qubit_part)

    factory_name, expected_qubits = _FROM_QASM.get(gate_name, ("", 0))
    if not factory_name:
        raise ValueError(f"Unsupported QASM gate: {gate_name}")

    if len(qubits) != expected_qubits:
        raise ValueError(
            f"Gate '{gate_name}' expects {expected_qubits} qubits, got {len(qubits)}"
        )

    _add_gate_by_name(qc, factory_name, qubits, angle)


def _parse_qubit_args(args: str) -> list[int]:
    """Parse 'q[0], q[1]' into [0, 1]."""
    qubits: list[int] = []
    for part in args.split(","):
        part = part.strip()
        if "[" in part and "]" in part:
            idx = int(part.split("[")[1].split("]")[0])
            qubits.append(idx)
    return qubits


def _add_gate_by_name(
    qc: QuantumCircuit,
    name: str,
    qubits: list[int],
    angle: float,
) -> None:
    """Add a gate to the circuit by its internal name."""
    if name == "i":
        qc.append(Operator.I(), qubits)
    elif name == "x":
        qc.x(qubits[0])
    elif name == "y":
        qc.append(Operator.Y(), qubits)
    elif name == "z":
        qc.append(Operator.Z(), qubits)
    elif name == "h":
        qc.h(qubits[0])
    elif name == "s":
        qc.s(qubits[0])
    elif name == "s_dag":
        qc.append(Operator.S().dag, qubits)
    elif name == "t":
        qc.append(Operator.T(), qubits)
    elif name == "t_dag":
        qc.append(Operator.T().dag, qubits)
    elif name == "rx":
        qc.append(Operator.Rx(angle), qubits)
    elif name == "ry":
        qc.ry(angle, qubits[0])
    elif name == "rz":
        qc.rz(angle, qubits[0])
    elif name == "cx":
        qc.cx(qubits[0], qubits[1])
    elif name == "cz":
        qc.append(Operator.CZ(), qubits)
    elif name == "swap":
        qc.swap(qubits[0], qubits[1])


def _extract_rotation_angle(op: Operator, gate_type: str) -> float:
    """Extract the rotation angle from an Rx/Ry/Rz gate matrix.

    Uses the matrix entries to recover the angle.
    """
    m = op.matrix

    if gate_type == "rx":
        c = float(np.real(m[0, 0]))
        return 2.0 * math.acos(max(-1.0, min(1.0, c)))
    elif gate_type == "ry":
        c = float(np.real(m[0, 0]))
        return 2.0 * math.acos(max(-1.0, min(1.0, c)))
    elif gate_type == "rz":
        return float(np.angle(m[1, 1])) * 2.0
    return 0.0
