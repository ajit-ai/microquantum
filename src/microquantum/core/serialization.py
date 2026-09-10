"""JSON-based quantum circuit serialization.

Provides save/load methods for persisting circuits as JSON files
or strings, useful for saving work, sharing circuits, and logging.

Format version: 1.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .circuit import QuantumCircuit
from .operators import Operator

FORMAT_VERSION = "1.0"


def to_dict(circuit: QuantumCircuit) -> dict[str, Any]:
    """Serialize a quantum circuit to a JSON-compatible dictionary.

    Args:
        circuit: The circuit to serialize.

    Returns:
        Dictionary representation of the circuit.
    """
    gates: list[dict[str, Any]] = []
    for op, targets in circuit.gates:
        gate_dict: dict[str, Any] = {
            "name": op.name,
            "targets": targets,
            "num_qubits": op.num_qubits,
        }
        if op.name in ("rx", "ry", "rz"):
            gate_dict["angle"] = _extract_angle(op, op.name)
        gates.append(gate_dict)

    return {
        "format_version": FORMAT_VERSION,
        "num_qubits": circuit.num_qubits,
        "num_gates": circuit.num_gates,
        "depth": circuit.depth,
        "gates": gates,
    }


def from_dict(data: dict[str, Any]) -> QuantumCircuit:
    """Deserialize a quantum circuit from a dictionary.

    Args:
        data: Dictionary created by to_dict().

    Returns:
        Reconstructed QuantumCircuit.

    Raises:
        ValueError: If the data is invalid or uses an unsupported format.
    """
    version = data.get("format_version", "")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format version '{version}'. Expected '{FORMAT_VERSION}'"
        )

    num_qubits = data["num_qubits"]
    qc = QuantumCircuit(num_qubits)

    for gate_dict in data["gates"]:
        name = gate_dict["name"]
        targets = gate_dict["targets"]
        angle = gate_dict.get("angle", 0.0)

        _add_gate(qc, name, targets, angle)

    return qc


def to_json(circuit: QuantumCircuit, indent: Optional[int] = 2) -> str:
    """Serialize a circuit to a JSON string.

    Args:
        circuit: The circuit to serialize.
        indent: JSON indentation level (None for compact).

    Returns:
        JSON string.
    """
    return json.dumps(to_dict(circuit), indent=indent)


def from_json(json_str: str) -> QuantumCircuit:
    """Deserialize a circuit from a JSON string.

    Args:
        json_str: JSON string created by to_json().

    Returns:
        Reconstructed QuantumCircuit.
    """
    return from_dict(json.loads(json_str))


def save(circuit: QuantumCircuit, path: Union[str, Path]) -> None:
    """Save a circuit to a JSON file.

    Args:
        circuit: The circuit to save.
        path: File path to write to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(circuit), encoding="utf-8")


def load(path: Union[str, Path]) -> QuantumCircuit:
    """Load a circuit from a JSON file.

    Args:
        path: File path to read from.

    Returns:
        Loaded QuantumCircuit.
    """
    path = Path(path)
    return from_json(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _extract_angle(op: Operator, gate_type: str) -> float:
    """Extract rotation angle from gate matrix."""
    import math
    m = op.matrix
    if gate_type in ("rx", "ry"):
        c = float(np.real(m[0, 0]))
        return 2.0 * math.acos(max(-1.0, min(1.0, c)))
    elif gate_type == "rz":
        return float(np.angle(m[1, 1])) * 2.0
    return 0.0


def _add_gate(qc: QuantumCircuit, name: str, targets: list[int], angle: float) -> None:
    """Add a gate to the circuit by name."""
    if name == "i":
        qc.append(Operator.I(), targets)
    elif name == "x":
        qc.x(targets[0])
    elif name == "y":
        qc.append(Operator.Y(), targets)
    elif name == "z":
        qc.append(Operator.Z(), targets)
    elif name == "h":
        qc.h(targets[0])
    elif name == "s":
        qc.s(targets[0])
    elif name == "t":
        qc.append(Operator.T(), targets)
    elif name == "rx":
        qc.append(Operator.Rx(angle), targets)
    elif name == "ry":
        qc.ry(angle, targets[0])
    elif name == "rz":
        qc.rz(angle, targets[0])
    elif name in ("cx", "cnot"):
        qc.cx(targets[0], targets[1])
    elif name == "cz":
        qc.append(Operator.CZ(), targets)
    elif name == "swap":
        qc.swap(targets[0], targets[1])
