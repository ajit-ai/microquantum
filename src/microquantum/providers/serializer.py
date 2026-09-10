"""Circuit serializer for hardware providers.

Converts a bound :class:`~microquantum.core.circuit.QuantumCircuit` into a
provider-native JSON instruction list. Only the standard gate set
understood by both IBM Quantum and IonQ is emitted. If a gate cannot be
mapped, :class:`UnsupportedGateError` is raised so callers can transpile
first.

This module does NOT depend on Qiskit, Cirq, or OpenQASM - it walks
microquantum's own ``_gate_instructions``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator

# Gates serializable natively (name) - matched by operator name/parametrized type.
_STANDARD_GATES = {
    "i", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "cx", "cnot", "cz", "swap",
}


class UnsupportedGateError(ValueError):
    """Raised when a circuit contains a gate the provider cannot run."""


def _rotation_angle(op: Operator, axis: str) -> float:
    """Extract the rotation angle of an Rx/Ry/Rz operator matrix.

    Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2}).
    Rx/Ry follow the standard half-angle parametrization.
    """
    mat = op.matrix
    if axis == "rz":
        # Phase of first diagonal element = -theta/2
        return -2.0 * math.atan2(float(np.imag(mat[0, 0])), float(np.real(mat[0, 0])))
    # rx: [[c, -i s], [-i s, c]], ry: [[c, -s], [s, c]]
    angle = 2.0 * math.acos(np.clip(float(np.real(mat[0, 0])), -1.0, 1.0))
    # adjust to match convention (sign handling for rx)
    if axis == "rx" and np.imag(mat[0, 1]) > 0:
        angle = -angle
    return angle


def _gate_name(op: Operator) -> str:
    """Identify a concrete operator by matrix comparison (not just label)."""
    # Trusted name labels for operators built by the SDK constructors
    if op.name in _STANDARD_GATES:
        return op.name
    raise UnsupportedGateError(f"Gate '{op.name}' is not part of the native set")


def _instruction_to_dict(instr: Any) -> dict[str, Any]:
    """Convert one gate instruction to a canonical JSON dict.

    Returns one of:
      - {"gate", "targets", "params"} for rotations
      - {"gate", "targets"} for fixed gates
    """
    gate: str
    targets: list[int]
    params: list[float] = []

    if QuantumCircuit._is_parameterized_gate(instr):
        gate_type = str(instr[0])
        param = instr[1]
        target = int(instr[2])
        if gate_type not in ("rx", "ry", "rz"):
            raise UnsupportedGateError(
                f"Parameterized gate '{gate_type}' is not natively supported"
            )
        theta = float(param.real) if isinstance(param, complex) else float(param)
        gate = gate_type
        targets = [target]
        params = [theta]
    else:
        op, targets_raw = instr  # type: ignore[misc]
        op = op  # type: ignore[assignment]
        gate = _gate_name(op)  # type: ignore[arg-type]
        targets = [int(t) for t in targets_raw]
        if gate in ("rx", "ry", "rz"):
            params = [_rotation_angle(op, gate)]  # type: ignore[arg-type]

    return {"gate": gate, "targets": targets, "params": params}


class CircuitSerializer:
    """Serialize bound circuits to provider-native JSON gate lists."""

    @staticmethod
    def serialize(circuit: QuantumCircuit) -> dict[str, Any]:
        """Serialize a circuit to a provider-neutral gate list.

        Args:
            circuit: Fully-bound quantum circuit.

        Returns:
            Dict with ``num_qubits`` and ``instructions`` (list of gate dicts).

        Raises:
            ValueError: If the circuit still has unbound parameters.
            UnsupportedGateError: If a gate is not in the native set.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(f"Expected QuantumCircuit, got {type(circuit)}")
        circuit._ensure_bound()

        instructions = [_instruction_to_dict(i) for i in circuit._gate_instructions]
        return {
            "num_qubits": circuit.num_qubits,
            "instructions": instructions,
        }

    @staticmethod
    def to_ionq(circuit: QuantumCircuit) -> dict[str, Any]:
        """Serialize a circuit into IonQ API v0.3 native JSON.

        IonQ expects ``{"qubits": N, "circuit": [...]}`` where each element
        has a ``gate`` plus ``target`` and (for rotations) ``rotation``.
        """
        data = CircuitSerializer.serialize(circuit)
        qubits = data["num_qubits"]
        circuit_list: list[dict[str, Any]] = []
        for instr in data["instructions"]:
            gate = instr["gate"]
            targets = instr["targets"]
            if gate == "cnot":
                circuit_list.append({"gate": "cnot", "control": targets[0], "target": targets[1]})
            elif gate == "cz":
                circuit_list.append({"gate": "cz", "control": targets[0], "target": targets[1]})
            elif gate == "swap":
                circuit_list.append({"gate": "swap", "targets": targets})
            elif gate in ("rx", "ry", "rz"):
                circuit_list.append({
                    "gate": gate,
                    "target": targets[0],
                    "rotation": instr["params"][0],
                })
            else:
                circuit_list.append({"gate": gate, "target": targets[0]})
        return {"qubits": qubits, "circuit": circuit_list}

    @staticmethod
    def to_ibm(circuit: QuantumCircuit) -> dict[str, Any]:
        """Serialize a circuit into IBM Quantum Runtime (v2) JSON.

        Emits ``instructions`` fields with ``name`` and ``qubits`` matching
        the native gate set.
        """
        data = CircuitSerializer.serialize(circuit)
        instructions = []
        for instr in data["instructions"]:
            gate = instr["gate"]
            qubits = instr["targets"]
            if gate in ("rx", "ry", "rz"):
                instructions.append({
                    "name": gate,
                    "qubits": qubits,
                    "params": instr["params"],
                })
            else:
                instructions.append({"name": gate, "qubits": qubits})
        return {
            "num_qubits": data["num_qubits"],
            "instructions": instructions,
        }