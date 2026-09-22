"""Backend-independent quantum circuit model.

The canonical :class:`QuantumCircuit` implementation lives in
``microquantum.core.circuit._model`` (moved from the stable 1.0.0 flat
module) to preserve the public API.  This package re-exports that model
and adds the circuit-level value objects required by the expanded Core:
explicit instructions, metadata, validation and iteration helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from ._model import (
    QuantumCircuit,
    _gate_name,
    _narrow_concrete,
    _narrow_parameterized,
)

__all__ = [
    "QuantumCircuit",
    "CircuitInstruction",
    "CircuitMetadata",
    "validate_circuit",
    "iter_instructions",
    "qubit_depth",
    # Re-exported canonical-model helpers (used across Core/backends).
    "_gate_name",
    "_narrow_concrete",
    "_narrow_parameterized",
]


@dataclass(frozen=True)
class CircuitInstruction:
    """A single backend-independent circuit instruction."""

    operation: str
    qubits: tuple[int, ...]
    clbits: tuple[int, ...] = ()
    params: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not self.operation:
            raise ValueError("operation must be non-empty")
        if any(q < 0 for q in self.qubits):
            raise ValueError("qubit indices must be >= 0")
        if any(c < 0 for c in self.clbits):
            raise ValueError("clbit indices must be >= 0")


@dataclass
class CircuitMetadata:
    """Free-form circuit metadata (name, shots, backend hints...)."""

    name: str = "circuit"
    shots: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shots < 1:
            raise ValueError("shots must be >= 1")


def validate_circuit(circuit: QuantumCircuit) -> None:
    """Validate qubit indices, sizes and measurement targets.

    Raises:
        ValueError: On any structural inconsistency.
        TypeError: If *circuit* is not a :class:`QuantumCircuit`.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"Expected QuantumCircuit, got {type(circuit).__name__}")
    n = circuit.num_qubits
    for instr in circuit._gate_instructions:  # noqa: SLF001 - canonical model access
        targets = circuit._get_targets(instr)  # noqa: SLF001
        for t in targets:
            if not 0 <= t < n:
                raise ValueError(f"Qubit index {t} out of range for {n} qubits")
    for m in circuit._measurements:  # noqa: SLF001
        if not 0 <= m < n:
            raise ValueError(f"Measurement qubit {m} out of range for {n} qubits")


def iter_instructions(circuit: QuantumCircuit) -> Iterator[CircuitInstruction]:
    """Iterate over a circuit as backend-independent instructions."""
    validate_circuit(circuit)
    for instr in circuit._gate_instructions:  # noqa: SLF001
        if circuit._is_parameterized_gate(instr):  # noqa: SLF001
            name, param, target = _narrow_parameterized(instr)
            yield CircuitInstruction(str(name), (int(target),), (), (param,))
        else:
            op, targets = _narrow_concrete(instr)
            yield CircuitInstruction(op.name, tuple(int(t) for t in targets))
    for m in circuit._measurements:  # noqa: SLF001
        yield CircuitInstruction("measure", (int(m),))


def qubit_depth(circuit: QuantumCircuit) -> int:
    """Return circuit depth (longest critical path)."""
    return circuit.depth()
