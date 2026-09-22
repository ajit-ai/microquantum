"""Circuit and resource analysis model.

Deterministic, backend-independent resource estimation: qubit/clbit
counts, width, depth, gate counts by arity, measurement and parameter
counts, plus memory estimates.  Re-exports the canonical
:class:`ResourceEstimate` / :class:`ResourceEstimator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._model import ResourceEstimate, ResourceEstimator

__all__ = [
    "ResourceEstimate",
    "ResourceEstimator",
    "CircuitResources",
    "estimate_resources",
    "estimate_state_memory_bytes",
]


@dataclass(frozen=True)
class CircuitResources:
    """Backend-independent resource summary for a circuit."""

    num_qubits: int
    num_clbits: int = 0
    width: int = 0
    depth: int = 0
    total_gates: int = 0
    single_qubit_gates: int = 0
    two_qubit_gates: int = 0
    multi_qubit_gates: int = 0
    measurement_count: int = 0
    parameter_count: int = 0
    gate_type_counts: dict[str, int] = field(default_factory=dict)
    estimated_state_memory_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "version": 1,
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "width": self.width,
            "depth": self.depth,
            "total_gates": self.total_gates,
            "single_qubit_gates": self.single_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "multi_qubit_gates": self.multi_qubit_gates,
            "measurement_count": self.measurement_count,
            "parameter_count": self.parameter_count,
            "gate_type_counts": dict(self.gate_type_counts),
            "estimated_state_memory_bytes": self.estimated_state_memory_bytes,
        }


def estimate_state_memory_bytes(num_qubits: int, density_matrix: bool = False) -> int:
    """Memory for one state vector (or density matrix) of *num_qubits*."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    dim = 2**num_qubits
    entries = dim * dim if density_matrix else dim
    return int(entries * 16)  # complex128


def estimate_resources(circuit: Any) -> CircuitResources:
    """Compute deterministic resources for a :class:`QuantumCircuit`."""
    from microquantum.core.circuit import QuantumCircuit  # noqa: PLC0415

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"Expected QuantumCircuit, got {type(circuit).__name__}")
    num_qubits = circuit.num_qubits
    num_clbits = int(getattr(circuit, "num_clbits", 0) or 0)
    single = two = multi = 0
    gate_counts: dict[str, int] = {}
    for instr in circuit._gate_instructions:  # noqa: SLF001
        if circuit._is_parameterized_gate(instr):  # noqa: SLF001
            from microquantum.core.circuit import _narrow_parameterized  # noqa: PLC0415

            name = str(_narrow_parameterized(instr)[0]).lower()
            arity = 1
        else:
            from microquantum.core.circuit import _narrow_concrete  # noqa: PLC0415

            op, targets = _narrow_concrete(instr)
            name = op.name.lower()
            arity = len(targets)
        gate_counts[name] = gate_counts.get(name, 0) + 1
        if arity == 1:
            single += 1
        elif arity == 2:
            two += 1
        else:
            multi += 1
    measurements = list(getattr(circuit, "_measurements", []) or [])
    try:
        params = tuple(circuit.parameters)
    except Exception:
        params = ()
    depth = circuit.depth()
    return CircuitResources(
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        width=num_qubits,
        depth=depth,
        total_gates=single + two + multi,
        single_qubit_gates=single,
        two_qubit_gates=two,
        multi_qubit_gates=multi,
        measurement_count=len(measurements),
        parameter_count=len(params),
        gate_type_counts=gate_counts,
        estimated_state_memory_bytes=estimate_state_memory_bytes(num_qubits),
    )
