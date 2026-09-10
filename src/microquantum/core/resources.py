"""Resource estimation for quantum circuits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..core.circuit import QuantumCircuit

_DEFAULT_GATE_COSTS: dict[str, float] = {
    "h": 1.0,
    "x": 1.0,
    "y": 1.0,
    "z": 1.0,
    "s": 1.0,
    "t": 1.0,
    "rx": 1.0,
    "ry": 1.0,
    "rz": 1.0,
    "i": 1.0,
    "cnot": 10.0,
    "cx": 10.0,
    "cz": 10.0,
    "swap": 10.0,
}

_T_GATE_NAMES: frozenset[str] = frozenset({"t", "tdg"})
_CNOT_GATE_NAMES: frozenset[str] = frozenset({"cnot", "cx"})


@dataclass
class ResourceEstimate:
    """Resource analysis of a quantum circuit.

    Attributes:
        total_gates: Total number of gates.
        single_qubit_gates: Number of single-qubit gates.
        two_qubit_gates: Number of two-qubit gates.
        depth: Circuit depth (critical path length).
        num_qubits: Number of qubits.
        t_count: Number of T gates (critical for fault-tolerant cost).
        cnot_count: Number of CNOT/CX gates.
        estimated_time_ns: Estimated execution time in nanoseconds.
        circuit_volume: ``num_qubits * depth``.
        gate_type_counts: Per-gate-type counts.
    """

    total_gates: int
    single_qubit_gates: int
    two_qubit_gates: int
    depth: int
    num_qubits: int
    t_count: int
    cnot_count: int
    estimated_time_ns: float
    circuit_volume: int
    gate_type_counts: dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ResourceEstimate(qubits={self.num_qubits}, "
            f"gates={self.total_gates}, depth={self.depth}, "
            f"t_count={self.t_count}, cnot_count={self.cnot_count})"
        )

    def __str__(self) -> str:
        lines = [
            "ResourceEstimate",
            f"  Qubits:           {self.num_qubits}",
            f"  Total gates:      {self.total_gates}",
            f"  Single-qubit:     {self.single_qubit_gates}",
            f"  Two-qubit:        {self.two_qubit_gates}",
            f"  T gates:          {self.t_count}",
            f"  CNOT gates:       {self.cnot_count}",
            f"  Depth:            {self.depth}",
            f"  Circuit volume:   {self.circuit_volume}",
            f"  Est. time (ns):   {self.estimated_time_ns:.1f}",
        ]
        if self.gate_type_counts:
            lines.append("  Gate counts:")
            for name in sorted(self.gate_type_counts):
                lines.append(f"    {name}: {self.gate_type_counts[name]}")
        return "\n".join(lines)


class ResourceEstimator:
    """Analyzes resource requirements of quantum circuits.

    Args:
        gate_costs: Mapping from gate name to execution time in nanoseconds.
            Defaults to single-qubit gates = 1 ns, two-qubit gates = 10 ns.
    """

    def __init__(self, gate_costs: Optional[dict[str, float]] = None) -> None:
        self._gate_costs: dict[str, float] = dict(
            gate_costs if gate_costs is not None else _DEFAULT_GATE_COSTS
        )

    @property
    def gate_costs(self) -> dict[str, float]:
        """Current gate cost mapping."""
        return dict(self._gate_costs)

    def estimate(self, circuit: QuantumCircuit) -> ResourceEstimate:
        """Analyze a circuit and return resource estimates.

        Args:
            circuit: The quantum circuit to analyze.

        Returns:
            :class:`ResourceEstimate` with detailed resource breakdown.
        """
        gate_type_counts: dict[str, int] = {}
        single_qubit = 0
        two_qubit = 0

        for instr in circuit._gate_instructions:
            name = (
                str(instr[0])
                if QuantumCircuit._is_parameterized_gate(instr)
                else instr[0].name  # type: ignore[union-attr]
            )

            gate_type_counts[name] = gate_type_counts.get(name, 0) + 1

            n_targets = len(circuit._get_targets(instr))
            if n_targets == 1:
                single_qubit += 1
            elif n_targets >= 2:
                two_qubit += 1

        total_gates = single_qubit + two_qubit
        t_count = sum(
            gate_type_counts.get(n, 0) for n in _T_GATE_NAMES
        )
        cnot_count = sum(
            gate_type_counts.get(n, 0) for n in _CNOT_GATE_NAMES
        )

        depth = circuit.depth
        num_qubits = circuit.num_qubits
        volume = num_qubits * depth

        estimated_time = 0.0
        for name, count in gate_type_counts.items():
            cost = self._gate_costs.get(name, 1.0)
            estimated_time += cost * count

        return ResourceEstimate(
            total_gates=total_gates,
            single_qubit_gates=single_qubit,
            two_qubit_gates=two_qubit,
            depth=depth,
            num_qubits=num_qubits,
            t_count=t_count,
            cnot_count=cnot_count,
            estimated_time_ns=estimated_time,
            circuit_volume=volume,
            gate_type_counts=gate_type_counts,
        )

    def __repr__(self) -> str:
        return f"ResourceEstimator(costs={self._gate_costs})"
