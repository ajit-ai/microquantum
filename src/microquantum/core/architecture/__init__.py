"""Hardware-independent quantum architecture descriptions.

:class:`QuantumArchitecture` describes qubit count, connectivity,
native gates, gate capabilities (durations/errors), measurement and
dynamic-circuit support.  Usable by the transpiler without any hardware
provider dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from microquantum.core.coupling import CouplingMap

__all__ = [
    "CouplingMap",
    "QubitTopology",
    "NativeGateSet",
    "QuantumArchitecture",
    "linear_architecture",
    "fully_connected_architecture",
]


@dataclass(frozen=True)
class QubitTopology:
    """Qubit connectivity graph."""

    num_qubits: int
    edges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        for a, b in self.edges:
            if not 0 <= a < self.num_qubits or not 0 <= b < self.num_qubits:
                raise ValueError(f"Edge ({a}, {b}) out of range for {self.num_qubits} qubits")
            if a == b:
                raise ValueError("Self-loops are not allowed")

    @property
    def pairs(self) -> set[frozenset[int]]:
        """Undirected edge set."""
        return {frozenset(e) for e in self.edges}

    def is_connected(self, a: int, b: int) -> bool:
        """True when qubits ``a`` and ``b`` are directly coupled."""
        return frozenset((a, b)) in self.pairs

    def to_coupling_map(self) -> CouplingMap:
        """Convert to the canonical :class:`CouplingMap`."""
        return CouplingMap(list(self.edges))

    @classmethod
    def linear(cls, num_qubits: int) -> QubitTopology:
        """1D chain topology."""
        return cls(num_qubits, tuple((i, i + 1) for i in range(num_qubits - 1)))

    @classmethod
    def fully_connected(cls, num_qubits: int) -> QubitTopology:
        """All-to-all topology."""
        edges = tuple((i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits))
        return cls(num_qubits, edges)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {"num_qubits": self.num_qubits, "edges": [list(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QubitTopology:
        """Deserialize from :meth:`to_dict` output."""
        raw_edges = data.get("edges", [])
        assert isinstance(raw_edges, list)
        edges = tuple((int(a), int(b)) for a, b in raw_edges)
        return cls(int(data["num_qubits"]), edges)


@dataclass(frozen=True)
class NativeGateSet:
    """Allowed native gates plus optional durations and error rates."""

    single_qubit: frozenset[str] = frozenset({"h", "rz", "ry"})
    two_qubit: frozenset[str] = frozenset({"cx"})
    gate_durations_ns: Mapping[str, float] = field(default_factory=dict)
    gate_errors: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for table in (self.gate_durations_ns, self.gate_errors):
            for key, value in table.items():
                if value < 0:
                    raise ValueError(f"Negative metadata for gate '{key}': {value}")

    @property
    def basis_gates(self) -> frozenset[str]:
        """Union of single- and two-qubit gate names."""
        return self.single_qubit | self.two_qubit

    def supports(self, gate_name: str) -> bool:
        """True when *gate_name* is native."""
        return gate_name.lower() in self.basis_gates


@dataclass(frozen=True)
class QuantumArchitecture:
    """Full hardware-independent architecture description."""

    num_qubits: int
    topology: QubitTopology
    native_gates: NativeGateSet = field(default_factory=NativeGateSet)
    supports_measurement: bool = True
    supports_reset: bool = True
    supports_dynamic_circuits: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        if self.topology.num_qubits != self.num_qubits:
            raise ValueError("topology qubit count does not match num_qubits")

    def is_native(self, gate_name: str) -> bool:
        """True when *gate_name* is natively supported."""
        return self.native_gates.supports(gate_name)

    def requires_routing(self, qubit_pair: tuple[int, int]) -> bool:
        """True when a two-qubit gate on *qubit_pair* needs routing."""
        a, b = qubit_pair
        if not 0 <= a < self.num_qubits or not 0 <= b < self.num_qubits:
            raise ValueError("qubit indices out of range")
        return not self.topology.is_connected(a, b)

    def validate_circuit_qubits(self, num_circuit_qubits: int) -> None:
        """Raise when a circuit does not fit this architecture."""
        if num_circuit_qubits > self.num_qubits:
            raise ValueError(f"Circuit needs {num_circuit_qubits} qubits, architecture has {self.num_qubits}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        from microquantum._json import json_safe  # noqa: PLC0415

        return {
            "version": 1,
            "num_qubits": self.num_qubits,
            "topology": self.topology.to_dict(),
            "native_gates": sorted(self.native_gates.basis_gates),
            "supports_measurement": self.supports_measurement,
            "supports_reset": self.supports_reset,
            "supports_dynamic_circuits": self.supports_dynamic_circuits,
            "metadata": json_safe(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuantumArchitecture:
        """Deserialize from :meth:`to_dict` output."""
        topology_data = data["topology"]
        assert isinstance(topology_data, dict)
        topo = QubitTopology.from_dict(topology_data)
        gates = NativeGateSet(
            single_qubit=frozenset(g for g in data.get("native_gates", []) if g in ("h", "x", "y", "z", "rx", "ry", "rz", "s", "t", "i")),
            two_qubit=frozenset(g for g in data.get("native_gates", []) if g not in ("h", "x", "y", "z", "rx", "ry", "rz", "s", "t", "i")),
        )
        return cls(
            num_qubits=int(data["num_qubits"]),
            topology=topo,
            native_gates=gates,
            supports_measurement=bool(data.get("supports_measurement", True)),
            supports_reset=bool(data.get("supports_reset", True)),
            supports_dynamic_circuits=bool(data.get("supports_dynamic_circuits", False)),
            metadata=dict(data.get("metadata", {})),
        )


def linear_architecture(num_qubits: int, gate_set: NativeGateSet | None = None) -> QuantumArchitecture:
    """Linear-chain architecture with a default native gate set."""
    return QuantumArchitecture(num_qubits, QubitTopology.linear(num_qubits), gate_set or NativeGateSet())


def fully_connected_architecture(
    num_qubits: int, gate_set: NativeGateSet | None = None
) -> QuantumArchitecture:
    """All-to-all architecture with a default native gate set."""
    return QuantumArchitecture(
        num_qubits, QubitTopology.fully_connected(num_qubits), gate_set or NativeGateSet()
    )


def describe_edges(edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Normalize an edge list for serialization."""
    return [[int(a), int(b)] for a, b in edges]
