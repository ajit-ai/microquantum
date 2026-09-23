"""Cost models and alias analysis for the IR compiler.

:class:`CostModel` assigns execution costs to gate names so the
:class:`Compiler` can compare compilation choices; :class:`AliasAnalysis`
is an :class:`IRPass` that records which qubits interact, for use by
routing and cancellation passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .circuit_ir import IRCircuit
from .nodes import Gate
from .passes import IRPass

__all__ = [
    "CostModel",
    "AliasAnalysis",
]


@dataclass(frozen=True)
class CostModel:
    """Per-gate execution costs with a fallback default.

    Attributes:
        gate_costs: Mapping of lowercase gate name to cost.
        default_cost: Cost used for unlisted gates.
    """

    gate_costs: Mapping[str, float] = field(default_factory=dict)
    default_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.default_cost < 0:
            raise ValueError("default_cost must be >= 0")
        for name, cost in self.gate_costs.items():
            if cost < 0:
                raise ValueError(f"Negative cost for gate '{name}': {cost}")

    def cost_of(self, gate_name: str) -> float:
        """Cost of a single gate by name (case-insensitive)."""
        return float(self.gate_costs.get(gate_name.lower(), self.default_cost))

    def total(self, ir: IRCircuit) -> float:
        """Total cost of all gates in *ir* (including nested blocks)."""
        return float(sum(self.cost_of(op.name) for op in ir.walk() if isinstance(op, Gate)))

    def breakdown(self, ir: IRCircuit) -> dict[str, float]:
        """Per-gate-type cost breakdown, sorted by gate name."""
        counts: dict[str, int] = {}
        for op in ir.walk():
            if isinstance(op, Gate):
                counts[op.name] = counts.get(op.name, 0) + 1
        return {name: counts[name] * self.cost_of(name) for name in sorted(counts)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "gate_costs": {str(k): float(v) for k, v in self.gate_costs.items()},
            "default_cost": self.default_cost,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostModel:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            gate_costs={str(k): float(v) for k, v in data.get("gate_costs", {}).items()},
            default_cost=float(data.get("default_cost", 1.0)),
        )


class AliasAnalysis(IRPass):
    """Record qubit interaction pairs in an IR circuit.

    The circuit passes through unchanged; the analysis result
    (``{qubit: {partners}}``) is available via :meth:`result` after
    :meth:`run`, and is also merged into the pass manager metadata when
    composed through :class:`IRPassManager`.
    """

    def __init__(self) -> None:
        self._pairs: dict[int, set[int]] = {}

    @property
    def name(self) -> str:
        """Pass name."""
        return "alias-analysis"

    def run(self, ir: IRCircuit) -> IRCircuit:
        """Analyze *ir* and return it unchanged."""
        pairs: dict[int, set[int]] = {}
        for op in ir.walk():
            qubits = list(getattr(op, "qubits", ()))
            for index, first in enumerate(qubits):
                partners = pairs.setdefault(int(first), set())
                for second in qubits[index + 1 :]:
                    partners.add(int(second))
                    pairs.setdefault(int(second), set()).add(int(first))
        self._pairs = pairs
        return ir

    def result(self) -> dict[int, set[int]]:
        """Interaction map from the most recent :meth:`run`."""
        return {qubit: set(partners) for qubit, partners in self._pairs.items()}

    def interacting_pairs(self) -> list[tuple[int, int]]:
        """Sorted ``(a, b)`` interaction pairs with ``a < b``."""
        pairs: set[tuple[int, int]] = set()
        for qubit, partners in self._pairs.items():
            for partner in partners:
                pairs.add((min(qubit, partner), max(qubit, partner)))
        return sorted(pairs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the latest analysis result."""
        return {
            str(qubit): sorted(partners) for qubit, partners in self._pairs.items()
        }
