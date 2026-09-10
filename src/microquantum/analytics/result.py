"""Standard decision-result contract for microquantum solvers.

``Result`` is the public, serializable schema that downstream
proprietary solvers (e.g. Fraud Desk, Screening Desk) consume and
extend. Publishing it in the SDK keeps the contract stable across
open and proprietary layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


@dataclass
class Result:
    """Standardized outcome of a quantum business-solver run.

    Attributes:
        problem: Machine-readable problem identifier (e.g. "fraud_detection").
        decision: Model-readable decision produced by the solver.
        confidence: Confidence score in [0, 1].
        fidelity: Circuit/quantum fidelity achieved, if measurable.
        qubit_count: Number of qubits used by the solver.
        runtime_ms: End-to-end solver runtime in milliseconds.
        classical_baseline: Corresponding classical result, if available.
        quantum_trace: Optional execution trace (shots, error budget, etc.).
        duration: ISO 8601 timestamp of completion.
    """

    problem: str
    decision: Any
    confidence: float = 1.0
    fidelity: Optional[float] = None
    qubit_count: int = 0
    runtime_ms: float = 0.0
    classical_baseline: Any = None
    quantum_trace: dict[str, Any] = field(default_factory=dict)
    duration: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Recursively convert numpy types to JSON-safe Python types."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): Result._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Result._json_safe(v) for v in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "problem": self.problem,
            "decision": self._json_safe(self.decision),
            "confidence": float(self.confidence),
            "fidelity": float(self.fidelity) if self.fidelity is not None else None,
            "qubit_count": int(self.qubit_count),
            "runtime_ms": float(self.runtime_ms),
            "classical_baseline": self._json_safe(self.classical_baseline),
            "quantum_trace": self._json_safe(self.quantum_trace),
            "duration": self.duration,
        }

    def to_json(self) -> dict[str, Any]:
        """Alias of :meth:`to_dict` for API consumers."""
        return self.to_dict()

    @property
    def improved_over_classical(self) -> Optional[bool]:
        """True if the quantum decision beat the classical baseline.

        Only meaningful when both ``decision`` and ``classical_baseline``
        are numeric, or ``quantum_trace`` carries a comparable metric.
        """
        q = self.decision
        c = self.classical_baseline
        if isinstance(q, (int, float)) and isinstance(c, (int, float)):
            return bool(q <= c if self.quantum_trace.get("lower_is_better") else q >= c)
        metric = self.quantum_trace.get("metric")
        if metric is not None:
            qm = metric.get("quantum_value")
            cm = metric.get("classical_value")
            if isinstance(qm, (int, float)) and isinstance(cm, (int, float)):
                return bool(qm >= cm)
        return None