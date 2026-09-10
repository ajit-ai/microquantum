"""Standard result contract for microquantum executions and solvers.

``Result`` is the public, serializable schema that represents the outcome
of a generic quantum execution or solver run: the computed solution,
confidence, fidelity, timing, optional reference baseline, and an
execution trace. It is intentionally domain-agnostic — applications may
attach their own meaning to the fields without the SDK assuming one.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

_UNSET = object()


@dataclass(init=False)
class Result:
    """Standardized outcome of a generic quantum execution / solver run.

    Attributes:
        problem: Machine-readable problem identifier.
        solution: Computed solution produced by the execution.
        confidence: Confidence score in [0, 1].
        fidelity: Circuit/quantum fidelity achieved, if measurable.
        qubit_count: Number of qubits used by the execution.
        runtime_ms: End-to-end execution runtime in milliseconds.
        baseline: Reference (classical) result, if available.
        quantum_trace: Optional execution trace (shots, error budget, etc.).
        duration: ISO 8601 timestamp of completion.
    """

    problem: str
    solution: Any = None
    confidence: float = 1.0
    fidelity: Optional[float] = None
    qubit_count: int = 0
    runtime_ms: float = 0.0
    baseline: Any = None
    quantum_trace: dict[str, Any] = field(default_factory=dict)
    duration: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __init__(
        self,
        problem: str,
        solution: Any = None,
        confidence: float = 1.0,
        fidelity: Optional[float] = None,
        qubit_count: int = 0,
        runtime_ms: float = 0.0,
        baseline: Any = None,
        quantum_trace: Optional[dict[str, Any]] = None,
        duration: Optional[str] = None,
        *,
        decision: Any = _UNSET,
        classical_baseline: Any = _UNSET,
    ) -> None:
        """Initialize a result.

        The keyword-only ``decision`` and ``classical_baseline`` arguments
        are deprecated aliases for ``solution`` and ``baseline``.
        """
        if decision is not _UNSET:
            warnings.warn(
                "Result(decision=...) is deprecated; use solution=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if solution is None:
                solution = decision
        if classical_baseline is not _UNSET:
            warnings.warn(
                "Result(classical_baseline=...) is deprecated; use baseline=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if baseline is None:
                baseline = classical_baseline
        self.problem = problem
        self.solution = solution
        self.confidence = confidence
        self.fidelity = fidelity
        self.qubit_count = qubit_count
        self.runtime_ms = runtime_ms
        self.baseline = baseline
        self.quantum_trace = {} if quantum_trace is None else quantum_trace
        if duration is None:
            self.duration = datetime.now(timezone.utc).isoformat()
        else:
            self.duration = duration

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
            "solution": self._json_safe(self.solution),
            "confidence": float(self.confidence),
            "fidelity": float(self.fidelity) if self.fidelity is not None else None,
            "qubit_count": int(self.qubit_count),
            "runtime_ms": float(self.runtime_ms),
            "baseline": self._json_safe(self.baseline),
            "quantum_trace": self._json_safe(self.quantum_trace),
            "duration": self.duration,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @property
    def improved_over_baseline(self) -> Optional[bool]:
        """True if the quantum solution beats the reference baseline.

        Only meaningful when both ``solution`` and ``baseline`` are numeric,
        or ``quantum_trace`` carries a comparable metric.
        """
        q = self.solution
        c = self.baseline
        if isinstance(q, (int, float)) and isinstance(c, (int, float)):
            return bool(q <= c if self.quantum_trace.get("lower_is_better") else q >= c)
        metric = self.quantum_trace.get("metric")
        if metric is not None:
            qm = metric.get("quantum_value")
            cm = metric.get("classical_value")
            if isinstance(qm, (int, float)) and isinstance(cm, (int, float)):
                return bool(qm >= cm)
        return None

    @property
    def improved_over_classical(self) -> Optional[bool]:
        """Deprecated alias for :attr:`improved_over_baseline`."""
        warnings.warn(
            "Result.improved_over_classical is deprecated; "
            "use Result.improved_over_baseline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.improved_over_baseline

    @property
    def decision(self) -> Any:
        """Deprecated alias for :attr:`solution`."""
        warnings.warn(
            "Result.decision is deprecated; use Result.solution instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.solution

    @decision.setter
    def decision(self, value: Any) -> None:
        """Deprecated setter alias for :attr:`solution`."""
        warnings.warn(
            "Result.decision is deprecated; use Result.solution instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.solution = value

    @property
    def classical_baseline(self) -> Any:
        """Deprecated alias for :attr:`baseline`."""
        warnings.warn(
            "Result.classical_baseline is deprecated; use Result.baseline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.baseline

    @classical_baseline.setter
    def classical_baseline(self, value: Any) -> None:
        """Deprecated setter alias for :attr:`baseline`."""
        warnings.warn(
            "Result.classical_baseline is deprecated; use Result.baseline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.baseline = value