"""Base analytics interface for generic quantum analysis.

Provides a common interface for data ingestion, quantum algorithm
execution, and analysis result production.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .._json import json_safe, json_string


@dataclass
class AnalysisResult:
    """Result from an analytics analysis.

    Contains the computed solution, quantum metrics, a classical
    reference comparison, and associated metadata.
    """
    solution: dict[str, Any] = field(default_factory=dict)
    quantum_metrics: dict[str, Any] = field(default_factory=dict)
    classical_metrics: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "solution": json_safe(self.solution),
            "quantum_metrics": json_safe(self.quantum_metrics),
            "classical_metrics": json_safe(self.classical_metrics),
            "comparison": json_safe(self.comparison),
            "metadata": json_safe(self.metadata),
            "success": self.success,
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())


class BaseAnalytics(ABC):
    """Base class for analytics modules.

    Provides a common interface for data ingestion, quantum algorithm
    execution, and analysis output.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the analytics module."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the analytics module solves."""
        ...

    @property
    @abstractmethod
    def required_inputs(self) -> list[str]:
        """List of required input parameters."""
        ...

    @property
    @abstractmethod
    def algorithm(self) -> str:
        """Quantum algorithm used."""
        ...

    @abstractmethod
    def analyze(self, **kwargs: Any) -> AnalysisResult:
        """Run the analysis.

        Args:
            **kwargs: Input parameters specific to the analytics module.

        Returns:
            AnalysisResult with solution, metrics, and comparison.
        """
        ...

    def _compute_comparison(
        self,
        quantum_value: float,
        classical_value: float,
        metric_name: str = "value",
        higher_is_better: bool = True,
    ) -> dict[str, Any]:
        """Compute comparison between quantum and classical results.

        Args:
            quantum_value: Quantum result value.
            classical_value: Classical result value.
            metric_name: Name of the metric being compared.
            higher_is_better: If True, higher values are better.

        Returns:
            Dictionary with comparison metrics.
        """
        if classical_value == 0:
            # Avoid division-by-zero and non-finite JSON values. Use a large
            # finite sentinel when only the quantum side is non-zero.
            improvement_pct = (
                100.0 * quantum_value
                if quantum_value >= 0
                else -100.0 * abs(quantum_value)
            )
        else:
            improvement_pct = (quantum_value - classical_value) / abs(classical_value) * 100

        if higher_is_better:
            quantum_wins = quantum_value >= classical_value
        else:
            quantum_wins = quantum_value <= classical_value

        return {
            "metric_name": metric_name,
            "quantum_value": quantum_value,
            "classical_value": classical_value,
            "improvement_pct": improvement_pct,
            "quantum_wins": quantum_wins,
        }
