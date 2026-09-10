"""Algorithm contract.

The built-in public algorithms follow a single convention: instantiate
with the problem definition, then execute it via ``run()`` (a few older
algorithms expose the legacy entry points ``solve()`` / ``estimate()`` or
``compute_minimum_eigenvalue()``), returning a typed ``*Result`` dataclass
that supports ``to_dict()`` / ``to_json()``.

Algorithms build and execute a :class:`~microquantum.core.circuit.QuantumCircuit`.
Where a circuit can be produced independently it is exposed via
``build_circuit()``.  Execution uses the SDK's backends / executor or the
internal state-vector engine; variational algorithms accept a classical
:class:`~microquantum.optimizers.base.Optimizer`.

The ``Algorithm`` base class also documents the generic problem-driven
lifecycle: ``validate → prepare → solve → post-process → AlgorithmResult``.
Subclassing is optional but recommended for custom algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AlgorithmResult:
    """Generic, serializable result produced by the ``Algorithm.solve``
    lifecycle.

    Existing typed ``*Result`` dataclasses (``VQEResult``, ``GroverResult``,
    etc.) remain the primary outputs of built-in algorithms.  ``AlgorithmResult``
    is the *uniform* container returned by the generic
    ``algorithm.solve(problem, runtime)`` path and carries everything a caller
    may want to inspect or persist.

    Attributes:
        algorithm: Algorithm name.
        problem: Problem name (if a ``Problem`` was supplied).
        solution: Primary output of the algorithm (interpretation varies).
        objective: Best objective / cost value found (``None`` when not applicable).
        optimal_parameters: Flat ``{name: value}`` dict of the best parameters found.
        iterations: Total optimization iterations executed.
        converged: Whether the optimizer converged.
        termination_reason: Human-readable explanation of why execution stopped.
        history: Objective value at each iteration / evaluation.
        optimizer_result: Serialized native optimizer output when available.
        quantum_results: Backend results collected during execution.
        execution_metadata: Free-form metadata (plan names, job ids, etc.).
        config: Algorithm configuration snapshot (ansatz, shots, etc.).
        native: The algorithm's own native result object (e.g. ``VQEResult``).
    """

    algorithm: str = ""
    problem: str = ""
    solution: Any = None
    objective: Optional[float] = None
    optimal_parameters: dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    converged: bool = False
    termination_reason: str = ""
    history: list[float] = field(default_factory=list)
    optimizer_result: Optional[dict[str, Any]] = None
    quantum_results: list[dict[str, Any]] = field(default_factory=list)
    execution_metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    native: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        from .._json import json_safe

        data: dict[str, Any] = {
            "algorithm": self.algorithm,
            "problem": self.problem,
            "solution": self.solution,
            "objective": self.objective,
            "optimal_parameters": dict(self.optimal_parameters),
            "iterations": self.iterations,
            "converged": self.converged,
            "termination_reason": self.termination_reason,
            "history": list(self.history),
            "optimizer_result": self.optimizer_result,
            "quantum_results": list(self.quantum_results),
            "execution_metadata": dict(self.execution_metadata),
            "config": dict(self.config),
        }
        return json_safe(data)  # type: ignore[no-any-return]

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        from .._json import json_string

        return json_string(self.to_dict())


class Algorithm(ABC):
    """Lightweight contract shared by all public quantum algorithms.

    Subclassing is optional — this class documents the convention used by
    the built-in algorithms rather than forcing a uniform implementation.

    For the generic problem-driven lifecycle, override :meth:`solve` to
    return an :class:`AlgorithmResult`.  The default ``solve`` raises
    ``NotImplementedError`` to clearly signal that a subclass has not
    implemented the generic path.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm identifier."""

    # -- lifecycle hooks (all concrete, non-breaking) -------------------

    def validate(self, problem: Any) -> list[str]:
        """Return a list of validation errors (empty means *ok*).

        Subclasses should override this to check that ``problem`` is
        compatible with the algorithm before execution.
        """
        return []

    def solve(self, problem: Any, runtime: Any = None) -> AlgorithmResult:
        """Generic problem-driven entry point.

        Subclasses override this to accept a ``Problem`` and (optionally)
        an :class:`~microquantum.runtime.ExecutionRuntime`, returning a
        serialized :class:`AlgorithmResult`.

        Raises:
            NotImplementedError: Always in the base implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement solve(problem, runtime)"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}')"


__all__ = ["Algorithm", "AlgorithmResult"]