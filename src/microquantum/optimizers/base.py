"""Abstract base class for classical optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from .._json import JSONSerializable
from ..core.parameter import Parameter


@dataclass
class OptimizerResult(JSONSerializable):
    """Container for optimizer output.

    Attributes:
        optimal_parameters: Parameter values at the optimum.
        optimal_value: Best objective value found.
        history: Objective values at each iteration.
        iterations: Total number of iterations executed.
        converged: Whether the optimizer reached convergence criteria.
    """

    optimal_parameters: dict[Parameter, float] = field(default_factory=dict)
    optimal_value: float = float("inf")
    history: list[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False


class Optimizer(ABC):
    """Abstract base class for classical optimizers.

    Subclasses must implement ``_step`` which updates parameters
    given a current value, gradient, and iteration state.
    """

    def minimize(
        self,
        cost_fn: Callable[[dict[Parameter, float]], float],
        gradient_fn: Optional[Callable[[dict[Parameter, float]], dict[Parameter, float]]] = None,
        initial_params: Optional[dict[Parameter, float]] = None,
    ) -> OptimizerResult:
        """Minimize a cost function.

        Args:
            cost_fn: Objective function mapping parameter dict to scalar.
            gradient_fn: Gradient function (optional for gradient-free optimizers).
            initial_params: Starting parameter values.

        Returns:
            OptimizerResult with the best parameters found.
        """
        if initial_params is None:
            raise ValueError("initial_params is required")
        params = dict(initial_params)
        result = OptimizerResult()
        result.history.append(cost_fn(params))

        for i in range(self.max_iter):
            grads = gradient_fn(params) if gradient_fn is not None else {}
            params = self._step(params, grads, i, cost_fn=cost_fn)
            value = cost_fn(params)
            result.history.append(value)
            result.iterations = i + 1

            if self._converged(result.history):
                result.optimal_parameters = params
                result.optimal_value = value
                result.converged = True
                return result

        result.optimal_parameters = params
        result.optimal_value = result.history[-1]
        result.converged = False
        return result

    @abstractmethod
    def _step(
        self,
        params: dict[Parameter, float],
        grads: dict[Parameter, float],
        iteration: int,
        cost_fn: Optional[Callable[[dict[Parameter, float]], float]] = None,
    ) -> dict[Parameter, float]:
        """Perform a single parameter update step."""
        ...

    @abstractmethod
    def _converged(self, history: list[float]) -> bool:
        """Check if the optimizer has converged."""
        ...

    @property
    @abstractmethod
    def max_iter(self) -> int:
        """Maximum number of iterations."""
        ...
