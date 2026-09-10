"""Gradient-descent and Adam optimizers."""

from __future__ import annotations

import math
from typing import Callable, Optional

from ..core.parameter import Parameter
from .base import Optimizer


class GradientDescent(Optimizer):
    """First-order gradient descent optimizer.

    Updates parameters as: theta_new = theta - lr * grad.

    Args:
        learning_rate: Step size for parameter updates.
        max_iter: Maximum number of optimization iterations.
        tol: Tolerance for convergence (change in objective < tol).
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._tol = tol

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def _step(
        self,
        params: dict[Parameter, float],
        grads: dict[Parameter, float],
        iteration: int,
        cost_fn: Optional[Callable[[dict[Parameter, float]], float]] = None,
    ) -> dict[Parameter, float]:
        new_params: dict[Parameter, float] = {}
        for p, val in params.items():
            g = grads.get(p, 0.0)
            new_params[p] = val - self._learning_rate * g
        return new_params

    def _converged(self, history: list[float]) -> bool:
        if len(history) < 2:
            return False
        return abs(history[-1] - history[-2]) < self._tol


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer.

    Maintains exponential moving averages of gradient (first moment)
    and squared gradient (second moment) for adaptive learning rates.

    Args:
        learning_rate: Base step size.
        beta1: Exponential decay rate for the first moment.
        beta2: Exponential decay rate for the second moment.
        epsilon: Small constant for numerical stability.
        max_iter: Maximum number of optimization iterations.
        tol: Tolerance for convergence.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> None:
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._tol = tol
        self._m: dict[Parameter, float] = {}
        self._v: dict[Parameter, float] = {}

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def _step(
        self,
        params: dict[Parameter, float],
        grads: dict[Parameter, float],
        iteration: int,
        cost_fn: Optional[Callable[[dict[Parameter, float]], float]] = None,
    ) -> dict[Parameter, float]:
        t = iteration + 1
        bc1 = 1 - self._beta1 ** t
        bc2 = 1 - self._beta2 ** t

        new_params: dict[Parameter, float] = {}
        for p, val in params.items():
            g = grads.get(p, 0.0)

            # Update biased first moment
            self._m[p] = self._beta1 * self._m.get(p, 0.0) + (1 - self._beta1) * g
            # Update biased second raw moment
            self._v[p] = self._beta2 * self._v.get(p, 0.0) + (1 - self._beta2) * g * g

            # Bias-corrected estimates
            m_hat = self._m[p] / bc1
            v_hat = self._v[p] / bc2

            new_params[p] = val - self._learning_rate * m_hat / (math.sqrt(v_hat) + self._epsilon)
        return new_params

    def _converged(self, history: list[float]) -> bool:
        if len(history) < 2:
            return False
        return abs(history[-1] - history[-2]) < self._tol
