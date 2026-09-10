"""Gradient-free optimizers: COBYLA and Nelder-Mead."""

from __future__ import annotations

import copy
import math
from typing import Callable, Optional

from ..core.parameter import Parameter
from .base import Optimizer


class COBYLA(Optimizer):
    """Constrained Optimization BY Linear Approximation.

    A gradient-free optimizer that builds linear approximations
    of the objective function using simplex evaluations.
    Well-suited for noisy quantum cost landscapes.

    Args:
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence (change in objective < tol).
        rhobeg: Initial simplex size.
        catol: Tolerance for constraint violation (unused for unconstrained).
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        rhobeg: float = 1.0,
        catol: float = 1e-4,
    ) -> None:
        self._max_iter = max_iter
        self._tol = tol
        self._rhobeg = rhobeg
        self._catol = catol

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
        if cost_fn is None:
            raise ValueError("COBYLA requires cost_fn")

        keys = list(params.keys())
        n = len(keys)
        if n == 0:
            return params

        # Build simplex: center + n directional points
        simplex: list[dict[Parameter, float]] = []
        f_values: list[float] = []

        # Evaluate center
        f_center = cost_fn(params)
        simplex.append(dict(params))
        f_values.append(f_center)

        # Evaluate simplex points
        step = self._rhobeg / (iteration + 1)
        for i in range(n):
            point = dict(params)
            point[keys[i]] += step
            simplex.append(point)
            f_values.append(cost_fn(point))

        # Find best point
        best_idx = min(range(len(f_values)), key=lambda i: f_values[i])
        best_point = dict(simplex[best_idx])

        # Simplex centroid (excluding worst)
        worst_idx = max(range(len(f_values)), key=lambda i: f_values[i])
        centroid: dict[Parameter, float] = {k: 0.0 for k in keys}
        for i, pt in enumerate(simplex):
            if i == worst_idx:
                continue
            for k in keys:
                centroid[k] += pt[k]
        for k in keys:
            centroid[k] /= n

        # Try reflection: reflect worst through centroid
        reflected: dict[Parameter, float] = {}
        for k in keys:
            reflected[k] = 2 * centroid[k] - simplex[worst_idx][k]

        f_reflected = cost_fn(reflected)
        if f_reflected < f_values[worst_idx]:
            # Accept reflected point
            return reflected

        # Fallback: try best point (simplex shrink already handled by step size)
        return best_point

    def _converged(self, history: list[float]) -> bool:
        if len(history) < 2:
            return False
        return abs(history[-1] - history[-2]) < self._tol


class NelderMead(Optimizer):
    """Nelder-Mead simplex method.

    A derivative-free optimization algorithm that maintains a simplex
    of n+1 points in parameter space and iteratively improves it
    through reflection, expansion, contraction, and shrink operations.

    Args:
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
        alpha: Reflection coefficient (default 1.0).
        gamma: Expansion coefficient (default 2.0).
        rho: Contraction coefficient (default 0.5).
        sigma: Shrink coefficient (default 0.5).
    """

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> None:
        self._max_iter = max_iter
        self._tol = tol
        self._alpha = alpha
        self._gamma = gamma
        self._rho = rho
        self._sigma = sigma
        self._simplex: Optional[list[dict[Parameter, float]]] = None
        self._f_simplex: Optional[list[float]] = None

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def _init_simplex(
        self, params: dict[Parameter, float], cost_fn: Callable
    ) -> None:
        """Initialize simplex around starting point."""
        keys = list(params.keys())
        n = len(keys)
        self._simplex = [dict(params)]
        self._f_simplex = [cost_fn(params)]

        for i in range(n):
            point = dict(params)
            step = 0.25 if point[keys[i]] == 0.0 else max(0.25, 0.05 * abs(point[keys[i]]))
            point[keys[i]] += step
            self._simplex.append(point)
            self._f_simplex.append(cost_fn(point))

    def _step(
        self,
        params: dict[Parameter, float],
        grads: dict[Parameter, float],
        iteration: int,
        cost_fn: Optional[Callable[[dict[Parameter, float]], float]] = None,
    ) -> dict[Parameter, float]:
        if cost_fn is None:
            raise ValueError("Nelder-Mead requires cost_fn")

        keys = list(params.keys())
        n = len(keys)
        if n == 0:
            return params

        # Initialize simplex on first call
        if self._simplex is None or len(self._simplex) != n + 1:
            self._init_simplex(params, cost_fn)

        assert self._simplex is not None
        assert self._f_simplex is not None
        simplex = self._simplex
        f_vals = self._f_simplex

        # Sort by function value
        indices = sorted(range(len(f_vals)), key=lambda i: f_vals[i])

        best_idx = indices[0]
        worst_idx = indices[-1]
        second_worst_idx = indices[-2] if len(indices) > 1 else best_idx

        # Centroid of all points except worst
        centroid: dict[Parameter, float] = {k: 0.0 for k in keys}
        for i in indices[:-1]:
            for k in keys:
                centroid[k] += simplex[i][k]
        for k in keys:
            centroid[k] /= n

        # Reflection
        reflected: dict[Parameter, float] = {}
        for k in keys:
            reflected[k] = centroid[k] + self._alpha * (centroid[k] - simplex[worst_idx][k])
        f_reflected = cost_fn(reflected)

        if f_vals[second_worst_idx] <= f_reflected <= f_vals[best_idx]:
            simplex[worst_idx] = reflected
            f_vals[worst_idx] = f_reflected
        elif f_reflected < f_vals[best_idx]:
            expanded: dict[Parameter, float] = {}
            for k in keys:
                expanded[k] = centroid[k] + self._gamma * (reflected[k] - centroid[k])
            f_expanded = cost_fn(expanded)
            if f_expanded < f_reflected:
                simplex[worst_idx] = expanded
                f_vals[worst_idx] = f_expanded
            else:
                simplex[worst_idx] = reflected
                f_vals[worst_idx] = f_reflected
        else:
            contracted: dict[Parameter, float] = {}
            for k in keys:
                contracted[k] = centroid[k] + self._rho * (simplex[worst_idx][k] - centroid[k])
            f_contracted = cost_fn(contracted)
            if f_contracted < f_vals[worst_idx]:
                simplex[worst_idx] = contracted
                f_vals[worst_idx] = f_contracted
            else:
                for i in range(1, len(simplex)):
                    for k in keys:
                        simplex[i][k] = simplex[best_idx][k] + self._sigma * (
                            simplex[i][k] - simplex[best_idx][k]
                        )
                    f_vals[i] = cost_fn(simplex[i])

        # Return actual best after all updates
        best_after = min(range(len(f_vals)), key=lambda i: f_vals[i])
        return dict(simplex[best_after])

    def _converged(self, history: list[float]) -> bool:
        if self._simplex is None or self._f_simplex is None:
            return False
        if len(self._f_simplex) < 2:
            return False
        f_spread = max(self._f_simplex) - min(self._f_simplex)
        # Also check vertex spread in parameter space
        keys = list(self._simplex[0].keys())
        max_param_spread = 0.0
        for k in keys:
            vals = [s[k] for s in self._simplex]
            max_param_spread = max(max_param_spread, max(vals) - min(vals))
        return f_spread < self._tol and max_param_spread < self._tol
