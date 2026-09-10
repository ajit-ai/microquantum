"""Quasi-Newton optimizers.

:class:`BFGS` implements the Broyden–Fletcher–Goldfarb–Shanno
algorithm in pure NumPy — no SciPy required.  It uses the standard
BFGS inverse-Hessian update formula with a backtracking line search
that satisfies the Armijo (sufficient decrease) condition.

When SciPy is available, :class:`L_BFGS_B` wraps ``scipy.optimize.minimize``
with the ``L-BFGS-B`` method for a more robust large-scale solver.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from ..core.parameter import Parameter
from .base import Optimizer, OptimizerResult


class BFGS(Optimizer):
    """BFGS quasi-Newton optimizer (pure NumPy).

    Maintains the inverse Hessian approximation via the standard BFGS
    rank-2 update and uses a backtracking Armijo line search.

    Args:
        max_iter: Maximum number of optimization iterations.
        tol: Convergence tolerance on the gradient norm.
        c1: Armijo sufficient-decrease constant.
        line_search_steps: Maximum number of backtracking steps.
    """

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        c1: float = 1e-4,
        line_search_steps: int = 30,
    ) -> None:
        self._max_iter = max_iter
        self._tol = tol
        self._c1 = c1
        self._ls_steps = line_search_steps
        self._H_inv: Optional[np.ndarray] = None

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def minimize(
        self,
        cost_fn: Callable[[dict[Parameter, float]], float],
        gradient_fn: Optional[Callable[[dict[Parameter, float]], dict[Parameter, float]]] = None,
        initial_params: Optional[dict[Parameter, float]] = None,
    ) -> OptimizerResult:
        """Run full BFGS optimization.

        Overrides :meth:`Optimizer.minimize` to implement the complete
        BFGS loop with inverse-Hessian tracking.
        """
        if initial_params is None:
            raise ValueError("initial_params is required")
        keys = list(initial_params.keys())
        n = len(keys)
        params = dict(initial_params)
        result = OptimizerResult()
        result.history.append(cost_fn(params))

        H_k = np.eye(n, dtype=float)

        def _grad(p: dict[Parameter, float]) -> np.ndarray:
            if gradient_fn is not None:
                return np.array([gradient_fn(p).get(k, 0.0) for k in keys])
            eps = max(1e-7, math.sqrt(np.finfo(float).eps))
            f0 = cost_fn(p)
            g = np.zeros(n, dtype=float)
            for idx, key in enumerate(keys):
                fwd = dict(p)
                fwd[key] = p[key] + eps
                g[idx] = (cost_fn(fwd) - f0) / eps
            return g

        g = _grad(params)
        for i in range(self._max_iter):
            if np.linalg.norm(g) < self._tol:
                result.optimal_parameters = dict(params)
                result.optimal_value = result.history[-1]
                result.converged = True
                result.iterations = i
                return result

            p_dir = -H_k @ g

            # backtracking line search (Armijo)
            alpha = 1.0
            dg = float(g @ p_dir)
            f0 = result.history[-1]
            for _ in range(self._ls_steps):
                trial = {k: params[k] + alpha * p_dir[j] for j, k in enumerate(keys)}
                f_trial = cost_fn(trial)
                if f_trial <= f0 + self._c1 * alpha * dg:
                    break
                alpha *= 0.5

            params_new = {k: params[k] + alpha * p_dir[j] for j, k in enumerate(keys)}
            f_new = cost_fn(params_new)
            result.history.append(f_new)
            result.iterations = i + 1

            if self._converged(result.history):
                result.optimal_parameters = dict(params_new)
                result.optimal_value = f_new
                result.converged = True
                return result

            g_new = _grad(params_new)
            s = alpha * p_dir
            y = g_new - g
            rho_denom = float(y @ s)
            if abs(rho_denom) > 1e-12:
                rho = 1.0 / rho_denom
                identity = np.eye(n, dtype=float)
                V = identity - rho * np.outer(s, y)
                W = identity + rho * np.outer(y, s)
                H_k = V @ H_k @ W + rho * np.outer(s, s)

            params = params_new
            g = g_new

        result.optimal_parameters = dict(params)
        result.optimal_value = result.history[-1]
        result.converged = False
        return result

    def _step(
        self,
        params: dict[Parameter, float],
        grads: dict[Parameter, float],
        iteration: int,
        cost_fn: Optional[Callable[[dict[Parameter, float]], float]] = None,
    ) -> dict[Parameter, float]:
        raise NotImplementedError("BFGS uses its own minimize() loop")

    def _converged(self, history: list[float]) -> bool:
        if len(history) < 2:
            return False
        return abs(history[-1] - history[-2]) < self._tol


def _scipy_available() -> bool:
    try:
        import scipy.optimize  # noqa: F401
        return True
    except ImportError:
        return False


class L_BFGS_B(Optimizer):
    """L-BFGS-B wrapper around SciPy (optional dependency).

    Requires ``scipy`` to be installed.  Raises ``ImportError`` on
    construction if SciPy is missing.

    Args:
        max_iter: Maximum number of optimization iterations.
        tol: Optimality tolerance.
        maxcor: Number of past updates stored (memory parameter).
    """

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        maxcor: int = 20,
    ) -> None:
        if not _scipy_available():
            raise ImportError(
                "L_BFGS_B requires scipy; install with: pip install scipy"
            )
        self._max_iter = max_iter
        self._tol = tol
        self._maxcor = maxcor

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def minimize(
        self,
        cost_fn: Callable[[dict[Parameter, float]], float],
        gradient_fn: Optional[Callable[[dict[Parameter, float]], dict[Parameter, float]]] = None,
        initial_params: Optional[dict[Parameter, float]] = None,
    ) -> OptimizerResult:
        if initial_params is None:
            raise ValueError("initial_params is required")

        import scipy.optimize

        keys = list(initial_params.keys())
        n = len(keys)
        x0 = np.array([initial_params[k] for k in keys], dtype=float)

        result = OptimizerResult()

        def objective(x: np.ndarray) -> float:
            p = {k: float(x[j]) for j, k in enumerate(keys)}
            val = cost_fn(p)
            result.history.append(val)
            return val

        def jacobian(x: np.ndarray) -> np.ndarray:
            p = {k: float(x[j]) for j, k in enumerate(keys)}
            if gradient_fn is not None:
                grads = gradient_fn(p)
                return np.array([grads.get(k, 0.0) for k in keys])
            eps = max(1e-7, math.sqrt(np.finfo(float).eps))
            f0 = cost_fn(p)
            g = np.zeros(n, dtype=float)
            for idx in range(n):
                x_plus = x.copy()
                x_plus[idx] += eps
                p_plus = {k: float(x_plus[j]) for j, k in enumerate(keys)}
                g[idx] = (cost_fn(p_plus) - f0) / eps
            return g

        initial_val = cost_fn(initial_params)
        result.history = [initial_val]

        res = scipy.optimize.minimize(
            objective,
            x0,
            jac=jacobian if gradient_fn is not None else None,
            method="L-BFGS-B",
            options={"maxiter": self._max_iter, "ftol": self._tol * 1e-2, "maxcor": self._maxcor},
        )

        result.optimal_parameters = {k: float(res.x[j]) for j, k in enumerate(keys)}
        result.optimal_value = float(res.fun)
        result.converged = bool(res.success)
        result.iterations = int(res.nit)
        return result

    def _step(
        self,
        params: dict[Parameter, float],
        grads: dict[Parameter, float],
        iteration: int,
        cost_fn: Optional[Callable[[dict[Parameter, float]], float]] = None,
    ) -> dict[Parameter, float]:
        raise NotImplementedError("L_BFGS_B uses its own minimize() loop")

    def _converged(self, history: list[float]) -> bool:
        if len(history) < 2:
            return False
        return abs(history[-1] - history[-2]) < self._tol
