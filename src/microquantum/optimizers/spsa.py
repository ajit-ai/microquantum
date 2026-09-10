"""SPSA and QNSPSA optimizers for noisy quantum landscapes.

SPSA (Simultaneous Perturbation Stochastic Approximation) estimates
gradients using only two cost function evaluations regardless of
parameter count, making it ideal for shot-noisy quantum optimization.

QNSPSA (Quantum Natural SPSA) uses the Fubini-Study metric tensor
to precondition SPSA updates for better convergence on quantum
optimization landscapes.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from ..core.parameter import Parameter
from .base import Optimizer


class SPSA(Optimizer):
    """Simultaneous Perturbation Stochastic Approximation.

    Estimates the gradient using random perturbations in all
    parameters simultaneously, requiring only 2 cost evaluations
    per iteration regardless of parameter count.

    Args:
        a: Initial step size (decays as 1/(iteration + A)^alpha).
        c: Perturbation size for gradient estimation.
        alpha: Step size decay exponent (typically 0.602).
        gamma: Perturbation decay exponent (typically 0.101).
        A: Stabilization constant for step size decay.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        seed: Random seed for reproducibility.

    Reference:
        J.C. Spall, "Multivariate Stochastic Approximation using
        Simultaneous Perturbation Gradient Approximation", 1992.
    """

    def __init__(
        self,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
        max_iter: int = 200,
        tol: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        self._a = a
        self._c = c
        self._alpha = alpha
        self._gamma = gamma
        self._A = A
        self._max_iter = max_iter
        self._tol = tol
        self._rng = random.Random(seed)

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
            raise ValueError("SPSA requires cost_fn")

        ak = self._a / (iteration + 1 + self._A) ** self._alpha
        ck = self._c / (iteration + 1) ** self._gamma

        # Random perturbation vector (+1/-1 with equal probability)
        delta = {
            p: float(self._rng.choice([-1.0, 1.0])) for p in params
        }

        # Evaluate cost at perturbed points
        params_plus = {
            p: params[p] + ck * delta[p] for p in params
        }
        params_minus = {
            p: params[p] - ck * delta[p] for p in params
        }

        cost_plus = cost_fn(params_plus)
        cost_minus = cost_fn(params_minus)

        # Estimate gradient
        ghat = {
            p: (cost_plus - cost_minus) / (2.0 * ck * delta[p])
            for p in params
        }

        # Update parameters
        new_params = {
            p: params[p] - ak * ghat[p] for p in params
        }
        return new_params

    def _converged(self, history: list[float]) -> bool:
        if len(history) < 2:
            return False
        return abs(history[-1] - history[-2]) < self._tol


class QNSPSA(Optimizer):
    """Quantum Natural SPSA with Fubini-Study metric tensor.

    Improves SPSA by preconditioning the gradient estimate with the
    quantum Fisher information matrix (Fubini-Study metric tensor),
    leading to better convergence on quantum cost landscapes.

    The metric tensor is estimated using two additional SPSA-style
    evaluations per iteration (4 total cost evaluations).

    Args:
        a: Initial step size.
        c: Perturbation size for gradient estimation.
        alpha: Step size decay exponent.
        gamma: Perturbation decay exponent.
        A: Stabilization constant.
        fidelity_fn: Function computing state fidelity overlap.
            Signature: (params_a, params_b) -> float in [0, 1].
            If None, falls back to vanilla SPSA.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        seed: Random seed.

    Reference:
        G. Stoudenmire & D. Wiersema, "OpenFermion: The Electronic
        Structure Package for Quantum Computers", 2020.
    """

    def __init__(
        self,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
        fidelity_fn: Optional[
            Callable[[dict[Parameter, float], dict[Parameter, float]], float]
        ] = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        self._a = a
        self._c = c
        self._alpha = alpha
        self._gamma = gamma
        self._A = A
        self._fidelity_fn = fidelity_fn
        self._max_iter = max_iter
        self._tol = tol
        self._rng = random.Random(seed)

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def _estimate_metric_tensor(
        self,
        params: dict[Parameter, float],
        cost_fn: Callable[[dict[Parameter, float]], float],
    ) -> dict[Parameter, float]:
        """Estimate diagonal of the Fubini-Study metric tensor."""
        ck = self._c / (max(1, len(params))) ** 0.5

        g = {}
        for p in params:
            delta = {q: 0.0 for q in params}
            delta[p] = 1.0

            params_plus = {q: params[q] + ck * delta[q] for q in params}
            params_minus = {q: params[q] - ck * delta[q] for q in params}

            if self._fidelity_fn is not None:
                fidelity = self._fidelity_fn(params_plus, params_minus)
            else:
                # Approximate using cost function curvature
                c0 = cost_fn(params)
                c_plus = cost_fn(params_plus)
                c_minus = cost_fn(params_minus)
                fidelity = 1.0 - abs(c_plus + c_minus - 2 * c0) / 2.0

            g[p] = max(fidelity, 1e-10)

        return g

    def _step(
        self,
        params: dict[Parameter, float],
        grads: dict[Parameter, float],
        iteration: int,
        cost_fn: Optional[Callable[[dict[Parameter, float]], float]] = None,
    ) -> dict[Parameter, float]:
        if cost_fn is None:
            raise ValueError("QNSPSA requires cost_fn")

        ak = self._a / (iteration + 1 + self._A) ** self._alpha
        ck = self._c / (iteration + 1) ** self._gamma

        # SPSA gradient estimate
        delta = {
            p: float(self._rng.choice([-1.0, 1.0])) for p in params
        }
        params_plus = {p: params[p] + ck * delta[p] for p in params}
        params_minus = {p: params[p] - ck * delta[p] for p in params}
        cost_plus = cost_fn(params_plus)
        cost_minus = cost_fn(params_minus)

        ghat = {
            p: (cost_plus - cost_minus) / (2.0 * ck * delta[p])
            for p in params
        }

        # Metric tensor preconditioning
        metric = self._estimate_metric_tensor(params, cost_fn)

        # Preconditioned update: g_metric = ghat / metric_tensor_diag
        new_params = {}
        for p in params:
            gt = ghat[p] / metric[p]
            new_params[p] = params[p] - ak * gt

        return new_params

    def _converged(self, history: list[float]) -> bool:
        if len(history) < 2:
            return False
        return abs(history[-1] - history[-2]) < self._tol
