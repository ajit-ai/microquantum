"""Uniform iteration callbacks with early stopping for optimizers.

Wraps any :class:`Optimizer.minimize` call: the cost function is
instrumented to record values and invoke callbacks.  A callback returns
``True`` to request an early stop, which surfaces as an
:class:`OptimizerResult` with ``converged=False`` and the best point
seen so far — the underlying optimizer implementation is untouched.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

from ..core.parameter import Parameter
from .base import Optimizer, OptimizerResult

__all__ = [
    "CallbackProtocol",
    "minimize_with_callbacks",
]


class CallbackProtocol(Protocol):
    """Per-iteration observer; return ``True`` to request a stop."""

    def on_iteration(self, params: dict[Parameter, float], value: float) -> bool:
        """Observe an iteration; return True to stop optimization."""
        ...  # pragma: no cover - protocol stub


class _EarlyStop(Exception):
    """Internal signal carrying the best point seen so far."""

    def __init__(self, params: dict[Parameter, float], value: float) -> None:
        super().__init__("early stop requested by callback")
        self.params = params
        self.value = value


def minimize_with_callbacks(
    optimizer: Optimizer,
    cost_fn: Callable[[dict[Parameter, float]], float],
    *,
    gradient_fn: Optional[Callable[[dict[Parameter, float]], dict[Parameter, float]]] = None,
    initial_params: Optional[dict[Parameter, float]] = None,
    callbacks: Sequence[CallbackProtocol] = (),
    callback_every: int = 1,
) -> OptimizerResult:
    """Minimize *cost_fn*, invoking *callbacks* every iteration.

    Args:
        optimizer: Any optimizer (gradient-based or gradient-free).
        cost_fn: Objective to minimize.
        gradient_fn: Optional gradient (forwarded when the optimizer
            supports it).
        initial_params: Starting point (forwarded).
        callbacks: Observers invoked with ``(params, value)``.
        callback_every: Invoke callbacks every N cost evaluations.

    Returns:
        The optimizer result, or an early-stopped result when a
        callback requests it.
    """
    if callback_every < 1:
        raise ValueError("callback_every must be >= 1")
    history: list[float] = []
    evaluations = 0

    def wrapped(params: dict[Parameter, float]) -> float:
        nonlocal evaluations
        value = float(cost_fn(params))
        history.append(value)
        evaluations += 1
        if callbacks and evaluations % callback_every == 0:
            snapshot = dict(params)
            if any(cb.on_iteration(snapshot, value) for cb in callbacks):
                raise _EarlyStop(snapshot, value)
        return value

    try:
        return optimizer.minimize(
            wrapped, gradient_fn=gradient_fn, initial_params=initial_params
        )
    except _EarlyStop as stop:
        return OptimizerResult(
            optimal_parameters=dict(stop.params),
            optimal_value=stop.value,
            history=list(history),
            iterations=len(history),
            converged=False,
        )
