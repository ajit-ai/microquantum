"""Execution strategies: how the runtime dispatches a plan.

An :class:`ExecutionStrategy` describes *how* a plan is executed.  The
runtime classifies plans (:meth:`ExecutionStrategy.classify`) and dispatches
to a pluggable handler table.  Handlers are plain callables
``(runtime, plan) -> BackendResult`` so users can override behaviour for a
strategy (for example, replacing the built-in GPU-less direct execution
with an accelerated path) without forking the runtime.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from .plan import ExecutionPlan


class ExecutionStrategy(Enum):
    """Execution mode selected for a plan.

    Members:
        DIRECT: Run the plan as-is on the backend (no compilation).
        COMPILED: Run after optimizing/decomposing toward a target.
        BATCH: Execute several plans and collect all results.
        PARAMETER_SWEEP: Run the same circuit over many parameter bindings.
        HYBRID: Interleave classical work with quantum sub-executions.
    """

    DIRECT = "direct"
    COMPILED = "compiled"
    BATCH = "batch"
    PARAMETER_SWEEP = "parameter_sweep"
    HYBRID = "hybrid"

    @classmethod
    def classify(cls, plan: ExecutionPlan) -> "ExecutionStrategy":
        """Pick the strategy implied by a plan's compilation intent."""
        if plan.compiled is not None or plan.optimization_level > 0 or plan.target is not None:
            return cls.COMPILED
        return cls.DIRECT


#: Handler type: ``(runtime, plan) -> BackendResult``.
StrategyHandler = Callable[[Any, ExecutionPlan], Any]

#: Registry of strategy handlers keyed by :class:`ExecutionStrategy`.
#: The runtime reads this table for every dispatch.  ``BATCH``,
#: ``PARAMETER_SWEEP`` and ``HYBRID`` entries are set by the runtime for
#: convenience; overriding them replaces the corresponding high-level method.
STRATEGY_HANDLERS: dict[ExecutionStrategy, StrategyHandler] = {}


def register_custom_strategy(strategy: str, handler: StrategyHandler) -> None:
    """Register a handler for a strategy name.

    The strategy name must be one of the :class:`ExecutionStrategy` values.
    Registered handlers take precedence over the built-in behaviour, giving a
    concrete extension point without adding placeholder strategy classes.

    Args:
        strategy: Strategy name, e.g. ``"direct"`` or ``"compiled"``.
        handler: Callable ``(runtime, plan) -> BackendResult``.
    """
    try:
        key = ExecutionStrategy(strategy)
    except ValueError as exc:
        raise ValueError(
            f"unknown strategy {strategy!r}; "
            f"expected one of {[e.value for e in ExecutionStrategy]}"
        ) from exc
    STRATEGY_HANDLERS[key] = handler


__all__ = [
    "ExecutionStrategy",
    "StrategyHandler",
    "STRATEGY_HANDLERS",
    "register_custom_strategy",
]