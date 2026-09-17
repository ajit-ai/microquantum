"""Stage-tagged runtime errors.

The runtime pipeline has well-defined stages::

    planning -> compilation -> runtime dispatch -> backend execution

Each stage has a dedicated exception type below.  Stageless custom exception
``ExecutionError`` is the base, and *every* stage error subclasses it.  All of
them also subclass :class:`ValueError` so existing ``except ValueError``
handlers keep working unchanged — this is a compatibility-preserving
addition, not a replacement of the existing exception behaviour.

The runtime wraps failures at its stage boundaries only; type-validation
failures (e.g. passing a non-plan where a plan is required) still raise the
corresponding built-in :class:`TypeError`.
"""

from __future__ import annotations

__all__ = [
    "ExecutionError",
    "PlanningError",
    "CompilationError",
    "RuntimeDispatchError",
    "BackendExecutionError",
]


class ExecutionError(ValueError):
    """Base class for runtime-pipeline execution failures."""


class PlanningError(ExecutionError):
    """The description of what to run (plan) is invalid or un-runnable."""


class CompilationError(ExecutionError):
    """Work could not be compiled or is incompatible with its target."""


class RuntimeDispatchError(ExecutionError):
    """The runtime could not dispatch/submit the work to a backend."""


class BackendExecutionError(ExecutionError):
    """Backend execution started but the job failed, cancelled or was empty."""