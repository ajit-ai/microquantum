"""Execution experiments & reproducibility (MQ-07).

This package provides the backend-independent *experiment layer*:

* :class:`ExecutionRecord` — one actual execution plus portable metadata
  (plan context, backend/target, shots/seed, bindings, timing, raw result).
* :class:`ExecutionFailure` — structured, inspectable failures.
* :class:`ParameterSweep` — deterministic parameter grids with explicit
  values or generated numeric ranges (Cartesian product, stable order).
* :class:`Experiment` / :class:`ExperimentResult` — collections of related
  executions, run through an existing
  :class:`~microquantum.runtime.ExecutionRuntime`, keeping raw records intact.
* ``execution_fingerprint`` / ``reproducibility_metadata`` — stable,
  configuration-based reproducibility metadata (never memory addresses; no
  bit-for-bit claims for hardware/nondeterministic backends).

Everything is fully in-memory and JSON-safe serializable.
"""

from .experiment import Experiment, ExperimentResult
from .record import (
    ExecutionFailure,
    ExecutionRecord,
    ExecutionStatus,
    execution_fingerprint,
    reproducibility_metadata,
)
from .sweep import ParameterSweep

__all__ = [
    "ExecutionRecord",
    "ExecutionFailure",
    "ExecutionStatus",
    "ParameterSweep",
    "Experiment",
    "ExperimentResult",
    "execution_fingerprint",
    "reproducibility_metadata",
]