"""MicroQuantum hybrid execution runtime.

The runtime turns work into results through the canonical pipeline
``Program -> ExecutionPlan -> Target -> Backend -> Job -> Execution ->
Result``:

* :class:`ExecutionPlan` — declarative description of what to run.
* :class:`ExecutionRuntime` — coordinates preparation, compilation,
  submission and collection, enriching results with metadata and traces.
* :class:`ExecutionStrategy` — decides whether a plan runs directly or
  after compilation toward a target.
* :class:`ExecutionTrace` — lightweight per-execution lifecycle record.

Module-level helpers route through a shared default runtime so simple
scripts can call :func:`execute` / :func:`submit` directly.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from ..backends.base import Backend, BackendResult, Job
from ..core.circuit import QuantumCircuit
from ..core.device import Target
from ..experiments.experiment import ExperimentResult
from ..experiments.record import ExecutionRecord
from .plan import ExecutionPlan, ParameterBinding
from .runtime import BatchResult, ExecutionRuntime, SweepValues
from .strategy import (
    STRATEGY_HANDLERS,
    ExecutionStrategy,
    StrategyHandler,
    register_custom_strategy,
)
from .trace import ExecutionTrace, TraceEvent

Work = Union[ExecutionPlan, QuantumCircuit]

_DEFAULT_RUNTIME = ExecutionRuntime()


def execute(
    circuit: Optional[Union[QuantumCircuit, ExecutionPlan]] = None,
    *,
    plan: Optional[ExecutionPlan] = None,
    backend: Optional[Backend] = None,
    shots: int = 1024,
    seed: Optional[int] = None,
    target: Optional[Target] = None,
    parameter_bindings: Optional[ParameterBinding] = None,
    optimization_level: int = 0,
    options: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> BackendResult:
    """Execute a circuit (or an explicit plan) and return the result.

    Convenience wrapper around :meth:`ExecutionRuntime.execute` using a
    shared default runtime.
    """
    if isinstance(circuit, ExecutionPlan):
        if plan is not None:
            raise ValueError("provide either a circuit or a plan, not both")
        plan = circuit
        circuit = None
    if plan is not None:
        if circuit is not None:
            raise ValueError("provide either a circuit or a plan, not both")
        return _DEFAULT_RUNTIME.execute(plan, backend=backend)
    if circuit is None:
        raise ValueError("execute() requires a circuit or a plan")
    work_plan = ExecutionPlan.from_circuit(
        circuit,
        backend=backend,
        shots=shots,
        seed=seed,
        target=target,
        parameter_bindings=parameter_bindings,
        optimization_level=optimization_level,
        options=options,
        metadata=metadata,
    )
    return _DEFAULT_RUNTIME.execute(work_plan)


def submit(
    circuit: Optional[Union[QuantumCircuit, ExecutionPlan]] = None,
    *,
    plan: Optional[ExecutionPlan] = None,
    backend: Optional[Backend] = None,
    shots: int = 1024,
    seed: Optional[int] = None,
) -> Job:
    """Submit a circuit (or plan) and return its :class:`Job`."""
    if isinstance(circuit, ExecutionPlan):
        if plan is not None:
            raise ValueError("provide either a circuit or a plan, not both")
        plan = circuit
        circuit = None
    if plan is not None:
        if circuit is not None:
            raise ValueError("provide either a circuit or a plan, not both")
        return _DEFAULT_RUNTIME.submit(plan, backend=backend)
    if circuit is None:
        raise ValueError("submit() requires a circuit or a plan")
    work_plan = ExecutionPlan.from_circuit(
        circuit, backend=backend, shots=shots, seed=seed
    )
    return _DEFAULT_RUNTIME.submit(work_plan)


def execute_batch(
    work: Sequence[Work],
    *,
    shots: int = 1024,
    seed: Optional[int] = None,
    backend: Optional[Backend] = None,
    raise_on_error: bool = True,
) -> list[BatchResult]:
    """Execute a sequence of circuits/plans, collecting all results."""
    return _DEFAULT_RUNTIME.execute_batch(
        work,
        shots=shots,
        seed=seed,
        backend=backend,
        raise_on_error=raise_on_error,
    )


def submit_batch(
    work: Sequence[Work],
    *,
    shots: int = 1024,
    seed: Optional[int] = None,
    backend: Optional[Backend] = None,
) -> list[Job]:
    """Submit a sequence of circuits/plans, returning one Job each."""
    return _DEFAULT_RUNTIME.submit_batch(
        work, shots=shots, seed=seed, backend=backend
    )


def run_parameter_sweep(
    circuit: QuantumCircuit,
    parameter_values: SweepValues,
    **kwargs: Any,
) -> list[BackendResult]:
    """Run a circuit across a sweep of parameter bindings.

    See :meth:`ExecutionRuntime.run_parameter_sweep` for the full signature
    (``parameter_name``, ``shots``, ``seed``, ``target``,
    ``optimization_level``, ``backend``).
    """
    return _DEFAULT_RUNTIME.run_parameter_sweep(circuit, parameter_values, **kwargs)


def run_hybrid(
    iterations: int,
    build: Any,
    update: Any,
    initial: Any,
    *,
    shots: int = 1024,
    seed: Optional[int] = None,
    name: str = "hybrid",
) -> list[BackendResult]:
    """Run a generic classical-quantum hybrid workflow.

    See :meth:`ExecutionRuntime.run_hybrid` for the contract of the
    ``build`` and ``update`` callables.
    """
    return _DEFAULT_RUNTIME.run_hybrid(
        iterations, build, update, initial, shots=shots, seed=seed, name=name
    )


def execute_records(
    work: Sequence[Work],
    *,
    backend: Optional[Backend] = None,
    shots: int = 1024,
    seed: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> list[ExecutionRecord]:
    """Execute a sequence of plans/circuits, returning structured records.

    One :class:`ExecutionRecord` per input, in order; failures are recorded,
    never silently dropped.  See :meth:`ExecutionRuntime.execute_records`.
    """
    return _DEFAULT_RUNTIME.execute_records(
        work,
        backend=backend,
        shots=shots,
        seed=seed,
        metadata=metadata,
    )


def run_experiment(experiment: Any) -> ExperimentResult:
    """Run an :class:`Experiment` through the shared default runtime."""
    return _DEFAULT_RUNTIME.run_experiment(experiment)


default_runtime = _DEFAULT_RUNTIME

__all__ = [
    # Plan
    "ExecutionPlan",
    "ParameterBinding",
    # Runtime
    "ExecutionRuntime",
    "default_runtime",
    # Strategy
    "ExecutionStrategy",
    "StrategyHandler",
    "STRATEGY_HANDLERS",
    "register_custom_strategy",
    # Trace
    "ExecutionTrace",
    "TraceEvent",
    # Module-level convenience entry points
    "execute",
    "submit",
    "execute_batch",
    "submit_batch",
    "run_parameter_sweep",
    "run_hybrid",
    "execute_records",
    "run_experiment",
]