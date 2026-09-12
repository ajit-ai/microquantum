"""Execution runtime: coordinate plans, backends, jobs and results.

The :class:`ExecutionRuntime` is the coordinator of the canonical pipeline
``Program -> ExecutionPlan -> Target -> Backend -> Job -> Execution ->
Result``.  It is deliberately *not* a backend: it prepares work, binds
parameters, optionally compiles against a target (reusing the MQ-03
:class:`~microquantum.ir.Compiler`), submits to the selected backend and
collects the resulting :class:`~microquantum.backends.base.Job` /
:class:`~microquantum.backends.base.BackendResult`, enriching them with
execution metadata and a lightweight trace.

Backends keep full responsibility for execution simulation/provider
interaction; the runtime never re-implements backend logic.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union, cast

import numpy as np

from ..backends.base import Backend, BackendResult, Job, JobStatus
from ..backends.registry import BackendRegistry, default_registry
from ..backends.statevector import StatevectorBackend
from ..core.circuit import QuantumCircuit
from ..core.device import Target
from ..core.parameter import Parameter
from ..ir import CompilationResult, Compiler, from_ir
from .plan import ExecutionPlan, ParameterBinding
from .strategy import STRATEGY_HANDLERS, ExecutionStrategy
from .trace import ExecutionTrace

if TYPE_CHECKING:
    from ..experiments.experiment import ExperimentResult
    from ..experiments.record import ExecutionRecord

Work = Union[ExecutionPlan, QuantumCircuit]
BatchResult = Union[BackendResult, str]
SweepValues = Union[Sequence[float], Sequence[ParameterBinding]]
#: Backend reference accepted by the runtime: an instance or a name.
BackendRef = Union[Backend, str]


def _direct_handler(
    runtime: "ExecutionRuntime", plan: ExecutionPlan
) -> BackendResult:
    return runtime._run_plan(plan)


def _compiled_handler(
    runtime: "ExecutionRuntime", plan: ExecutionPlan
) -> BackendResult:
    return runtime._run_plan(plan)


class ExecutionRuntime:
    """Orchestrates plan preparation, compilation, submission and collection.

    Args:
        backend: Default :class:`Backend` when a plan does not name one
            (highest-priority explicit default).
        registry: Optional :class:`BackendRegistry` used to resolve backend
            *names* given in plans and to pick the fallback default backend.
            Without a registry, a plan may still name one of the backends
            registered in the module-level ``default_registry``.
        default_target: Default :class:`Target` applied to plans that do not
            specify a target.  ``None`` disables default target processing.
        history_size: Maximum number of records kept in :attr:`history`.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        *,
        registry: Optional[BackendRegistry] = None,
        default_target: Optional[Target] = None,
        history_size: int = 200,
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._default_target = default_target
        self._history_size = max(0, int(history_size))
        self._history: list[dict[str, Any]] = []
        self._simulator: Optional[StatevectorBackend] = None
        self.handlers: dict[ExecutionStrategy, Any] = dict(STRATEGY_HANDLERS)
        self.handlers.setdefault(ExecutionStrategy.DIRECT, _direct_handler)
        self.handlers.setdefault(ExecutionStrategy.COMPILED, _compiled_handler)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict[str, Any]]:
        """Read-only execution history (newest execution last)."""
        return [dict(entry) for entry in self._history]

    @property
    def registry(self) -> Optional[BackendRegistry]:
        """Registry used to resolve backend names (may be ``None``)."""
        return self._registry

    @property
    def default_backend(self) -> Backend:
        """Resolve the backend used for plans that name none.

        Precedence: explicit ``backend`` argument > registry default >
        a lazily created :class:`StatevectorBackend` (name ``"statevector"``).
        """
        if self._backend is not None:
            return self._backend
        if self._registry is not None:
            backend = self._registry.default
            if backend is not None:
                return backend
        if self._simulator is None:
            self._simulator = StatevectorBackend()
        return self._simulator

    def resolve_backend(self, ref: Optional[BackendRef]) -> Backend:
        """Resolve a backend reference (instance, name or ``None``).

        Passed a *name*, the runtime first checks the attached registry and
        then the module-level ``default_registry``.

        Raises:
            KeyError: If *ref* is a string naming no registered backend.
            TypeError: If *ref* is neither a Backend nor a string.
        """
        if ref is None:
            return self.default_backend
        if isinstance(ref, Backend):
            return ref
        if isinstance(ref, str):
            if self._registry is not None and self._registry.has(ref):
                return self._registry.get(ref)
            if default_registry.has(ref):
                return default_registry.get(ref)
            known = (
                self._registry.names()
                if self._registry is not None
                else default_registry.names()
            )
            raise KeyError(
                f"unknown backend {ref!r}; "
                f"known backends: {sorted(known)}"
            )
        raise TypeError(
            "backend reference must be a Backend, a name string or None, "
            f"got {type(ref).__name__}"
        )

    def clear_history(self) -> None:
        """Clear the recorded execution history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # plan lifecycle
    # ------------------------------------------------------------------

    def prepare(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Validate a plan and return it ready for execution.

        Raises:
            TypeError: If ``plan`` is not an :class:`ExecutionPlan`.
            ValueError: If the plan has validation problems (unbound
                parameters, unknown bindings, invalid shots, ...).
        """
        if not isinstance(plan, ExecutionPlan):
            raise TypeError(
                f"runtime execution requires an ExecutionPlan, "
                f"got {type(plan).__name__}"
            )
        problems = plan.validate()
        if problems:
            raise ValueError(
                f"ExecutionPlan '{plan.name}' has problems: {'; '.join(problems)}"
            )
        return plan

    def _select_backend(self, plan: ExecutionPlan) -> Backend:
        return self.resolve_backend(plan.backend)

    def _select_target(self, plan: ExecutionPlan) -> Optional[Target]:
        target = plan.target
        if target is None and self._default_target is not None:
            return self._default_target
        return target

    def compile(
        self,
        work: Union[ExecutionPlan, QuantumCircuit],
        *,
        optimization_level: Optional[int] = None,
        target: Optional[Target] = None,
    ) -> CompilationResult:
        """Compile work toward a plan (or the provided target).

        Runs the MQ-03 compilation pipeline over a bound copy of the work.
        Plans that already carry a ``compiled`` result are returned as-is.

        Returns:
            A :class:`CompilationResult` holding source IR, compiled IR,
            passes applied and target diagnostics.

        Raises:
            ValueError: If the work cannot be compiled, or is incompatible
                with the target.
        """
        plan = self._as_plan(work)
        if plan.compiled is not None:
            return cast(CompilationResult, plan.compiled)
        bound = self._build_bound(plan)
        level = (
            plan.optimization_level
            if optimization_level is None
            else optimization_level
        )
        compiled = Compiler(optimization_level=level).compile(
            bound, target=target if target is not None else self._select_target(plan)
        )
        explicit = target is not None or plan.target is not None
        if explicit and not compiled.is_compatible:
            raise ValueError(
                f"compilation produced target problems: "
                f"{'; '.join(compiled.diagnostics)}"
            )
        return compiled

    # ------------------------------------------------------------------
    # submission / execution
    # ------------------------------------------------------------------

    def submit(self, work: Work, *, backend: Optional[BackendRef] = None) -> Job:
        """Submit a plan or circuit to a backend and return its Job.

        Simulator backends return an already-completed job synchronously;
        asynchronous providers may return a job that completes later.  The
        job carries lifecycle metadata (timestamps, backend/target, info).

        Args:
            work: An :class:`ExecutionPlan` or :class:`QuantumCircuit`.
            backend: Optional backend override for circuit inputs.

        Returns:
            A :class:`Job` for the submission.
        """
        plan = self._as_plan(work, backend=backend)
        job, _trace, _elapsed = self._submit_plan(plan)
        return job

    def submit_batch(
        self,
        work: Sequence[Work],
        *,
        shots: int = 1024,
        seed: Optional[int] = None,
        backend: Optional[BackendRef] = None,
    ) -> list[Job]:
        """Submit a sequence of plans/circuits, returning one Job each.

        Failed submissions appear as failed jobs in the returned list rather
        than aborting the batch.
        """
        jobs: list[Job] = []
        for index, item in enumerate(work):
            try:
                plan = self._as_plan(item, backend=backend, shots=shots, seed=seed)
                plan = replace(plan, metadata={**plan.metadata, "batch_index": index})
                jobs.append(self.submit(plan))
            except Exception as exc:
                jobs.append(
                    Job(
                        status=JobStatus.FAILED,
                        error=f"batch item {index} failed: {exc}",
                        info={"batch_index": index},
                    )
                )
        return jobs

    def execute(
        self,
        work: Work,
        *,
        backend: Optional[BackendRef] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> BackendResult:
        """Execute work and return its result.

        This is the primary one-shot entry point.  It prepares the plan,
        dispatches it through the registered :class:`ExecutionStrategy`
        handler, submits to the backend and unwraps the completed job.

        Args:
            work: An :class:`ExecutionPlan` or :class:`QuantumCircuit`.
            backend: Backend override for circuit inputs (default: resolve
                from the plan or the runtime default).
            shots: Shots override for circuit inputs.
            seed: Seed override for circuit inputs.
            metadata: Metadata merged into every result for circuit inputs.

        Returns:
            A :class:`BackendResult` enriched with execution metadata.

        Raises:
            ValueError: If the plan fails validation, the backend job fails,
                or the work is incompatible with the selected target.
        """
        plan = self._as_plan(
            work,
            backend=backend,
            shots=shots,
            seed=seed,
            metadata=metadata,
        )
        return self._dispatch(plan)

    def _dispatch(self, plan: ExecutionPlan) -> BackendResult:
        strategy = ExecutionStrategy.classify(plan)
        handler = self.handlers.get(strategy)
        if handler is None:
            raise ValueError(
                f"no execution handler registered for strategy {strategy.value!r}"
            )
        return cast(BackendResult, handler(self, plan))

    def _run_plan(self, plan: ExecutionPlan) -> BackendResult:
        job, trace, elapsed = self._submit_plan(plan)
        return self._collect(job, plan, self._select_backend(plan), trace, elapsed)

    def execute_batch(
        self,
        work: Sequence[Work],
        *,
        shots: int = 1024,
        seed: Optional[int] = None,
        backend: Optional[BackendRef] = None,
        raise_on_error: bool = True,
    ) -> list[BatchResult]:
        """Execute a sequence of plans/circuits and collect their results.

        By default the first failure aborts execution (:data:`True`).  With
        ``raise_on_error=False`` failed items are returned as ``str`` error
        messages in their original position so a single bad circuit does not
        discard the rest of the batch.

        Returns:
            List with one :class:`BackendResult` (or error string) per input.
        """
        batch_id = uuid.uuid4().hex[:8]
        results: list[BatchResult] = []
        for index, item in enumerate(work):
            try:
                plan = self._as_plan(item, backend=backend, shots=shots, seed=seed)
                plan = replace(
                    plan,
                    metadata={**plan.metadata, "batch_id": batch_id, "batch_index": index},
                )
                results.append(self.execute(plan))
            except Exception as exc:
                if raise_on_error:
                    raise
                results.append(f"item {index}: {exc}")
        return results

    def execute_record(
        self,
        work: Work,
        *,
        backend: Optional[BackendRef] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ExecutionRecord":
        """Execute work and return a structured :class:`ExecutionRecord`.

        Mirrors :meth:`execute` but wraps the *entire* pipeline (including
        preparation and validation) so that the outcome is always represented
        by a record — completed with its raw result, or failed with a
        structured :class:`ExecutionFailure`.  Failures are never silently
        discarded.

        Args:
            work: An :class:`ExecutionPlan` or :class:`QuantumCircuit`.
            backend: Backend override for circuit inputs.
            shots: Shots override for circuit inputs.
            seed: Seed override for circuit inputs.
            metadata: Metadata merged into every result for circuit inputs.

        Returns:
            An :class:`ExecutionRecord` carrying the result or the failure.
        """
        from ..experiments.record import ExecutionRecord

        plan = self._as_plan(
            work,
            backend=backend,
            shots=shots,
            seed=seed,
            metadata=metadata,
        )
        started = time.monotonic()
        try:
            job, trace, elapsed = self._submit_plan(plan)
            backend_obj = self._select_backend(plan)
            result = self._collect(job, plan, backend_obj, trace, elapsed)
        except Exception as exc:
            return ExecutionRecord.failed(
                plan,
                exc,
                backend_name=self._plan_backend_name(plan),
                elapsed_seconds=time.monotonic() - started,
            )
        return ExecutionRecord.completed(plan, backend_obj, result, elapsed)

    def execute_records(
        self,
        work: Sequence[Work],
        *,
        backend: Optional[BackendRef] = None,
        shots: int = 1024,
        seed: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        parameter_bindings: Optional[dict[str, Any]] = None,
    ) -> list["ExecutionRecord"]:
        """Execute a sequence of work items, collecting structured records.

        Unlike :meth:`execute_batch` (raw results or error strings), this
        returns one :class:`ExecutionRecord` per input, in order.  The batch
        preserves parameter bindings and backend selection, and *never* drops
        failures: each failed item yields a record in :class:`failed` state
        with an inspectable :class:`ExecutionFailure`.

        When ``parameter_bindings`` is given, every input is executed once per
        binding combination: each value must be a sequence (or a single scalar),
        and the Cartesian product is submitted in deterministic order.

        Args:
            work: Sequence of plans/circuits to execute.
            backend: Backend override for circuit inputs.
            shots: Shots for circuit inputs.
            seed: Seed for circuit inputs.
            metadata: Metadata merged into circuit-input results.
            parameter_bindings: Optional ``{parameter: [values...]}`` mapping
                expanded into one record per binding combination.

        Returns:
            One :class:`ExecutionRecord` per input (per combination), in order.
        """
        from itertools import product
        from ..experiments.record import ExecutionRecord

        if parameter_bindings:
            combos: list[dict[str, Any]] = [
                dict(zip(parameter_bindings.keys(), values, strict=True))
                for values in product(
                    *(
                        v if isinstance(v, (list, tuple)) else [v]
                        for v in parameter_bindings.values()
                    )
                )
            ]
            combos = [{str(k): v for k, v in combo.items()} for combo in combos]
        else:
            combos = [{}]

        batch_id = uuid.uuid4().hex[:8]
        records: list[ExecutionRecord] = []
        for combo in combos:
            for index, item in enumerate(work):
                context: Optional[ExecutionPlan] = None
                try:
                    plan = self._as_plan(
                        item,
                        backend=backend,
                        shots=shots,
                        seed=seed,
                        metadata=metadata,
                    )
                    plan = replace(
                        plan,
                        parameter_bindings=combo,
                        metadata={
                            **plan.metadata,
                            "batch_id": batch_id,
                            "batch_index": index,
                        },
                    )
                    context = plan
                    records.append(self.execute_record(plan))
                except Exception as exc:
                    records.append(
                        ExecutionRecord.failed(
                            context if context is not None else self._context_plan(item),
                            exc,
                            backend_name=(
                                backend if isinstance(backend, str) else None
                            ),
                            elapsed_seconds=0.0,
                        )
                    )
        return records

    def run_experiment(self, experiment: Any) -> "ExperimentResult":
        """Run an :class:`Experiment` through this runtime.

        Convenience delegating to :meth:`Experiment.run`; returns an
        :class:`ExperimentResult` with its raw execution records preserved.
        """
        from ..experiments.experiment import Experiment

        if not isinstance(experiment, Experiment):
            raise TypeError(
                f"run_experiment expects an Experiment, "
                f"got {type(experiment).__name__}"
            )
        return experiment.run(self)

    def run_parameter_sweep(
        self,
        circuit: QuantumCircuit,
        parameter_values: SweepValues,
        *,
        parameter_name: Optional[str] = None,
        shots: int = 1024,
        seed: Optional[int] = None,
        target: Optional[Target] = None,
        optimization_level: int = 0,
        backend: Optional[BackendRef] = None,
    ) -> list[BackendResult]:
        """Run the same circuit across a sweep of parameter bindings.

        Args:
            circuit: The (parameterized) circuit to sweep.
            parameter_values: Either a sequence of float values (shortcut
                usable when the circuit has exactly one free parameter) or a
                sequence of binding mappings.
            parameter_name: When sweeping raw floats, the name of the
                parameter each value binds to (required for multi-parameter
                circuits).
            shots: Shots per execution.
            seed: Seed for reproducible executions.
            target: Optional target compiled against.
            optimization_level: Compiler optimization level.
            backend: Optional backend override.

        Returns:
            One :class:`BackendResult` per binding, in order.
        """
        params = tuple(circuit.parameters)
        bindings_all: list[ParameterBinding] = []
        for value in parameter_values:
            if isinstance(value, dict):
                bindings_all.append(dict(value))
            else:
                if not isinstance(value, (int, float, np.generic)):
                    raise ValueError(
                        f"sweep values must be numbers or bindings, "
                        f"got {type(value).__name__}"
                    )
                name = (
                    parameter_name
                    if parameter_name is not None
                    else self._single_param(params)
                )
                bindings_all.append({name: float(value)})

        sweep_id = uuid.uuid4().hex[:8]
        results: list[BackendResult] = []
        for index, bindings in enumerate(bindings_all):
            plan = ExecutionPlan.from_circuit(
                circuit,
                name=f"sweep-{index}",
                target=target,
                backend=backend,
                shots=shots,
                seed=seed,
                parameter_bindings=bindings,
                optimization_level=optimization_level,
                metadata={
                    "sweep_id": sweep_id,
                    "sweep_index": index,
                    "parameter_bindings": dict(bindings),
                },
            )
            results.append(self.execute(plan))
        return results

    def run_hybrid(
        self,
        iterations: int,
        build: Any,
        update: Any,
        initial: Any,
        *,
        shots: int = 1024,
        seed: Optional[int] = None,
        name: str = "hybrid",
    ) -> list[BackendResult]:
        """Run a classical-quantum hybrid workflow.

        The workflow is described by two plain callables:

        * ``build(state, step_index) -> QuantumCircuit | ExecutionPlan``
          maps the current classical state to the next quantum work item.
        * ``update(state, step_index, backend_result) -> state`` folds the
          measured outcome back into the classical state for the next step.

        The loop runs exactly ``iterations`` steps and returns the quantum
        measurements in order (each enriched with ``hybrid_step`` metadata).
        This is intentionally generic — it implements no specific algorithm
        such as VQE or QAOA.

        Args:
            iterations: Number of classical-quantum rounds.
            build: ``(state, index) -> work`` callable.
            update: ``(state, index, result) -> state`` callable.
            initial: Initial classical state.
            shots: Shots per quantum execution.
            seed: Seed passed to each execution.
            name: Plan name prefix used in metadata/traces.

        Returns:
            One :class:`BackendResult` per step, in order.
        """
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        hybrid_id = uuid.uuid4().hex[:8]
        state = initial
        results: list[BackendResult] = []
        for index in range(iterations):
            work = build(state, index)
            plan = self._as_plan(work, shots=shots, seed=seed)
            plan = replace(
                plan,
                name=f"{name}-step{index}",
                metadata={
                    **plan.metadata,
                    "hybrid_id": hybrid_id,
                    "hybrid_step": index,
                    "hybrid_rounds": iterations,
                },
            )
            result = self.execute(plan)
            results.append(result)
            state = update(state, index, result)
        return results

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _context_plan(self, item: Any) -> Optional[ExecutionPlan]:
        """Best-effort plan snapshot for a failed planning attempt.

        Returns a minimal plan (or ``None``) so failed records still carry
        user-facing context (name, shots, bindings) when plan construction
        itself raised.
        """
        if isinstance(item, ExecutionPlan):
            return item
        if isinstance(item, QuantumCircuit):
            try:
                return ExecutionPlan.from_circuit(item, name="item")
            except Exception:
                return None
        return None

    def _plan_backend_name(self, plan: Any) -> str:
        """Best-effort backend name for a plan (resilient to failures)."""
        try:
            return self._select_backend(cast(ExecutionPlan, plan)).name
        except Exception:
            backend = getattr(plan, "backend", None)
            if isinstance(backend, str):
                return backend
            return getattr(backend, "name", "<unknown>")

    def _single_param(self, params: tuple[Parameter, ...]) -> str:
        params = tuple(params)
        if len(params) != 1:
            raise ValueError(
                f"raw sweep values require a single free parameter, "
                f"found {sorted(p.name for p in params)}"
            )
        return params[0].name

    def _as_plan(
        self,
        work: Work,
        *,
        backend: Optional[BackendRef] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExecutionPlan:
        if isinstance(work, ExecutionPlan):
            plan = work
            if backend is not None and plan.backend is None:
                plan = replace(plan, backend=backend)
            if metadata:
                plan = replace(plan, metadata={**plan.metadata, **metadata})
            return plan
        if isinstance(work, QuantumCircuit):
            from ..core.dynamic import DynamicCircuit

            if isinstance(work, DynamicCircuit):
                raise ValueError(
                    "dynamic circuits must be converted first; "
                    "use to_ir_dynamic() + a compiled plan"
                )
            return ExecutionPlan.from_circuit(
                work,
                backend=backend,
                shots=shots if shots is not None else 1024,
                seed=seed if seed is not None else None,
                metadata=metadata or {},
            )
        raise TypeError(
            f"cannot execute {type(work).__name__}; "
            f"expected ExecutionPlan or QuantumCircuit"
        )

    def _build_bound(self, plan: ExecutionPlan) -> QuantumCircuit:
        base: Any
        if plan.circuit is not None:
            base = plan.circuit
        elif plan.ir is not None:
            base = from_ir(plan.ir)
        else:
            raise ValueError(
                f"plan '{plan.name}' carries no circuit/ir work to compile"
            )
        if plan.parameter_bindings:
            base = base.bind_parameters(dict(plan.parameter_bindings))
        if base.is_parameterized:
            remaining = sorted(p.name for p in base.parameters)
            raise ValueError(
                f"plan '{plan.name}' remains parameterized; "
                f"unbound parameters: {remaining}"
            )
        return cast(QuantumCircuit, base)

    def _submit_plan(
        self, plan: ExecutionPlan
    ) -> tuple[Job, ExecutionTrace, float]:
        plan = self.prepare(plan)
        backend = self._select_backend(plan)
        trace = ExecutionTrace(name=plan.name)
        started = time.monotonic()
        strategy = ExecutionStrategy.classify(plan)
        trace.record("prepared", strategy=strategy.value)

        if strategy is ExecutionStrategy.COMPILED:
            if plan.compiled is not None:
                compiled = plan.compiled
                self._check_compiled_incompatible(compiled, plan)
                executable = compiled.circuit()
            else:
                compiled = Compiler(
                    optimization_level=plan.optimization_level
                ).compile(
                    self._build_bound(plan),
                    target=self._select_target(plan),
                )
                explicit_target = plan.target is not None or self._default_target is not None
                if explicit_target and not compiled.is_compatible:
                    raise ValueError(
                        f"plan '{plan.name}' is incompatible with its target: "
                        f"{'; '.join(compiled.diagnostics)}"
                    )
                executable = compiled.circuit()
            if executable.is_parameterized:
                executable = self._bind_executable(executable, plan, trace)
            trace.record(
                "compiled",
                passes=list(compiled.passes_applied),
                compiled_gates=compiled.result.num_gates,
            )
        else:
            executable = self._build_bound(plan)
            trace.record("bound")
            compiled = None

        self._check_capabilities(backend, executable, plan)
        trace.record("validated")

        job = backend.submit_circuit(
            executable,
            shots=plan.shots,
            initial_state=plan.initial_state,
            seed=plan.seed,
        )
        self._annotate_job(job, plan, backend, strategy)
        trace.record(
            "submitted",
            job_id=job.job_id,
            status=job.status.value,
            strategy=strategy.value,
        )
        elapsed = time.monotonic() - started
        return job, trace, elapsed

    def _bind_executable(
        self, executable: QuantumCircuit, plan: ExecutionPlan, trace: ExecutionTrace
    ) -> QuantumCircuit:
        if plan.parameter_bindings:
            executable = executable.bind_parameters(dict(plan.parameter_bindings))
        if executable.is_parameterized:
            raise ValueError(
                f"plan '{plan.name}' remains parameterized; "
                f"unbound parameters: {sorted(p.name for p in executable.parameters)}"
            )
        trace.record("bound")
        return executable

    def _annotate_job(
        self,
        job: Job,
        plan: ExecutionPlan,
        backend: Backend,
        strategy: ExecutionStrategy,
    ) -> None:
        job.backend_name = backend.name
        job.target_name = plan.target.name if plan.target is not None else backend.target.name
        job.info["plan"] = plan.name
        job.info["strategy"] = strategy.value
        job.info["optimization_level"] = plan.optimization_level
        job.info["shots"] = plan.shots
        if plan.parameter_bindings:
            job.info["parameter_bindings"] = {
                k if isinstance(k, str) else k.name: v
                for k, v in plan.parameter_bindings.items()
            }

    def _check_compiled_incompatible(
        self, compiled: CompilationResult, plan: ExecutionPlan
    ) -> None:
        if not compiled.is_compatible and plan.options.get(
            "raise_on_incompatible", True
        ):
            raise ValueError(
                f"compiled plan '{plan.name}' is incompatible with its target: "
                f"{'; '.join(compiled.diagnostics)}"
            )

    def _check_capabilities(
        self, backend: Backend, executable: QuantumCircuit, plan: ExecutionPlan
    ) -> None:
        problems: list[str] = []
        target = self._select_target(plan)
        if target is not None:
            if (
                target.num_qubits is not None
                and executable.num_qubits > target.num_qubits
            ):
                problems.append(
                    f"circuit uses {executable.num_qubits} qubits but target "
                    f"'{target.name}' supports at most {target.num_qubits}"
                )
            if target.max_shots is not None and plan.shots > target.max_shots:
                problems.append(
                    f"plan requests {plan.shots} shots but target "
                    f"'{target.name}' supports at most {target.max_shots}"
                )
        problems.extend(backend.validate(plan))
        if problems:
            raise ValueError(
                f"plan '{plan.name}' cannot run on backend '{backend.name}': "
                f"{'; '.join(problems)}"
            )

    def _collect(
        self,
        job: Job,
        plan: ExecutionPlan,
        backend: Backend,
        trace: ExecutionTrace,
        elapsed: float,
    ) -> BackendResult:
        if job.status is JobStatus.CANCELLED:
            trace.record("cancelled")
            self._record_history(plan, backend, job, trace, elapsed, error=job.error)
            raise ValueError("job was cancelled before completion")
        if job.status is JobStatus.FAILED:
            trace.record("failed", error=job.error)
            self._record_history(plan, backend, job, trace, elapsed, error=job.error)
            raise ValueError(job.error or "backend job failed")
        if job.result is None:
            trace.record("empty")
            self._record_history(plan, backend, job, trace, elapsed)
            raise ValueError(f"job {job.job_id} produced no result")

        trace.record("completed", job_id=job.job_id)
        strategy = ExecutionStrategy.classify(plan)
        result = self._decorate(job, plan, backend, strategy, trace, elapsed)
        self._record_history(plan, backend, job, trace, elapsed, result=result)
        return result

    def _decorate(
        self,
        job: Job,
        plan: ExecutionPlan,
        backend: Backend,
        strategy: ExecutionStrategy,
        trace: ExecutionTrace,
        elapsed: float,
    ) -> BackendResult:
        if job.result is None:
            raise ValueError(f"job {job.job_id} produced no result")
        meta: dict[str, Any] = dict(job.result.metadata)
        meta.update(
            {
                "runtime": "ExecutionRuntime",
                "plan": plan.name,
                "job_id": job.job_id,
                "backend": backend.name,
                "shots": plan.shots,
                "strategy": strategy.value,
                "optimization_level": plan.optimization_level,
                "seed": plan.seed,
                "elapsed_seconds": round(elapsed, 6),
                "trace": trace.to_dict(),
            }
        )
        target = plan.target if plan.target is not None else backend.target
        if target is not None:
            meta["target"] = target.to_dict()
        if plan.parameter_bindings:
            meta["parameter_bindings"] = {
                k if isinstance(k, str) else k.name: v
                for k, v in plan.parameter_bindings.items()
            }
        for key, value in (plan.metadata or {}).items():
            if key not in ("trace", "parameter_bindings"):
                meta[key] = value
        return replace(
            job.result,
            metadata=meta,
            shots=plan.shots if job.result.shots is None else job.result.shots,
            seed=plan.seed if job.result.seed is None else job.result.seed,
            target_name=job.result.target_name
            if job.result.target_name is not None
            else (target.name if target is not None else backend.name),
        )

    def _record_history(
        self,
        plan: ExecutionPlan,
        backend: Backend,
        job: Job,
        trace: ExecutionTrace,
        elapsed: float,
        *,
        error: Optional[str] = None,
        result: Optional[BackendResult] = None,
    ) -> None:
        entry: dict[str, Any] = {
            "plan": plan.name,
            "backend": backend.name,
            "job_id": job.job_id,
            "status": job.status.value,
            "strategy": ExecutionStrategy.classify(plan).value,
            "shots": plan.shots,
            "optimization_level": plan.optimization_level,
            "elapsed_seconds": round(elapsed, 6),
            "trace": trace.to_dict(),
        }
        if plan.parameter_bindings:
            entry["parameter_bindings"] = {
                k if isinstance(k, str) else k.name: v
                for k, v in plan.parameter_bindings.items()
            }
        if error is not None:
            entry["error"] = error
        if result is not None:
            entry["counts"] = dict(result.counts)
        self._history.append(entry)
        if self._history_size >= 0 and len(self._history) > self._history_size:
            del self._history[: len(self._history) - self._history_size]

    def __repr__(self) -> str:
        return (
            f"ExecutionRuntime(backend={self.default_backend.name!r}, "
            f"history={len(self._history)})"
        )


__all__ = ["ExecutionRuntime"]