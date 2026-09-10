"""Experiments: collections of repeated, related executions (MQ-07).

An :class:`Experiment` describes *what a collection of executions looks like*
(fixed plans and/or parameter sweeps, a backend/target, shots/seed) and, once
run against an :class:`~microquantum.runtime.ExecutionRuntime`, accumulates the
resulting :class:`~microquantum.experiments.record.ExecutionRecord` objects.

An :class:`ExperimentResult` is the **structured experiment output**: the raw
execution records are preserved verbatim (aggregation/analysis never destroys
them), plus derived aggregate metadata and an inspectable list of failures.

Experiments are fully in-memory (no database) and serializable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from .._json import JSONSerializable, json_safe, json_string
from ..core.circuit import QuantumCircuit
from .record import (
    ExecutionFailure,
    ExecutionRecord,
)
from .sweep import ParameterSweep


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flatten_bindings(bindings: Any) -> dict[str, float]:
    """Flatten a binding mapping to ``{name: float}``."""
    out: dict[str, float] = {}
    for key, value in (bindings or {}).items():
        name = key if isinstance(key, str) else getattr(key, "name", str(key))
        out[name] = float(value)
    return out


@dataclass
class ExperimentResult(JSONSerializable):
    """Structured output of an :class:`Experiment`.

    Attributes:
        experiment_id: Experiment identifier.
        name: Experiment name.
        description: Free-form description.
        status: ``"completed"`` (no failures), ``"failed"`` (at least one
            failure), ``"running"`` or ``"pending"``.
        records: The raw :class:`ExecutionRecord` objects (never summarized).
        failures: Structured :class:`ExecutionFailure` objects (each also
            inside its failed record).
        backend_names: Unique backend names used, in first-use order.
        shots: Requested shot count (if a single value applies).
        seed: Requested seed (if a single value applies).
        metadata: Free-form JSON-safe experiment metadata.
        created_at / completed_at: ISO-8601 timestamps.

    The raw-records-preserved contract is central: ``records`` are never
    replaced by summaries; analysis operates on them downstream.
    """

    experiment_id: str
    name: str
    description: str = ""
    status: str = "pending"
    records: list[ExecutionRecord] = field(default_factory=list)
    failures: list[ExecutionFailure] = field(default_factory=list)
    backend_names: list[str] = field(default_factory=list)
    shots: Optional[int] = None
    seed: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    completed_at: str = ""

    @classmethod
    def from_experiment(
        cls,
        experiment: "Experiment",
        records: Sequence[ExecutionRecord],
    ) -> "ExperimentResult":
        """Assemble a result from an experiment and its executed records."""
        successes = [r for r in records if r.is_success]
        failures = [r.error for r in records if r.error is not None]
        all_ok = len(records) > 0 and len(successes) == len(records)
        backends: list[str] = []
        for record in records:
            if record.backend not in backends:
                backends.append(record.backend)
        return cls(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            description=experiment.description,
            status="completed" if all_ok else "failed",
            records=list(records),
            failures=failures,
            backend_names=backends,
            shots=experiment.shots,
            seed=experiment.seed,
            metadata=json_safe(dict(experiment.metadata)),
            completed_at=_now_iso(),
        )

    # -- derived accessors ---------------------------------------------------

    @property
    def executions(self) -> list[ExecutionRecord]:
        """Alias for :attr:`records` (the raw execution list)."""
        return self.records

    @property
    def successes(self) -> list[ExecutionRecord]:
        """Completed execution records."""
        return [r for r in self.records if r.is_success]

    @property
    def success_count(self) -> int:
        """Number of completed executions."""
        return len(self.successes)

    @property
    def failure_count(self) -> int:
        """Number of failed executions."""
        return len(self.failures)

    @property
    def all_successful(self) -> bool:
        """True when every execution completed."""
        return len(self.records) > 0 and self.failure_count == 0

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary (records included verbatim)."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "records": [r.to_dict() for r in self.records],
            "failures": [f.to_dict() for f in self.failures],
            "backend_names": list(self.backend_names),
            "shots": self.shots,
            "seed": self.seed,
            "metadata": json_safe(dict(self.metadata)),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        """Rebuild a result from its ``to_dict()`` mapping."""
        return cls(
            experiment_id=str(data.get("experiment_id", "")),
            name=str(data.get("name", "experiment")),
            description=str(data.get("description", "")),
            status=str(data.get("status", "pending")),
            records=[
                ExecutionRecord.from_dict(r)
                for r in (data.get("records") or [])
            ],
            failures=[
                ExecutionFailure.from_dict(f)
                for f in (data.get("failures") or [])
            ],
            backend_names=list(data.get("backend_names") or []),
            shots=data.get("shots"),
            seed=data.get("seed"),
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at") or _now_iso()),
            completed_at=str(data.get("completed_at") or ""),
        )

    def to_json(self) -> str:
        """Serialize the result to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"ExperimentResult('{self.name}', status={self.status}, "
            f"executions={len(self.records)}, "
            f"successes={self.success_count}, failures={self.failure_count})"
        )


class Experiment:
    """A collection of related executions (in-memory experiment abstraction).

    Supports fixed plans/circuits *and* parameter sweeps; execution is
    delegated entirely to an existing :class:`ExecutionRuntime` (backend
    routing logic is never duplicated here).

    Example:
        .. code-block:: python

            exp = Experiment("vqe-sweep", shots=4096, seed=42)
            exp.add_circuit(ansatz, backend="statevector")
            exp.add_sweep(ParameterSweep({"theta": [...]}))
            result = exp.run(runtime)
    """

    def __init__(
        self,
        name: str = "experiment",
        *,
        experiment_id: Optional[str] = None,
        description: str = "",
        backend: Any = None,
        target: Any = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        optimization_level: int = 0,
        options: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create an experiment.

        Args:
            name: Human-readable experiment name.
            experiment_id: Optional explicit identifier (auto-generated).
            description: Free-form description.
            backend: Default backend (instance or registered name) applied to
                circuits added without an explicit backend.
            target: Optional default :class:`Target`.
            shots: Default number of shots for circuit-based executions.
            seed: Default seed for circuit-based executions.
            optimization_level: Default compiler optimization level.
            options: Default plan options.
            metadata: Free-form metadata attached to the result.
        """
        self._experiment_id = experiment_id or uuid.uuid4().hex
        self._name = str(name)
        self._description = str(description)
        self._backend = backend
        self._target = target
        self._shots = shots
        self._seed = seed
        self._optimization_level = optimization_level
        self._options = dict(options or {})
        self._metadata = dict(metadata or {})
        self._fixed: list[ExecutionPlanLike] = []
        self._sweeps: list[tuple[ParameterSweep, Any]] = []
        self._records: list[ExecutionRecord] = []
        self._status = "pending"

    # -- identity & configuration -------------------------------------------

    @property
    def experiment_id(self) -> str:
        """Unique experiment identifier."""
        return self._experiment_id

    @property
    def name(self) -> str:
        """Experiment name."""
        return self._name

    @property
    def description(self) -> str:
        """Free-form description."""
        return self._description

    @property
    def status(self) -> str:
        """Current lifecycle status (``pending``/``running``/``completed``/``failed``)."""
        return self._status

    @property
    def shots(self) -> Optional[int]:
        """Default shot count for circuit-based executions."""
        return self._shots

    @property
    def seed(self) -> Optional[int]:
        """Default seed for circuit-based executions."""
        return self._seed

    @property
    def metadata(self) -> dict[str, Any]:
        """Free-form experiment metadata (read-only view)."""
        return dict(self._metadata)

    @property
    def records(self) -> list[ExecutionRecord]:
        """Execution records accumulated by :meth:`run`."""
        return list(self._records)

    @property
    def execution_count(self) -> int:
        """Number of executions the current configuration will produce."""
        return len(self._fixed) + sum(
            sweep.num_combinations for sweep, _ in self._sweeps
        )

    # -- content ----------------------------------------------------------

    def add_plan(self, plan: Any) -> "Experiment":
        """Add a fixed :class:`ExecutionPlan` to the experiment."""
        from ..runtime.plan import ExecutionPlan

        if not isinstance(plan, ExecutionPlan):
            raise TypeError(
                f"Experiment.add_plan expects an ExecutionPlan, "
                f"got {type(plan).__name__}"
            )
        self._fixed.append(plan)
        return self

    def add_plans(self, plans: Sequence[Any]) -> "Experiment":
        """Add several fixed plans in order."""
        for plan in plans:
            self.add_plan(plan)
        return self

    def add_circuit(
        self,
        circuit: QuantumCircuit,
        *,
        name: str = "plan",
        backend: Any = None,
        target: Any = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        optimization_level: Optional[int] = None,
        parameter_bindings: Any = None,
        options: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Experiment":
        """Add a circuit executed with the experiment defaults (overridable)."""
        from ..runtime.plan import ExecutionPlan

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                f"Experiment.add_circuit expects a QuantumCircuit, "
                f"got {type(circuit).__name__}"
            )
        plan = ExecutionPlan.from_circuit(
            circuit,
            name=name,
            backend=backend if backend is not None else self._backend,
            target=target if target is not None else self._target,
            shots=shots if shots is not None else (self._shots or 1024),
            seed=seed if seed is not None else self._seed,
            optimization_level=(
                optimization_level
                if optimization_level is not None
                else self._optimization_level
            ),
            parameter_bindings=parameter_bindings,
            options=options if options is not None else self._options,
            metadata=metadata if metadata is not None else self._metadata,
        )
        self._fixed.append(plan)
        return self

    def add_sweep(
        self,
        sweep: ParameterSweep,
        base: Any = None,
    ) -> "Experiment":
        """Add a parameter sweep over a base plan or circuit.

        Args:
            sweep: The :class:`ParameterSweep` to expand.
            base: Base work the sweep binds to — an :class:`ExecutionPlan` or
                a :class:`QuantumCircuit`.  ``None`` uses the first fixed plan
                added to the experiment (if any).

        Raises:
            TypeError: If ``base`` is neither a plan nor a circuit.
            ValueError: If the sweep is incompatible with the base parameters.
        """
        from ..runtime.plan import ExecutionPlan

        if base is None:
            if not self._fixed:
                raise ValueError(
                    "add_sweep requires a base plan/circuit when the experiment "
                    "has no fixed plans yet"
                )
            base = self._fixed[0]
        if not isinstance(base, (ExecutionPlan, QuantumCircuit)):
            raise TypeError(
                "add_sweep base must be an ExecutionPlan or QuantumCircuit, "
                f"got {type(base).__name__}"
            )
        if not isinstance(sweep, ParameterSweep):
            raise TypeError(
                "add_sweep expects a ParameterSweep, "
                f"got {type(sweep).__name__}"
            )
        available = (
            {p.name for p in base.parameters}
            if isinstance(base, ExecutionPlan)
            else {p.name for p in base.parameters}
        )
        sweep.verify(available)
        self._sweeps.append((sweep, base))
        return self

    # -- lifecycle ----------------------------------------------------------

    def _expand(self) -> list[Any]:
        """Expand the experiment configuration into ordered plans."""
        from ..runtime.plan import ExecutionPlan

        plans: list[Any] = []
        plans.extend(plan for plan in self._fixed)
        for sweep, base in self._sweeps:
            base_plan = (
                base
                if isinstance(base, ExecutionPlan)
                else ExecutionPlan.from_circuit(
                    base,
                    name=sweep.name,
                    backend=self._backend,
                    target=self._target,
                    shots=self._shots or 1024,
                    seed=self._seed,
                    optimization_level=self._optimization_level,
                    options=self._options,
                    metadata=dict(self._metadata),
                )
            )
            base_bindings = _flatten_bindings(base_plan.parameter_bindings)
            prefix = (
                base_plan.name
                if base_plan.name not in ("main", sweep.name)
                else sweep.name
            )
            for index, combination in enumerate(sweep.combinations()):
                bindings = dict(base_bindings)
                bindings.update(combination)
                plan = replace_plan(
                    base_plan,
                    name=f"{prefix}-{index}",
                    parameter_bindings=bindings,
                    metadata={
                        **dict(base_plan.metadata),
                        "sweep_name": sweep.name,
                        "sweep_index": index,
                        "parameter_bindings": bindings,
                    },
                )
                plans.append(plan)
        return plans

    def run(self, runtime: Any) -> ExperimentResult:
        """Execute the experiment through an existing runtime.

        Args:
            runtime: An :class:`ExecutionRuntime` (or any object exposing
                ``execute_records``) used for backends, plans and batches.

        Raises:
            ValueError: If the experiment has no executions to run.
        """
        from ..runtime.runtime import ExecutionRuntime

        if not isinstance(runtime, ExecutionRuntime):
            raise TypeError(
                f"Experiment.run expects an ExecutionRuntime, "
                f"got {type(runtime).__name__}"
            )
        plans = self._expand()
        if not plans:
            raise ValueError(
                f"experiment '{self._name}' has no executions to run"
            )
        self._status = "running"
        self._records = list(runtime.execute_records(plans))
        self._status = (
            "completed"
            if self._records
            and all(r.is_success for r in self._records)
            else "failed"
        )
        return ExperimentResult.from_experiment(self, self._records)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the experiment *configuration* (not its records)."""
        return {
            "experiment_id": self._experiment_id,
            "name": self._name,
            "description": self._description,
            "status": self._status,
            "backend": (
                self._backend if isinstance(self._backend, str) else
                (getattr(self._backend, "name", None) if self._backend is not None else None)
            ),
            "target": self._target.to_dict() if self._target is not None else None,
            "shots": self._shots,
            "seed": self._seed,
            "optimization_level": self._optimization_level,
            "options": json_safe(dict(self._options)),
            "metadata": json_safe(dict(self._metadata)),
            "execution_count": self.execution_count,
        }

    def to_json(self) -> str:
        """Serialize the experiment configuration to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"Experiment('{self._name}', status={self._status}, "
            f"executions={self.execution_count})"
        )


#: A plan-like object accepted as fixed content (kept generic for typing ease).
ExecutionPlanLike = Any


def replace_plan(plan: Any, **changes: Any) -> Any:
    """Return a shallow copy of a plan with ``dataclasses.replace``.

    Referenced work objects (circuits, IR) are shared by reference, avoiding
    unnecessary copies of large state during sweep expansion.
    """
    from dataclasses import replace

    return replace(plan, **changes)


__all__ = ["Experiment", "ExperimentResult"]