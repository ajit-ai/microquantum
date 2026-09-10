"""Structured execution records (MQ-07).

An :class:`ExecutionRecord` is the portable description of *one actual
execution*: plan context, backend/target, shots/seed, parameter bindings,
timing, status and the raw :class:`~microquantum.backends.base.BackendResult`
(or a structured :class:`ExecutionFailure`).

The record deliberately separates **runtime objects** (the live
:class:`BackendResult`) from **portable execution metadata**: :meth:`to_dict`
never embeds live objects, and :meth:`from_dict` currently restores the
portable metadata plus a JSON-safe copy of the result.  The architecture gap
``BackendResult -> ExecutionRecord -> ExperimentResult`` leaves the raw output
untouched for later aggregation and analysis.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, cast

import numpy as np

from .._json import JSONSerializable, json_safe, json_string


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _sdk_version() -> str:
    """Best-effort SDK version string (never raises)."""
    try:
        import microquantum  # type: ignore[import-not-found]

        return str(getattr(microquantum, "__version__", "unknown"))
    except Exception:
        return "unknown"


class ExecutionStatus(Enum):
    """Terminal state of a single execution record."""

    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# result round trips
# ---------------------------------------------------------------------------


def backend_result_from_dict(data: dict[str, Any]) -> Any:
    """Rebuild a :class:`BackendResult` from its ``to_dict()`` mapping.

    This is a thin wrapper around :meth:`BackendResult.from_dict` kept here so
    the experiments layer does not depend on internal backend helpers.
    """
    from ..backends.base import BackendResult

    return BackendResult.from_dict(data)


# ---------------------------------------------------------------------------
# structured failures
# ---------------------------------------------------------------------------


@dataclass
class ExecutionFailure:
    """A structured, inspectable execution failure.

    Failed executions are never silently dropped: batches surface them as
    records with :attr:`status` ``failed`` and a populated :class:`ExecutionFailure`
    describing *what* failed and *where*.

    Attributes:
        execution_id: Identifier of the failed execution record.
        backend: Backend name the work targeted (best effort).
        plan_name: Name of the plan being executed.
        error_type: Exception class name of the failure.
        message: Human-readable error message.
        parameter_bindings: Parameter bindings at the time of the failure.
        shots: Requested shot count (if known).
        seed: Requested seed (if known).
        timestamp: ISO-8601 time the failure was recorded.
        metadata: Free-form failure metadata.
    """

    execution_id: str
    backend: str
    plan_name: str = "unknown"
    error_type: str = "Exception"
    message: str = ""
    parameter_bindings: dict[str, float] = field(default_factory=dict)
    shots: Optional[int] = None
    seed: Optional[int] = None
    timestamp: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the failure to a JSON-safe dictionary."""
        return {
            "execution_id": self.execution_id,
            "backend": self.backend,
            "plan_name": self.plan_name,
            "error_type": self.error_type,
            "message": self.message,
            "parameter_bindings": json_safe(dict(self.parameter_bindings)),
            "shots": self.shots,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "metadata": json_safe(dict(self.metadata)),
        }

    def to_json(self) -> str:
        """Serialize the failure to a JSON string."""
        return json_string(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionFailure":
        """Rebuild a failure from its ``to_dict()`` mapping."""
        return cls(
            execution_id=str(data.get("execution_id", "")),
            backend=str(data.get("backend", "")),
            plan_name=str(data.get("plan_name", "unknown")),
            error_type=str(data.get("error_type", "Exception")),
            message=str(data.get("message", "")),
            parameter_bindings=dict(data.get("parameter_bindings") or {}),
            shots=data.get("shots"),
            seed=data.get("seed"),
            timestamp=str(data.get("timestamp") or _now_iso()),
            metadata=dict(data.get("metadata") or {}),
        )

    def __repr__(self) -> str:
        return (
            f"ExecutionFailure({self.error_type}, backend={self.backend!r}, "
            f"plan={self.plan_name!r}, msg={self.message!r})"
        )


# ---------------------------------------------------------------------------
# execution record
# ---------------------------------------------------------------------------


def _normalized_bindings(bindings: Optional[Mapping[str, Any]]) -> dict[str, float]:
    """Flatten parameter bindings to ``{name: float}`` (JSON-safe)."""
    out: dict[str, float] = {}
    for key, value in (bindings or {}).items():
        name = key if isinstance(key, str) else getattr(key, "name", str(key))
        if isinstance(value, complex):
            raise ValueError(f"binding '{name}' must be real-valued")
        out[name] = float(value)
    return out


def _plan_configuration(plan_name: str, **fields: Any) -> dict[str, Any]:
    """JSON-safe, fingerprintable subset of the execution configuration."""
    return {"plan_name": plan_name, **json_safe(dict(fields))}


def execution_fingerprint(configuration: Mapping[str, Any]) -> str:
    """Stable, deterministic SHA-256 fingerprint of an execution configuration.

    The fingerprint is derived from the **serialized configuration** (sorted
    keys, JSON-native values) — never from object memory addresses — and
    includes the SDK version so a record stays tied to the library that
    produced it.

    Note:
        A matching fingerprint means the *configuration* is reproducible
        (``configured reproducibility``).  Whether a backend actually produces
        identical output depends on its determinism (see
        ``deterministic_execution`` in :meth:`ExecutionRecord.reproducibility`).
    """
    import hashlib
    import json

    from .._json import json_safe as _safe

    payload = json.dumps(
        _safe(dict(configuration)),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reproducibility_metadata(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Reproducibility metadata for an execution configuration.

    Returns:
        A dict with ``fingerprint`` (SHA-256 of the serialized configuration),
        ``sdk_version``, ``configured_reproducibility=True`` (the same
        configuration maps to the same fingerprint) and
        ``deterministic_execution=None`` — hardware and nondeterministic
        backends are *not* claimed bit-for-bit reproducible; callers with
        backend-level determinism knowledge should set that value themselves.
    """
    return {
        "fingerprint": execution_fingerprint(configuration),
        "sdk_version": _sdk_version(),
        "configured_reproducibility": True,
        "deterministic_execution": None,
    }


@dataclass
class ExecutionRecord(JSONSerializable):
    """One actual execution plus its structured metadata.

    Attributes:
        execution_id: Unique identifier of the execution.
        plan_name: Name of the :class:`ExecutionPlan` that was run.
        status: :class:`ExecutionStatus` (``completed`` or ``failed``).
        backend: Name of the backend that was selected.
        target: Serialized target descriptor (or ``None``).
        shots: Number of shots requested.
        seed: RNG seed requested (or ``None``).
        parameter_bindings: Flat ``{param_name: value}`` map bound for this run.
        plan: JSON-safe snapshot of the plan configuration (includes the
            serialized circuit / IR).
        result: The raw :class:`~microquantum.backends.base.BackendResult`
            (live object, in memory only — never serialized directly).
        error: Structured :class:`ExecutionFailure` for failed executions.
        timing: Timing metadata (`total_seconds`, and `queue_seconds` /
            `execution_seconds` when the backend provides them, else ``None``).
        timestamp: ISO-8601 time the record was created.
        metadata: Free-form JSON-safe execution metadata.
        reproducibility: Reproducibility info (``fingerprint``, seed, SDK
            version, configured-vs-deterministic distinction).

    The record separates runtime objects from portable metadata: ``result``
    is the live runtime object; :meth:`to_dict` stores only its JSON-safe
    serialization and :meth:`from_dict` restores the metadata.
    """

    execution_id: str
    plan_name: str
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    backend: str = ""
    target: Optional[dict[str, Any]] = None
    shots: Optional[int] = None
    seed: Optional[int] = None
    parameter_bindings: dict[str, float] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[ExecutionFailure] = None
    timing: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
    reproducibility: dict[str, Any] = field(default_factory=dict)

    # -- construction helpers ----------------------------------------------

    @classmethod
    def completed(
        cls,
        plan: Any,
        backend: Any,
        result: Any,
        elapsed_seconds: float,
    ) -> "ExecutionRecord":
        """Build a completed record from an executed plan.

        Args:
            plan: The :class:`ExecutionPlan` that was executed.
            backend: The :class:`Backend` that produced the result.
            result: The raw :class:`BackendResult`.
            elapsed_seconds: Total wall-clock execution time.
        """
        target = plan.target if plan.target is not None else backend.target
        target_dict: Optional[dict[str, Any]] = None
        if target is not None:
            target_dict = json_safe(target.to_dict())
        bindings = _normalized_bindings(plan.parameter_bindings)
        configuration = {
            "plan_name": plan.name,
            "backend": backend.name,
            "target": target_dict,
            "shots": plan.shots,
            "seed": plan.seed,
            "parameter_bindings": bindings,
            "optimization_level": plan.optimization_level,
            "options": json_safe(dict(plan.options)),
            "metadata": json_safe(dict(plan.metadata)),
        }
        execution_id = uuid.uuid4().hex
        return cls(
            execution_id=execution_id,
            plan_name=plan.name,
            status=ExecutionStatus.COMPLETED,
            backend=backend.name,
            target=target_dict,
            shots=plan.shots,
            seed=plan.seed,
            parameter_bindings=bindings,
            plan=_plan_snapshot(plan),
            result=result,
            timing={
                "total_seconds": round(float(elapsed_seconds), 6),
                "queue_seconds": None,
                "execution_seconds": None,
            },
            metadata=json_safe(dict(result.metadata)),
            reproducibility=reproducibility_metadata(configuration),
        )

    @classmethod
    def failed(
        cls,
        plan: Any,
        error: BaseException,
        *,
        backend_name: Optional[str] = None,
        elapsed_seconds: float = 0.0,
    ) -> "ExecutionRecord":
        """Build a failed record carrying a structured :class:`ExecutionFailure`.

        Args:
            plan: The plan that failed (may be partially prepared).
            error: The raised exception (never swallowed — always recorded).
            backend_name: Best-effort backend name.
            elapsed_seconds: Elapsed time before the failure.
        """
        name = getattr(plan, "name", "unknown")
        backend = backend_name or _backend_name_of(plan)
        try:
            bindings = _normalized_bindings(getattr(plan, "parameter_bindings", None))
        except (TypeError, ValueError):
            bindings = {}
        plan_dict = _plan_snapshot(plan)
        execution_id = uuid.uuid4().hex
        configuration = {
            "plan_name": name,
            "backend": backend,
            "shots": getattr(plan, "shots", None),
            "seed": getattr(plan, "seed", None),
            "parameter_bindings": bindings,
            "metadata": json_safe(dict(getattr(plan, "metadata", {}) or {})),
        }
        return cls(
            execution_id=execution_id,
            plan_name=name,
            status=ExecutionStatus.FAILED,
            backend=backend,
            shots=getattr(plan, "shots", None),
            seed=getattr(plan, "seed", None),
            parameter_bindings=bindings,
            plan=plan_dict,
            error=ExecutionFailure(
                execution_id=execution_id,
                backend=backend,
                plan_name=name,
                error_type=type(error).__name__,
                message=str(error),
                parameter_bindings=bindings,
                shots=getattr(plan, "shots", None),
                seed=getattr(plan, "seed", None),
            ),
            timing={
                "total_seconds": round(float(elapsed_seconds), 6),
                "queue_seconds": None,
                "execution_seconds": None,
            },
            reproducibility=reproducibility_metadata(configuration),
        )

    # -- derived accessors ------------------------------------------------

    @property
    def is_success(self) -> bool:
        """True if the execution completed successfully."""
        return self.status is ExecutionStatus.COMPLETED

    @property
    def expectations(self) -> dict[str, float]:
        """Expectation values from the raw result (empty if none)."""
        result = self.result
        if result is None:
            return {}
        return dict(getattr(result, "expectations", {}) or {})

    @property
    def counts(self) -> dict[str, int]:
        """Measurement counts from the raw result (empty if none)."""
        result = self.result
        if result is None:
            return {}
        return dict(getattr(result, "counts", {}) or {})

    @property
    def statevector(self) -> Optional[np.ndarray]:
        """Statevector from the raw result (``None`` if unavailable)."""
        result = self.result
        if result is None:
            return None
        return getattr(result, "statevector", None)

    @property
    def density_matrix(self) -> Optional[np.ndarray]:
        """Density matrix from the raw result (``None`` if unavailable)."""
        result = self.result
        if result is None:
            return None
        return getattr(result, "density_matrix", None)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary (no live objects).

        The raw result is stored as its ``to_dict()`` form; the plan as a
        serialized snapshot.
        """
        return {
            "execution_id": self.execution_id,
            "plan_name": self.plan_name,
            "status": self.status.value,
            "backend": self.backend,
            "target": json_safe(self.target) if self.target is not None else None,
            "shots": self.shots,
            "seed": self.seed,
            "parameter_bindings": json_safe(dict(self.parameter_bindings)),
            "plan": json_safe(self.plan),
            "result": self.result.to_dict() if self.result is not None else None,
            "error": self.error.to_dict() if self.error is not None else None,
            "timing": json_safe(dict(self.timing)),
            "timestamp": self.timestamp,
            "metadata": json_safe(dict(self.metadata)),
            "reproducibility": json_safe(dict(self.reproducibility)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionRecord":
        """Rebuild a record from its ``to_dict()`` mapping.

        The raw result is restored to a :class:`BackendResult` (arrays are
        decoded back to complex128) so records remain analysable after a
        round trip.
        """
        status = ExecutionStatus(data.get("status", ExecutionStatus.COMPLETED.value))
        error_data = data.get("error")
        result_data = data.get("result")
        return cls(
            execution_id=str(data.get("execution_id", "")),
            plan_name=str(data.get("plan_name", "unknown")),
            status=status,
            backend=str(data.get("backend", "")),
            target=dict(data["target"]) if data.get("target") is not None else None,
            shots=data.get("shots"),
            seed=data.get("seed"),
            parameter_bindings=dict(data.get("parameter_bindings") or {}),
            plan=dict(data.get("plan") or {}),
            result=backend_result_from_dict(result_data) if result_data else None,
            error=ExecutionFailure.from_dict(error_data) if error_data else None,
            timing=dict(data.get("timing") or {}),
            timestamp=str(data.get("timestamp") or _now_iso()),
            metadata=dict(data.get("metadata") or {}),
            reproducibility=dict(data.get("reproducibility") or {}),
        )

    def to_json(self) -> str:
        """Serialize the record to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"ExecutionRecord({self.execution_id[:8]}..., "
            f"status={self.status.value}, backend={self.backend!r}, "
            f"plan={self.plan_name!r})"
        )


def _backend_name_of(plan: Any) -> str:
    """Best-effort backend name from a plan (instance or name)."""
    backend = getattr(plan, "backend", None)
    if backend is None:
        return "<none>"
    if isinstance(backend, str):
        return backend
    return getattr(backend, "name", "<unknown>")


def _plan_snapshot(plan: Any) -> dict[str, Any]:
    """JSON-safe plan snapshot, resilient to partially prepared plans."""
    to_dict = getattr(plan, "to_dict", None)
    if callable(to_dict):
        try:
            return cast(dict[str, Any], json_safe(to_dict()))
        except Exception:
            pass
    return {"name": getattr(plan, "name", "unknown")}


# re-exported for convenience (see experiments/__init__)
__all__ = [
    "ExecutionFailure",
    "ExecutionRecord",
    "ExecutionStatus",
    "execution_fingerprint",
    "reproducibility_metadata",
]