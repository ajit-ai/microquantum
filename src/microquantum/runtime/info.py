"""Runtime introspection: a stable, derived description of the runtime.

:class:`RuntimeInfo` / :func:`runtime_info` expose a small, JSON-safe snapshot
of what the runtime can actually do — nothing is guessed:

* MicroQuantum version (from installed distribution metadata);
* Python / NumPy versions of the running interpreter;
* the runtime implementation and its execution strategies
  (:class:`~microquantum.runtime.ExecutionStrategy`);
* the backends resolved through the runtime's registry (or the module-level
  ``default_registry``), with their canonical
  :class:`~microquantum.backends.capabilities.BackendCapabilities` summaries;
* the default backend used when a plan names none.

The backend list is derived from the live registry — it cannot go stale, and
no hand-maintained capability matrix is kept anywhere.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Optional

import numpy as np

from ..backends.registry import BackendRegistry, default_registry
from .strategy import ExecutionStrategy

try:  # pragma: no cover - defensive fallback
    _MQ_VERSION = metadata.version("microquantum")
except metadata.PackageNotFoundError:  # pragma: no cover - bare source tree
    _MQ_VERSION = "unknown"

RUNTIME_IMPLEMENTATION = "ExecutionRuntime"


def _backend_summary(backend: Any) -> dict[str, Any]:
    """Canonical capability summary of one backend."""
    capabilities = backend.capabilities.to_dict()
    return {
        "name": backend.name,
        "target_class": capabilities.get("target_class"),
        "execution": list(capabilities.get("execution", ())),
        "circuit_features": list(capabilities.get("circuit_features", ())),
        "max_qubits": capabilities.get("max_qubits"),
        "precision": capabilities.get("precision"),
    }


@dataclass(frozen=True)
class RuntimeInfo:
    """JSON-safe snapshot of the runtime implementation.

    Attributes:
        version: MicroQuantum package version.
        runtime: Runtime implementation name (``"ExecutionRuntime"``).
        python: Python version string of the running interpreter.
        numpy: Installed NumPy version.
        strategies: Supported execution-strategy names, sorted.
        backends: One capability summary per registered backend.
        default_backend: Backend used when a plan names none.
        default_optimization_level: Optimization level used for bare circuits.
    """

    version: str
    runtime: str
    python: str
    numpy: str
    strategies: list[str]
    backends: list[dict[str, Any]]
    default_backend: Optional[str]
    default_optimization_level: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snapshot to a JSON-safe dictionary."""
        return {
            "version": self.version,
            "runtime": self.runtime,
            "python": self.python,
            "numpy": self.numpy,
            "strategies": list(self.strategies),
            "backends": list(self.backends),
            "default_backend": self.default_backend,
            "default_optimization_level": self.default_optimization_level,
        }


def runtime_info(
    runtime: Optional[Any] = None, *, registry: Optional[BackendRegistry] = None
) -> RuntimeInfo:
    """Return a snapshot of the runtime implementation.

    Args:
        runtime: Optional :class:`~microquantum.runtime.ExecutionRuntime` to
            introspect.  ``None`` introspects the shared default runtime.
        registry: Optional :class:`~microquantum.backends.registry.BackendRegistry`
            override.  Defaults to the default runtime's registry, then the
            module-level ``default_registry``.

    Returns:
        A :class:`RuntimeInfo` with version, runtime, strategies and a
        capability summary for every registered backend.
    """
    registry = registry or (
        runtime.registry if runtime is not None else None
    ) or default_registry

    backends = [_backend_summary(backend) for backend in registry.list()]

    default_backend: Optional[str] = None
    default_optimization_level = 0
    config = getattr(runtime, "config", None)
    if config is not None:
        default_optimization_level = config.default_optimization_level
    if runtime is not None:
        try:
            default_backend = runtime.default_backend.name
        except Exception:  # pragma: no cover - defensive
            default_backend = None
    elif registry.default is not None:
        default_backend = registry.default.name

    return RuntimeInfo(
        version=_MQ_VERSION,
        runtime=RUNTIME_IMPLEMENTATION,
        python=platform.python_version(),
        numpy=np.__version__,
        strategies=sorted(strategy.value for strategy in ExecutionStrategy),
        backends=backends,
        default_backend=default_backend,
        default_optimization_level=default_optimization_level,
    )


__all__ = ["RUNTIME_IMPLEMENTATION", "RuntimeInfo", "runtime_info"]