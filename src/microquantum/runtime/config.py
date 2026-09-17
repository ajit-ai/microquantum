"""Runtime configuration: a plain, JSON-safe description of runtime defaults.

:class:`RuntimeConfig` is the single documented way to describe the default
``ExecutionRuntime`` behaviour: the fallback backend, the registry used to
resolve backend *names*, the default compile target, the history cap and the
default compiler optimization level.  It is a plain frozen dataclass — no
framework, no global state.  Pass it to :class:`ExecutionRuntime` via
``config=...`` or obtain a configured copy of a runtime with
:meth:`ExecutionRuntime.configure`.

Only options the runtime actually honours are present: options the runtime
cannot support are deliberately absent (``execution_mode``, backend arrays,
unsupported execution levels, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from .._json import json_string
from ..backends.base import Backend
from ..backends.registry import BackendRegistry

if TYPE_CHECKING:
    from ..core.device import Target

#: Backend reference accepted by the runtime: an instance or a name.
BackendRef = Union[Backend, str]


def _validate_history_size(value: int) -> int:
    if not isinstance(value, int):
        raise TypeError(
            f"history_size must be an integer, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"history_size must be >= 0, got {value}")
    return value


def _validate_optimization_level(value: int) -> int:
    if not isinstance(value, int):
        raise TypeError(
            f"default_optimization_level must be an int, got {type(value).__name__}"
        )
    if value < 0 or value > 2:
        raise ValueError(
            f"default_optimization_level must be in [0, 2], got {value}"
        )
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    """Default configuration for an :class:`~microquantum.runtime.ExecutionRuntime`.

    Attributes:
        backend: Default :class:`~microquantum.backends.base.Backend` (or a
            registered backend *name*) used when a plan names none.
            ``None`` falls back to the registry default, then a lazily-created
            ``statevector`` simulator.
        registry: Optional :class:`~microquantum.backends.registry.BackendRegistry`
            used to resolve backend names and to pick the fallback default
            backend.
        default_target: Default :class:`~microquantum.core.device.Target`
            applied to plans that do not name a target.  ``None`` disables
            default target processing.
        history_size: Maximum number of entries retained in the runtime's
            :attr:`~microquantum.runtime.ExecutionRuntime.history` (``0``
            disables history recording).
        default_optimization_level: Compiler optimization level (0-2) used
            when a plan is built from a raw circuit and carries none.
    """

    backend: Optional[BackendRef] = None
    registry: Optional[BackendRegistry] = None
    default_target: Optional[Target] = None
    history_size: int = 200
    default_optimization_level: int = 0

    def __post_init__(self) -> None:
        if self.backend is not None and not isinstance(self.backend, (Backend, str)):
            raise TypeError(
                "RuntimeConfig.backend must be a Backend, a name string or "
                f"None, got {type(self.backend).__name__}"
            )
        if self.registry is not None and not isinstance(self.registry, BackendRegistry):
            raise TypeError(
                "RuntimeConfig.registry must be a BackendRegistry or None, "
                f"got {type(self.registry).__name__}"
            )
        if self.default_target is not None:
            from ..core.device import Target

            if not isinstance(self.default_target, Target):
                raise TypeError(
                    "RuntimeConfig.default_target must be a Target or None, "
                    f"got {type(self.default_target).__name__}"
                )
        object.__setattr__(self, "history_size", _validate_history_size(self.history_size))
        object.__setattr__(
            self,
            "default_optimization_level",
            _validate_optimization_level(self.default_optimization_level),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-safe dictionary."""
        backend_value: Optional[str] = None
        if self.backend is not None:
            backend_value = (
                self.backend if isinstance(self.backend, str) else self.backend.name
            )
        registry_names = (
            list(self.registry.names()) if self.registry is not None else []
        )
        return {
            "backend": backend_value,
            "registry_backends": registry_names,
            "default_target": (
                self.default_target.to_dict()
                if self.default_target is not None
                else None
            ),
            "history_size": self.history_size,
            "default_optimization_level": self.default_optimization_level,
        }

    def to_json(self) -> str:
        """Serialize the configuration to a JSON string."""
        return json_string(self.to_dict())


__all__ = ["RuntimeConfig"]