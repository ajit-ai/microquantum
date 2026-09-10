"""Lightweight backend registry (MQ-06).

The registry is the single place to register, discover and inspect
:class:`~microquantum.backends.base.Backend` instances:

* :meth:`register` — add a backend under its name (duplicates rejected).
* :meth:`get` / :meth:`has` — look up by name with helpful errors.
* :meth:`list` / :meth:`names` — enumerate available backends.
* :attr:`default` — the backend used when a plan names none.

It is deliberately simple: no plugin framework, no auto-discovery of
vendors.  Providers (see :mod:`microquantum.backends.provider`) feed the
registry; the runtime routes plans through it.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .base import Backend
from .local import LocalSimulatorBackend


class BackendRegistry:
    """Register and look up backends by name.

    Args:
        default: Optional default :class:`Backend`; equal to registering it
            as the default.  Falls back to a shared
            :class:`LocalSimulatorBackend` instance.
        backends: Optional iterable of backends to register up front.
    """

    def __init__(
        self,
        *,
        default: Optional[Backend] = None,
        backends: Optional[list[Backend]] = None,
    ) -> None:
        self._backends: dict[str, Backend] = {}
        self._default_backend: Optional[Backend] = None
        pending = list(backends or [])
        if default is not None:
            if not any(default is b for b in pending):
                self.register(default, default=True)
            else:
                self.register(default, default=True, replace=True)
        for backend in pending:
            if backend is not default:
                self.register(backend)
        # A bare registry always offers the local simulator as its default.
        if not self._backends:
            self.register(LocalSimulatorBackend(), default=True)

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(
        self,
        backend: Backend,
        *,
        name: Optional[str] = None,
        default: bool = False,
        replace: bool = False,
    ) -> str:
        """Register a backend.

        Args:
            backend: The backend to register.
            name: Explicit registration name; defaults to ``backend.name``.
            default: Also make this backend the registry default.
            replace: Allow overwriting an existing registration.

        Returns:
            The name the backend was registered under.

        Raises:
            TypeError: If ``backend`` is not a :class:`Backend`.
            ValueError: If the name is already registered and ``replace``
                is ``False``.
        """
        if not isinstance(backend, Backend):
            raise TypeError(
                f"BackendRegistry.register expects a Backend, "
                f"got {type(backend).__name__}"
            )
        key = name if name is not None else backend.name
        if not key:
            raise ValueError("backend registration requires a non-empty name")
        if key in self._backends and not replace:
            raise ValueError(
                f"backend '{key}' is already registered; "
                f"use replace=True to overwrite it"
            )
        self._backends[key] = backend
        if default:
            self._default_backend = backend
        return key

    def unregister(self, name: str) -> Backend:
        """Remove and return the backend registered under *name*.

        Raises:
            ValueError: If no backend is registered under *name*.
        """
        if name not in self._backends:
            raise ValueError(self._missing_message(name))
        backend = self._backends.pop(name)
        if self._default_backend is backend:
            self._default_backend = None
        return backend

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Backend:
        """Return the backend registered under *name*.

        Raises:
            KeyError: With the list of known backends when unknown.
        """
        try:
            return self._backends[name]
        except KeyError:
            raise KeyError(self._missing_message(name)) from None

    def has(self, name: str) -> bool:
        """Return True if a backend is registered under *name*."""
        return name in self._backends

    def get_or_none(self, name: str) -> Optional[Backend]:
        """Return the backend under *name*, or ``None`` if unknown."""
        return self._backends.get(name)

    # ------------------------------------------------------------------
    # enumeration
    # ------------------------------------------------------------------

    def list(self) -> List[Backend]:
        """Return all registered backends (insertion order)."""
        return list(self._backends.values())

    def names(self) -> List[str]:
        """Return the names of all registered backends."""
        return list(self._backends)

    def __len__(self) -> int:
        return len(self._backends)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._backends

    # ------------------------------------------------------------------
    # default
    # ------------------------------------------------------------------

    @property
    def default(self) -> Optional[Backend]:
        """The backend used when no plan/backend names one."""
        if self._default_backend is not None:
            return self._default_backend
        if not self._backends:
            return None
        # An explicit default, else the first registered backend.
        return next(iter(self._backends.values()))

    def set_default(self, name: str) -> Backend:
        """Set the default backend by registered name.

        Raises:
            KeyError: If *name* is not registered.
        """
        backend = self.get(name)
        self._default_backend = backend
        return backend

    def reset_default(self) -> None:
        """Clear the explicit default (falls back to first registered)."""
        self._default_backend = None

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry state (names + capability summaries)."""
        return {
            "backends": [
                {
                    "name": backend.name,
                    "capabilities": backend.capabilities.to_dict(),
                }
                for backend in self._backends.values()
            ],
            "default": self.default.name if self.default is not None else None,
        }

    def to_json(self) -> str:
        """Serialize registry state to a JSON string."""
        from .._json import json_string

        return json_string(self.to_dict())

    def _missing_message(self, name: str) -> str:
        known = self.names()
        if not known:
            return f"no backend registered as '{name}'"
        return (
            f"unknown backend '{name}'; "
            f"available backends: {sorted(known)}"
        )

    def __repr__(self) -> str:
        default = self.default.name if self.default is not None else None
        return f"BackendRegistry(backends={list(self._backends)}, default={default!r})"


#: Shared default registry instance used by convenience helpers.
default_registry = BackendRegistry(default=LocalSimulatorBackend())

__all__ = ["BackendRegistry", "default_registry"]