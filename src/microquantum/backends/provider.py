"""Provider abstraction: backend discovery and registration (MQ-06).

A :class:`Provider` owns a family of :class:`Backend` instances — local
simulators today, and (in future phases) hardware, cloud and custom
enterprise providers.  Providers only *discover and expose* backends; they
never execute anything.  Discovery never imports every possible vendor SDK:
future vendor providers will construct backends lazily from credentials the
user supplies.

:class:`LocalProvider` is the built-in provider wrapping the local
simulator backends on this machine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .base import Backend
from .density_matrix import DensityMatrixBackend
from .local import LocalSimulatorBackend
from .mock import MockBackend
from .registry import BackendRegistry
from .statevector import StatevectorBackend


class Provider(ABC):
    """Interface contract for backend discovery.

    Subclasses expose a set of :class:`Backend` instances under a provider
    name.  ``get_backend`` raises a descriptive error when the requested
    backend is not offered by the provider.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. ``"local"``)."""

    @abstractmethod
    def backends(self) -> list[Backend]:
        """Return the backends this provider manages."""

    def get_backend(self, name: str) -> Backend:
        """Look up a backend by name.

        Raises:
            KeyError: Naming the provider, the requested backend and the
                backends actually offered.
        """
        for backend in self.backends():
            if backend.name == name:
                return backend
        known = sorted(backend.name for backend in self.backends())
        raise KeyError(
            f"provider '{self.name}' has no backend '{name}'; "
            f"available backends: {known}"
        )

    def has_backend(self, name: str) -> bool:
        """Return True if this provider offers a backend named *name*."""
        return any(b.name == name for b in self.backends())

    def register_all(self, registry: BackendRegistry) -> list[str]:
        """Register every backend into *registry* and return registered names."""
        registered: list[str] = []
        for backend in self.backends():
            key = registry.register(backend, replace=True)
            registered.append(key)
        return registered

    def to_dict(self) -> dict[str, Any]:
        """Serialize provider identity and offered backends."""
        return {
            "name": self.name,
            "backends": [backend.name for backend in self.backends()],
        }

    def __repr__(self) -> str:
        names = [b.name for b in self.backends()]
        return f"{self.__class__.__name__}(name='{self.name}', backends={names})"


class LocalProvider(Provider):
    """Provider exposing the built-in local simulator backends."""

    def __init__(
        self,
        *,
        include_density_matrix: bool = True,
        local_name: str = "local_simulator",
    ) -> None:
        self._local_name = local_name
        self._include_density_matrix = include_density_matrix

    @property
    def name(self) -> str:
        return "local"

    def backends(self) -> list[Backend]:
        backends: list[Backend] = [
            LocalSimulatorBackend(),
            StatevectorBackend(),
            MockBackend(),
        ]
        if self._include_density_matrix:
            backends.append(DensityMatrixBackend())
        return backends


__all__ = [
    "LocalProvider",
    "Provider",
]