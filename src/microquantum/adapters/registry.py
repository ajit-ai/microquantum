"""Name-based registry for domain adapters."""

from __future__ import annotations

from typing import Union

from .base import DomainAdapter, QuantumProblem

__all__ = [
    "AdapterRegistry",
]

_AdapterLike = Union[DomainAdapter, type[DomainAdapter]]


class AdapterRegistry:
    """Maps adapter names to domain adapters and resolves by problem.

    Registration accepts an instance or a no-argument adapter class.
    Resolution prefers an exact ``domain_name`` match, then falls back to
    the first adapter whose :meth:`can_handle` accepts the problem.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, DomainAdapter] = {}

    def register(self, name: str, adapter: _AdapterLike) -> None:
        """Register *adapter* under *name* (overwrites on conflict)."""
        if not name:
            raise ValueError("Adapter name must be non-empty")
        instance = adapter() if isinstance(adapter, type) else adapter
        if not isinstance(instance, DomainAdapter):
            raise TypeError(
                f"Expected a DomainAdapter, got {type(instance).__name__}"
            )
        self._adapters[name] = instance

    def resolve(self, problem: QuantumProblem) -> DomainAdapter:
        """Return the best adapter for *problem*.

        Raises:
            LookupError: If no registered adapter can handle the problem.
        """
        for adapter in self._adapters.values():
            if adapter.domain_name == problem.domain and adapter.can_handle(problem):
                return adapter
        for adapter in self._adapters.values():
            if adapter.can_handle(problem):
                return adapter
        raise LookupError(
            f"No registered adapter can handle domain '{problem.domain}' "
            f"(registered: {sorted(self._adapters)})"
        )

    def registered(self) -> list[str]:
        """Registered adapter names in registration order."""
        return list(self._adapters)

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, name: object) -> bool:
        return name in self._adapters
