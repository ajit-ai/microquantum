"""Result caching for domain adapters.

Caches quantum execution results so that identical problems
run on the same backend produce cached results instead of
re-executing. Supports TTL-based invalidation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .base import QuantumProblem, QuantumResult


@dataclass
class CacheEntry:
    """A single cached result entry.

    Attributes:
        result: The cached quantum result.
        created_at: Timestamp of cache creation.
        ttl: Time-to-live in seconds. None = no expiry.
        hit_count: Number of times this entry was retrieved.
    """

    result: QuantumResult
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    @property
    def age(self) -> float:
        """Age of the cache entry in seconds."""
        return time.time() - self.created_at


class ResultCache:
    """LRU-style cache for quantum execution results.

    Keys are derived from problem content + backend name + shots,
    ensuring that identical configurations return cached results.

    Attributes:
        max_size: Maximum number of entries. None = unlimited.
        default_ttl: Default time-to-live in seconds. None = no expiry.
    """

    def __init__(
        self,
        max_size: Optional[int] = 1000,
        default_ttl: Optional[float] = None,
    ) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(
        self,
        problem: QuantumProblem,
        backend_name: str,
        shots: int,
    ) -> Optional[QuantumResult]:
        """Retrieve a cached result.

        Args:
            problem: The problem to look up.
            backend_name: Backend used for execution.
            shots: Number of shots.

        Returns:
            Cached QuantumResult or None if not found/expired.
        """
        key = self._make_key(problem, backend_name, shots)
        entry = self._entries.get(key)

        if entry is None:
            return None

        if entry.is_expired:
            del self._entries[key]
            return None

        entry.hit_count += 1
        return entry.result

    def put(
        self,
        problem: QuantumProblem,
        backend_name: str,
        shots: int,
        result: QuantumResult,
        ttl: Optional[float] = None,
    ) -> None:
        """Store a result in the cache.

        Args:
            problem: The problem that was solved.
            backend_name: Backend used.
            shots: Number of shots.
            result: The result to cache.
            ttl: Optional per-entry TTL override.
        """
        key = self._make_key(problem, backend_name, shots)

        if self._max_size and len(self._entries) >= self._max_size:
            self._evict_oldest()

        self._entries[key] = CacheEntry(
            result=result,
            ttl=ttl or self._default_ttl,
        )

    def invalidate(self, problem: QuantumProblem) -> int:
        """Remove all cached entries for a given problem.

        Args:
            problem: The problem to invalidate.

        Returns:
            Number of entries removed.
        """
        prefix = problem.problem_id
        to_remove = [
            k for k in self._entries if k.startswith(prefix)
        ]
        for k in to_remove:
            del self._entries[k]
        return len(to_remove)

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        count = len(self._entries)
        self._entries.clear()
        return count

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._entries)

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        total_hits = sum(e.hit_count for e in self._entries.values())
        expired = sum(1 for e in self._entries.values() if e.is_expired)
        return {
            "size": len(self._entries),
            "max_size": self._max_size,
            "total_hits": total_hits,
            "expired": expired,
        }

    def _make_key(
        self,
        problem: QuantumProblem,
        backend_name: str,
        shots: int,
    ) -> str:
        """Generate cache key from problem + execution config."""
        return f"{problem.problem_id}:{backend_name}:{shots}"

    def _evict_oldest(self) -> None:
        """Remove the oldest entry."""
        if not self._entries:
            return
        oldest_key = min(self._entries, key=lambda k: self._entries[k].created_at)
        del self._entries[oldest_key]

    def __repr__(self) -> str:
        return f"ResultCache(size={len(self._entries)}, max_size={self._max_size})"

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries
