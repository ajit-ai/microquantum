"""Dependency-aware batch scheduling for execution plans.

:class:`DAGScheduler` orders plans into levels via explicit index
dependencies (Kahn's algorithm) and packs each level into batches of at
most ``max_parallel`` plans, preserving submission order within a
level.  Batches are plain lists consumed by ``execute_batch``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "DAGScheduler",
    "ScheduledBatch",
]


@dataclass(frozen=True)
class ScheduledBatch:
    """One schedulable batch: plan indices plus their level."""

    indices: tuple[int, ...]
    level: int

    def __len__(self) -> int:
        return len(self.indices)


class DAGScheduler:
    """Level scheduler over explicit plan dependencies.

    Args:
        max_parallel: Maximum plans per batch (must be >= 1).
    """

    def __init__(self, max_parallel: int = 4) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self._max_parallel = max_parallel

    @property
    def max_parallel(self) -> int:
        """Maximum plans per batch."""
        return self._max_parallel

    def schedule(
        self,
        plans: Sequence[Any],
        dependencies: Optional[Mapping[int, set[int]]] = None,
    ) -> list[ScheduledBatch]:
        """Schedule *plans* into ordered batches.

        Args:
            plans: Plans to schedule (only positions matter).
            dependencies: Optional ``{index: {prerequisite indices}}``
                edges.  Out-of-range indices are rejected; cycles raise
                ``ValueError``.

        Returns:
            Batches in execution order; plans within a batch are
            independent of each other.
        """
        count = len(plans)
        edges: dict[int, set[int]] = {i: set() for i in range(count)}
        if dependencies:
            for target, prereqs in dependencies.items():
                if not 0 <= target < count:
                    raise ValueError(f"Dependency target {target} out of range")
                for prereq in prereqs:
                    if not 0 <= prereq < count:
                        raise ValueError(f"Dependency prerequisite {prereq} out of range")
                    if prereq == target:
                        raise ValueError(f"Plan {target} cannot depend on itself")
                    edges[target].add(prereq)
        levels = self._levels(count, edges)
        batches: list[ScheduledBatch] = []
        for level, members in enumerate(levels):
            for start in range(0, len(members), self._max_parallel):
                chunk = tuple(members[start : start + self._max_parallel])
                batches.append(ScheduledBatch(indices=chunk, level=level))
        return batches

    @staticmethod
    def _levels(count: int, edges: dict[int, set[int]]) -> list[list[int]]:
        """Kahn's algorithm returning index levels in dependency order."""
        remaining = {index: set(prereqs) for index, prereqs in edges.items()}
        levels: list[list[int]] = []
        done: set[int] = set()
        while len(done) < count:
            ready = sorted(i for i in range(count) if i not in done and remaining[i] <= done)
            if not ready:
                raise ValueError("Dependency cycle detected among plans")
            levels.append(ready)
            done.update(ready)
        return levels

    def depth(self, plans: Sequence[Any], dependencies: Optional[Mapping[int, set[int]]] = None) -> int:
        """Number of dependency levels (critical-path length in levels)."""
        count = len(plans)
        if count == 0:
            return 0
        edges: dict[int, set[int]] = {i: set() for i in range(count)}
        if dependencies:
            for target, prereqs in dependencies.items():
                edges[target] = set(prereqs)
        return len(self._levels(count, edges))

    def batch_shots(self, plans: Sequence[Any], batch: ScheduledBatch) -> int:
        """Total shots in *batch* (``None`` shots count as 0)."""
        total = 0
        for index in batch.indices:
            shots = getattr(plans[index], "shots", None)
            if isinstance(shots, int):
                total += shots
        return total

    def to_dict(self, batches: Sequence[ScheduledBatch]) -> dict[str, Any]:
        """Serialize batches to a JSON-safe dictionary."""
        return {
            "max_parallel": self._max_parallel,
            "batches": [
                {"indices": list(batch.indices), "level": batch.level} for batch in batches
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> tuple[DAGScheduler, list[ScheduledBatch]]:
        """Rebuild a scheduler plus batches from :meth:`to_dict` output."""
        scheduler = cls(max_parallel=int(data.get("max_parallel", 4)))
        batches = [
            ScheduledBatch(indices=tuple(int(i) for i in item["indices"]), level=int(item["level"]))
            for item in data.get("batches", [])
        ]
        return scheduler, batches

    def __repr__(self) -> str:
        return f"DAGScheduler(max_parallel={self._max_parallel})"


def _level_of(batches: Sequence[ScheduledBatch], index: int) -> int:
    """Level containing *index* (helper for trace enrichment)."""
    for batch in batches:
        if index in batch.indices:
            return batch.level
    raise ValueError(f"Plan index {index} not present in any batch")
