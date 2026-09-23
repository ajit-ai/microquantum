"""Content-addressed result cache for execution plans.

Keys are SHA-256 fingerprints of the plan's JSON-safe dictionary, so
identical work (same circuit, bindings, shots, seed and options) maps
to one entry.  Only plans with ``cacheable=True`` (the default) are
stored or served.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ResultCache",
    "plan_fingerprint",
]


def plan_fingerprint(plan: Any) -> str:
    """SHA-256 fingerprint of a plan's JSON-safe dictionary."""
    to_dict = getattr(plan, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"Cannot fingerprint {type(plan).__name__}: no to_dict()")
    payload = to_dict()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ResultCache:
    """Bounded in-memory cache of plan fingerprints to records."""

    max_entries: int = 256
    _entries: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError("max_entries must be >= 1")

    def get(self, plan: Any) -> Any | None:
        """Return the cached record for *plan*, or ``None``."""
        if not bool(getattr(plan, "cacheable", True)):
            return None
        return self._entries.get(plan_fingerprint(plan))

    def put(self, plan: Any, record: Any) -> str:
        """Store *record* under *plan*'s fingerprint; return the key.

        Non-cacheable plans are ignored (returns their fingerprint
        without storing).
        """
        key = plan_fingerprint(plan)
        if not bool(getattr(plan, "cacheable", True)):
            return key
        if len(self._entries) >= self.max_entries:
            oldest = next(iter(self._entries))
            del self._entries[oldest]
        self._entries[key] = record
        return key

    def clear(self) -> None:
        """Drop all entries."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, plan: Any) -> bool:
        try:
            return plan_fingerprint(plan) in self._entries
        except TypeError:
            return False
