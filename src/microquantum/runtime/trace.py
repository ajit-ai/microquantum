"""Lightweight execution trace for the runtime.

An :class:`ExecutionTrace` records the ordered lifecycle of a single
execution (``prepared -> submitted -> completed ...``) together with per-step
timing and free-form data.  Traces are cheap by design: they are a
foundation for observability and are not intended to replace profiling or
logging tools.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .._json import JSONSerializable, json_string


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceEvent:
    """A single ordered trace step.

    Attributes:
        name: Short event name, e.g. ``"prepare"`` or ``"submitted"``.
        timestamp: ISO-8601 wall-clock timestamp of the event.
        elapsed: Seconds since the trace started (monotonic).
        data: Free-form JSON-safe payload for the event.
    """

    name: str
    timestamp: str
    elapsed: float
    data: dict[str, Any] = field(default_factory=dict)


class ExecutionTrace(JSONSerializable):
    """Ordered record of execution lifecycle events."""

    def __init__(self, name: str = "execution") -> None:
        self._name = name
        self._started: float = time.monotonic()
        self._created_at: str = _now_iso()
        self._events: list[TraceEvent] = []

    @property
    def name(self) -> str:
        """Trace identifier."""
        return self._name

    @property
    def events(self) -> list[TraceEvent]:
        """Recorded events in submission order."""
        return list(self._events)

    @property
    def started(self) -> str:
        """ISO-8601 wall-clock time the trace was created."""
        return self._created_at

    def record(self, name: str, **data: Any) -> TraceEvent:
        """Record an event with free-form keyword data."""
        event = TraceEvent(
            name=name,
            timestamp=_now_iso(),
            elapsed=time.monotonic() - self._started,
            data=data,
        )
        self._events.append(event)
        return event

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time since the trace started (seconds)."""
        return time.monotonic() - self._started

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to a JSON-safe dictionary."""
        return {
            "name": self._name,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp,
                    "elapsed_seconds": round(e.elapsed, 6),
                    "data": e.data,
                }
                for e in self._events
            ],
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }

    def to_json(self) -> str:
        """Serialize the trace to a pretty-printed JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return f"ExecutionTrace('{self._name}', {len(self._events)} events)"