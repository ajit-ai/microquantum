"""Shared polling jobs and error mapping for hardware providers."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from .base import HardwareJob, HardwareProvider, HardwareStatus
from .http import HttpError

__all__ = [
    "ProviderErrorMapper",
    "PollingJob",
    "TERMINAL_STATUSES",
]

TERMINAL_STATUSES: frozenset[HardwareStatus] = frozenset(
    {
        HardwareStatus.COMPLETED,
        HardwareStatus.FAILED,
        HardwareStatus.CANCELLED,
    }
)

_DEFAULT_TABLE: dict[str, HardwareStatus] = {
    "completed": HardwareStatus.COMPLETED,
    "complete": HardwareStatus.COMPLETED,
    "done": HardwareStatus.COMPLETED,
    "success": HardwareStatus.COMPLETED,
    "succeeded": HardwareStatus.COMPLETED,
    "failed": HardwareStatus.FAILED,
    "failure": HardwareStatus.FAILED,
    "error": HardwareStatus.FAILED,
    "cancelled": HardwareStatus.CANCELLED,
    "canceled": HardwareStatus.CANCELLED,
    "running": HardwareStatus.RUNNING,
    "executing": HardwareStatus.RUNNING,
    "queued": HardwareStatus.QUEUED,
    "pending": HardwareStatus.QUEUED,
    "waiting": HardwareStatus.QUEUED,
    "ready": HardwareStatus.QUEUED,
    "submitted": HardwareStatus.QUEUED,
}


class ProviderErrorMapper:
    """Map vendor HTTP/payload responses to SDK statuses and errors.

    Args:
        status_field: Payload key holding the vendor status string.
        table: Extra ``{lowercased string: HardwareStatus}`` entries
            merged over the default table (vendor spellings).
    """

    def __init__(
        self,
        status_field: str = "status",
        table: Optional[Mapping[str, HardwareStatus]] = None,
    ) -> None:
        if not status_field:
            raise ValueError("status_field must be non-empty")
        merged = dict(_DEFAULT_TABLE)
        for key, status in (table or {}).items():
            merged[str(key).lower()] = status
        self._status_field = status_field
        self._table = merged

    @property
    def status_field(self) -> str:
        """Payload key holding the vendor status string."""
        return self._status_field

    def to_status(self, payload: Mapping[str, Any]) -> HardwareStatus:
        """Map a vendor payload to :class:`HardwareStatus`.

        Raises:
            ValueError: If the payload carries no recognizable status.
        """
        raw = payload.get(self._status_field)
        if raw is None:
            raise ValueError(f"Payload has no '{self._status_field}' field: {dict(payload)}")
        key = str(raw).lower()
        if key not in self._table:
            raise ValueError(f"Unknown vendor status '{raw}'")
        return self._table[key]

    def check(
        self, http_code: int, payload: Mapping[str, Any], *, operation: str = "request"
    ) -> None:
        """Raise :class:`HttpError` for failing HTTP responses."""
        if http_code < 400:
            return
        detail = payload.get("error", payload.get("message", f"HTTP {http_code}"))
        raise HttpError(f"Provider {operation} failed ({http_code}): {detail}", http_code)


class PollingJob(HardwareJob):
    """Hardware job with bounded polling shared by all providers.

    Attributes:
        poll_interval: Seconds between status polls.
        max_polls: Maximum polls per :meth:`wait` call.
    """

    poll_interval: float = 5.0
    max_polls: int = 100

    def __init__(
        self,
        provider: HardwareProvider,
        job_id: str,
        status: HardwareStatus = HardwareStatus.PENDING,
        *,
        poll_interval: float = 5.0,
        max_polls: int = 100,
    ) -> None:
        super().__init__(provider, job_id, status)
        if poll_interval < 0:
            raise ValueError("poll_interval must be >= 0")
        if max_polls < 1:
            raise ValueError("max_polls must be >= 1")
        self.poll_interval = poll_interval
        self.max_polls = max_polls
        self._polls = 0

    def poll(self) -> HardwareStatus:
        """Refresh once via the provider and return the status."""
        self._polls += 1
        return self.refresh()

    def wait(self, timeout: float = 120.0) -> dict[str, Any]:
        """Wait (bounded by polls and *timeout*) and return the result.

        Raises:
            TimeoutError: If limits are hit before a terminal status.
            RuntimeError: If the job failed or was cancelled.
        """
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        deadline = time.monotonic() + timeout
        polls = 0
        while polls < self.max_polls:
            if self.status == HardwareStatus.COMPLETED:
                return self._provider.result(self.job_id)
            if self.status in (HardwareStatus.FAILED, HardwareStatus.CANCELLED):
                raise RuntimeError(
                    f"Hardware job {self.job_id} ended with status {self.status.value}"
                )
            self.poll()
            polls += 1
            if self.status in TERMINAL_STATUSES:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self.poll_interval, remaining))
        if self.status == HardwareStatus.COMPLETED:
            return self._provider.result(self.job_id)
        raise TimeoutError(
            f"Hardware job {self.job_id} did not finish within {timeout}s "
            f"({polls} polls, status={self.status.value})"
        )

    @classmethod
    def from_job(
        cls,
        job: HardwareJob,
        provider: Optional[HardwareProvider] = None,
    ) -> PollingJob:
        """Wrap a plain :class:`HardwareJob` with polling behavior."""
        resolved = provider if provider is not None else job._provider
        wrapped = cls(provider=resolved, job_id=job.job_id, status=job.status)
        wrapped.error = job.error
        return wrapped
