"""Base abstractions for real quantum hardware providers.

Provides the :class:`HardwareProvider` interface contract, job and
credential models. Providers talk to vendor REST APIs directly using
only the standard library - no Qiskit/Cirq/OpenQASM anywhere.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from ..core.circuit import QuantumCircuit


class HardwareStatus(Enum):
    """Status of a submitted hardware job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HardwareJob:
    """A job submitted to a hardware provider.

    Tracks vendor job id and status, and lazily fetches results.
    """

    def __init__(
        self,
        provider: "HardwareProvider",
        job_id: str,
        status: HardwareStatus = HardwareStatus.PENDING,
    ) -> None:
        self._provider = provider
        self.job_id = job_id
        self.status = status
        self._result: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None

    def refresh(self) -> HardwareStatus:
        """Poll the provider for the latest job status."""
        self.status = self._provider.status(self.job_id)
        return self.status

    def wait_for_result(
        self,
        timeout: float = 120.0,
        poll_interval: float = 5.0,
    ) -> dict[str, Any]:
        """Block until the job completes and return its result.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between status polls.

        Returns:
            The vendor result dictionary.

        Raises:
            TimeoutError: If the job does not finish in time.
            RuntimeError: If the job failed or was cancelled.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.status == HardwareStatus.COMPLETED:
                if self._result is None:
                    self._result = self._provider.result(self.job_id)
                return self._result
            if self.status in (HardwareStatus.FAILED, HardwareStatus.CANCELLED):
                raise RuntimeError(
                    f"Hardware job {self.job_id} ended with status {self.status.value}"
                )
            self.refresh()
            time.sleep(poll_interval)
        raise TimeoutError(f"Hardware job {self.job_id} did not complete within {timeout}s")

    def __repr__(self) -> str:
        return f"HardwareJob(id={self.job_id}, status={self.status.value})"


@dataclass
class ProviderCredentials:
    """Credentials for a real hardware provider.

    Provider-specific subclasses add extra fields (e.g. an IBM ``channel``,
    a region, or instance id) — vendor concepts never leak into the generic
    core API.

    Attributes:
        api_token: Secret API token / key.
        base_url: Optional API base URL override (for mirrors/e2e).
    """

    api_token: str
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ProviderCredentials":
        """Build credentials from standard provider environment variables."""
        raise NotImplementedError

    def validate(self) -> None:
        """Ensure the token is present, else raise ValueError."""
        if not self.api_token:
            raise ValueError(
                "Provider API token is required. Set the appropriate "
                "environment variable before constructing a provider."
            )


class HardwareProvider(ABC):
    """Interface contract for submitting circuits to real quantum hardware.

    Implementations talk to vendor REST APIs using :func:`http_request`.
    """

    def __init__(self, credentials: ProviderCredentials) -> None:
        self._credentials = credentials
        self._credentials.validate()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'ibm', 'ionq')."""

    @property
    def credentials(self) -> ProviderCredentials:
        """Access the configured credentials."""
        return self._credentials

    @abstractmethod
    def submit(self, circuit: QuantumCircuit, shots: int = 1024) -> HardwareJob:
        """Submit a circuit for hardware execution."""

    @abstractmethod
    def status(self, job_id: str) -> HardwareStatus:
        """Return the current status of a submitted job."""

    @abstractmethod
    def result(self, job_id: str) -> dict[str, Any]:
        """Return the completed result dictionary for a job."""

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a job. Defaults to unsupported."""
        raise NotImplementedError("cancel() is not supported by this provider")

    @abstractmethod
    def list_devices(self) -> list[dict[str, Any]]:
        """Return information about available hardware devices."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider='{self.name}')"


HttpResponder = Callable[[str, str, dict[str, str], Optional[bytes]], tuple[int, dict[str, Any]]]
