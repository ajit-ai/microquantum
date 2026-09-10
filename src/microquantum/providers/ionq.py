"""IonQ cloud hardware provider (raw REST API v0.3).

Talks to https://api.ionq.co/v0.3 directly with only the standard library.
Native gate JSON is produced by :class:`CircuitSerializer`.

Reference API docs (public, no auth required to read):
    https://docs.ionq.com/api-reference
"""
from __future__ import annotations

import os
from typing import Any, Optional

from ..core.circuit import QuantumCircuit
from .base import HardwareJob, HardwareProvider, HardwareStatus, ProviderCredentials
from .http import http_request
from .serializer import CircuitSerializer

_DEFAULT_BASE_URL = "https://api.ionq.co/v0.3"


class IonQCredentials(ProviderCredentials):
    """Credentials for IonQ, sourced from the ``IONQ_API_TOKEN`` env var."""

    @classmethod
    def from_env(cls) -> "IonQCredentials":
        token = os.environ.get("IONQ_API_TOKEN", "")
        base_url = os.environ.get("IONQ_BASE_URL")
        return cls(api_token=token, base_url=base_url)


class IonQProvider(HardwareProvider):
    """Provider for IonQ trapped-ion quantum computers.

    Args:
        credentials: IonQ API credentials.
        target: Device to target (e.g. "qpu.aria-1", "simulator").
        transport: Optional callable overridding :func:`http_request`
            (used by tests).
    """

    def __init__(
        self,
        credentials: Optional[ProviderCredentials] = None,
        target: str = "qpu.aria-1",
        transport: Any = None,
    ) -> None:
        super().__init__(credentials or IonQCredentials.from_env())
        self._target = target
        self._transport = transport or http_request
        self._base_url = self._credentials.base_url or _DEFAULT_BASE_URL
        self._shots_by_job: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "ionq"

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._credentials.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = None
        if body is not None:
            import json

            payload = json.dumps(body).encode("utf-8")
        _, data = self._transport(method, url, headers, payload)
        return data

    def submit(self, circuit: QuantumCircuit, shots: int = 1024) -> HardwareJob:
        """Submit a circuit for execution on the selected IonQ device."""
        ionq_json = CircuitSerializer.to_ionq(circuit)
        body = {
            "target": self._target,
            "shots": int(shots),
            "input": {
                "format": "ionq.circuit.v0",
                **ionq_json,
            },
        }
        data = self._request("POST", "/jobs", body)
        job_id = str(data.get("id", ""))
        if not job_id:
            raise RuntimeError(f"IonQ submit did not return a job id: {data}")
        self._shots_by_job[job_id] = int(shots)
        return HardwareJob(self, job_id, HardwareStatus.QUEUED)

    def status(self, job_id: str) -> HardwareStatus:
        """Poll job status from IonQ."""
        data = self._request("GET", f"/jobs/{job_id}")
        status_map = {
            "ready": HardwareStatus.QUEUED,
            "submitted": HardwareStatus.QUEUED,
            "running": HardwareStatus.RUNNING,
            "completed": HardwareStatus.COMPLETED,
            "failed": HardwareStatus.FAILED,
            "cancelled": HardwareStatus.CANCELLED,
        }
        return status_map.get(
            str(data.get("status", "")).lower(), HardwareStatus.PENDING
        )

    def result(self, job_id: str) -> dict[str, Any]:
        """Fetch completed job results (bitstring counts)."""
        data = self._request("GET", f"/jobs/{job_id}/results")
        histogram = data.get("histogram") or {}
        num_qubits = int(data.get("num_qubits", data.get("qubits", 0)))
        shots = self._shots_by_job.get(job_id, 1024)
        counts: dict[str, int] = {}
        import numpy as np

        probabilities: dict[str, float] = {}
        for index, prob in histogram.items():
            idx = int(index)
            bits = format(idx, f"0{num_qubits}b")
            probabilities[bits] = float(prob)
            counts[bits] = int(round(float(prob) * shots))
        return {
            "counts": counts,
            "probabilities": probabilities,
            "num_qubits": num_qubits,
            "shots": shots,
        }

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job via DELETE."""
        data = self._request("DELETE", f"/jobs/{job_id}")
        return str(data.get("status", "")).lower() == "cancelled"

    def list_devices(self) -> list[dict[str, Any]]:
        """List IonQ devices and their status."""
        data = self._request("GET", "/devices")
        devices = data.get("devices", [])
        if not devices:
            return []
        out = []
        for name, info in devices.items():
            out.append({
                "name": name,
                "status": info.get("status") or info.get("charactersistics", {}).get("status", "unknown"),
                "num_qubits": info.get("qubits", info.get("charactersistics", {}).get("qubits", 0)),
            })
        return out


# Remove the fragile module-level shot tracking: shots are stored per job id
# on the provider instance in ``self._shots_by_job``.