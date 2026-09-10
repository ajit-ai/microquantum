"""IBM Quantum provider (raw REST API).

Talks to the IBM Quantum HTTP API directly with only the standard library.
The circuits are serialized with :class:`CircuitSerializer` into IBM's
JSON instruction format (no OpenQASM text, no Qiskit).

Supports both IBM Quantum (``ibm_quantum``) channel directly and the
IBM Cloud Runtime-style flows. Given that IBM's API is authentication
heavy (the legacy quantum channel uses an IBMid login token), this
client works with an already-issued access token and exposes a clean
submit/status/result contract that is fully testable with a mocked
HTTP transport.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ..core.circuit import QuantumCircuit
from .base import HardwareJob, HardwareProvider, HardwareStatus, ProviderCredentials
from .http import http_request
from .serializer import CircuitSerializer

_DEFAULT_BASE_URL = "https://api.quantum-computing.ibm.com/api"


@dataclass
class IBMQuantumCredentials(ProviderCredentials):
    """IBM Quantum credentials from the ``IBM_QUANTUM_TOKEN`` env var.

    Attributes:
        channel: IBM-specific channel ("ibm_quantum" or "ibm_cloud").
    """

    channel: str = "ibm_quantum"

    @classmethod
    def from_env(cls) -> "IBMQuantumCredentials":
        token = os.environ.get("IBM_QUANTUM_TOKEN", "")
        channel = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum")
        base_url = os.environ.get("IBM_QUANTUM_BASE_URL")
        return cls(api_token=token, channel=channel, base_url=base_url)


class IBMQuantumProvider(HardwareProvider):
    """Provider for IBM Quantum cloud hardware.

    Args:
        credentials: IBM token credentials.
        backend: Target backend name (e.g. "ibm_brisbane").
        transport: Optional callable overriding :func:`http_request`
            (used by tests).
    """

    def __init__(
        self,
        credentials: Optional[ProviderCredentials] = None,
        backend: str = "ibm_brisbane",
        transport: Any = None,
    ) -> None:
        super().__init__(credentials or IBMQuantumCredentials.from_env())
        self._backend = backend
        self._transport = transport or http_request
        self._base_url = self._credentials.base_url or _DEFAULT_BASE_URL
        self._shots_by_job: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "ibm"

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
        """Submit a circuit to the selected IBM backend."""
        ibm_json = CircuitSerializer.to_ibm(circuit)
        body = {
            "backend": self._backend,
            "shots": int(shots),
            "circuits": [ibm_json],
            "qobj_type": "run",
        }
        data = self._request("POST", "/jobs", body)
        job_id = str(data.get("id", ""))
        if not job_id:
            raise RuntimeError(f"IBM submit did not return a job id: {data}")
        self._shots_by_job[job_id] = int(shots)
        return HardwareJob(self, job_id, HardwareStatus.QUEUED)

    def status(self, job_id: str) -> HardwareStatus:
        """Poll job status from IBM."""
        data = self._request("GET", f"/jobs/{job_id}")
        status_map = {
            "queued": HardwareStatus.QUEUED,
            "validating": HardwareStatus.QUEUED,
            "running": HardwareStatus.RUNNING,
            "completed": HardwareStatus.COMPLETED,
            "failed": HardwareStatus.FAILED,
            "cancelled": HardwareStatus.CANCELLED,
        }
        return status_map.get(
            str(data.get("status", data.get("state", ""))).lower(),
            HardwareStatus.PENDING,
        )

    def result(self, job_id: str) -> dict[str, Any]:
        """Fetch completed job results (bitstring counts)."""
        data = self._request("GET", f"/jobs/{job_id}/results")
        counts: dict[str, int] = {}
        probabilities: dict[str, float] = {}
        num_qubits = int(data.get("num_qubits", 0))

        raw = data.get("counts") or {}
        for key, count in raw.items():
            # IBM may return hex keys like "0x0" or "0 1"; normalize.
            bits = self._normalize_key(key, num_qubits)
            counts[bits] = counts.get(bits, 0) + int(count)
        total = sum(counts.values())
        if total > 0:
            probabilities = {k: v / total for k, v in counts.items()}

        return {
            "counts": counts,
            "probabilities": probabilities,
            "num_qubits": num_qubits,
            "shots": sum(counts.values()),
        }

    @staticmethod
    def _normalize_key(key: str, num_qubits: int) -> str:
        """Normalize IBM hex / spaced bitstring keys to a bitstring."""
        clean = key.replace(" ", "")
        if clean.lower().startswith("0x"):
            value = int(clean, 16)
            return format(value, f"0{num_qubits}b")
        if all(c in "01" for c in clean):
            return clean
        raise ValueError(f"Cannot normalize IBM counts key: {key!r}")

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued/running job."""
        data = self._request("POST", f"/jobs/{job_id}/cancel", body={})
        return str(data.get("status", "")).lower() in ("cancelled", "canceling")

    def list_devices(self) -> list[dict[str, Any]]:
        """List available IBM Quantum backends."""
        data = self._request("GET", "/Backends")
        backends = data or {}
        out: list[dict[str, Any]] = []
        for backend_name in backends:
            info = backends[backend_name]
            out.append(
                {
                    "name": backend_name,
                    "status": str(info.get("status", info.get("state", "unknown"))),
                    "num_qubits": int(info.get("n_qubits", info.get("num_qubits", 0))),
                }
            )
        return out
