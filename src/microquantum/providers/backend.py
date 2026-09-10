"""Hardware backend adapter conforming to the simulation Backend API.

Bridges :class:`~microquantum.backends.base.Backend` so the existing
:class:`~microquantum.backends.executor.Executor` and platform layers can
target real hardware through any :class:`HardwareProvider`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..backends.base import Backend, BackendResult, Job, JobStatus
from ..core.circuit import QuantumCircuit
from ..core.state import StateVector
from .base import HardwareProvider


class HardwareBackend(Backend):
    """Backend that executes circuits on real quantum hardware.

    Unlike simulation backends, :meth:`submit` returns a job that stays
    ``PENDING`` until :meth:`execute` is called (hardware is asynchronous).
    The provider polls the vendor until completion.
    """

    def __init__(
        self,
        provider: HardwareProvider,
        shots: int = 1024,
    ) -> None:
        self._provider = provider
        self._shots = shots
        self._circuits: dict[str, QuantumCircuit] = {}

    @property
    def name(self) -> str:
        return f"{self._provider.name}_hardware"

    @property
    def provider(self) -> HardwareProvider:
        """Access the wrapped hardware provider."""
        return self._provider

    def submit(
        self,
        num_qubits: int,
        gates: list[tuple[NDArray[np.complex128], list[int]]],
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> Job:
        """Submit a circuit for execution.

        Note:
            Hardware submission requires a circuit (gate names), not raw
            matrices. Call :meth:`run_circuit` with a circuit, or use
            :meth:`submit_circuit` directly.

        Returns:
            A job marked PENDING (hardware is asynchronous).
        """
        circuit = QuantumCircuit(num_qubits)
        for matrix, targets in gates:
            from ..core.operators import Operator

            circuit.append(Operator(np.asarray(matrix, dtype=np.complex128)), targets)
        return self.submit_circuit(circuit, shots=shots)

    def submit_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Job:
        """Submit a real circuit to hardware and return an async job.

        Args:
            circuit: Fully-bound circuit to execute.
            shots: Number of measurement shots.

        Returns:
            A :class:`Job` that will be completed once the provider returns.
        """
        job = Job()
        job.status = JobStatus.RUNNING
        hw_job = self._provider.submit(circuit, shots=shots)
        self._circuits[hw_job.job_id] = circuit

        def _collect() -> None:
            try:
                hw_job.wait_for_result()
                data = hw_job._result if hw_job._result is not None else self._provider.result(hw_job.job_id)
                counts = data.get("counts") or {}
                backend_result = BackendResult(
                    num_qubits=circuit.num_qubits,
                    backend_name=self.name,
                    counts=counts,
                    metadata={
                        "provider": self._provider.name,
                        "job_id": hw_job.job_id,
                        "shots": shots,
                    },
                )
                job.result = backend_result
                job.status = JobStatus.COMPLETED
            except Exception as exc:
                job.error = str(exc)
                job.status = JobStatus.FAILED

        _collect()
        return job

    def run_circuit(
        self,
        num_qubits: int,
        gates: list[tuple[NDArray[np.complex128], list[int]]],
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Execute a circuit on hardware and block for the result."""
        raise RuntimeError(
            "HardwareBackend cannot execute raw gate matrices. "
            "Build a QuantumCircuit and call run() with a bound circuit "
            "instead (e.g. via quantumcircuit.run())."
        )

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Execute a bound circuit on hardware and block for the result.

        Args:
            circuit: Fully-bound quantum circuit.
            shots: Number of measurement shots.
            initial_state: Not applicable on hardware hardware; ignored.
            seed: Not applicable on real hardware; ignored.

        Returns:
            BackendResult with hardware measurement counts.
        """
        circuit._ensure_bound()
        hw_job = self._provider.submit(circuit, shots=shots)
        data = hw_job.wait_for_result()
        counts = data.get("counts") or {}
        return BackendResult(
            num_qubits=circuit.num_qubits,
            backend_name=self.name,
            counts=counts,
            metadata={
                "provider": self._provider.name,
                "job_id": hw_job.job_id,
                "shots": shots,
            },
        )

    def __repr__(self) -> str:
        return f"HardwareBackend(provider='{self._provider.name}', shots={self._shots})"