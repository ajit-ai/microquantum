"""Tests for Phase 14: Real Hardware Integration.

Covers IBM Quantum and IonQ providers (with mocked HTTP transports), the
circuit serializer, the HardwareBackend adapter, and Executor integration.
No live credentials or network access are required.
"""
import numpy as np
import pytest

from microquantum import (
    CircuitSerializer,
    Executor,
    HardwareBackend,
    IBMQuantumCredentials,
    IBMQuantumProvider,
    IonQCredentials,
    IonQProvider,
    QuantumCircuit,
    UnsupportedGateError,
)
from microquantum.core.operators import Operator
from microquantum.providers.base import HardwareStatus


def bell_circuit() -> QuantumCircuit:
    """H + CNOT produces a Bell state."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def rotation_circuit() -> QuantumCircuit:
    """RY/RZ rotations + CZ."""
    qc = QuantumCircuit(3)
    qc.ry(np.pi / 3, 0)
    qc.ry(np.pi / 4, 1)
    qc.rz(np.pi / 2, 2)
    qc.cz(0, 1)
    return qc


# =============================================================================
# Circuit Serializer Tests
# =============================================================================
class TestCircuitSerializer:
    """Test serializer to provider-native JSON."""

    def test_ionq_bell(self) -> None:
        """Bell circuit serializes to IonQ JSON."""
        payload = CircuitSerializer.to_ionq(bell_circuit())
        assert payload["qubits"] == 2
        assert payload["circuit"][0] == {"gate": "h", "target": 0}
        assert payload["circuit"][1] == {"gate": "cnot", "control": 0, "target": 1}

    def test_ibm_bell(self) -> None:
        """Bell circuit serializes to IBM JSON."""
        payload = CircuitSerializer.to_ibm(bell_circuit())
        assert payload["num_qubits"] == 2
        assert payload["instructions"][0] == {"name": "h", "qubits": [0]}
        assert payload["instructions"][1] == {"name": "cnot", "qubits": [0, 1]}

    def test_rotations_roundtrip_angles(self) -> None:
        """RY/RZ angles are extracted and emitted for hardware."""
        data = CircuitSerializer.to_ionq(rotation_circuit())
        rotations = [g for g in data["circuit"] if g["gate"] in ("rx", "ry", "rz")]
        ry_angles = sorted(round(g["rotation"], 6) for g in rotations if g["gate"] == "ry")
        assert ry_angles == [round(np.pi / 4, 6), round(np.pi / 3, 6)]
        rz = [g for g in rotations if g["gate"] == "rz"][0]
        assert abs(rz["rotation"] - np.pi / 2) < 1e-9

    def test_custom_gate_rejected(self) -> None:
        """Non-native gates raise UnsupportedGateError."""
        qc = QuantumCircuit(1)
        qc.append(Operator(np.array([[0.5, -0.5], [0.5, 0.5]], dtype=complex), name="custom"), [0])
        with pytest.raises(UnsupportedGateError):
            CircuitSerializer.to_ionq(qc)

    def test_unbound_parameters_rejected(self) -> None:
        """Serializing an unbound circuit raises ValueError."""
        from microquantum import Parameter

        qc = QuantumCircuit(1)
        qc.ry(Parameter("theta"), 0)
        with pytest.raises(ValueError):
            CircuitSerializer.to_ionq(qc)

    def test_serialize_neutral(self) -> None:
        """Neutral serializer returns num_qubits + instructions."""
        data = CircuitSerializer.serialize(bell_circuit())
        assert data["num_qubits"] == 2
        assert len(data["instructions"]) == 2


# =============================================================================
# IonQ Provider Tests
# =============================================================================
class TestIonQProvider:
    """Test IonQ provider against a mocked HTTP transport."""

    @pytest.fixture()
    def provider(self):  # type: ignore[no-untyped-def]
        calls = []

        def fake_ionq(method: str, url: str, headers: dict, body) -> tuple[int, dict]:  # type: ignore[no-untyped-def]
            calls.append((method, url))
            if method == "POST" and url.endswith("/jobs"):
                return 201, {"id": "ionq-job-1"}
            if method == "GET" and url.endswith("/jobs/ionq-job-1"):
                return 200, {"id": "ionq-job-1", "status": "completed"}
            if method == "GET" and url.endswith("/jobs/ionq-job-1/results"):
                return 200, {"num_qubits": 2, "histogram": {"0": 0.25, "3": 0.75}}
            if method == "GET" and url.endswith("/devices"):
                return 200, {"devices": {"qpu.aria-1": {"status": "available", "qubits": 25}}}
            if method == "DELETE" and url.endswith("/jobs/ionq-job-1"):
                return 200, {"id": "ionq-job-1", "status": "cancelled"}
            return 404, {}

        prov = IonQProvider(
            IonQCredentials(api_token="test-token", base_url="https://mock.ionq/v0.3"),
            transport=fake_ionq,
        )
        return prov, calls

    def test_submit_returns_job(self, provider) -> None:  # type: ignore[no-untyped-def]
        """submit() returns a HardwareJob with an id."""
        prov, _ = provider
        job = prov.submit(bell_circuit(), shots=1000)
        assert job.job_id == "ionq-job-1"
        assert job.status in (HardwareStatus.QUEUED, HardwareStatus.PENDING)

    def test_status_poll(self, provider) -> None:  # type: ignore[no-untyped-def]
        """status() maps vendor status to HardwareStatus."""
        prov, _ = provider
        assert prov.status("ionq-job-1") == HardwareStatus.COMPLETED

    def test_result_counts(self, provider) -> None:  # type: ignore[no-untyped-def]
        """result() converts histogram to bitstring counts/probabilities."""
        prov, _ = provider
        prov.submit(bell_circuit(), shots=1000)
        data = prov.result("ionq-job-1")
        assert data["counts"] == {"00": 250, "11": 750}
        assert data["probabilities"]["11"] == 0.75

    def test_wait_for_result(self, provider) -> None:  # type: ignore[no-untyped-def]
        """wait_for_result() blocks and fetches counts."""
        prov, _ = provider
        job = prov.submit(bell_circuit(), shots=1000)
        result = job.wait_for_result(timeout=10, poll_interval=0.001)
        assert result["counts"] == {"00": 250, "11": 750}

    def test_cancel(self, provider) -> None:  # type: ignore[no-untyped-def]
        """cancel() issues HTTP DELETE."""
        prov, calls = provider
        result = prov.cancel("ionq-job-1")
        assert any(m == "DELETE" for m, u in calls)
        assert result is True

    def test_list_devices(self, provider) -> None:  # type: ignore[no-untyped-def]
        """list_devices() returns device metadata."""
        prov, _ = provider
        devices = prov.list_devices()
        assert devices[0]["name"] == "qpu.aria-1"
        assert devices[0]["num_qubits"] == 25

    def test_missing_token_rejected(self) -> None:
        """Constructing with an empty token raises ValueError."""
        with pytest.raises(ValueError):
            IonQProvider(IonQCredentials(api_token=""))

    def test_http_error_raises(self) -> None:
        """Non-2xx responses raise an error from the transport."""
        from microquantum.providers.http import HttpError

        def failing(  # type: ignore[no-untyped-def]
            method: str, url: str, headers: dict, body
        ) -> tuple[int, dict]:
            raise HttpError("Provider API error (401)", 401)

        prov = IonQProvider(
            IonQCredentials(api_token="t", base_url="https://mock/v0.3"),
            transport=failing,
        )
        with pytest.raises(HttpError):
            prov.submit(bell_circuit())


# =============================================================================
# IBM Quantum Provider Tests
# =============================================================================
class TestIBMQuantumProvider:
    """Test IBM provider against a mocked HTTP transport."""

    @pytest.fixture()
    def provider(self):  # type: ignore[no-untyped-def]
        calls = []

        def fake_ibm(method: str, url: str, headers: dict, body) -> tuple[int, dict]:  # type: ignore[no-untyped-def]
            calls.append((method, url))
            if method == "POST" and url.endswith("/jobs"):
                return 200, {"id": "ibm-job-7"}
            if method == "GET" and url.endswith("/jobs/ibm-job-7"):
                return 200, {"id": "ibm-job-7", "status": "COMPLETED"}
            if method == "GET" and url.endswith("/jobs/ibm-job-7/results"):
                return 200, {"num_qubits": 2, "counts": {"0x0": 512, "0x3": 512}}
            if method == "GET" and url.endswith("/Backends"):
                return 200, {"ibm_brisbane": {"status": "available", "n_qubits": 127}}
            if method == "POST" and url.endswith("/jobs/ibm-job-7/cancel"):
                return 200, {"status": "cancelled"}
            return 404, {}

        prov = IBMQuantumProvider(
            IBMQuantumCredentials(
                api_token="ibm-token",
                channel="ibm_quantum",
                base_url="https://mock.ibm.com/api",
            ),
            backend="ibm_brisbane",
            transport=fake_ibm,
        )
        return prov, calls

    def test_submit(self, provider) -> None:  # type: ignore[no-untyped-def]
        """submit() sends IBM JSON and returns a job."""
        prov, calls = provider
        job = prov.submit(bell_circuit(), shots=1000)
        assert job.job_id == "ibm-job-7"
        assert any(m == "POST" and u.endswith("/jobs") for m, u in calls)

    def test_status_maps_to_enum(self, provider) -> None:  # type: ignore[no-untyped-def]
        """Uppercase vendor status maps to HardwareStatus.completed."""
        prov, _ = provider
        assert prov.status("ibm-job-7") == HardwareStatus.COMPLETED

    def test_hex_counts_normalized(self, provider) -> None:  # type: ignore[no-untyped-def]
        """IBM hex keys are normalized to bitstrings."""
        prov, _ = provider
        data = prov.result("ibm-job-7")
        assert data["counts"]["00"] == 512
        assert data["counts"]["11"] == 512
        assert abs(data["probabilities"]["00"] - 0.5) < 1e-9

    def test_cancel(self, provider) -> None:  # type: ignore[no-untyped-def]
        """cancel() marks the job cancelled."""
        prov, _ = provider
        assert prov.cancel("ibm-job-7") is True

    def test_list_devices(self, provider) -> None:  # type: ignore[no-untyped-def]
        """list_devices() returns IBM backend metadata."""
        prov, _ = provider
        devices = prov.list_devices()
        assert devices[0]["name"] == "ibm_brisbane"
        assert devices[0]["num_qubits"] == 127


# =============================================================================
# HardwareBackend Adapter Tests
# =============================================================================
class TestHardwareBackend:
    """Test the Backend-compatible hardware adapter."""

    def test_run_returns_backend_result(self) -> None:
        """HardwareBackend.run() returns BackendResult with counts."""
        def fake_ionq(method: str, url: str, headers: dict, body) -> tuple[int, dict]:  # type: ignore[no-untyped-def]
            if method == "POST" and url.endswith("/jobs"):
                return 201, {"id": "job-2"}
            if method == "GET" and url.endswith("/jobs/job-2"):
                return 200, {"status": "completed"}
            if method == "GET" and url.endswith("/results"):
                return 200, {"num_qubits": 2, "histogram": {"0": 0.5, "3": 0.5}}
            return 404, {}

        prov = IonQProvider(
            IonQCredentials(api_token="t", base_url="https://mock/v0.3"),
            transport=fake_ionq,
        )
        backend = HardwareBackend(prov)
        result = backend.run(bell_circuit(), shots=1000)
        assert result.backend_name == "ionq_hardware"
        assert result.counts == {"00": 500, "11": 500}
        assert result.most_frequent() in ("00", "11")

    def test_raw_gate_matrices_rejected(self) -> None:
        """run_circuit() raises because hardware needs named gates."""
        prov = IonQProvider(
            IonQCredentials(api_token="t", base_url="https://mock/v0.3"),
            transport=lambda *a: (200, {}),  # type: ignore[misc]
        )
        backend = HardwareBackend(prov)
        eye = np.eye(2, dtype=np.complex128)
        with pytest.raises(RuntimeError):
            backend.run_circuit(1, [(eye, [0])], shots=100)

    def test_executor_hardware_integration(self) -> None:
        """Executor.run() delegates to hardware when backend is hardware."""
        def fake_ionq(method: str, url: str, headers: dict, body) -> tuple[int, dict]:  # type: ignore[no-untyped-def]
            if method == "POST" and url.endswith("/jobs"):
                return 201, {"id": "job-3"}
            if method == "GET" and url.endswith("/jobs/job-3"):
                return 200, {"status": "completed"}
            if method == "GET" and url.endswith("/results"):
                return 200, {"num_qubits": 2, "histogram": {"0": 0.25, "3": 0.75}}
            return 404, {}

        prov = IonQProvider(
            IonQCredentials(api_token="t", base_url="https://mock/v0.3"),
            transport=fake_ionq,
        )
        executor = Executor(backend=HardwareBackend(prov))
        result = executor.run(bell_circuit(), shots=1000)
        assert result.metadata["backend"] == "ionq_hardware"
        assert result.probabilities.get("11") == 0.75


# =============================================================================
# Credentials Tests
# =============================================================================
class TestCredentials:
    """Test provider credential handling."""

    def test_ibm_from_env(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """IBM credentials read from environment variables."""
        monkeypatch.setenv("IBM_QUANTUM_TOKEN", "env-token")
        monkeypatch.setenv("IBM_QUANTUM_CHANNEL", "ibm_cloud")
        creds = IBMQuantumCredentials.from_env()
        assert creds.api_token == "env-token"
        assert creds.channel == "ibm_cloud"

    def test_ionq_from_env(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """IonQ credentials read from environment variables."""
        monkeypatch.setenv("IONQ_API_TOKEN", "ionq-env-token")
        creds = IonQCredentials.from_env()
        assert creds.api_token == "ionq-env-token"

    def test_validate_requires_token(self) -> None:
        """Empty tokens raise ValueError."""
        creds = IonQCredentials(api_token="")
        with pytest.raises(ValueError):
            creds.validate()


# =============================================================================
# Export Tests
# =============================================================================
class TestExports:
    """Test top-level package exports."""

    def test_provider_exports(self) -> None:
        """All Phase 14 classes exported at the package top level."""
        import microquantum

        for name in [
            "HardwareProvider",
            "HardwareJob",
            "HardwareStatus",
            "HardwareBackend",
            "IBMQuantumProvider",
            "IonQProvider",
            "CircuitSerializer",
            "UnsupportedGateError",
        ]:
            assert hasattr(microquantum, name), f"missing top-level export {name}"