"""Tests for Backend ABC, Job, and BackendResult."""

import pytest

from microquantum.backends.base import Backend, BackendResult, Job, JobStatus


class TestJob:
    def test_job_creation(self) -> None:
        job = Job()
        assert job.status == JobStatus.PENDING
        assert job.result is None
        assert job.error is None
        assert len(job.job_id) == 8

    def test_job_repr(self) -> None:
        job = Job()
        assert "pending" in repr(job)

    def test_job_status_values(self) -> None:
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"


class TestBackendResult:
    def test_result_creation(self) -> None:
        result = BackendResult(num_qubits=2, backend_name="test")
        assert result.num_qubits == 2
        assert result.backend_name == "test"
        assert result.counts == {}
        assert result.statevector is None
        assert result.density_matrix is None

    def test_result_probabilities(self) -> None:
        result = BackendResult(
            num_qubits=1,
            backend_name="test",
            counts={"0": 750, "1": 250},
        )
        probs = result.probabilities
        assert abs(probs["0"] - 0.75) < 1e-9
        assert abs(probs["1"] - 0.25) < 1e-9

    def test_result_probabilities_empty(self) -> None:
        result = BackendResult(num_qubits=1, backend_name="test")
        assert result.probabilities == {}

    def test_result_most_frequent(self) -> None:
        result = BackendResult(
            num_qubits=2,
            backend_name="test",
            counts={"00": 100, "01": 500, "10": 300, "11": 100},
        )
        assert result.most_frequent() == "01"

    def test_result_most_frequent_empty_raises(self) -> None:
        result = BackendResult(num_qubits=1, backend_name="test")
        with pytest.raises(ValueError, match="No measurement"):
            result.most_frequent()

    def test_result_repr(self) -> None:
        result = BackendResult(
            num_qubits=2, backend_name="test", counts={"00": 500, "11": 500}
        )
        r = repr(result)
        assert "2" in r
        assert "test" in r

    def test_result_str(self) -> None:
        result = BackendResult(
            num_qubits=2, backend_name="test", counts={"00": 500, "11": 500}
        )
        s = str(result)
        assert "test" in s
        assert "00" in s
        assert "11" in s


class TestBackend:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            Backend()

    def test_submit_completed(self) -> None:
        class DummyBackend(Backend):
            @property
            def name(self) -> str:
                return "dummy"

            def run_circuit(self, num_qubits, gates, shots=1024, initial_state=None, seed=None):
                counts = {"0" * num_qubits: shots}
                return BackendResult(
                    num_qubits=num_qubits, backend_name=self.name, counts=counts
                )

        backend = DummyBackend()
        job = backend.submit(num_qubits=1, gates=[], shots=100)
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert "0" in job.result.counts

    def test_submit_failed(self) -> None:
        class FailingBackend(Backend):
            @property
            def name(self) -> str:
                return "failing"

            def run_circuit(self, num_qubits, gates, shots=1024, initial_state=None, seed=None):
                raise RuntimeError("boom")

        backend = FailingBackend()
        job = backend.submit(num_qubits=1, gates=[])
        assert job.status == JobStatus.FAILED
        assert job.error == "boom"

    def test_backend_repr(self) -> None:
        class DummyBackend(Backend):
            @property
            def name(self) -> str:
                return "dummy"

            def run_circuit(self, num_qubits, gates, shots=1024, initial_state=None, seed=None):
                return BackendResult(num_qubits=num_qubits, backend_name=self.name)

        backend = DummyBackend()
        assert "DummyBackend" in repr(backend)
        assert "dummy" in repr(backend)
