"""Phase 121 W5: experience and hardware."""

from __future__ import annotations

import pytest

from microquantum.analytics import ReportBuilder, Result, to_records
from microquantum.benchmarks import BenchmarkSuite, MirrorBenchmarking, SuiteReport
from microquantum.experiments import AdaptiveSweep, Checkpoint, ParameterSweep
from microquantum.providers import (
    IonQCredentials,
    PollingJob,
    ProviderCredentials,
    ProviderErrorMapper,
)
from microquantum.providers.base import HardwareJob, HardwareStatus
from microquantum.providers.http import HttpError


class _FakeProvider:
    """Minimal provider double for polling tests."""

    def __init__(self, statuses: list[HardwareStatus]) -> None:
        self._statuses = list(statuses)
        self.calls = 0

    def status(self, job_id: str) -> HardwareStatus:
        """Return the next queued status."""
        self.calls += 1
        assert job_id == "job-1"
        if len(self._statuses) > 1:
            return self._statuses.pop(0)
        return self._statuses[0]

    def result(self, job_id: str) -> dict[str, object]:
        """Return a canned result payload."""
        assert job_id == "job-1"
        return {"counts": {"00": 10}}


class TestPollingJob:
    def test_poll_and_wait(self) -> None:
        provider = _FakeProvider(
            [HardwareStatus.QUEUED, HardwareStatus.RUNNING, HardwareStatus.COMPLETED]
        )
        job = PollingJob(provider=provider, job_id="job-1", poll_interval=0.0)  # type: ignore[arg-type]
        assert job.poll() == HardwareStatus.QUEUED
        assert job.wait(timeout=5.0) == {"counts": {"00": 10}}
        assert provider.calls >= 3

    def test_wait_timeout_and_failure(self) -> None:
        stuck = PollingJob(
            provider=_FakeProvider([HardwareStatus.RUNNING]),  # type: ignore[arg-type]
            job_id="job-1",
            poll_interval=0.0,
            max_polls=2,
        )
        with pytest.raises(TimeoutError):
            stuck.wait(timeout=5.0)
        failed = PollingJob(
            provider=_FakeProvider([HardwareStatus.FAILED]),  # type: ignore[arg-type]
            job_id="job-1",
            poll_interval=0.0,
        )
        with pytest.raises(RuntimeError):
            failed.wait(timeout=5.0)
        with pytest.raises(ValueError):
            PollingJob(provider=_FakeProvider([]), job_id="x", max_polls=0)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            PollingJob(provider=_FakeProvider([]), job_id="x").wait(timeout=-1.0)  # type: ignore[arg-type]

    def test_from_job(self) -> None:
        provider = _FakeProvider([HardwareStatus.COMPLETED])
        plain = HardwareJob(provider=provider, job_id="job-1")  # type: ignore[arg-type]
        wrapped = PollingJob.from_job(plain)
        assert isinstance(wrapped, PollingJob)
        assert wrapped.job_id == "job-1"
        assert wrapped.wait(timeout=5.0) == {"counts": {"00": 10}}


class TestErrorMapper:
    def test_status_mapping(self) -> None:
        mapper = ProviderErrorMapper()
        assert mapper.to_status({"status": "COMPLETED"}) == HardwareStatus.COMPLETED
        assert mapper.to_status({"status": "running"}) == HardwareStatus.RUNNING
        assert mapper.to_status({"status": "queued"}) == HardwareStatus.QUEUED
        assert mapper.status_field == "status"
        custom = ProviderErrorMapper(status_field="state", table={"finished": HardwareStatus.COMPLETED})
        assert custom.to_status({"state": "finished"}) == HardwareStatus.COMPLETED
        with pytest.raises(ValueError):
            mapper.to_status({"status": "teleported"})
        with pytest.raises(ValueError):
            mapper.to_status({"other": 1})
        with pytest.raises(ValueError):
            ProviderErrorMapper(status_field="")

    def test_check_raises(self) -> None:
        mapper = ProviderErrorMapper()
        mapper.check(200, {})
        with pytest.raises(HttpError):
            mapper.check(500, {"error": "boom"}, operation="submit")


class TestCredentialsProxy:
    def test_proxy_validation(self) -> None:
        credentials = ProviderCredentials(api_token="token", proxy="http://proxy:8080")
        credentials.validate()
        with pytest.raises(ValueError):
            ProviderCredentials(api_token="token", proxy="socks5://x").validate()
        with pytest.raises(ValueError):
            ProviderCredentials(api_token="").validate()

    def test_providers_bind_proxy_transport(self) -> None:
        from microquantum.providers import IBMQuantumProvider, IonQProvider

        ibm = IBMQuantumProvider(
            credentials=ProviderCredentials(api_token="t", proxy="http://p:8080"),
            transport=lambda method, url, headers, body: (200, {}),
        )
        assert ibm.credentials.proxy == "http://p:8080"
        ionq = IonQProvider(
            credentials=IonQCredentials(api_token="t"),
            transport=lambda method, url, headers, body: (200, {}),
        )
        assert ionq.credentials.proxy is None


class TestReportBuilder:
    def test_records_and_markdown(self) -> None:
        result = Result(problem="demo", solution={"bits": "01"}, confidence=0.9)
        rows = to_records(result)
        assert {"field": "problem", "value": "demo"} in rows
        assert {"field": "solution.bits", "value": "01"} in rows
        report = (
            ReportBuilder(title="Demo")
            .add_section("Notes", "All good.")
            .add_result("Outcome", result)
            .add_table("Empty", [])
        )
        markdown = report.to_markdown()
        assert markdown.startswith("# Demo")
        assert "## Outcome" in markdown
        assert "| field | value |" in markdown
        assert len(report) == 3
        with pytest.raises(ValueError):
            ReportBuilder().add_section("", "body")
        with pytest.raises(ValueError):
            ReportBuilder().add_table("", [])


class TestMirrorBenchmarking:
    def test_mirror_circuits(self) -> None:
        benchmark = MirrorBenchmarking(num_qubits=2, depths=(1, 2), num_circuits=2, num_shots=64, seed=0)
        circuit = benchmark.mirror_circuit(2, seed=1)
        assert circuit.num_qubits == 2
        assert "qubits=2" in repr(benchmark)
        with pytest.raises(ValueError):
            MirrorBenchmarking(num_qubits=0)
        with pytest.raises(ValueError):
            MirrorBenchmarking(depths=())
        with pytest.raises(ValueError):
            benchmark.mirror_circuit(0)

    def test_polarization(self) -> None:
        assert MirrorBenchmarking.polarization({"00": 100}, 2) == pytest.approx(1.0)
        assert MirrorBenchmarking.polarization({"00": 25, "01": 25, "10": 25, "11": 25}, 2) == pytest.approx(0.0)
        with pytest.raises(ValueError):
            MirrorBenchmarking.polarization({}, 2)

    def test_run_returns_result(self) -> None:
        benchmark = MirrorBenchmarking(num_qubits=1, depths=(1,), num_circuits=2, num_shots=128, seed=0)
        result = benchmark.run()
        assert result.metric_name == "mirror_fidelity"
        assert 0.0 <= result.value <= 1.0
        assert result.num_qubits == 1


class TestBenchmarkSuite:
    def test_suite_run_and_report(self) -> None:
        mirror = MirrorBenchmarking(num_qubits=1, depths=(1,), num_circuits=1, num_shots=32, seed=0)
        suite = BenchmarkSuite(seed=7)
        suite.add("mirror", mirror)
        assert suite.names == ["mirror"]
        assert len(suite) == 1
        report = suite.run()
        assert isinstance(report, SuiteReport)
        assert report.succeeded == ["mirror"]
        assert report.failed == []
        assert "mirror_fidelity" in report.summary()
        rebuilt = SuiteReport.from_dict(report.to_dict())
        assert rebuilt.succeeded == ["mirror"]
        assert "mirror_fidelity" in rebuilt.to_json()

    def test_suite_records_failures(self) -> None:
        class _Broken:
            def run(self) -> object:
                raise RuntimeError("nope")

        suite = BenchmarkSuite()
        suite.add("broken", _Broken())  # type: ignore[arg-type]
        report = suite.run()
        assert report.failed == ["broken"]
        assert report.summary() == {}
        with pytest.raises(ValueError):
            suite.add("", _Broken())  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            suite.add("broken", _Broken())  # type: ignore[arg-type]


class TestCheckpoint:
    def test_checkpoint_lifecycle(self, tmp_path: object) -> None:
        from pathlib import Path

        checkpoint = Checkpoint("exp-1")
        assert not checkpoint.is_complete
        checkpoint.mark_done("plan-a")
        checkpoint.mark_done("plan-a")
        assert checkpoint.completed == ["plan-a"]
        assert checkpoint.pending(["plan-a", "plan-b"]) == ["plan-b"]
        path = Path(str(tmp_path)) / "nested" / "checkpoint.json"
        checkpoint.save(path)
        rebuilt = Checkpoint.load(path)
        assert rebuilt.completed == ["plan-a"]
        assert rebuilt.experiment_id == "exp-1"
        assert "completed=1" in repr(rebuilt)
        assert len(checkpoint) == 1
        with pytest.raises(ValueError):
            Checkpoint("")
        with pytest.raises(ValueError):
            Checkpoint.load(path.parent / "missing.json")

    def test_checkpoint_from_dict(self) -> None:
        checkpoint = Checkpoint.from_dict({"experiment_id": "e", "completed": ["a", "b"]})
        assert checkpoint.pending(["a", "b", "c"]) == ["c"]


class TestAdaptiveSweep:
    def test_refinement_narrows_grid(self) -> None:
        base = ParameterSweep({"theta": [0.0, 1.0, 2.0]})
        adaptive = AdaptiveSweep(base, refine_fraction=0.5, max_rounds=2)
        assert adaptive.rounds_completed == 0
        assert len(adaptive.combinations()) == 3
        refined = adaptive.refine(
            [({"theta": 0.0}, 1.0), ({"theta": 1.0}, 0.0), ({"theta": 2.0}, 0.5)]
        )
        assert adaptive.rounds_completed == 1
        assert len(refined.combinations()) == 3
        points = sorted(combination["theta"] for combination in refined.combinations())
        assert points[1] == pytest.approx(1.0)
        assert adaptive.to_dict()["rounds_completed"] == 1

    def test_refine_validation(self) -> None:
        adaptive = AdaptiveSweep(ParameterSweep({"theta": [0.0, 1.0]}), max_rounds=1)
        with pytest.raises(ValueError):
            adaptive.refine([])
        adaptive.refine([({"theta": 0.0}, 1.0)])
        with pytest.raises(ValueError):
            adaptive.refine([({"theta": 0.0}, 1.0)])
        with pytest.raises(ValueError):
            AdaptiveSweep(ParameterSweep({"theta": [0.0]}), refine_fraction=0.0)
        with pytest.raises(ValueError):
            AdaptiveSweep(ParameterSweep({"theta": [0.0]}), max_rounds=0)
