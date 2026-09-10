"""Public API & contracts regression tests.

Locks the stabilized public programming model:

    QuantumCircuit -> Backend.run / submit_circuit -> Job -> BackendResult
    Device / Target            -> Backend capabilities
    Algorithm / Optimizer      -> computational contracts
    Provider / Adapter         -> hardware & domain boundaries
"""

import dataclasses
import json

import pytest

import microquantum
from microquantum import (
    Algorithm,
    BackendResult,
    Device,
    DeviceType,
    DomainAdapter,
    HardwareProvider,
    Job,
    JobStatus,
    Optimizer,
    OptimizerResult,
    ProviderCredentials,
    QuantumCircuit,
    QuantumProblem,
    StatevectorBackend,
    Target,
)
from microquantum.backends.base import JobStatus as BareJobStatus
from microquantum.core import Target as CoreTarget
from microquantum.providers import IBMQuantumCredentials, IonQCredentials


class TestPublicImports:
    """All programming-model contracts are reachable from the top level."""

    @pytest.mark.parametrize(
        "name",
        [
            "QuantumCircuit",
            "Backend",
            "BackendResult",
            "Job",
            "Device",
            "DeviceType",
            "Target",
            "Algorithm",
            "Optimizer",
            "OptimizerResult",
            "HardwareProvider",
            "ProviderCredentials",
            "DomainAdapter",
        ],
    )
    def test_public_name(self, name: str) -> None:
        assert name in microquantum.__all__
        assert hasattr(microquantum, name)

    def test_device_contracts_in_core(self) -> None:
        assert {"Device", "DeviceType", "Target"} <= set(microquantum.core.__all__)
        assert CoreTarget is Target


class TestDevice:
    def test_construction_and_defaults(self) -> None:
        device = Device(name="ionq-aria")
        assert device.device_type == DeviceType.SIMULATOR
        assert device.max_qubits is None
        assert device.available is True
        assert device.metadata == {}

    def test_capabilities(self) -> None:
        device = Device(
            name="qpu-1",
            device_type=DeviceType.QPU,
            max_qubits=32,
            available=False,
            metadata={"vendor": "acme"},
        )
        assert device.device_type.value == "qpu"
        assert device.max_qubits == 32
        assert device.available is False

    def test_device_types(self) -> None:
        values = {t.value for t in DeviceType}
        assert {"simulator", "cpu", "gpu", "npu", "accelerator", "qpu"} <= values

    def test_serialization_json_safe(self) -> None:
        raw = Device(name="gpu-0", device_type=DeviceType.GPU, max_qubits=64)
        encoded = raw.to_json()
        assert json.loads(encoded)["device_type"] == "gpu"
        assert raw.to_dict()["max_qubits"] == 64

    def test_str(self) -> None:
        assert "qpu-1" in str(Device(name="qpu-1", device_type=DeviceType.QPU))


class TestTarget:
    def test_construction_and_capabilities(self) -> None:
        target = Target(
            name="hardware-target",
            num_qubits=20,
            native_gates=("h", "cx", "rz", "measure"),
            connectivity=((0, 1), (1, 2)),
            max_shots=5000,
        )
        assert target.num_qubits == 20
        assert target.supports_gate("cx")
        assert not target.supports_gate("ccx")
        assert target.connectivity == ((0, 1), (1, 2))
        assert target.supports_measurement is True
        assert target.supports_dynamic_circuits is False

    def test_universal_target(self) -> None:
        target = Target.universal(num_qubits=10)
        assert target.supports_gate("h")
        assert target.supports_gate("cx")
        assert target.supports_gate("measure")
        assert target.supports_measurement is True
        assert target.supports_dynamic_circuits is True

    def test_frozen(self) -> None:
        target = Target(name="t")
        with pytest.raises(dataclasses.FrozenInstanceError):
            target.name = "other"

    def test_serialization_json_safe(self) -> None:
        target = Target(name="t", native_gates=("h", "cx"), connectivity=((0, 1),))
        d = target.to_dict()
        assert d["native_gates"] == ["h", "cx"]
        assert d["connectivity"] == [[0, 1]]
        assert json.loads(target.to_json())["name"] == "t"


class TestBackendContract:
    """Circuit -> Backend -> Job -> BackendResult on a real simulator."""

    @staticmethod
    def bell_circuit() -> QuantumCircuit:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        return circuit

    def test_device_capabilities(self) -> None:
        backend = StatevectorBackend()
        device = backend.device
        assert device.name == "statevector"
        assert device.device_type == DeviceType.SIMULATOR

    def test_target_capabilities_universal(self) -> None:
        backend = StatevectorBackend()
        target = backend.target
        assert target.name == "statevector_simulator"
        assert target.supports_gate("cx")
        assert target.supports_dynamic_circuits is True

    def test_metadata_json_safe(self) -> None:
        backend = StatevectorBackend()
        meta = backend.metadata()
        assert meta["name"] == "statevector"
        assert "device" in meta and "target" in meta
        json.dumps(meta)

    def test_run_returns_result(self) -> None:
        backend = StatevectorBackend()
        result = backend.run(self.bell_circuit(), shots=200)
        assert isinstance(result, BackendResult)
        assert result.num_qubits == 2
        assert set(result.counts.keys()) <= {"00", "11"}
        assert result.backend_name == "statevector"

    def test_submit_circuit_returns_completed_job(self) -> None:
        backend = StatevectorBackend()
        job = backend.submit_circuit(self.bell_circuit(), shots=200)
        assert isinstance(job, Job)
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert set(job.result.counts.keys()) <= {"00", "11"}


class TestJobContract:
    def test_lifecycle_and_cancel(self) -> None:
        job = Job()
        assert job.status == JobStatus.PENDING
        job.cancel()
        assert job.status == JobStatus.CANCELLED
        job.cancel()  # idempotent

    def test_cancel_completed_is_noop(self) -> None:
        job = Job(status=JobStatus.COMPLETED)
        job.cancel()
        assert job.status == JobStatus.COMPLETED

    def test_scoped_enum(self) -> None:
        assert JobStatus is BareJobStatus
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_metadata(self) -> None:
        job = Job(job_id="abc", status=JobStatus.RUNNING, error="boom")
        meta = job.metadata()
        assert meta["job_id"] == "abc"
        assert meta["status"] == "running"
        assert meta["error"] == "boom"


class TestAlgorithmContract:
    def test_algorithm_marker_default_name(self) -> None:
        class MyAlgo(Algorithm):
            @property
            def name(self) -> str:
                return "my-algo"

        assert MyAlgo().name == "my-algo"

    def test_algorithm_requires_name(self) -> None:
        with pytest.raises(TypeError):
            Algorithm()

    def test_representative_algorithm_returns_serializable_result(
        self,
    ) -> None:
        from microquantum.algorithms import DeutschJozsa

        result = DeutschJozsa(n_qubits=3, balanced=True).run()
        assert isinstance(result, microquantum.DJResult)
        json.dumps(result.to_dict())


class TestOptimizerContract:
    def test_optimizer_abstract_contract(self) -> None:
        assert Optimizer.__abstractmethods__ == frozenset({"_step", "_converged", "max_iter"})
        assert callable(Optimizer.minimize)
        with pytest.raises(TypeError):
            Optimizer()

    def test_optimizer_result_serializable(self) -> None:
        result = OptimizerResult(
            optimal_parameters={},
            optimal_value=-1.5,
            history=[-1.2, -1.5],
            iterations=10,
            converged=True,
        )
        loaded = json.loads(result.to_json())
        assert loaded["optimal_value"] == -1.5
        assert loaded["iterations"] == 10


class TestProviderBoundary:
    def test_provider_is_abstract(self) -> None:
        assert "submit" in HardwareProvider.__abstractmethods__

    def test_provider_credentials_have_no_vendor_leak(self) -> None:
        creds = ProviderCredentials(api_token="token")
        assert not hasattr(creds, "channel")
        assert "channel" not in {f.name for f in dataclasses.fields(ProviderCredentials)}

    def test_ibm_specific_channel_scoped_to_subclass(self) -> None:
        ibm = IBMQuantumCredentials(api_token="t", channel="ibm_cloud")
        assert ibm.channel == "ibm_cloud"
        ionq = IonQCredentials(api_token="t")
        assert not hasattr(ionq, "channel")

    def test_credentials_validate(self) -> None:
        with pytest.raises(ValueError):
            ProviderCredentials(api_token="").validate()
        ProviderCredentials(api_token="ok").validate()

    def test_provider_construction_without_network(self) -> None:
        from microquantum.providers import IonQProvider

        provider = IonQProvider(credentials=IonQCredentials(api_token="t"))
        assert provider.name == "ionq"


class TestAdapterContract:
    def test_adapter_solve_uses_high_level_run(self) -> None:
        class EchoAdapter(DomainAdapter):
            @property
            def domain_name(self) -> str:
                return "test"

            @property
            def supported_problems(self) -> list[str]:
                return ["echo"]

            def validate(self, problem: QuantumProblem) -> list[str]:
                return []

            def encode(self, problem: QuantumProblem) -> QuantumCircuit:
                circuit = QuantumCircuit(2)
                circuit.h(0)
                circuit.cx(0, 1)
                return circuit

            def decode(self, problem, result: BackendResult) -> dict:
                return {"bit": result.most_frequent()}

        problem = QuantumProblem(name="echo", domain="test", num_qubits=2)
        result = EchoAdapter().solve(problem, StatevectorBackend(), shots=100)
        assert result.problem.status.value == "decoded"
        assert result.backend_result is not None
        assert result.most_frequent_state in ("00", "11")
        assert result.decoded["bit"] == result.most_frequent_state
