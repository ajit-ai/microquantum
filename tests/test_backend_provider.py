"""MQ-06 tests: backend capabilities, registry, providers and runtime routing.

Covers the backend/provider abstraction layer introduced in MQ-06:

* :class:`BackendCapabilities` + :class:`TargetClass`
* :class:`Backend` plan contract (``validate`` / ``supports`` / ``execute``)
* :class:`MockBackend` (deterministic, contract-obeying)
* :class:`LocalSimulatorBackend`
* :class:`BackendRegistry`
* :class:`Provider` + :class:`LocalProvider`
* :class:`BackendAdapter` boundary
* runtime integration (backend names, registry defaults, validation)
* MQ-05 algorithms executing through the new backend path
"""

from __future__ import annotations

import pytest

from microquantum import (
    Backend,
    BackendCapabilities,
    BackendRegistry,
    BackendResult,
    ExecutionPlan,
    ExecutionRuntime,
    GroverSearch,
    LocalProvider,
    LocalSimulatorBackend,
    MockBackend,
    Provider,
    QuantumCircuit,
    TargetClass,
)
from microquantum.backends.adapter import BackendAdapter
from microquantum.backends.capabilities import (
    EXECUTION_DENSITY_MATRIX,
    EXECUTION_STATEVECTOR,
    FEATURE_MID_CIRCUIT_MEASUREMENT,
    FEATURE_PARAMETERIZED_CIRCUITS,
    simulator_capabilities,
)


def bell_circuit(phase: float = 0.5, qubits: int = 2) -> QuantumCircuit:
    from microquantum import Parameter

    theta = Parameter("theta")
    qc = QuantumCircuit(qubits)
    qc.h(0)
    qc.rz(theta, 1)
    qc.cnot(0, 1)
    return qc.bind_parameters({theta: phase})


# ----------------------------------------------------------------------
# BackendCapabilities
# ----------------------------------------------------------------------


class TestBackendCapabilities:
    def test_default_capabilities(self) -> None:
        caps = BackendCapabilities()
        assert caps.target_class is TargetClass.SIMULATOR
        assert caps.supports_shots
        assert caps.supports_statevector

    def test_simulator_factory_statevector_flag(self) -> None:
        sv = simulator_capabilities(statevector=True)
        assert sv.supports_statevector
        assert not sv.supports_execution(EXECUTION_DENSITY_MATRIX)
        dm = simulator_capabilities(statevector=False, density_matrix=True)
        assert dm.supports_execution(EXECUTION_DENSITY_MATRIX)
        assert not dm.supports_statevector

    def test_helpers(self) -> None:
        caps = BackendCapabilities(
            max_qubits=32,
            execution=frozenset({EXECUTION_STATEVECTOR, "custom_mode"}),
        )
        assert caps.supports_execution("custom_mode")
        assert not caps.supports_shots
        assert not caps.supports_feature(FEATURE_MID_CIRCUIT_MEASUREMENT)
        assert caps.is_simulator
        assert not caps.is_hardware
        assert not caps.is_remote

    def test_hardware_classification(self) -> None:
        caps = BackendCapabilities(target_class=TargetClass.QUANTUM_HARDWARE)
        assert caps.is_hardware
        assert not caps.is_simulator

    def test_merge_intersects(self) -> None:
        a = simulator_capabilities(max_qubits=40, statevector=True)
        b = simulator_capabilities(max_qubits=20, statevector=False)
        merged = a.merge(b)
        assert merged.max_qubits == 20
        assert not merged.supports_statevector
        assert merged.supports_shots

    def test_merge_preserves_metadata(self) -> None:
        a = BackendCapabilities(metadata={"engine": "numpy"})
        b = BackendCapabilities(metadata={"vendor": "private"})
        merged = a.merge(b)
        assert merged.metadata == {"engine": "numpy", "vendor": "private"}

    def test_json_round_trip(self) -> None:
        caps = simulator_capabilities(max_qubits=16, statevector=True)
        caps.metadata["engine"] = "numpy"
        restored = BackendCapabilities.from_dict(caps.to_dict())
        assert restored == caps
        assert restored.max_qubits == 16
        assert restored.supports_statevector

    def test_json_string_is_parseable(self) -> None:
        import json

        caps = simulator_capabilities(max_qubits=8)
        payload = json.loads(caps.to_json())
        assert payload["target_class"] == "simulator"
        assert payload["max_qubits"] == 8


# ----------------------------------------------------------------------
# Backend plan contract + BackendResult payload
# ----------------------------------------------------------------------


class _RecordingBackend(Backend):
    """Minimal backend that records the calls it receives."""

    @property
    def name(self) -> str:
        return "recording"

    def __init__(self) -> None:
        self.executed: list[tuple[int, int, object]] = []

    def run_circuit(self, num_qubits, gates, shots=1024, initial_state=None, seed=None):
        self.executed.append((shots, seed, len(gates)))
        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            samples=[0] * shots,
            shots=shots,
            seed=seed,
            target_name=self.name,
            expectations={"E": 0.5},
            eigenvalues=[0.0, 1.0],
            native={"vendor": "n/a"},
        )


class TestBackendContract:
    def test_backend_metadata_includes_capabilities(self) -> None:
        backend = _RecordingBackend()
        meta = backend.metadata()
        assert meta["name"] == "recording"
        assert "capabilities" in meta
        assert meta["capabilities"]["target_class"] == "simulator"

    def test_validate_accepts_fit(self) -> None:
        backend = _RecordingBackend()
        plan = ExecutionPlan.from_circuit(bell_circuit(), shots=64)
        assert backend.supports(plan)
        assert backend.validate(plan) == []

    def test_validate_rejects_oversized(self) -> None:
        backend = MockBackend(name="tiny", max_qubits=1)
        plan = ExecutionPlan.from_circuit(QuantumCircuit(2), shots=64)
        problems = backend.validate(plan)
        assert problems
        assert "2 qubits" in problems[0]
        assert not backend.supports(plan)

    def test_validate_rejects_shots_when_unsupported(self) -> None:
        backend = MockBackend(name="sample_only", supports_shots=False)
        plan = ExecutionPlan.from_circuit(QuantumCircuit(1), shots=32)
        problems = backend.validate(plan)
        assert any("shots" in p for p in problems)

    def test_execute_runs_validated_plan(self) -> None:
        backend = _RecordingBackend()
        plan = ExecutionPlan.from_circuit(bell_circuit(), shots=128, seed=7)
        result = backend.execute(plan)
        assert result.backend_name == "recording"
        assert result.shots == 128
        assert result.seed == 7
        assert result.target_name == "recording"
        assert result.expectations == {"E": 0.5}
        assert result.eigenvalues == [0.0, 1.0]
        assert result.native == {"vendor": "n/a"}
        assert len(result.samples) == 128
        assert backend.executed == [(128, 7, 3)]

    def test_execute_rejects_invalid_plan(self) -> None:
        backend = MockBackend(name="tiny", max_qubits=1)
        plan = ExecutionPlan.from_circuit(QuantumCircuit(2), shots=64)
        with pytest.raises(ValueError, match="rejected the plan"):
            backend.execute(plan)


class TestBackendResultPayload:
    def test_to_dict_includes_payload(self) -> None:
        result = _RecordingBackend().run_circuit(2, [], shots=4, seed=3)
        data = result.to_dict()
        assert data["samples"] == [0, 0, 0, 0]
        assert data["expectations"] == {"E": 0.5}
        assert data["eigenvalues"] == [0.0, 1.0]
        assert data["native"] == {"vendor": "n/a"}
        assert data["shots"] == 4
        assert data["seed"] == 3

    def test_to_json_round_trip(self) -> None:
        import json

        result = _RecordingBackend().run_circuit(2, [], shots=4, seed=3)
        payload = json.loads(result.to_json())
        assert payload["samples"] == [0, 0, 0, 0]
        assert payload["shots"] == 4


# ----------------------------------------------------------------------
# MockBackend
# ----------------------------------------------------------------------


class TestMockBackend:
    def test_deterministic_seed(self) -> None:
        plan = ExecutionPlan.from_circuit(bell_circuit(), shots=200, seed=42)
        a = MockBackend().execute(plan)
        b = MockBackend().execute(plan)
        assert a.counts == b.counts

    def test_seed_changes_results(self) -> None:
        plan = ExecutionPlan.from_circuit(bell_circuit(), shots=200, seed=1)
        base = MockBackend().execute(plan)
        other = MockBackend().execute(
            ExecutionPlan.from_circuit(bell_circuit(), shots=200, seed=2)
        )
        assert base.counts != other.counts

    def test_capabilities(self) -> None:
        mb = MockBackend(name="stub", max_qubits=8)
        assert mb.capabilities.max_qubits == 8
        assert mb.capabilities.is_simulator
        assert mb.capabilities.supports_feature(FEATURE_PARAMETERIZED_CIRCUITS)

    def test_result_plumbing(self) -> None:
        plan = ExecutionPlan.from_circuit(bell_circuit(), shots=64, seed=9)
        result = MockBackend().execute(plan)
        assert result.backend_name == "mock"
        assert result.samples is not None
        assert len(result.samples) == 64
        assert sum(result.counts.values()) == 64


# ----------------------------------------------------------------------
# LocalSimulatorBackend
# ----------------------------------------------------------------------


class TestLocalSimulatorBackend:
    def test_identity_and_capabilities(self) -> None:
        backend = LocalSimulatorBackend()
        assert backend.name == "local_simulator"
        assert backend.device.device_type.value == "cpu"
        caps = backend.capabilities
        assert caps.is_simulator
        assert caps.supports_statevector
        assert caps.supports_shots

    def test_runs_bell_circuit(self) -> None:
        plan = ExecutionPlan.from_circuit(bell_circuit(), shots=1024, seed=5)
        result = LocalSimulatorBackend().execute(plan)
        assert result.backend_name == "local_simulator"
        assert result.statevector is not None
        assert sum(result.counts.values()) == 1024

    def test_matches_statevector_engine(self) -> None:
        from microquantum import StatevectorBackend

        plan = ExecutionPlan.from_circuit(
            bell_circuit(0.5), shots=2048, seed=11
        )
        local = LocalSimulatorBackend().execute(plan)
        reference = StatevectorBackend().execute(plan)
        assert local.counts == reference.counts


# ----------------------------------------------------------------------
# BackendRegistry
# ----------------------------------------------------------------------


class TestBackendRegistry:
    def test_bare_registry_has_local_simulator_default(self) -> None:
        registry = BackendRegistry()
        assert registry.default is not None
        assert registry.default.name == "local_simulator"
        assert "local_simulator" in registry.names()

    def test_register_and_lookup(self) -> None:
        registry = BackendRegistry()
        register = MockBackend(name="register-demo", max_qubits=4)
        registry.register(register)
        assert registry.get("register-demo") is register
        assert registry.has("register-demo")
        assert "register-demo" in registry
        assert len(registry) == 2

    def test_duplicate_rejected(self) -> None:
        registry = BackendRegistry()
        registry.register(MockBackend(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockBackend(name="dup"))

    def test_duplicate_with_replace(self) -> None:
        registry = BackendRegistry()
        a = MockBackend(name="dup")
        b = MockBackend(name="dup")
        registry.register(a)
        registry.register(b, replace=True)
        assert registry.get("dup") is b

    def test_register_custom_name(self) -> None:
        registry = BackendRegistry()
        backend = MockBackend(name="internal-name")
        registry.register(backend, name="exposed")
        assert registry.get("exposed") is backend
        assert not registry.has("internal-name")

    def test_unregister(self) -> None:
        registry = BackendRegistry()
        backend = MockBackend(name="gone")
        registry.register(backend)
        assert registry.unregister("gone") is backend
        assert not registry.has("gone")

    def test_unknown_get_raises_with_listing(self) -> None:
        registry = BackendRegistry()
        with pytest.raises(KeyError, match="local_simulator"):
            registry.get("does-not-exist")

    def test_default_settable_and_resettable(self) -> None:
        registry = BackendRegistry()
        mock = MockBackend(name="preferred", max_qubits=3)
        registry.register(mock)
        registry.set_default("preferred")
        assert registry.default is mock
        registry.reset_default()
        assert registry.default.name == "local_simulator"

    def test_rejects_non_backend(self) -> None:
        registry = BackendRegistry()
        with pytest.raises(TypeError, match="Backend"):
            registry.register("not-a-backend")  # type: ignore[arg-type]

    def test_serialization(self) -> None:
        registry = BackendRegistry()
        registry.register(MockBackend(name="extra"))
        data = registry.to_dict()
        assert data["default"] == "local_simulator"
        names = {b["name"] for b in data["backends"]}
        assert {"local_simulator", "extra"} <= names
        import json

        assert json.loads(registry.to_json())["default"] == "local_simulator"


# ----------------------------------------------------------------------
# Provider abstraction
# ----------------------------------------------------------------------


class TestProviderAbstraction:
    def test_provider_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Provider()

    def test_local_provider_backends(self) -> None:
        provider = LocalProvider()
        names = {b.name for b in provider.backends()}
        assert {
            "local_simulator",
            "statevector",
            "mock",
            "density_matrix",
        } <= names

    def test_get_backend_and_missing(self) -> None:
        provider = LocalProvider()
        assert provider.has_backend("mock")
        with pytest.raises(KeyError, match="mock"):
            provider.get_backend("no-such-backend")

    def test_register_all_into_registry(self) -> None:
        registry = BackendRegistry()
        provider = LocalProvider()
        registered = provider.register_all(registry)
        assert "local_simulator" in registered
        assert registry.has("local_simulator")
        assert registry.has("statevector")
        assert registry.has("mock")

    def test_provider_serialization(self) -> None:
        data = LocalProvider().to_dict()
        assert data["name"] == "local"
        assert "statevector" in data["backends"]


# ----------------------------------------------------------------------
# BackendAdapter boundary
# ----------------------------------------------------------------------


class _VendorBackend(BackendAdapter):
    """An adapter delegating to a fake vendor."""

    @property
    def name(self) -> str:
        return "vendor"

    def submit_to_vendor(self, circuit, *, shots, seed):
        return {"circuit": circuit, "shots": shots, "seed": seed}

    def collect_from_vendor(self, vendor_handle, circuit):
        num_qubits = circuit.num_qubits
        counts = {"0" * num_qubits: vendor_handle["shots"]}
        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            counts=counts,
            native={"vendor_received": True},
        )


class TestBackendAdapter:
    def test_raw_gate_matrices_rejected(self) -> None:
        with pytest.raises(ValueError, match="raw gate matrices"):
            _VendorBackend().run_circuit(1, [], shots=8)

    def test_plan_execution_delegates_to_vendor(self) -> None:
        backend = _VendorBackend()
        plan = ExecutionPlan.from_circuit(bell_circuit(), shots=32, seed=3)
        result = backend.execute(plan)
        assert result.backend_name == "vendor"
        assert result.native == {"vendor_received": True}
        assert sum(result.counts.values()) == 32

    def test_runtime_through_adapter(self) -> None:
        backend = _VendorBackend()
        result = ExecutionRuntime().execute(
            bell_circuit(), backend=backend, shots=16, seed=1
        )
        assert result.metadata["backend"] == "vendor"
        assert result.native == {"vendor_received": True}


# ----------------------------------------------------------------------
# Runtime integration
# ----------------------------------------------------------------------


class TestRuntimeIntegration:
    def test_no_arg_runtime_keeps_statevector(self) -> None:
        assert ExecutionRuntime().default_backend.name == "statevector"

    def test_registry_default_selected(self) -> None:
        registry = BackendRegistry()
        registry.register(MockBackend(name="family", max_qubits=4))
        registry.set_default("family")
        runtime = ExecutionRuntime(registry=registry)
        assert runtime.default_backend.name == "family"

    def test_plan_backend_by_name_resolved(self) -> None:
        registry = BackendRegistry()
        runtime = ExecutionRuntime(registry=registry)
        plan = ExecutionPlan.from_circuit(
            bell_circuit(), backend="local_simulator", shots=64, seed=2
        )
        result = runtime.execute(plan)
        assert result.backend_name == "local_simulator"

    def test_string_backend_without_registry(self) -> None:
        runtime = ExecutionRuntime()
        plan = ExecutionPlan.from_circuit(
            bell_circuit(), backend="local_simulator", shots=64, seed=2
        )
        result = runtime.execute(plan)
        assert result.backend_name == "local_simulator"

    def test_unknown_backend_name_raises(self) -> None:
        runtime = ExecutionRuntime()
        with pytest.raises(KeyError, match="unknown backend"):
            runtime.execute(bell_circuit(), backend="no-such-backend")

    def test_registry_name_takes_precedence_over_default_registry(self) -> None:
        registry = BackendRegistry()
        local = MockBackend(name="local_simulator", max_qubits=5)
        registry.register(local, replace=True)
        runtime = ExecutionRuntime(registry=registry)
        plan = ExecutionPlan.from_circuit(bell_circuit(), backend="local_simulator")
        assert runtime.resolve_backend(plan.backend) is local

    def test_execution_validates_before_running(self) -> None:
        tiny = MockBackend(name="tiny", max_qubits=1)
        runtime = ExecutionRuntime()
        with pytest.raises(ValueError, match="cannot run on backend 'tiny'"):
            runtime.execute(QuantumCircuit(2), backend=tiny)

    def test_shots_support_validated(self) -> None:
        sample_only = MockBackend(name="sample_only", supports_shots=False)
        runtime = ExecutionRuntime()
        with pytest.raises(ValueError, match="shots"):
            runtime.execute(
                ExecutionPlan.from_circuit(QuantumCircuit(1), shots=16),
                backend=sample_only,
            )

    def test_batch_and_sweep_respect_name_resolution(self) -> None:
        runtime = ExecutionRuntime(registry=BackendRegistry())
        results = runtime.execute_batch(
            [bell_circuit(), bell_circuit(0.2)],
            backend="local_simulator",
            shots=64,
            seed=6,
        )
        assert all(r.backend_name == "local_simulator" for r in results)

    def test_result_payload_populated(self) -> None:
        result = ExecutionRuntime().execute(bell_circuit(), shots=128, seed=9)
        assert result.shots == 128
        assert result.seed == 9
        assert result.target_name == "statevector_simulator"


# ----------------------------------------------------------------------
# MQ-05 algorithms through the new backend path
# ----------------------------------------------------------------------


class TestAlgorithmThroughBackend:
    def test_grover_through_plan_and_backend(self) -> None:
        algorithm = GroverSearch(num_qubits=3, target=5)
        plan = ExecutionPlan.from_circuit(
            algorithm.build_circuit(),
            backend=LocalSimulatorBackend(),
            shots=4096,
            seed=123,
        )
        result = ExecutionRuntime().execute(plan)
        assert result.most_frequent() == "101"
        assert result.backend_name == "local_simulator"
        assert result.metadata["backend"] == "local_simulator"

    def test_grover_direct_backend_execute(self) -> None:
        algorithm = GroverSearch(num_qubits=3, target=5)
        plan = ExecutionPlan.from_circuit(
            algorithm.build_circuit(),
            backend="local_simulator",
            shots=4096,
            seed=123,
        )
        result = LocalSimulatorBackend().execute(plan)
        assert result.most_frequent() == "101"

    def test_algorithm_uses_runtime_default(self) -> None:
        algorithm = GroverSearch(num_qubits=2, target=3)
        result = ExecutionRuntime().execute(
            algorithm.build_circuit(), shots=4096, seed=17
        )
        assert result.most_frequent() == "11"
        assert result.backend_name == "statevector"

    def test_registry_routes_algorithm_to_mock(self) -> None:
        registry = BackendRegistry()
        registry.register(MockBackend(name="route_mock", max_qubits=3))
        registry.set_default("route_mock")
        runtime = ExecutionRuntime(registry=registry)
        algorithm = GroverSearch(num_qubits=3, target=5)
        result = runtime.execute(algorithm.build_circuit(), shots=512, seed=3)
        assert result.backend_name == "route_mock"
        assert result.metadata["backend"] == "route_mock"

    def test_algorithm_run_pipeline_complete(self) -> None:
        algorithm = GroverSearch(num_qubits=3, target=5)
        alg_result = algorithm.run(shots=4096, seed=42)
        assert alg_result.target == 5