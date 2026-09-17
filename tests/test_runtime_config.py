"""Phase 119 tests: runtime configuration surface."""

import pytest

import microquantum
from microquantum import ExecutionRuntime, RuntimeConfig
from microquantum.backends import MockBackend
from microquantum.backends.registry import BackendRegistry


class TestRuntimeConfigValidation:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.backend is None
        assert cfg.registry is None
        assert cfg.default_target is None
        assert cfg.history_size == 200
        assert cfg.default_optimization_level == 0

    def test_rejects_bad_backend(self):
        with pytest.raises(TypeError):
            RuntimeConfig(backend=object())  # type: ignore[arg-type]

    def test_accepts_backend_instance_and_name(self):
        assert RuntimeConfig(backend=MockBackend()).backend.name
        assert RuntimeConfig(backend="local_simulator").backend == "local_simulator"

    def test_rejects_bad_registry(self):
        with pytest.raises(TypeError):
            RuntimeConfig(registry=object())  # type: ignore[arg-type]

    def test_rejects_bad_optimization_level(self):
        with pytest.raises(ValueError):
            RuntimeConfig(default_optimization_level=7)
        with pytest.raises(TypeError):
            RuntimeConfig(default_optimization_level="1")  # type: ignore[arg-type]

    def test_rejects_negative_history(self):
        with pytest.raises(ValueError):
            RuntimeConfig(history_size=-1)

    def test_to_dict_is_json_safe(self):
        cfg = RuntimeConfig(backend="mock", history_size=50)
        data = cfg.to_dict()
        assert data["backend"] == "mock"
        assert data["history_size"] == 50
        assert data["default_optimization_level"] == 0
        assert cfg.to_json()


class TestExecutionRuntimeConfig:
    def test_runtime_exposes_config(self):
        cfg = RuntimeConfig(history_size=7)
        rt = ExecutionRuntime(config=cfg)
        assert rt.config == cfg

    def test_legacy_kwargs_still_work(self):
        backend = MockBackend()
        rt = ExecutionRuntime(backend=backend, history_size=3)
        assert rt.config.backend is backend
        assert rt.config.history_size == 3
        assert rt.default_backend is backend

    def test_config_plus_kwargs_merges(self):
        cfg = RuntimeConfig(history_size=4, default_optimization_level=2)
        rt = ExecutionRuntime(config=cfg, history_size=10)
        assert rt.config.history_size == 10
        assert rt.config.default_optimization_level == 2

    def test_rejects_non_config(self):
        with pytest.raises(TypeError):
            ExecutionRuntime(config=object())  # type: ignore[arg-type]

    def test_configure_returns_fresh_runtime(self):
        rt = ExecutionRuntime(config=RuntimeConfig(history_size=5))
        other = rt.configure(history_size=8, default_optimization_level=1)
        assert isinstance(other, ExecutionRuntime)
        assert other.config is not rt.config
        assert other.config.history_size == 8
        assert rt.config.history_size == 5
        assert other.config.default_optimization_level == 1

    def test_string_backend_resolved_via_registry(self):
        registry = BackendRegistry()
        registered = MockBackend()
        registry.register(registered, name="mock_sim")
        rt = ExecutionRuntime(config=RuntimeConfig(backend="mock_sim", registry=registry))
        assert rt.default_backend is registered
        assert rt.config.backend == "mock_sim"

    def test_history_size_from_config(self):
        rt = ExecutionRuntime(config=RuntimeConfig(history_size=2))
        rt.execute(microquantum.QuantumCircuit(1), shots=8)
        rt.execute(microquantum.QuantumCircuit(1), shots=8)
        rt.execute(microquantum.QuantumCircuit(1), shots=8)
        assert len(rt.history) == 2

    def test_default_optimization_level_wired_to_circuits(self):
        rt = ExecutionRuntime(config=RuntimeConfig(default_optimization_level=2))
        result = rt.execute(microquantum.QuantumCircuit(2).h(0))
        assert result.metadata["strategy"] == "compiled"
        assert result.metadata["optimization_level"] == 2

    def test_default_optimization_level_zero_stays_direct(self):
        rt = ExecutionRuntime()
        result = rt.execute(microquantum.QuantumCircuit(1).h(0), shots=16, seed=1)
        assert result.metadata["strategy"] == "direct"

    def test_top_level_identity(self):
        from microquantum.runtime.config import RuntimeConfig as DeepConfig

        assert microquantum.RuntimeConfig is DeepConfig is RuntimeConfig