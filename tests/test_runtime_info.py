"""Phase 119 tests: runtime introspection."""

import json

import pytest

import microquantum
from microquantum import ExecutionRuntime, RuntimeConfig, runtime_info
from microquantum.backends import MockBackend
from microquantum.backends.registry import BackendRegistry, default_registry
from microquantum.runtime.strategy import ExecutionStrategy


class TestRuntimeInfo:
    def test_default_snapshot(self):
        info = runtime_info()
        assert info.version == microquantum.__version__
        assert info.runtime == "ExecutionRuntime"
        assert info.python
        assert info.numpy

    def test_strategies_derived_from_enum(self):
        info = runtime_info()
        assert info.strategies == sorted(s.value for s in ExecutionStrategy)

    def test_backends_derived_from_registry(self):
        info = runtime_info()
        assert [b["name"] for b in info.backends] == list(default_registry.names())

    def test_backend_summary_has_capabilities(self):
        info = runtime_info()
        assert info.backends
        summary = info.backends[0]
        assert "target_class" in summary
        assert "execution" in summary
        assert "circuit_features" in summary

    def test_runtime_aware_snapshot(self):
        rt = ExecutionRuntime(config=RuntimeConfig(default_optimization_level=1))
        info = runtime_info(rt)
        assert info.default_backend == rt.default_backend.name
        assert info.default_optimization_level == 1

    def test_registry_override(self):
        registry = BackendRegistry()
        registry.register(MockBackend(name="mock_info"), default=True)
        info = runtime_info(registry=registry)
        assert [b["name"] for b in info.backends] == ["local_simulator", "mock_info"]
        assert info.default_backend == "mock_info"

    def test_to_dict_json_safe(self):
        info = runtime_info()
        data = info.to_dict()
        assert data["version"] == microquantum.__version__
        json.dumps(data)

    def test_top_level_identity(self):
        from microquantum.runtime.info import RuntimeInfo
        from microquantum.runtime.info import runtime_info as deep

        assert microquantum.runtime_info is deep is runtime_info
        assert isinstance(runtime_info(), RuntimeInfo)

    @pytest.mark.parametrize("field", ["version", "runtime"])
    def test_required_fields_never_empty(self, field):
        assert getattr(runtime_info(), field)