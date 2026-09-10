"""Tests for the domain adapter framework."""

import numpy as np
import pytest

from microquantum.adapters.base import (
    DomainAdapter,
    ProblemStatus,
    QuantumProblem,
    QuantumResult,
)
from microquantum.adapters.caching import CacheEntry, ResultCache
from microquantum.backends.base import Backend, BackendResult


# ── Helpers ──────────────────────────────────────────────────────────


class DummyBackend(Backend):
    """Minimal backend for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def run_circuit(self, num_qubits, gates, shots=1024, initial_state=None, seed=None):
        counts = {"0" * num_qubits: shots}
        return BackendResult(
            num_qubits=num_qubits,
            backend_name=self.name,
            counts=counts,
        )


class DummyAdapter(DomainAdapter):
    """Minimal adapter for testing."""

    @property
    def domain_name(self) -> str:
        return "test_domain"

    @property
    def supported_problems(self) -> list[str]:
        return ["test_problem"]

    def validate(self, problem):
        errors = []
        if "value" not in problem.parameters:
            errors.append("Missing 'value' parameter")
        if problem.parameters.get("value", 0) < 0:
            errors.append("Value must be non-negative")
        return errors

    def encode(self, problem):
        from microquantum.core.circuit import QuantumCircuit
        qc = QuantumCircuit(problem.num_qubits or 1)
        return qc

    def decode(self, problem, result):
        return {"decoded_value": 42.0}


# ── QuantumProblem ──────────────────────────────────────────────────


class TestQuantumProblem:
    def test_creation(self) -> None:
        p = QuantumProblem(name="test", domain="aerospace")
        assert p.name == "test"
        assert p.domain == "aerospace"
        assert p.status == ProblemStatus.CREATED

    def test_problem_id_deterministic(self) -> None:
        p1 = QuantumProblem(name="test", domain="aero", parameters={"r1": 100})
        p2 = QuantumProblem(name="test", domain="aero", parameters={"r1": 100})
        assert p1.problem_id == p2.problem_id

    def test_problem_id_different(self) -> None:
        p1 = QuantumProblem(name="test", domain="aero", parameters={"r1": 100})
        p2 = QuantumProblem(name="test", domain="aero", parameters={"r1": 200})
        assert p1.problem_id != p2.problem_id

    def test_repr(self) -> None:
        p = QuantumProblem(name="hohmann", domain="aerospace")
        r = repr(p)
        assert "hohmann" in r
        assert "aerospace" in r

    def test_str(self) -> None:
        p = QuantumProblem(
            name="hohmann", domain="aerospace", parameters={"r1": 100, "r2": 200}
        )
        s = str(p)
        assert "hohmann" in s
        assert "aerospace" in s


class TestProblemStatus:
    def test_all_values(self) -> None:
        assert ProblemStatus.CREATED.value == "created"
        assert ProblemStatus.VALIDATED.value == "validated"
        assert ProblemStatus.ENCODED.value == "encoded"
        assert ProblemStatus.EXECUTED.value == "executed"
        assert ProblemStatus.DECODED.value == "decoded"
        assert ProblemStatus.FAILED.value == "failed"


# ── QuantumResult ───────────────────────────────────────────────────


class TestQuantumResult:
    def test_creation(self) -> None:
        p = QuantumProblem(name="test", domain="test")
        r = QuantumResult(problem=p, fidelity=0.95)
        assert r.fidelity == 0.95

    def test_most_frequent_state(self) -> None:
        p = QuantumProblem(name="test", domain="test")
        br = BackendResult(num_qubits=1, backend_name="test", counts={"0": 700, "1": 300})
        r = QuantumResult(problem=p, backend_result=br)
        assert r.most_frequent_state == "0"

    def test_probabilities(self) -> None:
        p = QuantumProblem(name="test", domain="test")
        br = BackendResult(num_qubits=1, backend_name="test", counts={"0": 500, "1": 500})
        r = QuantumResult(problem=p, backend_result=br)
        probs = r.probabilities
        assert abs(probs["0"] - 0.5) < 1e-9

    def test_repr(self) -> None:
        p = QuantumProblem(name="test", domain="test")
        r = QuantumResult(problem=p, fidelity=0.9)
        assert "test" in repr(r)
        assert "0.9" in repr(r)


# ── DomainAdapter ───────────────────────────────────────────────────


class TestDomainAdapter:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            DomainAdapter()

    def test_can_handle(self) -> None:
        adapter = DummyAdapter()
        p = QuantumProblem(name="test_problem", domain="test_domain")
        assert adapter.can_handle(p)

    def test_cannot_handle_wrong_domain(self) -> None:
        adapter = DummyAdapter()
        p = QuantumProblem(name="test_problem", domain="other")
        assert not adapter.can_handle(p)

    def test_cannot_handle_wrong_problem(self) -> None:
        adapter = DummyAdapter()
        p = QuantumProblem(name="other_problem", domain="test_domain")
        assert not adapter.can_handle(p)

    def test_solve_valid(self) -> None:
        adapter = DummyAdapter()
        backend = DummyBackend()
        p = QuantumProblem(name="test_problem", domain="test_domain", parameters={"value": 10})
        result = adapter.solve(p, backend, shots=100, seed=42)
        assert result.problem.status == ProblemStatus.DECODED
        assert result.fidelity > 0
        assert result.decoded["decoded_value"] == 42.0

    def test_solve_validation_fails(self) -> None:
        adapter = DummyAdapter()
        backend = DummyBackend()
        p = QuantumProblem(name="test_problem", domain="test_domain", parameters={})
        with pytest.raises(ValueError, match="validation failed"):
            adapter.solve(p, backend)

    def test_solve_negative_value_fails(self) -> None:
        adapter = DummyAdapter()
        backend = DummyBackend()
        p = QuantumProblem(name="test_problem", domain="test_domain", parameters={"value": -1})
        with pytest.raises(ValueError, match="non-negative"):
            adapter.solve(p, backend)

    def test_repr(self) -> None:
        adapter = DummyAdapter()
        r = repr(adapter)
        assert "test_domain" in r


# ── ResultCache ─────────────────────────────────────────────────────


class TestResultCache:
    def test_empty_cache(self) -> None:
        cache = ResultCache()
        assert cache.size == 0
        assert len(cache) == 0

    def test_put_and_get(self) -> None:
        cache = ResultCache()
        p = QuantumProblem(name="test", domain="aero", parameters={"r1": 100})
        br = BackendResult(num_qubits=1, backend_name="sv", counts={"0": 500})
        result = QuantumResult(problem=p, backend_result=br, fidelity=0.5)

        cache.put(p, "sv", 1000, result)
        cached = cache.get(p, "sv", 1000)
        assert cached is not None
        assert cached.fidelity == 0.5

    def test_cache_miss(self) -> None:
        cache = ResultCache()
        p = QuantumProblem(name="test", domain="aero")
        assert cache.get(p, "sv", 1000) is None

    def test_cache_different_backend(self) -> None:
        cache = ResultCache()
        p = QuantumProblem(name="test", domain="aero")
        br = BackendResult(num_qubits=1, backend_name="sv", counts={"0": 500})
        result = QuantumResult(problem=p, backend_result=br, fidelity=0.5)

        cache.put(p, "sv", 1000, result)
        assert cache.get(p, "dm", 1000) is None

    def test_cache_different_shots(self) -> None:
        cache = ResultCache()
        p = QuantumProblem(name="test", domain="aero")
        br = BackendResult(num_qubits=1, backend_name="sv", counts={"0": 500})
        result = QuantumResult(problem=p, backend_result=br, fidelity=0.5)

        cache.put(p, "sv", 1000, result)
        assert cache.get(p, "sv", 500) is None

    def test_invalidate(self) -> None:
        cache = ResultCache()
        p = QuantumProblem(name="test", domain="aero", parameters={"r1": 100})
        br = BackendResult(num_qubits=1, backend_name="sv", counts={"0": 500})
        result = QuantumResult(problem=p, backend_result=br, fidelity=0.5)

        cache.put(p, "sv", 1000, result)
        assert cache.size == 1
        removed = cache.invalidate(p)
        assert removed == 1
        assert cache.size == 0

    def test_clear(self) -> None:
        cache = ResultCache()
        for i in range(5):
            p = QuantumProblem(name=f"test{i}", domain="aero")
            br = BackendResult(num_qubits=1, backend_name="sv", counts={"0": 500})
            result = QuantumResult(problem=p, backend_result=br, fidelity=0.5)
            cache.put(p, "sv", 1000, result)
        assert cache.size == 5
        cleared = cache.clear()
        assert cleared == 5
        assert cache.size == 0

    def test_max_size_eviction(self) -> None:
        cache = ResultCache(max_size=3)
        for i in range(5):
            p = QuantumProblem(name=f"test{i}", domain="aero")
            br = BackendResult(num_qubits=1, backend_name="sv", counts={"0": 500})
            result = QuantumResult(problem=p, backend_result=br, fidelity=0.5)
            cache.put(p, "sv", 1000, result)
        assert cache.size == 3

    def test_ttl_expiry(self) -> None:
        cache = ResultCache()
        p = QuantumProblem(name="test", domain="aero")
        br = BackendResult(num_qubits=1, backend_name="sv", counts={"0": 500})
        result = QuantumResult(problem=p, backend_result=br, fidelity=0.5)

        cache.put(p, "sv", 1000, result, ttl=0.01)
        import time
        time.sleep(0.02)
        assert cache.get(p, "sv", 1000) is None
        assert cache.size == 0

    def test_hit_count(self) -> None:
        cache = ResultCache()
        p = QuantumProblem(name="test", domain="aero")
        br = BackendResult(num_qubits=1, backend_name="sv", counts={"0": 500})
        result = QuantumResult(problem=p, backend_result=br, fidelity=0.5)

        cache.put(p, "sv", 1000, result)
        cache.get(p, "sv", 1000)
        cache.get(p, "sv", 1000)
        stats = cache.stats
        assert stats["total_hits"] == 2

    def test_stats(self) -> None:
        cache = ResultCache(max_size=None)
        p = QuantumProblem(name="test", domain="aero")
        br = BackendResult(num_qubits=1, backend_name="sv", counts={"0": 500})
        result = QuantumResult(problem=p, backend_result=br, fidelity=0.5)

        cache.put(p, "sv", 1000, result)
        stats = cache.stats
        assert stats["size"] == 1
        assert stats["max_size"] is None

    def test_repr(self) -> None:
        cache = ResultCache(max_size=100)
        assert "100" in repr(cache)

    def test_cache_entry_age(self) -> None:
        entry = CacheEntry(
            result=QuantumResult(
                problem=QuantumProblem(name="t", domain="d"),
                fidelity=1.0,
            )
        )
        assert entry.age >= 0
        assert not entry.is_expired
