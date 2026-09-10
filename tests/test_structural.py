"""Structural regression tests for the MQ-01 consolidated SDK architecture.

Verifies the public API surface, the canonical generic result contract,
uniform result serialization, representative backend/algorithm execution,
the provider extension boundary, package import isolation (no circular
imports), and the absence of domain-specific business terminology.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest


class TestPublicImports:
    """Public package exports resolve consistently."""

    def test_top_level_imports(self) -> None:
        import microquantum

        names = [
            "QuantumCircuit",
            "Operator",
            "StateVector",
            "Backend",
            "BackendResult",
            "Executor",
            "ExecutorResult",
            "StatevectorBackend",
            "Result",
            "AnalysisResult",
            "QUBOProblem",
            "HardwareProvider",
            "DomainAdapter",
            "QuantumProblem",
            "QuantumResult",
            "Optimizer",
            "OptimizerResult",
            "GroverSearch",
            "VQE",
        ]
        for name in names:
            assert hasattr(microquantum, name), f"missing top-level export: {name}"

    def test_core_imports(self) -> None:
        from microquantum.core import (  # noqa: F401
            Operator,
            Parameter,
            PauliSum,
            QuantumCircuit,
            StateVector,
        )

    def test_backend_imports(self) -> None:
        from microquantum.backends import (  # noqa: F401
            Backend,
            BackendResult,
            DensityMatrixBackend,
            Executor,
            ExecutorResult,
            Job,
            NoiseModel,
            StatevectorBackend,
        )

    def test_analytics_exports(self) -> None:
        from microquantum.analytics import (  # noqa: F401
            AnalysisResult,
            BaseAnalytics,
            Result,
            plot_allocation,
            plot_distribution,
            plot_scatter,
        )


class TestImportIsolation:
    """No accidental circular-import failures across subpackages."""

    TARGETS = [
        "microquantum",
        "microquantum.core",
        "microquantum.backends",
        "microquantum.algorithms",
        "microquantum.analytics",
        "microquantum.providers",
        "microquantum.optimization",
        "microquantum.optimizers",
        "microquantum.adapters",
        "microquantum.qml",
        "microquantum.qec",
        "microquantum.chemistry",
        "microquantum.mitigation",
        "microquantum.benchmarks",
    ]

    def test_each_submodule_imports_in_isolation(self) -> None:
        for target in self.TARGETS:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", f"import {target}"],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, (
                f"'import {target}' failed in isolation:\n{proc.stderr}"
            )

    def test_reverse_dependency_order_import(self) -> None:
        """Import top layers before lower layers must also succeed."""
        for target in reversed(self.TARGETS):
            __import__(target)


class TestResultContract:
    """The canonical Result contract is generic and serializable."""

    def test_canonical_construction(self) -> None:
        from microquantum.analytics.result import Result

        result = Result(
            problem="optimization",
            solution=0.9,
            confidence=0.91,
            qubit_count=8,
            runtime_ms=48.3,
            baseline=0.7,
        )
        assert result.solution == 0.9
        assert result.baseline == 0.7
        assert result.improved_over_baseline is True

    def test_legacy_kwargs_deprecated(self) -> None:
        from microquantum.analytics.result import Result

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Result(problem="t", decision=0.9, classical_baseline=0.7)

        assert result.solution == 0.9
        assert result.baseline == 0.7
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_legacy_property_aliases(self) -> None:
        from microquantum.analytics.result import Result

        result = Result(problem="t", solution=0.9, baseline=0.7)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert result.decision == 0.9
            assert result.classical_baseline == 0.7

        assert len(caught) >= 2
        assert all(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_to_dict_json_safe_numpy(self) -> None:
        from microquantum.analytics.result import Result

        result = Result(
            problem="routing",
            solution={"tour": np.array([0, 2, 1]), "cost": np.float32(42.0)},
        )
        data = result.to_dict()
        assert data["solution"]["tour"] == [0, 2, 1]
        assert data["solution"]["cost"] == 42.0
        json.dumps(data)

    def test_to_json_returns_string(self) -> None:
        from microquantum.analytics.result import Result

        result = Result(problem="q", solution=1.0)
        payload = result.to_json()
        assert isinstance(payload, str)


class TestUniformSerialization:
    """Every public result type has JSON-safe to_dict()/to_json()."""

    def test_backend_result_serialization(self) -> None:
        from microquantum import QuantumCircuit, StatevectorBackend

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = StatevectorBackend().run(qc, shots=1024, seed=7)

        data = result.to_dict()
        assert data["num_qubits"] == 2
        assert json.dumps(data)
        assert isinstance(result.to_json(), str)

    def test_executor_result_serialization(self) -> None:
        from microquantum import Executor, QuantumCircuit, StatevectorBackend

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = Executor(backend=StatevectorBackend()).run(qc, shots=512, seed=3)

        data = result.to_dict()
        assert data["num_qubits"] == 2
        assert json.dumps(data)
        assert isinstance(result.to_json(), str)

    def test_analysis_result_serialization(self) -> None:
        from microquantum import AnalysisResult

        result = AnalysisResult(solution={"energy": np.float64(-1.857)}, success=True)
        data = result.to_dict()
        assert data["solution"]["energy"] == -1.857
        assert json.dumps(data)
        assert isinstance(result.to_json(), str)

    def test_quantum_result_serialization(self) -> None:
        from microquantum import QuantumProblem, QuantumResult

        problem = QuantumProblem(name="binary_optimization", domain="optimization")
        result = QuantumResult(problem=problem, fidelity=0.9, decoded={"bit": "01"})
        data = result.to_dict()
        assert data["problem"]["name"] == "binary_optimization"
        assert data["problem"]["domain"] == "optimization"
        assert data["decoded"]["bit"] == "01"
        assert json.dumps(data)
        assert isinstance(result.to_json(), str)


class TestRepresentativeExecution:
    """Representative backend and algorithm workflows still execute."""

    def test_statevector_backend_bell(self) -> None:
        from microquantum import QuantumCircuit, StatevectorBackend

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = StatevectorBackend().run(qc, shots=1024, seed=7)
        assert result.counts
        assert set(result.counts.keys()) <= {"00", "11"}

    def test_executor_bell(self) -> None:
        from microquantum import Executor, QuantumCircuit, StatevectorBackend

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = Executor(backend=StatevectorBackend()).run(qc, shots=512, seed=3)
        assert result.counts
        assert result.most_frequent() in {"00", "11"}

    def test_grover_algorithm(self) -> None:
        from microquantum import GroverSearch

        search = GroverSearch(num_qubits=3, target=5)
        result = search.run()
        assert result.target == 5
        assert result.success_probability > 0.5


class TestProviderBoundary:
    """HardwareProvider is an abstract extension boundary."""

    def test_provider_is_abstract(self) -> None:
        from microquantum.providers import HardwareProvider

        assert hasattr(HardwareProvider, "__abstractmethods__")
        with pytest.raises(TypeError):
            HardwareProvider(credentials=None)  # type: ignore[arg-type]


class TestDomainIndependence:
    """The SDK package contains no domain-specific business terminology."""

    FORBIDDEN_TERMS = [
        "fraud",
        "quantsmind",
        "karkain",
        "microquantum_pro",
        "drug discovery",
        "proprietary",
        "business",
        "screening desk",
        "decision desk",
        "solver desk",
        "portfolio_selection",
        "hohmann_transfer",
        "aerospace",
        "scheduling_optimization",
    ]

    def test_no_domain_terminology_in_package_source(self) -> None:
        import microquantum

        package_root = Path(microquantum.__file__).parent
        offenders: list[str] = []
        for path in package_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for term in self.FORBIDDEN_TERMS:
                if term in text.lower():
                    offenders.append(f"{path.relative_to(package_root)}: {term!r}")

        assert not offenders, (
            "Domain/business terminology leaked into the public SDK:\n"
            + "\n".join(sorted(set(offenders)))
        )