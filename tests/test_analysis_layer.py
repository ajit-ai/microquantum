"""MQ-07 tests: statistics, sampling/expectation/state analysis and result
aggregation."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from microquantum import (
    BackendResult,
    ExecutionPlan,
    ExecutionRecord,
    ExecutionRuntime,
    ExpectationAnalysis,
    ExperimentResult,
    LocalSimulatorBackend,
    MockBackend,
    QuantumCircuit,
    ResultAggregator,
    SamplingAnalysis,
    StateAnalysis,
)
from microquantum.analysis.statistics import (
    confidence_interval,
    count,
    maximum,
    mean,
    minimum,
    standard_deviation,
    standard_error,
    variance,
)
from microquantum.core import StateVector


class TestStatistics:
    def test_mean(self) -> None:
        assert mean([2, 4, 6]) == 4.0
        assert mean(np.array([1.0, 2.0])) == 1.5

    def test_population_vs_sample_variance(self) -> None:
        data = [2.0, 4.0, 6.0]
        assert variance(data, ddof=0) == pytest.approx(8.0 / 3.0)
        assert variance(data, ddof=1) == pytest.approx(4.0)

    def test_standard_deviation(self) -> None:
        assert standard_deviation([2.0, 4.0, 6.0], ddof=1) == pytest.approx(2.0)

    def test_standard_error(self) -> None:
        assert standard_error([1.0, 2.0, 3.0]) == pytest.approx(1.0 / math.sqrt(3))

    def test_confidence_interval(self) -> None:
        low, high = confidence_interval([1.0] * 100, confidence=0.95)
        assert low == pytest.approx(1.0, abs=1e-3)
        assert high == pytest.approx(1.0, abs=1e-3)
        low, high = confidence_interval([0.0, 2.0, 4.0], z=1.0)
        assert low == pytest.approx(mean([0.0, 2.0, 4.0]) - standard_error([0.0, 2.0, 4.0]))
        with pytest.raises(ValueError, match="preset"):
            confidence_interval([0.0, 1.0, 2.0], confidence=0.88)

    def test_min_max_count(self) -> None:
        assert minimum([3, 1, 2]) == 1
        assert maximum([3, 1, 2]) == 3
        assert count([3, 1, 2]) == 3

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mean([])
        with pytest.raises(ValueError, match="not a real number"):
            mean(["a"])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="finite"):
            mean([1.0, float("inf")])


class _Counts:
    def __init__(self, counts):
        self.counts = counts


def _result(counts, **kwargs) -> BackendResult:
    return BackendResult(
        num_qubits=kwargs.get("num_qubits", 2),
        backend_name=kwargs.get("backend_name", "statevector"),
        counts=dict(counts),
        shots=kwargs.get("shots"),
        seed=kwargs.get("seed"),
        expectations=kwargs.get("expectations", {}),
    )


class TestSamplingAnalysis:
    def test_sources(self) -> None:
        result = _result({"00": 25, "01": 25, "11": 50}, shots=100)
        assert SamplingAnalysis(result).total_shots() == 100
        assert SamplingAnalysis(result.to_dict()).total_shots() == 100
        assert SamplingAnalysis({"00": 3, "11": 7}).total_shots() == 10
        assert SamplingAnalysis(_Counts({"00": 2})).total_shots() == 2
        with pytest.raises(TypeError, match="SamplingAnalysis"):
            SamplingAnalysis(object())

    def test_probabilities_and_unique(self) -> None:
        analysis = SamplingAnalysis({"00": 1, "01": 3, "11": 1})
        assert analysis.total_shots() == 5
        assert analysis.unique_outcomes() == ["00", "01", "11"]
        probs = analysis.probabilities()
        assert probs == {"00": 0.2, "01": 0.6, "11": 0.2}
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_most_likely(self) -> None:
        analysis = SamplingAnalysis({"000": 2, "111": 9, "010": 4})
        assert analysis.most_likely() == "111"
        assert analysis.most_likely_probability() == pytest.approx(9 / 15)
        with pytest.raises(ValueError, match="no sampling"):
            SamplingAnalysis({}).most_likely()

    def test_entropy(self) -> None:
        assert SamplingAnalysis({"0": 100}).entropy() == 0.0
        assert SamplingAnalysis({"0": 50, "1": 50}).entropy() == pytest.approx(1.0)
        uniform = SamplingAnalysis({"0": 1, "1": 1, "2": 1, "3": 1})
        assert uniform.entropy() == pytest.approx(2.0)
        assert uniform.entropy(base=math.e) == pytest.approx(math.log(4))
        with pytest.raises(ValueError):
            uniform.entropy(base=1)

    def test_mean_default_and_value_of(self) -> None:
        analysis = SamplingAnalysis({"00": 1, "10": 1})
        assert analysis.mean() == pytest.approx(1.0)  # (0 + 2) / 2
        assert analysis.mean(lambda b: int(b[0])) == pytest.approx(0.5)
        assert analysis.variance(lambda b: float(int(b, 2))) == pytest.approx(1.0)

    def test_marginal(self) -> None:
        analysis = SamplingAnalysis({"00": 30, "11": 70})
        assert analysis.marginal([0]) == {"0": 0.3, "1": 0.7}
        assert analysis.marginal([1]) == {"0": 0.3, "1": 0.7}
        assert analysis.marginal([0, 1]) == {"00": 0.3, "11": 0.7}
        with pytest.raises(ValueError, match="requires at least"):
            analysis.marginal([])
        with pytest.raises(ValueError, match="within 0.."):
            analysis.marginal([5])

    def test_expected_value(self) -> None:
        analysis = SamplingAnalysis({"0": 25, "1": 75})
        assert analysis.expected_value(lambda b: 0.0 if b == "0" else 1.0) == pytest.approx(0.75)

    def test_to_dict(self) -> None:
        data = SamplingAnalysis({"00": 3, "11": 7}).to_dict()
        json.dumps(data)
        assert data["type"] == "sampling"
        assert data["total_shots"] == 10
        assert data["entropy"] > 0


class TestExpectationAnalysis:
    def _records(self, expectations, theta=None):
        qc = QuantumCircuit(1).h(0)
        result = BackendResult(
            num_qubits=1, backend_name="mock", expectations=expectations
        )
        plan = ExecutionPlan.from_circuit(
            qc,
            name="e",
            parameter_bindings={"theta": theta} if theta is not None else None,
        )
        runtime = ExecutionRuntime(backend=MockBackend())
        return ExecutionRecord.completed(plan, runtime.default_backend, result, 0.001)

    def test_aggregation_stats(self) -> None:
        records = [
            self._records({"Z": 1.0, "X": 0.0}),
            self._records({"Z": -1.0}),
            self._records({"Z": 0.0, "X": 1.0}),
        ]
        analysis = ExpectationAnalysis(records)
        assert analysis.result_count == 3
        assert analysis.keys == ("X", "Z")
        assert analysis.mean("Z") == pytest.approx(0.0)
        assert analysis.standard_deviation("Z") == pytest.approx(math.sqrt(2 / 3))
        assert analysis.standard_error("Z") > 0
        assert analysis.minimum("Z") == -1.0
        assert analysis.maximum("Z") == 1.0
        with pytest.raises(KeyError, match="no expectation values"):
            analysis.values("Q")

    def test_from_experiment_result(self) -> None:
        records = [
            self._records({"Z": 1.0}),
            self._records({"Z": -1.0}),
        ]
        result = ExperimentResult(
            experiment_id="e1",
            name="e",
            status="completed",
            records=records,
            failures=[r.error for r in records if r.error is not None],
            backend_names=["mock"],
        )
        analysis = ExpectationAnalysis(result)
        assert analysis.result_count == 2
        assert analysis.mean("Z") == pytest.approx(0.0)

    def test_parameter_mapping(self) -> None:
        records = [
            self._records({"Z": 1.0}, theta=0.0),
            self._records({"Z": 0.5}, theta=0.5),
            self._records({"Z": 0.0}, theta=1.0),
            self._records({"Z": 0.75}, theta=0.5),  # repeat binding → averaged
        ]
        analysis = ExpectationAnalysis(records)
        mapping = analysis.parameter_to_expectation("theta", "Z")
        assert mapping[0.0] == pytest.approx(1.0)
        assert mapping[0.5] == pytest.approx(0.625)
        assert mapping[1.0] == pytest.approx(0.0)
        points = analysis.parameter_points("theta", "Z", include_error=True)
        assert [p[0] for p in points] == [0.0, 0.5, 1.0]
        _, _, err = points[1]
        assert err is not None and err > 0


class TestStateAnalysis:
    def test_statevector_from_record(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        record = ExecutionRuntime(backend=LocalSimulatorBackend()).execute_record(qc)
        analysis = StateAnalysis(record)
        assert analysis.kind == "statevector"
        assert analysis.dim == 2
        assert analysis.is_normalized()
        assert analysis.norm_squared() == pytest.approx(1.0)
        assert analysis.probabilities()["0"] == pytest.approx(0.5)
        assert analysis.most_probable_bitstring() in ("0", "1")

    def test_statevector_from_instances_and_dicts(self) -> None:
        sv = StateVector(1, amplitudes=np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
        analysis = StateAnalysis(sv)
        assert analysis.is_normalized()
        probs = analysis.probabilities()
        assert probs["0"] == pytest.approx(0.5)
        assert probs["1"] == pytest.approx(0.5)
        assert StateAnalysis({"statevector": [{"real": 1.0, "imag": 0.0}, {"real": 0.0, "imag": 0.0}]}).most_probable_state() == 0

    def test_statevector_expectation_diagonal(self) -> None:
        sv = StateVector(1, amplitudes=np.array([1.0, 0.0]))
        assert StateAnalysis(sv).expectation([1.0, -1.0]) == pytest.approx(1.0)
        assert StateAnalysis(sv).expectation(lambda i: float(i)) == pytest.approx(0.0)
        with pytest.raises(ValueError, match="diagonal observable"):
            StateAnalysis(sv).expectation([1.0, 2.0, 3.0])

    def test_density_matrix(self) -> None:
        rho = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
        analysis = StateAnalysis(rho)
        assert analysis.kind == "density_matrix"
        assert analysis.trace() == pytest.approx(1.0)
        assert analysis.purity() == pytest.approx(0.5)
        assert not analysis.is_pure()
        assert analysis.diagonal_probabilities() == {"0": 0.5, "1": 0.5}

    def test_unsupported_source_and_wrong_kind(self) -> None:
        with pytest.raises(TypeError, match="statevector or a density"):
            StateAnalysis(BackendResult(num_qubits=1, backend_name="mock"))
        rho = np.eye(2, dtype=np.complex128) / 2
        analysis = StateAnalysis(rho)
        with pytest.raises(ValueError, match="requires a statevector"):
            analysis.probabilities()


class TestResultAggregator:
    def _records(self):
        results = []
        for theta, backend, expectations in [
            (0.0, "mock", {"Z": 1.0}),
            (0.0, "mock", {"Z": 2.0}),
            (1.0, "statevector", {"Z": 0.0}),
        ]:
            qc = QuantumCircuit(1).h(0)
            runtime = ExecutionRuntime(
                backend=MockBackend() if backend == "mock" else LocalSimulatorBackend()
            )
            result = BackendResult(
                num_qubits=1,
                backend_name=backend,
                expectations=expectations,
                metadata={"plan": "a"},
            )
            plan = ExecutionPlan.from_circuit(
                qc, name="a", parameter_bindings={"theta": theta}
            )
            results.append(ExecutionRecord.completed(plan, runtime.default_backend, result, 0.001))
        return results

    def test_group_by_parameter_preserves_raw(self) -> None:
        records = self._records()
        aggregator = ResultAggregator(records)
        groups = aggregator.group_by_parameter("theta")
        assert set(groups) == {0.0, 1.0}
        assert len(groups[0.0]) == 2
        assert len(groups[1.0]) == 1
        # raw objects preserved (same identity)
        assert groups[0.0][0] in records
        assert aggregator.records == records

    def test_group_by_backend_status_paths(self) -> None:
        records = self._records()
        aggregator = ResultAggregator(records)
        assert set(aggregator.group_by_backend()) == {"local_simulator", "mock"}
        assert set(aggregator.group_by_status()) == {"completed"}
        assert aggregator.group_by("metadata.plan") == {"a": records}

    def test_group_by_callable_accessor(self) -> None:
        aggregator = ResultAggregator(self._records())
        groups = aggregator.group_by(lambda r: r.parameter_bindings)
        assert set(groups) == {"{'theta': 0.0}", "{'theta': 1.0}"}

    def test_mean_expectation_per_group(self) -> None:
        records = self._records()
        aggregator = ResultAggregator(records)
        groups = aggregator.group_by_parameter("theta")
        means = aggregator.mean_expectation(groups, "Z")
        assert means[0.0] == pytest.approx(1.5)
        assert means[1.0] == pytest.approx(0.0)
        assert aggregator.parameter_expectations("theta", "Z")[0.0] == pytest.approx(1.5)

    def test_expectation_keys(self) -> None:
        aggregator = ResultAggregator(self._records())
        assert aggregator.expectation_keys() == ("Z",)

    def test_group_counts_and_to_dict(self) -> None:
        aggregator = ResultAggregator(self._records())
        groups = aggregator.group_by_parameter("theta")
        assert aggregator.group_counts(groups) == {0.0: 2, 1.0: 1}
        data = aggregator.to_dict("backend")
        json.dumps(data)
        assert data["groups"]["mock"]["count"] == 2
        assert data["groups"]["mock"]["Z"]["mean"] == pytest.approx(1.5)

    def test_empty_and_invalid_source(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ResultAggregator([])
        with pytest.raises(TypeError, match="accessor"):
            ResultAggregator(self._records()).group_by(42)  # type: ignore[arg-type]