"""MQ-07 integration: MQ-05 algorithms flowing through execution records,
experiments and the analysis layer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from microquantum import (
    BackendResult,
    ExecutionPlan,
    ExecutionRecord,
    ExecutionRuntime,
    Experiment,
    ExperimentResult,
    LocalSimulatorBackend,
    MockBackend,
    Operator,
    Parameter,
    ParameterSweep,
    QuantumCircuit,
    SamplingAnalysis,
    StateAnalysis,
)
from microquantum.algorithms import QAOA, VQE, GroverSearch, PhaseEstimation
from microquantum.optimization.qubo import QUBOBuilder
from microquantum.optimizers import Adam, GradientDescent
from microquantum.problems import EigenvalueProblem, OptimizationProblem, SearchProblem


def _local_runtime() -> ExecutionRuntime:
    return ExecutionRuntime(backend=LocalSimulatorBackend())


class TestRecordsFromAlgorithms:
    def test_grover_circuit_record_and_analysis(self) -> None:
        circuit = GroverSearch(num_qubits=3, target=5).build_circuit()
        record = _local_runtime().execute_record(circuit, shots=4096, seed=123)
        assert record.is_success
        assert record.backend == "local_simulator"
        assert SamplingAnalysis(record).most_likely() == "101"
        assert SamplingAnalysis(record).total_shots() == 4096
        stateview = StateAnalysis(record)
        assert stateview.kind == "statevector"
        assert stateview.most_probable_bitstring() == "101"

    def test_grover_record_survives_serialization_roundtrip(self) -> None:
        circuit = GroverSearch(num_qubits=3, target=5).build_circuit()
        record = _local_runtime().execute_record(circuit, shots=1024, seed=7)
        payload = json.loads(record.to_json())
        assert payload["status"] == "completed"
        restored = type(record).from_dict(payload)
        assert restored.counts == record.counts
        assert SamplingAnalysis(restored).most_likely() == "101"

    def test_algorithm_batch_ordering_and_mixed_status(self) -> None:
        good = GroverSearch(num_qubits=2, target=3).build_circuit()
        bad = QuantumCircuit(1).ry(Parameter("theta"), 0)
        bad_plan = ExecutionPlan.from_circuit(
            bad, name="bad", parameter_bindings={"missing": 0.5}
        )
        records = _local_runtime().execute_records([good, bad_plan])
        assert [r.status.value for r in records] == ["completed", "failed"]
        assert records[0].plan_name == "main"
        assert records[1].error is not None
        assert "missing" in records[1].error.message

    def test_vqe_output_feeds_analysis(self) -> None:
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)
        result = VQE(
            ansatz,
            Operator.Z(),
            GradientDescent(0.3, max_iter=60, tol=1e-6),
            runtime=_local_runtime(),
            seed=3,
        ).compute_minimum_eigenvalue(initial_params={theta: 0.5})
        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-3)
        bound = ansatz.bind_parameters(result.eigenstate)  # type: ignore[arg-type]
        record = _local_runtime().execute_record(bound, shots=4096, seed=1)
        assert record.is_success
        assert SamplingAnalysis(record).total_shots() == 4096

    def test_vqe_result_serializes(self) -> None:
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)
        result = VQE(
            ansatz,
            Operator.Z(),
            GradientDescent(0.3, max_iter=20, tol=1e-6),
        ).compute_minimum_eigenvalue()
        payload = json.loads(json.dumps(result.to_dict()))
        assert -1.0 - 1e-3 <= payload["eigenvalue"] <= 1.0

    def test_grover_result_serializes(self) -> None:
        result = GroverSearch(num_qubits=2, target=0).run(seed=5)
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["most_probable"] == 0
        assert payload["target"] == 0


class TestAlgorithmsThroughExperiments:
    def test_grover_experiment_run(self) -> None:
        exp = Experiment("grover", shots=4096, seed=42)
        exp.add_circuit(GroverSearch(num_qubits=2, target=1).build_circuit())
        result = exp.run(_local_runtime())
        assert isinstance(result, ExperimentResult)
        assert result.success_count == 1
        assert result.all_successful
        assert SamplingAnalysis(result.records[0]).most_likely() == "01"

    def test_algorithm_experiment_result_roundtrip(self) -> None:
        exp = Experiment("grover-sweep", shots=2048, seed=3)
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)
        exp.add_plan(
            ExecutionPlan.from_circuit(
                ansatz, name="ansatz", parameter_bindings={"theta": 0.5}
            )
        )
        exp.add_circuit(GroverSearch(num_qubits=2, target=3).build_circuit())
        result = exp.run(_local_runtime())
        data = result.to_dict()
        json.loads(json.dumps(data))
        restored = ExperimentResult.from_dict(data)
        assert restored.success_count == 2
        assert restored.records[1].plan_name == "plan"
        assert SamplingAnalysis(restored.records[0]).total_shots() > 0

    def test_parameter_sweep_of_algorithm_circuit(self) -> None:
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)
        exp = Experiment("sweep-ry", shots=1024, seed=11)
        exp.add_sweep(ParameterSweep({"theta": [0.0, np.pi]}), base=ansatz)
        result = exp.run(_local_runtime())
        assert result.success_count == 2
        samples = [SamplingAnalysis(r).most_likely() for r in result.records]
        assert samples == ["0", "1"]


class TestClassicProblemAlgorithmsWithRuntime:
    def test_qaoa_qubo_solve(self) -> None:
        builder = QUBOBuilder(num_variables=2)
        builder.add_quadratic(0, 1, 2.0)
        builder.add_linear(0, -1.0)
        builder.add_linear(1, -1.0)
        problem = OptimizationProblem.from_qubo(builder.build(), name="maxcut2")
        qaoa = QAOA.from_problem(
            problem,
            num_layers=1,
            optimizer=Adam(0.1, max_iter=80, tol=1e-6),
            runtime=_local_runtime(),
            seed=5,
        )
        result = qaoa.solve(problem)
        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-2)

    def test_phase_estimation_solve(self) -> None:
        problem = EigenvalueProblem(Operator.S())
        result = PhaseEstimation.from_problem(
            problem, num_counting_qubits=4, runtime=_local_runtime(), seed=9
        ).solve(problem)
        assert result.phase == pytest.approx(0.25)

    def test_grover_solve(self) -> None:
        problem = SearchProblem(num_qubits=3, target=5, name="find-5")
        result = GroverSearch.from_problem(problem).solve(
            problem, runtime=_local_runtime(), seed=7
        )
        assert result.most_probable == 5


class TestAnalysisComposition:
    def test_aggregate_over_algorithm_records(self) -> None:
        from microquantum import ResultAggregator

        records = [
            _local_runtime().execute_record(
                GroverSearch(num_qubits=2, target=1).build_circuit(), shots=512, seed=s
            )
            for s in (1, 2, 3)
        ]
        aggregator = ResultAggregator(records)
        groups = aggregator.group_by("metadata.plan")
        assert aggregator.record_count == 3
        assert set(groups) == {"main"}
        assert len(groups["main"]) == 3
        assert all(g.is_success for g in groups["main"])

    def test_expectation_analysis_consumes_aggregations(self) -> None:
        from microquantum import ExpectationAnalysis

        records = []
        for theta in (0.0, np.pi):
            qc = QuantumCircuit(1)
            qc.ry(Parameter("theta"), 0)
            plan = ExecutionPlan.from_circuit(
                qc, name="z", parameter_bindings={"theta": theta}
            )
            result = BackendResult(
                num_qubits=1,
                backend_name="mock",
                expectations={"Z": float(np.cos(theta))},
            )
            records.append(
                ExecutionRecord.completed(
                    plan, MockBackend(), result, 0.001  # type: ignore[arg-type]
                )
            )
        analysis = ExpectationAnalysis(records)
        assert analysis.mean("Z") == pytest.approx(0.0)
        mapping = analysis.parameter_to_expectation("theta", "Z")
        assert mapping[0.0] == pytest.approx(1.0)
        assert mapping[np.pi] == pytest.approx(-1.0)