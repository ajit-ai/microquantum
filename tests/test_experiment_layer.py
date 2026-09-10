"""MQ-07 tests: execution records, parameter sweeps, experiments, batch
execution, reproducibility, failures and serialization."""

from __future__ import annotations

import json

import pytest

from microquantum import (
    ExecutionFailure,
    ExecutionPlan,
    ExecutionRecord,
    ExecutionRuntime,
    Experiment,
    ExperimentResult,
    LocalSimulatorBackend,
    MockBackend,
    Parameter,
    ParameterSweep,
    QuantumCircuit,
    execute_records,
    execution_fingerprint,
    reproducibility_metadata,
)
from microquantum.experiments.record import ExecutionStatus


def _runtime(backend=None):
    return ExecutionRuntime(backend=backend or MockBackend())


class TestExecutionRecord:
    def test_creation_and_metadata(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        record = _runtime().execute_record(qc, shots=512, seed=9)
        assert record.is_success
        assert record.status is ExecutionStatus.COMPLETED
        assert record.execution_id
        assert record.timestamp
        assert record.backend == "mock"
        assert record.shots == 512
        assert record.seed == 9
        assert set(record.timing) == {"total_seconds", "queue_seconds", "execution_seconds"}
        assert record.reproducibility["fingerprint"]

    def test_parameter_bindings_captured(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        plan = ExecutionPlan.from_circuit(
            qc, name="bound", parameter_bindings={"theta": 0.5}
        )
        record = _runtime().execute_record(plan)
        assert record.parameter_bindings == {"theta": 0.5}
        assert record.plan_name == "bound"
        assert record.plan["parameter_bindings"] == {"theta": 0.5}

    def test_serialization_roundtrip(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        record = _runtime(backend=LocalSimulatorBackend()).execute_record(
            qc, shots=64, seed=4
        )
        data = record.to_dict()
        json.dumps(data)  # fully JSON-safe
        assert "statevector" in data["result"]
        restored = ExecutionRecord.from_dict(data)
        assert restored.execution_id == record.execution_id
        assert restored.status == record.status
        assert restored.backend == record.backend
        assert restored.shots == record.shots
        assert restored.seed == record.seed
        assert restored.parameter_bindings == record.parameter_bindings
        assert restored.counts == record.counts
        assert restored.reproducibility == record.reproducibility
        assert restored.result is not None
        assert restored.result.backend_name == "local_simulator"

    def test_no_live_objects_in_serialized_form(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        record = _runtime().execute_record(qc)
        data = record.to_dict()
        json.dumps(data)
        assert isinstance(data["backend"], str)
        assert isinstance(data["plan"]["backend"], str) or data["plan"]["backend"] is None

    def test_trace_metadata_without_trace_objects(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        record = _runtime().execute_record(qc)
        payload = json.loads(record.to_json())
        assert "trace" in payload["result"]["metadata"]
        assert isinstance(payload["result"]["metadata"]["trace"], dict)

    def test_failed_record_structured(self) -> None:
        qc = QuantumCircuit(1).ry(Parameter("theta"), 0)
        plan = ExecutionPlan.from_circuit(
            qc, name="unbound", parameter_bindings={"nope": 1.0}
        )
        record = _runtime().execute_record(plan)
        assert not record.is_success
        assert record.status is ExecutionStatus.FAILED
        assert record.error is not None
        assert record.error.error_type == "ValueError"
        assert record.error.execution_id == record.execution_id
        assert record.error.plan_name == "unbound"
        assert "nope" in record.error.message
        json.dumps(record.to_dict())

    def test_failed_non_work_item_still_recorded(self) -> None:
        records = _runtime().execute_records([42])  # type: ignore[list-item]
        assert len(records) == 1
        assert not records[0].is_success
        assert records[0].error is not None
        assert records[0].error.error_type == "TypeError"


class TestParameterSweep:
    def test_single_parameter_values(self) -> None:
        sweep = ParameterSweep({"theta": [0.0, 0.5, 1.0]})
        assert sweep.parameters == ("theta",)
        assert sweep.num_combinations == 3
        assert len(sweep) == 3
        assert sweep.combinations() == [
            {"theta": 0.0},
            {"theta": 0.5},
            {"theta": 1.0},
        ]

    def test_multiple_parameters_cartesian_product(self) -> None:
        sweep = ParameterSweep({"theta": [0.0, 1.0], "phi": [0.0, 0.5, 2.0]})
        assert sweep.parameters == ("theta", "phi")
        assert sweep.num_combinations == 6
        assert sweep.combinations() == [
            {"theta": 0.0, "phi": 0.0},
            {"theta": 0.0, "phi": 0.5},
            {"theta": 0.0, "phi": 2.0},
            {"theta": 1.0, "phi": 0.0},
            {"theta": 1.0, "phi": 0.5},
            {"theta": 1.0, "phi": 2.0},
        ]

    def test_generated_ranges(self) -> None:
        sweep = ParameterSweep(
            {
                "a": {"range": (0.0, 2.1, 1.0)},
                "b": {"start": 0.0, "stop": 1.0, "num_points": 3},
            }
        )
        assert sweep.values["a"] == [0.0, 1.0, 2.0]
        assert sweep.values["b"] == [0.0, 0.5, 1.0]
        assert sweep.num_combinations == 9

    def test_deterministic_ordering(self) -> None:
        a = ParameterSweep({"x": [1, 2], "y": [3, 4]})
        b = ParameterSweep({"x": [1, 2], "y": [3, 4]})
        assert a.combinations() == b.combinations()

    def test_verify_and_validate(self) -> None:
        sweep = ParameterSweep({"theta": [0.0, 1.0]})
        assert sweep.validate({"theta"}) == []
        problems = sweep.validate({"physics"})
        assert any("theta" in p for p in problems)
        with pytest.raises(ValueError, match="incompatible"):
            sweep.verify({"physics"})

    def test_invalid_definitions_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ParameterSweep({})
        with pytest.raises(ValueError, match="non-empty strings"):
            ParameterSweep({"": [0.0]})
        with pytest.raises(ValueError, match="no values"):
            ParameterSweep({"theta": []})
        with pytest.raises(ValueError, match="must be numeric"):
            ParameterSweep({"theta": ["x"]})  # type: ignore[list-item]
        with pytest.raises(ValueError, match="non-finite"):
            ParameterSweep({"theta": [1.0, float("nan")]})
        with pytest.raises(ValueError, match="step"):
            ParameterSweep({"theta": {"range": (0.0, 2.0, 0.0)}})
        with pytest.raises(ValueError, match="num_points"):
            ParameterSweep({"theta": {"start": 0.0, "stop": 1.0, "num_points": 1}})
        with pytest.raises(ValueError, match="not a"):
            ParameterSweep({"theta": {"bogus": 1}})

    def test_serialization_roundtrip(self) -> None:
        sweep = ParameterSweep({"theta": [0.0, 0.5], "phi": {"start": 0.0, "stop": 2.0, "num_points": 3}})
        data = sweep.to_dict()
        json.dumps(data)
        assert data["parameters"] == ["theta", "phi"]
        assert data["values"]["phi"] == [0.0, 1.0, 2.0]

    def test_iteration(self) -> None:
        sweep = ParameterSweep({"theta": [0.0, 1.0]})
        assert list(sweep) == sweep.combinations()


class TestBatchExecution:
    def test_ordering_and_bindings_preserved(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        plans = [
            ExecutionPlan.from_circuit(qc, name=f"p{i}", parameter_bindings={"theta": v})
            for i, v in enumerate([0.0, 0.5, 1.0])
        ]
        records = _runtime().execute_records(plans)
        assert [r.plan_name for r in records] == ["p0", "p1", "p2"]
        assert [r.parameter_bindings["theta"] for r in records] == [0.0, 0.5, 1.0]
        assert all(r.is_success for r in records)

    def test_partial_failure_not_dropped(self) -> None:
        qc = QuantumCircuit(1).ry(Parameter("theta"), 0)
        good = ExecutionPlan.from_circuit(qc, name="good", parameter_bindings={"theta": 0.3})
        bad = ExecutionPlan.from_circuit(qc, name="bad", parameter_bindings={"nope": 1.0})
        records = _runtime().execute_records([good, bad, good])
        assert len(records) == 3
        assert [r.status for r in records] == [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.COMPLETED,
        ]
        assert [r.plan_name for r in records] == ["good", "bad", "good"]
        assert records[1].error is not None
        assert records[1].error.plan_name == "bad"
        assert "nope" in records[1].error.message

    def test_backend_and_seed_propagation(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        records = _runtime(backend=LocalSimulatorBackend()).execute_records(
            [qc, qc], seed=5, shots=256
        )
        assert all(r.backend == "local_simulator" for r in records)
        assert all(r.shots == 256 for r in records)
        assert all(r.seed == 5 for r in records)

    def test_batch_metadata_marks_indices(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        records = _runtime().execute_records([qc, qc])
        assert records[0].metadata["plan"] == "main"
        assert records[0].metadata["batch_index"] == 0
        assert records[1].metadata["batch_index"] == 1
        assert "batch_id" in records[0].metadata

    def test_module_level_convenience(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        records = execute_records([qc, qc])
        assert len(records) == 2
        assert all(r.is_success for r in records)


class TestExperiment:
    def test_name_identity_and_status_lifecycle(self) -> None:
        exp = Experiment("demo", description="a demo", seed=3)
        assert exp.name == "demo"
        assert exp.description == "a demo"
        assert exp.status == "pending"
        assert exp.experiment_id
        with pytest.raises(ValueError, match="no executions"):
            exp.run(_runtime())

    def test_single_plan_experiment(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        exp = Experiment("one")
        exp.add_circuit(qc, name="h")
        result = exp.run(_runtime())
        assert exp.status == "completed"
        assert result.status == "completed"
        assert result.success_count == 1
        assert result.all_successful
        assert result.backend_names == ["mock"]
        assert len(result.records) == 1

    def test_sweep_experiment(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        exp = Experiment("sweep", shots=1024, seed=7)
        exp.add_sweep(ParameterSweep({"theta": [0.0, 2.0]}), base=qc)
        result = exp.run(_runtime())
        assert result.success_count == 2
        assert [r.parameter_bindings["theta"] for r in result.records] == [0.0, 2.0]
        assert result.records[0].metadata["sweep_index"] == 0
        assert result.records[1].metadata["sweep_name"] == "sweep"

    def test_partial_failure_sets_result_status(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        exp = Experiment("partial")
        exp.add_plan(ExecutionPlan.from_circuit(qc, name="good", parameter_bindings={"theta": 0.1}))
        exp.add_plan(ExecutionPlan.from_circuit(qc, name="bad", parameter_bindings={"oops": 1.0}))
        result = exp.run(_runtime())
        assert result.status == "failed"
        assert result.success_count == 1
        assert result.failure_count == 1
        assert not result.all_successful
        assert [f.plan_name for f in result.failures] == ["bad"]

    def test_add_plan_and_circuit_type_errors(self) -> None:
        qc = QuantumCircuit(1)
        exp = Experiment("t")
        with pytest.raises(TypeError, match="ExecutionPlan"):
            exp.add_plan(object())
        with pytest.raises(TypeError, match="QuantumCircuit"):
            exp.add_circuit(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ParameterSweep"):
            exp.add_sweep("nope", base=qc)  # type: ignore[arg-type]

    def test_sweep_without_base_raises(self) -> None:
        exp = Experiment("t")
        with pytest.raises(ValueError, match="base"):
            exp.add_sweep(ParameterSweep({"theta": [0.0]}))

    def test_sweep_incompatible_parameters_rejected(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        exp = Experiment("t")
        with pytest.raises(ValueError, match="incompatible"):
            exp.add_sweep(ParameterSweep({"phi": [0.0]}), base=qc)

    def test_result_serialization_roundtrip(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        exp = Experiment("rt")
        exp.add_circuit(qc, name="h")
        result = exp.run(_runtime())
        data = result.to_dict()
        json.loads(json.dumps(data))
        restored = ExperimentResult.from_dict(data)
        assert restored.experiment_id == result.experiment_id
        assert restored.status == result.status
        assert len(restored.records) == len(result.records)
        assert restored.records[0].counts == result.records[0].counts
        assert restored.success_count == result.success_count

    def test_run_experiment_via_runtime(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        exp = Experiment("via-runtime")
        exp.add_circuit(qc)
        result = _runtime().run_experiment(exp)
        assert isinstance(result, ExperimentResult)
        assert result.success_count == 1

    def test_experiment_config_serialization(self) -> None:
        exp = Experiment("cfg", shots=100, seed=2)
        data = exp.to_dict()
        json.dumps(data)
        assert data["name"] == "cfg"
        assert data["execution_count"] == 0


class TestReproducibility:
    def test_fingerprint_deterministic_and_sensitive(self) -> None:
        config = {"plan_name": "p", "backend": "b", "shots": 100, "seed": 3}
        assert execution_fingerprint(config) == execution_fingerprint(config)
        assert execution_fingerprint(config) != execution_fingerprint(
            {**config, "seed": 4}
        )

    def test_fingerprint_not_memory_based(self) -> None:
        base = {"plan_name": "p", "backend": "b", "parameter_bindings": {"theta": 0.5}}
        results = {execution_fingerprint(base) for _ in range(5)}
        assert len(results) == 1

    def test_same_configuration_same_fingerprint(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)

        def run(seed):
            runtime = _runtime()
            plan = ExecutionPlan.from_circuit(
                qc, name="rep", shots=256, seed=seed, parameter_bindings={"theta": 0.5}
            )
            return runtime.execute_record(plan)

        a, b = run(3), run(3)
        assert a.reproducibility["fingerprint"] == b.reproducibility["fingerprint"]
        assert a.counts == b.counts
        c = run(4)
        assert c.reproducibility["fingerprint"] != a.reproducibility["fingerprint"]

    def test_reproducibility_metadata_fields(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        record = _runtime().execute_record(qc, seed=1)
        meta = record.reproducibility
        assert meta["configured_reproducibility"] is True
        assert meta["deterministic_execution"] is None
        assert meta["sdk_version"]
        assert len(meta["fingerprint"]) == 64

    def test_hardware_not_claimed_deterministic(self) -> None:
        config = {"plan_name": "p", "backend": "hardware", "seed": None}
        meta = reproducibility_metadata(config)
        assert meta["deterministic_execution"] is None
        assert meta["configured_reproducibility"] is True


class TestFailureInspection:
    def test_execution_failure_serialization(self) -> None:
        failure = ExecutionFailure(
            execution_id="abc",
            backend="mock",
            plan_name="p",
            error_type="ValueError",
            message="boom",
            parameter_bindings={"theta": 1.0},
        )
        data = failure.to_dict()
        json.dumps(data)
        restored = ExecutionFailure.from_dict(data)
        assert restored.message == "boom"
        assert restored.parameter_bindings == {"theta": 1.0}

    def test_failures_inspectable_individually(self) -> None:
        qc = QuantumCircuit(1).ry(Parameter("theta"), 0)
        plans = [
            ExecutionPlan.from_circuit(qc, name=f"p{i}", parameter_bindings={"theta": 0.1})
            for i in range(3)
        ]
        plans[1] = ExecutionPlan.from_circuit(qc, name="p1", parameter_bindings={"nope": 1.0})
        records = _runtime().execute_records(plans)
        failed = [r for r in records if not r.is_success]
        assert len(failed) == 1
        err = failed[0].error
        assert err is not None
        assert err.execution_id == failed[0].execution_id
        assert err.backend == "mock"
        assert err.error_type == "ValueError"
        assert failed[0].plan_name == "p1"