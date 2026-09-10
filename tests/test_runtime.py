"""Hybrid execution runtime tests (MQ-04).

Covers the public execution surface: ExecutionPlan construction &
validation, ExecutionRuntime prepare/compile/submit/execute, Job lifecycle
metadata, parameter binding + sweeps, batch execution, classical-quantum
hybrid workflows, error handling and regression against the existing
backend/result contracts.
"""

import json

import pytest

import microquantum
from microquantum import (
    BackendResult,
    Compiler,
    ExecutionPlan,
    ExecutionRuntime,
    ExecutionStrategy,
    ExecutionTrace,
    Job,
    JobStatus,
    Parameter,
    QuantumCircuit,
    StateVector,
    StatevectorBackend,
    Target,
    execute,
    execute_batch,
    run_hybrid,
    run_parameter_sweep,
    submit,
    submit_batch,
)


def bell(theta: Parameter) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)
    qc.rx(theta, 0)
    return qc


def bound_bell(value: float) -> QuantumCircuit:
    return bell(Parameter("theta")).bind_parameters({"theta": value})


# ----------------------------------------------------------------------
# ExecutionPlan
# ----------------------------------------------------------------------


class TestExecutionPlan:
    def test_exactly_one_of_circuit_ir_compiled(self):
        with pytest.raises(ValueError):
            ExecutionPlan(circuit=QuantumCircuit(1), ir=None, compiled=object())
        with pytest.raises(ValueError):
            ExecutionPlan()
        ExecutionPlan(circuit=QuantumCircuit(1))

    def test_shots_validated(self):
        with pytest.raises(ValueError):
            ExecutionPlan(circuit=QuantumCircuit(1), shots=0)
        with pytest.raises(ValueError):
            ExecutionPlan(circuit=QuantumCircuit(1), shots=-5)
        plan = ExecutionPlan(circuit=QuantumCircuit(1), shots=1)
        assert plan.shots == 1

    def test_optimization_level_range(self):
        with pytest.raises(ValueError):
            ExecutionPlan(circuit=QuantumCircuit(1), optimization_level=3)
        with pytest.raises(ValueError):
            ExecutionPlan(circuit=QuantumCircuit(1), optimization_level=-1)

    def test_from_circuit_preserves_fields(self):
        plan = ExecutionPlan.from_circuit(
            bell(Parameter("theta")),
            name="p",
            shots=64,
            seed=3,
            parameter_bindings={"theta": 0.5},
            optimization_level=1,
            metadata={"kind": "demo"},
        )
        assert plan.name == "p"
        assert plan.shots == 64
        assert plan.seed == 3
        assert plan.parameter_bindings == {"theta": 0.5}
        assert plan.optimization_level == 1
        assert plan.metadata == {"kind": "demo"}

    def test_parameters_property(self):
        theta = Parameter("angle")
        plan = ExecutionPlan.from_circuit(bell(theta))
        names = {p.name for p in plan.parameters}
        assert names == {"angle"}

    def test_validate_unbound(self):
        plan = ExecutionPlan.from_circuit(bell(Parameter("theta")))
        problems = plan.validate()
        assert any("unbound parameters" in p for p in problems)

    def test_validate_unknown_binding(self):
        plan = ExecutionPlan.from_circuit(
            bound_bell(0.3), parameter_bindings={"nope": 1.0}
        )
        problems = plan.validate()
        assert any("nope" in p for p in problems)

    def test_validate_non_numeric_binding(self):
        plan = ExecutionPlan.from_circuit(
            bell(Parameter("theta")), parameter_bindings={"theta": "foo"}
        )
        problems = plan.validate()
        assert any("must be numeric" in p for p in problems)

    def test_bound_produces_static_circuit(self):
        plan = ExecutionPlan.from_circuit(
            bell(Parameter("theta")), parameter_bindings={"theta": 0.7}
        )
        circuit = plan.bound()
        assert not circuit.is_parameterized

    def test_bound_missing_binding_raises(self):
        plan = ExecutionPlan.from_circuit(bell(Parameter("theta")))
        with pytest.raises(ValueError):
            plan.bound()

    def test_serialization_round_trip(self):
        plan = ExecutionPlan.from_circuit(
            bound_bell(0.4), name="ser", shots=42, seed=9, metadata={"a": 1}
        )
        data = plan.to_dict()
        assert data["name"] == "ser"
        assert data["shots"] == 42
        payload = plan.to_json()
        assert json.loads(payload)["shots"] == 42

    def test_repr(self):
        plan = ExecutionPlan.from_circuit(QuantumCircuit(1))
        assert "ExecutionPlan" in repr(plan)
        assert "shots=1024" in repr(plan)


# ----------------------------------------------------------------------
# ExecutionRuntime basics
# ----------------------------------------------------------------------


class TestExecutionRuntime:
    def test_prepare_validates(self):
        rt = ExecutionRuntime()
        good = ExecutionPlan.from_circuit(QuantumCircuit(1))
        assert rt.prepare(good) is good
        bad = ExecutionPlan.from_circuit(bell(Parameter("theta")))
        with pytest.raises(ValueError):
            rt.prepare(bad)
        with pytest.raises(TypeError):
            rt.prepare(object())  # type: ignore[arg-type]

    def test_default_backend_is_simulator(self):
        rt = ExecutionRuntime()
        assert rt.default_backend.name == "statevector"

    def test_history_records_and_clears(self):
        rt = ExecutionRuntime()
        rt.execute(QuantumCircuit(1), shots=8)
        assert len(rt.history) == 1
        entry = rt.history[0]
        assert entry["status"] == "completed"
        assert entry["shots"] == 8
        assert "trace" in entry
        rt.clear_history()
        assert rt.history == []

    def test_history_size_capped(self):
        rt = ExecutionRuntime(history_size=3)
        for _ in range(6):
            rt.execute(QuantumCircuit(1), shots=8)
        assert len(rt.history) == 3

    def test_execute_lambda_runs_direct(self):
        result = microquantum.execute(QuantumCircuit(1), shots=64, seed=1)
        assert isinstance(result, BackendResult)
        assert result.metadata["strategy"] == "direct"
        assert result.metadata["backend"] == "statevector"
        assert result.metadata["shots"] == 64
        assert result.metadata["job_id"]
        assert "trace" in result.metadata
        assert "target" in result.metadata

    def test_execute_plan_path_matches_circuit_path(self):
        qc = QuantumCircuit(1)
        result_circuit = execute(qc, shots=256, seed=2)
        plan = ExecutionPlan.from_circuit(qc, shots=256, seed=2)
        result_plan = execute(plan)
        assert result_circuit.counts == result_plan.counts

    def test_custom_backend_selection(self):
        rt = ExecutionRuntime()
        backend = StatevectorBackend()
        result = rt.execute(QuantumCircuit(1), backend=backend, shots=32)
        assert result.metadata["backend"] == "statevector"

    def test_execute_rejects_bad_work(self):
        rt = ExecutionRuntime()
        with pytest.raises(TypeError):
            rt.execute("not a circuit")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Compilation integration
# ----------------------------------------------------------------------


_WIDE_BASIS = (
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "cnot",
    "cx",
    "cy",
    "ccx",
    "cz",
    "measure",
)


class TestCompilation:
    def test_compiled_strategy_with_optimization_level(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)  # adjacent identity pair cancelled at level 1
        qc.cnot(0, 1)
        plan = ExecutionPlan.from_circuit(qc, optimization_level=1, shots=64, seed=1)
        result = execute(plan)
        assert result.metadata["strategy"] == "compiled"

    def test_compiled_plan_reuse(self):
        qc = bound_bell(0.2)
        compiled = Compiler(optimization_level=1).compile(qc)
        plan = ExecutionPlan(compiled=compiled, shots=64, seed=3)
        result = execute(plan)
        assert result.metadata["strategy"] == "compiled"
        assert len(result.metadata["trace"]["events"]) >= 4

    def test_circuit_equal_after_compilation(self):
        theta = Parameter("theta")
        raw = bell(theta).bind_parameters({"theta": 1.7})
        rt = ExecutionRuntime()
        compiled = rt.compile(raw, optimization_level=1)
        assert compiled.is_compatible
        direct = rt.execute(bound_bell(1.7), shots=512, seed=11)
        optimized = rt.execute(
            ExecutionPlan.from_circuit(
                raw, optimization_level=1, shots=512, seed=11
            )
        )
        assert direct.most_frequent() == optimized.most_frequent()

    def test_target_pinned_plan(self):
        target = Target(name="mq04", num_qubits=3, max_shots=100000)
        plan = ExecutionPlan.from_circuit(
            bound_bell(0.5), target=target, shots=32, seed=1
        )
        result = execute(plan)
        assert result.metadata["strategy"] == "compiled"
        assert result.metadata["target"]["name"] == "mq04"

    def test_target_qubit_capacity_enforced(self):
        small = Target(name="tiny", num_qubits=1, native_gates=_WIDE_BASIS)
        big = QuantumCircuit(3)
        big.h(0)
        big.cnot(0, 1)
        plan = ExecutionPlan.from_circuit(big, target=small, shots=16)
        with pytest.raises(ValueError, match="qubits"):
            execute(plan)

    def test_target_shots_capacity_enforced(self):
        few = Target(name="few", max_shots=10)
        plan = ExecutionPlan.from_circuit(QuantumCircuit(1), target=few, shots=64)
        with pytest.raises(ValueError, match="shots"):
            execute(plan)

    def test_compile_raises_on_incompatible_target(self):
        strict = Target(name="strict", native_gates=("cz",))
        qc = QuantumCircuit(1)
        qc.h(0)  # 'h' not in native basis
        with pytest.raises(ValueError):
            execute(ExecutionPlan.from_circuit(qc, target=strict, shots=8))


# ----------------------------------------------------------------------
# Submission + Jobs
# ----------------------------------------------------------------------


class TestSubmission:
    def test_submit_returns_completed_job(self):
        job = submit(bound_bell(0.3), shots=16)
        assert isinstance(job, Job)
        assert job.status is JobStatus.COMPLETED
        assert job.result is not None
        assert job.backend_name == "statevector"
        assert job.target_name
        assert job.created_at
        assert job.started_at
        assert job.finished_at

    def test_submit_failed_job_has_error(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        plan = ExecutionPlan(circuit=qc, shots=8, initial_state=StateVector(1))
        job = submit(plan)
        assert job.status is JobStatus.FAILED
        assert job.error

    def test_job_metadata_contains_lifecycle(self):
        job = submit(bound_bell(0.9), shots=16)
        meta = job.metadata()
        assert meta["job_id"] == job.job_id
        assert meta["status"] == "completed"
        assert meta["backend_name"] == "statevector"
        assert "created_at" in meta
        assert "finished_at" in meta
        assert meta["info"]["strategy"] == "direct"

    def test_job_cancel_noop_after_complete(self):
        job = submit(bound_bell(0.1), shots=8)
        job.cancel()
        assert job.status is JobStatus.COMPLETED

    def test_runtime_submit_batch(self):
        jobs = submit_batch(
            [bound_bell(0.3), bound_bell(0.6)], shots=32, seed=5
        )
        assert len(jobs) == 2
        assert all(j.status is JobStatus.COMPLETED for j in jobs)

    def test_runtime_submit_batch_contains_failures(self):
        big = QuantumCircuit(3)
        big.h(0)
        big.cnot(0, 1)
        tiny = Target(name="tiny", num_qubits=1)
        plan_bad = ExecutionPlan.from_circuit(big, target=tiny, shots=8)
        jobs = submit_batch([bound_bell(0.3), plan_bad], shots=8)
        assert jobs[0].status is JobStatus.COMPLETED
        assert jobs[1].status is JobStatus.FAILED


# ----------------------------------------------------------------------
# Parameters + sweeps
# ----------------------------------------------------------------------


class TestParameterExecution:
    def test_execute_with_bindings(self):
        result = execute(
            bell(Parameter("theta")),
            shots=256,
            seed=4,
            parameter_bindings={"theta": 0.8},
        )
        assert result.metadata["parameter_bindings"] == {"theta": 0.8}
        assert result.metadata["strategy"] == "direct"

    def test_execute_unbound_raises(self):
        with pytest.raises(ValueError, match="unbound"):
            execute(bell(Parameter("theta")), shots=64)

    def test_partial_binding_raises_clear_error(self):
        qc = QuantumCircuit(1)
        qc.rx(Parameter("a"), 0)
        qc.ry(Parameter("b"), 0)
        with pytest.raises(ValueError, match="unbound"):
            execute(qc, shots=64, parameter_bindings={"a": 1.0})

    def test_sweep_raw_values(self):
        theta = Parameter("theta")
        qc = bell(theta)
        results = run_parameter_sweep(qc, [0.0, 1.57, 3.14], shots=512, seed=9)
        assert len(results) == 3
        bindings = [r.metadata["parameter_bindings"]["theta"] for r in results]
        assert bindings == [0.0, 1.57, 3.14]
        assert [r.metadata["sweep_index"] for r in results] == [0, 1, 2]

    def test_sweep_binding_dicts(self):
        qc = QuantumCircuit(1)
        qc.rx(Parameter("a"), 0)
        qc.ry(Parameter("b"), 0)
        results = run_parameter_sweep(
            qc,
            [{"a": 0.1, "b": 0.2}, {"a": 0.3, "b": 0.4}],
            shots=32,
            seed=1,
        )
        assert len(results) == 2
        assert results[1].metadata["parameter_bindings"] == {"a": 0.3, "b": 0.4}

    def test_sweep_reproducible(self):
        theta = Parameter("theta")
        qc = bell(theta)
        first = run_parameter_sweep(qc, [0.5, 1.5], shots=256, seed=12)
        second = run_parameter_sweep(qc, [0.5, 1.5], shots=256, seed=12)
        assert first[0].counts == second[0].counts
        assert first[1].counts == second[1].counts

    def test_sweep_raw_requires_single_param(self):
        qc = QuantumCircuit(1)
        qc.rx(Parameter("a"), 0)
        qc.ry(Parameter("b"), 0)
        with pytest.raises(ValueError, match="single free parameter"):
            run_parameter_sweep(qc, [0.1, 0.2], shots=16)

    def test_sweep_explicit_parameter_name(self):
        qc = QuantumCircuit(1)
        qc.rx(Parameter("a"), 0)
        qc.ry(Parameter("b"), 0)
        partial = qc.bind_parameters({"a": 0.5})
        results = run_parameter_sweep(
            partial, [0.5, 1.0], parameter_name="b", shots=16
        )
        assert [r.metadata["parameter_bindings"]["b"] for r in results] == [0.5, 1.0]


# ----------------------------------------------------------------------
# Batch execution
# ----------------------------------------------------------------------


class TestBatchExecution:
    def test_execute_batch_default_raises_on_error(self):
        with pytest.raises(ValueError):
            execute_batch(
                [bound_bell(0.1), bell(Parameter("theta"))], shots=16
            )

    def test_execute_batch_continues_on_error(self):
        results = execute_batch(
            [bound_bell(0.1), bell(Parameter("theta")), bound_bell(0.2)],
            shots=16,
            raise_on_error=False,
        )
        assert isinstance(results[0], BackendResult)
        assert isinstance(results[1], str)
        assert "unbound" in results[1]
        assert isinstance(results[2], BackendResult)

    def test_execute_batch_metadata(self):
        results = execute_batch(
            [bound_bell(0.1), bound_bell(0.2)], shots=16, seed=1
        )
        for index, result in enumerate(results):
            assert result.metadata["batch_id"]
            assert result.metadata["batch_index"] == index

    def test_execute_batch_plan_inputs(self):
        plans = [
            ExecutionPlan.from_circuit(bound_bell(0.1), shots=16, seed=1),
            ExecutionPlan.from_circuit(bound_bell(0.2), shots=16, seed=2),
        ]
        results = execute_batch(plans)
        assert len(results) == 2


# ----------------------------------------------------------------------
# Hybrid classical-quantum workflow
# ----------------------------------------------------------------------


class TestHybrid:
    def test_run_hybrid_loop(self):
        theta = Parameter("theta")
        qc = bell(theta)
        visited = []

        def build(state, index):
            visited.append((state["count"], index))
            return qc.bind_parameters({"theta": float(index) * 0.5})

        def update(state, index, result):
            return {"count": state["count"] + 1}

        results = run_hybrid(
            4, build, update, {"count": 0}, shots=128, seed=3
        )
        assert len(results) == 4
        assert [r.metadata["hybrid_step"] for r in results] == [0, 1, 2, 3]
        assert [r.metadata["hybrid_rounds"] for r in results] == [4, 4, 4, 4]
        assert visited[0] == (0, 0)
        assert visited[3] == (3, 3)

    def test_hybrid_plan_builds(self):
        def build(state, index):
            return ExecutionPlan.from_circuit(
                bound_bell(0.3 + index * 0.1),
                shots=64,
                seed=2,
                metadata={"tag": f"step-{index}"},
            )

        def update(state, _index, _result):
            return state

        results = run_hybrid(2, build, update, None, shots=64, seed=2)
        assert [r.metadata["tag"] for r in results] == ["step-0", "step-1"]

    def test_hybrid_rejects_zero_iterations(self):
        with pytest.raises(ValueError):
            run_hybrid(0, lambda s, i: bound_bell(0.1), lambda s, i, r: s, None)

    def test_hybrid_is_generic(self):
        seen = []

        class State:
            pass

        def build(state, index):
            seen.append(state)
            return bound_bell(0.1)

        def update(state, _index, result):
            result.metadata["custom"] = True
            return state

        results = run_hybrid(2, build, update, State(), shots=8)
        assert len(results) == 2
        assert results[0].metadata["custom"] is True


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------


class TestErrors:
    def test_failed_job_raises_value_error(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        plan = ExecutionPlan(circuit=qc, shots=4, initial_state=StateVector(1))
        job = submit(plan)
        assert job.status is JobStatus.FAILED
        rt = ExecutionRuntime()
        # execute() surfaces the failure as a ValueError
        with pytest.raises(ValueError):
            rt.execute(plan)

    def test_runtime_requires_plan_or_circuit(self):
        with pytest.raises(ValueError):
            microquantum.execute(None)

    def test_execute_none(self):
        with pytest.raises(ValueError):
            execute()

    def test_plan_and_circuit_conflict(self):
        plan = ExecutionPlan.from_circuit(QuantumCircuit(1))
        with pytest.raises(ValueError):
            execute(QuantumCircuit(1), plan=plan)

    def test_submit_unknown_strategy_default_ok(self):
        # default handlers are installed for direct + compiled
        rt = ExecutionRuntime()
        assert ExecutionStrategy.DIRECT in rt.handlers
        assert ExecutionStrategy.COMPILED in rt.handlers


# ----------------------------------------------------------------------
# Regression against existing contracts
# ----------------------------------------------------------------------


class TestRegression:
    def test_backend_result_contract_unchanged(self):
        backend = StatevectorBackend()
        raw = backend.run(bound_bell(0.4), shots=128, seed=6)
        assert isinstance(raw, BackendResult)
        assert raw.num_qubits == 2
        assert raw.backend_name == "statevector"
        assert raw.counts

    def test_job_default_status_still_pending(self):
        job = Job()
        assert job.status is JobStatus.PENDING
        assert len(job.job_id) == 8

    def test_job_lifecycle_flags_present(self):
        for member in (JobStatus.CREATED, JobStatus.QUEUED):
            assert member.value
        assert JobStatus.COMPLETED.final
        assert not JobStatus.RUNNING.final

    def test_statevector_backend_submit_circuit_sets_lifecycle(self):
        backend = StatevectorBackend()
        job = backend.submit_circuit(bound_bell(0.2), shots=16)
        assert job.status is JobStatus.COMPLETED
        assert job.backend_name == "statevector"
        assert job.created_at is not None
        assert job.finished_at is not None

    def test_trace_events_ordered(self):
        result = execute(bound_bell(0.3), shots=16)
        names = [e["name"] for e in result.metadata["trace"]["events"]]
        assert "prepared" in names
        assert "validated" in names
        assert "submitted" in names

    def test_trace_object_api(self):
        trace = ExecutionTrace("t")
        trace.record("a", x=1)
        trace.record("b")
        assert len(trace.events) == 2
        assert trace.to_dict()["events"][0]["name"] == "a"
        assert trace.to_dict()["events"][0]["elapsed_seconds"] >= 0.0

    def test_result_serialization_includes_runtime_meta(self):
        result = execute(bound_bell(0.5), shots=32, seed=2)
        payload = json.loads(result.to_json())
        assert payload["metadata"]["runtime"] == "ExecutionRuntime"

    def test_top_level_exports(self):
        for name in (
            "ExecutionPlan",
            "ExecutionRuntime",
            "ExecutionStrategy",
            "ExecutionTrace",
            "execute",
            "execute_batch",
            "submit",
            "submit_batch",
            "run_parameter_sweep",
            "run_hybrid",
        ):
            assert hasattr(microquantum, name)