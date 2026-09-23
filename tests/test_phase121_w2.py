"""Phase 121 W2: compilation and execution extensions."""

from __future__ import annotations

import pytest

from microquantum.backends import (
    AsyncJob,
    CalibrationData,
    IdentityReadoutMitigator,
    JobStatus,
    ReadoutMitigator,
    RetryPolicy,
    with_retry,
)
from microquantum.backends.capabilities import BackendCapabilities
from microquantum.core.circuit import QuantumCircuit
from microquantum.ir import (
    AliasAnalysis,
    Compiler,
    ConditionalBlock,
    CostModel,
    Gate,
    IRCircuit,
    Loop,
    Measurement,
    Switch,
    loop_from_dict,
    switch_from_dict,
    to_ir,
)
from microquantum.ir.control import Condition
from microquantum.runtime import (
    Budget,
    DAGScheduler,
    ExecutionPlan,
    ResultCache,
    plan_fingerprint,
)


def _bell_ir() -> IRCircuit:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return to_ir(circuit)


class TestControlNodes:
    def test_loop_construction_and_unrolling(self) -> None:
        body = (Gate(name="h", qubits=(0,)), Gate(name="x", qubits=(1,)))
        loop = Loop(body=body, trip_count=3)
        assert loop.kind == "loop"
        assert loop.qubits == (0, 1)
        assert len(loop.unrolled()) == 6
        assert Loop(body=body, trip_count=0).unrolled() == ()
        with pytest.raises(ValueError):
            Loop(body=body, trip_count=-1)
        with pytest.raises(TypeError):
            Loop(body=body, trip_count=1.5)  # type: ignore[arg-type]

    def test_symbolic_trip_count(self) -> None:
        from microquantum.core.parameter import Parameter

        loop = Loop(body=(Gate(name="rx", qubits=(0,), params=(Parameter("t"),)),), trip_count=Parameter("n"))
        assert len(loop.parameters) == 2
        with pytest.raises(ValueError):
            loop.unrolled()

    def test_switch_construction(self) -> None:
        switch = Switch(
            condition=Condition(bit=0, value=1),
            cases=((1, (Gate(name="x", qubits=(0,)),)),),
            default=(Gate(name="h", qubits=(0,)),),
        )
        assert switch.kind == "switch"
        assert switch.qubits == (0,)
        assert switch.cbits == (0,)
        assert switch.to_dict()["cases"][0]["value"] == 1
        with pytest.raises(ValueError):
            Switch(condition=Condition(bit=0), cases=((2, ()),))

    def test_dict_round_trips(self) -> None:
        loop = Loop(body=(Gate(name="h", qubits=(0,)),), trip_count=2)
        rebuilt = loop_from_dict(loop.to_dict())
        assert rebuilt.trip_count == 2
        assert rebuilt.body[0].name == "h"  # type: ignore[union-attr]
        switch = Switch(
            condition=Condition(bit=1, value=0),
            cases=((0, (Measurement(qubit=0, classical=1),)),),
        )
        rebuilt_switch = switch_from_dict(switch.to_dict())
        assert rebuilt_switch.condition.bit == 1
        assert rebuilt_switch.cases[0][0] == 0
        with pytest.raises(ValueError):
            loop_from_dict({"kind": "switch"})
        with pytest.raises(ValueError):
            switch_from_dict({"kind": "loop"})
        with pytest.raises(ValueError):
            loop_from_dict({"kind": "loop", "body": [{"kind": "nope"}]})

    def test_symbolic_loop_round_trip(self) -> None:
        from microquantum.core.parameter import Parameter

        loop = Loop(body=(), trip_count=Parameter("rounds"))
        rebuilt = loop_from_dict(loop.to_dict())
        assert rebuilt.trip_count == Parameter("rounds")


class TestControlValidationAndTraversal:
    def test_validation_accepts_control_nodes(self) -> None:
        from microquantum.ir import assert_valid

        circuit = IRCircuit(num_qubits=2)
        circuit.add(Loop(body=(Gate(name="h", qubits=(0,)),), trip_count=2))
        circuit.add(
            Switch(
                condition=Condition(bit=0, value=1),
                cases=((1, (Gate(name="x", qubits=(1,)),)),),
            )
        )
        assert_valid(circuit)
        assert circuit.num_gates == 2

    def test_validation_rejects_bad_nesting(self) -> None:
        from microquantum.ir import validate

        circuit = IRCircuit(num_qubits=1)
        circuit.add(Loop(body=(Gate(name="h", qubits=(5,)),), trip_count=1))
        errors = validate(circuit)
        assert any("loop" in error for error in errors)
        bad_switch = IRCircuit(num_qubits=1)
        bad_switch.add(Switch(condition=Condition(bit=9), cases=((1, ()),)))
        assert any("classical bit" in error for error in validate(bad_switch))

    def test_walk_descends_into_control_nodes(self) -> None:
        circuit = IRCircuit(num_qubits=2)
        circuit.add(Loop(body=(Gate(name="h", qubits=(0,)),), trip_count=2))
        circuit.add(ConditionalBlock(condition=Condition(bit=0), operations=(Gate(name="x", qubits=(1,)),)))
        kinds = [op.kind for op in circuit.walk()]
        assert kinds == ["loop", "gate", "conditional", "gate"]


class TestCostModel:
    def test_total_and_breakdown(self) -> None:
        model = CostModel(gate_costs={"h": 1.0, "cnot": 10.0})
        assert model.cost_of("H") == pytest.approx(1.0)
        assert model.cost_of("x") == pytest.approx(1.0)
        ir = _bell_ir()
        assert model.total(ir) == pytest.approx(11.0)
        assert model.breakdown(ir) == {"cnot": 10.0, "h": 1.0}
        assert CostModel.from_dict(model.to_dict()).default_cost == pytest.approx(1.0)
        with pytest.raises(ValueError):
            CostModel(default_cost=-1.0)
        with pytest.raises(ValueError):
            CostModel(gate_costs={"h": -2.0})

    def test_compiler_records_cost(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        plain = Compiler().compile(circuit)
        assert "estimated_cost" not in plain.metadata
        priced = Compiler(cost_model=CostModel(gate_costs={"h": 2.0, "cnot": 5.0})).compile(circuit)
        assert priced.metadata["estimated_cost"] == pytest.approx(7.0)
        assert priced.metadata["cost_breakdown"] == {"cnot": 5.0, "h": 2.0}
        assert Compiler(cost_model=CostModel()).cost_model is not None
        assert Compiler().cost_model is None


class TestAliasAnalysis:
    def test_interaction_pairs(self) -> None:
        analysis = AliasAnalysis()
        assert analysis.name == "alias-analysis"
        ir = _bell_ir()
        assert analysis.run(ir) is ir
        assert analysis.result() == {0: {1}, 1: {0}}
        assert analysis.interacting_pairs() == [(0, 1)]
        assert analysis.to_dict() == {"0": [1], "1": [0]}

    def test_empty_circuit(self) -> None:
        analysis = AliasAnalysis()
        analysis.run(IRCircuit(num_qubits=2))
        assert analysis.result() == {}
        assert analysis.interacting_pairs() == []


class TestBackendsW2:
    def test_calibration_data(self) -> None:
        calibration = CalibrationData(
            gate_errors={"cx": 0.01},
            readout_errors={"q0": 0.02},
            t1_us={"q0": 100.0},
            timestamp="2026-01-01T00:00:00",
        )
        assert CalibrationData.from_dict(calibration.to_dict()).gate_errors == {"cx": 0.01}
        capabilities = BackendCapabilities(calibration=calibration, max_circuit_depth=100)
        rebuilt = BackendCapabilities.from_dict(capabilities.to_dict())
        assert rebuilt.max_circuit_depth == 100
        assert rebuilt.calibration is not None
        assert rebuilt.calibration.readout_errors == {"q0": 0.02}
        assert BackendCapabilities().to_dict()["calibration"] is None
        with pytest.raises(ValueError):
            CalibrationData(gate_errors={"x": 1.5})
        with pytest.raises(ValueError):
            CalibrationData(t1_us={"q0": -1.0})

    def test_async_job_polling(self) -> None:
        job = AsyncJob()
        assert job.status == JobStatus.PENDING
        assert job.poll(lambda: JobStatus.RUNNING) == JobStatus.RUNNING
        assert job.poll(lambda: JobStatus.COMPLETED) == JobStatus.COMPLETED
        assert job.finished_at is not None
        with pytest.raises(ValueError):
            AsyncJob(max_polls=0)

    def test_async_job_wait_timeout(self) -> None:
        job = AsyncJob(poll_interval_s=0.0)
        with pytest.raises(TimeoutError):
            job.wait(lambda: JobStatus.RUNNING, timeout=0.0)
        done = AsyncJob(poll_interval_s=0.0)
        assert done.wait(lambda: JobStatus.COMPLETED, timeout=5.0) == JobStatus.COMPLETED
        with pytest.raises(ValueError):
            done.wait(lambda: JobStatus.COMPLETED, timeout=-1.0)

    def test_retry_policy(self) -> None:
        attempts = 0

        def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("down")
            return "up"

        policy = RetryPolicy(max_attempts=3, backoff_s=0.0)
        assert with_retry(policy, flaky) == "up"
        assert attempts == 3
        with pytest.raises(RuntimeError):
            with_retry(policy, lambda: (_ for _ in ()).throw(ConnectionError("down")))
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_readout_mitigator(self) -> None:
        mitigator = IdentityReadoutMitigator()
        assert mitigator.name == "identity"
        assert mitigator.mitigate({"00": 70, "11": 30}) == {"00": 0.7, "11": 0.3}
        assert mitigator.mitigate({}) == {}
        assert issubclass(IdentityReadoutMitigator, ReadoutMitigator)
        with pytest.raises(NotImplementedError):
            ReadoutMitigator().mitigate({"0": 1})


class TestRuntimeW2:
    def _plans(self, count: int) -> list[ExecutionPlan]:
        return [ExecutionPlan.from_circuit(QuantumCircuit(1), shots=10) for _ in range(count)]

    def test_dag_levels_and_batches(self) -> None:
        scheduler = DAGScheduler(max_parallel=2)
        plans = self._plans(5)
        batches = scheduler.schedule(plans, dependencies={2: {0, 1}, 4: {2}})
        assert scheduler.depth(plans, dependencies={2: {0, 1}, 4: {2}}) == 3
        assert [batch.level for batch in batches] == [0, 0, 1, 2]
        assert sum(len(batch) for batch in batches) == 5
        assert scheduler.batch_shots(plans, batches[0]) == 20
        rebuilt_scheduler, rebuilt = DAGScheduler.from_dict(scheduler.to_dict(batches))
        assert rebuilt_scheduler.max_parallel == 2
        assert [batch.indices for batch in rebuilt] == [batch.indices for batch in batches]

    def test_dag_validation(self) -> None:
        scheduler = DAGScheduler()
        plans = self._plans(2)
        assert scheduler.depth([]) == 0
        with pytest.raises(ValueError):
            DAGScheduler(max_parallel=0)
        with pytest.raises(ValueError):
            scheduler.schedule(plans, dependencies={0: {0}})
        with pytest.raises(ValueError):
            scheduler.schedule(plans, dependencies={0: {1}, 1: {0}})
        with pytest.raises(ValueError):
            scheduler.schedule(plans, dependencies={7: set()})
        assert repr(scheduler) == "DAGScheduler(max_parallel=4)"

    def test_result_cache(self) -> None:
        cache = ResultCache(max_entries=2)
        plans = [
            ExecutionPlan.from_circuit(QuantumCircuit(1), shots=10),
            ExecutionPlan.from_circuit(QuantumCircuit(1), shots=11),
            ExecutionPlan.from_circuit(QuantumCircuit(1), shots=12),
        ]
        assert cache.get(plans[0]) is None
        key = cache.put(plans[0], {"value": 1})
        assert key == plan_fingerprint(plans[0])
        assert cache.get(plans[0]) == {"value": 1}
        assert plans[0] in cache
        # Identical work maps to one entry (content addressed).
        duplicate = ExecutionPlan.from_circuit(QuantumCircuit(1), shots=10)
        cache.put(duplicate, {"value": 9})
        assert len(cache) == 1
        assert cache.get(plans[0]) == {"value": 9}
        cache.put(plans[1], 2)
        cache.put(plans[2], 3)
        assert len(cache) == 2
        assert cache.get(plans[0]) is None
        cache.clear()
        assert len(cache) == 0
        with pytest.raises(ValueError):
            ResultCache(max_entries=0)
        with pytest.raises(TypeError):
            plan_fingerprint(object())

    def test_non_cacheable_plans_bypass(self) -> None:
        cache = ResultCache()
        plan = ExecutionPlan.from_circuit(QuantumCircuit(1), cacheable=False)
        cache.put(plan, {"value": 1})
        assert len(cache) == 0
        assert cache.get(plan) is None

    def test_budget_enforcement(self) -> None:
        budget = Budget(max_shots=100)
        budget.check(shots=50, circuits=1)
