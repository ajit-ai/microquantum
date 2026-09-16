"""Conformance: verification of the MQ-11 / MQ-12 / MQ-13 milestones.

Each milestone's user-visible contract is pinned here so a regression in any
shipped milestone fails loudly instead of passing silently.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import microquantum as mq
from microquantum.core import Parameter, QuantumCircuit

# ---------------------------------------------------------------------------
# MQ-11: Execution pipeline (plan -> job -> result, runtime, records)
# ---------------------------------------------------------------------------


class TestMQ11Execution:
    def test_plan_to_result_round_trip(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        plan = mq.ExecutionPlan(name="bell", circuit=qc, shots=1000, seed=42)
        runtime = mq.ExecutionRuntime()
        job = runtime.submit(plan)
        assert isinstance(job, mq.Job)
        result = job.result
        assert result.get_counts()["00"] + result.get_counts()["11"] == 1000
        assert result.shots == 1000

    def test_batch_execution(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        plans = [
            mq.ExecutionPlan(name=f"run-{i}", circuit=qc, shots=256, seed=i)
            for i in range(3)
        ]
        runtime = mq.ExecutionRuntime()
        results = runtime.execute_batch(plans)
        assert len(results) == 3
        for r in results:
            assert r.get_counts()["0"] + r.get_counts()["1"] == 256

    def test_submit_batch_jobs(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        plans = [
            mq.ExecutionPlan(name=f"job-{i}", circuit=qc, shots=64, seed=i)
            for i in range(2)
        ]
        runtime = mq.ExecutionRuntime()
        jobs = runtime.submit_batch(plans)
        assert all(isinstance(j, mq.Job) for j in jobs)
        for job in jobs:
            assert job.result.shots == 64

    def test_job_status_lifecycle(self) -> None:
        qc = QuantumCircuit(1)
        qc.x(0)
        plan = mq.ExecutionPlan(name="status", circuit=qc, shots=16)
        job = mq.ExecutionRuntime().submit(plan)
        assert job.status is not None
        assert isinstance(job.status, mq.JobStatus)

    def test_execution_record_serialization(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        runtime = mq.ExecutionRuntime()
        record = runtime.execute_record(
            mq.ExecutionPlan(name="rec", circuit=qc, shots=512, seed=7)
        )
        assert record.is_success is True
        assert record.shots == 512
        data = record.to_dict()
        assert data["status"] in ("completed", "success")
        assert isinstance(record.to_json(), str)

    def test_parameter_sweep_execution(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        runtime = mq.ExecutionRuntime()
        results = runtime.run_parameter_sweep(
            qc, [0.0, 1.0, 2.0], parameter_name="theta", shots=256, seed=1
        )
        assert len(results) == 3
        for r in results:
            assert r.shots == 256


# ---------------------------------------------------------------------------
# MQ-12: Algorithm & solver layer (problem in, validated solution out)
# ---------------------------------------------------------------------------


class TestMQ12Algorithms:
    def test_grover_solver_contract(self) -> None:
        problem = mq.SearchProblem(num_qubits=4, target=7, name="find-7")
        solver = mq.GroverSearch.from_problem(problem)
        result = solver.run(shots=256, seed=1)
        assert result.target == 7
        assert result.most_probable == 7
        assert result.success_probability >= 0.85
        assert isinstance(result.to_dict(), dict)

    def test_qaoa_solver_contract(self) -> None:
        b = mq.QUBOBuilder(2)
        b.add_linear(0, -2.0)
        b.add_linear(1, -2.0)
        b.add_quadratic(0, 1, 3.0)
        problem = mq.OptimizationProblem.from_qubo(b.build())
        solver = mq.QAOA.from_problem(problem, seed=2, num_layers=1)
        ansatz = solver.build_ansatz()
        assert isinstance(ansatz, QuantumCircuit)
        result = solver.solve()
        assert "eigenvalue" in result.to_dict()

    def test_vqe_solver_contract(self) -> None:
        h2 = mq.H2Hamiltonian()
        ansatz = QuantumCircuit(2)
        for q in range(2):
            ansatz.ry(Parameter(f"t{q}"), q)
        solver = mq.VQE(
            ansatz=ansatz,
            hamiltonian=h2.hamiltonian,
            optimizer=mq.Adam(max_iter=8, learning_rate=0.1),
            seed=5,
            shots=256,
        )
        result = solver.compute_minimum_eigenvalue()
        assert len(result.optimal_params) == 2
        json.loads(result.to_json())

    def test_alogorithm_result_unifies(self) -> None:
        r = mq.AlgorithmResult(
            algorithm="vqe",
            problem="h2",
            solution={"energy": -1.5},
            objective=-1.5,
            converged=False,
        )
        d = r.to_dict()
        assert d["algorithm"] == "vqe"
        assert d["objective"] == -1.5


# ---------------------------------------------------------------------------
# MQ-13: Analysis & statistics (records, experiments, sampling analysis)
# ---------------------------------------------------------------------------


class TestMQ13Analysis:
    def test_experiment_collects_records(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        exp = mq.Experiment(name="demo")
        exp.add_plan(mq.ExecutionPlan(name="p1", circuit=qc, shots=128, seed=3))
        exp.add_plan(mq.ExecutionPlan(name="p2", circuit=qc, shots=64, seed=4))
        result = exp.run(runtime=mq.ExecutionRuntime())
        assert isinstance(result, mq.ExperimentResult)
        assert result.success_count == 2
        assert len(result.executions) == 2

    def test_sampling_analysis_statistics(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        record = mq.ExecutionRuntime().execute_record(
            mq.ExecutionPlan(name="rand", circuit=qc, shots=2000, seed=9)
        )
        analysis = mq.SamplingAnalysis(record)
        probs = analysis.probabilities()
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
        assert analysis.total_shots() == 2000
        analysis.to_json()
        assert "entropy" in analysis.to_dict()

    def test_state_analysis(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        plan = mq.ExecutionPlan(name="sv", circuit=qc, shots=128, seed=1)
        record = mq.ExecutionRuntime().execute_record(plan)
        state = np.asarray(record.statevector, dtype=np.complex128)
        assert state.size > 0
        assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)

    def test_experiment_serialization(self) -> None:
        exp = mq.Experiment(name="serializable")
        exp.add_circuit(QuantumCircuit(1).x(0))
        data = exp.to_dict()
        assert data["name"] == "serializable"
        assert isinstance(exp.to_json(), str)