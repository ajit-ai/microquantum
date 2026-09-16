"""Conformance: serialization round-trips across the public API.

Every advertised serializer must round-trip: encode -> decode preserves the
semantic payload (counts, names, values), and every encoder emits JSON that
actually parses.  No values may be dropped or mutated by the round trip.
"""

from __future__ import annotations

import json

import numpy as np

import microquantum as mq
from microquantum.core import Parameter, QuantumCircuit
from microquantum.core.measurement import measure_qubits


def assert_json_clean(payload: str) -> dict:
    """Assert a to_json payload parses and returns the decoded dict."""
    data = json.loads(payload)
    assert isinstance(data, dict)
    return data


class TestProblemSerialization:
    def test_base_problem_round_trip(self) -> None:
        p = mq.Problem(name="round", num_qubits=3, metadata={"k": "v"})
        assert_json_clean(p.to_json())
        data = p.to_dict()
        assert data["name"] == "round"
        assert data["num_qubits"] == 3

    def test_sampling_problem_round_trip(self) -> None:
        sp = mq.SamplingProblem(num_qubits=2, num_samples=512, name="sp")
        restored = mq.SamplingProblem.from_dict(sp.to_dict())
        assert restored.num_qubits == 2
        assert restored.num_samples == 512
        assert_json_clean(sp.to_json())

    def test_optimization_problem_round_trip(self) -> None:
        b = mq.QUBOBuilder(2)
        b.add_linear(0, -1.0)
        b.add_quadratic(0, 1, 3.0)
        op = mq.OptimizationProblem.from_qubo(b.build())
        data = op.to_dict()
        assert data["num_variables"] == 2
        assert_json_clean(op.to_json())
        rebuilt = op.to_qubo()
        assert isinstance(rebuilt, mq.QUBOProblem)
        assert rebuilt.Q.shape == (2, 2)

    def test_search_problem_round_trip(self) -> None:
        sp = mq.SearchProblem(num_qubits=4, target=11, name="tap")
        restored = mq.SearchProblem.from_dict(sp.to_dict())
        assert restored.target == 11
        assert_json_clean(sp.to_json())

    def test_qubo_problem_to_dict(self) -> None:
        b = mq.QUBOBuilder(3)
        b.add_linear(1, 2.0)
        b.add_quadratic(0, 2, -4.0)
        q = b.build()
        data = q.to_dict()
        assert data["num_variables"] == 3
        assert np.asarray(data["Q"]).shape == (3, 3)


class TestResultSerialization:
    def test_algorithm_result_json(self) -> None:
        r = mq.AlgorithmResult(
            algorithm="grover",
            problem="search",
            solution=[1, 1, 0],
            objective=-1.0,
            iterations=3,
            converged=True,
        )
        data = assert_json_clean(r.to_json())
        assert data["algorithm"] == "grover"
        assert data["solution"] == [1, 1, 0]

    def test_backend_result_round_trip(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        res = mq.ExecutionRuntime().execute(
            mq.ExecutionPlan(name="br", circuit=qc, shots=512, seed=7)
        )
        restored = mq.BackendResult.from_dict(res.to_dict())
        assert restored.get_counts() == res.get_counts()
        assert restored.shots == 512
        assert restored.backend_name == res.backend_name
        assert_json_clean(res.to_json())

    def test_measurement_result_serializable(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        mr = measure_qubits(qc.run(), [0, 1], shots=200, seed=3)
        data = mr.to_dict()
        assert sum(data["counts"].values()) == 200
        assert_json_clean(mr.to_json())

    def test_execution_record_round_trip(self) -> None:
        qc = QuantumCircuit(1)
        qc.h(0)
        rec = mq.ExecutionRuntime().execute_record(
            mq.ExecutionPlan(name="rec-s", circuit=qc, shots=256, seed=2)
        )
        restored = mq.ExecutionRecord.from_dict(rec.to_dict())
        assert restored.shots == 256
        assert restored.is_success is True
        assert_json_clean(rec.to_json())

    def test_grover_result_json(self) -> None:
        from microquantum.algorithms import GroverSearch

        gr = GroverSearch(num_qubits=4, target=3).run(shots=128, seed=5)
        data = assert_json_clean(gr.to_json())
        assert data["most_probable"] == 3

    def test_shor_result_json(self) -> None:
        from microquantum.algorithms import ShorsAlgorithm

        sh = ShorsAlgorithm(15, max_attempts=3, seed=2).run()
        data = assert_json_clean(sh.to_json())
        assert data["success"] is True


class TestOptimizerSerialization:
    def test_optimizer_result_json(self) -> None:
        from microquantum.optimizers import Adam

        p = Parameter("w")
        res = Adam(max_iter=30).minimize(
            lambda params: (params[p] - 2.0) ** 2,
            initial_params={p: 0.0},
        )
        assert_json_clean(res.to_json())
        data = res.to_dict()
        assert data["iterations"] >= 0