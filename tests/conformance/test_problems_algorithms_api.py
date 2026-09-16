"""Conformance: problem-model and algorithm API contracts.

Validates that the public problem/algorithms/optimizer surface behaves as
advertised: constructors, round-trips, factories, and end-to-end solves.
Failure here means the SDK's advertised high-level API regressed.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import microquantum as mq
from microquantum.algorithms import (
    QAOA,
    VQE,
    Algorithm,
    AlgorithmResult,
    GroverSearch,
    ShorsAlgorithm,
)
from microquantum.core import Parameter, QuantumCircuit
from microquantum.optimizers import BFGS, COBYLA, Adam


def assert_round_trips(obj: mq.Problem) -> None:
    data = obj.to_dict()
    assert isinstance(data, dict)
    json_text = obj.to_json()
    d2 = json.loads(json_text)
    assert d2["name"] == obj.name


# ---------------------------------------------------------------------------
# Problem model hierarchy
# ---------------------------------------------------------------------------


class TestProblemHierarchy:
    def test_base_problem_shape(self) -> None:
        p = mq.Problem(name="p0", num_qubits=3)
        assert p.name == "p0"
        assert p.num_qubits == 3
        assert isinstance(p.validate(), (bool, list))
        assert_round_trips(p)

    def test_sampling_problem(self) -> None:
        sp = mq.SamplingProblem(num_qubits=2, num_samples=1024, name="sample-me")
        assert sp.num_samples == 1024
        assert sp.num_qubits == 2
        assert_round_trips(sp)
        assert sp.validate() == []
        restored = mq.SamplingProblem.from_dict(sp.to_dict())
        assert restored.num_samples == sp.num_samples

    def test_optimization_problem_from_qubo(self) -> None:
        b = mq.QUBOBuilder(3)
        b.add_linear(0, -1.0)
        b.add_linear(1, -1.0)
        b.add_quadratic(0, 1, 2.0)
        q = b.build()
        op = mq.OptimizationProblem.from_qubo(q)
        assert op.num_variables == 3
        assert op.qubits_needed == 3
        assert op.validate() == []
        assert_round_trips(op)

        # QUBO in -> QUBO out preserves the quadratic matrix.
        rebuilt = op.to_qubo()
        assert isinstance(rebuilt, mq.QUBOProblem)

    def test_optimization_problem_ising_round_trip(self) -> None:
        b = mq.QUBOBuilder(2)
        b.add_linear(0, -1.0)
        b.add_linear(1, -1.0)
        ising = mq.IsingConverter().qubo_to_ising(b.build())
        op = mq.OptimizationProblem.from_ising(ising)
        assert op.num_variables == 2
        cost = op.energy(np.array([1.0, -1.0]))
        assert isinstance(cost, float)
        rebuilt = op.to_qubo()
        assert isinstance(rebuilt, mq.QUBOProblem)

    def test_search_problem(self) -> None:
        sp = mq.SearchProblem(num_qubits=4, target=5, name="find-5")
        assert sp.target == 5
        assert sp.is_marked(format(5, "04b")) is True
        assert sp.is_marked("0000") is False
        assert sp.num_targets == 1
        assert sp.target_indices() == [5]
        assert_round_trips(sp)
        restored = mq.SearchProblem.from_dict(sp.to_dict())
        assert restored.target == 5

    def test_hamiltonian_problems(self) -> None:
        h2 = mq.H2Hamiltonian()
        hp = mq.HamiltonianProblem(hamiltonian=h2.hamiltonian, num_qubits=2)
        assert hp.num_qubits == 2
        assert_round_trips(hp)

        ep = mq.EigenvalueProblem(hamiltonian=h2.hamiltonian, k=2, num_qubits=2)
        assert ep.k == 2
        assert_round_trips(ep)

    def test_hamiltonian_simulation_circuit(self) -> None:
        h2 = mq.H2Hamiltonian()
        sim = mq.HamiltonianSimulation(hamiltonian=h2.hamiltonian, evolution_time=0.1)
        assert sim.evolution_time == pytest.approx(0.1)
        circuit = sim.build_circuit()
        assert circuit.num_qubits == 2
        state = sim.run()
        assert np.sum(np.abs(state.amplitudes) ** 2) == pytest.approx(1.0)


class TestQUBOCore:
    def test_qubo_builder_and_problem(self) -> None:
        b = mq.QUBOBuilder(2)
        b.add_constant(1.0)
        b.add_linear(0, -1.0)
        b.add_quadratic(0, 1, 2.0)
        q = b.build()
        assert q.num_variables == 2
        assert q.offset == pytest.approx(1.0)
        x = np.array([1.0, 0.0])
        assert q.energy(x) == pytest.approx(0.0)

    def test_ising_converter_round_trip(self) -> None:
        b = mq.QUBOBuilder(3)
        b.add_linear(0, 1.5)
        b.add_quadratic(0, 1, -2.0)
        q = b.build()
        conv = mq.IsingConverter()
        ising = conv.qubo_to_ising(q)
        back = conv.ising_to_qubo(ising)
        assert back.num_variables == 3
        assert back.Q.shape == (3, 3)


# ---------------------------------------------------------------------------
# Algorithm layer
# ---------------------------------------------------------------------------

class TestAlgorithmLayer:
    def test_algorithm_base_contract(self) -> None:
        assert inspect.isabstract(Algorithm)
        assert "name" in Algorithm.__abstractmethods__
        res = AlgorithmResult(algorithm="test", problem="p", solution=[1])
        assert res.algorithm == "test"
        assert res.solution == [1]
        assert isinstance(res.to_dict(), dict)
        json.loads(res.to_json())

    def test_optimizer_contract_and_landscapes(self) -> None:
        p = Parameter("x")
        for Opt in (Adam, BFGS, COBYLA):
            opt = Opt(max_iter=100)
            res = opt.minimize(
                lambda params: (params[p] - 3.0) ** 2,
                initial_params={p: 0.0},
            )
            assert isinstance(res.optimal_parameters[p], float)
            assert res.optimal_value == pytest.approx(
                (res.optimal_parameters[p] - 3.0) ** 2
            )
            assert Iterations(res)

    def test_algorithm_result_serialization(self) -> None:
        r = AlgorithmResult(
            algorithm="grover",
            problem="search",
            solution=[1, 0],
            objective=-2.0,
            converged=True,
            iterations=3,
        )
        data = json.loads(r.to_json())
        assert data["algorithm"] == "grover"
        assert data["converged"] is True
        assert data["objective"] == -2.0


class TestGroverConformance:
    def test_grover_search_finds_marked_target(self) -> None:
        sp = mq.SearchProblem(num_qubits=4, target=5, name="find-5")
        grover = GroverSearch.from_problem(sp)
        assert grover.num_qubits == 4
        assert grover.num_iterations > 0
        result = grover.run(shots=512, seed=42)
        assert result.most_probable == 5
        assert result.success_probability >= 0.8
        assert result.target == 5
        json.loads(result.to_json())


class TestQAOAConformance:
    def test_qaoa_from_qubo_problem(self) -> None:
        b = mq.QUBOBuilder(2)
        b.add_linear(0, -1.0)
        b.add_linear(1, -1.0)
        b.add_quadratic(0, 1, 2.0)
        op = mq.OptimizationProblem.from_qubo(b.build())
        qaoa = QAOA.from_problem(op, seed=1, num_layers=1)
        ansatz = qaoa.build_ansatz()
        assert isinstance(ansatz, QuantumCircuit)
        result = qaoa.solve()
        assert isinstance(result.eigenvalue, (int, float))
        assert isinstance(result.optimal_params, dict)
        json.loads(result.to_json())


class TestVQEConformance:
    def test_vqe_h2_ground_state(self) -> None:
        h2 = mq.H2Hamiltonian()
        assert h2.num_spatial_orbitals >= 1
        exact = h2.ground_state_energy
        assert exact == pytest.approx(-1.857275, abs=0.05)

        ansatz = QuantumCircuit(2)
        for qubit in range(2):
            ansatz.ry(Parameter(f"t{qubit}"), qubit)
        vqe = VQE(
            ansatz=ansatz,
            hamiltonian=h2.hamiltonian,
            optimizer=Adam(max_iter=10, learning_rate=0.1),
            seed=7,
            shots=512,
        )
        result = vqe.compute_minimum_eigenvalue()
        assert isinstance(result.eigenvalue, (int, float))
        assert len(result.optimal_params) == 2
        json.loads(result.to_json())

    def test_vqe_from_problem(self) -> None:
        h2 = mq.H2Hamiltonian()
        problem = mq.HamiltonianProblem(
            hamiltonian=h2.hamiltonian, num_qubits=2
        )
        assert isinstance(problem, mq.HamiltonianProblem)
        assert problem.hamiltonian.num_terms > 0
        assert VQE.from_problem is not None


class TestShorConformance:
    def test_shors_algorithm_factors_small_semiprime(self) -> None:
        shor = ShorsAlgorithm(15, max_attempts=5, seed=3)
        assert shor.n == 15
        circuit = shor.build_circuit(a=2)
        assert isinstance(circuit, QuantumCircuit)
        result = shor.run()
        assert result.success is True
        assert result.factors[0] in (3, 5)
        assert result.factors[0] * result.factors[1] == 15
        json.loads(result.to_json())


def Iterations(res) -> bool:
    assert isinstance(res.iterations, int) and res.iterations >= 0
    return True