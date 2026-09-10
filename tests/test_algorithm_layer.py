"""Tests for the generic algorithm lifecycle and problem/algorithm
separation (MQ-05)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from microquantum.algorithms import Algorithm, AlgorithmResult
from microquantum.core import Operator, Parameter, QuantumCircuit
from microquantum.optimization.qubo import QUBOBuilder
from microquantum.optimizers import Adam
from microquantum.problems import (
    EigenvalueProblem,
    OptimizationProblem,
    SearchProblem,
)


class _DemoAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "demo"

    def solve(self, problem, runtime=None) -> AlgorithmResult:
        return AlgorithmResult(
            algorithm=self.name,
            problem=problem.name,
            solution={"found": True},
            objective=1.0,
            converged=True,
            iterations=1,
        )


class TestAlgorithmLifecycle:
    def test_base_is_abstract_on_name_only(self) -> None:
        with pytest.raises(TypeError):
            Algorithm()

    def test_subclass_needs_only_name(self) -> None:
        class MyAlgo(Algorithm):
            @property
            def name(self) -> str:
                return "my-algo"

        assert MyAlgo().name == "my-algo"
        # concrete hooks have safe defaults
        assert MyAlgo().validate("anything") == []
        with pytest.raises(NotImplementedError):
            MyAlgo().solve("problem")

    def test_generic_solve_lifecycle(self) -> None:
        algo = _DemoAlgorithm()
        result = algo.solve(EigenvalueProblem(Operator.Z()))
        assert isinstance(result, AlgorithmResult)
        assert result.algorithm == "demo"
        assert result.problem != ""
        assert result.objective == 1.0
        assert result.converged

    def test_algorithm_result_serializable(self) -> None:
        result = _DemoAlgorithm().solve(EigenvalueProblem(Operator.Z()))
        data = result.to_dict()
        assert "native" not in data
        json.loads(result.to_json())
        json.dumps(data)

    def test_algorithm_result_keeps_native(self) -> None:
        result = AlgorithmResult(algorithm="x", native={"custom": 7})
        assert result.native == {"custom": 7}


class TestProblemSeparation:
    def test_vqe_rejects_optimization_problem(self) -> None:
        from microquantum.algorithms import VQE

        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)
        vqe = VQE(ansatz, Operator.Z(), Adam(0.1, max_iter=5))
        p = OptimizationProblem(num_variables=1, objective=lambda b: float(b[0]))
        assert vqe.validate("not-a-problem") != []
        assert vqe.validate(p) != []

    def test_vqe_solve_uses_problem_hamiltonian(self) -> None:
        from microquantum.algorithms import VQE
        from microquantum.optimizers import GradientDescent

        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)
        vqe = VQE(ansatz, Operator.Z(), GradientDescent(0.3, max_iter=100, tol=1e-9))
        problem = EigenvalueProblem(Operator.Z(), k=1)
        result = vqe.solve(problem, initial_params={theta: 0.5})
        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-3)

    def test_vqe_from_problem_type_check(self) -> None:
        from microquantum.algorithms import VQE

        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(TypeError):
            VQE.from_problem(
                OptimizationProblem(num_variables=1, objective=lambda b: 0.0),
                ansatz,
            )

    def test_qaoa_rejects_non_optimization_problem(self) -> None:
        from microquantum.algorithms import QAOA

        qaoa = QAOA(Operator.Z(), num_qubits=1, num_layers=1)
        assert qaoa.validate(SearchProblem(num_qubits=2, target=0)) != []

    def test_qaoa_problem_without_ising_rejected(self) -> None:
        from microquantum.algorithms import QAOA

        p = OptimizationProblem(num_variables=2, objective=lambda b: float(b[0]))
        qaoa = QAOA(Operator.Z(), num_qubits=2, num_layers=1)
        assert qaoa.validate(p) != []

    def test_grover_rejects_wrong_problem_type(self) -> None:
        from microquantum.algorithms import GroverSearch

        g = GroverSearch(num_qubits=2, target=0)
        assert g.validate(OptimizationProblem(num_variables=2, objective=lambda b: 0.0)) != []


class TestFromProblemConstructors:
    def test_qaoa_from_problem_and_validate(self) -> None:
        from microquantum.algorithms import QAOA
        from microquantum.optimizers import GradientDescent

        builder = QUBOBuilder(num_variables=2)
        builder.add_quadratic(0, 1, 2.0)
        builder.add_linear(0, -1.0)
        builder.add_linear(1, -1.0)
        p = OptimizationProblem.from_qubo(builder.build(), name="maxcut2")

        qaoa = QAOA.from_problem(p, num_layers=1, optimizer=GradientDescent(0.1, max_iter=50))
        assert qaoa.validate(p) == []
        ansatz = qaoa.build_ansatz()
        assert ansatz.num_qubits == 2
        assert ansatz.is_parameterized

    def test_grover_from_problem(self) -> None:
        from microquantum.algorithms import GroverSearch

        p = SearchProblem(num_qubits=3, target=5, name="find-5")
        g = GroverSearch.from_problem(p)
        assert g.targets == [5]
        result = g.solve(p)
        assert result.most_probable == 5
        assert result.success_probability > 0.9

    def test_phase_estimation_from_problem(self) -> None:
        from microquantum.algorithms import PhaseEstimation

        p = EigenvalueProblem(Operator.S())
        pe = PhaseEstimation.from_problem(p, num_counting_qubits=4)
        result = pe.solve(p)
        assert result.phase == pytest.approx(0.25)

    def test_phase_estimation_rejects_non_unitary(self) -> None:
        from microquantum.algorithms import PhaseEstimation

        h = Operator(np.array([[1.0, 0.5], [0.5, 1.0]], dtype=complex))
        p = EigenvalueProblem(h)
        # from_problem validates the Hamiltonian must be a unitary Operator
        with pytest.raises(ValueError):
            PhaseEstimation.from_problem(p)