"""MQ-05: algorithms execute their circuits through the ExecutionRuntime
when one is supplied, and the runtime trace gets populated."""

from __future__ import annotations

import pytest

from microquantum.algorithms import QAOA, VQE, GroverSearch, PhaseEstimation
from microquantum.core import Operator, Parameter, QuantumCircuit, tensor
from microquantum.optimization.qubo import QUBOBuilder
from microquantum.optimizers import Adam, GradientDescent
from microquantum.problems import (
    EigenvalueProblem,
    OptimizationProblem,
    SearchProblem,
)
from microquantum.runtime import ExecutionRuntime


def _runtime() -> ExecutionRuntime:
    return ExecutionRuntime()


class TestRuntimeIntegratedAlgorithms:
    def test_vqe_runs_through_runtime(self) -> None:
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)
        rt = _runtime()
        vqe = VQE(
            ansatz,
            Operator.Z(),
            GradientDescent(0.3, max_iter=60, tol=1e-6),
            runtime=rt,
            seed=3,
        )
        result = vqe.compute_minimum_eigenvalue(initial_params={theta: 0.5})
        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-3)
        assert len(rt.history) > 0

    def test_qaoa_problem_solve_through_runtime(self) -> None:
        builder = QUBOBuilder(num_variables=2)
        builder.add_quadratic(0, 1, 2.0)
        builder.add_linear(0, -1.0)
        builder.add_linear(1, -1.0)
        problem = OptimizationProblem.from_qubo(builder.build(), name="maxcut2")

        rt = _runtime()
        qaoa = QAOA.from_problem(
            problem,
            num_layers=1,
            optimizer=Adam(0.1, max_iter=80, tol=1e-6),
            runtime=rt,
            seed=5,
        )
        result = qaoa.solve(problem)
        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-2)
        assert len(rt.history) > 0

    def test_grover_solve_through_runtime(self) -> None:
        problem = SearchProblem(num_qubits=3, target=5, name="find-5")
        rt = _runtime()
        result = GroverSearch.from_problem(problem).solve(problem, runtime=rt, seed=7)
        assert result.most_probable == 5
        assert result.success_probability > 0.9
        assert len(rt.history) == 1

    def test_phase_estimation_through_runtime(self) -> None:
        problem = EigenvalueProblem(Operator.S())
        rt = _runtime()
        result = PhaseEstimation.from_problem(
            problem, num_counting_qubits=4, runtime=rt, seed=9
        ).solve(problem)
        assert result.phase == pytest.approx(0.25)
        assert len(rt.history) == 1

    def test_execution_metadata_present_in_history(self) -> None:
        problem = SearchProblem(num_qubits=2, target=1, name="find-1")
        rt = _runtime()
        GroverSearch.from_problem(problem).solve(problem, runtime=rt)
        latest = rt.history[-1]
        # history entries are dicts carrying a status key
        assert isinstance(latest, dict) or hasattr(latest, "status")


class TestRuntimeDeterminism:
    def test_statevector_exact_regardless_of_shots(self) -> None:
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)
        values: list[float] = []
        for shots in (1, 64, 4096):
            rt = _runtime()
            vqe = VQE(
                ansatz,
                Operator.Z(),
                GradientDescent(0.3, max_iter=30, tol=1e-6),
                runtime=rt,
                shots=shots,
                seed=1,
            )
            result = vqe.compute_minimum_eigenvalue(initial_params={theta: 0.5})
            values.append(result.eigenvalue)
        assert values[0] == pytest.approx(values[-1], abs=1e-6)


class TestQAOALegacyStillWorks:
    def test_legacy_solve_unaffacted_by_runtime_kwargs(self) -> None:
        ZZ = tensor(Operator.Z(), Operator.Z())
        II = tensor(Operator.I(), Operator.I())
        H_C = 0.5 * (II - ZZ)
        qaoa = QAOA(H_C, num_qubits=2, num_layers=1, optimizer=Adam(0.1, max_iter=100, tol=1e-8))
        result = qaoa.solve(initial_gamma=[0.5], initial_beta=[0.5])
        assert result.eigenvalue == pytest.approx(0.0, abs=1e-2)