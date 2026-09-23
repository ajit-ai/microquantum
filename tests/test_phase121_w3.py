"""Phase 121 W3: domain methods I."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.algorithms import (
    InitialPoint,
    QPEPhaseFilter,
    QuantumCounting,
    initial_parameters,
)
from microquantum.analysis import HypothesisTest, bootstrap_ci, float_mean
from microquantum.core.operators import Operator
from microquantum.core.parameter import Parameter
from microquantum.optimization import (
    PUBOBuilder,
    PUBOProblem,
    QUBOBuilder,
    qubo_to_pauli_sum,
)
from microquantum.optimization.ising_pauli import IsingToPauli, pubo_to_qubo_projection
from microquantum.optimizers import SPSA, GradientDescent
from microquantum.problems import SearchProblem


class TestQuantumCounting:
    def test_counts_single_target(self) -> None:
        counter = QuantumCounting(num_evaluation_qubits=3, shots=256)
        assert counter.name == "quantum_counting"
        result = counter.count(3, [0])
        assert result.estimated_count == 1
        assert result.num_qubits == 3
        assert result.fraction == pytest.approx(0.125)
        assert result.confidence == pytest.approx(1.0)

    def test_counts_multiple_targets(self) -> None:
        counter = QuantumCounting(num_evaluation_qubits=3, shots=256)
        result = counter.count(2, [0, 3])
        assert result.estimated_count == 2
        assert result.fraction == pytest.approx(0.5)
        assert counter.count(2, []).estimated_count == 0
        assert counter.count(2, ["00", "11"]).estimated_count == 2

    def test_solve_search_problem(self) -> None:
        problem = SearchProblem(num_qubits=2, target=1)
        counter = QuantumCounting(num_evaluation_qubits=2, shots=128)
        assert counter.validate(problem) == []
        result = counter.solve(problem)
        assert result.estimated_count == 1

    def test_oracle_construction(self) -> None:
        counter = QuantumCounting()
        oracle = counter.build_oracle(2, [1])
        assert oracle.num_qubits == 2
        assert counter.build_state_preparation(2).num_gates == 2
        assert counter.count(2, [0]).to_dict()["num_qubits"] == 2
        with pytest.raises(ValueError):
            QuantumCounting(num_evaluation_qubits=0)
        with pytest.raises(ValueError):
            counter.build_oracle(0, [0])
        with pytest.raises(ValueError):
            counter.build_oracle(2, [4])
        with pytest.raises(ValueError):
            counter.solve(SearchProblem(num_qubits=1))
        with pytest.raises(ValueError):
            counter.solve(object())

    def test_estimate_count_agrees_on_exact_fractions(self) -> None:
        counter = QuantumCounting(num_evaluation_qubits=4, shots=64)
        assert counter.estimate_count(1, [1]).estimated_count == 1
        assert counter.estimate_count(2, [0, 3]).estimated_count == 2
        assert counter.estimate_count(2, []).estimated_count == 0
        near = counter.estimate_count(2, [3], num_evaluation_qubits=5)
        assert near.estimated_count == 1
        assert near.metadata["method"] == "qpe-sampling"


class TestQPEPhaseFilter:
    def test_schedule_and_circuit(self) -> None:
        filtr = QPEPhaseFilter(kappa=1.0)
        assert filtr.rotation_angles(3) == [1.0, 0.5, 0.25]
        circuit = filtr.filter_circuit(2)
        assert circuit.num_qubits == 3
        assert circuit.num_gates == 6
        assert "kappa=1.0" in repr(filtr)
        with pytest.raises(ValueError):
            QPEPhaseFilter(kappa=0.0)
        with pytest.raises(ValueError):
            filtr.rotation_angles(0)


class TestInitialPoint:
    def test_strategies(self) -> None:
        theta = Parameter("t")
        phi = Parameter("p")
        assert initial_parameters([theta, phi], InitialPoint(strategy="zeros")) == {
            theta: 0.0,
            phi: 0.0,
        }
        random_point = initial_parameters(
            [theta], InitialPoint(strategy="random", seed=0, range=0.5)
        )
        assert -0.5 <= random_point[theta] <= 0.5
        custom = initial_parameters(
            [theta, phi], InitialPoint(strategy="custom", values={"t": 1.0, phi: 2.0})
        )
        assert custom == {theta: 1.0, phi: 2.0}
        with pytest.raises(ValueError):
            InitialPoint(strategy="bogus")
        with pytest.raises(ValueError):
            InitialPoint(strategy="custom")
        with pytest.raises(KeyError):
            initial_parameters([theta], InitialPoint(strategy="custom", values={"other": 1.0}))


class TestVQECallbacks:
    def test_vqe_accepts_callbacks(self) -> None:
        from microquantum.algorithms import VQE
        from microquantum.chemistry import H2Hamiltonian
        from microquantum.optimizers import GradientDescent as GD

        hamiltonian = H2Hamiltonian()
        ansatz_circuit = __import__("microquantum").QuantumCircuit(2)
        ansatz_circuit.ry(Parameter("a"), 0)
        ansatz_circuit.cx(0, 1)
        vqe = VQE(ansatz_circuit, hamiltonian.hamiltonian, GD(learning_rate=0.1, max_iter=3))
        seen: list[float] = []

        class _Recorder:
            def on_iteration(self, params: dict[Parameter, float], value: float) -> bool:
                seen.append(value)
                return False

        result = vqe.compute_minimum_eigenvalue(callbacks=[_Recorder()])
        assert len(seen) > 0
        assert np.isfinite(result.eigenvalue)
        plain = vqe.compute_minimum_eigenvalue()
        assert np.isfinite(plain.eigenvalue)


class TestInference:
    def test_chi_square_comparison(self) -> None:
        test = HypothesisTest(alpha=0.05, permutations=200, seed=0)
        same_a = {"00": 50, "11": 50}
        same_b = {"00": 52, "11": 48}
        outcome = test.compare(same_a, same_b)
        assert outcome.p_value > 0.05
        assert not outcome.significant
        assert outcome.degrees_of_freedom == 1
        different = test.compare({"00": 90, "11": 10}, {"00": 10, "11": 90})
        assert different.significant
        assert different.to_dict()["degrees_of_freedom"] == 1
        with pytest.raises(ValueError):
            HypothesisTest(alpha=1.5)
        with pytest.raises(ValueError):
            test.compare({"00": 5}, {"00": 5})
        with pytest.raises(ValueError):
            test.compare({}, {})

    def test_bootstrap_ci(self) -> None:
        low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0], resamples=200, seed=0)
        assert low < 2.5 < high
        assert float_mean([1.0, 3.0]) == pytest.approx(2.0)
        with pytest.raises(ValueError):
            bootstrap_ci([])
        with pytest.raises(ValueError):
            bootstrap_ci([1.0], confidence=2.0)


class TestPUBO:
    def test_energy_and_round_trip(self) -> None:
        builder = PUBOBuilder(3)
        builder.add_linear(0, 1.0)
        builder.add_quadratic(0, 1, -2.0)
        builder.add_term((0, 1, 2), 0.5)
        builder.add_constant(0.25)
        problem = builder.build(name="demo")
        assert problem.max_order() == 3
        assert problem.energy(np.array([1, 1, 1])) == pytest.approx(-0.25)
        rebuilt = PUBOProblem.from_dict(problem.to_dict())
        assert rebuilt.energy(np.array([1, 1, 1])) == pytest.approx(-0.25)
        assert rebuilt.name == "demo"
        with pytest.raises(ValueError):
            PUBOBuilder(0)
        with pytest.raises(ValueError):
            builder.add_term((), 1.0)
        with pytest.raises(ValueError):
            builder.add_term((0, 9), 1.0)
        with pytest.raises(ValueError):
            problem.energy(np.array([1, 0]))

    def test_bridges(self) -> None:
        qubo_builder = QUBOBuilder(2)
        qubo_builder.add_linear(0, -1.0)
        qubo_builder.add_quadratic(0, 1, 2.0)
        qubo = qubo_builder.build()
        pauli_sum = qubo_to_pauli_sum(qubo)
        assert pauli_sum.num_qubits == 2
        builder = PUBOBuilder(2)
        builder.add_term((0, 1), 1.0)
        builder.add_term((0, 1, 0), 0.0)
        projected = pubo_to_qubo_projection(builder.build())
        assert projected.num_variables == 2
        assert projected.metadata["source"] == "pubo-projection"
        cubic = PUBOBuilder(3)
        cubic.add_term((0, 1, 2), 4.0)
        projected_cubic = pubo_to_qubo_projection(cubic.build())
        assert projected_cubic.metadata["dropped_higher_order"] == pytest.approx(4.0)
        with pytest.raises(TypeError):
            IsingToPauli.to_pauli_sum(object())
        with pytest.raises(ValueError):
            from microquantum.core.pauli import PauliString

            IsingToPauli.to_pauli_sum(
                __import__("microquantum").PauliSum([PauliString("X")])
            )


class TestSPSAOptions:
    def test_resampling_and_blocking(self) -> None:
        theta = Parameter("t")
        plain = SPSA(max_iter=20, seed=0)
        resampled = SPSA(max_iter=20, seed=0, resamples=3)
        blocked = SPSA(max_iter=20, seed=0, blocking=True)

        def cost(params: dict[Parameter, float]) -> float:
            return float(params[theta] ** 2)

        for optimizer in (plain, resampled, blocked):
            result = optimizer.minimize(cost, initial_params={theta: 1.0})
            assert np.isfinite(result.optimal_value)
        with pytest.raises(ValueError):
            SPSA(resamples=0)

    def test_unused_import_guard(self) -> None:
        assert Operator.Z().name == "z"
        assert GradientDescent(learning_rate=0.1, max_iter=2) is not None
