"""Phase 121 W1: contracts and shared protocols."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.adapters import AdapterRegistry, QuantumProblem
from microquantum.backends.noise import NoiseModel
from microquantum.core.parameter import Parameter
from microquantum.mitigation import (
    MEMProtocol,
    MitigationData,
    MitigationProtocol,
    PECProtocol,
    ZNEProtocol,
)
from microquantum.optimizers import (
    GradientDescent,
    minimize_with_callbacks,
)
from microquantum.problems import (
    ConstrainedOptimizationProblem,
    ExcitedStateProblem,
    LinearConstraint,
    OptimizationProblem,
    TimeEvolutionProblem,
)
from microquantum.problems.eigenvalue import HamiltonianProblem
from microquantum.runtime import Budget, ExecutionPlan
from microquantum.runtime.errors import PlanningError


class TestConstrainedOptimization:
    def test_penalty_and_feasibility(self) -> None:
        constraint = LinearConstraint(indices=(0, 1), sense="<=", rhs=1.0, penalty=5.0)
        problem = ConstrainedOptimizationProblem(
            num_variables=2,
            objective=lambda bits: float(bits[0]),
            constraints=[constraint],
        )
        assert problem.validate() == []
        assert problem.penalty(np.array([1, 1])) == pytest.approx(5.0)
        assert problem.penalty(np.array([1, 0])) == pytest.approx(0.0)
        assert problem.is_feasible(np.array([0, 1]))
        assert not problem.is_feasible(np.array([1, 1]))
        assert problem.penalized_energy(np.array([1, 1])) == pytest.approx(6.0)

    def test_invalid_constraints(self) -> None:
        with pytest.raises(ValueError):
            LinearConstraint(indices=(0,), sense="<>", rhs=1.0)
        with pytest.raises(ValueError):
            LinearConstraint(indices=(), sense="<=", rhs=0.0)
        problem = ConstrainedOptimizationProblem(
            num_variables=1,
            objective=lambda bits: 0.0,
            constraints=[LinearConstraint(indices=(0, 5), sense="==", rhs=1.0)],
        )
        assert any("5" in issue for issue in problem.validate())
        data = problem.to_dict()
        assert data["constraints"][0]["sense"] == "=="
        assert LinearConstraint.from_dict(data["constraints"][0]).rhs == pytest.approx(1.0)

    def test_base_problem_still_works(self) -> None:
        problem = OptimizationProblem(num_variables=2, objective=lambda bits: float(sum(bits)))
        assert problem.validate() == []


class TestDynamicsProblems:
    def test_time_evolution_validation(self) -> None:
        problem = TimeEvolutionProblem(time=1.5, num_steps=4)
        assert any("hamiltonian" in issue for issue in problem.validate())
        bad = TimeEvolutionProblem(time=-1.0, num_steps=0)
        assert len(bad.validate()) >= 2
        assert TimeEvolutionProblem(time=0.0, num_steps=1).to_dict()["num_steps"] == 1

    def test_time_evolution_round_trip(self) -> None:
        source = HamiltonianProblem()
        problem = TimeEvolutionProblem(
            hamiltonian=source.hamiltonian, time=0.5, num_steps=2
        )
        rebuilt = TimeEvolutionProblem.from_dict(problem.to_dict())
        assert rebuilt.time == pytest.approx(0.5)
        assert rebuilt.num_steps == 2

    def test_excited_state_problem(self) -> None:
        from microquantum.core.operators import Operator

        problem = ExcitedStateProblem(hamiltonian=Operator.Z(), num_states=3, k=2)
        assert problem.num_states == 3
        assert problem.validate() == []
        bad = ExcitedStateProblem(num_states=1, k=2)
        assert any("num_states" in issue for issue in bad.validate())
        with pytest.raises(ValueError):
            ExcitedStateProblem(num_states=0)
        assert problem.to_dict()["num_states"] == 3


class TestAdapterRegistry:
    def test_register_and_resolve(self) -> None:
        from microquantum.adapters import DomainAdapter

        class _DemoAdapter(DomainAdapter):
            @property
            def domain_name(self) -> str:
                return "demo"

            @property
            def supported_problems(self) -> list[str]:
                return ["demo-task"]

            def validate(self, problem: QuantumProblem) -> list[str]:
                return []

            def encode(self, problem: QuantumProblem) -> object:
                from microquantum.core.circuit import QuantumCircuit

                return QuantumCircuit(1)

            def decode(self, problem: QuantumProblem, result: object) -> dict[str, object]:
                return {"ok": True}

        registry = AdapterRegistry()
        registry.register("demo", _DemoAdapter)
        assert registry.registered() == ["demo"]
        assert len(registry) == 1
        assert "demo" in registry
        problem = QuantumProblem(name="demo-task", domain="demo")
        assert isinstance(registry.resolve(problem), _DemoAdapter)
        with pytest.raises(LookupError):
            registry.resolve(QuantumProblem(name="other-task", domain="unknown"))
        with pytest.raises(ValueError):
            registry.register("", _DemoAdapter)
        with pytest.raises(TypeError):
            registry.register("bad", object())  # type: ignore[arg-type]


class TestBudget:
    def test_check_and_serialization(self) -> None:
        budget = Budget(max_shots=100, max_circuits=2)
        budget.check(shots=100, circuits=2)
        with pytest.raises(PlanningError):
            budget.check(shots=101)
        with pytest.raises(PlanningError):
            budget.check(circuits=3)
        assert Budget.from_dict(budget.to_dict()).max_shots == 100
        with pytest.raises(ValueError):
            Budget(max_shots=0)

    def test_plan_carries_budget(self) -> None:
        from microquantum.core.circuit import QuantumCircuit

        plan = ExecutionPlan.from_circuit(
            QuantumCircuit(1), budget=Budget(max_shots=10), cacheable=False
        )
        assert plan.cacheable is False
        assert plan.to_dict()["budget"] == {"max_shots": 10, "max_circuits": None, "max_wall_s": None}
        plain = ExecutionPlan.from_circuit(QuantumCircuit(1))
        assert plain.cacheable is True
        assert plain.to_dict()["budget"] is None


class TestBoundsAndCallbacks:
    def test_bounds_projection(self) -> None:
        from microquantum.optimizers import Bounds as BoundsAgain

        bounds = BoundsAgain(lower=0.0, upper=1.0)
        theta = Parameter("t")
        assert bounds.project({theta: 1.5})[theta] == pytest.approx(1.0)
        assert bounds.project({theta: -0.5})[theta] == pytest.approx(0.0)
        assert BoundsAgain.from_dict(bounds.to_dict()).upper == pytest.approx(1.0)
        with pytest.raises(ValueError):
            BoundsAgain(lower=2.0, upper=1.0)

    def test_callbacks_and_early_stop(self) -> None:
        seen: list[float] = []

        class _Stopper:
            def on_iteration(self, params: dict[Parameter, float], value: float) -> bool:
                seen.append(value)
                return len(seen) >= 2

        theta = Parameter("t")
        result = minimize_with_callbacks(
            GradientDescent(learning_rate=0.1, max_iter=50),
            lambda params: float((params[theta] - 1.0) ** 2),
            initial_params={theta: 0.0},
            callbacks=[_Stopper()],
        )
        assert not result.converged
        assert len(seen) == 2
        assert len(result.history) == 2

    def test_no_callbacks_full_run(self) -> None:
        theta = Parameter("t")
        result = minimize_with_callbacks(
            GradientDescent(learning_rate=0.5, max_iter=30),
            lambda params: float(params[theta] ** 2),
            gradient_fn=lambda params: {theta: 2.0 * float(params[theta])},
            initial_params={theta: 1.0},
        )
        assert result.optimal_value < 1e-3
        with pytest.raises(ValueError):
            minimize_with_callbacks(
                GradientDescent(max_iter=2),
                lambda params: 0.0,
                initial_params={theta: 0.0},
                callback_every=0,
            )


class TestMitigationProtocol:
    def test_zne_protocol(self) -> None:
        protocol = ZNEProtocol()
        assert protocol.name == "zne"
        outcome = protocol.mitigate(
            MitigationData(noisy_values=[1.0, 1.2, 1.4], noise_factors=[1.0, 3.0, 5.0])
        )
        assert outcome.mitigated_value == pytest.approx(0.9, abs=0.05)
        assert outcome.method.startswith("zne-")
        with pytest.raises(ValueError):
            protocol.mitigate(MitigationData())
        assert issubclass(ZNEProtocol, MitigationProtocol)

    def test_mem_protocol(self) -> None:
        confusion = np.array([[0.9, 0.1], [0.1, 0.9]])
        protocol = MEMProtocol()
        outcome = protocol.mitigate(
            MitigationData(
                counts={"0": 80, "1": 20},
                observable=np.array([[1, 0], [0, -1]], dtype=complex),
                confusion=confusion,
                num_qubits=1,
            )
        )
        assert outcome.method == "mem"
        assert -1.0 <= outcome.mitigated_value <= 1.0
        with pytest.raises(ValueError):
            protocol.mitigate(MitigationData(counts={"0": 1}))

    def test_pec_protocol(self) -> None:
        protocol = PECProtocol()
        outcome = protocol.mitigate(
            MitigationData(
                channel_matrix=np.eye(4, dtype=complex),
                noisy_values=[0.8, 0.9],
            )
        )
        assert outcome.method == "pec"
        assert "sampling_overhead" in outcome.metadata
        with pytest.raises(ValueError):
            protocol.mitigate(MitigationData())
        assert isinstance(NoiseModel(), NoiseModel)
