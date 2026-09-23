"""Phase 121 W1 tour: contracts and shared protocols."""

from __future__ import annotations

import numpy as np

from microquantum.adapters import AdapterRegistry, DomainAdapter, QuantumProblem
from microquantum.core.circuit import QuantumCircuit
from microquantum.core.parameter import Parameter
from microquantum.mitigation import MitigationData, ZNEProtocol
from microquantum.optimizers import GradientDescent, minimize_with_callbacks
from microquantum.problems import ConstrainedOptimizationProblem, LinearConstraint
from microquantum.runtime import Budget, ExecutionPlan


class _DemoAdapter(DomainAdapter):
    @property
    def domain_name(self) -> str:
        return "demo"

    @property
    def supported_problems(self) -> list[str]:
        return ["demo-task"]

    def validate(self, problem: QuantumProblem) -> list[str]:
        return []

    def encode(self, problem: QuantumProblem) -> QuantumCircuit:
        return QuantumCircuit(1)

    def decode(self, problem: QuantumProblem, result: object) -> dict[str, object]:
        return {"ok": True}


def main() -> None:
    constraint = LinearConstraint(indices=(0, 1), sense="<=", rhs=1.0, penalty=5.0)
    problem = ConstrainedOptimizationProblem(
        num_variables=2, objective=lambda bits: float(bits[0]), constraints=[constraint]
    )
    print("penalty([1, 1]):", problem.penalty(np.array([1, 1])))
    print("feasible([1, 0]):", problem.is_feasible(np.array([1, 0])))

    registry = AdapterRegistry()
    registry.register("demo", _DemoAdapter)
    print("resolved:", type(registry.resolve(QuantumProblem(name="demo-task", domain="demo"))).__name__)

    plan = ExecutionPlan.from_circuit(QuantumCircuit(1), budget=Budget(max_shots=10))
    print("plan budget:", plan.to_dict()["budget"])

    theta = Parameter("t")
    result = minimize_with_callbacks(
        GradientDescent(learning_rate=0.5, max_iter=30),
        lambda params: float(params[theta] ** 2),
        gradient_fn=lambda params: {theta: 2.0 * float(params[theta])},
        initial_params={theta: 1.0},
    )
    print("minimized value:", round(result.optimal_value, 6))

    outcome = ZNEProtocol().mitigate(
        MitigationData(noisy_values=[1.0, 1.2, 1.4], noise_factors=[1.0, 3.0, 5.0])
    )
    print("mitigated:", round(outcome.mitigated_value, 6))


if __name__ == "__main__":
    main()
