"""Generic algorithm lifecycle (MQ-05).

Every algorithm follows the same lifecycle when driven by a problem:
``validate(problem) -> [errors]`` then ``solve(problem, runtime)``.
Validation collects *descriptive* problems instead of raising so callers
can aggregate diagnostics before execution.
"""

from microquantum import EigenvalueProblem, OptimizationProblem
from microquantum.algorithms import QAOA, VQE
from microquantum.core import Operator, Parameter, QuantumCircuit
from microquantum.optimization.qubo import QUBOBuilder
from microquantum.optimizers import GradientDescent


def main() -> None:
    print("=== Algorithm lifecycle ===\n")

    # VQE solves eigenvalue problems
    theta = Parameter("theta")
    ansatz = QuantumCircuit(1).ry(theta, 0)
    vqe = VQE(
        ansatz,
        Operator.Z(),
        GradientDescent(learning_rate=0.3, max_iter=60, tol=1e-8),
    )
    eigen = EigenvalueProblem(Operator.Z(), k=1, name="z-hamiltonian")

    issues = vqe.validate(eigen)
    print(f"VQE.validate(eigenvalue problem) -> {issues} (empty = valid)")

    result = vqe.solve(eigen, initial_params={theta: 0.5})
    print(f"VQE.solve eigenvalue: {result.eigenvalue:.4f}  "
          f"(analytic ground state of Z is -1)")
    print(f"result type: {type(result).__name__}")

    # Same ansatz/optimizer reused for a different problem: no rewrite needed
    other = EigenvalueProblem(Operator.Z(), k=1, name="again")
    result2 = vqe.solve(other, initial_params={theta: 0.5})
    print(f"Second problem solved with the same configured VQE: "
          f"{result2.eigenvalue:.4f}")

    print("\nValidation turns mistakes into actionable messages:")
    bad = OptimizationProblem(num_variables=1, objective=lambda b: float(b[0]))
    print(f"  VQE.validate(optimization problem) -> {vqe.validate(bad)}")

    builder = QUBOBuilder(num_variables=2)
    builder.add_quadratic(0, 1, 2.0)
    builder.add_linear(0, -1.0)
    builder.add_linear(1, -1.0)
    plain = builder.build()
    opt = OptimizationProblem(num_variables=2, objective=lambda b: float(plain.energy(b)))
    qaoa = QAOA(Operator.Z(), num_qubits=2, num_layers=1)
    print(f"  QAOA.validate(problem w/o Ising view) -> {qaoa.validate(opt)}")


if __name__ == "__main__":
    main()