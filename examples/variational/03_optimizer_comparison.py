"""Choosing a classical optimizer for VQE (MQ-05).

Gradient-based (GradientDescent, Adam), quasi-Newton (BFGS) and
gradient-free (COBYLA, NelderMead) optimizers are all drop-in
Optimizer implementations for variational algorithms.
"""

from microquantum import EigenvalueProblem
from microquantum.algorithms import VQE
from microquantum.core import Operator, Parameter, QuantumCircuit
from microquantum.optimizers import (
    BFGS,
    COBYLA,
    Adam,
    GradientDescent,
    NelderMead,
)


def main() -> None:
    print("=== Optimizer comparison for a 1-qubit VQE ===\n")

    theta = Parameter("theta")
    ansatz = QuantumCircuit(1).ry(theta, 0)
    problem = EigenvalueProblem(Operator.Z(), k=1, name="z")

    optimizers = {
        "GradientDescent": GradientDescent(0.3, max_iter=60, tol=1e-6),
        "Adam": Adam(learning_rate=0.2, max_iter=80, tol=1e-6),
        "BFGS": BFGS(max_iter=40, tol=1e-6),
        "COBYLA": COBYLA(max_iter=200, tol=1e-4),
        "NelderMead": NelderMead(max_iter=200, tol=1e-4),
    }

    print(f"{'optimizer':16s} {'energy':>8s}  {'iters':>6s}")
    for name, optimizer in optimizers.items():
        vqe = VQE(ansatz, Operator.Z(), optimizer)
        result = vqe.solve(problem, initial_params={theta: 0.5})
        iters = result.optimizer_result.iterations if result.optimizer_result else -1
        print(f"{name:16s} {result.eigenvalue:8.4f}  {iters:6d}")

    print("\nAll converge to -1.0 (the exact ground state of Z).")


if __name__ == "__main__":
    main()