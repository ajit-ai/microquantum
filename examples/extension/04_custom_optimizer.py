"""Plugging a custom optimizer into variational algorithms (MQ-05).

An Optimizer subclasses the abstract base and implements `_step`,
`_converged`, and `max_iter`; the base `minimize` loop drives it.
A tiny coordinate-descent optimizer is dropped straight into VQE next to
the built-in optimizers.
"""

from microquantum import EigenvalueProblem
from microquantum.algorithms import VQE
from microquantum.core import Operator, Parameter, QuantumCircuit
from microquantum.optimizers import Optimizer


class CoordinateDescent(Optimizer):
    """Greedy axis-aligned coordinate descent (simple demo optimizer)."""

    def __init__(self, step: float = 0.1, max_iter: int = 80, tol: float = 1e-12) -> None:
        self.step = step
        self._max_iter = max_iter
        self.tol = tol

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def _step(self, params, grads, iteration, cost_fn=None) -> dict:
        best = dict(params)
        best_value = cost_fn(best)
        for name in list(best):
            for delta in (self.step, -self.step):
                trial = dict(best)
                trial[name] += delta
                value = cost_fn(trial)
                if value < best_value - 1e-12:
                    best_value, best = value, trial
        return best

    def _converged(self, history) -> bool:
        return len(history) >= 2 and abs(history[-1] - history[-2]) < self.tol


def main() -> None:
    print("=== Custom optimizer inside VQE ===\n")

    theta = Parameter("theta")
    ansatz = QuantumCircuit(1).ry(theta, 0)
    problem = EigenvalueProblem(Operator.Z(), k=1, name="z")

    vqe = VQE(ansatz, Operator.Z(), CoordinateDescent(step=0.2, max_iter=200))
    result = vqe.solve(problem, initial_params={theta: 0.5})
    print(f"  eigenvalue with custom optimizer: {result.eigenvalue:.4f} "
          f"(analytic: -1.0)")
    print(f"  optimizer iterations: {result.optimizer_result.iterations} "
          f"converged: {result.optimizer_result.converged}")


if __name__ == "__main__":
    main()