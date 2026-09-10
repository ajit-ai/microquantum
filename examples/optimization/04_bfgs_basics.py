"""BFGS classic optimizer (MQ-05).

A pure-NumPy quasi-Newton optimizer, usable as the classical optimizer
behind any variational algorithm (no SciPy dependency).
"""

from microquantum.core import Parameter
from microquantum.optimizers import BFGS

x, y = Parameter("x"), Parameter("y")


def rosenbrock(p):
    """f(x, y) = (1 - x)^2 + 100 (y - x^2)^2."""
    return (1 - p[x]) ** 2 + 100.0 * (p[y] - p[x] ** 2) ** 2


def rosenbrock_grad(p):
    gx = -2.0 * (1 - p[x]) - 400.0 * p[x] * (p[y] - p[x] ** 2)
    gy = 200.0 * (p[y] - p[x] ** 2)
    return {x: gx, y: gy}


def main() -> None:
    print("=== BFGS on Rosenbrock ===\n")

    optimizer = BFGS(max_iter=300, tol=1e-8)
    result = optimizer.minimize(
        cost_fn=rosenbrock,
        gradient_fn=rosenbrock_grad,
        initial_params={x: 1.5, y: 1.5},
    )

    print("Distance to the exact minimum (1, 1):")
    print(f"  x = {result.optimal_parameters[x]:.6f} "
          f"(err {abs(result.optimal_parameters[x] - 1.0):.2e})")
    print(f"  y = {result.optimal_parameters[y]:.6f} "
          f"(err {abs(result.optimal_parameters[y] - 1.0):.2e})")
    print(f"  value = {result.optimal_value:.3e}")
    print(f"  converged = {result.converged}, iterations = {result.iterations}")

    print("\nGradient-free mode (finite differences when no gradient given):")
    result2 = BFGS(max_iter=120, tol=1e-5).minimize(
        cost_fn=rosenbrock, initial_params={x: 1.5, y: 1.5}
    )
    print(f"  x = {result2.optimal_parameters[x]:.6f}, "
          f"y = {result2.optimal_parameters[y]:.6f}, "
          f"converged = {result2.converged}")


if __name__ == "__main__":
    main()