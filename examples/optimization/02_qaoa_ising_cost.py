"""QAOA with a direct Ising cost Hamiltonian (MQ-05).

OptimizationProblem supports an `ising` view directly, which is useful
when the interaction structure of the problem is already known.
"""

from microquantum import OptimizationProblem
from microquantum.algorithms import QAOA
from microquantum.core import PauliString, PauliSum
from microquantum.optimizers import Adam


def main() -> None:
    print("=== QAOA on a direct Ising cost ===\n")

    # H = 1.0 * Z0 Z1 + 0.5 * Z1 Z2 - 1.0 * Z0  (all coefficients real)
    ising = PauliSum(
        [
            PauliString("ZZI", 1.0),
            PauliString("IZZ", 0.5),
            PauliString("ZII", -1.0),
        ]
    )
    problem = OptimizationProblem.from_ising(ising, name="ising-3spin")
    print(f"Problem: {problem}")
    print(f"Cost Hamiltonian: {problem.cost_hamiltonian()}")

    # Classical reference: enumerate all 2^3 configurations.
    import numpy as np

    from microquantum.core import StateVector

    def spin_energy(bits_int: int) -> float:
        amps = np.zeros(8, dtype=np.complex128)
        amps[bits_int] = 1.0
        return problem.cost_hamiltonian().expectation(StateVector(3, amplitudes=amps))

    best = min(spin_energy(i) for i in range(8))
    print(f"Exact ground energy: {best:.3f}")

    qaoa = QAOA.from_problem(
        problem, num_layers=1, optimizer=Adam(0.1, max_iter=250, tol=1e-6)
    )
    result = qaoa.solve(problem)
    print(f"QAOA energy: {result.eigenvalue: .3f}")


if __name__ == "__main__":
    main()