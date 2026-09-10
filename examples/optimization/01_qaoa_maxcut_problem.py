"""QAOA on MaxCut via the problem abstraction (MQ-05).

The MaxCut instance is expressed as a QUBO, wrapped as an
OptimizationProblem, and handed to QAOA — no manual Hamiltonian assembly.
"""

from microquantum import OptimizationProblem
from microquantum.algorithms import QAOA
from microquantum.optimization.qubo import QUBOBuilder
from microquantum.optimizers import Adam


def maxcut_qubo(edges: list[tuple[int, int]], num_vertices: int) -> QUBOBuilder:
    """Minimize -(cut size).  A vertex in {0,1} partitions the graph."""
    builder = QUBOBuilder(num_variables=num_vertices)
    for i, j in edges:
        # cut contribution = x_i + x_j - 2 x_i x_j ; maximize -> minimize negative
        builder.add_linear(i, -1.0)
        builder.add_linear(j, -1.0)
        builder.add_quadratic(i, j, 2.0)
    return builder


def brute_force_min(problem: OptimizationProblem) -> float:
    import numpy as np

    best = float("inf")
    for bits_int in range(2 ** problem.num_variables):
        bits = np.array([(bits_int >> k) & 1 for k in range(problem.num_variables)], dtype=float)
        best = min(best, problem.energy(bits))
    return best


def main() -> None:
    print("=== QAOA on MaxCut via OptimizationProblem ===\n")

    edges = [(0, 1), (1, 2), (2, 0)]  # triangle
    num_vertices = 3
    qubo = maxcut_qubo(edges, num_vertices).build("triangle-maxcut")

    problem = OptimizationProblem.from_qubo(qubo, name="triangle-maxcut")
    print(f"Problem: {problem}")
    print(f"Ising cost Hamiltonian: {problem.cost_hamiltonian()}")

    exact = brute_force_min(problem)
    print(f"Brute-force optimum: {exact:.3f}  (a cut of {int(-exact)})")

    qaoa = QAOA.from_problem(  # num_layers=1, default optimizer
        problem, num_layers=1, optimizer=Adam(learning_rate=0.1, max_iter=200, tol=1e-6)
    )
    result = qaoa.solve(problem)
    print(f"QAOA energy: {result.eigenvalue: .3f}  "
          f"(gap from exact: {result.eigenvalue - exact: .3f})")

    # Extract the best classical cut from the final state and check it.
    bound = qaoa.build_ansatz().bind_parameters(result.eigenstate)
    probabilities = abs(bound.run().amplitudes) ** 2
    best_state = int(probabilities.argmax())
    bits = [(best_state >> q) & 1 for q in range(num_vertices)]
    cut = sum(1 for i, j in edges if bits[i] != bits[j])
    print(f"Most likely outcome |{' '.join(map(str, bits))}> -> cut = {cut}")


if __name__ == "__main__":
    main()