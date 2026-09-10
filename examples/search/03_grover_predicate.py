"""GroverSearch with a predicate-based SearchProblem (MQ-05).

The predicate view (`is_marked`) maps every index to True/False, which
supports marked sets that are not stored as an explicit list.
"""

from microquantum import SearchProblem
from microquantum.algorithms import GroverSearch


def main() -> None:
    print("=== GroverSearch with a predicate oracle ===\n")

    n_qubits = 4

    def is_square(i: int) -> bool:
        root = round(i ** 0.5)
        return root * root == i

    problem = SearchProblem(
        num_qubits=n_qubits,
        predicate=is_square,
        name="find-perfect-squares",
    )
    targets = problem.target_indices()
    print(f"predicate marks: {targets}")

    grover = GroverSearch(num_qubits=n_qubits, target=targets)
    print(f"optimal iterations: {grover.num_iterations}")

    result = grover.solve(problem, seed=5)
    print(f"most probable: {result.most_probable} (square? "
          f"{result.most_probable in targets}, p = "
          f"{result.success_probability:.3f})")


if __name__ == "__main__":
    main()