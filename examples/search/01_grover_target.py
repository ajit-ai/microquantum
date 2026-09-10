"""GroverSearch over a SearchProblem with a marked target (MQ-05).

The problem carries the marked target; Grover builds the oracle, mixes,
and reports the found mark.
"""

from microquantum import SearchProblem
from microquantum.algorithms import GroverSearch


def main() -> None:
    print("=== GroverSearch: find a marked state ===\n")

    for n_qubits, target in ((4, 10), (5, 2)):
        problem = SearchProblem(num_qubits=n_qubits, target=target, name=f"find-{target}")
        print(f"problem: {problem}  (valid: {problem.validate() == []})")

        grover = GroverSearch.from_problem(problem)
        print(f"automatic iterations: {grover.num_iterations} "
              f"(optimal M = floor(pi/4 * sqrt(2^n)))")

        result = grover.solve(problem, seed=7)
        state = result.most_probable
        marked = state in problem.target_indices()
        print(f"  found {state} (as {target:0{n_qubits}b})  "
              f"success p = {result.success_probability:.3f}  "
              f"in marked set: {marked}")
        print()


if __name__ == "__main__":
    main()