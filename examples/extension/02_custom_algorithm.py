"""Defining a custom Algorithm (MQ-05).

Algorithms extend the generic `Algorithm` base and integrate with
`AlgorithmResult`; callers invoke them through the uniform
`validate(problem)` + `solve(problem, runtime)` interface.
"""

from microquantum import Problem
from microquantum.algorithms import Algorithm, AlgorithmResult


class ParitySolver(Algorithm):
    """Counts ones in a binary string (classical, for the demo)."""

    @property
    def name(self) -> str:
        return "parity-solver"

    def validate(self, problem) -> list[str]:
        issues = super().validate(problem)
        if "bits" not in (problem.metadata or {}):
            issues.append("metadata['bits'] is required")
        return issues

    def solve(self, problem, runtime=None) -> AlgorithmResult:
        bits = problem.metadata["bits"]
        ones = sum(1 for b in bits if b == "1")
        return AlgorithmResult(
            algorithm=self.name,
            problem=problem.name,
            solution={"ones": ones, "parity": ones % 2},
            objective=float(ones),
            converged=True,
            iterations=1,
            termination_reason="trivially exact",
            history=[float(ones)],
            config={"over": bits},
        )


def main() -> None:
    print("=== Custom algorithm through the generic lifecycle ===\n")

    solver = ParitySolver()
    problem = Problem(name="parity-demo", num_qubits=4, metadata={"bits": "1011001"})

    print(f"  validate: {solver.validate(problem)}")
    result = solver.solve(problem)
    print(f"  solution: {result.solution}")
    print(f"  to_dict:  {result.to_dict()['algorithm']!r} / "
          f"{result.to_dict()['objective']}")


if __name__ == "__main__":
    main()