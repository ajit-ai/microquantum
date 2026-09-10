"""AlgorithmResult serialization (MQ-05).

`AlgorithmResult` is the uniform container returned by the generic
`algorithm.solve(problem, runtime)` lifecycle.  It carries the algorithm
name, the solved problem, the outcome, optimizer info, and free-form
configuration/metadata, and always serializes to plain JSON.
"""

import json

from microquantum.algorithms import Algorithm, AlgorithmResult


class ParitySolver(Algorithm):
    """Counts ones in a binary string (quantum-flavoured demo)."""

    @property
    def name(self) -> str:
        return "parity-solver"

    def solve(self, problem, runtime=None) -> AlgorithmResult:
        data = problem.metadata
        bits = data.get("bits", "")
        total = sum(1 for b in bits if b == "1")
        return AlgorithmResult(
            algorithm=self.name,
            problem=problem.name,
            solution={"parity": total % 2, "ones": total},
            objective=float(total),
            converged=True,
            iterations=1,
            termination_reason="deterministic",
            history=[float(total)],
            config={"ansatz": "none", "shots": 0},
            execution_metadata={"engine": "demo", "plan": "parity-v1"},
            native={"raw": bits},
        )


def main() -> None:
    print("=== AlgorithmResult serialization ===\n")

    from microquantum import Problem

    problem = Problem(name="0101-parity", num_qubits=4, metadata={"bits": "0101"})
    result = ParitySolver().solve(problem)

    print("to_dict():")
    for key, value in result.to_dict().items():
        print(f"  {key:18s} = {value}")

    # The `native` field holds the algorithm's own object and is excluded
    # from serialization (it may hold non-JSON-safe data).
    print(f"\nnative excluded:        {'native' not in result.to_dict()}")
    print(f"native still accessible: {result.native}")

    loaded = json.loads(result.to_json())
    print("\nto_json() round-trips:")
    print(
        f"  algorithm={loaded['algorithm']!r} "
        f"problem={loaded['problem']!r} solution={loaded['solution']!r}"
    )


if __name__ == "__main__":
    main()