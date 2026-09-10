"""GroverSearch through the MQ-04 ExecutionRuntime (MQ-05).

The same algorithm object can run against the internal state-vector
engine or a runtime pipeline, with a seeded sampler for statistical
reproducibility.
"""

from microquantum import SearchProblem
from microquantum.algorithms import GroverSearch
from microquantum.runtime import ExecutionRuntime


def run(via_runtime: bool) -> None:
    problem = SearchProblem(num_qubits=4, target=13, name="find-13")
    grover = GroverSearch(num_qubits=4, target=13)
    kwargs = {}
    if via_runtime:
        kwargs["runtime"] = ExecutionRuntime()

    result = grover.solve(problem, seed=42, shots=8192, **kwargs)
    label = "runtime" if via_runtime else "engine"
    print(f"  [{label:8s}] most probable = {result.most_probable} "
          f"(found 13: {result.most_probable == 13})  "
          f"p = {result.success_probability:.3f}")


def main() -> None:
    print("=== GroverSearch: engine vs runtime ===\n")
    run(via_runtime=False)
    run(via_runtime=True)


if __name__ == "__main__":
    main()