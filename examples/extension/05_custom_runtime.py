"""Extending the runtime integration: routing solves to a custom engine (MQ-05).

Algorithms hold no engine details: they receive a `runtime` when solving.
This example routes a Grover solve through a minimal custom runtime
subclass whose history hook makes every executed plan visible.
"""

from typing import Any, Optional

from microquantum import SearchProblem
from microquantum.algorithms import GroverSearch
from microquantum.runtime import ExecutionRuntime


class LoggingRuntime(ExecutionRuntime):
    """ExecutionRuntime that echoes every executed plan/status."""

    def _record_history(
        self,
        plan,
        backend,
        job,
        trace,
        elapsed,
        *,
        error: Optional[str] = None,
        result: Any = None,
    ) -> None:
        super()._record_history(
            plan, backend, job, trace, elapsed, error=error, result=result
        )
        print(f"    [logging-runtime] plan={plan.name} "
              f"job={job.job_id} status={job.status.value}")


def main() -> None:
    print("=== Routing solves through a custom runtime ===\n")

    problem = SearchProblem(num_qubits=4, target=5, name="find-5")
    grover = GroverSearch(num_qubits=4, target=5)
    runtime = LoggingRuntime()
    result = grover.solve(problem, runtime=runtime, seed=3)
    print(f"  most probable = {result.most_probable} "
          f"(found 5: {result.most_probable == 5})")


if __name__ == "__main__":
    main()