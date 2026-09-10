"""Problem/algorithm separation + runtime integration (MQ-05).

The same configured algorithm can be applied to any compatible problem,
and every circuit evaluation can be routed through the MQ-04
ExecutionRuntime (plan -> backend -> job -> enriched result) instead of the
internal engine.
"""

from microquantum import EigenvalueProblem, SearchProblem
from microquantum.algorithms import VQE, GroverSearch
from microquantum.core import Operator, Parameter, QuantumCircuit
from microquantum.optimizers import GradientDescent
from microquantum.runtime import ExecutionRuntime


def main() -> None:
    print("=== Separation of problem and algorithm ===\n")

    # One ansatz+optimizer configuration, two different problems.
    theta = Parameter("theta")
    ansatz = QuantumCircuit(1).ry(theta, 0)
    vqe = VQE(ansatz, Operator.Z(), GradientDescent(0.3, max_iter=60, tol=1e-8))

    for name, hamiltonian in (("Z", Operator.Z()), ("-Z", -1.0 * Operator.Z())):
        problem = EigenvalueProblem(hamiltonian, k=1, name=name)
        result = vqe.solve(problem, initial_params={theta: 0.5})
        print(f"  VQE on {name:3s} -> eigenvalue {result.eigenvalue: .4f} "
              f"(validation: {vqe.validate(problem)})")

    print("\nRuntime-integrated solve (trace is populated):")
    runtime = ExecutionRuntime()
    problem = SearchProblem(num_qubits=3, target=3, name="find-3")
    grover = GroverSearch.from_problem(problem)
    result = grover.solve(problem, runtime=runtime, seed=7)
    print(
        f"  Grover via runtime -> most probable {result.most_probable}, "
        f"success p={result.success_probability:.3f}"
    )
    print(f"  runtime history entries: {len(runtime.history)}")
    print(f"  trace events of last job: "
          f"{[e for e in runtime.history[-1]['trace']]}"
          if isinstance(runtime.history[-1], dict) else "  (trace detail skipped)")

    # VQE with every energy evaluation through the runtime.
    runtime2 = ExecutionRuntime()
    vqe_rt = VQE(
        ansatz,
        Operator.Z(),
        GradientDescent(0.3, max_iter=40, tol=1e-6),
        runtime=runtime2,
        seed=3,
    )
    result = vqe_rt.compute_minimum_eigenvalue(initial_params={theta: 0.5})
    print(f"\nRuntime VQE: eigenvalue {result.eigenvalue:.4f} over "
          f"{len(runtime2.history)} runtime executions")


if __name__ == "__main__":
    main()