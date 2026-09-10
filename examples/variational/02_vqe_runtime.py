"""VQE driving the MQ-04 ExecutionRuntime (MQ-05).

Every energy evaluation is executed through the runtime pipeline
(plan -> backend -> job -> result), and the statevector result is used
for exact, shot-independent expectation values.
"""

from microquantum import EigenvalueProblem
from microquantum.algorithms import VQE
from microquantum.core import Operator, Parameter, QuantumCircuit
from microquantum.optimizers import GradientDescent
from microquantum.runtime import ExecutionRuntime


def main() -> None:
    print("=== VQE through the ExecutionRuntime ===\n")

    theta = Parameter("theta")
    ansatz = QuantumCircuit(1).ry(theta, 0)
    problem = EigenvalueProblem(Operator.Z(), k=1, name="z")

    runtime = ExecutionRuntime()
    vqe = VQE(
        ansatz,
        Operator.Z(),
        GradientDescent(learning_rate=0.3, max_iter=60, tol=1e-8),
        runtime=runtime,
        shots=4096,
        seed=7,
    )
    result = vqe.solve(problem, initial_params={theta: 0.5})

    print(f"Ground-state energy: {result.eigenvalue:.4f} (expected -1.0)")
    print(f"Runtime executions:  {len(runtime.history)}")
    print("Each job produced an enriched result with a statevector "
          "(exact regardless of shots).")


if __name__ == "__main__":
    main()