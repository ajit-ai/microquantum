"""VQE ground-state estimation on a 2-qubit Hamiltonian (MQ-05).

Uses the generic EigenvalueProblem abstraction and a parameterized
entangled ansatz.
"""

from microquantum import EigenvalueProblem
from microquantum.algorithms import VQE
from microquantum.core import Operator, Parameter, QuantumCircuit, tensor
from microquantum.optimizers import Adam


def main() -> None:
    print("=== VQE on a 2-qubit ZZ + X hamiltonian ===\n")

    # H = alpha Z0 Z1 + beta X0  with alpha = 0.6, beta = 0.8
    alpha, beta = 0.6, 0.8
    hamiltonian = (
        alpha * tensor(Operator.Z(), Operator.Z())
        + beta * tensor(Operator.X(), Operator.I())
    )
    problem = EigenvalueProblem(hamiltonian, k=1, name="zz-x")

    # Exact ground energy (small Hilbert space -> closed form)
    import numpy as np

    exact = float(np.linalg.eigvalsh(hamiltonian.matrix)[0])
    print(f"Exact ground energy: {exact:.4f}")

    # Ansatz: |0>0 -> RY(theta0) |0>1 -> CNOT -> RZ(theta1) on qubit 0
    theta0, theta1 = Parameter("theta0"), Parameter("theta1")
    ansatz = QuantumCircuit(2)
    ansatz.ry(theta0, 0)
    ansatz.ry(np.pi / 2, 1)  # fixed local rotation
    ansatz.cx(0, 1)
    ansatz.rz(theta1, 0)

    vqe = VQE(ansatz, hamiltonian, Adam(learning_rate=0.2, max_iter=300, tol=1e-8))
    result = vqe.solve(
        problem, initial_params={theta0: 1.0, theta1: 0.5}
    )
    print(f"VQE energy:      {result.eigenvalue:.4f}")
    print(f"gap:             {result.eigenvalue - exact:.2e}")
    print(f"validated:       {vqe.validate(problem) == []}")
    print(f"optimizer iters: {result.optimizer_result.iterations}")


if __name__ == "__main__":
    main()