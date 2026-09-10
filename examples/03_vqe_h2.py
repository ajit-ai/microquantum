"""Example 3: VQE for H2 molecule ground state energy.

Demonstrates:
- Building a parameterized ansatz circuit
- Constructing a Hamiltonian from Pauli terms
- Running VQE with COBYLA (gradient-free) optimizer
- Extracting the ground state energy and optimal parameters

VQE finds the minimum eigenvalue of a Hamiltonian by optimizing
a quantum circuit to minimize <psi(theta)|H|psi(theta)>.
"""

from microquantum.core import QuantumCircuit, Parameter
from microquantum.core.operators import Operator
from microquantum.core.tensor import tensor
from microquantum.optimizers import COBYLA, GradientDescent
from microquantum.algorithms import VQE


def build_ansatz(num_qubits: int, depth: int = 1) -> QuantumCircuit:
    """Build a hardware-efficient ansatz circuit.

    Each layer applies Ry rotations and CNOT entangling gates.
    """
    params = []
    qc = QuantumCircuit(num_qubits)

    for layer in range(depth):
        for q in range(num_qubits):
            p = Parameter(f"theta_{layer}_{q}")
            params.append(p)
            qc.ry(p, q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    return qc


def h2_hamiltonian() -> Operator:
    """Simplified H2 Hamiltonian in the STO-3G basis.

    H = -1.0523 * I + 0.3979 * Z0 - 0.3979 * Z1
        + 0.1809 * Z0Z1 + 0.1809 * X0X1 + 0.1809 * Y0Y1
    """
    II = tensor(Operator.I(), Operator.I())
    ZI = tensor(Operator.Z(), Operator.I())
    IZ = tensor(Operator.I(), Operator.Z())
    ZZ = tensor(Operator.Z(), Operator.Z())
    XX = tensor(Operator.X(), Operator.X())
    YY = tensor(Operator.Y(), Operator.Y())

    H = (
        -1.0523 * II
        + 0.3979 * ZI
        - 0.3979 * IZ
        + 0.1809 * ZZ
        + 0.1809 * XX
        + 0.1809 * YY
    )
    return H


def main():
    print("=== VQE: H2 Ground State Energy ===\n")

    num_qubits = 2
    hamiltonian = h2_hamiltonian()

    # Build ansatz
    ansatz = build_ansatz(num_qubits, depth=2)
    print(f"Ansatz: {ansatz.num_qubits} qubits, {ansatz.num_gates} gates")
    print(f"Parameters: {len(ansatz.parameters)}")

    # VQE with COBYLA (gradient-free)
    vqe = VQE(
        ansatz=ansatz,
        hamiltonian=hamiltonian,
        optimizer=COBYLA(max_iter=100, tol=1e-8),
    )

    result = vqe.compute_minimum_eigenvalue()

    print(f"\n--- Results ---")
    print(f"Ground state energy: {result.eigenvalue:.6f} Ha")
    print(f"Exact value:         -1.857275 Ha")
    print(f"Error:               {abs(result.eigenvalue - (-1.857275)):.6f} Ha")

    print(f"\nOptimal parameters:")
    for p, v in sorted(result.eigenstate.items(), key=lambda x: x[0].name):
        print(f"  {p.name}: {v:.6f}")

    # Also show optimizer history
    if result.optimizer_result:
        print(f"\nOptimizer: {result.optimizer_result.iterations} iterations")
        print(f"Final energy: {result.optimizer_result.optimal_value:.6f}")


if __name__ == "__main__":
    main()
