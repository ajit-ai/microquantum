"""Problem hierarchy overview (MQ-05).

Demonstrates the five generic problem abstractions: what they describe,
how they validate, and how they serialize.  Problems never execute
anything — they are plain, JSON-safe data handed to algorithms.
"""

from microquantum import (
    EigenvalueProblem,
    HamiltonianProblem,
    OptimizationProblem,
    SamplingProblem,
    SearchProblem,
)
from microquantum.core import Operator, PauliString, PauliSum, QuantumCircuit
from microquantum.optimization.qubo import QUBOBuilder


def header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    header("1. SamplingProblem — draw samples from a circuit's distribution")
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0, 1)  # Bell state
    sampling = SamplingProblem(circ, num_samples=2048)
    print(f"  {sampling}")
    print(f"  validate: {sampling.validate()} (empty = ok)")

    header("2. OptimizationProblem — minimize over binary variables")
    builder = QUBOBuilder(num_variables=3)
    builder.add_quadratic(0, 1, 1.0)
    builder.add_quadratic(1, 2, -1.0)
    builder.add_linear(0, -0.5)
    opt = OptimizationProblem.from_qubo(builder.build(), name="maxcut-tiny")
    print(f"  {opt}")
    print(f"  qubits needed (exact solver): {opt.qubits_needed}")
    print(f"  Ising cost Hamiltonian: {opt.cost_hamiltonian()}")

    header("3. HamiltonianProblem — the spectrum of a Hermitian operator")
    ham = HamiltonianProblem(Operator.Z(), name="z")
    print(f"  {ham}, acts on {ham.num_qubits} qubit(s)")

    header("4. EigenvalueProblem — how many low eigenvalues are wanted")
    eig = EigenvalueProblem(PauliSum([PauliString("Z", 1.0)]), k=2, name="z-k2")
    print(f"  {eig}: k = {eig.k}")

    header("5. SearchProblem — find marked state(s) in a database")
    search = SearchProblem(num_qubits=3, target=[6, 7], name="last-two")
    print(f"  {search}")
    print(f"  marked indices: {search.target_indices()}")

    header("Serialization")
    for problem in (sampling, opt, ham, eig, search):
        dumped = problem.to_json().replace("\n", " ")
        print(f"  {problem.to_dict()['type'] + 'Problem':20s} -> {dumped[:90]}...")


if __name__ == "__main__":
    main()