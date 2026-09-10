"""GroverSearch with a user-supplied oracle (MQ-05).

The oracle in GroverSearch can be *any* callable that builds a
`QuantumCircuit`.  Here we hand-roll a phase-flip oracle for a single
marked state (X-flip onto the mark bit pattern, diagonal phase oracle,
uncompute) instead of relying on the built-in target-based oracle.
"""

import numpy as np

from microquantum import SearchProblem
from microquantum.algorithms import GroverSearch
from microquantum.core import Operator, QuantumCircuit


def phase_flip_oracle(n_qubits: int, mark: int) -> QuantumCircuit:
    """Flip the phase of |mark> only.

    Decomposition: rotate into the diagonal basis of |mark> (X on every
    qubit where the mark stores 0), apply diag(+1,...,+1,-1,+1,...) on
    the control pattern, then undo the X flips.
    """
    oracle = QuantumCircuit(n_qubits)

    pattern = [(mark >> q) & 1 for q in range(n_qubits)]
    for q, bit in enumerate(pattern):
        if bit == 0:
            oracle.x(q)  # prepare |1...1> = the mark pattern

    matrix = np.eye(2**n_qubits, dtype=np.complex128)
    matrix[2**n_qubits - 1, 2**n_qubits - 1] = -1.0
    oracle.append(Operator(matrix), list(range(n_qubits)))

    for q, bit in enumerate(pattern):
        if bit == 0:
            oracle.x(q)  # uncompute
    return oracle


def main() -> None:
    print("=== GroverSearch with a custom oracle ===\n")

    n_qubits, mark = 4, 6
    oracle = phase_flip_oracle(n_qubits, mark)
    print(f"custom oracle: {oracle.num_gates} gates over {n_qubits} qubits")

    problem = SearchProblem(num_qubits=n_qubits, target=mark, name="custom-oracle")
    grover = GroverSearch(
        num_qubits=n_qubits,
        target=lambda n: phase_flip_oracle(n, mark),
        num_iterations=2,
    )
    assert grover.num_iterations == 2

    result = grover.solve(problem, seed=11)
    print(f"most probable: {result.most_probable}  (marked = "
          f"{result.most_probable == mark}, p = "
          f"{result.success_probability:.3f})")


if __name__ == "__main__":
    main()