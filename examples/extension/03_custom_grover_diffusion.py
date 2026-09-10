"""Extending Grover with a custom oracle and custom diffusion shape (MQ-05).

You can hand GroverSearch *any* oracle callable — even one built from a
non-trivial classical predicate — and plug in a different amplitude
amplification strategy by subclassing `_apply_diffusion`.
"""

import numpy as np

from microquantum import SearchProblem
from microquantum.algorithms import GroverSearch
from microquantum.core import QuantumCircuit


class WeightedDominantGrover(GroverSearch):
    """Grover variant whose diffusion gives more weight to low indices."""

    def _apply_diffusion(self, qc: QuantumCircuit) -> QuantumCircuit:
        n = self.num_qubits
        dim = 2**n
        basis = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            for j in range(dim):
                basis[i, j] = 2.0 / dim * (1.0 / (1.0 + abs(i - j)))
        block = np.eye(dim, dtype=np.complex128)
        matrix = basis - block  # reflect about a weighted average direction
        from microquantum.core import Operator

        qc.append(Operator(matrix), list(range(n)))
        return qc


def built_in_diffusion():
    from microquantum.algorithms import GroverSearch as GS
    return GS


def main() -> None:
    print("=== Custom Grover components ===\n")

    def oracle(n_qubits: int) -> QuantumCircuit:
        """Built from the SDK oracle-builder language (no matrices)."""
        qc = QuantumCircuit(n_qubits)
        mark = 9  # to reproduce, use the same target in SearchProblem
        pattern = [(mark >> q) & 1 for q in range(n_qubits)]
        for q, bit in enumerate(pattern):
            if bit == 0:
                qc.x(q)
        from microquantum.core import Operator

        m = np.eye(2**n_qubits, dtype=complex)
        m[2**n_qubits - 1, 2**n_qubits - 1] = -1.0
        qc.append(Operator(m), list(range(n_qubits)))
        for q, bit in enumerate(pattern):
            if bit == 0:
                qc.x(q)
        return qc

    problem = SearchProblem(num_qubits=4, target=9, name="extended-grover")
    grover = WeightedDominantGrover(num_qubits=4, target=oracle)
    result = grover.solve(problem, seed=3)
    print(f"  custom-diffusion Grover: most probable = {result.most_probable}, "
          f"p = {result.success_probability:.3f}")


if __name__ == "__main__":
    main()