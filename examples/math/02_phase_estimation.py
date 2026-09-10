"""Phase estimation: eigenvalues of a unitary from its eigenstate (MQ-05).

PhaseEstimation reads the phase  phi  such that  U|phi> = e**(2 pi i phi)|phi>
by running the textbook phase estimation algorithm (Hadamard register +
controlled-U's + inverse QFT).
"""

import numpy as np

from microquantum.algorithms import PhaseEstimation
from microquantum.core import Operator, QuantumCircuit


def main() -> None:
    print("=== Phase estimation on engineered unitaries ===\n")

    s_gate = Operator(np.array([[1.0, 0.0], [0.0, 1j]], dtype=np.complex128))
    rz = Operator(
        np.diag(
            [
                1.0,
                np.exp(1j * 2.0 * np.pi * 3 / 8),
            ]
        )
    )

    state_0 = QuantumCircuit(1).run()          # |0>
    state_1 = QuantumCircuit(1).x(0).run()     # |1>

    for label, unitary, eigenstate, expect in (
        ("S gate", s_gate, state_0, 0.25),
        ("Phase(3pi/4)", rz, state_1, 3 / 8),
    ):
        pe = PhaseEstimation(unitary, num_counting_qubits=4)
        result = pe.estimate_from_state(eigenstate, seed=3)
        print(f"  {label:10s}: estimated phase {result.phase:.3f} "
              f"(expected {expect:.3f})")

    print("\nPhaseEstimation.from_problem works on any unitary Operator "
          "Hamiltonian (see docs for the unitary-only restriction).")


if __name__ == "__main__":
    main()