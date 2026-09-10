"""Hamiltonian expectation with the PauliSum formalism (MQ-05).

The expectation-value plumbing that energy-based algorithms rely on:
expectations of Pauli sums over explicit states, the energy landscape an
optimizer walks, and the bitstring-energy view of OptimizationProblem.
"""

import numpy as np

from microquantum import OptimizationProblem
from microquantum.core import Parameter, PauliString, PauliSum, QuantumCircuit


def main() -> None:
    print("=== Hamiltonian expectations ===\n")

    ham = PauliSum([PauliString("Z", 0.5), PauliString("X", 0.5)])

    theta = Parameter("theta")
    for value in (0.0, np.pi / 2, np.pi):
        state = QuantumCircuit(1).ry(theta, 0).bind_parameters({theta: value}).run()
        print(f"  E[H] at theta={value:6.3f}: {ham.expectation(state): .4f}")

    print("\nEnergy landscape along the ansatz drives the optimizer:")
    energies = [
        ham.expectation(
            QuantumCircuit(1).ry(theta, 0).bind_parameters({theta: value}).run()
        )
        for value in np.linspace(-np.pi, np.pi, 9)
    ]
    print(f"  sampled energies: {[f'{e: .2f}' for e in energies]}")

    print("\nBitstring-energy view (OptimizationProblem.energy):")
    problem = OptimizationProblem.from_ising(
        PauliSum([PauliString("Z", -1.0)]), name="one-spin"
    )
    for b in (0.0, 1.0):
        print(f"  energy(bits={int(b)}) = {problem.energy(np.array([b])):.2f}")


if __name__ == "__main__":
    main()