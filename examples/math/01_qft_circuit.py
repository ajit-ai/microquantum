"""Quantum Fourier transform (QFT): fundamentals (MQ-05).

Builds the QFT of a computational basis state with the standard
textbook decomposition (Hadamard + controlled-Rz + SWAP bit reversal),
and checks that forward-then-inverse QFT recovers the input state.
"""

import numpy as np

from microquantum.algorithms.qft import QFT
from microquantum.core import QuantumCircuit


def main() -> None:
    print("=== QFT fundamentals on 3 qubits ===\n")

    n = 3
    qc = QuantumCircuit(n)
    qc.x(1)  # |2> = |010>

    qft = QFT(num_qubits=n)
    print(f"QFT built by the SDK: {qft.build_circuit().num_gates} gates")

    amps = qft.run_forward(qc.run()).amplitudes
    print("\nQFT of |2> is a flat superposition with basis-dependent phases:")
    for k, v in enumerate(amps):
        if abs(v) > 1e-6:
            print(f"  amplitude at |{k:0{n}b}> = {abs(v):.3f} "
                  f"* exp(i*{np.angle(v):.3f})")

    recovered = qft.run_inverse(qft.run_forward(qc.run())).amplitudes
    max_amp = int(np.abs(recovered).argmax())
    print(f"\nQFT then inverse-QFT recovers |{max_amp:0{n}b}>  "
          f"(fidelity {np.abs(np.vdot(recovered, qc.run().amplitudes)):.6f})")

    print("\nUsed by PhaseEstimation to read eigenphases from & measurement "
          "registers.")


if __name__ == "__main__":
    main()