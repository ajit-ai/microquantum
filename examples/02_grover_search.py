"""Example 2: Grover's search algorithm.

Demonstrates:
- Oracle construction for a specific target state
- Diffusion operator (amplitude amplification)
- Multi-iteration search

Grover's algorithm finds a marked item in an unsorted database
of N items using O(sqrt(N)) queries.
"""

import math
import numpy as np
from microquantum.core import QuantumCircuit
from microquantum.core.operators import Operator
from microquantum.core.tensor import tensor


def oracle(num_qubits: int, target: int) -> QuantumCircuit:
    """Oracle that flips the phase of the target state.

    Constructs the oracle as a diagonal matrix that is -1
    at the target index and +1 everywhere else.
    """
    dim = 2**num_qubits
    matrix = np.eye(dim, dtype=np.complex128)
    matrix[target, target] = -1.0

    qc = QuantumCircuit(num_qubits)
    qc.append(Operator(matrix), list(range(num_qubits)))
    return qc


def diffusion(num_qubits: int) -> QuantumCircuit:
    """Grover diffusion operator: 2|s><s| - I.

    Reflects amplitudes about the mean using H-X-(phase flip)-X-H.
    """
    dim = 2**num_qubits
    # 2|s><s| - I = H * (2|0><0| - I) * H
    matrix = np.full((dim, dim), 2.0 / dim, dtype=np.complex128)
    matrix -= np.eye(dim, dtype=np.complex128)

    qc = QuantumCircuit(num_qubits)
    qc.append(Operator(matrix), list(range(num_qubits)))
    return qc


def search(num_qubits: int, target: int) -> tuple[QuantumCircuit, int]:
    """Run Grover's search for the target state."""
    N = 2**num_qubits
    iterations = max(1, int(math.floor(math.pi / 4 * math.sqrt(N))))

    qc = QuantumCircuit(num_qubits)

    # Uniform superposition
    for i in range(num_qubits):
        qc.h(i)

    # Grover iterations
    for _ in range(iterations):
        qc = qc + oracle(num_qubits, target)
        qc = qc + diffusion(num_qubits)

    return qc, iterations


def main():
    import numpy as np

    num_qubits = 3
    N = 2**num_qubits
    target = 5  # |101>

    print(f"=== Grover's Search ===")
    print(f"Database size: {N} items")
    print(f"Target state: |{target:0{num_qubits}b}>")

    qc, iterations = search(num_qubits, target)
    state = qc.run()
    probs = abs(state.amplitudes) ** 2

    print(f"Iterations: {iterations}")
    print(f"Circuit: {qc.num_qubits} qubits, {qc.num_gates} gates\n")

    print(f"Amplitudes after {iterations} Grover iterations:")
    for i in range(N):
        prob = probs[i]
        marker = " <-- TARGET" if i == target else ""
        print(f"  |{i:0{num_qubits}b}>: {prob:.4f}{marker}")

    print(f"\nTarget probability: {probs[target]:.4f}")


if __name__ == "__main__":
    main()
