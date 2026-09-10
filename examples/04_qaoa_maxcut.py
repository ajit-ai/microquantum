"""Example 4: QAOA for MaxCut problem.

Demonstrates:
- Encoding a graph problem as a QAOA circuit
- Building the cost and mixer Hamiltonians
- Running QAOA with the COBYLA optimizer
- Analyzing the solution quality

MaxCut: partition graph vertices into two sets to maximize
the number of edges between the sets.
"""

from microquantum.core import QuantumCircuit, Parameter
from microquantum.core.operators import Operator
from microquantum.core.tensor import tensor
from microquantum.optimizers import COBYLA
from microquantum.algorithms import QAOA


def build_maxcut_hamiltonian(
    edges: list[tuple[int, int]],
    num_qubits: int,
) -> Operator:
    """Build the cost Hamiltonian for MaxCut.

    For each edge (i, j), add a ZZ term: (1 - ZiZj) / 2
    """
    terms = []
    II = Operator.I()

    for i, j in edges:
        # Build ZiZj
        ops = [II] * num_qubits
        ops[i] = Operator.Z()
        ops[j] = Operator.Z()
        ZZ = ops[0]
        for k in range(1, num_qubits):
            ZZ = tensor(ZZ, ops[k])
        terms.append(0.5 * (tensor(*[Operator.I()] * num_qubits) - ZZ))

    H = terms[0]
    for t in terms[1:]:
        H = H + t
    return H


def build_maxcut_circuit(
    edges: list[tuple[int, int]],
    num_qubits: int,
    depth: int = 2,
) -> tuple[QuantumCircuit, list[Parameter]]:
    """Build a QAOA circuit for MaxCut.

    Layer structure:
    1. Uniform superposition (H on all qubits)
    2. Cost unitary: exp(-i * gamma * C)
    3. Mixer unitary: exp(-i * beta * B)
    """
    qc = QuantumCircuit(num_qubits)
    params = []

    # Initial superposition
    for q in range(num_qubits):
        qc.h(q)

    for layer in range(depth):
        gamma = Parameter(f"gamma_{layer}")
        beta = Parameter(f"beta_{layer}")
        params.extend([gamma, beta])

        # Cost unitary: ZZ rotations for each edge
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(gamma, j)
            qc.cx(i, j)

        # Mixer unitary: X rotations on all qubits
        for q in range(num_qubits):
            qc.rx(beta, q)

    return qc, params


def evaluate_cut(
    bitstring: list[int],
    edges: list[tuple[int, int]],
) -> int:
    """Evaluate the cut value for a given bitstring partition."""
    cut = 0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            cut += 1
    return cut


def main():
    print("=== QAOA: MaxCut Problem ===\n")

    # Create a triangle + chain graph:  0--1--2--3
    #                                   |     |
    #                                   +--4--+
    num_qubits = 5
    edges = [(0, 1), (1, 2), (2, 3), (0, 4), (3, 4)]

    print(f"Graph: {num_qubits} vertices, {len(edges)} edges")
    print(f"Edges: {edges}")

    # Build QAOA circuit
    depth = 2
    qc, params = build_maxcut_circuit(edges, num_qubits, depth)
    print(f"\nQAOA circuit: {qc.num_qubits} qubits, {qc.num_gates} gates")
    print(f"Parameters: {len(params)}")

    # Build cost Hamiltonian
    H = build_maxcut_hamiltonian(edges, num_qubits)

    # Run VQE with the QAOA circuit
    from microquantum.algorithms import VQE

    vqe = VQE(ansatz=qc, hamiltonian=H, optimizer=COBYLA(max_iter=200))
    result = vqe.compute_minimum_eigenvalue()

    print(f"\n--- Results ---")
    print(f"Ground state energy: {result.eigenvalue:.4f}")
    print(f"(Negative energy = high cut value)")

    # Show the optimal state
    bound = qc.bind_parameters(result.eigenstate)
    state = bound.run()
    probs = abs(state.amplitudes) ** 2

    print(f"\nMeasurement probabilities (top 5):")
    prob_list = list(enumerate(probs))
    prob_list.sort(key=lambda x: -x[1])
    for idx, (i, p) in enumerate(prob_list[:5]):
        bitstring = [(i >> q) & 1 for q in range(num_qubits)]
        cut_val = evaluate_cut(bitstring, edges)
        print(f"  |{i:0{num_qubits}b}>: {p:.4f} (cut={cut_val})")


if __name__ == "__main__":
    main()
