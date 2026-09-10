"""Example 5: Noise simulation with density matrix backend.

Demonstrates:
- Adding noise channels (depolarizing, amplitude damping)
- Running circuits on the density matrix backend
- Comparing ideal vs noisy results
- Using the noise model to simulate realistic quantum hardware

Noise is essential for understanding how quantum algorithms
perform on real hardware with imperfect gates.
"""

import numpy as np
from microquantum.core import QuantumCircuit
from microquantum.core.operators import Operator
from microquantum.core.tensor import expand_operator
from microquantum.backends.noise import NoiseModel
from microquantum.backends.density_matrix import DensityMatrixBackend


def circuit_to_gates(qc: QuantumCircuit) -> list[tuple[np.ndarray, list[int]]]:
    """Convert a QuantumCircuit to the (matrix, targets) format
    expected by the density matrix backend."""
    gates = []
    for op, targets in qc.gates:
        gates.append((op.matrix, targets))
    return gates


def ideal_vs_noisy():
    """Compare ideal vs noisy Bell state preparation."""
    print("=== Ideal vs Noisy Bell State ===\n")

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Ideal simulation
    ideal_state = qc.run()
    ideal_probs = abs(ideal_state.amplitudes) ** 2
    print("Ideal state probabilities:")
    for i, p in enumerate(ideal_probs):
        print(f"  |{i:02b}>: {p:.4f}")

    # Add noise
    noise = NoiseModel()
    noise.depolarizing(probability=0.05)
    noise.amplitude_damping(probability=0.02)

    # Noisy simulation via density matrix backend
    backend = DensityMatrixBackend()
    gates = circuit_to_gates(qc)
    result = backend.run_circuit(
        num_qubits=2,
        gates=gates,
        noise_model=noise,
        shots=10000,
    )

    print("\nNoisy state probabilities (from samples):")
    for bitstring, prob in sorted(result.probabilities.items()):
        print(f"  |{bitstring}>: {prob:.4f}")

    rho_diag = np.real(np.diag(result.density_matrix))
    print("\nNoisy state (from density matrix diagonal):")
    for i, p in enumerate(rho_diag):
        print(f"  |{i:02b}>: {p:.4f}")


def noise_channels_demo():
    """Demonstrate different noise channels."""
    print("\n=== Noise Channels Demo ===\n")

    qc = QuantumCircuit(1)
    qc.h(0)

    ideal_state = qc.run()
    ideal_probs = abs(ideal_state.amplitudes) ** 2
    print("Ideal |+> state:")
    print(f"  |0>: {ideal_probs[0]:.4f}")
    print(f"  |1>: {ideal_probs[1]:.4f}")

    channels = [
        ("Depolarizing (p=0.1)", "depolarizing", 0.1),
        ("Amplitude damping (p=0.2)", "amplitude_damping", 0.2),
        ("Phase damping (p=0.15)", "phase_damping", 0.15),
    ]

    backend = DensityMatrixBackend()
    gates = circuit_to_gates(qc)

    for name, channel_type, prob in channels:
        noise = NoiseModel()
        getattr(noise, channel_type)(probability=prob)

        result = backend.run_circuit(
            num_qubits=1,
            gates=gates,
            noise_model=noise,
            shots=10000,
        )

        rho_diag = np.real(np.diag(result.density_matrix))
        print(f"\n{name}:")
        print(f"  |0>: {rho_diag[0]:.4f}")
        print(f"  |1>: {rho_diag[1]:.4f}")
        print(f"  Purity: {result.metadata.get('purity', 'N/A')}")


def multi_qubit_noise():
    """Noise on a multi-qubit GHZ circuit."""
    print("\n=== Multi-Qubit Noisy Circuit ===\n")

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    ideal = qc.run()
    ideal_probs = abs(ideal.amplitudes) ** 2
    print("Ideal GHZ state:")
    for i, p in enumerate(ideal_probs):
        if p > 0.01:
            print(f"  |{i:03b}>: {p:.4f}")

    noise = NoiseModel()
    noise.depolarizing(probability=0.02)

    backend = DensityMatrixBackend()
    gates = circuit_to_gates(qc)
    result = backend.run_circuit(
        num_qubits=3,
        gates=gates,
        noise_model=noise,
        shots=10000,
    )

    print("\nNoisy GHZ state (from density matrix):")
    rho_diag = np.real(np.diag(result.density_matrix))
    for i, p in enumerate(rho_diag):
        if p > 0.01:
            print(f"  |{i:03b}>: {p:.4f}")

    print(f"\nPurity: {result.metadata.get('purity', 'N/A')}")


if __name__ == "__main__":
    ideal_vs_noisy()
    noise_channels_demo()
    multi_qubit_noise()
    print("\nAll noise examples completed!")
