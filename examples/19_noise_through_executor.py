"""Example 19: Noise through the Executor.

Demonstrates:
- Appending a noise model through ``Executor(noise_model=...)``
- Deterministic ``shots=None``: exact noisy probabilities, no sampling
- Seeded noisy sampling for Monte-Carlo estimates
- Calibrated channels (bit flip, amplitude damping, depolarizing)
- Explicit failure when a noise model is combined with a backend

``Example 5`` ran noise on the density-matrix backend directly; this example
uses the uniform ``Executor`` API so the same channels work on circuits.
"""

import numpy as np

from microquantum import (
    DensityMatrixBackend,
    Executor,
    NoiseModel,
    QuantumCircuit,
    StatevectorBackend,
)


def deterministic_noise_probabilities():
    """shots=None returns exact noisy probabilities."""
    print("=== Deterministic noisy probabilities (shots=None) ===\n")
    model = NoiseModel().bit_flip(0.2)
    executor = Executor(noise_model=model)
    result = executor.run(QuantumCircuit(1).x(0), shots=None)
    print(f"  shots={result.shots!r} counts={result.counts}")
    print("  noisy probabilities:", {k: round(v, 4) for k, v in sorted(result.probabilities.items())})


def seeded_noisy_sampling():
    """Sampling under noise is reproducible with a seed."""
    print("=== Seeded noisy sampling ===\n")
    model = NoiseModel().depolarizing(0.3)
    executor = Executor(noise_model=model)
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    a = executor.run(qc, shots=5000, seed=99)
    b = executor.run(qc, shots=5000, seed=99)
    print(f"  seed=99 -> {dict(sorted(a.counts.items()))}")
    print(f"  seed=99 -> {dict(sorted(b.counts.items()))}  (identical)")


def channel_calibration():
    """Each channel pushes the measured distribution in a known way."""
    print("=== Channel calibration on |1> ===\n")
    gate_matrix = QuantumCircuit(1).x(0).gates[0][0].matrix
    for name, builder in [
        ("bit_flip", lambda: NoiseModel().bit_flip(0.2)),
        ("amplitude_damping", lambda: NoiseModel().amplitude_damping(0.5)),
    ]:
        result = DensityMatrixBackend().run_circuit(
            num_qubits=1,
            gates=[(gate_matrix, [0])],
            shots=None,
            noise_model=builder(),
        )
        diag = np.real(np.diag(result.density_matrix))
        print(f"  {name:18s} P(0)={diag[0]:.3f} P(1)={diag[1]:.3f}")


def conflict_error():
    """backend + noise_model is ambiguous and rejected loudly."""
    print("=== Explicit failure ===\n")
    try:
        Executor(backend=StatevectorBackend(), noise_model=NoiseModel().bit_flip(0.1))
    except ValueError as exc:
        print(f"  Executor(backend=..., noise_model=...) -> ValueError: {exc}")


if __name__ == "__main__":
    deterministic_noise_probabilities()
    seeded_noisy_sampling()
    channel_calibration()
    conflict_error()
    print("\nExample 19 completed!")