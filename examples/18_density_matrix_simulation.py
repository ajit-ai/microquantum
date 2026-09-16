"""Example 18: Density-matrix simulation in depth.

Demonstrates:
- Running circuits on the ``DensityMatrixBackend``
- Exact equivalence with the state vector for pure states
- Coherence: off-diagonal density-matrix elements
- Mixed states: a depolarizing channel drops the purity below 1
- Deterministic ``shots=None`` execution and the ``EXECUTION_DENSITY_MATRIX``
  capability

The density-matrix simulator tracks the full 2**n x 2**n matrix, so it can
represent noisy and mixed states that a pure state vector cannot.
"""

from __future__ import annotations

import numpy as np

from microquantum import (
    Backend,
    DensityMatrixBackend,
    NoiseModel,
    QuantumCircuit,
    StatevectorBackend,
)


def bell():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def pure_equivalence(backend: Backend):
    """A pure circuit run both ways must agree on probabilities."""
    print("=== Pure-state equivalence ===\n")
    qc = QuantumCircuit(2).ry(2.1, 0).ry(0.6, 1).cnot(0, 1)
    sv = StatevectorBackend().run(qc, shots=None).statevector
    dm = DensityMatrixBackend().run(qc, shots=None).density_matrix
    probs = np.abs(sv) ** 2
    diag = np.real(np.diag(dm))
    print(f"  max |abs-statevector^2 - diag(density)| = {np.max(np.abs(probs - diag)):.2e}")


def coherence():
    """Off-diagonal elements reveal superposition/entanglement."""
    print("=== Coherence (off-diagonal elements) ===\n")
    dm = DensityMatrixBackend().run(bell(), shots=None).density_matrix
    print(f"  Bell-state |rho[0, 3]| = {abs(dm[0, 3]):.3f}  (entangled)")
    single = DensityMatrixBackend().run(QuantumCircuit(1).h(0), shots=None).density_matrix
    print(f"  |+><+| |rho[0, 1]|     = {abs(single[0, 1]):.3f}  (superposition)")


def mixed_state():
    """Noise produces a mixed state with purity strictly below 1."""
    print("=== Noise -> mixed state ===\n")
    model = NoiseModel().depolarizing(0.15)
    qc = bell()
    result = DensityMatrixBackend().run_circuit(
        num_qubits=2,
        gates=[(op.matrix, targets) for op, targets in qc.gates],
        shots=None,
        noise_model=model,
    )
    rho = result.density_matrix
    purity = np.real(np.trace(rho @ rho))
    diag = np.real(np.diag(rho))
    print(f"  purity = {purity:.4f}  (pure Bell = 1.0)")
    print("  noisy diagonal: " + ", ".join(f"|{i:02b}>={diag[i]:.3f}" for i in range(4) if diag[i] > 1e-3))


def deterministic_and_capability():
    """Deterministic execution plus the advertised execution mode."""
    print("=== Deterministic + capability ===\n")
    result = DensityMatrixBackend().run(bell(), shots=None)
    print(f"  shots={result.shots!r} counts={result.counts} samples={result.samples!r}")
    from microquantum.backends.capabilities import EXECUTION_DENSITY_MATRIX

    supports = DensityMatrixBackend().capabilities.supports_execution(EXECUTION_DENSITY_MATRIX)
    print(f"  supports EXECUTION_DENSITY_MATRIX: {supports}")


if __name__ == "__main__":
    pure_equivalence(StatevectorBackend())
    coherence()
    mixed_state()
    deterministic_and_capability()
    print("\nExample 18 completed!")