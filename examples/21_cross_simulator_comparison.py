"""Example 21: Cross-simulator comparison.

Demonstrates:
- One circuit run on all four simulators (statevector, density matrix,
  MPS, TTN) with ``shots=None``
- Exact agreement of the resulting probability vectors
- Same-seed sampling parity between the state-vector and TTN backends
- Statistical consistency of seeded sampling across simulators
- Advertised capabilities and targets per simulator

This is the "which simulator should I use?" story: all four agree on the
physics; they differ in memory scaling, noise support and feature breadth.
"""

from __future__ import annotations

import numpy as np

from microquantum import (
    DensityMatrixBackend,
    MPSBackend,
    QuantumCircuit,
    StatevectorBackend,
    TreeTensorNetworkBackend,
)
from microquantum.backends.capabilities import EXECUTION_DENSITY_MATRIX

BACKENDS = [
    ("statevector", lambda: StatevectorBackend()),
    ("density_matrix", lambda: DensityMatrixBackend()),
    ("mps", lambda: MPSBackend()),
    ("ttn", lambda: TreeTensorNetworkBackend()),
]


def challenge():
    qc = QuantumCircuit(4)
    qc.ry(0.9, 0)
    qc.ry(1.7, 1)
    qc.cnot(0, 1)
    qc.cnot(1, 2)
    qc.ry(0.4, 2)
    qc.cnot(2, 3)
    qc.ry(2.2, 3)
    return qc


def probability_vector(result):
    if result.statevector is not None:
        return np.abs(result.statevector) ** 2
    if result.density_matrix is not None:
        return np.real(np.diag(result.density_matrix))
    raise TypeError("no exact representation")


def exact_agreement():
    """All four simulators produce the same exact probabilities."""
    print("=== Exact deterministic agreement (shots=None) ===\n")
    qc = challenge()
    vectors = {}
    for name, factory in BACKENDS:
        result = factory().run(qc, shots=None)
        vectors[name] = probability_vector(result)
    ref = vectors["statevector"]
    for name, vec in vectors.items():
        print(f"  {name:14s} max |delta-p| = {np.max(np.abs(ref - vec)):.2e}")


def seeded_parity():
    """State-vector and TTN share a sampling path; seed parity is exact."""
    print("=== Same-seed sampling ===\n")
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    sv = StatevectorBackend().run(qc, shots=1000, seed=42).counts
    ttn = TreeTensorNetworkBackend().run(qc, shots=1000, seed=42).counts
    print(f"  statevector: {sorted(sv.items())}")
    print(f"  ttn:         {sorted(ttn.items())}  (identical: {sv == ttn})")


def statistical_consistency():
    """MPS peeling sampling uses a different RNG stream: statistical match."""
    print("=== Statistical sampling consistency ===\n")
    qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(1, 2)
    ref = StatevectorBackend().run(qc, shots=4000, seed=10).counts
    for name, factory in BACKENDS:
        counts = factory().run(qc, shots=4000, seed=10).counts
        worst = max(abs(counts.get(k, 0) / 4000 - ref[k] / 4000) for k in ref)
        print(f"  {name:14s} worst |delta-p| over outcomes = {worst:.3f}")


def simulator_portrait():
    """Capabilities and targets tell you which simulator fits the job."""
    print("=== Capabilities & targets ===\n")
    for name, factory in BACKENDS:
        backend = factory()
        caps = backend.capabilities
        print(
            f"  {name:14s} engine={caps.metadata.get('engine'):>14s} "
            f"target={backend.target.name:<22s} "
            f"density_exec={caps.supports_execution(EXECUTION_DENSITY_MATRIX)}"
        )


if __name__ == "__main__":
    exact_agreement()
    seeded_parity()
    statistical_consistency()
    simulator_portrait()
    print("\nExample 21 completed!")