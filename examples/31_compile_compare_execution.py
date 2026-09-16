"""Example 31: Comparing execution of compiled vs original circuits.

Runs the same circuit through the compiler at every optimization level and
on every local simulator, proving that compiled programs preserve the
*observable* behavior of the original:

- exact probability vectors on ``shots=None`` agree within 1e-9,
- seeded sampling estimates agree within a principled bound.

This is the differential oracle also enforced by the MQ-16 test suites.
"""

from __future__ import annotations

import numpy as np

from microquantum import (
    Compiler,
    DensityMatrixBackend,
    MPSBackend,
    QuantumCircuit,
    StatevectorBackend,
    TreeTensorNetworkBackend,
)

BACKENDS = [
    ("statevector", lambda: StatevectorBackend()),
    ("density_matrix", lambda: DensityMatrixBackend()),
    ("mps", lambda: MPSBackend()),
    ("ttn", lambda: TreeTensorNetworkBackend()),
]


def swap_dressed() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.swap(0, 1)      # lowers to 3 controlled-NOTs (incl. directed cx(1,0))
    qc.rz(-1.1, 1)
    qc.rz(0.5, 1)     # same-axis rotations fused at level 2
    return qc


def probability_vector(result) -> np.ndarray:
    if result.statevector is not None:
        return np.abs(np.asarray(result.statevector)) ** 2
    if result.density_matrix is not None:
        return np.real(np.diag(result.density_matrix))
    raise TypeError("no exact representation")


def main() -> None:
    print("=== 31 Comparing execution of compiled vs original ===\n")

    qc = swap_dressed()
    print(f"source gates: {len(qc.gates)}  (swap + rotations + negative angle)\n")

    for level in (0, 1, 2):
        compiled = Compiler(optimization_level=level).compile(qc)
        print(f"level {level}: passes={compiled.passes_applied}  "
              f"gates {compiled.source.num_gates} -> {compiled.result.num_gates}")

        exact_ok = True
        ref = probability_vector(StatevectorBackend().run(qc, shots=None))
        tracking = {name: None for name, _ in BACKENDS}
        for name, factory in BACKENDS:
            rebuilt = factory().run(compiled.circuit(), shots=None)
            out = probability_vector(rebuilt)
            delta = np.max(np.abs(ref - out))
            tracking[name] = delta
            if delta > 1e-9:
                exact_ok = False
        print(f"  max |delta-p| vs original (exact): "
              f"{max(tracking.values()):.2e}")
        assert exact_ok, "exact probabilities diverged after compilation"

        # Seeded statistical agreement on one representative backend
        backend = StatevectorBackend()
        shots, seed = 4000, 11
        po = normalize(backend.run(qc, shots=shots, seed=seed).counts)
        pc = normalize(backend.run(compiled.circuit(), shots=shots, seed=seed).counts)
        worst = max((abs(po.get(o, 0.0) - pc.get(o, 0.0))
                     for o in set(po) | set(pc)), default=0.0)
        print(f"  max |delta-p| vs original (sampled): {worst:.3f}")
        assert worst <= 0.05

    print("\nOriginal and compiled circuits are observably equivalent "
          "at every level and backend.")

    print("\nExample 31 completed!")


def normalize(counter) -> dict[str, float]:
    total = sum(counter.values())
    return {bits: count / total for bits, count in counter.items()}


if __name__ == "__main__":
    main()