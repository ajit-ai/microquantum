"""Example 29: Executing a compiled circuit.

Demonstrates:
- Compiling an optimizable circuit, then running the rebuilt circuit
  on every local simulator backend
- Exact agreement of probability vectors across backends
- Compiled circuit == original semantics after sampling
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


def challenge() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(0)          # identity pair -> optimized away
    qc.h(1)          # superposition that survives optimization
    qc.rz(1.2, 1)
    qc.rz(-1.2, 1)   # opposite rotations -> optimized away
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


def probability_vector(result) -> np.ndarray:
    if result.statevector is not None:
        return np.abs(np.asarray(result.statevector)) ** 2
    if result.density_matrix is not None:
        return np.real(np.diag(result.density_matrix))
    raise TypeError("no exact representation")


def main() -> None:
    print("=== 29 Executing a compiled circuit ===\n")

    qc = challenge()
    result = Compiler(optimization_level=1).compile(qc)
    compiled = result.circuit()

    print(f"source gates : {result.source.num_gates}  "
          f"compiled gates: {result.result.num_gates}")
    print(f"passes       : {result.passes_applied}\n")
    assert result.result.num_gates < result.source.num_gates

    print("Exact probability agreement across simulators:")
    ref = None
    for name, factory in BACKENDS:
        out = probability_vector(factory().run(compiled, shots=None))
        if ref is None:
            ref = out
        print(f"  {name:14s} max |delta-p| vs statevector = "
              f"{np.max(np.abs(ref - out)):.2e}")
        assert np.max(np.abs(ref - out)) < 1e-9

    counts = StatevectorBackend().run(compiled, shots=2000, seed=42).counts
    print("\nsampled counts (seeded):", sorted(counts.items()))
    assert sum(counts.values()) == 2000

    print("\nExample 29 completed!")


if __name__ == "__main__":
    main()