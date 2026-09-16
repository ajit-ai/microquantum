"""Example 17: State-vector simulation in depth.

Demonstrates:
- Running circuits on the ``StatevectorBackend``
- Deterministic execution with ``shots=None`` (exact amplitudes, no sampling)
- Seeded sampling and reproducibility
- Parameterized circuits bound before execution
- Measurement-subset results
- Explicit memory-budget failure for overly large dense states

The state-vector simulator keeps the full 2**n complex amplitudes in memory,
so the SDK refuses allocations above a 2 GiB budget instead of crashing.
"""

import numpy as np

from microquantum import (
    DensityMatrixBackend,
    Parameter,
    QuantumCircuit,
    StateVector,
    StatevectorBackend,
)


def ghz(num_qubits):
    """GHZ state: (|0...0> + |1...1>) / sqrt(2)."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    return qc


def deterministic_execution():
    """shots=None returns exact amplitudes instead of sampling."""
    print("=== Deterministic execution (shots=None) ===\n")
    backend = StatevectorBackend()
    result = backend.run(ghz(4), shots=None)
    print(f"  shots={result.shots!r}  counts={result.counts}  samples={result.samples!r}")
    probs = np.abs(result.statevector) ** 2
    print("  exact probabilities: " + ", ".join(f"|{i:04b}>={p:.3f}" for i, p in enumerate(probs) if p > 1e-9))


def seeded_sampling():
    """Same seed reproduces the same sampled counts."""
    print("=== Seeded sampling ===\n")
    backend = StatevectorBackend()
    first = backend.run(ghz(4), shots=2000, seed=123)
    second = backend.run(ghz(4), shots=2000, seed=123)
    different = backend.run(ghz(4), shots=2000, seed=7)
    print(f"  seed=123 -> {dict(sorted(first.counts.items()))}")
    print(f"  seed=123 -> {dict(sorted(second.counts.items()))}  (identical)")
    print(f"  seed=7   -> {dict(sorted(different.counts.items()))}  (afresh)")


def parameterized_circuit():
    """A single ry gate driven by a symbolic parameter."""
    print("=== Parameterized circuit ===\n")
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    backend = StatevectorBackend()
    at_zero = backend.run(qc, parameter_values={theta: 0.0}, shots=500)
    at_pi = backend.run(qc, parameter_values={theta: np.pi}, shots=500)
    print(f"  theta=0.0  -> {at_zero.counts}")
    print(f"  theta=pi   -> {at_pi.counts}")


def measurement_subset():
    """Only measured qubits appear in the counts."""
    print("=== Measurement subset ===\n")
    qc = ghz(3)
    qc.measure(0)
    result = StatevectorBackend().run(qc, shots=400)
    print("  measured_qubits:", result.metadata.get("measured_qubits"))
    print("  counts:", sorted(result.counts.items()))


def memory_budget():
    """Huge dense states are refused up front, not at allocation time."""
    print("=== Dense-memory budget ===\n")
    try:
        StateVector(28)  # 4 GiB, twice the 2 GiB budget
    except ValueError as exc:
        print(f"  StateVector(28) refused: {exc}")
    ok = DensityMatrixBackend()
    print(f"  {ok.target.name} still fine (uses the budget for vectors, not matrices)")


if __name__ == "__main__":
    deterministic_execution()
    seeded_sampling()
    parameterized_circuit()
    measurement_subset()
    memory_budget()
    print("\nExample 17 completed!")