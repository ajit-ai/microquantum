"""Example 14: Parameter sweeps.

Demonstrates:
- Running the same parameterized circuit across a grid of values
- The low-level loop: bind each value, run on a backend, collect counts
- The runtime helper ``run_parameter_sweep`` (one call, one result per value)
- Seeded reproducibility of every execution

``theta = 0`` prepares |0>; ``theta = pi/2`` prepares (|0>+|1>)/sqrt(2);
``theta = pi`` prepares |1> — the sweep makes the transition visible.
"""

import numpy as np

from microquantum import (
    Parameter,
    QuantumCircuit,
    StatevectorBackend,
    run_parameter_sweep,
)

theta = Parameter("theta")
qc = QuantumCircuit(1).ry(theta, 0).measure_all()

values = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
backend = StatevectorBackend()

print("manual bind + run loop:")
for value in values:
    bound = qc.bind_parameters({theta: value})
    counts = backend.run(bound, shots=1024, seed=42).get_counts()
    print(f"  theta={value:6.3f}  P(1)={counts.get('1', 0) / 1024:.3f}")

print("\nrun_parameter_sweep (values are per-parameter floats):")
results = run_parameter_sweep(qc, values, shots=1024, seed=42)
for value, result in zip(values, results, strict=True):
    counts = result.get_counts()
    print(f"  theta={value:6.3f}  P(1)={counts.get('1', 0) / 1024:.3f}")

print("\nre-running with the same seed reproduces the counts:")
repeat = backend.run(qc.bind_parameters({theta: np.pi / 2}), shots=1024, seed=42)
print("  identical counts:", repeat.get_counts() == results[2].get_counts())