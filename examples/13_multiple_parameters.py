"""Example 13: Multiple parameters & parameter expressions.

Demonstrates:
- Several symbolic parameters in one circuit
- Parameter expressions like ``2 * phi + 0.5`` in the angle slot
- Partial binding (bind a subset, keep the rest symbolic)
- Strict validation: unknown / non-numeric bindings fail loudly

Binding policy (MQ-12):
- Every value is validated before any gate is substituted.
- Unknown parameters that are not in the circuit raise ``ValueError``.
- The original circuit is never modified: ``bind_parameters`` returns a
  new circuit.
"""

import numpy as np

from microquantum import Parameter, QuantumCircuit, StatevectorBackend

theta = Parameter("theta")
phi = Parameter("phi")

qc = QuantumCircuit(2).ry(theta, 0).rz(2 * phi + 0.5, 1).cx(0, 1).ry(phi, 0)
print(f"parameters (deterministic order): {[p.name for p in qc.parameters]}")

partial = qc.bind_parameters({phi: 0.2})
print(f"after binding phi=0.2 -> unbound: {[p.name for p in partial.parameters]}")

full = partial.bind_parameters({"theta": np.pi / 2})
print(f"after full binding -> is_parameterized: {full.is_parameterized}")

result = StatevectorBackend().run(full, shots=1000, seed=7)
print(f"counts: {result.get_counts()}")

for bad in (
    {"does_not_exist": 1.0},
    {"theta": "not-a-number"},
    {theta: 1.0, "theta": 2.0},
):
    try:
        qc.bind_parameters(bad)
    except (TypeError, ValueError) as exc:
        print(f"rejected {bad} -> {type(exc).__name__}: {exc}")