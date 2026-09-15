"""Example 12: Basic parameterized circuit.

Demonstrates:
- Declaring a symbolic parameter with ``Parameter("theta")``
- Building a circuit whose rotation angle is symbolic
- Introspecting the circuit's unbound parameters
- Binding the parameter to a concrete angle and executing on a backend

``Parameter`` is MicroQuantum's first-class symbolic scalar.  Rotation
gates (Rx/Ry/Rz) accept a parameter anywhere a numeric angle is allowed;
``bind_parameters`` returns a *new* circuit with the symbolic value
resolved, leaving the original circuit untouched.
"""

import numpy as np

from microquantum import Parameter, QuantumCircuit, StatevectorBackend
from microquantum.backends.mock import MockBackend

theta = Parameter("theta")
print(f"parameter: {theta!r} (name={theta.name!r})")

qc = QuantumCircuit(1).ry(theta, 0).measure_all()
print(f"circuit: {qc}")
print(f"unbound parameters: {qc.parameters}")
print(f"is_parameterized: {qc.is_parameterized}")

bound = qc.bind_parameters({theta: np.pi / 2})
print(f"\nafter binding theta = pi/2:")
print(f"  bound.is_parameterized: {bound.is_parameterized}")
print(f"  original.is_parameterized: {qc.is_parameterized}  (unchanged)")

for backend in (StatevectorBackend(), MockBackend()):
    result = backend.run(bound, shots=1024, seed=42)
    print(f"{backend.name}: counts={result.get_counts()}")