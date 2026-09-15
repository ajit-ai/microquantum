"""Example 16: Gradient-descent loop over a parameterized circuit.

Demonstrates the MQ-13 gradient interface end to end:
- A small PauliSum Hamiltonian and a parameterized RY ansatz.
- The analytic ``gradient()`` function plugged into a gradient-aware
  optimizer as ``gradient_fn`` (the same contract VQE consumes).
- A manual classical -> quantum -> classical loop with ``run_hybrid``
  style parameter updates using the analytic gradient.

This is the pattern that unlocks gradient-based VQE/QAOA on top of the
MQ-11 execution core and the MQ-12 parameterized circuits.
"""

import numpy as np

from microquantum import (
    Parameter,
    PauliString,
    PauliSum,
    QuantumCircuit,
    gradient,
)
from microquantum.optimizers import Adam

# Hamiltonian: H = Z Z + 0.5 Z on qubit 0 + 0.25 X on qubit 1.
observable = PauliSum(
    [
        PauliString("ZZ", 1.0),
        PauliString("ZI", 0.5),
        PauliString("IX", 0.25),
    ]
)

# Ansatz: Ry(theta) on qubit 0 entangles with qubit 1 via a bell pair,
# then an rx rotation over phi.
theta = Parameter("theta")
phi = Parameter("phi")
ansatz = QuantumCircuit(2).ry(theta, 0).h(1).cx(1, 0).rx(phi, 1)


def energy(values) -> float:
    """Exact expectation of the Hamiltonian (independent of gradient())."""
    state = ansatz.bind_parameters(values).run()
    return sum(PauliString(term.label, term.coefficient).expectation(state)
               for term in observable.terms)


def grad_fn(values) -> dict[Parameter, float]:
    """Analytic parameter-shift gradient of the energy landscape."""
    return gradient(ansatz, observable, values)


print("== energy landscape sanity check ==")
for point in (0.0, np.pi / 4, np.pi / 2):
    print(f"  E({point:+.4f}, {point:+.4f}) = {energy({theta: point, phi: point}):+.6f}")

print("\n== Adam with analytic gradient_fn ==")
result = Adam(learning_rate=0.05, max_iter=120).minimize(
    cost_fn=energy,
    gradient_fn=grad_fn,
    initial_params={theta: 0.4, phi: 0.2},
)
print(f"converged: {result.converged} in {result.iterations} iterations")
print(f"optimal parameters: {result.optimal_parameters}")
print(f"optimal energy: {result.optimal_value:+.8f}")
print(f"first three energies: {[f'{v:+.6f}' for v in result.history[:3]]}")

print("\n== manual loop using the analytic gradient ==")
values = {theta: 0.8, phi: 0.1}
step = 0.3
for iteration in range(6):
    g = grad_fn(values)
    values = {p: v - step * g.get(p, 0.0) for p, v in values.items()}
    print(
        f"  iter {iteration}: theta={values[theta]:+.4f} phi={values[phi]:+.4f} "
        f"E={energy(values):+.6f}"
    )