"""Example 15: Analytic gradients with the parameter-shift rule.

Demonstrates:
- Symbolic derivatives: ``Parameter.gradient()`` and
  ``ParameterExpression.gradient()`` return the chain-rule coefficient.
- ``gradient()`` / ``parameter_shift_gradient()``: exact analytic
  derivative of an expectation value via the parameter-shift rule.
- Observables: ``Operator``, ``PauliString`` and ``PauliSum`` (no dense
  matrices required).
- ``backend=``: evaluate the shifted circuits through the MQ-11/12
  execution core (``StatevectorBackend``), exact and seed-reproducible.

The parameter-shift rule for a rotation gate:

    d<O>/d(theta) = (<O>(theta + s) - <O>(theta - s)) / (2 * sin(s))

with the default shift s = pi/2. For an angle expression ``a*theta + c``
each occurrence contributes ``a`` times the shift difference (chain rule).
"""

import numpy as np

from microquantum import (
    Operator,
    Parameter,
    PauliString,
    PauliSum,
    QuantumCircuit,
    StatevectorBackend,
    gradient,
    parameter_shift_gradient,
)
from microquantum.core.measurement import expectation_value

theta = Parameter("theta")
phi = Parameter("phi")

print("== symbolic derivatives ==")
print(f"theta.gradient()            = {theta.gradient()}")
print(f"(2 * theta).gradient()      = {(2 * theta).gradient()}")
print(f"(-theta).gradient()         = {(-theta).gradient()}")
print(f"(2*theta+0.5).gradient(phi) = {(2 * theta + 0.5).gradient(phi)}")

print("\n== analytic gradient vs closed form ==")
# Ry(theta)|0>: <Z> = cos(theta) => d/dtheta = -sin(theta)
qc = QuantumCircuit(1).ry(theta, 0)
val = np.pi / 4
grad_shift = parameter_shift_gradient(qc, Operator.Z(), theta, {theta: val})
print(
    f"d<Z>/dtheta at {val:.4f} = {grad_shift:.6f} "
    f"(exact {-np.sin(val):.6f})"
)

print("\n== PauliSum observable (no dense matrices) ==")
# H = 0.5 X - 0.3 Z on a single qubit.
obs = PauliSum([PauliString("X", 0.5), PauliString("Z", -0.3)])
e = 0.6
g_obs = parameter_shift_gradient(qc, obs, theta, {theta: e})
closed = 0.5 * np.cos(e) + 0.3 * np.sin(e)
print(f"d<H>/dtheta = {g_obs:.6f} (exact {closed:.6f})")

print("\n== full gradient vector over a multi-qubit circuit ==")
qc2 = QuantumCircuit(2).ry(theta, 0).rx(phi, 1).cx(0, 1)
params = {theta: 0.5, phi: 0.9}
grads = gradient(qc2, PauliSum.from_label("ZZ"), params)

print("central-difference reference:")
h = 1e-6
zz_operator = PauliSum.from_label("ZZ").to_operator()


def energy(mapping):
    state = qc2.bind_parameters(mapping).run()
    return expectation_value(state, zz_operator, targets=[0, 1])


for p in params:
    plus = dict(params)
    minus = dict(params)
    plus[p] += h
    minus[p] -= h
    ref = (energy(plus) - energy(minus)) / (2 * h)
    print(f"  d/d{p.name} = shift {grads[p]:+.8f} | finite-diff {ref:+.8f}")

print("\n== backend-integrated evaluation ==")
via_backend = gradient(
    qc2,
    PauliSum.from_label("ZZ"),
    params,
    backend=StatevectorBackend(),
    seed=7,
)
for p in params:
    print(
        f"  d/d{p.name} via backend = {via_backend[p]:+.8f} "
        f"(engine {grads[p]:+.8f})"
    )