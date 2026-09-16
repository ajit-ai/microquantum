Analytical Gradients (parameter-shift rule)
===========================================

Variational algorithms (VQE, QAOA, ...) repeatedly minimize an expectation
value over circuit parameters.  The SDK provides *exact* analytic
gradients via the **parameter-shift rule** — no finite differences, no
automatic differentiation — together with symbolic derivatives on
:class:`~microquantum.Parameter` and
:class:`~microquantum.ParameterExpression`.

The parameter-shift rule
------------------------

For a single-qubit rotation gate :math:`U(\\theta)=e^{-i\\theta P/2}`
with Pauli generator :math:`P \\in \\{X, Y, Z\\}`, the derivative of any
expectation value :math:`\\langle O\\rangle` with respect to
:math:`\\theta` is

.. math::

   \\frac{d\\langle O\\rangle}{d\\theta}
     = \\frac{\\langle O\\rangle(\\theta+s)
             - \\langle O\\rangle(\\theta-s)}{2\\,\\sin(s)}\\,,

for any shift :math:`s` that is not an integer multiple of
:math:`\\pi`.  The default :math:`s=\\pi/2` gives the classic
two-term rule.

The estimation
--------------

.. code-block:: python

   from microquantum import Operator, Parameter, QuantumCircuit, gradient

   theta = Parameter("theta")
   qc = QuantumCircuit(1).ry(theta, 0)

   # Scalar derivative d<Z>/d(theta) at theta = pi/4.
   from microquantum import parameter_shift_gradient
   d = parameter_shift_gradient(qc, Operator.Z(), theta, {theta: 0.5})

   # Full gradient vector over every circuit parameter, in the
   # deterministic order of ``qc.parameters``.
   grads = gradient(qc, Operator.Z(), {theta: 0.5})
   print(grads[theta])   # d<Z>/dtheta at theta=0.5

``gradient()`` returns a ``dict`` keyed by the circuit's
:class:`~microquantum.Parameter` objects, so it plugs directly into the
gradient-mode optimizers as ``gradient_fn``
(:class:`~microquantum.optimizers.GradientDescent`,
:class:`~microquantum.optimizers.Adam`,
:class:`~microquantum.optimizers.BFGS`, ...).

Symbolic derivatives
--------------------

A :class:`~microquantum.Parameter` is the identity scalar field in its own
variable; a :class:`~microquantum.ParameterExpression` ``a * p + c`` has
derivative ``a`` with respect to ``p``:

.. code-block:: python

   (2 * theta).gradient()         # 2.0  (chain-rule coefficient)
   (2 * theta).gradient("phi")    # 0.0  (unrelated parameter)
   theta.gradient()               # 1.0

The additive constant never contributes — the derivative only depends on
the coefficient.

Chain rule for expressions
--------------------------

When a gate angle is an expression ``a * theta + c``, each occurrence of
``theta`` contributes ``a`` times its parameter-shift difference:

.. code-block:: python

   qc = QuantumCircuit(1).ry(2 * theta, 0).ry(theta, 0)
   d = parameter_shift_gradient(qc, Operator.Z(), theta, {theta: 0.4})

A parameter used in *several* gates contributes the sum of its
per-occurrence gradients (product rule).  Because gates compose, the exact
gradient of the composed unitary is what is computed — nothing is
approximated.

Observables
-----------

The observable may be:

* a dense :class:`~microquantum.Operator`,
* a :class:`~microquantum.PauliString`, or
* a :class:`~microquantum.PauliSum` (a Hamiltonian),

and may act on a *subset* of the circuit's qubits via ``targets=``:

.. code-block:: python

   from microquantum import PauliString, PauliSum

   phi = Parameter("phi")
   qc2 = QuantumCircuit(2).ry(theta, 0).rz(phi, 1)

   obs = PauliSum([PauliString("X", 0.5), PauliString("Z", -0.3)])
   d = parameter_shift_gradient(qc, obs, theta, {theta: 0.6})

   # Pauli observable on qubit 1 of a 2-qubit circuit.
   d = parameter_shift_gradient(qc2, PauliString("X"), phi,
                                {theta: 0.5, phi: 0.8}, targets=[1])

`PauliString`/`PauliSum` terms are evaluated without building dense
matrices.

Execution through a backend
---------------------------

By default the shifted circuits are simulated with the built-in state
vector engine.  Pass ``backend=`` to route the shifted-circuit evaluations
through the MQ-11/12 execution core (``backend.run``), with optional
``seed`` and ``shots``:

.. code-block:: python

   from microquantum import StatevectorBackend

   grads = gradient(qc, obs, {theta: 0.5},
                    backend=StatevectorBackend(), seed=7)

Expectation values are computed from the exact state vector, so the
gradient is deterministic at any seed.  A backend that cannot return a
state vector raises ``ValueError``.

Requirements & validation
-------------------------

* ``param_values`` must be a ``Mapping`` providing a value for **every**
  circuit parameter (the differentiation point).
* ``shift`` must not be an integer multiple of :math:`\\pi`.
* The observable must be an ``Operator``, ``PauliString`` or ``PauliSum``.
* Parameters are matched **by name**, consistent with the MQ-12 parameter
  model.

Ordering & determinism
----------------------

The gradient vector follows the deterministic, name-sorted order of
:attr:`~microquantum.QuantumCircuit.parameters`, and — because evaluation
uses exact state vectors — is reproducible regardless of RNG seeding.

Examples
--------

* ``examples/15_parameter_shift_gradient.py`` — symbolic derivatives,
  closed-form agreement, ``PauliSum`` observables and backend-integrated
  evaluation.
* ``examples/16_gradient_descent_loop.py`` — a gradient-driven variational
  loop with ``gradient_fn`` (the pattern used by gradient-based VQE/QAOA).

Run them directly:

.. code-block:: console

   python examples/15_parameter_shift_gradient.py
   python examples/16_gradient_descent_loop.py