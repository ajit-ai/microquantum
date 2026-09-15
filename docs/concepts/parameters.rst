Parameters
==========

Parameters make circuits tunable without rebuilding them.  A
:class:`~microquantum.Parameter` is a named symbolic placeholder; rotation
gates (``rx``/``ry``/``rz``) accept a parameter — or any
:class:`~microquantum.ParameterExpression` built with ``*``, ``+``, ``-`` —
in the angle slot.

Creating and using parameters
-----------------------------

.. code-block:: python

   from microquantum import Parameter, QuantumCircuit

   theta = Parameter("theta")
   phi = Parameter("phi")

   qc = QuantumCircuit(2)
   qc.ry(theta, 0)          # symbolic angle
   qc.rz(2 * phi + 0.1, 1)  # or a parameter expression

   print(qc.parameters)          # deterministic name order -> (phi, theta)
   print(qc.is_parameterized)    # True

``qc.parameters`` is a read-only tuple of the circuit's unbound
parameters, sorted alphabetically by name.  Parameters are identified by
name: the same-name parameter used in several gates appears exactly once.

Creating a parameter expression
-------------------------------

.. code-block:: python

   expr = 2 * theta + 0.5   # ParameterExpression over theta
   qc = QuantumCircuit(1).ry(expr, 0)

Supported operations are ``+``, ``-`` and ``*`` (with numbers), negation,
and scaling.  An expression keeps its math symbolic until the parameter it
depends on is bound.

Binding values
--------------

A circuit whose parameters have all been resolved is "bound" and can be
executed:

.. code-block:: python

   bound = qc.bind_parameters({theta: 0.5, phi: 1.2})
   print(bound.is_parameterized)  # False
   print(bound.get_unitary())

Binding is non-destructive: it returns a *new* circuit and never modifies
the original.  Keys may be :class:`~microquantum.Parameter` objects or
plain strings.

Partial binding (MQ-12 policy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binding a subset of the parameters keeps the rest symbolic; the returned
circuit can be bound again (or executed once fully bound):

.. code-block:: python

   partial = qc.bind_parameters({theta: 0.5})   # phi stays symbolic
   full = partial.bind_parameters({phi: 1.2})

Binding validation
~~~~~~~~~~~~~~~~~~

Binding is strict and fails fast instead of silently ignoring mistakes:

* **Unknown parameters** that match no circuit parameter raise
  ``ValueError``.
* **Non-numeric values** raise ``TypeError``.
* **Complex values** (non-real angles) raise ``ValueError``.
* **Ambiguous bindings** — the same logical parameter given twice, e.g.
  ``{theta: 1.0, "theta": 2.0}`` — raise ``ValueError``.

Execution
---------

Once bound, a circuit runs through the standard execution path
(``Circuit -> Backend.run(shots, seed) -> BackendResult``).  A circuit with
unbound parameters raises ``ValueError`` on execution — it is never
silently converted.

Binding can also be supplied at execution time via ``parameter_values=``,
which delegates to the same canonical ``bind_parameters`` logic:

.. code-block:: python

   from microquantum import StatevectorBackend

   result = StatevectorBackend().run(
       qc, shots=1024, seed=42, parameter_values={theta: 0.5, phi: 1.2}
   )

Repeated execution / sweeps
---------------------------

To sweep a circuit across values, bind and run per value:

.. code-block:: python

   for value in [0.0, 0.5, 1.0, 1.5]:
       bound = qc.bind_parameters({theta: value})
       result = StatevectorBackend().run(bound, shots=1024, seed=42)

or use the runtime helper, which returns one :class:`BackendResult` per
binding:

.. code-block:: python

   from microquantum import run_parameter_sweep

   results = run_parameter_sweep(qc, [0.0, 0.5, 1.0, 1.5], shots=1024, seed=42)

Serialization
-------------

Parameterized circuits round-trip through JSON: ``to_json`` records
parameterized rotation gates symbolically (including parameter
expressions) and ``from_json`` restores them, so a reloaded circuit can be
bound and executed exactly like the original:

.. code-block:: python

   restored = QuantumCircuit.from_json(qc.to_json())
   restored.is_parameterized        # True
   restored.bind_parameters({theta: 0.5, phi: 1.2}).run()

OpenQASM 2.0
------------

OpenQASM 2.0 has no symbolic parameters.  Exporting a *parameterized*
circuit with ``qc.qasm()`` raises ``ValueError`` rather than silently
dropping gates; bind the circuit first to export a concrete, numeric
program.

Composition
-----------

Concatenating circuits with ``+`` preserves parameters.  Because parameter
identity is name-based, a parameter with the same name on both sides is
the same logical parameter after concatenation.

Parameter-shift gradients
-------------------------

:func:`~microquantum.parameter_shift_gradient` differentiates the expectation
of an observable with respect to the circuit's parameters — the standard
entry point for variational algorithms:

.. code-block:: python

   from microquantum import Operator, parameter_shift_gradient

   grads = parameter_shift_gradient(qc, Operator.Z(), {theta: 0.5, phi: 1.2})

Sweeps & bindings in the runtime layer
--------------------------------------

* :class:`~microquantum.ParameterSweep` builds deterministic grids over
  parameters (explicit values, ``range``-style or ``linspace``-style) and
  expands them as an ordered Cartesian product — used heavily by the
  :doc:`/experiments/parameter-sweeps` layer.
* The :class:`~microquantum.ExecutionPlan` carries a ``parameter_bindings``
  map, so binding is part of the *plan*, not buried in the algorithm; the
  runtime binds a plan before dispatch (``plan.bound()``).

More examples
-------------

* ``examples/12_parameterized_circuit.py`` — a single symbolic ``ry`` gate,
  introspection and binding.
* ``examples/13_multiple_parameters.py`` — multiple parameters, expressions,
  partial binding and validation.
* ``examples/14_parameter_sweep.py`` — binding + execution across a sweep of
  values.

Run any of them directly:

.. code-block:: console

   python examples/12_parameterized_circuit.py