Parameters
==========

Parameters make circuits tunable without rebuilding them.  A
:class:`~microquantum.Parameter` is a named symbolic placeholder; rotation
gates accept a parameter (or any :class:`~microquantum.ParameterExpression`
built with ``+``, ``-``, ``*``, ``/``, ``**``) in the angle slot.

Creating and using parameters
-----------------------------

.. code-block:: python

   from microquantum import Parameter, QuantumCircuit

   theta = Parameter("theta")
   phi = Parameter("phi")

   qc = QuantumCircuit(2)
   qc.ry(theta, 0)
   qc.rz(phi + 0.1, 1)

   print(qc.parameters())      # {theta, phi}
   print(qc.is_parameterized())# True

Binding values
--------------

A circuit whose parameters have all been resolved is "bound" and can be
executed:

.. code-block:: python

   bound = qc.bind_parameters({theta: 0.5, phi: 1.2})
   print(bound.is_parameterized())      # False
   print(bound.get_unitary())

Parameter-shift gradients
-------------------------

:func:`~microquantum.parameter_shift_gradient` differentiates the expectation
of an observable with respect to the circuit's parameters — the standard
entry point for variational algorithms:

.. code-block:: python

   from microquantum import Operator, parameter_shift_gradient

   grads = parameter_shift_gradient(qc, Operator.Z(), {theta: 0.5, phi: 1.2})

Sweeps & bindings
-----------------

* :class:`~microquantum.ParameterSweep` builds deterministic grids over
  parameters (explicit values, ``range``-style or ``linspace``-style) and
  expands them as an ordered Cartesian product — used heavily by the
  :doc:`/experiments/parameter-sweeps` layer.
* The :class:`~microquantum.ExecutionPlan` carries a ``parameter_bindings``
  map, so binding is part of the *plan*, not buried in the algorithm; the
  runtime binds a plan before dispatch (``plan.bound()``).

Parameter expressions
---------------------

Simple arithmetic on a :class:`~microquantum.Parameter` produces a
:class:`~microquantum.ParameterExpression` that keeps the math symbolic until
the parameters it depends on are bound.