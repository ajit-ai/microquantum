Execution Plan
==============

An :class:`~microquantum.ExecutionPlan` is the declarative, JSON-safe
description of **what** to run, **where** and **how**.  Plans are plain data
— they *never execute anything*.

Exactly one of these describes the work:

* ``circuit`` — a :class:`~microquantum.QuantumCircuit`,
* ``ir`` — an :class:`~microquantum.IRCircuit`,
* ``compiled`` — a :class:`~microquantum.CompilationResult` to reuse instead
  of recompiling.

Where & how:

* ``target`` — a :class:`~microquantum.Target` the work is compiled for.
* ``backend`` — a :class:`~microquantum.Backend`, or the *name* of a
  registered backend (resolved through the runtime's
  :class:`~microquantum.BackendRegistry`).
* ``shots`` — number of measurement shots (default 1024).
* ``parameter_bindings`` — mapping of parameter name (or
  :class:`~microquantum.Parameter`) to value, bound before execution.
* ``initial_state`` — optional starting :class:`~microquantum.StateVector`.
* ``seed`` — RNG seed for reproducible execution.
* ``optimization_level`` — compiler level (0-2).
* ``options`` / ``metadata`` — free-form runtime options and user metadata.

Building plans
--------------

.. code-block:: python

   from microquantum import ExecutionPlan, Parameter, QuantumCircuit, StatevectorBackend

   theta = Parameter("theta")
   qc = QuantumCircuit(1).ry(theta, 0)

   plan = ExecutionPlan.from_circuit(
       qc,
       name="ry-diag",
       backend=StatevectorBackend(),
       shots=4096,
       parameter_bindings={"theta": 0.5},
       seed=7,
       optimization_level=1,
   )
   print(plan.validate())            # []  (bound, valid)

   data = plan.to_dict()             # JSON-safe
   restored = ExecutionPlan(**data)

Binding
-------

The plan carries the bindings; the runtime binds before dispatch.  A bound
plan's work is concrete: ``plan.bound()`` returns the resolved circuit.