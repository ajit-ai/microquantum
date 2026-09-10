Capabilities
============

:class:`~microquantum.BackendCapabilities` describes **what a backend can do**
without running anything.  A backend answers three questions before any work
is dispatched:

.. code-block:: text

              inspect                     validate                  execute
   Work -> ExecutionPlan ------> BackendCapabilities -------> Backend.execute(plan)
                     |                  (no work runs)             (binds + runs)
                     v
             BackendRegistry <------ providers discover backends

What is described
-----------------

* **Target class** — :class:`~microquantum.TargetClass` (CPU, GPU,
  SIMULATOR, QUANTUM_HARDWARE, REMOTE, CUSTOM).
* **Execution modes** — statevector, sampling, shots, expectation values,
  unitary, density matrix.
* **Circuit features** — parameterized circuits, measurement, mid-circuit
  measurement, reset, controlled operations, custom gates.
* **Hardware characteristics** — qubit capacity, connectivity, native gate
  set, precision.

Usage
-----

.. code-block:: python

   from microquantum import StatevectorBackend, TargetClass

   caps = StatevectorBackend().capabilities
   print(caps.target_class)          # TargetClass.SIMULATOR
   print("statevector" in caps.execution)     # True
   print("shots" in caps.execution)           # True
   print(caps.circuit_features)      # set of feature tokens
   print(caps.qubit_capacity)        # unbounded / advertised
   print(caps.to_dict())             # JSON-safe (no execution)

Contract integration
--------------------

* ``supports(plan) -> bool`` — quick feature/capability check.
* ``validate(plan) -> list[str]`` — compatibility diagnostics before a run.
* ``Backend.metadata()`` — JSON-safe backend description including its
  capabilities.
* ``merge`` — derive the capability *intersection* of two backends (used when
  composing simulators or checking a pipeline against a target).

Custom capability tokens
------------------------

Unknown capability tokens are stored as plain strings, so custom backends can
tag their own modes without forking the SDK.