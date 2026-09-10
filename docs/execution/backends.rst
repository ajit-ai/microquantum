Backends
========

A :class:`~microquantum.Backend` is the execution contract.  Subclasses
implement ``run_circuit`` (raw gate matrices) or ``run`` (a bound
:class:`~microquantum.QuantumCircuit`); the base provides the plan-level
surface:

* ``capabilities`` — a :class:`~microquantum.BackendCapabilities` descriptor.
* ``validate(plan) -> list[str]`` — plan/backend compatibility diagnostics.
* ``supports(plan) -> bool`` — quick plan capability check.
* ``execute(plan) -> BackendResult`` — the canonical single-call entry point
  (validates, binds via ``plan.bound()``, runs).

Results
-------

:class:`~microquantum.BackendResult` carries the state vector / density
matrix, raw ``samples``, measurement ``counts`` / ``probabilities``, labeled
``expectations``, ``eigenvalues``, a JSON-safe ``native`` payload, and the
``shots`` / ``seed`` / ``target_name`` of the run.  Everything serializes via
``to_dict()`` / ``to_json()``.

Built-ins
---------

* :class:`~microquantum.StatevectorBackend` — exact state-vector simulation.
* :class:`~microquantum.DensityMatrixBackend` — density-matrix (mixed-state)
  simulation, including noise channels.
* :class:`~microquantum.MPSBackend` / :class:`~microquantum.TreeTensorNetworkBackend`
  — approximate tensor-network simulators.
* :class:`~microquantum.MockBackend` — deterministic stub for tests and
  pipelines.
* :class:`~microquantum.LocalSimulatorBackend` — the reference local
  simulator.

.. code-block:: python

   from microquantum import ExecutionPlan, QuantumCircuit, StatevectorBackend

   qc = QuantumCircuit(2).h(0).cx(0, 1)
   backend = StatevectorBackend()

   result = backend.run(qc, shots=1024, seed=1)
   print(result.state)
   print(result.counts)
   print(result.most_frequent())

   plan = ExecutionPlan.from_circuit(qc, backend=backend, shots=512)
   print(backend.validate(plan))     # []
   print(backend.supports(plan))     # True
   print(backend.execute(plan) == backend.run(qc, shots=512))  # same data path

Noise
-----

:class:`~microquantum.NoiseModel` and :class:`~microquantum.NoiseChannel`
attach to density-matrix / tensor-network simulators for noisy execution
studies.

Interoperability
----------------

`Executor` and continuous-focused backends stay NumPy-only; GPU acceleration is
opt-in at the array layer (:func:`~microquantum.set_array_backend`).  Vendor
execution happens through the provider boundary (:doc:`providers`).