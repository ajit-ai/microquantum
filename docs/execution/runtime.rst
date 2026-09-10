Runtime
=======

:class:`~microquantum.ExecutionRuntime` is the coordinator of the canonical
pipeline:

.. code-block:: text

   Program -> ExecutionPlan -> Target -> Backend -> Job -> Execution -> Result

It prepares a plan, optionally compiles it against a target (reusing the
:class:`~microquantum.Compiler`), submits it to the selected backend and
collects the completed job into an enriched :class:`~microquantum.BackendResult`.

One-shot execution
------------------

.. code-block:: python

   from microquantum import ExecutionRuntime, StatevectorBackend

   runtime = ExecutionRuntime(backend=StatevectorBackend())
   result = runtime.execute(qc, shots=1024, seed=1)
   print(result.counts)

   plan = ExecutionPlan.from_circuit(qc, backend="statevector", shots=512)
   result = runtime.execute(plan)

Runtime internals
-----------------

* ``prepare(plan)`` — validate; ``compile(work, optimization_level=...)`` —
  optional compilation; ``submit(work, backend=...)`` -> :class:`~microquantum.Job`;
  ``execute(work)`` — the one-shot entry point (prepare -> dispatch -> collect).
* :class:`~microquantum.ExecutionStrategy` decides ``DIRECT`` vs ``COMPILED``
  dispatch (pluggable handlers; ``register_custom_strategy``);
  :class:`~microquantum.ExecutionTrace` records each execution's
  ``prepared -> bound/compiled -> validated -> submitted -> completed/failed``
  lifecycle with timing.
* Backend resolution: ``plan.backend`` (instance or name) > explicit default >
  registry default > lazily-created ``statevector`` simulator.

Orchestration helpers
---------------------

* ``execute_batch(works)`` / ``submit_batch(works)`` — many plans together;
  with ``raise_on_error=False`` a failing item is reported in place.
* ``execute_records(works)`` — one :class:`~microquantum.ExecutionRecord` per
  input, in order, never dropping failures (see
  :doc:`/experiments/execution-records`).
* ``run_parameter_sweep(circuit, bindings-or-floats)`` — one circuit over
  many bindings.
* ``run_experiment(experiment)`` / ``run_hybrid(build, update)`` — the
  higher-level orchestration (:doc:`/experiments/experiments`).
* Module-level :func:`~microquantum.execute`, :func:`~microquantum.submit`,
  :func:`~microquantum.execute_batch`, :func:`~microquantum.submit_batch`,
  :func:`~microquantum.execute_records`, :func:`~microquantum.run_hybrid` wrap
  a shared ``default_runtime``.

Design note
-----------

The runtime is **not** a backend: it never re-implements simulation or
provider logic, and any user-provided :class:`~microquantum.Backend` can be
dropped in through a plan's ``backend`` field.