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

   from microquantum import (
       ExecutionPlan,
       ExecutionRuntime,
       QuantumCircuit,
       StatevectorBackend,
   )

   qc = QuantumCircuit(2).h(0).cx(0, 1)
   runtime = ExecutionRuntime(backend=StatevectorBackend())
   result = runtime.execute(qc, shots=1024, seed=1)
   print(result.counts)

   plan = ExecutionPlan.from_circuit(qc, backend="local_simulator", shots=512, seed=2)
   result = runtime.execute(plan)

Configuration
-------------

:class:`~microquantum.runtime.config.RuntimeConfig` holds the runtime's
defaults as one frozen, JSON-safe value: the fallback backend (an instance or
a registered name), the :class:`~microquantum.backends.registry.BackendRegistry`
used to resolve names, the default compile target, the history cap and the
default compiler optimization level.  Only options the runtime actually
honours exist — there is no configuration surface for behaviour the runtime
cannot support.

.. code-block:: python

   from microquantum import ExecutionRuntime, RuntimeConfig

   rt = ExecutionRuntime(
       config=RuntimeConfig(
           backend="local_simulator",
           history_size=50,
           default_optimization_level=1,
       )
   )
   reconfigured = rt.configure(history_size=200)   # new runtime, original untouched
   assert rt.config.history_size == 50

Legacy keyword arguments (``backend=``, ``registry=``, ``history_size=``,
``default_optimization_level=``, ``default_target=``) remain supported and
override the config.  A runtime built from a bare circuit applies
``default_optimization_level`` when the plan carries none.

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

Errors
------

Failures are tagged with the pipeline stage they occur in.  Every stage error
subclasses :class:`~microquantum.runtime.errors.ExecutionError`, which itself
subclasses ``ValueError`` — so existing ``except ValueError`` code keeps working
while callers that care can catch precisely:

* :class:`~microquantum.runtime.errors.PlanningError` — invalid plans, unbound
  or unknown parameter bindings, dynamic circuits that were not converted.
* :class:`~microquantum.runtime.errors.CompilationError` — work incompatible
  with an explicit target, or a compiled plan found incompatible at dispatch.
* :class:`~microquantum.runtime.errors.RuntimeDispatchError` — the plan cannot
  run on the selected backend (capability or ``backend.validate(...)``
  problems, no matching strategy handler).
* :class:`~microquantum.runtime.errors.BackendExecutionError` — the submitted
  job failed, was cancelled or produced no result.

``TypeError`` plan-shape errors (e.g. ``prepare(object())``) are intentionally
not wrapped, so misuse of the API stays a distinct signal.  Wide ``except
ValueError`` from earlier versions continues to catch every stage error.

Introspection
-------------

:func:`~microquantum.runtime.info.runtime_info` returns a
:class:`~microquantum.runtime.info.RuntimeInfo` snapshot describing the runtime
as it is — nothing is guessed or hand-maintained:

.. code-block:: python

   from microquantum import runtime_info

   info = runtime_info(runtime)
   print(info.runtime, info.version)      # e.g. 'ExecutionRuntime' '0.4.1'
   print(info.strategies)                 # sorted ExecutionStrategy values
   print(info.default_backend)            # backend used when a plan names none
   print(info.backends)                   # live capability summaries per backend

Backend summaries are derived from each backend's
:class:`~microquantum.backends.capabilities.BackendCapabilities` in the live
registry, so they can never go stale; ``info.to_dict()`` is JSON-safe.

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

Command line
------------

The ``microquantum`` console script (``python -m microquantum``) exposes
``info``, ``backends`` and ``run`` from the shell — see
:doc:`/execution/cli`.

Design note
-----------

The runtime is **not** a backend: it never re-implements simulation or
provider logic, and any user-provided :class:`~microquantum.Backend` can be
dropped in through a plan's ``backend`` field.