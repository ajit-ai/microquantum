Runtime
=======

The execution runtime layers configuration, plans, strategies, diagnostics and
introspection on top of the backend contract.  Full API details live in the
generated reference; the links below jump straight to the relevant module.

* :class:`~microquantum.runtime.runtime.ExecutionRuntime` — coordinates
  preparation, compilation, submission and collection, enriching results with
  metadata and traces.
* :class:`~microquantum.runtime.config.RuntimeConfig` — frozen, JSON-safe
  defaults (backend, registry, default target, history cap, optimization
  level); passed as ``config=...`` or derived via
  :meth:`~microquantum.runtime.runtime.ExecutionRuntime.configure`.
* :class:`~microquantum.runtime.plan.ExecutionPlan` — declarative description
  of what to run (see :doc:`/execution/execution-plan`).
* :class:`~microquantum.runtime.strategy.ExecutionStrategy` —
  ``DIRECT`` / ``COMPILED`` / ``BATCH`` / ``PARAMETER_SWEEP`` / ``HYBRID``
  dispatch and pluggable handlers (``STRATEGY_HANDLERS``,
  ``register_custom_strategy``).
* :class:`~microquantum.runtime.trace.ExecutionTrace` — per-execution lifecycle
  record.
* Stage-tagged errors in :mod:`microquantum.runtime.errors` — all subclasses of
  :class:`~microquantum.runtime.errors.ExecutionError` (which subclasses
  ``ValueError``, so existing ``except ValueError`` code keeps working):
  :class:`~microquantum.runtime.errors.PlanningError`,
  :class:`~microquantum.runtime.errors.CompilationError`,
  :class:`~microquantum.runtime.errors.RuntimeDispatchError` and
  :class:`~microquantum.runtime.errors.BackendExecutionError`.
* Introspection — :func:`~microquantum.runtime.info.runtime_info` returns a
  :class:`~microquantum.runtime.info.RuntimeInfo` snapshot (version, Python /
  NumPy, strategies, live backend capability summaries, default backend).
* Module-level convenience entry points :func:`~microquantum.runtime.execute`,
  :func:`~microquantum.runtime.submit`, ``execute_batch``, ``submit_batch``,
  ``execute_records``, ``run_parameter_sweep``, ``run_hybrid``,
  ``run_experiment`` route through the shared ``default_runtime``.

See :doc:`/execution/runtime` for the narrative guide, and
:doc:`/execution/cli` for the ``microquantum`` developer command line.

Generated reference (by module)
-------------------------------

.. toctree::
   :hidden:

   /api/microquantum/runtime/runtime/index
   /api/microquantum/runtime/config/index
   /api/microquantum/runtime/plan/index
   /api/microquantum/runtime/strategy/index
   /api/microquantum/runtime/trace/index
   /api/microquantum/runtime/errors/index
   /api/microquantum/runtime/info/index