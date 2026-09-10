Results
=======

An :class:`~microquantum.ExperimentResult` is the outcome of running an
:class:`~microquantum.Experiment`.  Its raw execution records are **preserved
verbatim** — aggregation and analysis never destroy them.

Structure
---------

* ``experiment`` / ``status`` — identity and lifecycle state.
* ``executions()`` — all :class:`~microquantum.ExecutionRecord` s in order.
* ``successes()`` / ``success_count`` / ``failure_count`` /
  ``all_successful`` — quick status summaries.
* ``to_dict()`` / ``to_json()`` / ``from_dict()`` — every record serialized
  (including each raw :class:`~microquantum.BackendResult`),
  fully restorable JSON-safe round trip.

Example
-------

.. code-block:: python

   result = exp.run(ExecutionRuntime(backend=MockBackend()))

   for record in result.executions():
       print(
           record.status, record.parameter_bindings,
           record.metadata.get("sweep_name", "-"),
       )

   consolidated = result.to_dict()
   restored = ExperimentResult.from_dict(consolidated)
   assert restored.failure_count == result.failure_count

Feeding the analysis layer
--------------------------

Records flow into analysis without re-running anything:

* :class:`~microquantum.SamplingAnalysis` — counts -> probabilities /
  entropy / marginals.
* :class:`~microquantum.ExpectationAnalysis` — per-label mean / variance /
  std / standard error.
* :class:`~microquantum.StateAnalysis` — statevector / density-matrix
  inspection.
* :class:`~microquantum.ResultAggregator` — group by bindings, backend or
  status (see :doc:`/analysis/aggregation`).

Design principle
----------------

``raw results -> records -> aggregation -> derived analysis`` — never
``raw results -> replace with summary``.  This keeps every experiment
re-analysable without re-execution.