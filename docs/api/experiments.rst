Experiments
===========

The experiments layer records and replays structured quantum experiments.
Full API details live in the generated reference; the links below jump
straight to the relevant module.

* :class:`~microquantum.experiments.record.ExecutionRecord` — a single
  immutable experiment execution (circuit, backend, shots, results, fidelity).
* :class:`~microquantum.experiments.record.ExecutionFailure` — recorded
  per-batch failures.
* :class:`~microquantum.experiments.sweep.ParameterSweep` — multi-point
  parameter sweeps executed in batches.
* :class:`~microquantum.experiments.experiment.Experiment` — end-to-end
  experiment driver with reproducibility helpers.
* :class:`~microquantum.experiments.experiment.ExperimentResult` — summary and
  statistics of an executed experiment.

See :doc:`/experiments/experiments` for the narrative guide.

Generated reference (by module)
-------------------------------

.. toctree::
   :hidden:

   /api/microquantum/experiments/record/index
   /api/microquantum/experiments/sweep/index
   /api/microquantum/experiments/experiment/index