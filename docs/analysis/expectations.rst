Expectations
============

:class:`~microquantum.ExpectationAnalysis` consumes the existing
``BackendResult.expectations`` ``{label: value}`` contract — the labeled
observable values a backend associates with a run.

Inputs
------

A :class:`~microquantum.BackendResult`, an :class:`~microquantum.ExecutionRecord`,
an :class:`~microquantum.ExperimentResult`, a ``{label: float}`` mapping, or a
serialized form carrying ``expectations``.

Usage
-----

.. code-block:: python

   from microquantum import ExpectationAnalysis, MockBackend, Parameter, QuantumCircuit

   theta = Parameter("theta")
   ansatz = QuantumCircuit(1).ry(theta, 0)

   # backend that reports labeled expectations (e.g. via Z-measurement)
   backend = MockBackend()
   r1 = backend.run(ansatz.bind_parameters({theta: 0.0}), shots=1024, seed=0)
   r2 = backend.run(ansatz.bind_parameters({theta: 0.5}), shots=1024, seed=0)

   analysis = ExpectationAnalysis([r1, r2])
   print(analysis.keys)                 # ('Z',) — the labels present
   print(analysis.result_count)         # 2
   print(analysis.mean("Z"))            # mean over the two executions
   print(analysis.variance("Z"))
   print(analysis.standard_deviation("Z"))
   print(analysis.standard_error("Z"))
   print(analysis.parameter_points("theta", "Z"))      # ordered points
   print(analysis.parameter_to_expectation("theta", "Z"))
   print(analysis.to_dict())

Notes
-----

* The analysis is *consensus-based*: it reads the labels backends already
  publish, so no new result format is introduced.
* Per-label mean/variance/std/standard-error follow the
  :mod:`microquantum.analysis.statistics` conventions (population variance by
  default, ``ddof`` adjustable).