Measurement
===========

Measurement projects a quantum state onto the computational basis.  The SDK
offers several levels:

* :func:`~microquantum.sample_state` — sample all qubits of a
  :class:`~microquantum.StateVector` by Born's rule, returning a
  :class:`~microquantum.MeasurementResult` (counts, probabilities,
  ``most_frequent()``).
* :func:`~microquantum.measure_qubits` — measure a subset of qubits.
* :func:`~microquantum.measure_and_collapse` — measure and collapse the
  unmeasured register consistently.
* :func:`~microquantum.expectation_value` — compute ``<psi|O|psi>`` without
  sampling.
* :class:`~microquantum.DynamicCircuit` — mid-circuit measurement + reset +
  classical control (``measure``, ``measure_all``, ``reset``, ``c_if``).

Example: Bell-state measurement
-------------------------------

.. code-block:: python

   from microquantum import QuantumCircuit, StatevectorBackend

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   result = StatevectorBackend().run(qc, shots=1024, seed=1)
   print(result.counts)         # {'00': ~512, '11': ~512}
   print(result.most_frequent())

   state = qc.run()             # direct state-vector evolution
   mr = sample_state(state, shots=128, seed=2)
   print(mr.get_probabilities())  # {'00': 0.5, '11': 0.5}

Measurement results
-------------------

:class:`~microquantum.MeasurementResult` exposes ``counts``, ``shots``,
``qubits``, ``get_counts()``, ``get_probabilities()``, ``most_frequent()`` and
JSON serialization.  Backend results extend this with the full
:class:`~microquantum.BackendResult` payload (state, samples, expectations,
eigenvalues) — see :doc:`/execution/backends`.

The output distribution also feeds the analysis layer:
:class:`~microquantum.SamplingAnalysis` computes outcome probabilities,
Shannon entropy, marginals and observable mean/variance (see
:doc:`/analysis/sampling`).