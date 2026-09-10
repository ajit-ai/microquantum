First Measurement
=================

Measurement turns a quantum state into a classical sample.  In MicroQuantum
you either let a backend sample the output distribution, or you measure the
state directly.

Sampling with a backend
-----------------------

.. code-block:: python

   from microquantum import QuantumCircuit, StatevectorBackend

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   result = StatevectorBackend().run(qc, shots=1024, seed=1)
   print(result.counts)                    # {'00': ~512, '11': ~512}
   print(result.probabilities)             # {'00': 0.5, '11': 0.5}
   print(result.most_frequent())           # '00' or '11'

Direct measurement
------------------

.. code-block:: python

   state = qc.run()
   sample = sample_state(state, shots=8, seed=1)   # MeasurementResult
   print(sample.counts)                            # {'00': ~4, '11': ~4}
   print(sample.most_frequent())

   from microquantum import StateAnalysis
   analysis = StateAnalysis(state)
   print(analysis.probabilities())               # {'00': 0.5, '11': 0.5}
   print(analysis.most_probable_bitstring())     # '00' or '11'

The :class:`~microquantum.BackendResult` keeps the raw counts, probabilities
and (for statevector backends) the state.  See :doc:`/concepts/measurement`
for the full measurement model and :doc:`/analysis/sampling` for analysing
measurement distributions.

Partial measurement collapses only the measured qubits; see
:func:`~microquantum.measure_qubits` and
:func:`~microquantum.measure_and_collapse`.

Next: :doc:`first-problem`.