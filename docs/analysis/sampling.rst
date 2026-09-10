Sampling
========

:class:`~microquantum.SamplingAnalysis` analyses measurement distributions.

Inputs
------

A :class:`~microquantum.BackendResult`, :class:`~microquantum.ExecutionRecord`,
:class:`~microquantum.ExperimentResult`, a ``{bitstring: count}`` mapping, or
a serialized ``to_dict()`` dict carrying ``counts``.

Usage
-----

.. code-block:: python

   from microquantum import QuantumCircuit, SamplingAnalysis, StatevectorBackend

   qc = QuantumCircuit(2).h(0).cx(0, 1)
   result = StatevectorBackend().run(qc, shots=4096, seed=1)

   analysis = SamplingAnalysis(result)
   print(analysis.counts)                    # raw counts
   print(analysis.total_shots())             # 4096
   print(analysis.unique_outcomes())         # ['00', '11']
   print(analysis.probabilities())           # {'00': ~0.5, '11': ~0.5}
   print(analysis.most_likely())             # '00' or '11'
   print(analysis.entropy())                 # ~1.0 bits

Numeric observable
------------------

A documented default maps bitstrings to unsigned-binary integers (MSB first);
a caller-supplied ``value_of`` overrides it:

.. code-block:: python

   analysis.mean()                    # ~1.5 with default mapping
   analysis.mean(value_of=str)        # caller-defined observable
   analysis.variance()                # population variance
   analysis.standard_deviation()
   print(analysis.to_dict())
   print(analysis.to_json())

Extremes
--------

``most_likely()`` / ``most_likely_probability()`` / ``least_likely()`` for the
distribution extremes.  An empty counts set raises ``ValueError`` for the
extremes and returns ``0.0`` entropy.