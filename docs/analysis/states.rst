States
======

:class:`~microquantum.StateAnalysis` inspects statevectors and density
matrices in a backend-independent way.

Inputs
------

A :class:`~microquantum.BackendResult`, an :class:`~microquantum.ExecutionRecord`,
a :class:`~microquantum.StateVector` / `~microquantum.DensityMatrix`, a raw
NumPy array (1-D = statevector, 2-D = density matrix), or a serialized dict
carrying ``statevector`` / ``density_matrix``.

Statevectors
------------

.. code-block:: python

   from microquantum import QuantumCircuit, StateAnalysis

   qc = QuantumCircuit(2).h(0).cx(0, 1)
   state = qc.run()                      # StateVector

   analysis = StateAnalysis(state)
   print(analysis.kind)                  # "statevector"
   print(analysis.dim)                   # 4
   print(analysis.norm_squared())        # 1.0
   print(analysis.is_normalized())       # True
   print(analysis.probabilities())       # {'00': 0.5, '11': 0.5}
   print(analysis.most_probable_state()) # 0 or 3 (index)
   print(analysis.most_probable_bitstring())  # '00' or '11'

Density matrices
----------------

For a result carrying a density matrix (or a
:class:`~microquantum.DensityMatrix` directly):

.. code-block:: python

   dm_analysis = StateAnalysis(density_matrix_result)
   print(dm_analysis.kind)                # "density_matrix"
   print(dm_analysis.trace())             # 1.0
   print(dm_analysis.purity())            # 1.0 pure / <1 mixed
   print(dm_analysis.is_pure())           # True
   print(dm_analysis.diagonal_probabilities())

Observables
-----------

A *diagonal* observable may be given as a 1-D array of ``dim`` real numbers,
or a callable ``basis_index -> real``:

.. code-block:: python

   print(analysis.expectation([0.0, 1.0, 1.0, 0.0]))   # ~1.0 for '11'
   print(analysis.expectation(lambda i: i % 2))         # callable form

Notes
-----

* ``kind`` / ``dim`` report what was analysed.
* Everything persists via ``to_dict()`` / ``to_json()``.