Quantum States
==============

MicroQuantum represents quantum states two ways: pure :class:`~microquantum.StateVector`
and mixed :class:`~microquantum.DensityMatrix`.

StateVector
-----------

A ``2**n`` complex amplitude vector, normalized to unit norm.  Big-endian
ordering means amplitude index ``1`` corresponds to the bitstring ``...001``.

.. code-block:: python

   import numpy as np
   from microquantum import StateVector

   sv = StateVector(2)                       # |00>
   sv.amplitudes[3] = 1.0 / 2**0.5           # |11> component
   sv = sv.normalize()                       # |psi> = (|00> + |11>)/sqrt(2)
   print(sv.dim())                           # 4
   print(sv.num_qubits)                      # 2
   print(sv.is_normalized)                   # True
   print(sv)                                 # (~0.707)|00> + (~0.707)|11>

Properties and operations
-------------------------

* ``amplitudes`` — the raw complex array (direct access for read/write).
* ``normalize()`` — renormalize in place (returns ``self`` for chaining).
* ``inner_product(other)`` / ``fidelity(other)`` — overlap and fidelity.
* ``copy()`` — deep copy.

Measurement distributions come from :func:`~microquantum.sample_state` and the
:doc:`/analysis/states` layer (see below).

DensityMatrix
-------------

Density matrices describe mixed states and underpin noisy simulation.

.. code-block:: python

   from microquantum import DensityMatrix

   dm = DensityMatrix.from_statevector(sv)
   print(dm.trace)          # 1.0 (property)
   print(dm.is_pure)        # True for |psi><psi|, False for mixed
   print(dm.matrix.shape)   # (4, 4)

``DensityMatrix`` also applies unitaries (``apply_unitary``), Kraus noise
(``apply_kraus``), forms partial traces, and computes observables
(``expectation``).

Analysis
--------

The :doc:`/analysis/states` page shows :class:`~microquantum.StateAnalysis`,
which inspects normalization, probabilities, the most probable basis state and
— for density matrices — trace and purity in a backend-independent way.  It
accepts raw :class:`~microquantum.StateVector` / :class:`~microquantum.DensityMatrix`
objects or backend results carrying them.

Tensor products
---------------

For composing systems the SDK provides :func:`~microquantum.tensor` (Kronecker
product) and :func:`~microquantum.expand_operator`; both operate on the
big-endian convention consistently.