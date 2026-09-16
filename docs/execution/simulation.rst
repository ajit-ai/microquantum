Simulation
==========

MicroQuantum ships four real local simulators plus a deterministic mock.
All of them implement one execution contract, so a circuit runs identically
across engines - only the memory scaling, the noise support and the feature
breadth differ.

Simulator overview
------------------

.. list-table::
   :widths: 18 22 26 18 16
   :header-rows: 1

   * - Backend
     - Engine
     - Representation
     - Noise
     - Sampling
   * - :class:`~microquantum.StatevectorBackend`
     - ``statevector``
     - ``2**n`` complex amplitudes
     - no
     - full-distribution ``numpy.default_rng`` draws
   * - :class:`~microquantum.DensityMatrixBackend`
     - ``density_matrix``
     - ``2**n x 2**n`` complex matrix
     - yes (:class:`~microquantum.NoiseModel`)
     - full-distribution ``numpy.default_rng`` draws
   * - :class:`~microquantum.MPSBackend`
     - ``mps``
     - ``n`` tensors, bond dimension ``chi``
     - no
     - sequential (peeling) Born-rule sampling
   * - :class:`~microquantum.TreeTensorNetworkBackend`
     - ``ttn``
     - tensor tree, leaves per qubit
     - no
     - capped at 18 qubits (dense reconstruction)
   * - :class:`~microquantum.MockBackend`
     - ``mock``
     - stub
     - no
     - stub counts

Each backend advertises its engine through ``capabilities.metadata`` and its
gate set through ``target``:

.. code-block:: python

   from microquantum import (
       DensityMatrixBackend, MPSBackend, StatevectorBackend,
       TreeTensorNetworkBackend,
   )
   from microquantum.backends.capabilities import EXECUTION_DENSITY_MATRIX

   for backend in (
       StatevectorBackend(), DensityMatrixBackend(),
       MPSBackend(), TreeTensorNetworkBackend(),
   ):
       caps = backend.capabilities
       print(backend.name, "->", caps.metadata["engine"])
       print("   target:", backend.target.name)
       print("   density-matrix execution:", caps.supports_execution(EXECUTION_DENSITY_MATRIX))

Execution contract
------------------

Every simulator backend implements ``run(circuit)`` / ``run_circuit(...)``
with the same shots rules:

* ``shots`` defaults to ``1024`` and must be a positive integer - ``0``,
  negative values and non-numeric values raise ``ValueError`` instead of
  being silently accepted.
* ``shots=None`` requests **deterministic execution**: no sampling happens,
  ``counts`` is ``{}``, ``samples`` is ``None`` and the exact final state
  (state vector for the statevector/MPS/TTN engines, plus the density matrix
  for the density-matrix engine) is returned.
* ``seed`` makes sampling reproducible: the same circuit, shots and seed
  reproduce the same counts.

.. code-block:: python

   from microquantum import QuantumCircuit, StatevectorBackend

   qc = QuantumCircuit(2).h(0).cnot(0, 1)

   exact = StatevectorBackend().run(qc, shots=None)
   print(exact.counts, exact.samples, exact.shots)  # {} None None
   print(abs(exact.statevector[0]) ** 2)            # 0.5

   seeded = StatevectorBackend().run(qc, shots=1000, seed=42)
   again = StatevectorBackend().run(qc, shots=1000, seed=42)
   print(seeded.counts == again.counts)             # True

Density-matrix simulator
------------------------

The density-matrix engine tracks the full ``2**n x 2**n`` matrix, so it can
represent noisy and mixed states.  For a pure circuit it is exactly
equivalent to the state vector, and it exposes off-diagonal (coherence)
elements:

.. code-block:: python

   import numpy as np
   from microquantum import DensityMatrixBackend, StatevectorBackend

   qc = QuantumCircuit(2).h(0).cnot(0, 1)

   sv = StatevectorBackend().run(qc, shots=None).statevector
   rho = DensityMatrixBackend().run(qc, shots=None).density_matrix
   print(np.max(np.abs(np.abs(sv) ** 2 - np.real(np.diag(rho)))))  # ~0
   print(abs(rho[0, 3]))                            # 0.5 (entanglement)

Noise
-----

The density-matrix backend is the sanctioned noise vehicle.  Build a
:class:`~microquantum.NoiseModel`, then pass it through
``DensityMatrixBackend.run_circuit(..., noise_model=...)``.  With
``shots=None`` the exact noisy probabilities are returned without sampling:

.. code-block:: python

   from microquantum import DensityMatrixBackend, NoiseModel

   x = [QuantumCircuit(1).x(0).gates[0][0].matrix]          # one X gate
   model = NoiseModel().depolarizing(0.6)
   result = DensityMatrixBackend().run_circuit(
       num_qubits=1, gates=[(x[0], [0])], noise_model=model, shots=None,
   )
   print(np.real(np.diag(result.density_matrix)))           # P(0) ~ 0.4

Unsupported options fail loudly: passing a non-``NoiseModel`` raises
``TypeError``, and combining a noise model with a backend in the
:class:`~microquantum.Executor` (which would silently ignore one of them)
raises ``ValueError``.

.. code-block:: python

   from microquantum import Executor

   try:
       Executor(noise_model="junk")
   except TypeError as exc:
       print("TypeError:", exc)

Tensor-network simulators
-------------------------

The MPS engine keeps memory polynomial in the qubit count.  A bond-dimension
cap trades a small, tracked fidelity loss for massive compression:

.. code-block:: python

   from microquantum import MPSBackend

   ladder = QuantumCircuit(20)
   ladder.h(0)
   for q in range(19):
       ladder.cx(q, q + 1)

   result = MPSBackend(max_bond_dim=1).run_circuit(
       num_qubits=20,
       gates=[(op.matrix, targets) for op, targets in ladder.gates],
       shots=None,
   )
   print(result.metadata["max_bond_dim"], result.metadata["truncation_error"])

The tree tensor network reconstructs a dense state vector for verification.
Because that reconstruction needs ``2**n`` memory, ``TreeTensorNetwork.sample``
is capped at 18 qubits and raises ``ValueError`` above it, pointing you to
``MatrixProductState.sample`` (sequential, memory-scalable sampling):

.. code-block:: python

   from microquantum import MatrixProductState, TreeTensorNetwork

   try:
       TreeTensorNetwork.from_zeros(19).sample(16)
   except ValueError as exc:
       print("ValueError:", exc)
   counts = MatrixProductState.from_zeros(19).sample(16, seed=1)
   print(sum(counts.values()))                         # 16

Memory boundaries
-----------------

Dense allocations are checked *before* memory is requested: the SDK refuses
a state vector larger than ``2**31`` bytes (``16 * 2**27``) and a density
matrix that would exceed the same budget, instead of letting the process run
out of memory.  ``StateVector(28)`` and ``DensityMatrix(14)`` therefore raise
``ValueError``.

.. code-block:: python

   from microquantum import StateVector

   try:
       StateVector(28)                      # ~4 GiB, over the 2 GiB budget
   except ValueError as exc:
       print("refused:", type(exc).__name__)

Cross-simulator consistency
---------------------------

All four simulators agree on the physics.  For a given seeded run the
state-vector and TTN engines share the same ``numpy.default_rng`` sampling
path (identical counts), while the MPS peeling sampler is statistically
consistent:

.. code-block:: python

   from microquantum import TreeTensorNetworkBackend

   bell = QuantumCircuit(2).h(0).cnot(0, 1)
   sv_counts = StatevectorBackend().run(bell, shots=2000, seed=1).counts
   tn_counts = TreeTensorNetworkBackend().run(bell).counts
   print(sv_counts)
   print("TTN without seed uses the default 1024 shots:",
         sum(tn_counts.values()))