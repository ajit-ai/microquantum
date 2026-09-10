Optimization
============

An :class:`~microquantum.OptimizationProblem` describes an objective over
binary variables, without encoding *how* it is optimized.  It provides two
equivalent, interchangeable views of the same problem:

* an objective function ``objective(bits) -> float`` over binary vectors,
* an Ising (spin) Hamiltonian ``cost_hamiltonian() -> PauliSum``.

Construction
------------

.. code-block:: python

   from microquantum import OptimizationProblem, PauliSum

   # From a spin-Ising Hamiltonian:
   problem = OptimizationProblem.from_ising(
       PauliSum.from_label("ZZ", 1.0), name="maxcut-edge"
   )
   print(problem.num_variables)          # 2
   print(problem.validate())             # []

Evaluating solutions
--------------------

.. code-block:: python

   import numpy as np

   x = np.array([1.0, 0.0])              # bitstring '10'
   print(problem.energy(x))              # objective at that bitstring
   spins = problem.encode_spins(x)       # s = 1 - 2*x  -> [-1, +1]
   print(problem.cost_hamiltonian())     # PauliSum spin view

   bits = problem.sample(seed=0)         # classical candidate (or sampler)
   print(bits.shape)

QUBO interchange
----------------

``OptimizationProblem`` keeps the objective and Ising views in sync:

* ``from_qubo(qubo, name=...)`` — build from a QUBO problem.
* ``to_qubo()`` — recover the binary ``QUBOProblem`` from the Ising view.
* ``energy(bits)`` — the QUBO energy when the problem came from a QUBO.

Size
----

``num_variables`` is the authoritative size (an ``OptimizationProblem`` must
not set ``num_qubits`` directly); ``qubits_needed`` reports the qubit count an
exact solver must allocate.