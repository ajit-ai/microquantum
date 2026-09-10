QAOA
====

The Quantum Approximate Optimization Algorithm solves combinatorial
optimization problems encoded as QUBO/Ising objectives.

Usage
-----

.. code-block:: python

   from microquantum import OptimizationProblem, Parameter, QuantumCircuit
   from microquantum.algorithms import QAOA
   from microquantum.optimization import QUBOBuilder
   from microquantum.optimizers import COBYLA

   builder = QUBOBuilder(2)
   builder.add_linear(0, -1.0)
   builder.add_quadratic(0, 1, 2.0)          # unconstrained target
   problem = OptimizationProblem.from_qubo(builder.build("cut"), name="cut")

   qaoa = QAOA(
       p=1,
       optimizer=COBYLA(max_iter=200),
   )
   # or: qaoa = QAOA.from_problem(problem, p=1, optimizer=COBYLA(max_iter=200))

   print(qaoa.validate(problem))             # []
   result = qaoa.solve(problem, seed=0)
   print(result.optimal_bitstring, result.optimal_energy)
   print(result.statevector)                 # QAOA output state

How it works
------------

QAOA builds a ``p``-layer ansatz:

* **Cost layer** — phase-separating rotations from the problem's Ising cost
  Hamiltonian (the ``ZZ`` terms become CNOT parity chains).
* **Mixer layer** — standard ``RX`` mixer over all qubits.

The classical optimizer tunes the layer angles ``gamma``/``beta`` to maximize
the objective expectation; the final state is sampled to read the solution.

Notes
-----

* The problem class is :class:`~microquantum.OptimizationProblem`; sole-size
  beyond a few hundred qubits is limited by the state-vector simulator just
  like every other algorithm.
* Pass an :class:`~microquantum.ExecutionRuntime` to run the circuit
  evaluations through the MQ-04 pipeline.