VQE
===

The Variational Quantum Eigensolver finds the ground-state energy of a
Hamiltonian by classically optimizing the parameters of a variational ansatz.

Usage
-----

.. code-block:: python

   from microquantum import EigenvalueProblem, Operator, Parameter, QuantumCircuit
   from microquantum.algorithms import VQE
   from microquantum.optimizers import COBYLA

   theta = Parameter("theta")
   ansatz = QuantumCircuit(1).ry(theta, 0)

   vqe = VQE(
       ansatz,
       Operator.Z(),
       COBYLA(max_iter=100),
   )
   problem = EigenvalueProblem(Operator.Z(), k=1, name="z")

   print(vqe.validate(problem))               # []
   result = vqe.solve(problem, initial_params={theta: 0.5})
   print(result.eigenvalue)                   # ~ -1.0
   print(result.optimal_params)               # {"theta": ~pi}

Constructor
-----------

``VQE(ansatz, hamiltonian, optimizer, *, runtime=None, shots=4096, seed=None)``

* ``ansatz`` — a (possibly parameterized) :class:`~microquantum.QuantumCircuit`.
* ``hamiltonian`` — the observable to minimize
  (:class:`~microquantum.Operator` or :class:`~microquantum.PauliSum`).
* ``optimizer`` — any :class:`~microquantum.Optimizer`
  (COBYLA, NelderMead, GradientDescent, Adam, BFGS, SPSA, QNSPSA, ...).
* ``runtime`` — optional :class:`~microquantum.ExecutionRuntime` to route
  every circuit evaluation through the pipeline.
* ``shots`` / ``seed`` — sampling configuration when run through a runtime.

Gradients
---------

Expectation gradients use the parameter-shift rule
(:func:`~microquantum.parameter_shift_gradient`) for rotation-parameterized
circuits, or operator-based gradients where the Hamiltonian structure allows;
gradient-free optimizers fall back to finite differences.