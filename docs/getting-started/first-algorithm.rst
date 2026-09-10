First Algorithm
===============

Algorithms consume problems through a uniform lifecycle:
``validate(problem) -> [problems]`` then ``solve(problem, runtime=None)``.
Results are typed, JSON-safe containers.

VQE on a one-qubit Hamiltonian
------------------------------

.. code-block:: python

   from microquantum import EigenvalueProblem, Operator, Parameter, QuantumCircuit
   from microquantum.algorithms import VQE
   from microquantum.optimizers import GradientDescent

   theta = Parameter("theta")
   ansatz = QuantumCircuit(1).ry(theta, 0)      # |0> -> RY(theta)|0>

   vqe = VQE(
       ansatz,
       Operator.Z(),
       GradientDescent(learning_rate=0.3, max_iter=60, tol=1e-8),
   )
   problem = EigenvalueProblem(Operator.Z(), k=1, name="z")

   print(vqe.validate(problem))                 # []
   result = vqe.solve(problem, initial_params={theta: 0.5})
   print(f"ground energy: {result.eigenvalue:.4f}")   # ~ -1.0

Running through the execution runtime
-------------------------------------

Pass an :class:`~microquantum.ExecutionRuntime` to route every circuit
evaluation (plan -> backend -> job -> result) through the MQ-04 pipeline:

.. code-block:: python

   from microquantum import ExecutionRuntime

   vqe_runtime = VQE(
       ansatz,
       Operator.Z(),
       GradientDescent(learning_rate=0.3, max_iter=60, tol=1e-8),
       runtime=ExecutionRuntime(),
   )
   result = vqe_runtime.solve(problem, initial_params={theta: 0.5})
   print(result.eigenvalue)

Every algorithm documents the problem type it accepts, its parameters, its
backend/runtime requirements and its limitations — see the :doc:`/api/algorithms`.

Next: :doc:`first-experiment`.