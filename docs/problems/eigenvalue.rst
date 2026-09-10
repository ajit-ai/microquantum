Eigenvalue
==========

An :class:`~microquantum.EigenvalueProblem` is a
:class:`~microquantum.HamiltonianProblem` that additionally requests the
lowest ``k`` eigenvalues.

Usage
-----

.. code-block:: python

   from microquantum import EigenvalueProblem, Operator

   problem = EigenvalueProblem(Operator.Z(), k=2, name="z-k2")
   print(problem.validate())              # []
   print(problem.hamiltonian)             # Operator.Z()

   data = problem.to_dict()
   print(data["type"])                    # "Eigenvalue"
   print(data["k"])                       # 2
   restored = EigenvalueProblem(
       hamiltonian=Operator.from_dict(data["hamiltonian"]),
       k=data["k"],
       name=data["name"],
   )

Solving with VQE
----------------

:doc:`/algorithms/vqe` is the canonical solver: give it the ansatz, the
Hamiltonian and a classical optimizer.

.. code-block:: python

   from microquantum import EigenvalueProblem, Operator, Parameter, QuantumCircuit
   from microquantum.algorithms import VQE
   from microquantum.optimizers import COBYLA

   theta = Parameter("theta")
   ansatz = QuantumCircuit(1).ry(theta, 0)

   vqe = VQE(ansatz, Operator.Z(), COBYLA(max_iter=100))
   result = vqe.solve(EigenvalueProblem(Operator.Z(), k=1), initial_params={theta: 0.5})
   print(result.eigenvalue)               # ~ -1.0

Alternative: phase estimation
-----------------------------

:doc:`/algorithms/phase-estimation` solves the eigenvalue problem when the
Hamiltonian is a **unitary** operator (its ``validate`` will reject
non-unitary Hamiltonians for that algorithm).