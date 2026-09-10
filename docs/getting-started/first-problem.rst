First Problem
=============

A **problem** is a plain, JSON-safe description of a computational task.  It
never executes anything — an algorithm consumes it later.

The five built-in problem types in
:class:`~microquantum.Problem` family:

* :class:`~microquantum.SamplingProblem` — sample ``|bitstring> -> probability``
  from a circuit's output distribution.
* :class:`~microquantum.OptimizationProblem` — minimize a binary-objective
  function (created via :meth:`OptimizationProblem.from_qubo
  <microquantum.OptimizationProblem.from_qubo>` or
  :meth:`OptimizationProblem.from_ising`).
* :class:`~microquantum.HamiltonianProblem` — the spectrum of a Hermitian
  operator.
* :class:`~microquantum.EigenvalueProblem` — the lowest ``k`` eigenvalues.
* :class:`~microquantum.SearchProblem` — find marked items in a ``2**n``
  database.

Optimization problem from a QUBO
--------------------------------

.. code-block:: python

   from microquantum import OptimizationProblem
   from microquantum.optimization.qubo import QUBOBuilder

   builder = QUBOBuilder(num_variables=2)
   builder.add_linear(0, -1.0)
   builder.add_linear(1, -1.0)
   builder.add_quadratic(0, 1, 2.0)        # cut-like objective
   qubo = builder.build("simple")

   problem = OptimizationProblem.from_qubo(qubo, name="simple-min")
   print(problem.num_variables)
   print(problem.energy([0, 1]))           # objective at bitstring '01'
   print(problem.cost_hamiltonian())       # spin-Ising Pauli view

Eigenvalue problem
------------------

.. code-block:: python

   from microquantum import EigenvalueProblem, Operator

   problem = EigenvalueProblem(Operator.Z(), k=2, name="z-spectrum")
   print(problem.validate())               # []  (valid)

Validation and serialization
----------------------------

Every problem validates (returning a list of problems, empty = valid) and
serializes JSON-safely:

.. code-block:: python

   data = problem.to_dict()
   print(data["type"])                     # "eigenvalue"
   restored = problem.__class__.from_dict(data)

Next: :doc:`first-algorithm`.