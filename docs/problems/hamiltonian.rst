Hamiltonian
===========

A :class:`~microquantum.HamiltonianProblem` asks for the spectrum of a
Hermitian operator.

Usage
-----

.. code-block:: python

   from microquantum import HamiltonianProblem, Operator

   problem = HamiltonianProblem(Operator.Z(), name="z-spectrum")
   print(problem.hamiltonian)             # Operator.Z()
   print(problem.validate())              # []

   data = problem.to_dict()               # labeled Pauli / matrix encoding
   print(data["type"])                    # "Hamiltonian"

   restored = HamiltonianProblem(
       hamiltonian=Operator.from_dict(data["hamiltonian"]),
       name=data["name"],
   )

Hamiltonians
------------

The Hamiltonian may be:

* a full :class:`~microquantum.Operator` matrix,
* a :class:`~microquantum.PauliSum` (efficient, matrix-free), or
* built from chemistry tooling — :class:`~microquantum.MolecularHamiltonian`,
  :class:`~microquantum.H2Hamiltonian`, :class:`~microquantum.LiHHamiltonian`.

Serialization
-------------

Hamiltonian-backed problems serialize through ``hamiltonian_to_dict`` into
labeled Pauli terms with complex coefficients (or an explicit matrix), so a
``HamiltonianProblem`` stays storable and shareable.

Solving
-------

:class:`~microquantum.EigenvalueProblem` (the subclass that also asks for the
lowest ``k`` eigenvalues) is the form consumed by :doc:`/algorithms/vqe`.