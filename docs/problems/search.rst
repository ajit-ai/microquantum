Search
======

A :class:`~microquantum.SearchProblem` describes an unstructured database
search: find one (or more) of the ``2**num_qubits`` computational basis
states that satisfy a predicate.

Providing the marked set
------------------------

Exactly one of three providers should be given (at most one):

* ``target`` — an integer index, or list of integer indices, of the sought
  state(s);
* ``oracle`` — a callable ``oracle(num_qubits) -> QuantumCircuit`` that marks
  the solution subspace;
* ``predicate`` — a callable ``predicate(index) -> bool`` used to classically
  expand the marked set.

Usage
-----

.. code-block:: python

   from microquantum import SearchProblem

   problem = SearchProblem(3, target=[1, 5], name="find-1-and-5")
   assert SearchProblem(3, target=1).validate() == []      # target: int
   assert SearchProblem(3, target=[1, 5]).validate() == [] # target: list
   assert (
       SearchProblem(3, predicate=lambda i: i % 2 == 1).validate() == []
   )

   print(problem.target_indices())        # [1, 5]
   print(problem.is_marked("001"))        # True
   print(problem.is_marked("010"))        # False
   print(problem.num_solutions())         # 2

   data = problem.to_dict()               # JSON-safe
   print(data["type"])                    # "Search"
   print(data["target"])                  # [1, 5]

Solving with Grover
-------------------

:doc:`/algorithms/grover` amplifies the marked states: the iterations are
derived from ``num_solutions()`` (or ``num_targets`` when only ``num_qubits``
is known).