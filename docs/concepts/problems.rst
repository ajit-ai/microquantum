Problems
========

A **problem** is a plain, JSON-safe description of a computational task.  It
isn't executed and it isn't an algorithm — an algorithm consumes a problem
later.  All problems share one contract: ``validate() -> list[str]`` (empty =
valid) and ``to_dict()`` / ``to_json()`` serialization.

The five built-in problem types:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Problem
     - Purpose
   * - :class:`~microquantum.SamplingProblem`
     - Sample ``|bitstring> -> probability`` from a circuit's output
       distribution.
   * - :class:`~microquantum.OptimizationProblem`
     - Minimize a binary-objective function (QUBO or spin-Ising view).
   * - :class:`~microquantum.HamiltonianProblem`
     - The spectrum of a Hermitian operator.
   * - :class:`~microquantum.EigenvalueProblem`
     - The lowest ``k`` eigenvalues (subclass of HamiltonianProblem).
   * - :class:`~microquantum.SearchProblem`
     - Find marked items in a ``2**n``-item database.

Creating problems
-----------------

.. code-block:: python

   from microquantum import (
       EigenvalueProblem,
       Operator,
       OptimizationProblem,
       PauliSum,
       SamplingProblem,
       SearchProblem,
   )
   from microquantum.optimization import QUBOBuilder

   sampling = SamplingProblem(QuantumCircuit(2), name="bell")
   spectrum = HamiltonianProblem(Operator.Z(), name="z")
   lowest   = EigenvalueProblem(Operator.Z(), k=2, name="z-k2")

   builder = QUBOBuilder(2)
   builder.add_linear(0, -1.0)
   builder.add_quadratic(0, 1, 2.0)
   opt  = OptimizationProblem.from_qubo(builder.build("cut"), name="maxcut")
   spin = OptimizationProblem.from_ising(PauliSum.from_label("ZZ", 1.0))

   search = SearchProblem(2, targets=[1, 2], name="search")

Validation and serialization
----------------------------

.. code-block:: python

   problems = opt.validate()      # [] when valid
   data = search.to_dict()        # JSON-safe
   restored = SearchProblem.from_dict(data)

Positional-first
----------------

Payload comes first, mirroring the primary use: ``HamiltonianProblem(H)``,
``EigenvalueProblem(H, k=2)`` and ``SamplingProblem(circuit)``.