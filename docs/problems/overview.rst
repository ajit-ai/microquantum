Overview
========

A *problem* is a plain, JSON-safe description of a computational task — never
an execution and never an algorithm.  Problems feed :doc:`/concepts/algorithms`,
which translate them into circuits and run them.

The problem contract
--------------------

Every problem subclasses :class:`~microquantum.Problem` and provides:

* ``validate() -> list[str]`` — human-readable diagnostics (empty list =
  *valid*).  Never raises.
* ``to_dict()`` / ``from_dict()`` and ``to_json()`` / ``from_json()`` —
  JSON-safe serialization.
* ``type`` / ``name`` metadata.

The five built-ins
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Problem
     - Generalization
   * - :class:`~microquantum.SamplingProblem`
     - Obtain samples ``|bitstring> -> probability`` from a circuit's output
       distribution.  Payload: a circuit.
   * - :class:`~microquantum.OptimizationProblem`
     - Minimize a binary-objective function, in QUBO or spin-Ising view.
       Payload: QUBO/Ising coefficients.
   * - :class:`~microquantum.HamiltonianProblem`
     - The spectrum of a Hermitian operator.
       Payload: an ``Operator`` or ``PauliSum``.
   * - :class:`~microquantum.EigenvalueProblem`
     - Like HamiltonianProblem, but requests the lowest ``k`` eigenvalues.
   * - :class:`~microquantum.SearchProblem`
     - Find marked items in a ``2**num_qubits``-item database.
       Payload: a target list or predicate.

Design notes
------------

* **Positional-first payloads**: ``SamplingProblem(circuit)``,
  ``HamiltonianProblem(H)``, ``EigenvalueProblem(H, k=2)`` — the payload is
  the first argument.
* Hamiltonian-backed problems serialize through
  ``hamiltonian_to_dict`` (labeled Pauli terms with complex coefficients or
  an explicit matrix).
* Problems are the *input* contract of the SDK; results are never a problem.