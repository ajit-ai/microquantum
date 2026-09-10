Overview
========

Algorithms translate a :doc:`/concepts/problems` into circuits, run those
circuits (through the state-vector engine or an
:class:`~microquantum.ExecutionRuntime`), and return typed, serializable
results.

Uniform lifecycle
-----------------

Every built-in algorithm follows the base contract from
:class:`~microquantum.Algorithm`:

.. code-block:: python

   algorithm = SomeAlgorithm(...)
   diagnostics = algorithm.validate(problem)   # list[str], [] when valid
   result = algorithm.solve(problem, runtime=None)

``solve`` accepts an optional :class:`~microquantum.ExecutionRuntime` — when
one is provided every circuit evaluation (prepare -> compile -> submit ->
collect) is routed through the MQ-04 pipeline; otherwise the internal
state-vector engine is used directly.

Result types
------------

Results are ``*Result`` dataclasses subclassing
:class:`~microquantum.AlgorithmResult` (or its companion typed containers).
Each carries the algorithm name, the solved problem, the outcome, optimizer
info and free-form ``config`` / ``execution_metadata`` / ``native`` payloads.
All serialize through ``to_dict()`` / ``to_json()``.

Algorithm gallery
-----------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Algorithm
     - Solves / returns
   * - :class:`~microquantum.VQE`
     - ``EigenvalueProblem`` -> lowest eigenvalue (VQEResult).
   * - :class:`~microquantum.QAOA`
     - ``OptimizationProblem`` -> QAOA states & energies.
   * - :class:`~microquantum.GroverSearch`
     - ``SearchProblem`` -> found/missing marked items (GroverResult).
   * - :class:`~microquantum.PhaseEstimation`
     - unitary ``Operator`` eigenphase (PhaseEstimationResult).
   * - :class:`~microquantum.QFT` / ``qft_circuit``
     - quantum Fourier transform circuit/result.
   * - :class:`~microquantum.HamiltonianSimulation`
     - Trotter / qDRIFT / 4th-order evolution of a Hamiltonian.
   * - :class:`~microquantum.HHL`, :class:`~microquantum.VQD`,
       :class:`~microquantum.AdaptVQE`
     - linear systems, excited states, adaptive VQE.
   * - :class:`~microquantum.ShorsAlgorithm`, :class:`~microquantum.BernsteinVazirani`,
       :class:`~microquantum.DeutschJozsa`
     - the classic oracle algorithms (ShorResult, BVResult, DJResult).
   * - :class:`~microquantum.DiscreteQuantumWalk`,
       :class:`~microquantum.ContinuousQuantumWalk`
     - quantum walk simulation (QuantumWalkResult).

Each class also exposes ``from_problem(problem, ...)`` to configure the
algorithm directly from a problem instance.

Classic-convention algorithms
-----------------------------

A few older algorithms (``AdaptVQE``, ``VQD``, oracles) still expose the
classic ``compute_minimum_eigenvalue()`` / ``run()`` entry points, but they
return the same serializable result shapes.