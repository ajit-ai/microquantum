Problems
========

Problem formulations that map onto quantum hardware.  Full API details live in
the generated reference; the links below jump straight to the relevant module.

* :class:`~microquantum.problems.base.SamplingProblem` — sampling problems
  solvable via measurement statistics.
* :class:`~microquantum.problems.optimization.CombinatorialOptimizationProblem` —
  QUBO/Ising-style optimization problems.
* :class:`~microquantum.problems.eigenvalue.HamiltonianProblem` — Pauli
  Hamiltonians for ground-state tasks.
* :class:`~microquantum.problems.eigenvalue.EigenvalueProblem` — general
  eigenvalue problems (tensor-product Hamiltonians, matrices, …).
* :class:`~microquantum.problems.search.SearchProblem` — search problems
  modeled for amplitude amplification.

They pair with the narrative guides in :doc:`/problems/overview` and the
algorithms in :doc:`/algorithms/overview`.

Generated reference (by module)
-------------------------------

.. toctree::
   :hidden:

   /api/microquantum/problems/base/index
   /api/microquantum/problems/optimization/index
   /api/microquantum/problems/eigenvalue/index
   /api/microquantum/problems/search/index