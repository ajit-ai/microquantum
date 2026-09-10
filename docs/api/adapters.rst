Adapted Problem Encodings
========================

Fixed-format problem encodings that adapt linear-programming style problem
definitions onto quantum solvers.  Full API details live in the generated
reference; the links below jump straight to the relevant module.

* :class:`~microquantum.adapters.base.QuantumProblem` — a problem in
  matrix + bound form, ready for quantum solver mapping.
* :class:`~microquantum.adapters.base.CombinatorialProblem` — combinatorial
  problems defined over binary decision variables.
* :mod:`microquantum.adapters.caching` — memoized evaluation helpers for
  adapted problems.

See :doc:`/execution/providers` for how these encodings plug into provider
backends.

Generated reference (by module)
-------------------------------

.. toctree::
   :hidden:

   /api/microquantum/adapters/base/index
   /api/microquantum/adapters/caching/index