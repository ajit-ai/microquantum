Grover
======

Grover's algorithm searches a ``2**n``-item database for marked items with
quadratic speed-up over classical linear search.

Usage
-----

.. code-block:: python

   from microquantum import SearchProblem
   from microquantum.algorithms import GroverSearch

   problem = SearchProblem(3, target=[1, 5], name="find-1-and-5")
   print(problem.validate())               # []

   grover = GroverSearch()
   # or: grover = GroverSearch.from_problem(problem)
   result = grover.solve(problem, seed=0)

   print(result.found_items)            # [1, 5] (indices of marked items)
   print(result.num_queries)            # ~O(sqrt(2**n / m)) oracle calls
   print(result.success)                # True/False

Problem types
-------------

:class:`~microquantum.SearchProblem` is constructed from an explicit
``targets`` list or, in the classic form, from the problem's decoder — the
oracle marks a subset of the computational-basis states.

Notes
-----

* The number of iterations is derived from the number of marked items;
  ``@staticmethod``-style ``from_problem`` sets it automatically.
* An optional user-supplied oracle callable is supported (see the
  :class:`~microquantum.algorithms.grover.GroverSearch` API reference).

Result
------

:class:`~microquantum.GroverResult` carries ``found_items``, the sampled
bitstring counts, ``num_queries`` and ``success`` flags — all JSON-safe.