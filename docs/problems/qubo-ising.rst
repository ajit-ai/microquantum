QUBO & Ising
============

MicroQuantum bridges the two standard binary-optimization formulations:

* **QUBO** — ``minimize x^T Q x + c^T x`` over ``x in {0,1}^n``.
* **Ising / spin** — ``H = sum J_ij Z_i Z_j + sum h_i Z_i`` over spins
  ``s in {-1,+1}`` (``x = (1 - s)/2``).

Both are first-class in the SDK: the :class:`~microquantum.QUBOBuilder`
assembles QUBO matrices, :class:`~microquantum.QUBOProblem` stores them,
:class:`~microquantum.IsingConverter` maps between the two, and
:class:`~microquantum.OptimizationProblem` keeps both views in sync.

Building a QUBO
---------------

.. code-block:: python

   from microquantum import QUBOBuilder

   builder = QUBOBuilder(num_variables=4)
   builder.add_quadratic(0, 1, 2.0)          # x0*x1 interaction
   builder.add_linear(2, -1.5)               # x2 linear term
   builder.add_penalty_equality(0, 1, target=1)
   builder.add_penalty_one_hot([2, 3])
   builder.add_constant(0.5)

   qubo = builder.build("selection_example")
   print(qubo.num_variables)                 # 4
   print(qubo.energy([1, 0, 0, 1]))          # objective at '1001'
   print(qubo.to_dict())                     # JSON-safe

Builder helpers
---------------

* ``add_linear(i, coeff)`` / ``add_quadratic(i, j, coeff)`` / ``add_constant``.
* ``add_penalty_equality(i, j, target=1, penalty=10.0)`` —
  ``penalty*(x_i + x_j - target)^2``.
* ``add_penalty_inequality_le(indices, max_sum)`` — sum constraint with
  quadratic penalties.
* ``add_penalty_one_hot(indices)`` / ``add_penalty_at_most_one(indices)``.

QUBO <-> Ising
--------------

.. code-block:: python

   from microquantum import IsingConverter

   ising = IsingConverter.qubo_to_ising(qubo)     # PauliSum
   print(ising.num_terms)

   back = IsingConverter.ising_to_qubo(ising)     # exact energies preserved
   print(back.energy([1, 0, 0, 1]))

   report = IsingConverter.evaluate(back, [1, 0, 0, 1])
   print(report["energy"], report["num_ones"])

Into problems / algorithms
--------------------------

Wrap the result as an :class:`~microquantum.OptimizationProblem`
(``from_qubo`` / ``from_ising``) and hand it to a solver such as
:doc:`/algorithms/qaoa`.