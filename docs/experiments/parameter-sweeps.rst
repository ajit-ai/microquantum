Parameter Sweeps
================

A :class:`~microquantum.ParameterSweep` builds a deterministic grid of
parameter combinations and expands it as an ordered Cartesian product.

Definitions
-----------

Each parameter maps to one of:

* a list/tuple of explicit numeric values — ``[0.0, 0.5, 1.0]``;
* ``{"values": [...]}`` — explicit values;
* ``{"range": (start, stop, step)}`` — NumPy ``arange``-style;
* ``{"start": .., "stop": .., "num_points": n}`` — ``linspace``;
* ``{"start": .., "stop": .., "step": ..}`` — ``arange``-style.

Usage
-----

.. code-block:: python

   from microquantum import ParameterSweep

   sweep = ParameterSweep({
       "theta": [0.0, 0.5, 1.0],              # explicit values
       "phi": {"range": (0.0, 1.0, 0.5)},     # arange-style (0.0, 0.5)
   })

   print(sweep.parameters)            # ("theta", "phi")
   print(sweep.values)                # {"theta": [...], "phi": [0.0, 0.5]}
   print(sweep.num_combinations)      # 3 * 2 == 6
   print(len(sweep))                  # 6
   print(sweep.combinations())        # 6 ordered dicts

   # iterable too:
   for combo in sweep:
       print(combo)

Verify against a base work item
-------------------------------

``verify``/``validate`` tie a sweep to the base work's parameters before any
execution:

.. code-block:: python

   from microquantum import Parameter, QuantumCircuit

   theta = Parameter("theta")
   qc = QuantumCircuit(1).ry(theta, 0)

   available = {p.name for p in qc.parameters}   # {"theta"}
   print(sweep.validate(available))    # [] if compatible
   sweep.verify(available)             # raises ValueError on mismatch

Use in experiments
------------------

Add a sweep to an :class:`~microquantum.Experiment` with a base plan/circuit;
each combination becomes one bound execution (see
:doc:`/experiments/experiments`).  Sweeps are JSON-safe
(``to_dict()`` / ``to_json()``).