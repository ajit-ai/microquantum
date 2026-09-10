Core Module
===========

The core engine: everything you need to build and inspect circuits, operators
and states.  Full API details live in the generated reference; the links below
jump straight to the relevant module.

Key entry points
----------------

* :class:`~microquantum.core.circuit.QuantumCircuit` — build circuits with
  gates, parameters, measurement and more.
* :func:`~microquantum.core.circuit.QuantumCircuit.run` — execute locally.
* :class:`~microquantum.core.state.StateVector` — complex amplitude vector.
* :class:`~microquantum.core.density_matrix.DensityMatrix` — mixed states.
* :class:`~microquantum.core.operators.Operator` — unitary / operator algebra.
* :class:`~microquantum.core.parameter.Parameter` — symbolic parameters.
* :func:`~microquantum.core.measurement.sample_state` — sample a state.
* :mod:`microquantum.core.pauli` — Pauli strings and sums.
* :mod:`microquantum.core.transpiler` — circuit transformations.
* :mod:`microquantum.core.qasm` — OpenQASM import / export.

Generated reference (by module)
-------------------------------

.. toctree::
   :hidden:

   /api/microquantum/core/engine/index
   /api/microquantum/core/circuit/index
   /api/microquantum/core/operators/index
   /api/microquantum/core/pauli/index
   /api/microquantum/core/parameter/index
   /api/microquantum/core/state/index
   /api/microquantum/core/density_matrix/index
   /api/microquantum/core/measurement/index
   /api/microquantum/core/gradient/index
   /api/microquantum/core/registers/index
   /api/microquantum/core/coupling/index
   /api/microquantum/core/device/index
   /api/microquantum/core/resources/index
   /api/microquantum/core/transpiler/index
   /api/microquantum/core/optimization/index
   /api/microquantum/core/dynamic/index
   /api/microquantum/core/qasm/index
   /api/microquantum/core/serialization/index
   /api/microquantum/core/tensor/index
   /api/microquantum/core/visualization/index