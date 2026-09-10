microquantum Documentation
==========================

A lightweight Python quantum computing SDK with domain-specific adapters for
aerospace, finance, biotech, chemistry, materials science, and optimization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   api/core
   api/benchmarks
   examples/index
   tutorials/index

Overview
--------

microquantum provides:

* **Core quantum operations**: State vectors, operators, gates, circuits
* **Measurement**: Sampling, expectation values, partial measurement
* **Noise modeling**: Depolarizing, amplitude damping, phase damping channels
* **Algorithms**: VQE, QAOA, Grover, Shor, Bernstein-Vazirani, Deutsch-Jozsa
* **Compilation**: Transpiler with routing, noise-aware placement, gate decomposition
* **Circuit optimization**: Gate fusion, identity removal, transpilation
* **Benchmarks**: Quantum volume, randomized benchmarking, XEB, CLOPS, GST

Getting Started
---------------

.. code-block:: bash

   pip install microquantum

.. code-block:: python

   from microquantum import QuantumCircuit

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   state = qc.run()
   print(state)

Result Contract
---------------

The SDK ships a standardized decision-result schema
(:mod:`microquantum.analytics.result`) that proprietary vertical solvers
published separately consume.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
