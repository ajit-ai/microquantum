Examples
========

The ``examples/`` directory ships with runnable demo scripts across ten
categories.  This page embeds the canonical snippets; run any demo directly:

.. code-block:: console

   python examples/analysis/01_expectation.py

.. toctree::
   :maxdepth: 2

   bell_state

.. note::

   The examples are exercised in CI like tests: every demo in ``examples/``
   is executed against the *installed* package to keep the SDK honest.

Example tree (excerpt, newest first)
------------------------------------

.. code-block:: text

   examples/
   ├── classification/         quantum kernels / classifiers
   ├── algorithms/             VQE, QAOA, Grover, phase estimation, ...
   ├── analysis/               expectation & sampling analysis
   ├── backends/               local backends, noise, MPS
   ├── benchmarks/             quantum volume, randomized benchmarking, XEB
   ├── chemistry/              molecular Hamiltonians, UCCSD
   ├── core/                   circuits, gates, measurement, parameters
   ├── experiments/            Experiment / ParameterSweep / records
   ├── integration/            runtime-plan-backend end to end
   ├── ir/                     IR, compilation, transpiler passes
   ├── optimization/           QUBO builders, Ising conversion
   ├── problems/               sampling/optimization/hamiltonian/search
   ├── qec/                    repetition/Shor codes
   ├── qml/                    angle / amplitude / IQP encodings
   ├── quantum_technologies/   walk + QFT + simulation
   ├── runtime/                ExecutionPlan / ExecutionRuntime / batch
   └── tutorials/              walkthroughs of the higher-level flows

Bell State
----------

Create and measure a Bell state:

.. code-block:: python

   from microquantum import QuantumCircuit

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   state = qc.run()
   print(f"Bell state: {state}")