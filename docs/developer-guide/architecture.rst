Architecture
============

MicroQuantum is deliberately layered so each level depends only on the levels
below it:

.. code-block:: text

   Experiments / Analysis
            │
   Runtime / Backends / Providers
            │
   Algorithms
            │
   Problems
            │
   Core engine (circuits · operators · states · IR · transpiler)
            │
         NumPy

Source layout
-------------

.. code-block:: text

   src/microquantum/
   ├── __init__.py          public package surface (numpy-only)
   ├── core/                circuits, operators, pauli, states, measurement,
   │                        parameters, gradients, registers, qasm, IR input,
   │                        transpiler passes, dynamic circuits, device/target
   ├── ir/                  intermediate representation & compiler
   ├── backends/            backend contract, capabilities, registry,
   │                        provider, adapters + local simulators
   ├── runtime/             ExecutionPlan, ExecutionRuntime, strategies
   ├── problems/            Sampling / Optimization / Hamiltonian /
   │                        Eigenvalue / Search
   ├── algorithms/          VQE, QAOA, Grover, QPE, QFT, HHL, walks, ...
   ├── optimizers/          classical optimizers (NumPy-only)
   ├── experiments/         records, sweeps, experiments (MQ-07)
   ├── analysis/            sampling / expectation / state / aggregation
   ├── optimization/        QUBO builder, Ising converter
   ├── providers/           optional vendor boundary (never imported by SDK)
   ├── adapters/            domain adapters (QuantumProblem/QuantumResult)
   ├── qml/ qec/ chemistry/ benchmarks/ mitigation/   feature libraries
   └── analytics/           CSV loaders & base analytics

Layering rules
--------------

1. **Independent implementation** — all quantum operations are implemented
   from scratch with NumPy; no Qiskit/Cirq/OpenQASM dependency.
2. **Layer separation** — analytics never touch raw matrices; problems never
   execute; backends never re-implement simulation logic.
3. **Big-endian qubit ordering** — qubit 0 is most-significant; tensor axis 0
   = qubit 0 (the mathematical convention).
4. **Standardized results** — every successful run funnels into a typed,
   JSON-serializable result.
5. **Public surface only** — ``__init__.py`` is the friendly gateway;
   internal helpers stay private.
6. **NumPy-only runtime** — GPU acceleration is opt-in at the array layer
   (:func:`~microquantum.set_array_backend`).

Execution flow
--------------

.. code-block:: text

   Problem -> Algorithm -> ExecutionPlan -> (compile?) -> Backend
                                                        -> Job -> BackendResult
        ExecutionRecord (one per run) -> ExperimentResult -> Analysis

The runtime and backend layers never re-implement simulation; a user-provided
:class:`~microquantum.Backend` drops in through a plan's ``backend`` field.