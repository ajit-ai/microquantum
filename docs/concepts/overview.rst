Overview
========

MicroQuantum exposes a small set of public contracts that compose into a
single execution model.  The same interfaces serve local NumPy simulation
today and CPU/GPU/NPU accelerators and quantum hardware in the future.

Canonical pipeline
------------------

.. code-block:: text

   Problem
      │   SamplingProblem · OptimizationProblem · HamiltonianProblem ·
      │   EigenvalueProblem · SearchProblem
      ▼
   Algorithm
      │   validate(problem) -> solve(problem, runtime)
      ▼
   ExecutionPlan
      │   declarative, JSON-safe: work + backend/target + shots + bindings + seed
      ▼
   ExecutionRuntime
      │   prepare -> compile -> submit -> collect
      ▼
   Backend
      │   capabilities -> validate(plan) -> supports(plan) -> execute(plan)
      ▼
   BackendResult
      │   statevector · density matrix · counts · probabilities ·
      │   expectations · samples · eigenvalues
      ▼
   Experiment / Analysis
          ExecutionRecord · ParameterSweep · SamplingAnalysis ·
          ExpectationAnalysis · StateAnalysis · ResultAggregator

Layering
--------

Each layer only depends on the layers below it:

.. code-block:: text

   Experiments / Analysis
            │
   Runtime / Backends / Providers
            │
   Algorithms
            │
   Problems
            │
   Core engine (circuits · operators · states · IR)
            │
         NumPy

Public contracts
----------------

* :class:`~microquantum.core.circuit.QuantumCircuit` — the program: gates,
  measurements, bound and symbolic parameters.
* :class:`~microquantum.core.device.Device` / :class:`~microquantum.core.device.Target`
  — capability-oriented descriptors of *where* a job runs and *which*
  execution constraints it satisfies.
* :class:`~microquantum.runtime.ExecutionPlan` — declarative, JSON-safe
  description of *what* to run, *where* and *how* (shots, bindings, seed,
  options, metadata).  Plans never execute anything.
* :class:`~microquantum.runtime.ExecutionRuntime` — the coordinator:
  prepare, optionally compile, submit, collect.
* :class:`~microquantum.backends.base.Backend` — the execution contract:
  ``capabilities``, ``validate(plan)``, ``supports(plan)``, ``execute(plan)``.
* :class:`~microquantum.backends.base.BackendResult` — execution output with
  JSON-safe serialization.
* :class:`~microquantum.experiments.record.ExecutionRecord` — portable record
  of one actual execution (raw result preserved).
* :class:`~microquantum.problems.base.Problem` — plain JSON-safe task
  description, never an execution.

Design principles
-----------------

1. **Independent implementation** — all quantum operations are implemented
   from scratch with NumPy; no Qiskit/Cirq/OpenQASM dependency.
2. **Layer separation** — analytics never touch raw matrices; problems never
   execute; backends never re-implement simulation logic.
3. **Big-endian qubit ordering** — qubit 0 is the most-significant bit;
   tensor axis 0 = qubit 0 (the mathematical convention).
4. **Standardized results** — every successful run funnels into a typed
   result with ``to_dict()`` / ``to_json()``.
5. **Reproducible by configuration** — execution fingerprints hash the stable
   serialized configuration, never object addresses.
6. **Test-driven** — a full test suite ships with the SDK and runs in CI.