MicroQuantum
============

.. raw:: html

   <style>
     .mq-hero { padding-bottom: 1.2em; }
     .mq-badges { margin: 0.8em 0 1.2em; }
   </style>

.. rst-class:: mq-hero

**A lightweight, NumPy-only Python quantum computing SDK foundation.** Build a
circuit. Execute it locally. Understand the result. Formulate a problem, run an
algorithm, run an experiment and analyze the outcome — with a pluggable backend
architecture that is ready for future hardware providers.

.. rst-class:: mq-badges

.. list-table::
   :widths: 30 70

   * - Status
     - **Developer Preview** (v0.4.0). Clean, tested, documented — but ``0.x``:
       APIs may still evolve before 1.0.
   * - Runtime
     - Pure Python + NumPy. No Qiskit, Cirq or OpenQASM dependency.
   * - Execution
     - Local CPUs (statevector / density-matrix / tensor-network / MPS), a
       deterministic mock, and a provider boundary for future hardware.
   * - License
     - MIT.

----------

.. toctree::
   :maxdepth: 2
   :caption: Get Started

   getting-started/introduction
   getting-started/installation
   getting-started/quickstart
   getting-started/first-circuit
   getting-started/first-measurement
   getting-started/first-problem
   getting-started/first-algorithm
   getting-started/first-experiment

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/overview
   concepts/quantum-states
   concepts/circuits
   concepts/gates
   concepts/parameters
   concepts/measurement
   concepts/problems
   concepts/algorithms
   concepts/serialization

.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   algorithms/overview
   algorithms/vqe
   algorithms/qaoa
   algorithms/grover
   algorithms/phase-estimation

.. toctree::
   :maxdepth: 2
   :caption: Problems

   problems/overview
   problems/sampling
   problems/optimization
   problems/qubo-ising
   problems/hamiltonian
   problems/eigenvalue
   problems/search

.. toctree::
   :maxdepth: 2
   :caption: Execution & Backends

   execution/execution-plan
   execution/runtime
   execution/backends
   execution/capabilities
   execution/providers
   execution/custom-backends

.. toctree::
   :maxdepth: 2
   :caption: Experiments

   experiments/execution-records
   experiments/parameter-sweeps
   experiments/experiments
   experiments/results
   experiments/reproducibility

.. toctree::
   :maxdepth: 2
   :caption: Analysis

   analysis/overview
   analysis/sampling
   analysis/expectations
   analysis/states
   analysis/statistics
   analysis/aggregation

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   examples/index
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core
   api/algorithms
   api/problems
   api/experiments
   api/analysis
   api/backends
   api/adapters
   api/benchmarks
   api/optimization

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer-guide/architecture
   developer-guide/development
   developer-guide/extending

.. toctree::
   :maxdepth: 2
   :caption: Project

   releases/developer-preview
   releases/changelog
   releases/compatibility
   contributing
   faq
   troubleshooting

----------

Overview
========

MicroQuantum is an independent quantum computing SDK: every circuit, operator,
simulator and algorithm is implemented from scratch with NumPy as the only hard
dependency. It is **not** a wrapper around Qiskit, Cirq or OpenQASM.

What the SDK provides today:

* **Core engine** — quantum circuits, operators, Pauli algebra,
  :class:`~microquantum.StateVector` / :class:`~microquantum.DensityMatrix`,
  measurement, gradients, registers, serialization, a transpiler and a
  :class:`~microquantum.PassManager` pipeline.
* **Problems** — plain, JSON-safe descriptions of computational tasks
  (:class:`~microquantum.SamplingProblem`, :class:`~microquantum.OptimizationProblem`,
  :class:`~microquantum.HamiltonianProblem`, :class:`~microquantum.EigenvalueProblem`,
  :class:`~microquantum.SearchProblem`).
* **Algorithms** — VQE, QAOA, Grover search, phase estimation, QFT, HHL,
  Hamiltonian simulation and more, through a uniform
  ``validate(problem) -> solve(problem, runtime)`` lifecycle.
* **Execution runtime** — declarative :class:`~microquantum.ExecutionPlan` s,
  batch/sweep orchestration and an :class:`~microquantum.ExecutionRuntime` that
  routes work to backends.
* **Backends & providers** — statevector, density-matrix, MPS and tree-tensor
  network simulators, a deterministic :class:`~microquantum.MockBackend`, a
  capability-driven :class:`~microquantum.Backend` contract and a
  :class:`~microquantum.Provider` discovery boundary for future hardware.
* **Experiments** — :class:`~microquantum.ExecutionRecord` s,
  :class:`~microquantum.ParameterSweep` s and :class:`~microquantum.Experiment` s
  with reproducibility fingerprints and JSON-safe serialization.
* **Analysis** — sampling, expectation, state and result-aggreation analysis on
  top of the existing result contracts.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`