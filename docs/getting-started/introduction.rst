Introduction
============

MicroQuantum is a **general-purpose Python quantum computing SDK foundation**.
It is implemented independently, from scratch, with NumPy as the only hard
dependency — it is not a wrapper around Qiskit, Cirq or OpenQASM.

This Developer Preview (v0.4.0) ships a complete local stack:

.. code-block:: text

   Problem
      │   SamplingProblem · OptimizationProblem · HamiltonianProblem ·
      │   EigenvalueProblem · SearchProblem
      ▼
   Algorithm
      │   VQE · QAOA · GroverSearch · PhaseEstimation · QFT · HHL · ...
      ▼
   ExecutionPlan
      │   declarative, JSON-safe: work + backend + shots + bindings + seed
      ▼
   ExecutionRuntime
      │   prepare · compile · submit · collect (batch / sweep / hybrid)
      ▼
   Backend
      │   capability-driven contract: validate · supports · execute
      ▼
   BackendResult
      │   statevector · density matrix · counts · expectations · samples
      ▼
   Experiment / Analysis
          ExecutionRecord · ParameterSweep · SamplingAnalysis ·
          ExpectationAnalysis · StateAnalysis · ResultAggregator

Goals of this release
---------------------

* A **usable public SDK**: `pip install microquantum` and go.
* A **single dependency** (NumPy) and an independent implementation.
* A **capability-oriented backend architecture** ready for future real
  hardware / cloud providers without rearchitecting the SDK.
* **Reproducible, serializable results** end to end.

What it is *not* (yet)
----------------------

This release does **not** provide hardware execution, a cloud platform, an
enterprise database-backed experiment service, or an analytics dashboard.  The
architecture contains the boundaries where those would plug in later — it does
not yet implement them.

See the :ref:`developer-preview` page for the full status and limitations.