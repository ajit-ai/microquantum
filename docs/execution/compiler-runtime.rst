Compiler ↔ Runtime Integration
==============================

The :class:`~microquantum.Compiler` produces a reusable
:class:`~microquantum.CompilationResult`.  MQ-16 proves that programs built
from that result — the "compiler ↔ runtime" hand-off — preserve the exact,
observable behavior of the original circuit on every simulator in the SDK.

Three execution routes
----------------------

A compiled program can be executed three equivalent ways:

.. list-table::
   :header-rows: 1

   * - Route
     - How
     - When to use
   * - A: explicit
     - ``Compiler(...).compile(qc).circuit()`` then ``backend.run(compiled)``
     - You keep full control of the compiled artifact and the backend.
   * - B: plan reuse
     - ``ExecutionPlan(compiled=result, backend=..., ...)`` passed as
       ``execute(plan=...)`` — no recompiling
     - Reusing one compilation across runs / parameter bounds.
   * - C: at run time
     - ``execute(qc, optimization_level=..., ...)`` — the runtime compiles
     - One-shot execution where compilation parameters live in the call.

All three routes return equivalent results: for a seeded sampler they produce
matching counts within the documented statistical tolerance (see
:doc:`runtime`).

Semantic equivalence contract
-----------------------------

MQ-16 defines *observable equivalence* between an original circuit and its
compiled form:

* **Exact.** On ``shots=None`` the probability vector of the compiled circuit
  matches the original within ``1e-9`` on every simulator
  (:class:`~microquantum.StatevectorBackend`, :class:`~microquantum.DensityMatrixBackend`,
  :class:`~microquantum.MPSBackend`, :class:`~microquantum.TreeTensorNetworkBackend`).
* **Statistical.** On equal shots and seed, per-outcome probabilities agree
  within ``0.05``, consistent with the sampler's statistical spread.
* **Directed gates.** Two-qubit gate *order* is part of the contract: e.g.
  ``cnot(1, 0)`` must act as the reverse-target application, never as
  ``cnot(0, 1)``.  This is enforced across all backends (including the
  density-matrix and executor paths) by the MQ-16 suites.
* **Parameters.** Compiling a symbolic circuit keeps its parameters; binding
  after compilation is equivalent to binding before
  (``compile(bind) == bind(compile)``), and the runtime resolves
  ``parameter_bindings`` on a compiled plan.

Compatibility and errors
------------------------

The integration reuses the existing runtime error contract — nothing new was
introduced:

* A target without ``supports_measurement`` is reported through
  ``CompilationResult.is_compatible`` and rejected by execution unless
  ``raise_on_incompatible=False``.
* ``ExecutionPlan.compiled`` must be a :class:`~microquantum.CompilationResult`
  (a ``TypeError`` otherwise); dynamic constructs the compiler cannot
  represent (reset nodes, classically conditioned blocks) raise ``ValueError``
  at rebuild time.
* Unbound or unknown parameters raise ``ValueError`` at bind / run time.

Example
-------

.. code-block:: python

   from microquantum import (
       Compiler,
       QuantumCircuit,
       StatevectorBackend,
       execute,
   )
   from microquantum.runtime.plan import ExecutionPlan

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.h(0)          # identity pair -> optimized away at level >= 1
   qc.cx(0, 1)

   compiled = Compiler(optimization_level=1).compile(qc)
   rebuilt = compiled.circuit()
   backend = StatevectorBackend()

   # Route A
   a = dict(backend.run(rebuilt, shots=1024, seed=7).counts)
   # Route B
   plan = ExecutionPlan(compiled=compiled, backend=backend, shots=1024, seed=7)
   b = dict(execute(plan=plan).counts)
   # Route C
   c = dict(execute(qc, backend=backend, shots=1024, seed=7,
                    optimization_level=1).counts)

   assert a == b == c
   print("compiled == original == runtime-compiled:", a)

*The same program, three routes, one answer.* The compiled artifact is
interchangeable with the original circuit as an execution input.

Verification
------------

MQ-16 is enforced permanently by two suites:

* ``tests/test_compiler_mq16_integration.py`` — the regression suite
  (equivalent semantics through the runtime, sampling, measurement,
  parameters, negative angles, error contracts, cross-backend determinism).
* ``tests/conformance/test_compiler_runtime_integration.py`` — the permanent
  conformance gate, run with the regular pytest command.

See the :doc:`MQ-16 record
<../quality/conformance/CONFORMANCE_MQ16_REPORT>` for counts and the single
core defect this milestone found and fixed (two-qubit target ordering on the
density-matrix and executor expansion paths).