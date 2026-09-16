=============================================
microquantum MQ-16 Compiler-Runtime Report
=============================================

Scope
=====

MQ-16 proves that programs handed from the :class:`~microquantum.Compiler`
to the execution runtime keep the *observable* semantics of the original
circuit: exact probability vectors on ``shots=None`` and matching seeded
sampling estimates, across all four simulators and all three optimization
levels, including directed two-qubit gates, measurements, symbolic
parameters and negative rotation angles.  It adds three executable
examples, a consolidated-notebook extension, documentation, a shared
equivalence helper module, and a regression + conformance suite that
enforce the contract permanently.

The milestone rides entirely on existing APIs.  No second runtime, no
duplicate abstraction, no nearby API breakage and no new dependency were
introduced.

A single genuine core defect surfaced during validation — two-qubit gate
*target ordering* was ignored on the density-matrix and executor expansion
paths — and is documented and fixed below (agent should report, not
paper-over: user approved fixing it now).

Deliverables
============

* **Three execution routes, one semantics.** A compiled program runs
  identically through (A) explicit ``backend.run(compiled)``,
  (B) ``ExecutionPlan(compiled=...)`` reuse, and (C) runtime compilation
  via ``execute(..., optimization_level=...)`` (examples 30–31 + tests).
* **Differential oracle.** For the corpus (Bell, GHZ(3), a rotation gate
  over ``±π`` across all quadrants, CZ, SWAP) every level `0/1/2`
  preserves: exact probability vectors (`max |Δp| < 1e-9`) on all four
  simulators — statevector, density matrix, MPS, TTN — and seeded
  statistical agreement (`≤ 0.05`, `4000` shots).
* **Directed two-qubit gates are part of the contract.** ``cnot(1, 0)``
  must act with reversed targets; SWAP and CZ basis decompositions must
  match on every backend.  This is what exposed the core defect below.
* **Measurements survive the hand-off.** Compiling an unmeasured circuit
  appends terminal measurements; both original and compiled produce
  full-width counts/probabilities.
* **Parameters through the runtime.** ``compile(bind) == bind(compile)``
  (state fidelity ``1.00``); the runtime resolves ``parameter_bindings`` on
  a compiled plan (``metadata["strategy"] == "compiled"``); unbound/unknown
  parameters raise ``ValueError``.
* **Error contracts preserved.** ``ExecutionPlan.compiled`` is validated
  to be a :class:`~microquantum.CompilationResult` (``TypeError``
  otherwise, after the plan's exclusivity ``ValueError``); incompatible
  targets are reported via ``is_compatible`` and rejected unless
  ``raise_on_incompatible=False``; reset nodes and classically conditioned
  blocks raise ``ValueError`` at rebuild.
* **Examples 30–32.** ``30_compile_and_execute.py`` (the three routes),
  ``31_compile_compare_execution.py`` (the differential oracle),
  ``32_parameterized_compile_execute.py`` (compile → bind → runtime).
* **Shared helpers.** ``tests/compiler_runtime_helpers.py`` — tolerances
  (`TOL_EXACT = 1e-9`, `TOL_FIDELITY = 1e-9`, `TOL_SAMPLING = 0.05`,
  `SHOTS_SAMPLED = 4000`), corpus, simulator collection and
  ``assert_execution_equivalent`` / ``assert_runtime_equivalent``.
* **Regression suite.** ``tests/test_compiler_mq16_integration.py`` —
  249 tests.
* **Conformance suite.** ``tests/conformance/test_compiler_runtime_integration.py`` —
  92 tests, run with the regular pytest command (self-contained; it imports
  the shared helpers by inserting the ``tests/`` directory on ``sys.path``).
* **Documentation.** ``docs/execution/compiler-runtime.rst`` registered in
  the Execution & Backends toctree; the examples README, examples index tree
  and this report updated.

Core defect found and fixed
===========================

``expand_operator`` (``src/microquantum/core/tensor.py``) short-circuited
when the gate qubit count equals the system width, using ``sorted(targets)``
to recognize the "full-width" fast path.  When the two qubits were the full
system but *not* in canonical order, both orderings collapsed to the
canonical gate: on two qubits ``cnot(1, 0)`` executed as ``cnot(0, 1)``.
The statevector path (einsum ``apply_gate``) and the MPS/TTN simulators
were already order-correct; the density-matrix and executor backends shared
the flawed ``expand_operator`` path.  This is why ``cx(1, 0)`` from a SWAP
decomposition went unnoticed at the routine level but was caught by the
MQ-16 cross-backend determinism gate.

Fix: the fast path now requires the *exact* canonical order
(``targets == list(range(n))``); any permutation is routed through the
general, order-correct expansion path.  Regression tests:
``tests/test_tensor.py`` (``test_expand_cnot_reversed_targets_respects_order``)
and the cross-backend determinism tests in the MQ-16 suite.

API change from this milestone (reported, additive)
---------------------------------------------------

* ``ExecutionPlan.compiled`` — plan validation now raises
  ``TypeError`` when ``compiled`` is not a
  :class:`~microquantum.CompilationResult` (in the compiled-only branch,
  after the exclusivity ``ValueError``).  Existing valid plans construct
  unchanged; the change is a stricter, local validation, no signature or
  behavior regression.

Gate results
============

================ ========================================================== ====================
Gate             Command                                                     Result
================ ========================================================== ====================
Tests (targeted) ``uv run pytest tests/test_tensor.py tests/test_backends.py`` 119 passed, 0 failed
                 ``tests/conformance/test_simulation_contracts.py``
                 ``tests/test_execution_core.py`` -q --no-cov
Tests (MQ-16)    ``uv run pytest tests/test_compiler_mq16_integration.py`` 249 passed in 56.11s
                 ``-q --no-cov``
Conformance      ``uv run pytest tests/conformance/test_compiler_runtime_``   92 passed in 8.60s
(MQ-16 file)     ``integration.py -q --no-cov``
Examples         ``uv run pytest tests/conformance/test_official_examples_``  86 passed
                 ``output.py -q --no-cov``
================ ========================================================== ====================

Note: the default pytest addopts add ``--cov=microquantum``; there is no
``fail_under`` threshold, so the functional gates are reported with
``--no-cov``.

MQ-16 suite contents
====================

``tests/test_compiler_mq16_integration.py``
-----------------------------------------------

* Equivalent semantics through the runtime: per-corpus exact probability
  agreement (all 4 backends × levels 0/1/2) and seeded sampling agreement.
* Route equivalence (A/B/C) and rich-result parity.
* Measurements: unmeasured compilation appends terminal measurements;
  compiled vs original counts match.
* Parameter pipeline: symbolic compile → bind; bind-after-compile fidelity;
  compiled-plan ``parameter_bindings``; equals IR-facing param counts.
* Negative rotation angles across the four quadrants.
* Metadata: ``compiled_gates_by_type`` counts for canonical IR names.
* ``ExecutionPlan``/error contracts and incompatible-target diagnostics.
* ``TestCrossBackendDeterminism``: directed ``cnot(1, 0)``, SWAP and CZ
  basis decompositions agree across all four simulators.

``tests/conformance/test_compiler_runtime_integration.py``
----------------------------------------------------------

The same contract as a compact, permanent gate (92 items): corpus matrix,
sampling, routes, directed gates, basis decompositions, parameters,
measurements, empty circuit, negative rotations, ``TypeError``/``ValueError``
contracts, incompatible plans and reset-IR rejection.

Design notes
============

* ``MockBackend`` is intentionally excluded from the simulator collection:
  it is a routing stub, not a simulator, and the equivalence contract is
  defined for real simulation targets.
* IR gate names are canonical (``cnot``), so metadata assertions use them
  rather than source spellings.
* The conformance file keeps a bare ``sys.path`` insertion (``tests/``) so
  it can reuse the shared helpers without duplicating logic; the import is
  marked ``# noqa: E402``.

Count reconciliation
====================

MQ-15 recorded ``tests/conformance/`` as **395 collected items**.  MQ-16
adds: 92 items from ``test_compiler_runtime_integration.py``,
+3 example-execute and +3 example-output parametrizations (examples 30–32),
and a +1 docs-consistency item for the new ``execution/compiler-runtime.rst``
page (one self-contained ``.. code-block:: python``).  The suite remains
**100% green with no skips, no xfails, no quarantines and no weakened
assertions.**

Examples
========

The three new examples are auto-discovered by the conformance gate
(``conformance_helpers.all_example_files()``); all **86 scripts** run clean
as standalone programs and again, in order, inside the consolidated
notebook.

Notebook
========

``examples/MicroQuantum_Examples.ipynb`` was regenerated to include
examples 30–32; validation (14 sections, full example coverage, in-order
execution) is unchanged and enforced by the conformance suite.

No dependency changes
=====================

MQ-16 adds no runtime, test or documentation dependencies.

Release prerequisite reminder (AGENTS.md)
=========================================

No PyPI release without: push + green CI + ``v<version>`` tag + TestPyPI
check first.