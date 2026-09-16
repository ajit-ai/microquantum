=============================================
microquantum MQ-15 Compiler Pipeline Report
=============================================

Scope
=====

MQ-15 ships the single, deterministic compilation pipeline for
MicroQuantum: the :class:`~microquantum.Compiler` entry point with explicit
optimization levels, a safe, state-preserving pass set, target/capability
aware lowering, metadata-rich ``CompilationResult``\ s, and loud rejection of
unsupported constructs.  It rides entirely on the existing
:class:`~microquantum.IRCircuit` / :class:`~microquantum.IRPass` /
:class:`~microquantum.Target` infrastructure — no competing or parallel
compiler API was introduced.  The milestone also adds eight new executable
examples, a consolidated developer notebook that walks the whole example
corpus, documentation and a permanent regression + conformance suite.

Deliverables
============

* **Single entry point.** ``Compiler(source=...)`` and
  ``Compiler.compile(...)`` remain the only public compile pathway;
  ``IRPassManager.run(ir, pass_specs=)`` stays the low-level escape hatch.
  No ``NewCompiler`` / ``TranspilerV2``-style APIs were added.
* **Optimization levels.** ``level=0`` is validation-only
  (``passes_applied == []``); ``level=1`` applies
  ``remove-identity-gates`` + ``cancel-adjacent-inverse``; ``level=2`` adds
  ``combine-rotations``.  Levels outside ``0..2`` raise ``ValueError`` in
  both ``Compiler`` and ``IRPassManager.run``.
* **Exact state preservation.** The final normalized state after level-1/2
  optimization (:class:`~microquantum.StatevectorBackend` fidelity) is
  *bit-for-bit identical* to the source circuit.
* **Measurements survive.** Terminal measurements keep their registry,
  ordering and subset through compilation; ``Compiler.from_ir`` restores
  them (``quantum == classical``) and rejects classical-bit mismatches.
  Reset and classically-conditioned blocks are rejected loudly, never
  silently dropped.
* **Symbolic parameters survive.** ``cq.compile()`` carries
  ``Parameter``/``ParameterExpression`` angles into the IR verbatim, and
  ``compile(bind) == bind(compile)``.  Symbolic rotations are never fused.
* **Target-aware lowering.** ``Target(native_gates=("h", "cx"))`` compiles
  ``cz`` (→ H + CNOT + H) and ``swap`` (→ three CNOTs); ``cx``/``cnot``
  spellings are normalized to one IR gate name; unsupported gates stay in
  the IR alongside a compatibility diagnostic (never silenced).
  Capabilities and target gate spans are both enforced.
* **Validation.** Rotation gates in the IR accept only numeric or symbolic
  parameters and reject ``complex`` angles; ``Compiler`` validates IR
  structure up front with clear, actionable errors.
* **Metadata.** ``CompilationResult.metadata`` reports
  ``optimization_level``, source/compiled qubit, gate and depth counts and
  per-type gate breakdowns.
* **Examples 22–29.** ``22_ir_roundtrip.py``, ``23_ir_to_circuit.py``,
  ``24_basic_compilation.py``, ``25_optimization_before_after.py``,
  ``26_parameterized_compilation.py``, ``27_measurement_preserving_optimization.py``,
  ``28_target_validation.py``, ``29_execute_compiled_circuit.py``.
* **Consolidated notebook.** ``examples/MicroQuantum_Examples.ipynb`` — a
  single walkthrough of the entire example corpus (83 examples).  Every code
  cell executes the corresponding authoritative ``.py`` file through a
  ``run_example`` helper, so the notebook cannot drift from the canonical
  examples and needs no Jupyter runtime on the SDK.
* **Documentation.** New ``docs/execution/compilation.rst`` registered in the
  Execution & Backends toctree; ``examples/README.md`` and
  ``docs/examples/index.rst`` updated for the new demos and the notebook.
* **Regression suite.** ``tests/test_compiler_mq15.py`` (78 tests).
* **Conformance suite.** ``tests/conformance/test_compiler_contracts.py``
  (25 tests) and ``tests/conformance/test_notebook_validation.py``
  (8 tests: valid notebook, full example coverage, no missing file
  references, in-order execution of every cell).

Gate results
============

========= ==================================================== ======================
Gate      Command                                              Result
========= ==================================================== ======================
Tests     ``uv run pytest tests/ -q --no-cov``                 2161 passed, 0 failed
(full)
mypy      ``uv run mypy src/microquantum/ --ignore-missing-     Success, 127 files,
          imports``                                            no issues
ruff      ``uv run ruff check src/microquantum/ tests/          All checks passed
          examples/``
Sphinx    ``uv run sphinx-build -E -W -b html docs <out>``      Build succeeded,
                                                               0 warnings
Conformance ``uv run pytest tests/conformance/ --collect-only`` 395 tests collected
===============================================================================

Note: the default pytest addopts add ``--cov=microquantum``; coverage
tracing slows the suite (~20x on some files) and there is no ``fail_under``
threshold, so the functional gate above is reported with ``--no-cov``.

Conformance suite ``tests/conformance/`` (collected: 395)
==========================================================

.. list-table::
   :widths: 33 8 60
   :header-rows: 1

   * - File
     - Collected
     - What it proves
   * - ``test_compiler_contracts.py``
     - 25
     - MQ-15 compiler contract: entry point shape, no silent gate drops,
       level bounds, determinism, metadata, measurement/parameter
       preservation, target decomposition, cross-backend execution.
   * - ``test_notebook_validation.py``
     - 8
     - The consolidated notebook is well formed, covers every official
       example, never references a missing file, and every code cell
       executes in order (stdlib only, no Jupyter runtime).
   * - ``test_docs_consistency.py``
     - 101
     - Every ``.. code-block:: python`` on 51 hand-written pages runs
       (shared namespace/page); every dotted ``microquantum.x.y`` doc
       reference resolves.
   * - ``test_official_examples_execute.py``
     - 83
     - All official example scripts in ``examples/`` execute without
       failure (one parametrized item per script).
   * - ``test_official_examples_output.py``
     - 83
     - Each example prints its contract (regex/attribute-based registry).
   * - ``test_core_api_contracts.py``
     - 11
     - Constructor signatures, errors, counts, params, circuits, states,
       problems, operators, optimizers surface.
   * - ``test_mq_milestone_verification.py``
     - 14
     - MQ-11 execution core + parameter sweep, MQ-12 Grover/QAOA/VQE,
       MQ-13 experiments/analysis.
   * - ``test_problems_algorithms_api.py``
     - 17
     - Grover, Shor, QAOA, VQE/H2, HamiltonianSimulation, optimizers,
       search/optimization problems.
   * - ``test_serialization_round_trips.py``
     - 12
     - to_dict/from_dict/to_json/from_json round trips incl. nested and
       typed content.
   * - ``test_simulation_contracts.py``
     - 38
     - MQ-14 simulator contract: deterministic ``shots=None``, shots
       validation, seeds, capabilities/targets, Executor conflicts, memory
       boundaries, TTN sampling cap, cross-simulator parity, serialization.
   * - ``test_source_vs_wheel_install.py``
     - 3
     - Fresh wheel installs cleanly in a clean venv; byte-identical
       behavior + version to source install.

Count reconciliation
====================

The :doc:`MQ-14 Conformance Report <CONFORMANCE_MQ14_REPORT>` recorded
``tests/conformance/`` as **344 collected items**.  MQ-15 grows that to
**395**: +25 compiler contracts, +8 notebook validation, +8 example-execute
parametrizations (examples 22–29) +8 example-output items, and +2 docs
consistency items for the new ``execution/compilation.rst`` page.  All three
of those additions sum exactly to the +51 headline delta.

The suite remains **100% green with no skips, no xfails, no quarantines and
no weakened assertions.**

Examples
========

The eight new examples are auto-discovered by the conformance gate
(``conformance_helpers.all_example_files()``); all **83 scripts** run clean
as standalone programs and again, in order, inside the consolidated
notebook.

Notebook
========

``examples/MicroQuantum_Examples.ipynb`` (``nbformat`` 4):

* 183 cells — 99 markdown, 84 code.
* 14 numbered sections plus a table of contents; the code cells cover all
  **83** official example files, each preceded by a ``**Source example:**
  `` <file>`` `` marker.
* Every code cell executes the canonical example with ``__name__ ==
  "__main__"`` from the repository root — the same behavior as
  ``uv run python examples/<file>.py`` — including all of the example's own
  assertions.
* No Jupyter runtime dependency; validation is stdlib-only and enforced by
  the conformance suite.

Defects fixed during the gate
=============================

* ``_rotation_angle`` recovered rotation angles with ``acos``, which loses
  the sign of negative ``rx``/``ry`` angles: ``ry(-1.1)`` round-tripped as
  ``ry(+1.1)`` (state fidelity 0.535, silently wrong).  Recovery now uses
  ``atan2`` and preserves the sign bit for ``rx``/``ry``/``rz``.
* ``QuantumCircuit.to_ir``/``CircuitBuilder.from_ir`` dropped terminal
  measurements or mishandled measured qubits; ``from_ir`` now restores them
  and rejects classical-bit mismatches explicitly.
* IR validation accepted non-numeric/complex rotation parameters; the
  validation pass now rejects them loudly instead of producing degenerate
  rotations downstream.
* A docs toctree indent regression excluded new execution pages from the
  navigation tree; the tree was repaired and the build is clean again.

No skips, no xfails, no quarantines were added in any file.

No dependency changes
=====================

MQ-15 adds no runtime, test or documentation dependencies.

Release prerequisite reminder (AGENTS.md)
=========================================

No PyPI release without: push + green CI + ``v<version>`` tag + TestPyPI
check first.