============================================
microquantum A-Z Conformance Report
============================================

Scope
=====

Prove the public ``microquantum`` SDK surface end-to-end with a
no-skip / no-xfail conformance test suite, fix every defect it surfaces,
and pass the standard verification gates.

Branch workflow completed
=========================

The conformance work landed as ``30b7e10`` and ``29e929a`` on ``develop``,
was merged into ``main`` with a no-fast-forward merge (``d2578de``), and
was pushed. The report itself was subsequently preserved into the
repository documentation at ``docs/quality/conformance/``.

Gate results
============

======== ====================================================== ======================================
Gate     Command                                                Result
======== ====================================================== ======================================
Tests    ``uv run pytest tests/ -q --no-cov``                   1900 passed, 0 failed
(full)
mypy     ``uv run mypy src/microquantum/ --ignore-missing-       Success, 126 files, no issues
         imports``
ruff     ``uv run ruff check src/microquantum/ tests/            All checks passed
         conformance/``
Sphinx   ``uv run sphinx-build -W -b html docs <out>``           Build succeeded, 0 warnings
======== ====================================================== ======================================

Note: the default pytest addopts add ``--cov=microquantum``; coverage
tracing is roughly 20x slower on this suite and there is no ``fail_under``
threshold, so the functional gate above is reported with ``--no-cov``.

Conformance suite ``tests/conformance/`` (219 tests)
=====================================================

.. list-table::
   :widths: 33 8 60
   :header-rows: 1

   * - File
     - Tests
     - What it proves
   * - ``test_core_api_contracts.py``
     - 11
     - Constructor signatures, errors, counts, params, circuits, states,
       problems, operators, optimizers surface.
   * - ``test_official_examples_execute.py``
     - 1
     - All 129 example scripts in ``examples/`` execute without failure.
   * - ``test_official_examples_output.py``
     - 70
     - Each example prints its contract (regex/attribute-based registry).
   * - ``test_problems_algorithms_api.py``
     - 17
     - Grover, Shor, QAOA, VQE/H2, HamiltonianSimulation, optimizers,
       search/optimization problems.
   * - ``test_mq_milestone_verification.py``
     - 14
     - MQ-11 execution core + parameter sweep, MQ-12 Grover/QAOA/VQE,
       MQ-13 experiments/analysis.
   * - ``test_serialization_round_trips.py``
     - 12
     - to_dict/from_dict/to_json/from_json round trips incl. nested and
       typed content.
   * - ``test_docs_consistency.py``
     - 97
     - Every ``.. code-block:: python`` on 48 hand-written pages runs
       (shared namespace/page); every dotted ``microquantum.x.y`` doc
       reference resolves.
   * - ``test_source_vs_wheel_install.py``
     - 3
     - Fresh wheel installs cleanly in a clean venv; byte-identical
       behavior + version to source install.

Defects found and fixed during the gate
=======================================

API-level (docs/examples were stale; the SDK was correct):

* ``depth`` was documented as an attribute in the class docstring but is a
  method (``examples/01_basic_circuits.py`` called ``qc.depth``); the SDK
  had already made ``depth()`` callable (commit ``994ef4c``). Aligned call
  sites + docstring; reworded the docstring "Methods:" list that autoapi
  duplicated (Sphinx -W fix).
* ``SearchProblem(name=...)`` received ``name`` positionally and by
  keyword.
* ``PhaseEstimation(num_ancillae=...)`` -> ``num_counting_qubits``;
  ``unitary`` required.
* ``QAOA(p=...)`` Qiskit-style kwarg -> ``QAOA.from_problem(...)``.
* ``BackendCapabilities.qubit_capacity`` -> ``max_qubits``.
* ``LocalProvider.available_backends()`` -> ``backends()``.
* ``BenchmarkResult.quantum_volume``/``clops`` -> ``value``;
  ``Operator.CNOT`` not valid for GST (single-qubit only).
* Calling properties as methods (``executions()``, ``dim()``,
  ``total_shots()``, ``target_indices()``, ``keys()``, ``depth``) and
  methods as attributes.
* ``ParameterSweep`` omitting the swept parameter from the base circuit;
  sweep keys are strings, not ``Parameter`` objects.
* Missing imports in doc blocks (``sample_state``, ``QuantumCircuit``,
  ...).
* ``SamplingProblem(**data)`` on a serialized dict ->
  ``from_dict(data)``.
* ``ExperimentResult``/``ExecutionRecord``/``OptimizerResult``
  attribute-vs-method semantics fixed across 32 docs pages.

Docs pages repaired (48/48 pages green): algorithms (overview, grover,
phase-estimation, qaoa, vqe), analysis (aggregation, expectations, states,
statistics), concepts (algorithms, circuits, gradients, measurement,
parameters, problems, quantum-states), execution (backends, capabilities,
custom-backends, execution-core, providers, runtime), experiments
(experiments, parameter-sweeps, reproducibility, results),
getting-started (first-experiment, first-measurement, quickstart),
problems (sampling, search), troubleshooting, tutorials/benchmarks, plus
``docs/examples/index.rst`` rewritten against the real example tree.

Future API cleanup (backlog)
============================

The conformance review identified ``Operator.from_dict`` without a
corresponding ``Operator.to_dict``.

* **Identified during:** conformance review of the public API
  serialization contracts.
* **Scope:** this item is **not** part of MQ-11 (Quantum Execution Core),
  MQ-12 (Parameterized Circuits), MQ-13 (Analytical Gradients), or the
  documentation task that preserved this report.
* **Impact today:** no compatibility or API change is introduced now; the
  SDK surface is unchanged.
* **Next step:** evaluate this item during a future API cleanup phase,
  together with the related serialization asymmetry noted below
  (``StateVector`` and ``QuantumCircuit`` have no ``to_dict``).

It is deliberately recorded as a backlog item only: no code, fake TODO,
or test changes accompany it.

Known observations (not blockers)
=================================

* Serialization asymmetry: ``Operator`` has ``from_dict`` but no
  ``to_dict`` (see Future API cleanup above); ``StateVector`` and
  ``QuantumCircuit`` have no ``to_dict``.
* Coverage gate is impractical in its current form (no ``fail_under``
  configured) — consider ``--cov-report=term`` on a smoke subset or
  ``--no-cov`` in CI.

Release prerequisite reminder (AGENTS.md)
=========================================

No PyPI release without: push + green CI + ``v<version>`` tag + TestPyPI
check first.