# Changelog

All notable changes to the open-source `microquantum` SDK are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Analytical gradients via the parameter-shift rule (MQ-13): exact
  `parameter_shift_gradient` for a single parameter and `gradient` for the
  full vector, with name-based matching, chain-rule scaling for
  `ParameterExpression` angles and per-occurrence summation for parameters
  appearing in several gates. Observables may be `Operator`, `PauliString`
  or `PauliSum` (no dense matrices), optionally placed on a subset of
  qubits via `targets=`. Evaluation defaults to the built-in state-vector
  engine or routes through the execution core with `backend=` / `seed=` /
  `shots=` (exact and reproducible). New `Parameter.gradient()` /
  `ParameterExpression.gradient()` return symbolic partial derivatives.
  Results are plain `{Parameter: float}` dicts that plug directly into
  gradient-mode optimizers as `gradient_fn`. New examples 15-16, a
  `gradients` concept page, and strict validation (Mapping-only
  `param_values`, shifts that are not integer multiples of pi).
- Parameterized circuits & parameter execution (MQ-12): first-class
  `Parameter` support with a deterministic, read-only `circuit.parameters`
  tuple (name-sorted, name-identity dedup), strict `bind_parameters`
  validation (unknown / non-numeric / complex / ambiguous bindings fail
  fast instead of being silently ignored), non-destructive partial binding,
  parameter-preserving JSON serialization for `Parameter` and
  `ParameterExpression` gates, an optional `parameter_values=` binding map
  on `Backend.run` / `submit_circuit` (delegating to the canonical binder),
  and honest OpenQASM 2.0 export that raises rather than dropping
  parameterized gates. New examples 12-14 and a rewritten parameters
  concept page.
- Quantum execution and measurement core (MQ-11): explicit circuit measurement
  annotations via `QuantumCircuit.measure` / `measure_all` (with an ordered,
  serializable `measurements` property), `BackendResult.get_counts()` and
  `BackendResult.state` accessors, deterministic per-shot `samples` on
  measurement results, and `Backend.run` restricting counts to measured qubits.
  Measurement annotations persist through JSON, OpenQASM 2.0, IR and
  parameter binding/circuit concatenation.

### Changed
- `QuantumCircuit.parameters` now returns a name-sorted tuple (was a set).
  `bind_parameters` now raises `ValueError` for keys that match no circuit
  parameter, `TypeError` for non-numeric values, and `ValueError` for
  complex or duplicate bindings; mixed unknown+known mappings fail instead
  of silently ignoring the unknown entries.
- `Parameter` gained `__mul__` (`theta * 2`), mirroring the existing
  `2 * theta` support.
- `append_parameterized` validates its gate type and parameter type up
  front.

### Fixed
- `parameter_shift_gradient` / `gradient` now apply the chain-rule
  coefficient of `ParameterExpression` angles (a gate at `2 * theta`
  contributes `2` times its shift difference) and match parameters by name
  rather than object identity, consistent with the MQ-12 parameter model.
- `sample_state` / `measure_qubits` now reject invalid shot counts (`< 1`).
- `StatevectorBackend.run_circuit` records the execution `seed` and raw
  `samples` on the returned `BackendResult`.

## [0.4.1] - 2026-09-12

### Fixed
- Batch sweep in `Runtime.execute_records` (and the Experiment layer) no longer
  clobbers each plan's own `parameter_bindings` with an empty `{}` when no
  caller sweep is provided. The sweep sentinel is now `None` (`combos_options =
  [None]`), so per-plan bindings are preserved verbatim. Resolves `KeyError
  'theta'`, zero-success sweep results, and dropped partial failures across the
  batch/sweep/experiment test surface.

## [0.4.0] - 2026-09-10

### Added
- Developer Preview release with complete packaging & documentation pipeline.
- Execution records / experiments layer: `ExecutionRecord`, `ExecutionFailure`,
  `ExecutionStatus`, `ExecutionPlan` fingerprinting (`execution_fingerprint`,
  `reproducibility_metadata`), `ParameterSweep`, `Experiment`, `ExperimentResult`.
- Analysis layer: `SamplingAnalysis`, `ExpectationAnalysis`, `StateAnalysis`,
  `ResultAggregator` and the `statistics` helpers (mean/variance/std/SE/CI).
- `BackendResult` payload expansion: raw `samples`, labeled `expectations`,
  `eigenvalues`, JSON-safe `native` payload.
- `Backend.capabilities` / `BackendCapabilities`, `TargetClass`,
  `supports(plan)`, `validate(plan)`, `BackendRegistry`, `Provider` /
  `LocalProvider`, `BackendAdapter` boundary.
- `ExecutionPlan`, `ExecutionRuntime`, `ExecutionStrategy`,
  `ExecutionTrace`, `execute_batch` / `submit_batch` / `execute_records` /
  `run_parameter_sweep` / `run_hybrid` / `run_experiment`.
- `docs` optional dependency extra (`sphinx`, `sphinx-autoapi`, `furo`).
- Sphinx documentation (Furo theme) with a full information architecture:
  getting-started, concepts, algorithms, problems, execution, experiments,
  analysis, examples, auto-generated API reference, developer guide, releases,
  FAQ and troubleshooting.
- GitHub Pages deployment workflow publishing the built HTML documentation
  (`.github/workflows/docs.yml`, run on pushes to `main`).
- PyPI release workflows using Trusted Publishing: `python-publish.yml`
  (publishes to PyPI on release publication) and `testpypi-publish.yml`
  (publishes to TestPyPI on release publication or manual dispatch).
- `microquantum 0.4.0` released to PyPI and TestPyPI (wheel + sdist).
- Consolidated CI: single `ci.yml` with a multi-Python test matrix, lint +
  warning-free docs build; the duplicate `python-app.yml` workflow was removed.

### Changed
- Version bumped to `0.4.0` across `microquantum/__init__.py`,
  `docs/conf.py` and packaging metadata.
- `docs/conf.py` rewritten: Furo theme, version `0.4.0`, autoapi root `api`
  (generated reference kept on disk and git-ignored), auto-api regenerated
  from source; the build is warning-free with `--keep-going` in CI.  `nitpicky`
  stays off because autoapi docstrings reference package-surface names
  (e.g. `microquantum.BackendResult`) that are valid Python but not resolvable
  cross-reference targets.
- Curated API pages (`docs/api/*.rst`) promoted to lightweight subsystem
  pointers that link into the single source-generated autoapi reference.
- `pyproject.toml` URLs now point at the public documentation site and
  changelog; `include-package-data = false`; dev dependency group uses
  `furo` in place of `sphinx-rtd-theme`.

## [0.3.0] - 2026-09-10

### Added
- Initial public open-source release of the `microquantum` SDK (MIT).
- Core engine, 23+ algorithms, simulators, tensor-network backends, real-hardware
  providers, QML, QEC, benchmarks, mitigation, chemistry, QUBO toolchain, and the
  standardized `Result` contract.

### Changed
- Repository split: the open SDK is published to PyPI as `microquantum`;
  industry-specific vertical solvers are developed in a separate repository.