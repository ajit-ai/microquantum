# Changelog

All notable changes to the open-source `microquantum` SDK are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Cross-platform support** — the GA (1.0.0) release is documented for
  Windows, Linux, macOS and BSD:
  - CI matrix expanded from Ubuntu-only to GitHub-hosted Ubuntu Linux,
    Windows and macOS across Python 3.10 - 3.13 (plus an import check and CLI
    smoke step in each test job).
  - `pyproject.toml` classifiers now declare `MacOS`, `Microsoft :: Windows`,
    `POSIX :: BSD` and `POSIX :: Linux`.
  - New `docs/releases/platform-support.rst` restates the designed-vs-verified
    distinction; `installation.rst` gains per-platform Linux / Windows / macOS
    / BSD instructions; README, `compatibility.rst` and `ga.rst` updated to
    the accurate cross-platform support statement.  BSD is reported as
    supported by design but not CI-verified.

### Fixed
- **Wheel-vs-source conformance test on POSIX** — the conformance suite in
  `tests/conformance/test_source_vs_wheel_install.py` only used the Windows
  layout (`venv\Scripts\python.exe`, `venv\Lib\site-packages`); it now locates
  the created venv's interpreter and site-packages per platform, so the
  wheel/source equivalence checks run on Linux and macOS as well as Windows.

## [1.0.0] - 2026-09-17

MicroQuantum **1.0.0** is the General Availability (GA) release.  It wraps up
the final Phase 120 roadmap phase; the roadmap is complete and future work, if
any, is post-GA.  Every feature and change below ships in this release.

### Added
- **Phase 120: Final GA / Release Readiness** — final planned roadmap phase:
  version bumped to `1.0.0` across `microquantum/__init__.py` and
  `docs/conf.py`; packaging classifier updated to "Development Status :: 5 -
  Production/Stable"; new `docs/releases/ga.rst` release notes with the final
  roadmap statement (Phase 120 COMPLETE; Phase 121+ NOT CREATED); the 0.4.x
  Developer Preview page archived; README, getting-started and compatibility
  docs refreshed to the GA status and the post-1.0 stability contract; the
  "Unreleased" section finalized as this release entry.
- **Phase 119: Runtime & Tooling** — makes the compiler/runtime stack directly
  usable as a developer-facing system without redesigning the Phase-118
  ecosystem:
  - **Runtime configuration** — new `RuntimeConfig` (frozen, JSON-safe) holding
    the backend (instance or registered name), `BackendRegistry`, default
    target, history cap and default optimization level; `ExecutionRuntime`
    accepts `config=...` (legacy kwargs still override), exposes `.config`, and
    `configure(**overrides)` returns a fresh runtime with merged settings.
  - **Stage-tagged errors** — new `microquantum.runtime.errors` hierarchy:
    `ExecutionError(ValueError)` with `PlanningError`, `CompilationError`,
    `RuntimeDispatchError`, `BackendExecutionError`, raised at the exact
    pipeline stage (messages unchanged).  All subclass `ValueError`, so
    existing `except ValueError` handlers keep working; `TypeError` plan-shape
    errors are deliberately not wrapped.
  - **Runtime introspection** — `runtime_info(runtime=None, registry=None)`
    returns a JSON-safe `RuntimeInfo` snapshot (SDK/Python/NumPy versions,
    sorted `ExecutionStrategy` values, live per-backend capability summaries
    from the real registries, default backend and optimization level).
  - **Developer CLI** — new `microquantum` console script (also
    `python -m microquantum`) implemented dependency-free in the private
    `microquantum._cli`: `version`/`--version`, `info`, `backends [--json]`
    and `run FILE [--shots N --seed N --backend NAME --optimization-level N]`
    for OpenQASM 2.0 files.  Runs through the canonical public Python APIs,
    prints one-line errors to stderr with a non-zero exit (no traceback noise;
    `--debug` restores it).
  - Docs: `execution/runtime.rst` extended with configuration, errors and
    introspection; new `execution/cli.rst` and curated `api/runtime.rst`;
    package-ecosystem private-boundary rule updated to `_json` + `_cli`.
    Tests: `test_runtime_config.py`, `test_runtime_info.py`,
    `test_runtime_errors.py`, `test_cli.py`.
- **Phase 118: Package Ecosystem** — establishes and locks the public package
  organization around the existing core, compiler, runtime, backends and the
  Phase-117 `microquantum.stdlib`.  The nine conceptual boundaries (core /
  circuit / gates / states / measurement / compiler / runtime / backends /
  stdlib) map onto the existing modules without file moves or renamed imports:
  a new `docs/developer-guide/package-ecosystem.rst` canonical import map, an
  updated architecture source tree and README project structure, and an
  expanded top-level package docstring.  Conformance tests
  (`tests/test_package_ecosystem.py`) lock the surface: every public
  subpackage imports with a curated `__all__` (no star imports, no private
  leaks), the top-level gateway stays fully resolvable, re-exports are
  identity-canonical (no duplicate implementations), stdlib utilities live
  only in `microquantum.stdlib.*`, and packaging discovery (setuptools
  `find` under `src/`) reaches every subpackage including `stdlib`.
- **Phase 117: System Standard Library** — new `microquantum.stdlib`
  package (also re-exported from the top-level `microquantum` module):
  `stdlib.bits` (MSB-first `int`↔bitstring conversions and Hamming
  weight/distance, matching measurement-outcome conventions),
  `stdlib.numbers` (`mod_2pi`, `wrap_angle`, `is_angle_close`,
  `is_identity_angle` — modulo-`2*pi` rotation arithmetic), and
  `stdlib.states` (factories for `basis_state`, `uniform_superposition`,
  `bell_state`, `ghz_state`, `w_state` on `StateVector`, honouring the
  dense-allocation budget).  Lays the stable, dependency-light foundation
  layer for user programs, the runtime and the compiler.  Adds a curated
  API docs page, example 33, and a dedicated test module.
- **MQ-17: Strict mypy conformance** — Full SDK now passes `mypy --strict` with zero errors. Removed per-module error-code overrides for `circuit.py` and `gradient.py`. Added `_narrow_parameterized()`/`_narrow_concrete()` type-safe helpers. New conformance test gates strict mypy in CI.
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