# Changelog

All notable changes to the open-source `microquantum` SDK are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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