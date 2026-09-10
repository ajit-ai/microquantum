# Changelog

All notable changes to the open-source `microquantum` SDK are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `Backend.run(circuit, shots, initial_state, seed)` high-level execution API on
  all simulator backends; `run_circuit` remains the low-level contract.
- `BackendResult.to_dict()` and `ExecutorResult.to_dict()` for JSON-safe
  serialization of execution results.
- `CONTRIBUTING.md` and this changelog.
- `[project.optional-dependencies]` extras: `gpu` (CuPy-backed array backend) and `all`.
- Ruff lint configuration and pytest-cov coverage reporting.

### Changed
- `pyproject.toml`: SPDX `license = "MIT"`, `keywords`, dynamic version read from
  `microquantum.__version__`, updated classifiers.
- `HardwareBackend.run()` now matches the unified backend signature
  (`circuit`, `shots`, `initial_state`, `seed`).

### Fixed
- Docs no longer reference industry-specific domain adapters that live outside
  the open SDK; examples use neutral problem identifiers.

## [0.3.0] - 2026-09-10

### Added
- Initial public open-source release of the `microquantum` SDK (MIT).
- Core engine, 23+ algorithms, simulators, tensor-network backends, real-hardware
  providers, QML, QEC, benchmarks, mitigation, chemistry, QUBO toolchain, and the
  standardized `Result` contract.

### Changed
- Repository split: the open SDK is published to PyPI as `microquantum`;
  industry-specific vertical solvers are developed in a separate repository.