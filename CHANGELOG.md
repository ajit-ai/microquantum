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
- `ExecutorResult.to_dict()`, `AnalysisResult.to_dict()`, `QuantumResult.to_dict()`
  and `to_json()` on all result types (uniform JSON-safe serialization).
- Uniform JSON-safe serialization (`to_dict()` / `to_json()`) on the remaining
  algorithm, optimizer, benchmark, QML, mitigation, and core result types:
  `VQEResult`, `VQDResult`, `AdaptResult`, `AmplitudeEstimationResult`,
  `BVResult`, `DJResult`, `GroverResult`, `TrotterResult`, `HHLResult`,
  `PhaseEstimationResult`, `QuantumWalkResult`, `ShorResult`,
  `CycleBenchmarkingResult`, `LayerFidelityResult`, `TwirlingResult`,
  `GSTResult`, `BenchmarkResult`, `RandomizedBenchmarkingResult`, `XEBResult`,
  `OptimizerResult`, `ClassifierResult`, `ExtrapolationResult`,
  `DynamicCircuitResult`, and `MeasurementResult`.
- Generic visualization helpers `plot_allocation`, `plot_scatter`, and
  `plot_distribution` alongside the pre-existing helpers.
- `Result.improved_over_baseline` property.
- `CONTRIBUTING.md` and this changelog.
- `[project.optional-dependencies]` extras: `gpu` (CuPy-backed array backend) and `all`.
- Ruff lint configuration and pytest-cov coverage reporting.

### Changed
- `pyproject.toml`: SPDX `license = "MIT"`, `keywords`, dynamic version read from
  `microquantum.__version__`, updated classifiers.
- `HardwareBackend.run()` now matches the unified backend signature
  (`circuit`, `shots`, `initial_state`, `seed`).
- The `Result` contract is now generic SDK infrastructure: canonical fields
  `solution` and `baseline` replace `decision` and `classical_baseline`.
  `to_json()` now returns a JSON **string** (was a dict) to match the SDK-wide
  `to_dict()`/`to_json()` convention.
- Complex-valued fields are now encoded element-wise as
  `{"real": ..., "imag": ...}` everywhere, including backend `statevector` /
  `density_matrix` output (previously `{"real": [...], "imag": [...]}`).
- Visualization helpers `plot_portfolio_allocation`, `plot_risk_return_scatter`,
  and `plot_var_distribution` were renamed to the generic `plot_allocation`,
  `plot_scatter`, and `plot_distribution`; the old names remain as deprecated
  aliases.
- Analytics and QUBO docstrings no longer reference industry-specific business
  terminology.

### Deprecated
- `Result(decision=...)`, `Result(classical_baseline=...)`,
  `Result.decision`, `Result.classical_baseline`, and
  `Result.improved_over_classical` are deprecated aliases for the canonical
  `solution`, `baseline`, and `improved_over_baseline` names. They emit
  `DeprecationWarning` and will be removed in a future major release.
- Plot helpers `plot_portfolio_allocation`, `plot_risk_return_scatter`, and
  `plot_var_distribution` are deprecated aliases for the generic names.

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