Changelog
=========

Every notable change to the open-source ``microquantum`` SDK is tracked in
``CHANGELOG.md`` (Keep a Changelog format, Semantic Versioning) in the
repository root — the single source of truth:

https://github.com/ajit-ai/microquantum/blob/main/CHANGELOG.md

Release history
---------------

0.4.0 (Developer Preview)
~~~~~~~~~~~~~~~~~~~~~~~~~

The first release that is installable from PyPI with full documentation and
CI coverage:

* Execution records / experiments layer and the analysis layer
  (SamplingAnalysis, ExpectationAnalysis, StateAnalysis, ResultAggregator).
* Backend capabilities, registry, providers and the BackendAdapter boundary.
* ExecutionPlan / ExecutionRuntime with batch, sweep and hybrid orchestration.
* Sphinx documentation (Furo) with the complete information architecture,
  GitHub Pages deployment workflow, packaging polish and version bump to 0.4.0.

0.3.0
~~~~~

Initial public open-source release of the SDK (MIT):

* Core engine, algorithms, simulators, tensor-network backends, real-hardware
  providers, QML, QEC, benchmarks, mitigation, chemistry, QUBO toolchain and
  the standardized ``Result`` contract.
* Repository split: the open SDK publishes to PyPI as ``microquantum``;
  vertical solvers live in a separate repository.