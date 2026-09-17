Changelog
=========

Every notable change to the open-source ``microquantum`` SDK is tracked in
``CHANGELOG.md`` (Keep a Changelog format, Semantic Versioning) in the
repository root — the single source of truth:

https://github.com/ajit-ai/microquantum/blob/main/CHANGELOG.md

Release history
---------------

1.0.0 (General Availability)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GA release completes the final Phase 120 roadmap phase — the roadmap is
complete:

* System standard library (``microquantum.stdlib``: ``bits``, ``numbers``,
  ``states``).
* Package ecosystem locked through the canonical package / import map.
* Runtime & tooling: ``RuntimeConfig``, stage-tagged runtime errors,
  ``runtime_info`` introspection and the ``microquantum`` CLI
  (``version`` / ``info`` / ``backends`` / ``run``).
* Final GA readiness: version 1.0.0, production/stable packaging
  classifiers, ``docs/releases/ga.rst`` release notes and the post-1.0
  stability contract.

The 0.4.x Developer Preview series that preceded this release is archived on
the :ref:`developer-preview` page.

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