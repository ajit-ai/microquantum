.. _ga:

General Availability (v1.0.0)
=============================

MicroQuantum **1.0.0** is the General Availability (GA) release of the
open-source ``microquantum`` SDK.  The final Phase 120 roadmap work described
below is complete, and the public surface documented in
:doc:`/developer-guide/package-ecosystem` is now stable.

Status
------

- **Version**: 1.0.0.
- **Runtime**: Pure Python + NumPy.  Python 3.10 - 3.13.
- **Platforms**: designed for Windows, Linux, macOS and BSD; CI-verified on
  GitHub-hosted Ubuntu Linux, Windows and macOS.  See
  :doc:`/releases/platform-support` for the per-platform status.
- **Dependency**: NumPy is the only hard runtime dependency.
- **Documentation**: https://ajit-ai.github.io/microquantum/
- **License**: MIT.
- **CLI**: a ``microquantum`` console script (``version``, ``info``,
  ``backends [--json]``, ``run FILE.qasm``) implemented on top of the same
  canonical public Python APIs used by programs.

What GA ships
-------------

* **Core engine** — quantum circuits, operators, Pauli algebra,
  :class:`~microquantum.StateVector` / :class:`~microquantum.DensityMatrix`,
  measurement, gradients, registers, serialization, a transpiler and a
  :class:`~microquantum.PassManager` pipeline.
* **Problems** — :class:`~microquantum.SamplingProblem`,
  :class:`~microquantum.OptimizationProblem`,
  :class:`~microquantum.HamiltonianProblem`,
  :class:`~microquantum.EigenvalueProblem` and
  :class:`~microquantum.SearchProblem`.
* **Algorithms** — VQE, QAOA, Grover, phase estimation, QFT, HHL,
  Hamiltonian simulation, quantum walks, Shor and more.
* **Execution runtime** — declarative :class:`~microquantum.ExecutionPlan`
  orchestration, configurable through
  :class:`~microquantum.RuntimeConfig` with stage-tagged runtime errors and
  :func:`~microquantum.runtime_info` introspection.
* **Backends & providers** — statevector, density-matrix, MPS and
  tree-tensor-network simulators, a deterministic
  :class:`~microquantum.MockBackend`, capability-driven contracts and a
  provider discovery boundary.
* **Experiments & analysis** — execution records, parameter sweeps,
  experiments, reproducibility fingerprints and the four analysers.
* **System standard library** — ``microquantum.stdlib`` (``bits``,
  ``numbers`` and ``states`` factories).
* **Interchange** — JSON serialization everywhere; OpenQASM 2.x
  import/export; the ``microquantum`` CLI for developer workflows.

Stability
---------

Starting with 1.0.0 the public API is stable under Semantic Versioning:

* Backward-compatible additions land in minor releases; breaking changes in
  major releases.
* Serialized result schemas are versioned; new fields are additive.

Known limitations
-----------------

* Simulator-only: no bundled vendor hardware adapters.  Statevector /
  density-matrix memory grows as ``2**n``; use the MPS / tree-tensor-network
  backends for shallow, wide circuits.
* No hardware execution, cloud platform, enterprise database-backed
  experiment service, or analytics dashboard — the architecture contains the
  boundaries where those would plug in later.

Roadmap status
--------------

Phase 120 is the final planned MicroQuantum roadmap phase.  The roadmap is
complete at 1.0.0:

* **Phase 117** — System Standard Library (``microquantum.stdlib``).
* **Phase 118** — Package Ecosystem (canonical package / import map).
* **Phase 119** — Runtime & Tooling (``RuntimeConfig``, stage-tagged errors,
  ``runtime_info``, the ``microquantum`` CLI).
* **Phase 120** — Final GA / Release Readiness (this release).

MicroQuantum roadmap: COMPLETE.  Phase 121+: NOT CREATED.  Future
enhancements, if any, are post-GA release work and are not additional roadmap
phases.

Release checklist status
------------------------

* [x] Version bump (1.0.0) consistent across
  ``src/microquantum/__init__.py`` / ``docs/conf.py`` / the wheel metadata.
* [x] Packaging metadata (description, URLs, extras, license, classifiers).
* [x] Sphinx docs (Furo) build warnings-as-errors.
* [x] CI: tests / lint / type check / docs build.
* [x] GitHub Pages workflow publishing ``docs/_build/html``.
* [x] Local: ``uv build`` -> wheel + sdist; fresh-venv install smoke (CLI and
  documented public imports).
* [ ] PyPI ``microquantum`` 1.0.0 and TestPyPI pre-check (needs credentials;
  outside this repository session).
* [ ] GitHub release ``v1.0.0`` (via ``gh``/web UI — outside this repository
  session).

The 0.4.x Developer Preview series that preceded this release is recorded on
the :ref:`developer-preview` page.