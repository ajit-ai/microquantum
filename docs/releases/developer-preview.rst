.. _developer-preview:

Developer Preview
=================

MicroQuantum **v0.4.0 Developer Preview** is the first release that is
installable from PyPI, fully documented on GitHub Pages, and covered by CI.

Status
------

- **Version**: 0.4.0 (``develop`` -> ``main``, tag ``v0.4.0``).
- **Runtime**: Pure Python + NumPy.  Python 3.10 - 3.13.
- **Documentation**: https://ajit-ai.github.io/microquantum/ (auto-generated
  API + hand-written guides).
- **Testing**: full test suite, ruff, mypy, warnings-as-errors docs build, and
  every ``examples/`` demo executed against the installed package in CI.

What this preview offers
------------------------

* A **complete local stack** — circuits, operators, states, measurement,
  parameters, gradients, IR & compilation, device/target contracts.
* **Problems** — Sampling / Optimization (QUBO/Ising) / Hamiltonian /
  Eigenvalue / Search.
* **Algorithms** — VQE, QAOA, Grover, phase estimation, QFT, HHL,
  Hamiltonian simulation, quantum walks, Shor, oracles, and more.
* **Execution** — declarative :class:`~microquantum.ExecutionPlan`,
  :class:`~microquantum.ExecutionRuntime`, batching, parameter sweeps, hybrid
  orchestration.
* **Backends** — statevector / density-matrix / MPS / tree-tensor-network
  simulators, a deterministic ``MockBackend``, capabilities and provider
  discovery.
* **Experiments & analysis** — execution records, sweeps, experiments,
  reproducibility fingerprints and the four analysers.
* **Interchange** — JSON serialization everywhere; OpenQASM import/export.

What is deliberately out of scope
---------------------------------

* Real quantum hardware execution / cloud platform.
* Enterprise database-backed experiment services, dashboards, CRUD APIs.
* Industry-specific vertical solvers (published separately by
  Microquantum-company on top of this SDK — never imported by it).

Limitations of the preview
--------------------------

* API may change before 1.0; ``0.x`` versioning means no stability guarantee
  across minor versions while the public surface settles.
* Simulator-only: density-matrix and statevector memory grow as ``2**n``.
* Hardware adapters exist as a boundary but no vendor is bundled.

Release checklist status
------------------------

* [x] Version bump (0.4.0) consistent across ``__init__.py`` /
  ``docs/conf.py`` / ``pyproject.toml``.
* [x] Packaging metadata (description, URLs, extras, license).
* [x] Sphinx docs (Furo) build warnings-as-errors.
* [x] CI: tests / lint / type check / docs build.
* [x] GitHub Pages workflow + action to publish ``docs/_build/html``.
* [x] Local: ``uv build`` -> wheel + sdist; fresh-venv install smoke;
  examples run against the installed package.
* [ ] PyPI ``microquantum`` 0.4.0 and TestPyPI pre-check (needs credentials).
* [ ] GitHub release ``v0.4.0`` (via ``gh``/web UI — no CLI available here).