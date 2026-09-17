Package Ecosystem
=================

MicroQuantum is distributed as a single flat Python package — ``pip install
microquantum`` — whose public surface is organized into stable, layered
subpackages.  This page is the canonical map: it tells users where each piece
of the SDK lives and how future extensions join the package without creating
duplicate APIs.

Import policy
-------------

* The top-level ``microquantum`` module (`src/microquantum/__init__.py`) is the
  **friendly gateway**.  Everything you need for daily work is available as
  ``from microquantum import ...``.
* Every top-level name is re-exported from exactly **one canonical location**.
  ``from microquantum import StateVector`` and ``from
  microquantum.core.state import StateVector`` return the *same object*
  (identity, not a copy).
* Subpackage imports stay valid and are the home of deeper, lower-level APIs
  (capability constants, strategy handlers, CSV/plot utilities) that are not
  part of the minimal top-level surface.
* Internal helpers are private: their names or modules start with ``_`` (for
  example ``microquantum._json`` — JSON helpers — and ``microquantum._cli`` —
  the ``microquantum`` console script's implementation).  They are not covered
  by the compatibility contract and must not be imported by user code.

Conceptual boundaries
---------------------

The ecosystem is described by nine boundaries.  Each maps onto existing
packages or modules — no duplicate implementations exist anywhere:

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Boundary
     - Canonical location
     - Contents
   * - **core**
     - ``microquantum.core``
     - The engine: ``core.circuit`` (``QuantumCircuit``), ``core.operators``
       (``Operator`` + the standard gate library), ``core.pauli``, parameters,
       registers, gradients, serialization, QASM, the transpiler / pass
       manager, device targets.
   * - **circuit**
     - ``microquantum.core.circuit``
     - The circuit model.  A circuit is a collection of registers,
       parameters and gates; ``QuantumCircuit.to_ir()`` lowers it into the
       compiler layer.
   * - **gates**
     - ``microquantum.core.operators``
     - ``Operator`` carries the matrix algebra and the standard gate
       constructors (H, X, CX, rotations, …).  In the compiler layer
       ``microquantum.ir.Gate`` is the *intermediate-representation node* — a
       different concept, not a second gate library.
   * - **states**
     - ``microquantum.core.state``,
       ``microquantum.core.density_matrix``
     - ``StateVector`` and ``DensityMatrix`` and their algebra.
       Convenience factories live in ``microquantum.stdlib.states``.
   * - **measurement**
     - ``microquantum.core.measurement``
     - Measurement sampling, projections and the measurement IR node
       (``microquantum.ir.Measurement``).
   * - **compiler**
     - ``microquantum.ir``
     - ``IRModule``/``IRCircuit``/``IRNode``, ``IRPass`` pipelines and the
       ``Compiler``, which compiles IR against a target device.  ``to_ir`` /
       ``from_ir`` convert between circuits and IR.  Advice: keep circuits
       and IR separate — the IR is the compiler 's private semantics, OpenQASM
       remains an optional interchange format.
   * - **runtime**
     - ``microquantum.runtime``
     - ``ExecutionPlan``, ``ExecutionRuntime``, strategies and the
       ``execute`` / ``execute_batch`` helpers that route work to backends.
   * - **backends**
     - ``microquantum.backends``
     - The ``Backend`` contract, capabilities and registry, the local
       simulators (statevector, density-matrix, MPS, tree-tensor network,
       mock) and the ``Executor``.  ``microquantum.providers`` is the optional
       vendor boundary for future hardware.
   * - **stdlib**
     - ``microquantum.stdlib``
     - The system standard library (Phase 117): ``stdlib.bits``, ``stdlib.numbers``
       and ``stdlib.states``.  This is the single canonical home for these
       utilities; nothing outside re-implements them.

One canonical implementation
----------------------------

Extension code — including future phases — must follow one rule: **add new
functionality in its own module and re-export it, never copy an existing
implementation into a second module.**  Concretely:

1. Put new code in the subpackage that owns the concern (a new gate goes into
   ``core.operators``, a new pass into ``ir.passes``, a new simulator into
   ``backends``, a shared utility into ``stdlib``).
2. Expose it through the subpackage's curated ``__all__`` and, when part of
   the minimal surface, re-export it from the top-level ``__init__.py``.
3. Contributor-facing boundaries (``adapters``, ``providers``, ``qml``,
   ``qec``, ``chemistry``, ``benchmarks``, ``mitigation``, ``analytics``)
   stay independent verticals: they consume the core through public APIs only.

Packaging
---------

The distribution metadata (`pyproject.toml`) uses setuptools package discovery
under ``src/``, so every ``microquantum`` subpackage — including
``microquantum.stdlib`` — is included in the sdist and wheel automatically.
The only hard runtime dependency is NumPy; everything else (docs, GPU
acceleration, optional integrations) is declared as an optional extra and
never imported by the SDK itself.

The package surface is locked by the ``tests/test_package_ecosystem.py``
conformance module: subpackage discovery, curated ``__all__`` sets, canonical
alias identity and stdlib exclusivity are all verified on every test run.