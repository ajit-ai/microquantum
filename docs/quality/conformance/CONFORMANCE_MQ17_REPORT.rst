MQ-17 Strict Mypy Conformance Report
=====================================

.. contents:: Table of Contents
   :depth: 2

Overview
--------
MQ-17 tightened ``[tool.mypy]`` to strict mode across the entire public SDK.
All per-module error-code overrides have been removed.  The remaining in-source
``# type: ignore`` comments were eliminated by fixing the underlying types, so
``mypy --strict`` now passes with zero errors and zero suppressions.

Scope
-----
- All source files under ``src/microquantum/`` (127 files).
- Removed ``[[tool.mypy.overrides]]`` for ``microquantum.core.circuit`` and
  ``microquantum.core.gradient`` which previously suppressed ``assignment``,
  ``arg-type``, ``misc``, ``union-attr`` and ``index`` error codes.
- Removed global ``ignore_missing_imports = true``.
- Added ``strict = true`` in ``[tool.mypy]``.
- Added scoped overrides only for genuinely-optional external packages
  (``cupy``, ``scipy``).
- Removed all 46 remaining ``# type: ignore`` comments inside the SDK by
  fixing the underlying strict-mode issues (narrowing helpers, asserted
  invariants, generic key typings and explicit ``cast`` at validated API
  boundaries).  The conformance suite permanently asserts this invariant.

Changes
-------
**pyproject.toml:**
  - ``[tool.mypy]``: enabled ``strict = true``, removed
    ``ignore_missing_imports``.
  - Removed module overrides for ``microquantum.core.circuit`` and
    ``microquantum.core.gradient``.
  - Added ``[[tool.mypy.overrides]]`` for ``cupy`` and ``scipy`` modules.

**core/circuit.py:**
  - Added ``_narrow_parameterized()`` and ``_narrow_concrete()`` helper
    functions for safe type narrowing of ``_GateInstruction`` union.
  - Updated all 9 gate-instruction loop sites to use the new narrowing
    helpers.
  - Changed ``from_ir()`` signature from ``object`` to ``IRCircuit``.
  - Added explicit ``complex`` angle rejection (Type-safe path).

**core/gradient.py:**
  - Updated ``_find_occurrences()`` and ``_build_shifted_circuit()`` to
    use ``_narrow_parameterized()``.

**core/dynamic.py:**
  - Widened ``_ops`` type from bare ``tuple`` to ``tuple[Any, ...]``.
  - Added explicit ``complex`` angle rejection in ``rx``/``ry``/``rz``.

**ir/builder.py:**
  - Typed ``_OP_FACTORIES`` as ``dict[str, Callable[..., Operator]]``
    instead of ``dict[str, object]``.

**backends/executor.py:**
  - Used ``_narrow_concrete()`` for loop narrowing.
  - Added ``assert self._noise_model is not None`` guard.

**Type-ignore elimination (18 files):**
  - Reused the ``_narrow_concrete()`` helper at every remaining gate loop:
    ``mitigation/zne.py``, ``qml/kernels.py``, ``qml/classifier.py``,
    ``algorithms/amplitude_estimation.py``.
  - Bound ``dict[Parameter, float]`` mappings explicitly via ``cast`` at the
    validated ``bind_parameters``/``gradient`` boundaries:
    ``algorithms/vqe.py``, ``algorithms/vqd.py``, ``algorithms/adapt_vqe.py``.
  - ``algorithms/qaoa.py``: narrowed the cost Hamiltonian with ``isinstance``
    before dispatching to the generic/legacy ansatz builders.
  - ``algorithms/base.py``: ``cast`` for the ``json_safe`` return.
  - ``algorithms/quantum_walk.py``, ``core/measurement.py``: used
    ``dict.__getitem__`` as the deterministic ``max`` key.
  - ``core/tensor.py``: filtered the homogenous argument tuple by
    ``isinstance`` before dispatch.
  - ``core/visualization.py``: unpacked the optional column entry behind an
    explicit ``is not None`` check.
  - ``analysis/state.py``: ``assert`` the density/statevector invariant that
    the constructor already enforces.
  - ``backends/array_backend.py``: ``cast`` the decomposed-tuple returns.
  - ``backends/tensor_network.py``: ``assert`` the non-leaf children
    invariant in ``_lca``/``_contract``/``_leaf_order``.
  - Added missing generic type arguments and return annotations as required.

Verification
------------
- ``uv run mypy src/microquantum/``: **Success: no issues found (127 files)**
- ``uv run ruff check src/microquantum/``: **All checks passed**
- ``uv run pytest tests/ -q``: **2514 passed** (2511 prior + 3 new
  conformance tests)