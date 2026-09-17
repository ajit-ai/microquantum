"""Package ecosystem conformance (Phase 118).

Makes the package organization a tested contract:

- every public subpackage is importable, exposes a curated ``__all__`` and
  never leaks private names or star-imports;
- the top-level ``microquantum`` gateway stays fully resolvable;
- re-exports are identity-as-canonical (no duplicate implementations); and
- the standard library is the single canonical home for its utilities.
"""

import importlib
import pkgutil
from pathlib import Path

import pytest

import microquantum
import microquantum.stdlib

PUBLIC_SUBPACKAGES = [
    "adapters",
    "algorithms",
    "analysis",
    "analytics",
    "backends",
    "benchmarks",
    "chemistry",
    "core",
    "experiments",
    "ir",
    "mitigation",
    "optimization",
    "optimizers",
    "problems",
    "providers",
    "qec",
    "qml",
    "runtime",
    "stdlib",
]

STDLIB_NAMES = [
    # bits (MSB-first conversions + Hamming)
    "int_to_bits",
    "bits_to_int",
    "hamming_weight",
    "hamming_distance",
    # numbers (mod-2*pi rotation arithmetic)
    "mod_2pi",
    "wrap_angle",
    "is_angle_close",
    "is_identity_angle",
    # states (factories on StateVector)
    "basis_state",
    "uniform_superposition",
    "bell_state",
    "ghz_state",
    "w_state",
]

_STDLIB_LEAF = {
    "int_to_bits": "bits",
    "bits_to_int": "bits",
    "hamming_weight": "bits",
    "hamming_distance": "bits",
    "mod_2pi": "numbers",
    "wrap_angle": "numbers",
    "is_angle_close": "numbers",
    "is_identity_angle": "numbers",
    "basis_state": "states",
    "uniform_superposition": "states",
    "bell_state": "states",
    "ghz_state": "states",
    "w_state": "states",
}


def _subpackage_module(name):
    return importlib.import_module(f"microquantum.{name}")


def _all_submodules():
    return {
        info.name
        for info in pkgutil.walk_packages(
            microquantum.__path__, prefix=f"{microquantum.__name__}."
        )
    }


class Test_subpackage_surface:
    @pytest.mark.parametrize("name", PUBLIC_SUBPACKAGES)
    def test_subpackage_importable_and_curated(self, name):
        module = _subpackage_module(name)
        assert hasattr(module, "__all__")
        assert len(module.__all__) > 0
        for exposed in module.__all__:
            assert not exposed.startswith("_")
            assert hasattr(module, exposed)

    @pytest.mark.parametrize("name", PUBLIC_SUBPACKAGES)
    def test_no_star_imports(self, name):
        module = _subpackage_module(name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "import *" not in source

    def test_every_top_level_subpackage_discovered(self):
        known = {f"microquantum.{name}" for name in PUBLIC_SUBPACKAGES}
        discovered = _all_submodules()
        assert known <= discovered

    def test_stdlib_trio_present(self):
        for submodule in ("bits", "numbers", "states"):
            assert f"microquantum.stdlib.{submodule}" in _all_submodules()


class Test_top_level_gateway:
    def test_all_exported_names_resolve(self):
        for name in microquantum.__all__:
            assert hasattr(microquantum, name)

    def test_no_private_names_exported(self):
        for name in microquantum.__all__:
            assert not name.startswith("_")

    def test_only_private_boundary_modules(self):
        # `_json` (JSON helpers) and `_cli` (developer CLI) are the entire
        # private module surface; everything else is public.
        allowed = {"_json", "_cli"}
        private = [
            m
            for m in dir(microquantum)
            if m.startswith("_")
            and not (m.startswith("__") and m.endswith("__"))
            and m not in allowed
        ]
        assert private == []


class Test_canonical_identity:
    """Top-level aliases must be the same objects as their canonical homes."""

    def test_core(self):
        canonical = importlib.import_module("microquantum.core.state")
        assert microquantum.StateVector is canonical.StateVector
        canonical = importlib.import_module("microquantum.core.density_matrix")
        assert microquantum.DensityMatrix is canonical.DensityMatrix
        canonical = importlib.import_module("microquantum.core.circuit")
        assert microquantum.QuantumCircuit is canonical.QuantumCircuit
        canonical = importlib.import_module("microquantum.core.operators")
        assert microquantum.Operator is canonical.Operator

    def test_compiler(self):
        canonical = importlib.import_module("microquantum.ir.compiler")
        assert microquantum.Compiler is canonical.Compiler
        canonical = importlib.import_module("microquantum.ir.builder")
        assert microquantum.to_ir is canonical.to_ir

    def test_runtime(self):
        canonical = importlib.import_module("microquantum.runtime.plan")
        assert microquantum.ExecutionPlan is canonical.ExecutionPlan
        canonical = importlib.import_module("microquantum.runtime.runtime")
        assert microquantum.ExecutionRuntime is canonical.ExecutionRuntime
        assert microquantum.execute is _subpackage_module("runtime").execute

    def test_backends(self):
        canonical = importlib.import_module("microquantum.backends.statevector")
        assert microquantum.StatevectorBackend is canonical.StatevectorBackend
        canonical = importlib.import_module("microquantum.backends.mock")
        assert microquantum.MockBackend is canonical.MockBackend
        canonical = importlib.import_module("microquantum.backends.executor")
        assert microquantum.Executor is canonical.Executor
        canonical = importlib.import_module("microquantum.backends.base")
        assert microquantum.Backend is canonical.Backend

    def test_stdlib_alias_identity(self):
        bits_module = _subpackage_module("stdlib.bits")
        numbers_module = _subpackage_module("stdlib.numbers")
        states_module = _subpackage_module("stdlib.states")
        assert microquantum.int_to_bits is bits_module.int_to_bits
        assert microquantum.hamming_weight is bits_module.hamming_weight
        assert microquantum.mod_2pi is numbers_module.mod_2pi
        assert microquantum.wrap_angle is numbers_module.wrap_angle
        assert microquantum.bell_state is states_module.bell_state
        assert microquantum.w_state is states_module.w_state


class Test_stdlib_is_canonical:
    @pytest.mark.parametrize("name", STDLIB_NAMES)
    def test_stdlib_symbol_not_duplicated_elsewhere(self, name):
        contributors: list[str] = []
        for submodule in _all_submodules():
            if submodule in ("microquantum", "microquantum._json"):
                continue
            module = importlib.import_module(submodule)
            if name in getattr(module, "__all__", ()):
                contributors.append(submodule)
        assert sorted(contributors) == sorted(
            [f"microquantum.stdlib.{_STDLIB_LEAF[name]}", "microquantum.stdlib"]
        )

    def test_stdlib_modules_are_top_level_subpackages(self):
        for submodule in ("bits", "numbers", "states"):
            parent = _subpackage_module("stdlib")
            child = importlib.import_module(f"microquantum.stdlib.{submodule}")
            assert getattr(parent, submodule) is child


class Test_compatibility_imports:
    """Long-standing import paths keep resolving unchanged."""

    def test_deep_module_imports(self):
        import microquantum.backends.local  # noqa: F401
        import microquantum.core.circuit  # noqa: F401
        import microquantum.core.operators  # noqa: F401
        import microquantum.core.state  # noqa: F401
        import microquantum.ir.compiler  # noqa: F401
        import microquantum.runtime.runtime  # noqa: F401
        import microquantum.stdlib.numbers  # noqa: F401

    def test_from_imports(self):
        from microquantum import (  # noqa: F401
            Compiler,
            ExecutionPlan,
            MockBackend,
            Operator,
            QuantumCircuit,
            StateVector,
        )
        from microquantum.core.circuit import QuantumCircuit as DeepCircuit
        from microquantum.stdlib.states import ghz_state  # noqa: F401

        assert DeepCircuit is microquantum.QuantumCircuit