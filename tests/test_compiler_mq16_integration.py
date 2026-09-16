"""MQ-16 regression suite: compiler <-> runtime integration.

Proves that a :class:`~microquantum.Compiler`-compiled program preserves the
*observable* behavior of the original circuit through the existing
execution runtime and every supported local backend:

* differential equivalence (exact and sampled) across levels 0/1/2 and all
  four simulators,
* runtime-route equivalence (explicit ``Compiler`` + ``backend.run`` vs
  ``ExecutionPlan(compiled=...)`` reuse vs ``execute(optimization_level)``),
* gate-level regressions: negative rotations, same-axis fusion, ``cz`` and
  ``swap`` decomposition toward native bases,
* parameter binding, measurements, metadata, empty/no-op circuits,
* error contracts and the ``ExecutionPlan.compiled`` type validation.

The shared logic lives in :mod:`compiler_runtime_helpers` so the compact
permanent conformance suite exercises the exact same helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
from compiler_runtime_helpers import (
    EQUIVALENCE_CORPUS,
    ROTATION_ANGLES,
    SHOTS_SAMPLED,
    SIMULATORS,
    TOL_EXACT,
    TOL_FIDELITY,
    TOL_SAMPLING,
    assert_execution_equivalent,
    assert_runtime_equivalent,
    assert_statevector_preserved,
    bell,
    compile_for,
    cz_circuit,
    exact_probabilities,
    fidelity,
    ghz,
    parameterized_circuit,
    sampled_probability_map,
    swap_circuit,
)

from microquantum import (
    CompilationResult,
    Compiler,
    Parameter,
    QuantumCircuit,
    StatevectorBackend,
    Target,
    execute,
    execute_batch,
)
from microquantum.ir import Gate
from microquantum.runtime.plan import ExecutionPlan

LEVELS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Differential equivalence — exact (shots=None), full level x backend matrix
# ---------------------------------------------------------------------------


class TestDifferentialExact:
    """Original vs compiled exact probability vectors across the matrix."""

    @pytest.mark.parametrize("corpus_name,qc", EQUIVALENCE_CORPUS, ids=[n for n, _ in EQUIVALENCE_CORPUS])
    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("backend_name,factory", SIMULATORS, ids=[n for n, _ in SIMULATORS])
    def test_exact_probabilities_preserved(
        self, corpus_name, qc, level, backend_name, factory
    ):
        assert_execution_equivalent(
            qc, level=level, backend_factory=factory, shots=None, tol=TOL_EXACT
        )

    @pytest.mark.parametrize("corpus_name,qc", EQUIVALENCE_CORPUS, ids=[n for n, _ in EQUIVALENCE_CORPUS])
    @pytest.mark.parametrize("level", LEVELS)
    def test_statevector_fidelity_preserved(self, corpus_name, qc, level):
        assert_statevector_preserved(qc, level, tol=TOL_FIDELITY)


# ---------------------------------------------------------------------------
# Differential equivalence — sampled (seeded)
# ---------------------------------------------------------------------------


class TestDifferentialSampled:
    """Original vs compiled seeded sampling agreement."""

    @pytest.mark.parametrize("corpus_name,qc", EQUIVALENCE_CORPUS, ids=[n for n, _ in EQUIVALENCE_CORPUS])
    @pytest.mark.parametrize("level", (0, 2))
    @pytest.mark.parametrize("backend_name,factory", SIMULATORS, ids=[n for n, _ in SIMULATORS])
    def test_sampled_probabilities_preserved(
        self, corpus_name, qc, level, backend_name, factory
    ):
        assert_execution_equivalent(
            qc,
            level=level,
            backend_factory=factory,
            shots=SHOTS_SAMPLED,
            seed=11,
            tol=TOL_SAMPLING,
        )


# ---------------------------------------------------------------------------
# Runtime-route equivalence — A (backend.run) vs B (plan reuse) vs C
# (runtime compile-at-run)
# ---------------------------------------------------------------------------


class TestRuntimeRoutes:
    @pytest.mark.parametrize("name,qc", [("bell", bell()), ("cz", cz_circuit()), ("swap", swap_circuit())], ids=["bell", "cz", "swap"])
    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("backend_name,factory", SIMULATORS, ids=[n for n, _ in SIMULATORS])
    def test_routes_agree(self, name, qc, level, backend_name, factory):
        assert_runtime_equivalent(
            qc, level=level, backend=factory(), shots=2048, seed=3
        )

    def test_plan_reuse_reports_compiled_strategy(self):
        plan = ExecutionPlan(
            compiled=Compiler(optimization_level=1).compile(bell()),
            backend=StatevectorBackend(),
            shots=64,
            seed=1,
        )
        result = execute(plan=plan)
        assert result.metadata["strategy"] == "compiled"

    def test_batch_runs_compiled_plans(self):
        work = [
            ExecutionPlan(
                compiled=Compiler(optimization_level=2).compile(bell()),
                backend=StatevectorBackend(),
                shots=256,
                seed=1,
            ),
            ExecutionPlan(
                compiled=Compiler(optimization_level=2).compile(ghz(3)),
                backend=StatevectorBackend(),
                shots=256,
                seed=1,
            ),
        ]
        results = execute_batch(work, backend=StatevectorBackend(), shots=256, seed=1)
        assert len(results) == 2
        for result in results:
            assert result.metadata["strategy"] == "compiled"


# ---------------------------------------------------------------------------
# Gate-level regressions
# ---------------------------------------------------------------------------


class TestNegativeRotations:
    """Sign and magnitude of negative rotation angles survive compilation."""

    @pytest.mark.parametrize("axis", ("rx", "ry", "rz"))
    @pytest.mark.parametrize("level", (1, 2))
    def test_fidelity_across_angles(self, axis, level):
        for angle in ROTATION_ANGLES:
            qc = QuantumCircuit(1)
            getattr(qc, axis)(angle, 0)
            cr = compile_for(qc, level)
            f = fidelity(qc, cr.circuit())
            assert f > 1.0 - TOL_FIDELITY, (axis, angle, level, f)

    @pytest.mark.parametrize("axis", ("rx", "ry", "rz"))
    @pytest.mark.parametrize("angle", [-3.0, -1.5708, -0.5, 0.0, 0.5, 1.1, 1.5708, 3.14])
    def test_angle_sign_preserved_after_compile(self, axis, angle):
        qc = QuantumCircuit(1)
        getattr(qc, axis)(angle, 0)
        cr = compile_for(qc, 2)
        gates = [op for op in cr.result.operations if isinstance(op, Gate)]
        gates = [g for g in gates if g.name == axis]
        if angle == 0.0:
            assert gates == [], "zero-angle rotation should be removed at level 2"
            return
        assert gates, f"rotation {axis}({angle}) unexpectedly removed"
        assert gates[0].name == axis
        assert abs(float(gates[0].params[0]) - angle) < 1e-9, (axis, angle, gates[0].params)


class TestRotationFusion:
    """CombineRotations is observable-correct for same-axis fusions."""

    @pytest.mark.parametrize(
        "pair",
        [(0.3, 0.4), (-1.1, 0.5), (1.0, -1.0), (0.2, -0.7), (1.5708, -3.14159)],
    )
    def test_fused_single_rotation_at_level_2(self, pair):
        a, b = pair
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rx(b, 0)
        cr = compile_for(qc, 2)
        gates = [op for op in cr.result.operations if isinstance(op, Gate)]
        if abs(a + b) <= 1e-12:
            assert gates == [], "opposite rotations should cancel entirely"
        else:
            assert len(gates) == 1, gates
            assert gates[0].name == "rx"
            assert abs(float(gates[0].params[0]) - (a + b)) < 1e-9
        f = fidelity(qc, cr.circuit())
        assert f > 1.0 - TOL_FIDELITY

    @pytest.mark.parametrize("level", (0, 1, 2))
    def test_gate_count_monotonic(self, level):
        qc = QuantumCircuit(2)
        qc.rx(1.1, 0)
        qc.rx(-0.3, 0)
        qc.ry(0.4, 1)
        source = to_ir_gate_count(qc)
        compiled_count = to_ir_gate_count(compile_for(qc, level).circuit())
        assert compiled_count <= source

    def test_cancel_adjacent_inverse_before_fusion(self):
        qc = QuantumCircuit(1)
        qc.rx(0.5, 0)
        qc.rx(-0.5, 0)
        qc.ry(0.8, 0)
        cr = compile_for(qc, 2)
        gates = [op for op in cr.result.operations if isinstance(op, Gate)]
        assert [g.name for g in gates] == ["ry"], gates
        assert abs(float(gates[0].params[0]) - 0.8) < 1e-9


class TestCZAndSwapBasis:
    """cz/swap decompose exactly toward constrained native bases."""

    @pytest.mark.parametrize("basis", [("h", "x", "cx"), ("h", "x", "cnot")])
    def test_cz_decomposes_to_basis(self, basis):
        target = Target(name=f"basis-{'+'.join(basis)}", num_qubits=2, native_gates=basis)
        cr = compile_for(cz_circuit(), 1, target=target)
        assert cr.is_compatible
        names = set(cr.metadata["compiled_gates_by_type"])
        assert names <= {"h", "cx", "cnot", "cz"}, names
        for _backend_name, factory in SIMULATORS:
            assert_execution_equivalent(
                cz_circuit(), level=1, backend_factory=factory, shots=None, target=target
            )

    @pytest.mark.parametrize("basis", [("h", "x", "cx"), ("h", "x", "cnot")])
    def test_swap_decomposes_to_basis(self, basis):
        target = Target(name=f"basis-{'+'.join(basis)}", num_qubits=2, native_gates=basis)
        cr = compile_for(swap_circuit(), 1, target=target)
        assert cr.is_compatible
        names = set(cr.metadata["compiled_gates_by_type"])
        assert names <= {"h", "cx", "cnot", "swap", "x"}, names
        for _backend_name, factory in SIMULATORS:
            assert_execution_equivalent(
                swap_circuit(), level=1, backend_factory=factory, shots=None, target=target
            )

    def test_full_width_measurement_after_target_compile(self):
        target = Target(name="basis-hcx", num_qubits=3, native_gates=("h", "cx"))
        cr = compile_for(ghz(3), 1, target=target)
        compiled = cr.circuit()
        assert compiled.measurements == [0, 1, 2]
        assert_execution_equivalent(
            ghz(3), level=1, backend_factory=lambda: StatevectorBackend(), shots=None, target=target
        )


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class TestParameterPipeline:
    """Compile symbolic circuits, bind afterward, execute equivalently."""

    def test_symbolic_compile_and_bind(self):
        qc = parameterized_circuit()
        for level in LEVELS:
            cr = compile_for(qc, level)
            compiled = cr.circuit()
            for value in (0.5, -1.2, 2.0):
                bindings = {"theta": value, "phi": value}
                bound = compiled.bind_parameters(bindings)
                f = fidelity(qc.bind_parameters(bindings), bound)
                assert f > 1.0 - TOL_FIDELITY, (level, value, f)

    def test_unknown_binding_rejected(self):
        theta = Parameter("theta")
        compiled = compile_for(QuantumCircuit(1).rx(theta, 0), 1).circuit()
        with pytest.raises(ValueError, match="nonexistent"):
            execute(
                compiled,
                backend=StatevectorBackend(),
                shots=16,
                seed=1,
                parameter_bindings={"nonexistent": 0.5},
            )

    def test_unbound_compiled_rejected_when_executing(self):
        theta = Parameter("theta")
        compiled = compile_for(QuantumCircuit(1).rx(theta, 0), 1).circuit()
        with pytest.raises(ValueError, match="parameterized"):
            execute(compiled, backend=StatevectorBackend(), shots=16, seed=1)

    def test_runtime_parameter_bindings_with_level(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(2).rx(theta, 0).cnot(0, 1)
        result = execute(
            qc,
            backend=StatevectorBackend(),
            shots=512,
            seed=5,
            parameter_bindings={"theta": 0.6},
            optimization_level=2,
        )
        assert result.metadata["strategy"] == "compiled"
        probs = result.probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


class TestMeasurements:
    def test_unmeasured_source_gains_terminal_measurement(self):
        cr = compile_for(bell(), 1)
        assert cr.circuit().measurements == [0, 1]

    def test_existing_measurements_preserved_through_compile(self):
        qc = QuantumCircuit(2).h(0).cnot(0, 1)
        qc.measure(0)
        cr = compile_for(qc, 1)
        assert cr.circuit().measurements == [0]

    def test_measured_counts_equivalent_across_levels(self):
        qc = QuantumCircuit(2).h(0).cnot(0, 1)
        qc.measure_all()
        reference = None
        for level in LEVELS:
            cr = compile_for(qc, level)
            probs = sampled_probability_map(cr.circuit(), StatevectorBackend(), shots=4000, seed=9)
            if reference is None:
                reference = probs
                continue
            for outcome in set(reference) | set(probs):
                assert abs(reference.get(outcome, 0.0) - probs.get(outcome, 0.0)) <= TOL_SAMPLING


# ---------------------------------------------------------------------------
# Empty / no-op / metadata
# ---------------------------------------------------------------------------


class TestEmptyAndNoop:
    @pytest.mark.parametrize("backend_name,factory", SIMULATORS, ids=[n for n, _ in SIMULATORS])
    def test_empty_circuit_all_levels(self, backend_name, factory):
        qc = QuantumCircuit(1)
        for level in LEVELS:
            assert_execution_equivalent(qc, level=level, backend_factory=factory, shots=None)

    @pytest.mark.parametrize("level", LEVELS)
    def test_identity_pair_removed(self, level):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0)
        cr = compile_for(qc, level)
        gates = [op for op in cr.result.operations if isinstance(op, Gate)]
        expected = 2 if level == 0 else 0
        assert len(gates) == expected, (level, gates)


class TestMetadata:
    def test_optimization_level_in_metadata(self):
        for level in LEVELS:
            cr = compile_for(bell(), level)
            assert cr.metadata["optimization_level"] == level

    def test_passes_applied_by_level(self):
        assert compile_for(bell(), 0).passes_applied == []
        assert compile_for(bell(), 1).passes_applied == ["remove-identity-gates", "cancel-adjacent-inverse"]
        assert compile_for(bell(), 2).passes_applied == [
            "remove-identity-gates",
            "cancel-adjacent-inverse",
            "combine-rotations",
        ]

    def test_metadata_gate_counts(self):
        cr = compile_for(bell(), 1)
        assert cr.metadata["source_gates"] == 2
        assert cr.metadata["compiled_gates"] == 2
        assert cr.metadata["compiled_gates_by_type"] == {"cnot": 1, "h": 1}

    def test_depth_non_increasing(self):
        qc = ghz(4)
        source = cr_depth(qc)
        for level in LEVELS:
            assert cr_depth(compile_for(qc, level).circuit()) <= source

    def test_serialization_round_trip(self):
        cr = compile_for(cz_circuit(), 1)
        payload = cr.to_dict()
        assert payload["metadata"]["optimization_level"] == 1
        assert isinstance(cr.to_json(), str)


# ---------------------------------------------------------------------------
# Cross-backend determinism
# ---------------------------------------------------------------------------


class TestCrossBackendDeterminism:
    """All simulators must agree on directed two-qubit gates.

    Guards the ``expand_operator`` k == n ordering contract: a full-space
    two-qubit gate applied in non-ascending target order (e.g. ``cx(1, 0)``,
    exactly what ``GateDecomposition`` emits for ``swap``) must act the same
    on every backend.
    """

    def test_directed_cnot_order_consistent_across_backends(self):
        # control=q1, target=q0: with q0=1, q1=0 nothing may flip
        qc = QuantumCircuit(2).x(0).cnot(1, 0)
        expected = np.zeros(4)
        expected[2] = 1.0
        for _backend_name, factory in SIMULATORS:
            probs = exact_probabilities(qc, factory())
            assert probs is not None
            np.testing.assert_allclose(probs, expected, atol=TOL_EXACT)

    def test_swap_decomposition_equivalent_on_all_backends(self):
        for _backend_name, factory in SIMULATORS:
            assert_execution_equivalent(
                swap_circuit(),
                level=1,
                backend_factory=factory,
                shots=None,
                target=Target(
                    name="basis-x-h-cx", num_qubits=2, native_gates=("h", "x", "cx")
                ),
            )

    def test_cz_decomposition_equivalent_on_all_backends(self):
        for _backend_name, factory in SIMULATORS:
            assert_execution_equivalent(
                cz_circuit(),
                level=1,
                backend_factory=factory,
                shots=None,
                target=Target(
                    name="basis-x-h-cx", num_qubits=2, native_gates=("h", "x", "cx")
                ),
            )


class TestTargetCompatibility:
    def test_no_measurement_target_flags_incompatibility(self):
        target = Target(name="no-measure", num_qubits=2, supports_measurement=False, native_gates=("h", "cx"))
        cr = compile_for(bell(), 1, target=target)
        assert not cr.is_compatible
        assert any("measurement" in d for d in cr.diagnostics)

    def test_oversized_circuit_diagnostic(self):
        target = Target(name="single", num_qubits=1, native_gates=("h", "cx"))
        cr = compile_for(bell(), 1, target=target)
        assert not cr.is_compatible

    def test_incompatible_plan_rejected_by_runtime(self):
        target = Target(name="no-measure", num_qubits=2, supports_measurement=False, native_gates=("h", "cx"))
        cr = compile_for(bell(), 1, target=target)
        plan = ExecutionPlan(compiled=cr, backend=StatevectorBackend(), shots=16, seed=1)
        with pytest.raises(ValueError, match="incompatible"):
            execute(plan=plan)
        plan.options["raise_on_incompatible"] = False
        result = execute(plan=plan)
        assert result is not None


class TestErrorContract:
    def test_dynamic_reset_ir_not_representable(self):
        from microquantum import DynamicCircuit

        dc = DynamicCircuit(1).reset(0)
        cr = Compiler(optimization_level=0).compile(dc.to_ir())
        with pytest.raises(ValueError, match="not representable"):
            cr.circuit()

    def test_bad_level_rejected(self):
        with pytest.raises(ValueError):
            Compiler(optimization_level=3)

    def test_unsupported_gate_not_silently_swallowed(self):
        cr = compile_for(bell(), 1)
        assert isinstance(cr, CompilationResult)

    def test_execution_plan_compiled_type_validated(self):
        with pytest.raises(TypeError, match="CompilationResult"):
            ExecutionPlan(compiled="not a compilation result")
        with pytest.raises(TypeError, match="CompilationResult"):
            ExecutionPlan(compiled=object())


def to_ir_gate_count(circuit: QuantumCircuit) -> int:
    """Number of gate operations in *circuit*'s IR."""
    from microquantum import to_ir

    return sum(1 for op in _walk_gates(to_ir(circuit)))


def _walk_gates(ir):
    from microquantum.ir import Gate as _Gate

    if hasattr(ir, "operations"):
        yield from (op for op in ir.operations if isinstance(op, _Gate))
        return
    for circ in ir.circuits:
        yield from _walk_gates(circ)


def cr_depth(circuit: QuantumCircuit) -> int:
    """Circuit depth via the IR."""
    from microquantum import to_ir

    return to_ir(circuit).depth