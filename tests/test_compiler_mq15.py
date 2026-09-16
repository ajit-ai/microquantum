"""Compiler pipeline + safe optimization tests (MQ-15-A).

Verifies the production compiler contract: a single public entry point
(``Compiler``), validation, safe optimization levels 0/1/2, parameter and
measurement preservation, target/capability validation, backend execution,
metrics, and QASM round trips.  Every optimization must preserve the state
vector of the program; nothing is ever silently dropped or invented.
"""

import json

import numpy as np
import pytest

import microquantum
from microquantum import (
    BindParameters,
    CompilationResult,
    Compiler,
    DensityMatrixBackend,
    Gate,
    GateDecomposition,
    IRCircuit,
    IRPass,
    Measurement,
    MPSBackend,
    Operator,
    Parameter,
    StatevectorBackend,
    Target,
    TreeTensorNetworkBackend,
    assert_valid,
    from_ir,
    to_ir,
    validate,
)
from microquantum.core.circuit import QuantumCircuit


def run_amplitudes(circuit: QuantumCircuit) -> np.ndarray:
    """Final amplitudes for a static circuit via the reference backend."""
    return np.asarray(
        StatevectorBackend().run(circuit, shots=1, seed=7).statevector,
        dtype=np.complex128,
    )


def state_fidelity(a: QuantumCircuit, b: QuantumCircuit) -> float:
    """Squared fidelity between the state vectors of two circuits."""
    amps = [run_amplitudes(a), run_amplitudes(b)]
    return float(abs(np.vdot(amps[0], amps[1])) ** 2)


def mq15_circuits():
    """Numeric circuits the optimizer can genuinely transform."""
    return [
        ("bell", QuantumCircuit(2).h(0).cx(0, 1)),
        ("identity-pair", QuantumCircuit(2).z(0).z(0).h(0).cx(0, 1).z(1).z(1)),
        ("rotation-fusion", QuantumCircuit(1).rx(0.3, 0).rx(0.4, 0).ry(1.1, 0).ry(-1.1, 0)),
        ("swap", QuantumCircuit(2).h(0).swap(0, 1)),
        ("inverse-pair", QuantumCircuit(1).s(0).sdg(0).t(0).tdg(0).h(0)),
    ]


class TestCompilerAPI:
    def test_single_public_entry_point(self) -> None:
        # The Compiler class is THE compile entry point; no competing API.
        assert "Compiler" in microquantum.__all__
        assert "CompilationResult" in microquantum.__all__
        assert "NewCompiler" not in microquantum.__all__
        assert "TranspilerV2" not in microquantum.__all__

    def test_compile_returns_compilation_result(self) -> None:
        result = Compiler().compile(QuantumCircuit(2).h(0).cx(0, 1))
        assert isinstance(result, CompilationResult)
        assert isinstance(result.source, IRCircuit)
        assert isinstance(result.result, IRCircuit)
        assert result.is_compatible  # no target -> no diagnostics
        assert isinstance(result.circuit(), QuantumCircuit)

    def test_compile_deterministic(self) -> None:
        qc = QuantumCircuit(3).h(0).cx(0, 2).rz(0.5, 1).rx(0.2, 2)
        a = Compiler(optimization_level=1).compile(qc)
        b = Compiler(optimization_level=1).compile(qc)
        assert a.to_dict() == b.to_dict()
        assert a.result.operations == b.result.operations

    @pytest.mark.parametrize(
        "level", [-1, 3, 4, 10],
    )
    def test_invalid_optimization_level_rejected(self, level: int) -> None:
        with pytest.raises(ValueError):
            Compiler(optimization_level=level)

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_valid_optimization_levels_accepted(self, level: int) -> None:
        assert Compiler(optimization_level=level).optimization_level == level

    def test_level_zero_validation_only(self) -> None:
        qc = QuantumCircuit(2).h(0).h(0).cx(0, 1)
        result = Compiler(optimization_level=0).compile(qc)
        assert result.passes_applied == []
        assert result.result.operations == result.source.operations
        assert result.metadata["optimization_level"] == 0

    def test_level_one_applied_passes(self) -> None:
        qc = QuantumCircuit(2).h(0).h(0).x(1)
        result = Compiler(optimization_level=1).compile(qc)
        assert result.passes_applied == ["remove-identity-gates", "cancel-adjacent-inverse"]
        assert result.result.num_gates == 1  # the x(1) survives

    def test_level_two_fuses_rotations(self) -> None:
        qc = QuantumCircuit(1).rx(0.3, 0).rx(0.4, 0)
        result = Compiler(optimization_level=2).compile(qc)
        assert "combine-rotations" in result.passes_applied
        assert result.result.gate_names() == {"rx": 1}
        assert state_fidelity(qc, result.circuit()) > 1.0 - 1e-9

    def test_custom_pass_pipeline(self) -> None:
        class PrependH(IRPass):
            @property
            def name(self) -> str:
                return "prepend-h"

            def run(self, ir: IRCircuit) -> IRCircuit:
                return IRCircuit(
                    num_qubits=ir.num_qubits,
                    operations=(Gate(name="h", qubits=(0,)), *ir.operations),
                )

        result = Compiler().compile(
            QuantumCircuit(2).x(0),
            passes=[PrependH()],
        )
        assert result.passes_applied[-1] == "prepend-h"
        assert result.result.operations[0].name == "h"

    def test_compile_accepts_ir_directly(self) -> None:
        ir = to_ir(QuantumCircuit(2).h(0).cx(0, 1))
        result = Compiler().compile(ir)
        assert result.source is ir


class TestIRRoundTrip:
    def test_circuit_ir_circuit_roundtrip(self) -> None:
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.7, 2)
        rebuilt = from_ir(to_ir(qc))
        assert rebuilt.num_qubits == 3
        assert rebuilt.to_ir().gate_names() == qc.to_ir().gate_names()
        assert state_fidelity(qc, rebuilt) > 1.0 - 1e-9

    def test_ir_validation_accepts_valid_ir(self) -> None:
        ir = to_ir(QuantumCircuit(2).h(0).cx(0, 1))
        assert validate(ir) == []
        assert_valid(ir)

    @pytest.mark.parametrize(
        "bad_ir",
        [
            IRCircuit(1, operations=[Gate(name="foo", qubits=(0,))]),
            IRCircuit(2, operations=[Gate(name="h", qubits=(0, 1))]),
            IRCircuit(1, operations=[Gate(name="h", qubits=(5,))]),
            IRCircuit(1, operations=[Gate(name="h", qubits=(0,), params=(0.5,))]),
            IRCircuit(1, operations=[Gate(name="rx", qubits=(0,))]),
            IRCircuit(1, operations=[Gate(name="rx", qubits=(0,), params=("nope",))]),
            IRCircuit(1, operations=[Gate(name="rx", qubits=(0,), params=(0.25j,))]),
            IRCircuit(1, operations=[Measurement(qubit=0, classical=7)]),
        ],
    )
    def test_malformed_ir_fails_explicitly(self, bad_ir: IRCircuit) -> None:
        with pytest.raises(ValueError):
            assert_valid(bad_ir)
        with pytest.raises(ValueError):
            from_ir(bad_ir)

    def test_measurement_classical_mismatch_rejected(self) -> None:
        ir = IRCircuit(
            num_qubits=2,
            num_classical_bits=2,
            operations=[
                Gate(name="h", qubits=(0,)),
                Measurement(qubit=1, classical=0),
            ],
        )
        # structurally valid...
        assert validate(ir) == []
        # ...but not representable as a static QuantumCircuit.
        with pytest.raises(ValueError):
            from_ir(ir)


class TestSafeOptimization:
    def test_identity_gates_removed(self) -> None:
        qc = QuantumCircuit(2).h(0).x(0).x(0)
        result = Compiler(optimization_level=1).compile(qc)
        assert "remove-identity-gates" in result.passes_applied
        assert state_fidelity(qc, result.circuit()) > 1.0 - 1e-9

    def test_adjacent_inverse_cancellation(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        result = Compiler(optimization_level=1).compile(qc)
        assert result.result.num_gates == 0
        assert state_fidelity(qc, result.circuit()) > 1.0 - 1e-9

    @pytest.mark.parametrize(
        "gate", ["h", "x", "y", "z", "cnot", "cz", "swap"],
    )
    def test_self_inverse_pairs_cancelled(self, gate: str) -> None:
        qc = QuantumCircuit(2)
        if gate in ("cnot", "cz", "swap"):
            getattr(qc, gate)(0, 1)
            getattr(qc, gate)(0, 1)
        else:
            getattr(qc, gate)(0)
            getattr(qc, gate)(0)
        result = Compiler(optimization_level=1).compile(qc)
        assert result.result.num_gates == 0

    @pytest.mark.parametrize("gate", ["rx", "ry", "rz"])
    def test_zero_rotations_removed(self, gate: str) -> None:
        qc = QuantumCircuit(1)
        getattr(qc, gate)(0.0, 0)
        assert Compiler(optimization_level=1).compile(qc).result.num_gates == 0

    def test_full_turn_rotation_is_identity(self) -> None:
        qc = QuantumCircuit(1).rz(2 * np.pi, 0)
        assert Compiler(optimization_level=1).compile(qc).result.num_gates == 0

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_semantics_preserved_at_every_level(
        self, level: int
    ) -> None:
        for name, qc in mq15_circuits():
            result = Compiler(optimization_level=level).compile(qc)
            assert state_fidelity(qc, result.circuit()) > 1.0 - 1e-9, name

    def test_semantics_preserved_under_decomposition(self) -> None:
        target = Target(
            name="test-basis",
            num_qubits=2,
            native_gates=("h", "cnot"),
        )
        qc = QuantumCircuit(2).h(0).cz(0, 1)
        for level in (0, 1, 2):
            result = Compiler(optimization_level=level).compile(qc, target=target)
            assert result.is_compatible
            assert state_fidelity(qc, result.circuit()) > 1.0 - 1e-9

    def test_optimization_does_not_invent_gates(self) -> None:
        qc = QuantumCircuit(2).h(0).x(1)
        result = Compiler(optimization_level=1).compile(qc)
        present = set(result.result.gate_names())
        assert present <= {"h", "x"}
        assert qc.to_ir().gate_names() == result.result.gate_names()


class TestParameters:
    def test_symbolic_parameter_survives_compilation(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        result = Compiler(optimization_level=2).compile(qc)
        rebuilt = result.circuit()
        assert rebuilt.is_parameterized
        assert "theta" in {p.name for p in rebuilt.parameters}
        bound = rebuilt.bind_parameters({"theta": 0.8})
        assert state_fidelity(qc.bind_parameters({"theta": 0.8}), bound) > 1.0 - 1e-9

    def test_repeated_parameter_preserved(self) -> None:
        theta = Parameter("alpha")
        ir = IRCircuit(
            num_qubits=2,
            num_classical_bits=2,
            operations=[
                Gate(name="rz", qubits=(0,), params=(theta,)),
                Gate(name="rz", qubits=(1,), params=(theta,)),
            ],
        )
        result = Compiler(optimization_level=2).compile(ir)
        rebuilt = result.circuit()
        bound = rebuilt.bind_parameters({"alpha": 1.2})
        assert state_fidelity(
            QuantumCircuit(2).rz(1.2, 0).rz(1.2, 1), bound
        ) > 1.0 - 1e-9

    def test_symbolic_rotations_never_fused_or_cancelled(self) -> None:
        theta = Parameter("theta")
        ir = IRCircuit(
            num_qubits=1,
            operations=[
                Gate(name="rx", qubits=(0,), params=(theta,)),
                Gate(name="rx", qubits=(0,), params=(theta,)),
            ],
        )
        for level in (1, 2):
            result = Compiler(optimization_level=level).compile(ir)
            assert result.result.num_gates == 2
            assert result.result.gate_names() == {"rx": 2}

    def test_compile_bind_equals_bind_compile(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2).ry(theta, 0).cx(0, 1)
        binding = {"theta": 0.5}
        pre = Compiler(optimization_level=2).compile(qc.bind_parameters(binding))
        post = Compiler(optimization_level=2).compile(qc).circuit().bind_parameters(binding)
        assert pre.circuit().to_ir().gate_names() == post.to_ir().gate_names()
        assert state_fidelity(pre.circuit(), post) > 1.0 - 1e-9

    def test_bind_parameters_pass_partial(self) -> None:
        alpha = Parameter("alpha")
        beta = Parameter("beta")
        ir = IRCircuit(
            num_qubits=1,
            operations=[
                Gate(name="rz", qubits=(0,), params=(alpha,)),
                Gate(name="rx", qubits=(0,), params=(beta,)),
            ],
        )
        result = Compiler().compile(ir, passes=[BindParameters({"alpha": 0.4})])
        rebuilt = result.circuit()
        assert rebuilt.is_parameterized
        assert {"beta"} == {p.name for p in rebuilt.parameters}


class TestMeasurements:
    def test_explicit_measurement_preserved(self) -> None:
        qc = QuantumCircuit(3).h(0).cx(0, 1).measure(1)
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        assert compiled.measurements == [1]

    def test_measure_all_preserved(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1).measure_all()
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        assert compiled.measurements == [0, 1]

    def test_measurement_subset_preserved(self) -> None:
        qc = QuantumCircuit(3).x(0).measure(0).measure(2)
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        assert compiled.measurements == [0, 2]

    def test_measurement_order_preserved(self) -> None:
        qc = QuantumCircuit(3).x(1).measure(2).measure(0).measure(1)
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        assert compiled.measurements == [2, 0, 1]

    def test_optimization_never_removes_measurements(self) -> None:
        qc = QuantumCircuit(2).h(0).h(0).cx(0, 1).measure_all()
        result = Compiler(optimization_level=1).compile(qc)
        assert result.result.num_gates < qc.gate_count()
        measurements = [m for m in result.result.operations if isinstance(m, Measurement)]
        assert [m.qubit for m in measurements] == [0, 1]

    def test_measurements_survive_roundtrip(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1).measure_all()
        rebuilt = from_ir(to_ir(qc, include_terminal_measurements=True))
        assert rebuilt.measurements == [0, 1]

    def test_measurement_acts_as_optimization_barrier(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure(0)
        qc.h(1)
        qc.measure(1)
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        assert compiled.measurements == [0, 1]
        assert compiled.to_ir().gate_names() == {"h": 2}  # separated by measurements

    def test_unmeasured_circuit_adds_all_terminal_measurements(self) -> None:
        qc = QuantumCircuit(3).h(0)
        result = Compiler(optimization_level=1).compile(qc)
        measurements = [m for m in result.result.operations if isinstance(m, Measurement)]
        assert [m.qubit for m in measurements] == [0, 1, 2]


class TestTargetValidation:
    def _target(self, gates=("h", "cnot"), num_qubits: int = 2) -> Target:
        return Target(
            name="test-target",
            num_qubits=num_qubits,
            native_gates=gates,
            supports_measurement=True,
        )

    def test_supported_gates_compile_clean(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        result = Compiler().compile(qc, target=self._target())
        assert result.is_compatible
        assert result.diagnostics == []

    def test_unsupported_gate_reported(self) -> None:
        qc = QuantumCircuit(1).t(0)
        result = Compiler().compile(qc, target=self._target())
        assert not result.is_compatible
        assert any("t" in d for d in result.diagnostics)

    def test_too_many_qubits_reported(self) -> None:
        qc = QuantumCircuit(3).h(0)
        result = Compiler().compile(qc, target=self._target(num_qubits=2))
        assert not result.is_compatible
        assert any("qubit" in d for d in result.diagnostics)

    def test_unsupported_measurement_reported(self) -> None:
        target = Target(
            name="no-measure",
            num_qubits=2,
            native_gates=("h", "cnot"),
            supports_measurement=False,
        )
        qc = QuantumCircuit(2).h(0).measure_all()
        result = Compiler().compile(qc, target=target)
        assert not result.is_compatible
        assert any("measure" in d for d in result.diagnostics)

    def test_no_silent_unsupported_operations(self) -> None:
        qc = QuantumCircuit(1).t(0).h(0)
        result = Compiler().compile(qc, target=self._target())
        names = set(result.result.gate_names())
        # The T gate is left in place and pointed at, never quietly dropped.
        assert "t" in names
        assert not result.is_compatible

    def test_cx_target_spelling_matches_ir_cnot(self) -> None:
        # Targets advertise "cx"; IR uses "cnot".  They are the same gate.
        result = Compiler().compile(
            QuantumCircuit(2).h(0).cx(0, 1),
            target=self._target(gates=("h", "cx")),
        )
        assert result.is_compatible
        assert result.circuit().to_ir().gate_names() == {"h": 1, "cnot": 1}

    def test_cz_lowered_to_hcnoth_basis_with_cx_spelling(self) -> None:
        target = self._target(gates=("h", "cx"))
        qc = QuantumCircuit(2).cz(0, 1)
        result = Compiler().compile(qc, target=target)
        assert result.circuit().to_ir().gate_names() == {"h": 2, "cnot": 1}
        assert result.is_compatible
        assert state_fidelity(qc, result.circuit()) > 1.0 - 1e-9

    def test_swap_lowered_to_three_cnots(self) -> None:
        target = self._target(gates=("h", "cx"))
        qc = QuantumCircuit(2).swap(0, 1)
        result = Compiler().compile(qc, target=target)
        assert result.circuit().to_ir().gate_names() == {"cnot": 3}
        assert result.is_compatible
        assert state_fidelity(qc, result.circuit()) > 1.0 - 1e-9

    def test_gate_decomposition_direct(self) -> None:
        qc = QuantumCircuit(2).cz(0, 1)
        ir = qc.to_ir()
        lowered = GateDecomposition(["h", "cx"]).run(ir)
        assert lowered.gate_names() == {"h": 2, "cnot": 1}


class TestBackendExecution:
    def test_compiled_circuit_runs_statevector(self) -> None:
        qc = QuantumCircuit(2).z(0).z(0).h(0).cx(0, 1).measure_all()
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        counts = StatevectorBackend().run(compiled, shots=1000, seed=42).counts
        assert counts.get("00", 0) > 300
        assert counts.get("11", 0) > 300
        assert counts.get("01", 0) + counts.get("10", 0) < 200

    def test_compiled_circuit_runs_density_matrix(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        dm = DensityMatrixBackend().run(compiled, shots=None).density_matrix
        diag = np.real(np.diag(dm))
        assert np.allclose(diag, [0.5, 0.0, 0.0, 0.5], atol=1e-9)

    def test_original_and_compiled_statevectors_agree(self) -> None:
        qc = QuantumCircuit(3).h(0).cx(0, 1).cx(1, 2)
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        assert state_fidelity(qc, compiled) > 1.0 - 1e-12

    def test_compiled_circuit_runs_mps(self) -> None:
        qc = QuantumCircuit(3).h(0).h(0).cx(0, 1).cx(1, 2).measure_all()
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        ref = StatevectorBackend().run(compiled, shots=4000, seed=10).counts
        mps = MPSBackend().run(compiled, shots=4000, seed=10).counts
        worst = max(abs(mps.get(k, 0) / 4000 - ref[k] / 4000) for k in ref)
        assert worst < 0.1

    def test_compiled_circuit_runs_ttn(self) -> None:
        qc = QuantumCircuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
        compiled = Compiler(optimization_level=2).compile(qc).circuit()
        sv = StatevectorBackend().run(compiled, shots=2000, seed=42).counts
        ttn = TreeTensorNetworkBackend().run(compiled, shots=2000, seed=42).counts
        assert sv == ttn  # shared sampling path with the state vector

    def test_expectation_values_preserved(self) -> None:
        qc = QuantumCircuit(2).h(0).h(0).cx(0, 1)
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        for q in (0, 1):
            assert np.isclose(
                qc.expectation_value(Operator.Z(), targets=[q]),
                compiled.expectation_value(Operator.Z(), targets=[q]),
                atol=1e-9,
            )


class TestMetrics:
    def test_metadata_reports_reduction(self) -> None:
        qc = QuantumCircuit(2).h(0).h(0).cx(0, 1)
        result = Compiler(optimization_level=1).compile(qc)
        meta = result.metadata
        assert meta["source_qubits"] == 2
        assert meta["compiled_qubits"] == 2
        assert meta["source_gates"] == 3
        assert meta["compiled_gates"] == 1
        assert meta["source_depth"] >= meta["compiled_depth"]
        assert meta["source_gates_by_type"] == {"h": 2, "cnot": 1}
        assert meta["compiled_gates_by_type"] == {"cnot": 1}

    def test_level_zero_metadata_unchanged(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        result = Compiler(optimization_level=0).compile(qc)
        meta = result.metadata
        assert meta["source_gates"] == meta["compiled_gates"]
        assert meta["source_depth"] == meta["compiled_depth"]

    def test_metadata_is_json_safe(self) -> None:
        result = Compiler().compile(QuantumCircuit(2).h(0).cx(0, 1))
        payload = json.loads(result.to_json())
        assert payload["metadata"]["source_gates_by_type"] == {"h": 1, "cnot": 1}
        assert isinstance(payload["metadata"]["optimization_level"], int)

    def test_depth_remains_method(self) -> None:
        assert callable(QuantumCircuit.depth)
        result = Compiler().compile(QuantumCircuit(2).h(0).cx(0, 1))
        compiled = result.circuit()
        assert isinstance(compiled.depth(), int)
        assert compiled.depth() == result.result.depth
        assert compiled.gate_count() == result.result.num_gates


class TestQASM:
    def test_compile_to_qasm_roundtrip(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        compiled = Compiler(optimization_level=1).compile(qc).circuit()
        rebuilt = QuantumCircuit.from_qasm(compiled.qasm())
        assert state_fidelity(compiled, rebuilt) > 1.0 - 1e-9
        assert rebuilt.to_ir().gate_names() == compiled.to_ir().gate_names()

    def test_qasm_roundtrip_preserves_state(self) -> None:
        qc = QuantumCircuit(2).h(0).rz(0.7, 1).cx(0, 1)
        rebuilt = QuantumCircuit.from_qasm(qc.qasm())
        assert state_fidelity(qc, rebuilt) > 1.0 - 1e-9