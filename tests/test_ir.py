"""Quantum IR + compilation foundation tests (MQ-03).

Covers the public IR surface: Construction, circuit -> IR -> circuit,
validation (valid + invalid), IRPass pipeline, parameter binding, gate
decomposition, target-aware compilation and serialization.
"""

import json

import numpy as np
import pytest

import microquantum
from microquantum import (
    Barrier,
    BindParameters,
    CompilationResult,
    Compiler,
    Condition,
    ConditionalBlock,
    Gate,
    IRCircuit,
    IRModule,
    IRPass,
    Measurement,
    Parameter,
    Reset,
    StatevectorBackend,
    Target,
    assert_valid,
    from_ir,
    optimize,
    validate,
)


def run_amplitudes(circuit) -> np.ndarray:
    """Final amplitudes for a static circuit via the reference backend."""
    return np.asarray(
        StatevectorBackend().run(circuit, shots=1, seed=7).statevector,
        dtype=np.complex128,
    )


def state_union(circuits) -> float:
    """Max pairwise squared fidelity among reference circuits."""
    amps = [run_amplitudes(qc) for qc in circuits]
    best = 0.0
    for a in amps:
        for b in amps:
            best = max(best, float(abs(np.vdot(a, b)) ** 2))
    return best


class TestIRImports:
    def test_top_level_names(self) -> None:
        for name in [
            "IRCircuit",
            "IRModule",
            "IRNode",
            "Gate",
            "Measurement",
            "Reset",
            "Barrier",
            "Condition",
            "ConditionalBlock",
            "to_ir",
            "to_ir_dynamic",
            "from_ir",
            "validate",
            "assert_valid",
            "IRPass",
            "IRPassManager",
            "RemoveIdentityGates",
            "CancelAdjacentInverse",
            "CombineRotations",
            "BindParameters",
            "GateDecomposition",
            "optimize",
            "Compiler",
            "CompilationResult",
        ]:
            assert name in microquantum.__all__
            assert hasattr(microquantum, name)

    def test_circuit_io_methods(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        assert callable(qc.to_ir)
        assert callable(microquantum.QuantumCircuit.from_ir)


class TestCircuitToIR:
    def test_gates_and_order(self) -> None:
        qc = microquantum.QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 2)
        ir = qc.to_ir()
        assert isinstance(ir, IRCircuit)
        assert ir.num_qubits == 3
        assert ir.num_classical_bits == 3
        assert ir.gate_names() == {"h": 1, "cnot": 1, "rz": 1}
        # h(0), cnot(0,1), rz(2) share qubits only in the first two → depth 2
        assert ir.depth == 2

    def test_qubit_ordering_and_source(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.swap(0, 1)
        ir = qc.to_ir()
        op = ir.operations[0]
        assert isinstance(op, Gate)
        assert op.name == "swap"
        assert op.qubits == (0, 1)
        assert op.source == {"circuit_index": 0}

    def test_terminal_measurements_optional(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.x(0)
        plain = qc.to_ir()
        measured = qc.to_ir(include_terminal_measurements=True)
        assert plain.measurements() == []
        assert [m.qubit for m in measured.measurements()] == [0, 1]
        assert [m.classical for m in measured.measurements()] == [0, 1]

    def test_parameterized_circuit(self) -> None:
        th = Parameter("theta")
        qc = microquantum.QuantumCircuit(1)
        qc.rx(th, 0)
        ir = qc.to_ir()
        assert ir.is_parameterized
        assert {p.name for p in ir.parameters} == {"theta"}

    def test_string_repr(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.h(0)
        ir = qc.to_ir()
        assert "main" in str(ir)
        assert "1 gate" in str(ir)

    def test_add_chaining(self) -> None:
        ir = IRCircuit(num_qubits=1)
        out = ir.add(Gate(name="h", qubits=(0,), params=(), condition=None, source={}))
        assert out is ir
        assert ir.num_gates == 1

    def test_roundtrip_equivalence(self) -> None:
        qc = microquantum.QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.rz(0.5, 1)
        qc.swap(1, 2)
        rebuilt = from_ir(qc.to_ir())
        assert state_union([qc, rebuilt]) == pytest.approx(1.0, abs=1e-6)

    def test_roundtrip_via_static_method(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.cx(0, 1)
        rebuilt = microquantum.QuantumCircuit.from_ir(qc.to_ir())
        assert state_union([qc, rebuilt]) == pytest.approx(1.0, abs=1e-6)


class TestDynamicIR:
    def test_measure_reset_conditional(self) -> None:
        dc = microquantum.DynamicCircuit(2, 2)
        assert callable(dc.reset)
        dc.h(0)
        dc.measure(0, 0)
        dc.reset(1)
        dc.classical_if(0, lambda d: d.x(1))
        ir = dc.to_ir()
        assert isinstance(ir, IRCircuit)
        assert ir.has_reset()
        assert ir.has_conditions()
        assert len(ir.measurements()) == 1
        reset_nodes = [op for op in ir.operations if isinstance(op, Reset)]
        assert reset_nodes[0].qubit == 1
        if_blocks = [op for op in ir.operations if isinstance(op, ConditionalBlock)]
        assert len(if_blocks) == 1
        nested = if_blocks[0].operations
        assert len(nested) == 1
        assert nested[0].name == "x"

    def test_reset_simulation(self) -> None:
        dc = microquantum.DynamicCircuit(2, 2)
        dc.x(0)
        dc.reset(0)
        result = dc.run(seed=3)
        amp0 = np.asarray(result.final_state.amplitudes, dtype=np.complex128)
        assert abs(amp0[0]) == pytest.approx(1.0, abs=1e-9)
        assert np.abs(amp0[1:]).max() < 1e-9

    def test_reset_after_h_renormalizes(self) -> None:
        dc = microquantum.DynamicCircuit(1, 1)
        dc.h(0)
        dc.reset(0)
        result = dc.run(seed=5)
        amp = np.asarray(result.final_state.amplitudes, dtype=np.complex128)
        assert amp.shape == (2,)
        assert abs(amp[0]) == pytest.approx(1.0, abs=1e-9)

    def test_reset_out_of_range(self) -> None:
        dc = microquantum.DynamicCircuit(1, 1)
        with pytest.raises(ValueError):
            dc.reset(1)


class TestValidation:
    def test_valid_circuits_pass(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert validate(qc.to_ir()) == []
        assert_valid(qc.to_ir())

    def test_dangling_gate_qubit(self) -> None:
        ir = IRCircuit(num_qubits=1)
        ir.add(Gate(name="h", qubits=(5,), condition=None))
        assert any("invalid qubit" in err for err in validate(ir))
        with pytest.raises(ValueError):
            assert_valid(ir)

    def test_negative_qubit(self) -> None:
        ir = IRCircuit(num_qubits=1)
        ir.add(Gate(name="h", qubits=(-1,), condition=None))
        assert any("invalid qubit" in err for err in validate(ir))

    def test_measurement_cbit_out_of_range(self) -> None:
        ir = IRCircuit(num_qubits=1, num_classical_bits=1)
        ir.add(Gate(name="h", qubits=(0,), condition=None))
        ir.add(Measurement(qubit=0, classical=3))
        assert any("invalid classical bit" in err for err in validate(ir))
        with pytest.raises(ValueError):
            assert_valid(ir)

    def test_condition_reference_out_of_range(self) -> None:
        ir = IRCircuit(num_qubits=1, num_classical_bits=1)
        ir.add(
            Gate(
                name="x",
                qubits=(0,),
                condition=Condition(bit=4),
            )
        )
        assert any("condition references invalid classical bit" in err for err in validate(ir))

    def test_unknown_gate_name(self) -> None:
        ir = IRCircuit(num_qubits=1)
        ir.add(Gate(name="not-a-gate", qubits=(0,), condition=None))
        assert any("not a known gate" in err for err in validate(ir))

    def test_wrong_arity(self) -> None:
        ir = IRCircuit(num_qubits=2)
        ir.add(Gate(name="cnot", qubits=(0,), condition=None))
        assert any("expects 2 qubit(s)" in err for err in validate(ir))

    def test_params_on_non_rotation_gate(self) -> None:
        ir = IRCircuit(num_qubits=1)
        ir.add(Gate(name="h", qubits=(0,), params=(0.2,), condition=None))
        assert any("does not accept parameters" in err for err in validate(ir))

    def test_conditional_block_nested_check(self) -> None:
        ir = IRCircuit(num_qubits=1, num_classical_bits=1)
        ir.add(
            ConditionalBlock(
                condition=Condition(bit=0),
                operations=[Gate(name="x", qubits=(9,), condition=None)],
            )
        )
        assert any("invalid qubit" in err for err in validate(ir))
        with pytest.raises(ValueError):
            assert_valid(ir)


class TestPasses:
    def test_identity_removal(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.t(0)
        qc.tdg(0)
        qc.h(0)
        qc.h(0)
        qc.s(1)
        qc.sdg(1)
        ir = optimize(qc.to_ir(), level=1)
        assert ir.num_gates == 0

    def test_cancel_adjacent_inverse(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.s(0)
        qc.sdg(0)
        qc.t(1)
        qc.tdg(1)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(0)
        ir = optimize(qc.to_ir(), level=1)
        assert ir.num_gates == 0

    def test_combine_rotations(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.rx(0.3, 0)
        qc.rx(0.4, 0)
        ir = optimize(qc.to_ir(), level=2)
        assert ir.gate_names() == {"rx": 1}
        rebuilt = from_ir(ir)
        assert state_union([qc, rebuilt]) == pytest.approx(1.0, abs=1e-6)

    def test_combine_rotations_not_at_level_1(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.rx(0.3, 0)
        qc.rx(0.4, 0)
        ir = optimize(qc.to_ir(), level=1)
        assert ir.num_gates == 2

    def test_rotation_cancellation_folds(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.rz(0.2, 0)
        qc.rz(2 * np.pi - 0.2, 0)
        ir = optimize(qc.to_ir(), level=1)
        assert ir.num_gates == 0

    def test_bind_parameters(self) -> None:
        alpha = Parameter("alpha")
        qc = microquantum.QuantumCircuit(1)
        qc.rx(alpha, 0)
        qc.rz(0.2, 0)
        ir = qc.to_ir()
        assert ir.is_parameterized
        bound = BindParameters({"alpha": 1.5}).run(ir)
        assert not bound.is_parameterized
        rebuilt = from_ir(bound)
        assert state_union([qc.bind_parameters({"alpha": 1.5}), rebuilt]) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_partial_bind_leaves_symbols(self) -> None:
        alpha = Parameter("alpha")
        beta = Parameter("beta")
        qc = microquantum.QuantumCircuit(1)
        qc.rx(alpha, 0)
        qc.rz(beta, 0)
        bound = BindParameters({"alpha": 1.0}).run(qc.to_ir())
        assert bound.is_parameterized
        assert {p.name for p in bound.parameters} == {"beta"}

    def test_cancel_adjacent_inverse_semantics(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.t(0)
        qc.tdg(0)
        ir = optimize(qc.to_ir(), level=1)
        assert ir.num_gates == 0

    def test_invalid_optimization_level(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.h(0)
        with pytest.raises(ValueError, match="level"):
            optimize(qc.to_ir(), level=99)

    def test_empty_conditional_block_removed(self) -> None:
        ir = IRCircuit(num_qubits=1, num_classical_bits=1)
        ir.add(ConditionalBlock(condition=Condition(bit=0), operations=[]))
        out = optimize(ir, level=1)
        assert out.num_gates == 0
        assert len(out.operations) == 0


class TestGateDecomposition:
    def test_cz_decomposition(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.cz(0, 1)
        target = Target(name="basis", num_qubits=2, native_gates=("h", "cnot", "rx", "rz"))
        result = Compiler(optimization_level=0).compile(qc, target=target)
        assert result.result.gate_names() == {"h": 2, "cnot": 1}
        rebuilt = result.circuit()
        assert result.is_compatible
        assert state_union([qc, rebuilt]) == pytest.approx(1.0, abs=1e-9)

    def test_swap_decomposition(self) -> None:
        qc = microquantum.QuantumCircuit(3)
        qc.swap(0, 2)
        target = Target(name="basis", num_qubits=3, native_gates=("h", "cnot", "rx", "rz"))
        result = Compiler(optimization_level=0).compile(qc, target=target)
        assert result.result.gate_names() == {"cnot": 3}
        rebuilt = result.circuit()
        assert state_union([qc, rebuilt]) == pytest.approx(1.0, abs=1e-9)

    def test_compatible_basis(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        target = Target(name="native", num_qubits=2, native_gates=("h", "cnot", "rx", "rz"))
        result = Compiler(optimization_level=0).compile(qc, target=target)
        assert result.is_compatible
        assert result.diagnostics == []
        # gates already in the native basis are untouched
        assert result.result.gate_names() == {"h": 1, "cnot": 1}

    def test_unsupported_gate_diagnostics(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.t(0)
        target = Target(name="narrow", num_qubits=1, native_gates=("h", "cnot"))
        result = Compiler(optimization_level=0).compile(qc, target=target)
        assert not result.is_compatible
        assert any("not in target basis" in d for d in result.diagnostics)

    def test_qubit_count_diagnostic(self) -> None:
        qc = microquantum.QuantumCircuit(3)
        qc.x(0)
        target = Target(name="small", num_qubits=2, native_gates=("x",))
        result = Compiler(optimization_level=0).compile(qc, target=target)
        assert any("supports at most 2" in d for d in result.diagnostics)

    def test_measurement_diagnostic(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.x(0)
        ir = qc.to_ir(include_terminal_measurements=True)
        target = Target(
            name="no-measure",
            num_qubits=1,
            native_gates=("x",),
            supports_measurement=False,
        )
        result = Compiler(optimization_level=0).compile(ir, target=target)
        assert any("does not support measurement" in d for d in result.diagnostics)


class TestCompiler:
    def test_source_kept(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = Compiler(optimization_level=1).compile(qc)
        assert isinstance(result, CompilationResult)
        assert result.source.num_gates == 2
        assert result.passes_applied == ["remove-identity-gates", "cancel-adjacent-inverse"]

    def test_no_default_passes_at_level_0(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.h(0)
        result = Compiler(optimization_level=0).compile(qc)
        assert result.passes_applied == []

    def test_invalid_level(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        with pytest.raises(ValueError):
            Compiler(optimization_level=-1).compile(qc)

    def test_custom_pass_in_pipeline(self) -> None:
        class Renamer(IRPass):
            name = "renamer"

            def run(self, ir):
                return IRCircuit(
                    num_qubits=ir.num_qubits,
                    num_classical_bits=ir.num_classical_bits,
                    name="renamed",
                    operations=list(ir.operations),
                    metadata=dict(ir.metadata),
                )

        qc = microquantum.QuantumCircuit(1)
        qc.h(0)
        result = Compiler(optimization_level=0).compile(
            qc,
            passes=[Renamer()],
        )
        assert "renamer" in result.passes_applied
        assert result.result.name == "renamed"

    def test_serialization_roundtrip(self) -> None:
        qc = microquantum.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        target = Target(name="t", num_qubits=2, native_gates=("h", "cnot"))
        result = Compiler(optimization_level=1).compile(qc, target=target)
        payload = result.to_json()
        data = json.loads(payload)
        assert data["passes_applied"]
        assert data["target"]["name"] == "t"
        # 2 original gates + 2 terminal measurements
        assert len(data["result"]["operations"]) == 4
        assert result.result.num_gates == 2

    def test_circuit_rebuild_with_measurements(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.h(0)
        ir = qc.to_ir(include_terminal_measurements=True)
        result = Compiler(optimization_level=0).compile(ir)
        rebuilt = result.circuit()
        assert rebuilt.gate_count() == 1

    def test_dynamic_ir_cannot_rebuild_static_circuit(self) -> None:
        dc = microquantum.DynamicCircuit(2, 2)
        dc.h(0)
        dc.reset(1)
        ir = dc.to_ir()
        with pytest.raises(ValueError):
            from_ir(ir)

    def test_invalid_input_rejected(self) -> None:
        ir = IRCircuit(num_qubits=1)
        ir.add(Gate(name="h", qubits=(7,)))
        with pytest.raises(ValueError):
            Compiler(optimization_level=0).compile(ir)


class TestModuleSerialization:
    def test_module_serialization_roundtrip(self) -> None:
        qc = microquantum.QuantumCircuit(1)
        qc.h(0)
        module = IRModule(name="unit", circuits=[qc.to_ir()])
        data = module.to_dict()
        assert data["name"] == "unit"
        assert len(data["circuits"]) == 1
        json.loads(module.to_json())

    def test_barrier_is_valid_and_roundtrips(self) -> None:
        ir = IRCircuit(num_qubits=2)
        ir.add(Gate(name="h", qubits=(0,), condition=None))
        ir.add(Barrier(qubits=(0, 1)))
        assert validate(ir) == []
        rebuilt = from_ir(ir)
        assert rebuilt.gate_count() == 1