"""Conformance - compiler pipeline & safe optimization (MQ-15).

Permanent public-face checks for the compiler contract:

* the single ``Compiler`` entry point and its ``CompilationResult``,
* safe optimization levels 0/1/2 and loud rejection of invalid levels,
* deterministic compilation and metrics in the result metadata,
* Circuit -> IR -> Circuit round trips that preserve behavior,
* symbolic-parameter safety (never fuse/cancel unbound rotations),
* measurement preservation through compilation and rebuilding,
* the two-tier target contract (soft diagnostics vs. hard runtime
  errors) with ``cx``/``cnot`` name normalization,
* compiled circuits executing on the available backends,
* QASM round trips of compiled circuits.

Every assertion matches behavior documented in
``docs/execution/compilation.rst`` and exercised by
``tests/test_compiler_mq15.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from microquantum import (
    CompilationResult,
    Compiler,
    DensityMatrixBackend,
    Gate,
    IRCircuit,
    Parameter,
    QuantumCircuit,
    StatevectorBackend,
    Target,
    from_ir,
    optimize,
    to_ir,
)
from microquantum.core.circuit import QuantumCircuit as QC


def amplitudes(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(
        StatevectorBackend().run(circuit, shots=1, seed=7).statevector,
        dtype=np.complex128,
    )


def fidelity(a: QuantumCircuit, b: QuantumCircuit) -> float:
    return float(abs(np.vdot(amplitudes(a), amplitudes(b))) ** 2)


def _target(gates=("h", "cnot"), qubits: int = 2) -> Target:
    return Target(
        name="conformance-target",
        num_qubits=qubits,
        native_gates=gates,
        supports_measurement=True,
    )


def bell() -> QuantumCircuit:
    return QuantumCircuit(2).h(0).cnot(0, 1)


# ---------------------------------------------------------------------------
# Entry point & result shape
# ---------------------------------------------------------------------------


def test_compile_is_the_public_entry_point() -> None:
    assert isinstance(Compiler().compile(bell()), CompilationResult)


def test_compilation_result_shape() -> None:
    result = Compiler().compile(bell())
    assert isinstance(result.source, IRCircuit)
    assert isinstance(result.result, IRCircuit)
    assert result.passes_applied
    assert result.is_compatible
    assert result.circuit().num_qubits == 2


def test_compilation_result_serializes() -> None:
    result = Compiler().compile(bell())
    payload = json.loads(result.to_json())
    assert payload["result"]["num_qubits"] == 2
    assert payload["metadata"]["optimization_level"] == 1


# ---------------------------------------------------------------------------
# Optimization levels
# ---------------------------------------------------------------------------


def test_level_zero_is_validation_only() -> None:
    qc = QuantumCircuit(2).h(0).h(0).cnot(0, 1)
    result = Compiler(optimization_level=0).compile(qc)
    assert result.passes_applied == []
    assert result.result.operations == result.source.operations


def test_level_one_removes_identity_and_inverse() -> None:
    qc = QuantumCircuit(2).h(0).h(0).cnot(0, 1).cnot(0, 1)
    result = Compiler(optimization_level=1).compile(qc)
    assert result.result.num_gates == 0
    assert fidelity(qc, result.circuit()) > 1.0 - 1e-9


def test_level_two_fuses_same_axis_rotations() -> None:
    qc = QuantumCircuit(1).rx(0.3, 0).rx(0.4, 0)
    result = Compiler(optimization_level=2).compile(qc)
    assert result.result.gate_names() == {"rx": 1}
    assert fidelity(qc, result.circuit()) > 1.0 - 1e-9


@pytest.mark.parametrize("level", [-1, 3])
def test_invalid_levels_are_rejected(level: int) -> None:
    with pytest.raises(ValueError):
        Compiler(optimization_level=level)


def test_optimize_util_level_bounds() -> None:
    ir = to_ir(bell())
    with pytest.raises(ValueError):
        optimize(ir, 3)
    assert optimize(ir, 0) is ir


# ---------------------------------------------------------------------------
# Determinism, metrics, behavioral round trips
# ---------------------------------------------------------------------------


def test_compilation_is_deterministic() -> None:
    qc = QuantumCircuit(3).h(0).cnot(0, 2).rz(0.5, 1).rx(0.2, 2)
    a = Compiler(optimization_level=2).compile(qc)
    b = Compiler(optimization_level=2).compile(qc)
    assert a.to_dict() == b.to_dict()


def test_metadata_reports_depth_qubits_and_gate_counts() -> None:
    qc = QuantumCircuit(2).h(0).h(0).cnot(0, 1)
    meta = Compiler(optimization_level=1).compile(qc).metadata
    assert meta["source_qubits"] == 2
    assert meta["source_gates"] == 3
    assert meta["compiled_gates"] == 1
    assert meta["source_depth"] >= meta["compiled_depth"]
    assert meta["compiled_gates_by_type"] == {"cnot": 1}
    assert isinstance(meta["source_gates_by_type"], dict)


def test_depth_is_a_method_not_a_property() -> None:
    assert callable(QC.depth)
    compiled = Compiler().compile(bell()).circuit()
    assert isinstance(compiled.depth(), int)


def test_circuit_ir_circuit_roundtrip_preserves_behavior() -> None:
    qc = QuantumCircuit(2).ry(-0.9, 0).rz(1.4, 1).cnot(0, 1)
    rebuilt = from_ir(to_ir(qc))
    assert fidelity(qc, rebuilt) > 1.0 - 1e-9
    assert to_ir(rebuilt).gate_names() == qc.to_ir().gate_names()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_symbolic_rotations_are_never_simplified() -> None:
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
        assert result.result.gate_names() == {"rx": 2}


def test_compile_bind_commutes_with_bind_compile() -> None:
    theta = Parameter("theta")
    qc = QuantumCircuit(2).ry(theta, 0).cnot(0, 1)
    binding = {"theta": 0.5}
    pre = Compiler(optimization_level=2).compile(qc.bind_parameters(binding))
    post = Compiler(optimization_level=2).compile(qc).circuit().bind_parameters(binding)
    assert fidelity(pre.circuit(), post) > 1.0 - 1e-9


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def test_measurements_survive_compilation() -> None:
    qc = QuantumCircuit(3).x(1).measure(2).measure(0).measure(1)
    compiled = Compiler(optimization_level=1).compile(qc).circuit()
    assert compiled.measurements == [2, 0, 1]


def test_terminal_measurements_restored_on_rebuild() -> None:
    qc = bell().measure_all()
    result = Compiler(optimization_level=1).compile(qc)
    assert result.circuit().measurements == [0, 1]


# ---------------------------------------------------------------------------
# Target contract (soft diagnostics + cx/cnot normalization)
# ---------------------------------------------------------------------------


def test_supported_target_compiles_clean() -> None:
    result = Compiler().compile(bell(), target=_target())
    assert result.is_compatible
    assert result.diagnostics == []


def test_unsupported_gate_yields_diagnostic_not_silent_drop() -> None:
    qc = QuantumCircuit(1).t(0).h(0)
    result = Compiler().compile(qc, target=_target())
    assert not result.is_compatible
    assert "t" in result.result.gate_names()
    assert any("t" in d for d in result.diagnostics)


def test_cx_target_spelling_equals_ir_cnot() -> None:
    # Target advertises "cx"; IR/passes use "cnot".  Same gate.
    result = Compiler().compile(
        bell(), target=_target(gates=("h", "cx"))
    )
    assert result.is_compatible
    assert result.circuit().to_ir().gate_names() == {"h": 1, "cnot": 1}


def test_cz_decomposes_into_limited_basis() -> None:
    qc = QuantumCircuit(2).cz(0, 1)
    result = Compiler().compile(qc, target=_target(gates=("h", "cx")))
    assert result.is_compatible
    assert result.circuit().to_ir().gate_names() == {"h": 2, "cnot": 1}
    assert fidelity(qc, result.circuit()) > 1.0 - 1e-9


# ---------------------------------------------------------------------------
# Backend execution of compiled circuits
# ---------------------------------------------------------------------------


def test_compiled_circuit_executes_on_statevector() -> None:
    qc = QuantumCircuit(2).z(0).z(0).h(0).cnot(0, 1)
    compiled = Compiler(optimization_level=1).compile(qc).circuit()
    counts = StatevectorBackend().run(compiled, shots=1000, seed=42).counts
    assert counts.get("00", 0) + counts.get("11", 0) > 600


def test_compiled_circuit_executes_on_density_matrix() -> None:
    compiled = Compiler(optimization_level=1).compile(bell()).circuit()
    dm = DensityMatrixBackend().run(compiled, shots=None).density_matrix
    assert np.allclose(np.real(np.diag(dm)), [0.5, 0.0, 0.0, 0.5], atol=1e-9)


def test_original_and_compiled_states_agree_exactly() -> None:
    qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(1, 2)
    compiled = Compiler(optimization_level=2).compile(qc).circuit()
    assert fidelity(qc, compiled) > 1.0 - 1e-12


# ---------------------------------------------------------------------------
# QASM
# ---------------------------------------------------------------------------


def test_compiled_circuit_qasm_roundtrip() -> None:
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    compiled = Compiler(optimization_level=1).compile(qc).circuit()
    rebuilt = QuantumCircuit.from_qasm(compiled.qasm())
    assert fidelity(compiled, rebuilt) > 1.0 - 1e-9