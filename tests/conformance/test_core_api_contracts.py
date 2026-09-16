"""Conformance - core public API contracts.

Systematically verifies the existing core surface: imports, construction,
public attributes, methods, method/property contracts (including explicit
collision detection), argument validation, return types and basic
integration.
"""

from __future__ import annotations

import numpy as np
import pytest

import microquantum as mq
from microquantum import (
    DensityMatrix,
    Operator,
    Parameter,
    QuantumCircuit,
    StateVector,
)


def _constructors() -> list[tuple[str, object]]:
    return [
        ("QuantumCircuit", QuantumCircuit(2)),
        ("StateVector", StateVector(2)),
        ("DensityMatrix", DensityMatrix(num_qubits=2, matrix=np.eye(4) / 4)),
        ("Operator", Operator(np.eye(2))),
        ("Parameter", Parameter("x")),
        ("ParameterExpression", 2 * Parameter("x")),
        ("PauliString", mq.PauliString("X")),
        ("PauliSum", mq.PauliSum([mq.PauliString("X", 0.5)])),
        ("MeasurementResult", mq.sample_state(QuantumCircuit(1).x(0).run(), seed=0)),
        ("BackendResult", mq.BackendResult(1, "b", counts={"0": 5, "1": 5})),
        ("QUBOBuilder", mq.QUBOBuilder(2)),
        ("ClassicalRegister", mq.ClassicalRegister(2)),
        ("QuantumRegister", mq.QuantumRegister("q", 2)),
        ("EigenvalueProblem", mq.EigenvalueProblem(Operator.Z(), k=1)),
        ("HamiltonianProblem", mq.HamiltonianProblem(Operator.Z())),
        ("OptimizationProblem", mq.OptimizationProblem.from_qubo(mq.QUBOBuilder(2).build())),
        ("SamplingProblem", mq.SamplingProblem(QuantumCircuit(1))),
        ("SearchProblem", mq.SearchProblem(name="s", num_qubits=3, target="101")),
        ("QUBOProblem", mq.QUBOBuilder(2).build()),
        ("IsingConverter", mq.IsingConverter()),
    ]


def test_all_public_names_importable() -> None:
    for name in mq.__all__:
        assert hasattr(mq, name), f"microquantum missing __all__ name {name!r}"
        obj = getattr(mq, name)
        assert obj is not None


def test_version_string() -> None:
    assert mq.__version__
    assert isinstance(mq.__version__, str)


def test_no_public_instance_attribute_shadows_method() -> None:
    """Regression protection: an instance attr must never shadow a method.

    The historical ``QuantumCircuit.depth`` regression (an int attribute
    replacing the ``depth()`` method) is the prototype for this check.
    """
    for name, obj in _constructors():
        cls = type(obj)
        methods = {n for n, v in vars(cls).items() if callable(v)}
        props = {n for n, v in vars(cls).items() if isinstance(v, property)}
        inst = set(vars(obj))
        for shadowed in inst & methods:
            raise AssertionError(
                f"{name}.{shadowed} is both an instance attribute and a "
                f"callable method (use a private name such as _shadowed)"
            )
        for shadowed in inst & props:
            raise AssertionError(
                f"{name}.{shadowed} is both an instance attribute and a property"
            )


# (class, instance-factory, documented-methods, documented-properties)
_CIRCUIT_CONTRACTS = (
    QuantumCircuit,
    lambda: QuantumCircuit(2),
    [
        "append", "append_parameterized", "h", "x", "y", "z", "s", "sdg",
        "t", "tdg", "rx", "ry", "rz", "cx", "cnot", "cz", "swap",
        "gate_count", "contains_gate", "inverse", "bind_parameters",
        "get_unitary", "run", "measure", "measure_all", "expectation_value",
        "to_ir", "from_ir", "qasm", "from_qasm", "draw", "to_json",
        "from_json", "save", "load", "depth",
    ],
    [
        "num_qubits", "gates", "num_gates", "parameters",
        "is_parameterized", "measurements", "qubits",
    ],
)


@pytest.mark.parametrize(
    "_cls,_factory,_methods,_props",
    [_CIRCUIT_CONTRACTS],
)
def test_quantum_circuit_method_property_contract(_cls, _factory, _methods, _props):
    qc = _factory()
    for m in _methods:
        assert callable(getattr(qc, m)), f"QuantumCircuit.{m} must be a method"
    for p in _props:
        value = getattr(qc, p)
        assert not callable(value), f"QuantumCircuit.{p} must be a property"


def test_quantum_circuit_depth_semantics() -> None:
    """Established depth semantics: longest-path scheduling per qubit."""
    assert QuantumCircuit(1).depth() == 0
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    assert qc.depth() == 2
    qc2 = QuantumCircuit(1)
    qc2.h(0)
    qc2.h(0)
    assert qc2.depth() == 2  # consecutive gates on a qubit each schedule a layer
    qc3 = QuantumCircuit(2)
    qc3.h(0)
    qc3.h(1)
    assert qc3.depth() == 1  # parallel gates share a layer
    qc4 = QuantumCircuit(2)
    qc4.h(0)
    qc4.cx(0, 1)
    qc4.h(0)
    assert qc4.depth() == 3


def test_quantum_circuit_construction_validation() -> None:
    with pytest.raises(ValueError):
        QuantumCircuit(0)
    with pytest.raises(ValueError):
        QuantumCircuit(-1)
    qc = QuantumCircuit(2)
    assert qc.num_qubits == 2
    assert qc.num_gates == 0
    with pytest.raises(ValueError):
        qc.h(2)


def test_core_types_construction_and_attributes() -> None:
    sv = StateVector(2)
    assert sv.num_qubits == 2
    assert sv.dim == 4
    assert sv.amplitudes.shape == (4,)
    norm = sv.normalize()
    assert np.sum(np.abs(norm.amplitudes) ** 2) == pytest.approx(1.0)

    dm = DensityMatrix(num_qubits=2, matrix=np.eye(4) / 4)
    assert dm.num_qubits == 2
    assert dm.dim == 4
    assert dm.matrix.shape == (4, 4)

    op = Operator(np.array([[0, 1], [1, 0]], dtype=complex))
    assert op.name == "custom"
    assert op.num_qubits == 1
    assert Operator.X().name == "x"
    assert Operator.Z().matrix.shape == (2, 2)

    p = Parameter("theta")
    assert p.name == "theta"
    expr = 2 * p + 0.5
    assert expr.gradient() == pytest.approx(2.0)
    assert expr.gradient(p) == pytest.approx(2.0)

    ps = mq.PauliString("XY", 0.5)
    assert ps.label == "XY"
    assert ps.coefficient == 0.5
    assert ps.num_qubits == 2
    psum = mq.PauliSum([mq.PauliString("X"), mq.PauliString("Z", -0.5)])
    assert psum.num_terms == 2
    assert psum.terms[0].label == "X"


def test_sample_and_measurement_functions() -> None:
    qc = QuantumCircuit(1)
    qc.x(0)
    state = qc.run()
    result = mq.sample_state(state, shots=1000, seed=42)
    assert result.get_counts()["1"] == 1000
    assert result.shots == 1000

    q0, collapsed = mq.measure_and_collapse(state, [0], seed=42)
    assert int(q0) == 1
    assert collapsed.num_qubits == 1

    bits = mq.measure_qubits(state, [0], shots=1000, seed=42)
    assert bits.get_counts()["1"] == 1000


def test_transpiler_public_api() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(0)
    stats = mq.circuit_stats(qc)
    assert stats["gate_count"] == 4
    assert stats["depth"] == 4

    simplified = mq.simplify_circuit(qc)
    assert simplified.num_gates <= qc.num_gates
    protonly = mq.transpile(qc)
    assert isinstance(protonly, QuantumCircuit)
    assert callable(mq.PassManager().run)


def test_parameter_expression_arithmetic() -> None:
    p = Parameter("p")
    assert (3 * p).coefficient == 3
    assert (-p).coefficient == -1
    total = (2 * p) + (1 * p)
    assert total.coefficient == pytest.approx(3.0)
    assert total.gradient(p) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        _ = (2 * p) + (2 * Parameter("q"))


def test_random_circuit_toolchain() -> None:
    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).x(2).ry(1.0, 1)
    unitary = qc.get_unitary()
    assert unitary.matrix.shape == (8, 8)
    state = qc.run()
    assert state.num_qubits == 3
    assert state.amplitudes.shape == (8,)
    assert np.sum(np.abs(state.amplitudes) ** 2) == pytest.approx(1.0)