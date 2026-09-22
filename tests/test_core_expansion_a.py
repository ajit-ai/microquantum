"""Core expansion tests, part A: model, math and measurement layers.

Covers ``core.circuit``, ``core.gates``, ``core.registers``,
``core.operators``, ``core.pauli``, ``core.observables``,
``core.states``, ``core.tensor`` and ``core.measurements``: normal
behavior, mathematical correctness and invalid input.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from microquantum.core.architecture import linear_architecture
from microquantum.core.channels import amplitude_damping, depolarizing
from microquantum.core.circuit import (
    QuantumCircuit,
    iter_instructions,
    validate_circuit,
)
from microquantum.core.density_matrix import DensityMatrix
from microquantum.core.execution import ExecutionOptions, ExecutionRequest, StateVectorExecutor
from microquantum.core.gates import (
    CX,
    RX,
    RY,
    RZ,
    ControlledGate,
    H,
    ParameterizedGate,
    Phase,
    Toffoli,
    U,
    UnitaryGate,
    gate_matrix,
    make_gate,
)
from microquantum.core.gradients import GradientEngine
from microquantum.core.information import (
    fidelity,
    shannon_entropy,
    trace_distance,
    von_neumann_entropy,
)
from microquantum.core.measurements import (
    POVM,
    ProjectiveMeasurement,
    computational_basis_measurement,
)
from microquantum.core.observables import (
    MatrixObservable,
    PauliObservable,
    SumObservable,
    observable_from_pauli_sum,
)
from microquantum.core.operators import (
    HermitianOperator,
    LinearOperator,
    Projector,
    UnitaryOperator,
)
from microquantum.core.parameters import (
    Cos,
    Parameter,
    ParameterBinding,
    ParameterVector,
    Sin,
    discover_parameters,
)
from microquantum.core.pauli import (
    PauliString,
    PauliSum,
    anticommutes,
    commutes,
    multiply_labels,
)
from microquantum.core.registers import (
    ClassicalRegister,
    QuantumRegister,
    register_from_dict,
    register_to_dict,
)
from microquantum.core.resources import estimate_resources
from microquantum.core.serialization import (
    deserialize_circuit,
    deserialize_observable,
    deserialize_state,
    serialize_circuit,
    serialize_observable,
    serialize_state,
)
from microquantum.core.state import StateVector
from microquantum.core.states import (
    is_normalized,
    partial_trace,
    probabilities,
    state_fidelity,
    state_purity,
)
from microquantum.core.tensor import (
    kron,
    partial_trace_matrix,
    permute_qubits,
    subsystem_probabilities,
)
from microquantum.core.transpiler import (
    BasisTranslationPass,
    PassContext,
    ValidationPass,
    default_pipeline,
    transpile_with,
)

# ----------------------------------------------------------------------
# gates
# ----------------------------------------------------------------------


class TestGateHierarchy:
    def test_single_qubit_unitarity(self) -> None:
        for factory in (H,):
            mat = factory().to_matrix()
            assert np.allclose(mat.conj().T @ mat, np.eye(2), atol=1e-12)

    def test_rotation_matrices(self) -> None:
        assert np.allclose(RX(0.0).to_matrix(), np.eye(2), atol=1e-12)
        assert np.allclose(RY(math.pi).to_matrix(), [[0, -1], [1, 0]], atol=1e-12)
        assert np.allclose(RZ(0.0).to_matrix(), np.eye(2), atol=1e-12)

    def test_two_qubit_and_three_qubit_shapes(self) -> None:
        assert CX().to_matrix().shape == (4, 4)
        assert Toffoli().to_matrix().shape == (8, 8)
        assert U(0.1, 0.2, 0.3).num_qubits == 1

    def test_inverse_and_controlled(self) -> None:
        h = H()
        assert np.allclose(h.inverse().to_matrix(), h.to_matrix(), atol=1e-12)
        assert isinstance(h.controlled(), ControlledGate)
        assert h.controlled().num_qubits == 2
        assert CX().inverse().name == "cx"

    def test_symbolic_gate_requires_binding(self) -> None:
        theta = Parameter("theta")
        gate = RX(theta)
        assert gate.is_parameterized
        with pytest.raises(ValueError):
            gate.to_matrix()
        bound = gate.bind({"theta": 0.5})
        assert np.allclose(bound.to_matrix(), gate_matrix("rx", (0.5,)), atol=1e-12)

    def test_make_gate_factory(self) -> None:
        assert make_gate("h").name == "h"
        assert make_gate("rx", (0.1,)).num_qubits == 1
        with pytest.raises(ValueError):
            make_gate("not-a-gate")

    def test_invalid_gate_inputs(self) -> None:
        with pytest.raises(ValueError):
            UnitaryGate(np.eye(3))
        with pytest.raises(ValueError):
            ParameterizedGate("rx", 0, (0.1,))
        with pytest.raises(ValueError):
            ControlledGate(H(), 0)
        with pytest.raises(ValueError):
            Phase("not-a-number").to_matrix()


# ----------------------------------------------------------------------
# circuit
# ----------------------------------------------------------------------


class TestCircuitPackage:
    def test_construction_and_properties(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert qc.num_qubits == 2
        assert qc.num_gates == 2
        assert qc.depth() == 2

    def test_validation_rejects_bad_indices(self) -> None:
        qc = QuantumCircuit(1)
        with pytest.raises(ValueError):
            qc.h(5)
        with pytest.raises(TypeError):
            validate_circuit("not-a-circuit")  # type: ignore[arg-type]

    def test_iteration_and_metadata(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure(0)
        instructions = list(iter_instructions(qc))
        assert instructions[0].operation == "h"
        assert instructions[-1].operation == "measure"

    def test_composition_and_inverse(self) -> None:
        a = QuantumCircuit(1)
        a.h(0)
        b = QuantumCircuit(1)
        b.x(0)
        both = a + b
        assert both.num_gates == 2
        assert (a + a.inverse()).get_unitary().matrix.shape == (2, 2)


# ----------------------------------------------------------------------
# registers
# ----------------------------------------------------------------------


class TestRegistersPackage:
    def test_legacy_registers_preserved(self) -> None:
        qr = QuantumRegister("q", 3)
        assert qr.size == 3
        assert qr[1] == 1
        assert list(qr) == [0, 1, 2]

    def test_register_serialization(self) -> None:
        cr = ClassicalRegister("c", 2)
        assert register_from_dict(register_to_dict(cr)).size == 2
        with pytest.raises(ValueError):
            register_from_dict({"kind": "bogus", "name": "x", "size": 1})

    def test_invalid_sizes(self) -> None:
        with pytest.raises(ValueError):
            QuantumRegister("q", 0)
        with pytest.raises(IndexError):
            QuantumRegister("q", 2)[7]


# ----------------------------------------------------------------------
# operators
# ----------------------------------------------------------------------


class TestOperatorsPackage:
    def test_linear_operator_algebra(self) -> None:
        a = LinearOperator(np.eye(2))
        b = LinearOperator(2 * np.eye(2))
        assert (a.compose(b)).matrix[0, 0] == pytest.approx(2.0)
        assert a.tensor(b).dim == 4
        with pytest.raises(ValueError):
            LinearOperator(np.eye(3))
        with pytest.raises(ValueError):
            a.compose(LinearOperator(np.eye(4)))

    def test_unitary_validation_and_inverse(self) -> None:
        u = UnitaryOperator(H().to_matrix())
        assert (u.inverse().compose(u)) == UnitaryOperator(np.eye(2))
        with pytest.raises(ValueError):
            UnitaryOperator(np.array([[1, 1], [0, 1]], dtype=complex))

    def test_hermitian_and_projector(self) -> None:
        herm = HermitianOperator(np.array([[1, 0], [0, -1]], dtype=complex))
        assert herm.is_hermitian
        assert list(herm.eigenvalues()) == pytest.approx([-1.0, 1.0])
        proj = Projector.zero_state(2)
        assert proj.rank == 1
        with pytest.raises(ValueError):
            HermitianOperator(np.array([[0, 1], [0, 0]], dtype=complex))
        with pytest.raises(ValueError):
            Projector(np.eye(2) / 2)


# ----------------------------------------------------------------------
# pauli
# ----------------------------------------------------------------------


class TestPauliPackage:
    def test_multiplication_with_phase(self) -> None:
        assert multiply_labels("X", "Y") == ("Z", 1j)
        assert multiply_labels("XY", "YX") == ("ZZ", 1)
        with pytest.raises(ValueError):
            multiply_labels("X", "XY")
        with pytest.raises(ValueError):
            multiply_labels("X", "Q")

    def test_commutation_rules(self) -> None:
        assert commutes(PauliString("XX"), PauliString("YY"))
        assert anticommutes(PauliString("X"), PauliString("Z"))
        assert not anticommutes(PauliString("XI"), PauliString("IX"))
        with pytest.raises(ValueError):
            commutes(PauliString("X"), PauliString("XX"))

    def test_pauli_sum_expectation(self) -> None:
        state = StateVector(1, amplitudes=np.array([1, 0], dtype=complex))
        total = PauliSum([PauliString("Z", 0.5), PauliString("X", 0.5)])
        assert total.expectation(state) == pytest.approx(0.5)


# ----------------------------------------------------------------------
# observables
# ----------------------------------------------------------------------


class TestObservablesPackage:
    def test_pauli_observable_no_dense_matrix(self) -> None:
        obs = PauliObservable("Z")
        state = StateVector(1, amplitudes=np.array([0, 1], dtype=complex))
        assert obs.expectation(state) == pytest.approx(-1.0)
        assert obs.variance(state) == pytest.approx(0.0)
        with pytest.raises(ValueError):
            PauliObservable("Z", coefficient=1j)

    def test_sum_and_matrix_observables(self) -> None:
        total = PauliObservable("Z") + PauliObservable("Z")
        assert isinstance(total, SumObservable)
        state = StateVector(1, amplitudes=np.array([1, 0], dtype=complex))
        assert total.expectation(state) == pytest.approx(2.0)
        mat_obs = MatrixObservable(np.array([[1, 0], [0, -1]], dtype=complex))
        assert mat_obs.expectation(state) == pytest.approx(1.0)
        assert mat_obs.commutes_with(PauliObservable("Z"))
        assert not mat_obs.commutes_with(PauliObservable("X"))
        with pytest.raises(ValueError):
            MatrixObservable(np.array([[0, 1], [0, 0]], dtype=complex))

    def test_observable_from_pauli_sum(self) -> None:
        psum = PauliSum([PauliString("ZZ", 1.0)])
        obs = observable_from_pauli_sum(psum)
        assert obs.num_qubits == 2
        with pytest.raises(ValueError):
            observable_from_pauli_sum(PauliSum([]))


# ----------------------------------------------------------------------
# states & tensor
# ----------------------------------------------------------------------


class TestStatesPackage:
    def test_bell_partial_trace_is_maximally_mixed(self) -> None:
        bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        reduced = partial_trace(bell, [0])
        assert np.allclose(reduced, np.eye(2) / 2, atol=1e-12)

    def test_fidelity_purity_probabilities(self) -> None:
        state = StateVector(1, amplitudes=np.array([1, 0], dtype=complex))
        assert state_fidelity(state, state) == pytest.approx(1.0)
        assert state_purity(state) == pytest.approx(1.0)
        assert probabilities(state)[0] == pytest.approx(1.0)
        assert is_normalized(state)
        mixed = DensityMatrix(1, np.eye(2, dtype=complex) / 2)
        assert state_purity(mixed) == pytest.approx(0.5)
        with pytest.raises(ValueError):
            partial_trace(state, [5])
        with pytest.raises(ValueError):
            partial_trace(state, [])

    def test_tensor_utilities(self) -> None:
        assert kron(np.eye(2), np.eye(2)).shape == (4, 4)
        vec = np.array([1, 0, 0, 0], dtype=complex)
        swapped = permute_qubits(np.array([0, 1, 0, 0], dtype=complex), [1, 0])
        assert swapped[1] == pytest.approx(0.0)
        assert swapped[2] == pytest.approx(1.0)
        assert vec is not None
        marginal = subsystem_probabilities(
            np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2), [0], 2
        )
        assert marginal["0"] == pytest.approx(0.5)
        with pytest.raises(ValueError):
            permute_qubits(vec, [0, 0])
        rho = np.outer(
            np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2),
            np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2),
        ).conj()
        assert np.allclose(partial_trace_matrix(rho, [1], 2), np.eye(2) / 2, atol=1e-12)


# ----------------------------------------------------------------------
# measurements
# ----------------------------------------------------------------------


class TestMeasurementsPackage:
    def test_computational_basis(self) -> None:
        measurement = computational_basis_measurement(1)
        state = StateVector(1, amplitudes=np.array([1, 0], dtype=complex))
        outcomes = measurement.probabilities(state)
        assert outcomes[0].probability == pytest.approx(1.0)
        result = measurement.sample(state, shots=50, seed=0)
        assert result.get_counts() == {"0": 50}

    def test_projective_post_state_and_povm(self) -> None:
        measurement = computational_basis_measurement(1)
        state = StateVector(1, amplitudes=np.array([0, 1], dtype=complex))
        collapsed = measurement.post_measurement_state(state, "1")
        assert np.allclose(np.abs(collapsed), [0, 1], atol=1e-12)
        effects = [np.eye(2, dtype=complex) / 2, np.eye(2, dtype=complex) / 2]
        povm = POVM(effects, labels=["a", "b"])
        probs = povm.probabilities(state)
        assert [p.probability for p in probs] == pytest.approx([0.5, 0.5])
        with pytest.raises(ValueError):
            ProjectiveMeasurement([np.array([[0, 1], [0, 0]], dtype=complex)])
        with pytest.raises(ValueError):
            POVM([np.zeros((2, 2), dtype=complex)])
        with pytest.raises(ValueError):
            measurement.sample(state, shots=0)


# ----------------------------------------------------------------------
# cross-module integration (part A)
# ----------------------------------------------------------------------


class TestCoreIntegrationA:
    def test_parameter_to_measurement_chain(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        bound = qc.bind_parameters({"theta": math.pi})
        executor = StateVectorExecutor()
        result = executor.run(
            ExecutionRequest(circuit=bound, options=ExecutionOptions(shots=64, seed=1))
        )
        assert sum(result.get_counts().values()) == 64

    def test_pauli_sum_to_gradient(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        engine = GradientEngine()
        grad = engine.compute(qc, PauliString("Z"), {"theta": 0.2})
        assert grad["theta"] == pytest.approx(-math.sin(0.2), abs=1e-6)

    def test_channel_to_measurement_result(self) -> None:
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        noisy = depolarizing(0.2)(rho)
        assert abs(float(np.real(np.trace(noisy))) - 1.0) < 1e-9
        assert float(np.real(noisy[0, 0])) < 1.0

    def test_resources_and_transpiler(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        resources = estimate_resources(qc)
        assert resources.total_gates == 3
        optimized = transpile_with(qc)
        assert optimized.num_gates <= 2
        arch = linear_architecture(3)
        routed = transpile_with(qc, architecture=arch)
        assert routed.num_gates >= 1

    def test_serialization_round_trip(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(theta, 1)
        qc.cx(0, 1)
        rebuilt = deserialize_circuit(serialize_circuit(qc))
        assert rebuilt.num_qubits == 2
        assert [p.name for p in rebuilt.parameters] == ["theta"]
        state = StateVector(1, amplitudes=np.array([0, 1], dtype=complex))
        assert np.allclose(deserialize_state(serialize_state(state)).amplitudes, state.amplitudes)
        obs = PauliObservable("Z")
        assert deserialize_observable(serialize_observable(obs)).label == "Z"

    def test_information_and_parameters(self) -> None:
        rho = np.eye(2, dtype=complex) / 2
        assert von_neumann_entropy(rho) == pytest.approx(1.0)
        assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)
        assert trace_distance(rho, rho) == pytest.approx(0.0)
        assert fidelity(rho, rho) == pytest.approx(1.0)
        vector = ParameterVector("phi", 2)
        binding = ParameterBinding({"phi[0]": 0.1, "phi[1]": 0.2})
        assert binding["phi[0]"] == pytest.approx(0.1)
        assert len(discover_parameters([vector[0], vector[1]])) == 2
        assert abs(complex(Sin(Parameter("t")).evaluate({"t": 0.0}))) < 1e-12
        assert abs(complex(Cos(Parameter("t")).evaluate({"t": 0.0})) - 1.0) < 1e-12
        assert BasisTranslationPass().name == "basis-translation"
        ctx = PassContext()
        ValidationPass().analyze(QuantumCircuit(1), ctx)
        assert ctx.analysis["valid"] is True
        assert amplitude_damping(0.0).is_trace_preserving()
        assert default_pipeline().num_passes == 6
