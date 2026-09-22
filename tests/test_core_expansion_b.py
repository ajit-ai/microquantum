"""Core expansion tests, part B: parameters, channels, information,
execution, architecture, resources, gradients, serialization and
transpiler — behavior, math correctness and invalid input.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from microquantum.core.architecture import (
    NativeGateSet,
    QuantumArchitecture,
    QubitTopology,
    fully_connected_architecture,
    linear_architecture,
)
from microquantum.core.channels import (
    KrausChannel,
    amplitude_damping,
    bit_flip,
    bit_phase_flip,
    depolarizing,
    phase_damping,
    phase_flip,
)
from microquantum.core.circuit import QuantumCircuit
from microquantum.core.coupling import CouplingMap
from microquantum.core.execution import (
    ExecutionContext,
    ExecutionOptions,
    ExecutionRequest,
    ExecutionResult,
    Executor,
    StateVectorExecutor,
)
from microquantum.core.gates import CRX, RX, RY, RZ, SWAP, CPhase, Fredkin, X, Z
from microquantum.core.gradients import (
    AnalyticGradient,
    FiniteDifferenceGradient,
    GradientEngine,
    GradientResult,
    ParameterShiftGradient,
)
from microquantum.core.information import (
    fidelity,
    mutual_information,
    purity,
    relative_entropy,
    state_overlap,
    trace_distance,
    von_neumann_entropy,
)
from microquantum.core.operators import Operator
from microquantum.core.parameters import (
    Parameter,
    ParameterBinding,
    ParameterExpression,
    ParameterVector,
    bind_all,
    ordered_parameters,
)
from microquantum.core.pauli import pauli_matrices
from microquantum.core.resources import CircuitResources, estimate_state_memory_bytes
from microquantum.core.serialization import (
    deserialize_gate,
    deserialize_parameters,
    deserialize_state,
    dumps_canonical,
    serialize_gate,
    serialize_parameters,
    serialize_result,
    serialize_state,
)
from microquantum.core.transpiler import (
    AdjacentCancellationPass,
    AnalysisPass,
    BasisTranslationPass,
    GateFusionPass,
    InverseCancellationPass,
    LayoutPass,
    PassContext,
    RoutingPass,
    SchedulingPass,
    TargetGateSet,
    TransformationPass,
    transpile_with,
)


class TestParametersPackage:
    def test_vector_and_binding(self) -> None:
        vector = ParameterVector("beta", 2)
        assert vector.size == 2
        assert vector[0].name == "beta[0]"
        binding = ParameterBinding({"a": 1.5})
        assert binding["a"] == pytest.approx(1.5)
        merged = binding.merge(ParameterBinding({"b": 2.0}))
        assert merged.names == ("a", "b")
        assert ParameterBinding.from_dict(binding.to_dict())["a"] == pytest.approx(1.5)
        with pytest.raises(KeyError):
            binding["missing"]
        with pytest.raises(ValueError):
            ParameterVector("x", 0)
        with pytest.raises(ValueError):
            ParameterBinding({"": 1.0})

    def test_ordering_and_bind_all(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        assert [p.name for p in ordered_parameters([phi, theta])] == ["phi", "theta"]
        assert bind_all(theta * 2, {"theta": 0.5}) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            bind_all(theta, {})

    def test_expression_discovery(self) -> None:
        theta = Parameter("theta")
        expr = theta * 2 + 0.5
        assert isinstance(expr, ParameterExpression)
        assert expr.evaluate({"theta": 1.0}) == pytest.approx(2.5)


class TestChannelsPackage:
    def test_standard_channels_trace_preserving(self) -> None:
        for factory in (
            lambda: bit_flip(0.1),
            lambda: phase_flip(0.2),
            lambda: bit_phase_flip(0.3),
            lambda: depolarizing(0.1),
            lambda: amplitude_damping(0.4),
            lambda: phase_damping(0.4),
        ):
            channel = factory()
            assert channel.num_qubits == 1
            assert channel.is_trace_preserving()
            channel.validate()

    def test_invalid_probabilities_rejected(self) -> None:
        for factory in (bit_flip, phase_flip, bit_phase_flip, depolarizing):
            with pytest.raises(ValueError):
                factory(1.5)
        with pytest.raises(ValueError):
            amplitude_damping(-0.1)

    def test_application_and_composition(self) -> None:
        ground = np.array([[1, 0], [0, 0]], dtype=complex)
        excited = np.array([[0, 0], [0, 1]], dtype=complex)
        assert np.allclose(bit_flip(1.0)(ground), excited, atol=1e-12)
        assert np.allclose(amplitude_damping(1.0)(excited), ground, atol=1e-12)
        assert abs(float(np.real(np.trace(depolarizing(0.5)(ground)))) - 1.0) < 1e-12
        composed = bit_flip(0.1).compose(phase_flip(0.1))
        assert composed.num_qubits == 1 and composed.is_trace_preserving()
        tensored = bit_flip(0.1).tensor(phase_flip(0.1))
        assert tensored.num_qubits == 2
        with pytest.raises(ValueError):
            KrausChannel([])
        with pytest.raises(ValueError):
            KrausChannel([np.eye(2, dtype=complex)], num_qubits=2)
        with pytest.raises(ValueError):
            KrausChannel([np.eye(2, dtype=complex), np.zeros((4, 4), dtype=complex)])
        with pytest.raises(ValueError):
            bit_flip(0.1).apply(np.eye(4, dtype=complex))


class TestInformationPackage:
    def test_fidelity_and_distance_bounds(self) -> None:
        zero = np.array([[1, 0], [0, 0]], dtype=complex)
        one = np.array([[0, 0], [0, 1]], dtype=complex)
        assert fidelity(zero, one) == pytest.approx(0.0)
        assert trace_distance(zero, one) == pytest.approx(1.0)
        assert purity(np.eye(2, dtype=complex) / 2) == pytest.approx(0.5)
        assert state_overlap(zero, zero) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            fidelity(zero, np.eye(4, dtype=complex) / 4)

    def test_entropies(self) -> None:
        assert von_neumann_entropy(np.eye(2, dtype=complex) / 2) == pytest.approx(1.0)
        assert von_neumann_entropy(np.array([[1, 0], [0, 0]], dtype=complex)) == pytest.approx(0.0)
        assert relative_entropy(
            np.array([[1, 0], [0, 0]], dtype=complex), np.eye(2, dtype=complex) / 2
        ) == pytest.approx(1.0)
        bell = np.outer(
            np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2),
            np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2),
        ).conj()
        assert mutual_information(bell, [0], 2) == pytest.approx(2.0, abs=1e-9)
        with pytest.raises(ValueError):
            von_neumann_entropy(np.eye(2, dtype=complex) / 2, base=1.0)
        with pytest.raises(ValueError):
            mutual_information(np.eye(4, dtype=complex) / 4, [], 2)


class TestExecutionPackage:
    def test_statevector_executor_counts(self) -> None:
        qc = QuantumCircuit(1)
        qc.x(0)
        result = StateVectorExecutor().run(
            ExecutionRequest(circuit=qc, options=ExecutionOptions(shots=32, seed=42))
        )
        assert result.get_counts() == {"1": 32}
        assert result.shots == 32 and result.seed == 42

    def test_state_option_and_context(self) -> None:
        qc = QuantumCircuit(1)
        result = StateVectorExecutor().run_circuit(
            qc, ExecutionOptions(shots=8, seed=0, want_state=True)
        )
        assert result.has_state
        assert result.metadata["executor"] == "statevector"
        assert StateVectorExecutor().name == "statevector"
        assert ExecutionContext(backend_name="demo").backend_name == "demo"
        empty = ExecutionResult()
        assert not empty.has_counts and not empty.has_state
        with pytest.raises(ValueError):
            empty.get_counts()
        with pytest.raises(ValueError):
            ExecutionOptions(shots=0)

    def test_parameterized_request_binding(self) -> None:
        theta = Parameter("t")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        request = ExecutionRequest(circuit=qc, parameters={"t": math.pi})
        result = StateVectorExecutor().run(request)
        assert sum(result.get_counts().values()) == 1024

    def test_executor_abc(self) -> None:
        with pytest.raises(TypeError):
            Executor()  # type: ignore[abstract]


class TestArchitecturePackage:
    def test_topologies(self) -> None:
        linear = linear_architecture(4)
        assert linear.num_qubits == 4
        assert linear.requires_routing((0, 3))
        assert not linear.requires_routing((1, 2))
        full = fully_connected_architecture(3)
        assert not full.requires_routing((0, 2))
        assert full.is_native("cx")
        assert not full.is_native("ccx")
        with pytest.raises(ValueError):
            linear.validate_circuit_qubits(5)
        with pytest.raises(ValueError):
            QubitTopology(2, ((0, 0),))
        with pytest.raises(ValueError):
            QubitTopology(2, ((0, 5),))

    def test_serialization_and_coupling(self) -> None:
        arch = linear_architecture(2)
        rebuilt = QuantumArchitecture.from_dict(arch.to_dict())
        assert rebuilt.num_qubits == 2
        assert isinstance(arch.topology.to_coupling_map(), CouplingMap)
        gates = NativeGateSet()
        assert "cx" in gates.basis_gates and gates.supports("CX")
        with pytest.raises(ValueError):
            NativeGateSet(gate_errors={"cx": -0.1})


class TestResourcesPackage:
    def test_memory_estimates(self) -> None:
        assert estimate_state_memory_bytes(1) == 32
        assert estimate_state_memory_bytes(2, density_matrix=True) == 256
        with pytest.raises(ValueError):
            estimate_state_memory_bytes(0)

    def test_circuit_resources_type(self) -> None:
        resources = CircuitResources(num_qubits=2, depth=3, total_gates=4)
        assert resources.to_dict()["num_qubits"] == 2


class TestGradientsPackage:
    def _circuit(self) -> QuantumCircuit:
        theta = Parameter("g")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        return qc

    def test_all_methods_agree(self) -> None:
        qc = self._circuit()
        observable = Operator(np.array([[1, 0], [0, -1]], dtype=complex))
        expected = -math.sin(0.4)
        engine = GradientEngine()
        assert engine.compute(qc, observable, {"g": 0.4})["g"] == pytest.approx(expected, abs=1e-6)
        assert engine.compute(qc, observable, {"g": 0.4}, method="finite-difference")[
            "g"
        ] == pytest.approx(expected, abs=1e-4)
        assert engine.compute(qc, observable, {"g": 0.4}, method="analytic")[
            "g"
        ] == pytest.approx(expected, abs=1e-6)
        assert set(engine.available_methods) == {"parameter-shift", "finite-difference", "analytic"}

    def test_result_and_method_objects(self) -> None:
        result = GradientResult((0.5,), ("g",), method="finite-difference")
        assert result.as_dict() == {"g": 0.5}
        assert ParameterShiftGradient().name == "parameter-shift"
        assert FiniteDifferenceGradient().name == "finite-difference"
        assert AnalyticGradient().name == "analytic"
        with pytest.raises(ValueError):
            GradientResult((0.5,), ("a", "b"))
        with pytest.raises(ValueError):
            FiniteDifferenceGradient(epsilon=0.0)
        with pytest.raises(ValueError):
            GradientEngine(method="bogus")
        with pytest.raises(ValueError):
            ParameterShiftGradient(shift=math.pi)

    def test_custom_method_registration(self) -> None:
        engine = GradientEngine()
        engine.register(FiniteDifferenceGradient(epsilon=1e-4))
        assert "finite-difference" in engine.available_methods


class TestSerializationPackage:
    def test_gate_and_parameter_round_trips(self) -> None:
        gate = CRX(0.25)
        assert deserialize_gate(serialize_gate(gate)).name == "crx"
        symbolic = RX(Parameter("s"))
        assert deserialize_gate(serialize_gate(symbolic)).is_parameterized
        values = deserialize_parameters(serialize_parameters({"a": 1.0}))
        assert values["a"] == pytest.approx(1.0)
        assert dumps_canonical({"z": [1]}) == '{"z":[1]}'
        with pytest.raises(ValueError):
            deserialize_parameters({"type": "circuit", "version": 1, "payload": {}})

    def test_result_serialization(self) -> None:
        result = StateVectorExecutor().run_circuit(QuantumCircuit(1).h(0))
        envelope = serialize_result(result)
        assert envelope["type"] == "result"
        assert set(envelope["payload"]["counts"]) <= {"0", "1"}
        with pytest.raises(TypeError):
            serialize_result(object())

    def test_density_matrix_state_round_trip(self) -> None:
        from microquantum.core.density_matrix import DensityMatrix

        rho = DensityMatrix(1, np.eye(2, dtype=complex) / 2)
        rebuilt = deserialize_state(serialize_state(rho))
        assert np.allclose(rebuilt.matrix, rho.matrix, atol=1e-12)


class TestTranspilerPackage:
    def test_individual_passes(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.x(1)
        context = PassContext()
        out = InverseCancellationPass().transform(qc, context)
        assert out.num_gates == 1
        assert context.analysis["cancelled_pairs"] == 1
        assert AdjacentCancellationPass is InverseCancellationPass
        fused = GateFusionPass().transform(qc, PassContext())
        assert fused.num_gates == 2
        translated = BasisTranslationPass().transform(qc, PassContext())
        assert translated.num_gates >= 1
        scheduled = SchedulingPass().run(qc)
        assert scheduled.num_gates == 3
        laid_out = LayoutPass().transform(qc, PassContext(architecture=linear_architecture(2)))
        assert laid_out.num_qubits == 2

    def test_pass_base_classes(self) -> None:
        assert issubclass(InverseCancellationPass, TransformationPass)
        assert issubclass(SchedulingPass, AnalysisPass)
        with pytest.raises(TypeError):
            AnalysisPass()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            TransformationPass()  # type: ignore[abstract]

    def test_routing_with_architecture(self) -> None:
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        routed = transpile_with(qc, architecture=linear_architecture(3))
        assert routed.num_qubits == 3
        assert TargetGateSet.default().basis_gates >= {"cx"}
        assert RoutingPass(CouplingMap([(0, 1), (1, 2)])).name == "routing"

    def test_gates_beyond_core_library(self) -> None:
        assert SWAP().to_matrix().shape == (4, 4)
        assert Fredkin().to_matrix().shape == (8, 8)
        assert CPhase(0.1).num_qubits == 2
        assert X().to_matrix()[0, 1] == pytest.approx(1.0)
        assert Z().to_matrix()[1, 1] == pytest.approx(-1.0)
        assert RY(0.0).name == "ry" and RZ(0.0).name == "rz"
        assert pauli_matrices()["X"][0, 1] == pytest.approx(1.0)
