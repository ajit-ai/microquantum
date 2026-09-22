"""Core quantum computing primitives for microquantum."""

# Expanded Core subpackages (backend-independent quantum foundation).
# Each subpackage has a deliberate public API (``__all__``); the names
# re-exported here are additive and preserve the stable 1.0.0 surface.
from .architecture import NativeGateSet, QuantumArchitecture, QubitTopology
from .channels import KrausChannel, QuantumChannel
from .circuit import (
    CircuitInstruction,
    CircuitMetadata,
    QuantumCircuit,
    iter_instructions,
    validate_circuit,
)
from .coupling import CouplingMap
from .device import Device, DeviceType, Target
from .dynamic import ClassicalRegister, DynamicCircuit, DynamicCircuitResult
from .engine import apply_gate
from .execution import ExecutionOptions, ExecutionRequest, ExecutionResult
from .gates import (
    CompositeGate,
    ControlledGate,
    ParameterizedGate,
    UnitaryGate,
)
from .gradient import gradient, parameter_shift_gradient
from .gradients import GradientEngine, GradientResult
from .information import fidelity, trace_distance, von_neumann_entropy
from .measurement import (
    MeasurementResult,
    expectation_value,
    measure_and_collapse,
    measure_qubits,
    sample_state,
)
from .measurements import POVM, ProjectiveMeasurement
from .observables import MatrixObservable, Observable, PauliObservable, SumObservable
from .operators import HermitianOperator, LinearOperator, Operator, Projector, UnitaryOperator
from .optimization import (
    cancel_inverse_pairs,
    circuit_stats,
    fuse_single_qubit_gates,
    remove_identity_gates,
    simplify_circuit,
    transpile,
)
from .parameter import Parameter, ParameterExpression
from .parameters import ParameterBinding, ParameterVector
from .pauli import PauliString, PauliSum, anticommutes, commutes
from .registers import ClassicalRegister as ClassicalRegisterBase
from .registers import Clbit, QuantumRegister, Qubit, Register
from .resources import CircuitResources, ResourceEstimate, ResourceEstimator, estimate_resources
from .serialization import deserialize_circuit, serialize_circuit
from .state import StateVector
from .states import State
from .tensor import expand_operator, tensor
from .transpiler import (
    AnalysisPass,
    Pass,
    PassContext,
    PassManager,
    TargetGateSet,
    TransformationPass,
)

__all__ = [
    "QuantumCircuit",
    "CouplingMap",
    "StateVector",
    "apply_gate",
    "Operator",
    "tensor",
    "expand_operator",
    "MeasurementResult",
    "sample_state",
    "measure_qubits",
    "measure_and_collapse",
    "expectation_value",
    "Parameter",
    "ParameterExpression",
    "parameter_shift_gradient",
    "gradient",
    "PauliString",
    "PauliSum",
    "QuantumRegister",
    "ClassicalRegisterBase",
    "fuse_single_qubit_gates",
    "remove_identity_gates",
    "cancel_inverse_pairs",
    "simplify_circuit",
    "transpile",
    "circuit_stats",
    # Transpiler
    "Pass",
    "PassManager",
    "TargetGateSet",
    # Dynamic circuits
    "DynamicCircuit",
    "DynamicCircuitResult",
    "ClassicalRegister",
    # Resources
    "ResourceEstimator",
    "ResourceEstimate",
    # Device & target contracts
    "Device",
    "DeviceType",
    "Target",
    # Expanded Core foundations (additive, backend-independent)
    "NativeGateSet",
    "QuantumArchitecture",
    "QubitTopology",
    "KrausChannel",
    "QuantumChannel",
    "CircuitInstruction",
    "CircuitMetadata",
    "iter_instructions",
    "validate_circuit",
    "ExecutionOptions",
    "ExecutionRequest",
    "ExecutionResult",
    "ControlledGate",
    "CompositeGate",
    "ParameterizedGate",
    "UnitaryGate",
    "GradientEngine",
    "GradientResult",
    "fidelity",
    "trace_distance",
    "von_neumann_entropy",
    "POVM",
    "ProjectiveMeasurement",
    "MatrixObservable",
    "Observable",
    "PauliObservable",
    "SumObservable",
    "HermitianOperator",
    "LinearOperator",
    "Projector",
    "UnitaryOperator",
    "ParameterBinding",
    "ParameterVector",
    "anticommutes",
    "commutes",
    "Clbit",
    "Qubit",
    "Register",
    "CircuitResources",
    "estimate_resources",
    "deserialize_circuit",
    "serialize_circuit",
    "State",
    "AnalysisPass",
    "PassContext",
    "TransformationPass",
]
