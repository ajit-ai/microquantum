"""Tests for amplitude estimation algorithm."""


import pytest

from microquantum.algorithms.amplitude_estimation import (
    AmplitudeEstimation,
    AmplitudeEstimationResult,
)
from microquantum.core.circuit import QuantumCircuit
from microquantum.core.operators import Operator


class TestAmplitudeEstimationResult:
    def test_default(self):
        result = AmplitudeEstimationResult()
        assert result.estimated_amplitude == 0.0
        assert result.confidence_interval == 0.0
        assert result.num_evaluations == 0
        assert result.phases == []


class TestAmplitudeEstimation:
    def test_init(self):
        ae = AmplitudeEstimation(num_evaluation_qubits=3)
        assert ae.num_evaluation_qubits == 3

    def test_state_prep_and_oracle(self):
        # Simple 1-qubit: state_prep = H, oracle = Z (marks |1>)
        n = 1
        sp = QuantumCircuit(n)
        sp.h(0)

        oracle = QuantumCircuit(n)
        oracle.append(Operator.Z(), [0])

        ae = AmplitudeEstimation(num_evaluation_qubits=3)
        grover = ae.build_grover_operator(n, sp, oracle)
        assert grover.num_qubits == n
        assert grover.num_gates > 0

    def test_estimate_known_amplitude(self):
        # State: H|0> = (|0> + |1>)/sqrt(2)
        # Oracle marks |1>, so a = 0.5
        n = 1
        sp = QuantumCircuit(n)
        sp.h(0)

        oracle = QuantumCircuit(n)
        oracle.append(Operator.Z(), [0])

        ae = AmplitudeEstimation(num_evaluation_qubits=4)
        result = ae.estimate(n, sp, oracle)

        assert 0.0 <= result.estimated_amplitude <= 1.0
        assert result.num_evaluations == 16

    def test_grover_operator_requires_circuits(self):
        ae = AmplitudeEstimation(num_evaluation_qubits=3)
        with pytest.raises(ValueError, match="required"):
            ae.build_grover_operator(2)

    def test_repr(self):
        ae = AmplitudeEstimation(num_evaluation_qubits=4)
        assert "AmplitudeEstimation" in repr(ae)
        assert "4" in repr(ae)


def _diagonal_oracle(num_qubits: int, targets: list[int]) -> QuantumCircuit:
    """Phase oracle marking *targets* (Grover convention)."""
    import numpy as np

    diagonal = np.ones(2**num_qubits, dtype=complex)
    for target in targets:
        diagonal[target] = -1.0
    oracle = QuantumCircuit(num_qubits)
    oracle.append(Operator(np.diag(diagonal), name="oracle"), list(range(num_qubits)))
    return oracle


def _uniform_preparation(num_qubits: int) -> QuantumCircuit:
    """Uniform superposition state preparation."""
    preparation = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        preparation.h(qubit)
    return preparation


class TestAmplitudeEstimationAccuracy:
    """QPE accuracy: exact fractions resolve exactly, others land near."""

    def test_half_fraction_exact(self) -> None:
        ae = AmplitudeEstimation(num_evaluation_qubits=4)
        result = ae.estimate(1, _uniform_preparation(1), _diagonal_oracle(1, [1]))
        assert result.estimated_amplitude == pytest.approx(0.5)
        assert result.phases[0] == pytest.approx(0.25)

    def test_empty_and_full_exact(self) -> None:
        ae = AmplitudeEstimation(num_evaluation_qubits=3)
        assert ae.estimate(2, _uniform_preparation(2), _diagonal_oracle(2, [])).estimated_amplitude == pytest.approx(0.0)
        full = ae.estimate(2, _uniform_preparation(2), _diagonal_oracle(2, [0, 1, 2, 3]))
        assert full.estimated_amplitude == pytest.approx(1.0)

    def test_quarter_fraction_nearBin(self) -> None:
        ae = AmplitudeEstimation(num_evaluation_qubits=5)
        result = ae.estimate(2, _uniform_preparation(2), _diagonal_oracle(2, [3]))
        assert result.estimated_amplitude == pytest.approx(0.25, abs=0.03)

    def test_controlled_unitary_validation(self) -> None:
        import numpy as np

        with pytest.raises(ValueError, match="square"):
            AmplitudeEstimation._controlled_unitary(np.ones((2, 3)), 0, [1], 2)
        with pytest.raises(ValueError, match="dimension"):
            AmplitudeEstimation._controlled_unitary(np.eye(2), 0, [1, 2], 3)
        with pytest.raises(ValueError, match="control"):
            AmplitudeEstimation._controlled_unitary(np.eye(4), 5, [0, 1], 3)
        with pytest.raises(ValueError, match="must not be a target"):
            AmplitudeEstimation._controlled_unitary(np.eye(4), 0, [0, 1], 3)
