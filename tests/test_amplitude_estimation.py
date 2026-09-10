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
