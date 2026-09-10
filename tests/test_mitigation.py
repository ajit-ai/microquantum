"""Tests for error mitigation module."""
import numpy as np
import pytest

from microquantum.backends.noise import NoiseModel
from microquantum.core.circuit import QuantumCircuit
from microquantum.mitigation.mem import MeasurementErrorMitigation
from microquantum.mitigation.pec import ProbabilisticErrorCancellation
from microquantum.mitigation.zne import ExtrapolationResult, ZeroNoiseExtrapolation


class TestZeroNoiseExtrapolation:
    def test_creation(self):
        zne = ZeroNoiseExtrapolation()
        assert zne.noise_factors == [1.0, 3.0, 5.0]

    def test_custom_factors(self):
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 4.0])
        assert zne.noise_factors == [1.0, 2.0, 4.0]

    def test_invalid_factors(self):
        with pytest.raises(ValueError, match="at least 2"):
            ZeroNoiseExtrapolation(noise_factors=[1.0])

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            ZeroNoiseExtrapolation(method="invalid")

    def test_fold_circuit_factor1(self):
        zne = ZeroNoiseExtrapolation()
        qc = QuantumCircuit(2)
        qc.h(0)
        folded = zne.fold_circuit(qc, 1)
        assert len(folded._gate_instructions) == len(qc._gate_instructions)

    def test_fold_circuit_factor3(self):
        zne = ZeroNoiseExtrapolation()
        qc = QuantumCircuit(1)
        qc.h(0)
        folded = zne.fold_circuit(qc, 3)
        # Factor 3: G G^dag G = 3 gates
        assert len(folded._gate_instructions) == 3

    def test_extrapolate_linear(self):
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 3.0, 5.0], method="linear")
        result = zne.extrapolate([0.8, 0.6, 0.4])
        assert isinstance(result, ExtrapolationResult)
        assert result.method == "linear"
        assert result.mitigated_value > 0.4

    def test_extrapolate_perfect_linear(self):
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 3.0], method="linear")
        # Linear: y = 1 - 0.1*x -> at x=0, y=1
        result = zne.extrapolate([0.9, 0.7])
        assert result.mitigated_value == pytest.approx(1.0, abs=0.01)

    def test_extrapolate_richardson(self):
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 3.0, 5.0], method="richardson")
        result = zne.extrapolate([0.9, 0.7, 0.5])
        assert isinstance(result, ExtrapolationResult)

    def test_extrapolate_wrong_length(self):
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 3.0, 5.0])
        with pytest.raises(ValueError, match="Expected 3 values"):
            zne.extrapolate([0.8, 0.6])

    def test_repr(self):
        zne = ZeroNoiseExtrapolation()
        assert "ZeroNoiseExtrapolation" in repr(zne)


class TestProbabilisticErrorCancellation:
    def test_creation(self):
        noise = NoiseModel()
        noise.depolarizing(0.1)
        pec = ProbabilisticErrorCancellation(noise)
        assert pec.precision == 0.01

    def test_quasi_probabilities(self):
        noise = NoiseModel()
        noise.depolarizing(0.1)
        pec = ProbabilisticErrorCancellation(noise)
        channel = np.eye(4, dtype=complex)
        probs, gamma = pec.compute_quasi_probabilities(channel)
        assert len(probs) == 4
        assert gamma > 0

    def test_mitigate_expectation(self):
        noise = NoiseModel()
        noise.depolarizing(0.1)
        pec = ProbabilisticErrorCancellation(noise)
        noisy_vals = [0.9, 0.85, 0.8]
        result = pec.mitigate_expectation(noisy_vals)
        assert isinstance(result, float)

    def test_mitigate_empty(self):
        noise = NoiseModel()
        pec = ProbabilisticErrorCancellation(noise)
        result = pec.mitigate_expectation([])
        assert result == 0.0

    def test_repr(self):
        noise = NoiseModel()
        pec = ProbabilisticErrorCancellation(noise)
        assert "ProbabilisticErrorCancellation" in repr(pec)


class TestMeasurementErrorMitigation:
    def test_creation(self):
        mem = MeasurementErrorMitigation(2)
        assert mem.num_qubits == 2
        assert not mem.is_calibrated

    def test_invalid_qubits(self):
        with pytest.raises(ValueError, match="Need >= 1 qubit"):
            MeasurementErrorMitigation(0)

    def test_calibrate(self):
        mem = MeasurementErrorMitigation(2, shots=100)
        result = mem.calibrate()
        assert mem.is_calibrated
        assert result.confusion.shape == (4, 4)
        assert result.num_qubits == 2

    def test_mitigate_counts(self):
        mem = MeasurementErrorMitigation(1, shots=100)
        mem.calibrate()
        raw = {"0": 400, "1": 100}
        mitigated = mem.mitigate_counts(raw)
        assert isinstance(mitigated, dict)
        assert sum(mitigated.values()) == pytest.approx(1.0, abs=1e-6)

    def test_mitigate_counts_not_calibrated(self):
        mem = MeasurementErrorMitigation(1)
        with pytest.raises(RuntimeError, match="Must calibrate"):
            mem.mitigate_counts({"0": 100, "1": 0})

    def test_mitigate_expectation(self):
        mem = MeasurementErrorMitigation(1, shots=100)
        mem.calibrate()
        raw = {"0": 400, "1": 100}
        obs = np.array([[1, 0], [0, -1]])
        result = mem.mitigate_expectation(raw, obs)
        assert isinstance(result, float)

    def test_repr(self):
        mem = MeasurementErrorMitigation(2)
        assert "MeasurementErrorMitigation" in repr(mem)
        assert "not calibrated" in repr(mem)

    def test_repr_calibrated(self):
        mem = MeasurementErrorMitigation(1, shots=100)
        mem.calibrate()
        assert "calibrated" in repr(mem)
