"""Tests for advanced noise channels: thermal relaxation, readout error, phase flip, pauli channel."""

import numpy as np
import pytest

from microquantum.backends.noise import NoiseModel
from microquantum.core.circuit import QuantumCircuit
from microquantum.core.density_matrix import DensityMatrix


class TestPhaseFlip:
    def test_no_error(self):
        noise = NoiseModel()
        noise.phase_flip(probability=0.0)
        rho = DensityMatrix.from_statevector(QuantumCircuit(1).h(0).run())
        result = noise.apply(rho)
        probs = np.real(np.diag(result.matrix))
        assert abs(probs[0] - 0.5) < 0.01
        assert abs(probs[1] - 0.5) < 0.01

    def test_full_error(self):
        noise = NoiseModel()
        noise.phase_flip(probability=1.0)
        rho = DensityMatrix.from_statevector(QuantumCircuit(1).h(0).run())
        result = noise.apply(rho)
        probs = np.real(np.diag(result.matrix))
        assert abs(probs[0] - 0.5) < 0.01
        assert abs(probs[1] - 0.5) < 0.01

    def test_partial_error(self):
        noise = NoiseModel()
        noise.phase_flip(probability=0.5)
        rho = DensityMatrix.from_statevector(QuantumCircuit(1).h(0).run())
        result = noise.apply(rho)
        assert result.trace == pytest.approx(1.0, abs=0.01)

    def test_specific_qubits(self):
        noise = NoiseModel()
        noise.phase_flip(probability=0.3, qubits=[0])
        qc = QuantumCircuit(2)
        qc.h(0)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        assert result.trace == pytest.approx(1.0, abs=0.01)

    def test_invalid_probability(self):
        noise = NoiseModel()
        with pytest.raises(ValueError):
            noise.phase_flip(probability=-0.1)
        with pytest.raises(ValueError):
            noise.phase_flip(probability=1.5)


class TestThermalRelaxation:
    def test_basic(self):
        noise = NoiseModel()
        noise.thermal_relaxation(t1=50.0, t2=70.0, gate_time=1.0)
        qc = QuantumCircuit(1)
        qc.h(0)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        assert result.trace == pytest.approx(1.0, abs=0.01)

    def test_long_t1_no_decay(self):
        noise = NoiseModel()
        noise.thermal_relaxation(t1=10000.0, t2=20000.0, gate_time=1.0)
        qc = QuantumCircuit(1)
        qc.h(0)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        probs = np.real(np.diag(result.matrix))
        assert abs(probs[0] - 0.5) < 0.05
        assert abs(probs[1] - 0.5) < 0.05

    def test_short_t1_decay(self):
        noise = NoiseModel()
        noise.thermal_relaxation(t1=2.0, t2=3.0, gate_time=1.0)
        qc = QuantumCircuit(1)
        qc.x(0)  # Start in |1>
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        probs = np.real(np.diag(result.matrix))
        # State should have changed from pure |1>
        assert probs[0] + probs[1] == pytest.approx(1.0, abs=0.01)
        assert probs[0] > 0.01  # Some population moved to |0>

    def test_t2_constraint(self):
        noise = NoiseModel()
        with pytest.raises(ValueError):
            noise.thermal_relaxation(t1=5.0, t2=11.0, gate_time=1.0)

    def test_gate_time_constraint(self):
        noise = NoiseModel()
        with pytest.raises(ValueError):
            noise.thermal_relaxation(t1=5.0, t2=8.0, gate_time=6.0)

    def test_specific_qubits(self):
        noise = NoiseModel()
        noise.thermal_relaxation(t1=50.0, t2=70.0, gate_time=1.0, qubits=[0])
        qc = QuantumCircuit(2)
        qc.h(0)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        assert result.trace == pytest.approx(1.0, abs=0.01)


class TestReadoutError:
    def test_no_error(self):
        noise = NoiseModel()
        noise.readout_error(prob_measured_0_when_actual_1=0.0, prob_measured_1_when_actual_0=0.0)
        qc = QuantumCircuit(1)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        probs = np.real(np.diag(result.matrix))
        assert abs(probs[0] - 1.0) < 0.01

    def test_with_error(self):
        noise = NoiseModel()
        noise.readout_error(prob_measured_0_when_actual_1=0.1, prob_measured_1_when_actual_0=0.1)
        qc = QuantumCircuit(1)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        assert result.trace == pytest.approx(1.0, abs=0.01)

    def test_invalid_probability(self):
        noise = NoiseModel()
        with pytest.raises(ValueError):
            noise.readout_error(prob_measured_0_when_actual_1=0.5, prob_measured_1_when_actual_0=1.5)

    def test_specific_qubits(self):
        noise = NoiseModel()
        noise.readout_error(prob_measured_0_when_actual_1=0.05, prob_measured_1_when_actual_0=0.05, qubits=[1])
        qc = QuantumCircuit(2)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        assert result.trace == pytest.approx(1.0, abs=0.01)


class TestPauliChannel:
    def test_identity_only(self):
        noise = NoiseModel()
        noise.pauli_channel(px=0.0, py=0.0, pz=0.0)
        qc = QuantumCircuit(1)
        qc.h(0)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        probs = np.real(np.diag(result.matrix))
        assert abs(probs[0] - 0.5) < 0.01

    def test_x_error(self):
        noise = NoiseModel()
        noise.pauli_channel(px=0.5, py=0.0, pz=0.0)
        qc = QuantumCircuit(1)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        assert result.trace == pytest.approx(1.0, abs=0.01)

    def test_z_error(self):
        noise = NoiseModel()
        noise.pauli_channel(px=0.0, py=0.0, pz=0.5)
        qc = QuantumCircuit(1)
        qc.h(0)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        probs = np.real(np.diag(result.matrix))
        assert abs(probs[0] - 0.5) < 0.1

    def test_total_exceeds_one(self):
        noise = NoiseModel()
        with pytest.raises(ValueError):
            noise.pauli_channel(px=0.5, py=0.5, pz=0.5)

    def test_negative_probability(self):
        noise = NoiseModel()
        with pytest.raises(ValueError):
            noise.pauli_channel(px=-0.1, py=0.0, pz=0.0)

    def test_specific_qubits(self):
        noise = NoiseModel()
        noise.pauli_channel(px=0.1, py=0.0, pz=0.1, qubits=[0])
        qc = QuantumCircuit(2)
        qc.h(0)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        assert result.trace == pytest.approx(1.0, abs=0.01)

    def test_multiple_channels(self):
        noise = NoiseModel()
        noise.phase_flip(probability=0.1)
        noise.depolarizing(probability=0.1)
        qc = QuantumCircuit(1)
        qc.h(0)
        rho = DensityMatrix.from_statevector(qc.run())
        result = noise.apply(rho)
        assert result.trace == pytest.approx(1.0, abs=0.01)
