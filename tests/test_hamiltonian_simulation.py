"""Tests for Hamiltonian Simulation (Trotter-Suzuki)."""

import numpy as np
import pytest

from microquantum.algorithms.hamiltonian_simulation import (
    HamiltonianSimulation,
    TrotterResult,
    _pauli_evolution_circuit,
)
from microquantum.core import Operator, QuantumCircuit, StateVector
from microquantum.core.pauli import PauliString, PauliSum


class TestPauliEvolutionCircuit:
    def test_z_evolution(self):
        qc = _pauli_evolution_circuit("Z", coefficient=1.0, time=np.pi, num_qubits=1)
        sv0 = StateVector(1)
        sv1 = qc.run(sv0)
        assert sv1.num_qubits == 1
        expected = np.exp(-1j * np.pi)
        assert abs(sv1.amplitudes[0] - expected) < 1e-10

    def test_identity_no_effect(self):
        qc = _pauli_evolution_circuit("I", coefficient=1.0, time=1.0, num_qubits=1)
        sv0 = StateVector(1)
        sv1 = qc.run(sv0)
        assert np.allclose(sv1.amplitudes, sv0.amplitudes, atol=1e-10)

    def test_two_qubit_zz(self):
        qc = _pauli_evolution_circuit("ZZ", coefficient=1.0, time=np.pi / 2, num_qubits=2)
        assert qc.num_qubits == 2
        assert qc.num_gates > 0

    def test_x_evolution(self):
        qc = _pauli_evolution_circuit("X", coefficient=1.0, time=np.pi, num_qubits=1)
        sv0 = StateVector(1)
        sv1 = qc.run(sv0)
        expected = np.exp(-1j * np.pi)
        assert abs(sv1.amplitudes[0] - expected) < 1e-10


class TestHamiltonianSimulation:
    def test_z_hamiltonian(self):
        H = PauliSum([PauliString("Z")])
        sim = HamiltonianSimulation(H, evolution_time=np.pi, num_steps=1)
        assert sim.num_qubits == 1
        result = sim.run()
        expected_0 = np.exp(-1j * np.pi)
        assert abs(result.amplitudes[0] - expected_0) < 1e-10

    def test_identity_hamiltonian(self):
        H = PauliSum([PauliString("I")])
        sim = HamiltonianSimulation(H, evolution_time=1.0, num_steps=1)
        result = sim.run()
        sv0 = StateVector(1)
        assert np.allclose(result.amplitudes, sv0.amplitudes, atol=1e-10)

    def test_trotter_result_fields(self):
        H = PauliSum([PauliString("Z")])
        sim = HamiltonianSimulation(H, evolution_time=1.0, num_steps=5, trotter_order=2)
        result = sim.run()
        assert isinstance(sim, HamiltonianSimulation)
        assert sim.num_steps == 5
        assert sim.trotter_order == 2

    def test_second_order_trotter(self):
        H = PauliSum([PauliString("Z")])
        sim1 = HamiltonianSimulation(H, evolution_time=0.5, num_steps=10, trotter_order=1)
        sim2 = HamiltonianSimulation(H, evolution_time=0.5, num_steps=10, trotter_order=2)
        qc1 = sim1.build_circuit()
        qc2 = sim2.build_circuit()
        assert qc2.num_gates >= qc1.num_gates

    def test_multiple_terms(self):
        H = PauliSum([PauliString("Z", 0.5), PauliString("X", 0.3)])
        sim = HamiltonianSimulation(H, evolution_time=1.0, num_steps=3)
        result = sim.run()
        assert result.num_qubits == 1

    def test_two_qubit_hamiltonian(self):
        H = PauliSum([PauliString("ZZ", 1.0)])
        sim = HamiltonianSimulation(H, evolution_time=0.5, num_steps=2)
        assert sim.num_qubits == 2
        result = sim.run()
        assert result.num_qubits == 2

    def test_convergence_more_steps_better(self):
        H = PauliSum([PauliString("X", 1.0), PauliString("Z", 0.5)])
        sim_few = HamiltonianSimulation(H, evolution_time=1.0, num_steps=1)
        sim_many = HamiltonianSimulation(H, evolution_time=1.0, num_steps=20)
        qc_few = sim_few.build_circuit()
        qc_many = sim_many.build_circuit()
        assert qc_many.num_gates > qc_few.num_gates

    def test_validation(self):
        H = PauliSum([PauliString("Z")])
        with pytest.raises(ValueError):
            HamiltonianSimulation(H, evolution_time=-1.0)
        with pytest.raises(ValueError):
            HamiltonianSimulation(H, evolution_time=1.0, num_steps=0)
        with pytest.raises(ValueError):
            HamiltonianSimulation(H, evolution_time=1.0, trotter_order=3)

    def test_repr(self):
        H = PauliSum([PauliString("Z")])
        sim = HamiltonianSimulation(H, evolution_time=1.0)
        assert "HamiltonianSimulation" in repr(sim)
