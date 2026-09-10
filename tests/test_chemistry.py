"""Tests for quantum chemistry module."""
import pytest

from microquantum.chemistry.ansatz import HardwareEfficientAnsatz, UCCSDAnsatz
from microquantum.chemistry.hamiltonians import (
    H2Hamiltonian,
    LiHHamiltonian,
)
from microquantum.core.parameter import Parameter


class TestMolecularHamiltonian:
    def test_h2_creation(self):
        h = H2Hamiltonian()
        assert h.name == "H2"
        assert h.num_qubits == 2
        assert h.num_electrons == 2

    def test_h2_terms(self):
        h = H2Hamiltonian()
        assert h.num_terms > 0

    def test_h2_energy(self):
        h = H2Hamiltonian()
        assert h.ground_state_energy is not None
        assert h.ground_state_energy < 0

    def test_h2_custom_bond_length(self):
        h = H2Hamiltonian(bond_length=0.8)
        assert h.bond_length == 0.8

    def test_lih_creation(self):
        h = LiHHamiltonian()
        assert h.name == "LiH"
        assert h.num_qubits == 6
        assert h.num_electrons == 4

    def test_lih_terms(self):
        h = LiHHamiltonian()
        assert h.num_terms > 0

    def test_nuclear_repulsion(self):
        h = H2Hamiltonian()
        assert h.nuclear_repulsion > 0

    def test_repr(self):
        h = H2Hamiltonian()
        assert "H2" in repr(h)
        assert "2" in repr(h)


class TestHardwareEfficientAnsatz:
    def test_creation(self):
        ansatz = HardwareEfficientAnsatz(num_qubits=4, num_layers=2)
        assert ansatz.num_qubits == 4
        assert ansatz.num_parameters == 16  # 4 qubits * 2 params * 2 layers

    def test_invalid_qubits(self):
        with pytest.raises(ValueError, match="Need >= 1 qubit"):
            HardwareEfficientAnsatz(num_qubits=0)

    def test_invalid_layers(self):
        with pytest.raises(ValueError, match="Need >= 1 layer"):
            HardwareEfficientAnsatz(num_qubits=4, num_layers=0)

    def test_invalid_entangler(self):
        with pytest.raises(ValueError, match="entangler must be"):
            HardwareEfficientAnsatz(num_qubits=4, entangler="invalid")

    def test_build_circuit(self):
        ansatz = HardwareEfficientAnsatz(num_qubits=3, num_layers=2)
        qc = ansatz.build_circuit()
        assert qc.num_qubits == 3
        assert len(qc._gate_instructions) > 0

    def test_build_circuit_with_params(self):
        ansatz = HardwareEfficientAnsatz(num_qubits=3, num_layers=1)
        params = {Parameter("theta_0"): 0.5, Parameter("phi_0"): 1.0}
        # The build method should not crash with params
        qc = ansatz.build_circuit(params)
        assert qc.num_qubits == 3

    def test_cz_entangler(self):
        ansatz = HardwareEfficientAnsatz(num_qubits=3, num_layers=1, entangler="cz")
        qc = ansatz.build_circuit()
        names = [instr[0].name for instr in qc._gate_instructions
                 if not qc._is_parameterized_gate(instr)]
        assert "cz" in names

    def test_repr(self):
        ansatz = HardwareEfficientAnsatz(num_qubits=4, num_layers=2)
        assert "HardwareEfficientAnsatz" in repr(ansatz)


class TestUCCSDAnsatz:
    def test_creation(self):
        ansatz = UCCSDAnsatz(num_qubits=4, num_electrons=2)
        assert ansatz.num_qubits == 4
        assert ansatz.num_electrons == 2

    def test_invalid_qubits(self):
        with pytest.raises(ValueError, match="Need >= 2 qubits"):
            UCCSDAnsatz(num_qubits=1, num_electrons=1)

    def test_invalid_electrons(self):
        with pytest.raises(ValueError, match="Need >= 1 electron"):
            UCCSDAnsatz(num_qubits=4, num_electrons=0)

    def test_electrons_exceed_qubits(self):
        with pytest.raises(ValueError, match="More electrons"):
            UCCSDAnsatz(num_qubits=4, num_electrons=5)

    def test_excitations(self):
        ansatz = UCCSDAnsatz(num_qubits=4, num_electrons=2)
        assert ansatz.num_parameters > 0

    def test_build_circuit(self):
        ansatz = UCCSDAnsatz(num_qubits=4, num_electrons=2)
        qc = ansatz.build_circuit()
        assert qc.num_qubits == 4
        # Should have X gates for reference state
        names = [instr[0].name for instr in qc._gate_instructions
                 if not qc._is_parameterized_gate(instr)]
        assert "x" in names

    def test_build_circuit_with_params(self):
        ansatz = UCCSDAnsatz(num_qubits=4, num_electrons=2)
        params = {Parameter("uccsd_0"): 0.1}
        qc = ansatz.build_circuit(params)
        assert qc.num_qubits == 4

    def test_repr(self):
        ansatz = UCCSDAnsatz(num_qubits=4, num_electrons=2)
        assert "UCCSDAnsatz" in repr(ansatz)
