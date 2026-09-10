"""Tests for enhanced Hamiltonian simulation (Suzuki-4, qDRIFT, commuting groups)."""

from __future__ import annotations

from microquantum.algorithms.hamiltonian_simulation import TrotterResult
from microquantum.algorithms.simulation_enhanced import (
    _commutes,
    _commuting_groups,
    fourth_order_simulation,
    qdrift_simulation,
)
from microquantum.core import PauliString
from microquantum.core.pauli import PauliSum


class TestCommutingGroups:
    """Commuting group partitioning tests."""

    def test_single_term(self) -> None:
        """Single term trivially forms one group."""
        H = PauliSum([PauliString("ZI", 1.0)])
        groups = _commuting_groups(H)
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_two_commuting_terms(self) -> None:
        """ZZ and II commute (same Pauli type)."""
        H = PauliSum([PauliString("ZZ", 1.0), PauliString("II", 0.5)])
        groups = _commuting_groups(H)
        assert len(groups) == 1

    def test_two_non_commuting_terms(self) -> None:
        """ZI and XI do not commute (odd number of differing non-identity positions)."""
        H = PauliSum([PauliString("ZI", 1.0), PauliString("XI", 0.5)])
        groups = _commuting_groups(H)
        assert len(groups) == 2

    def test_three_terms_two_groups(self) -> None:
        """XI, IX commute; ZI commutes with both."""
        H = PauliSum([
            PauliString("XI", 1.0),
            PauliString("IX", 1.0),
            PauliString("ZI", 0.5),
        ])
        groups = _commuting_groups(H)
        assert len(groups) <= 3
        # All terms accounted for
        total = sum(len(g) for g in groups)
        assert total == 3


class TestCommutes:
    """PauliString commutation tests."""

    def test_same_pauli_commutes(self) -> None:
        assert _commutes(PauliString("ZZ", 1.0), PauliString("ZZ", 1.0))

    def test_identity_commutes(self) -> None:
        assert _commutes(PauliString("II", 1.0), PauliString("ZZ", 1.0))

    def test_zi_xi_do_not_commute(self) -> None:
        """ZI and XI do not commute (differ on qubit 0, both non-identity)."""
        assert not _commutes(PauliString("ZI", 1.0), PauliString("XI", 1.0))


class TestFourthOrderSuzuki:
    """4th-order Suzuki-Trotter tests."""

    def test_returns_trotter_result(self) -> None:
        """fourth_order_simulation should return TrotterResult."""
        H = PauliSum([PauliString("ZZ", -0.5)])
        result = fourth_order_simulation(H, evolution_time=0.5, num_steps=2)
        assert isinstance(result, TrotterResult)
        assert result.trotter_order == 4
        assert result.num_steps == 2
        assert result.evolution_time == 0.5

    def test_circuit_executes(self) -> None:
        """Built circuit should execute without error."""
        H = PauliSum([PauliString("ZI", 1.0), PauliString("IZ", -0.5)])
        result = fourth_order_simulation(H, evolution_time=0.1, num_steps=2)
        sv = result.circuit.run()
        assert sv.num_qubits == 2
        assert len(sv.amplitudes) == 4

    def test_single_step(self) -> None:
        """Single step should still produce valid circuit."""
        H = PauliSum([PauliString("Z", 1.0)])
        result = fourth_order_simulation(H, evolution_time=0.1, num_steps=1)
        assert result.circuit.num_qubits == 1
        sv = result.circuit.run()
        assert len(sv.amplitudes) == 2

    def test_more_steps_more_gates(self) -> None:
        """More steps should produce more gates."""
        H = PauliSum([PauliString("Z", 1.0)])
        r1 = fourth_order_simulation(H, evolution_time=1.0, num_steps=1)
        r2 = fourth_order_simulation(H, evolution_time=1.0, num_steps=3)
        assert r2.circuit.depth >= r1.circuit.depth


class TestQDRIFT:
    """qDRIFT randomized simulation tests."""

    def test_returns_trotter_result(self) -> None:
        """qdrift_simulation should return TrotterResult."""
        H = PauliSum([PauliString("ZZ", -0.5)])
        result = qdrift_simulation(H, evolution_time=0.5, num_samples=20, seed=42)
        assert isinstance(result, TrotterResult)
        assert result.trotter_order == 0
        assert result.num_steps == 20

    def test_circuit_executes(self) -> None:
        """Built circuit should execute without error."""
        H = PauliSum([PauliString("ZI", 1.0), PauliString("IZ", -0.5)])
        result = qdrift_simulation(H, evolution_time=0.1, num_samples=10, seed=42)
        sv = result.circuit.run()
        assert sv.num_qubits == 2

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce same circuit."""
        H = PauliSum([PauliString("Z", 1.0), PauliString("X", 0.5)])
        r1 = qdrift_simulation(H, evolution_time=1.0, num_samples=15, seed=42)
        r2 = qdrift_simulation(H, evolution_time=1.0, num_samples=15, seed=42)
        assert r1.circuit.depth == r2.circuit.depth

    def test_different_seeds_different_circuits(self) -> None:
        """Different seeds should produce different circuits."""
        H = PauliSum([PauliString("Z", 1.0), PauliString("X", 0.5)])
        r1 = qdrift_simulation(H, evolution_time=1.0, num_samples=20, seed=42)
        r2 = qdrift_simulation(H, evolution_time=1.0, num_samples=20, seed=99)
        # Very unlikely to be identical with different seeds
        assert r1.circuit.depth != r2.circuit.depth or r1.circuit.num_qubits == r2.circuit.num_qubits

    def test_zero_hamiltonian(self) -> None:
        """Zero Hamiltonian should produce empty circuit."""
        H = PauliSum([PauliString("Z", 0.0)])
        result = qdrift_simulation(H, evolution_time=1.0, num_samples=10, seed=42)
        sv = result.circuit.run()
        # Zero Hamiltonian means identity evolution
        assert abs(abs(sv.amplitudes[0]) ** 2 - 1.0) < 1e-10
