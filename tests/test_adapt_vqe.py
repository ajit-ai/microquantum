"""Tests for ADAPT-VQE."""

import numpy as np
import pytest

from microquantum.algorithms.adapt_vqe import AdaptResult, AdaptVQE
from microquantum.core.pauli import PauliString, PauliSum


class TestAdaptVQE:
    def test_single_qubit_x_ground_state(self):
        H = PauliSum([PauliString("X", -1.0)])
        adapt = AdaptVQE(H, max_layers=5, gradient_threshold=1e-3)
        result = adapt.run()
        assert isinstance(result, AdaptResult)
        assert result.energy <= 0.0

    def test_convergence(self):
        H = PauliSum([PauliString("X", -1.0)])
        adapt = AdaptVQE(H, max_layers=10, gradient_threshold=1e-3)
        result = adapt.run()
        assert result.energy == pytest.approx(-1.0, abs=0.15)

    def test_energy_history(self):
        H = PauliSum([PauliString("X", -1.0)])
        adapt = AdaptVQE(H, max_layers=5, gradient_threshold=1e-3)
        result = adapt.run()
        assert len(result.energy_history) >= 1

    def test_build_default_pool(self):
        H = PauliSum([PauliString("ZZ")])
        adapt = AdaptVQE(H)
        pool = adapt.build_default_pool()
        assert len(pool) > 0

    def test_two_qubit_hamiltonian(self):
        H = PauliSum([PauliString("IZ", 0.5), PauliString("ZZ", 0.5)])
        adapt = AdaptVQE(H, max_layers=5, gradient_threshold=1e-3)
        result = adapt.run()
        assert isinstance(result, AdaptResult)

    def test_custom_pool(self):
        H = PauliSum([PauliString("X", -1.0)])
        pool = [PauliString("X"), PauliString("Y"), PauliString("Z")]
        adapt = AdaptVQE(H, pool=pool, max_layers=5, gradient_threshold=1e-3)
        result = adapt.run()
        assert result.num_layers >= 1

    def test_validation_empty_hamiltonian(self):
        with pytest.raises(ValueError):
            AdaptVQE(PauliSum())

    def test_properties(self):
        H = PauliSum([PauliString("Z")])
        adapt = AdaptVQE(H, max_layers=5)
        assert adapt.num_qubits == 1
        assert adapt.max_layers == 5

    def test_repr(self):
        H = PauliSum([PauliString("Z")])
        adapt = AdaptVQE(H)
        assert "AdaptVQE" in repr(adapt)

    def test_result_dataclass_fields(self):
        H = PauliSum([PauliString("X", -1.0)])
        adapt = AdaptVQE(H, max_layers=5, gradient_threshold=1e-3)
        result = adapt.run()
        assert hasattr(result, "circuit")
        assert hasattr(result, "energy")
        assert hasattr(result, "state")
        assert hasattr(result, "num_layers")
        assert hasattr(result, "operator_indices")
        assert hasattr(result, "converged")
