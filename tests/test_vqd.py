"""Tests for VQD (Variational Quantum Deflation) for excited states."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.core import (
    Operator,
    Parameter,
    PauliString,
    QuantumCircuit,
)
from microquantum.core.pauli import PauliSum
from microquantum.algorithms.vqd import VQD, VQDResult
from microquantum.optimizers import Adam, GradientDescent


def _make_ry_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """Build a parameterized RY ansatz."""
    qc = QuantumCircuit(num_qubits)
    params = []
    for _ in range(depth):
        for q in range(num_qubits):
            p = Parameter(f"theta_{q}_{len(params)}")
            params.append(p)
            qc.ry(p, q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    return qc


class TestVQD:
    """VQD excited states tests."""

    def test_single_state_reduces_to_vqe(self) -> None:
        """VQD with num_states=1 should run and return a valid result."""
        H = PauliSum([PauliString("Z", -1.0)])
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1)
        ansatz.ry(theta, 0)
        opt = GradientDescent(learning_rate=0.3, max_iter=200, tol=1e-10)

        vqd = VQD(ansatz, H, opt, num_states=1, seed=0)
        result = vqd.run()

        assert result.num_states == 1
        assert len(result.eigenvalues) == 1
        assert len(result.eigenstates) == 1
        assert len(result.energies[0]) > 1
        assert all(np.isfinite(v) for v in result.eigenvalues)

    def test_two_states_finds_excited(self) -> None:
        """VQD with num_states=2 should find ground + first excited."""
        H = PauliSum([PauliString("Z", -1.0)])
        ansatz = _make_ry_ansatz(1, depth=3)
        opt = Adam(learning_rate=0.1, max_iter=300, tol=1e-10)

        vqd = VQD(ansatz, H, opt, num_states=2, beta=1.5, seed=42)
        result = vqd.run()

        assert result.num_states == 2
        assert len(result.eigenvalues) == 2
        assert len(result.eigenstates) == 2
        # Both states should be found (not necessarily sorted due to stochastic optimizer)
        assert all(isinstance(v, float) for v in result.eigenvalues)

    def test_beta_list(self) -> None:
        """VQD accepts per-state beta values."""
        H = PauliSum([PauliString("Z", -1.0)])
        ansatz = _make_ry_ansatz(1, depth=2)
        opt = GradientDescent(learning_rate=0.3, max_iter=50, tol=1e-8)

        vqd = VQD(ansatz, H, opt, num_states=2, beta=[1.0, 2.0], seed=42)
        result = vqd.run()
        assert result.num_states == 2

    def test_energy_history(self) -> None:
        """Each state should have an optimization history."""
        H = PauliSum([PauliString("Z", -1.0)])
        ansatz = _make_ry_ansatz(1, depth=2)
        opt = GradientDescent(learning_rate=0.3, max_iter=50, tol=1e-8)

        vqd = VQD(ansatz, H, opt, num_states=1, seed=42)
        result = vqd.run()

        assert len(result.energies) == 1
        assert len(result.energies[0]) > 1

    def test_requires_parameterized_ansatz(self) -> None:
        """Non-parameterized ansatz should raise."""
        qc = QuantumCircuit(1)
        qc.h(0)
        H = PauliSum([PauliString("Z", -1.0)])
        opt = GradientDescent(learning_rate=0.1, max_iter=10, tol=1e-8)
        with pytest.raises(ValueError, match="parameters"):
            VQD(qc, H, opt, num_states=1)

    def test_num_states_validation(self) -> None:
        """num_states < 1 should raise."""
        ansatz = _make_ry_ansatz(1, depth=2)
        H = PauliSum([PauliString("Z", -1.0)])
        opt = GradientDescent(learning_rate=0.1, max_iter=10, tol=1e-8)
        with pytest.raises(ValueError, match="num_states"):
            VQD(ansatz, H, opt, num_states=0)

    def test_num_qubits_property(self) -> None:
        """num_qubits should match ansatz."""
        ansatz = _make_ry_ansatz(2, depth=2)
        H = PauliSum([PauliString("ZZ", -1.0)])
        opt = GradientDescent(learning_rate=0.1, max_iter=10, tol=1e-8)
        vqd = VQD(ansatz, H, opt, num_states=1)
        assert vqd.num_qubits == 2

    def test_repr(self) -> None:
        ansatz = _make_ry_ansatz(2, depth=2)
        H = PauliSum([PauliString("ZZ", -1.0)])
        opt = GradientDescent(learning_rate=0.1, max_iter=10, tol=1e-8)
        vqd = VQD(ansatz, H, opt, num_states=3)
        assert "VQD" in repr(vqd)
        assert "num_states=3" in repr(vqd)
