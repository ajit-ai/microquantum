"""Tests for QAOA algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.algorithms import QAOA, VQEResult
from microquantum.core import Operator, tensor
from microquantum.optimizers import Adam


class TestQAOA:
    """QAOA on small graph problems."""

    def test_maxcut_two_nodes(self) -> None:
        """Max-Cut on a 2-node graph with 1 edge.

        H_C = (1/2)(I - Z0 Z1).
        Ground state energy = 0.0 (degenerate |01> and |10>).

        Must start from non-zero parameters because at gamma=beta=0
        the gradient is zero (|++> is a fixed point).
        """
        ZZ = tensor(Operator.Z(), Operator.Z())
        II = tensor(Operator.I(), Operator.I())
        H_C = 0.5 * (II - ZZ)

        exact = np.linalg.eigvalsh(H_C.matrix)
        ground_energy = float(exact[0])

        optimizer = Adam(learning_rate=0.1, max_iter=500, tol=1e-8)
        qaoa = QAOA(
            cost_hamiltonian=H_C,
            num_qubits=2,
            num_layers=1,
            optimizer=optimizer,
        )
        result = qaoa.solve(initial_gamma=[0.5], initial_beta=[0.5])

        assert result.eigenvalue == pytest.approx(ground_energy, abs=1e-2)

    def test_qaoa_returns_vqe_result(self) -> None:
        """QAOA.solve() returns a VQEResult."""
        ZZ = tensor(Operator.Z(), Operator.Z())
        II = tensor(Operator.I(), Operator.I())
        H_C = 0.5 * (II - ZZ)

        qaoa = QAOA(cost_hamiltonian=H_C, num_qubits=2, num_layers=1)
        result = qaoa.solve(initial_gamma=[0.5], initial_beta=[0.5])

        assert isinstance(result, VQEResult)
        assert result.optimizer_result is not None
        assert len(result.optimizer_result.history) > 0

    def test_qaoa_ansatz_is_parameterized(self) -> None:
        """QAOA ansatz contains gamma and beta parameters."""
        ZZ = tensor(Operator.Z(), Operator.Z())
        II = tensor(Operator.I(), Operator.I())
        H_C = 0.5 * (II - ZZ)

        qaoa = QAOA(cost_hamiltonian=H_C, num_qubits=2, num_layers=2)
        ansatz = qaoa.build_ansatz()

        assert ansatz.is_parameterized
        param_names = {p.name for p in ansatz.parameters}
        assert "gamma_0" in param_names
        assert "beta_0" in param_names
        assert "gamma_1" in param_names
        assert "beta_1" in param_names

    def test_qaoa_two_layers(self) -> None:
        """2-layer QAOA on 2-node max-cut."""
        ZZ = tensor(Operator.Z(), Operator.Z())
        II = tensor(Operator.I(), Operator.I())
        H_C = 0.5 * (II - ZZ)

        exact = np.linalg.eigvalsh(H_C.matrix)
        ground_energy = float(exact[0])

        optimizer = Adam(learning_rate=0.1, max_iter=500, tol=1e-8)
        qaoa = QAOA(
            cost_hamiltonian=H_C,
            num_qubits=2,
            num_layers=2,
            optimizer=optimizer,
        )
        result = qaoa.solve(initial_gamma=[0.5, 0.3], initial_beta=[0.3, 0.5])

        assert result.eigenvalue <= ground_energy + 0.1

    def test_qaoa_with_initial_params(self) -> None:
        """QAOA with custom initial gamma/beta."""
        ZZ = tensor(Operator.Z(), Operator.Z())
        II = tensor(Operator.I(), Operator.I())
        H_C = 0.5 * (II - ZZ)

        qaoa = QAOA(cost_hamiltonian=H_C, num_qubits=2, num_layers=1)
        result = qaoa.solve(initial_gamma=[0.5], initial_beta=[0.3])

        assert isinstance(result, VQEResult)
        # Should find something reasonable (energy <= 0.5 for this problem)
        assert result.eigenvalue <= 0.5

    def test_qaoa_history_convergence(self) -> None:
        """Energy history should show convergence."""
        ZZ = tensor(Operator.Z(), Operator.Z())
        II = tensor(Operator.I(), Operator.I())
        H_C = 0.5 * (II - ZZ)

        optimizer = Adam(learning_rate=0.1, max_iter=300, tol=1e-8)
        qaoa = QAOA(
            cost_hamiltonian=H_C,
            num_qubits=2,
            num_layers=1,
            optimizer=optimizer,
        )
        result = qaoa.solve(initial_gamma=[0.5], initial_beta=[0.5])

        history = result.optimizer_result.history
        # Final energy should be better than or equal to initial
        assert history[-1] <= history[0]

    def test_qaoa_energy_in_range(self) -> None:
        """QAOA energy should be between ground and maximum eigenvalue."""
        ZZ = tensor(Operator.Z(), Operator.Z())
        II = tensor(Operator.I(), Operator.I())
        H_C = 0.5 * (II - ZZ)

        exact = np.linalg.eigvalsh(H_C.matrix)
        ground_energy = float(exact[0])
        max_energy = float(exact[-1])

        qaoa = QAOA(cost_hamiltonian=H_C, num_qubits=2, num_layers=1)
        result = qaoa.solve(initial_gamma=[0.5], initial_beta=[0.5])

        assert ground_energy - 0.1 <= result.eigenvalue <= max_energy + 0.1

    def test_qaoa_maxcut_state_is_bitstring(self) -> None:
        """Optimal QAOA state should have max overlap with cut states.

        For Max-Cut on a 2-node graph, the cut states are |01> and |10>.
        The MaxCut cost Hamiltonian is H_C = 0.5*(ZZ - I), whose ground
        states are |01> and |10> with energy -1.
        """
        ZZ = tensor(Operator.Z(), Operator.Z())
        II = tensor(Operator.I(), Operator.I())
        H_C = 0.5 * (ZZ - II)

        optimizer = Adam(learning_rate=0.1, max_iter=500, tol=1e-8)
        qaoa = QAOA(
            cost_hamiltonian=H_C,
            num_qubits=2,
            num_layers=1,
            optimizer=optimizer,
        )
        result = qaoa.solve(initial_gamma=[0.5], initial_beta=[0.5])

        # Build the optimal state
        ansatz = qaoa.build_ansatz()
        bound = ansatz.bind_parameters(result.eigenstate)
        state = bound.run()

        # Check that |01> and |10> have significant amplitude
        # |01> = index 1, |10> = index 2
        prob_01 = abs(state.amplitudes[1]) ** 2
        prob_10 = abs(state.amplitudes[2]) ** 2
        cut_probability = prob_01 + prob_10

        # At least 50% of the probability should be in cut states
        assert cut_probability >= 0.5
