"""Variational Quantum Deflation (VQD) for excited states.

VQD extends VQE to find excited states by adding overlap penalty
terms that discourage orthogonality with previously found eigenstates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit
from ..core.parameter import Parameter
from ..core.pauli import PauliSum
from ..optimizers.base import Optimizer


@dataclass
class VQDResult(JSONSerializable):
    """Result container for VQD execution.

    Attributes:
        eigenvalues: Found eigenvalues (ground + excited states).
        eigenstates: Optimal parameter sets for each state.
        energies: Energy history per state.
        num_states: Number of states found.
    """

    eigenvalues: list[float] = field(default_factory=list)
    eigenstates: list[dict[Parameter, float]] = field(default_factory=list)
    energies: list[list[float]] = field(default_factory=list)
    num_states: int = 0


class VQD:
    """Variational Quantum Deflation for excited states.

    Finds the k lowest eigenvalues of a Hamiltonian by sequentially
    optimizing ansatz parameters with overlap penalty terms that
    enforce orthogonality to previously found states.

    The cost function for state k is:
        E_k(theta) = <theta|H|theta> + sum_{j<k} beta_j * |<theta_j|theta>|^2

    where beta_j are penalty strengths and theta_j are previously
    optimized parameters.

    Args:
        ansatz: Parameterized quantum circuit.
        hamiltonian: Target Hamiltonian (PauliSum).
        optimizer: Classical optimizer.
        num_states: Number of eigenstates to find.
        beta: Penalty strength for overlap terms. If a list,
            different penalties per state.
        seed: Optional RNG seed.

    Example::

        H = PauliSum([PauliString("Z", -1.0)])
        ansatz = build_ry_ansatz(1, depth=2)
        vqd = VQD(ansatz, H, SPSA(max_iter=100), num_states=2)
        result = vqd.run()
        print(result.eigenvalues)  # [-1.0, 1.0]
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: PauliSum,
        optimizer: Optimizer,
        num_states: int = 2,
        beta: float | list[float] = 1.0,
        seed: int | None = None,
    ) -> None:
        if not ansatz.is_parameterized:
            raise ValueError("Ansatz must have parameters")
        if num_states < 1:
            raise ValueError("num_states must be >= 1")

        self._ansatz = ansatz
        self._hamiltonian = hamiltonian
        self._optimizer = optimizer
        self._num_states = num_states
        self._beta = beta
        self._seed = seed

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the ansatz."""
        return self._ansatz.num_qubits

    @property
    def num_states(self) -> int:
        """Number of eigenstates to find."""
        return self._num_states

    def _get_beta(self, state_idx: int) -> float:
        """Get penalty strength for a given state index."""
        if isinstance(self._beta, list):
            return self._beta[min(state_idx, len(self._beta) - 1)]
        return self._beta

    def _compute_overlap(
        self,
        params_a: dict[Parameter, float],
        params_b: dict[Parameter, float],
    ) -> float:
        """Compute |<psi_a|psi_b>|^2 overlap."""
        sv_a = self._ansatz.bind_parameters(params_a).run()  # type: ignore[arg-type]
        sv_b = self._ansatz.bind_parameters(params_b).run()  # type: ignore[arg-type]
        overlap = abs(np.vdot(sv_a.amplitudes, sv_b.amplitudes)) ** 2
        return float(overlap)

    def _cost_fn(
        self,
        params: dict[Parameter, float],
        previous_states: list[dict[Parameter, float]],
        state_idx: int,
    ) -> float:
        """Evaluate cost with overlap penalties."""
        sv = self._ansatz.bind_parameters(params).run()  # type: ignore[arg-type]
        energy = self._hamiltonian.expectation(sv)

        # Add overlap penalties
        beta = self._get_beta(state_idx)
        for prev_params in previous_states:
            overlap = self._compute_overlap(params, prev_params)
            energy += beta * overlap

        return float(energy)

    def run(self) -> VQDResult:
        """Run VQD to find multiple eigenstates.

        Returns:
            VQDResult with eigenvalues, parameters, and histories.
        """
        result = VQDResult()
        previous_states: list[dict[Parameter, float]] = []
        circuit_params = sorted(self._ansatz.parameters, key=lambda p: p.name)

        for k in range(self._num_states):
            # Capture k in closure for cost function
            state_idx = k

            def cost_factory(
                p: dict[Parameter, float],
                prev: list[dict[Parameter, float]],
                si: int,
            ) -> float:
                return self._cost_fn(p, prev, si)

            def make_cost_fn(
                prev_states: list[dict[Parameter, float]],
                si: int,
            ) -> Callable[[dict[Parameter, float]], float]:
                def fn(params: dict[Parameter, float]) -> float:
                    return cost_factory(params, prev_states, si)
                return fn

            cost_fn = make_cost_fn(previous_states, state_idx)

            # Initialize parameters (random for excited states)
            init: dict[Parameter, float] = {}
            rng = np.random.RandomState(
                (self._seed or 0) + k * 1000
            )
            for p in circuit_params:
                init[p] = float(rng.uniform(-np.pi, np.pi))

            # Optimize
            opt_result = self._optimizer.minimize(
                cost_fn=cost_fn,
                initial_params=init,
            )

            # Store results
            result.eigenvalues.append(opt_result.optimal_value)
            result.eigenstates.append(opt_result.optimal_parameters)
            result.energies.append(opt_result.history)
            previous_states.append(opt_result.optimal_parameters)

        result.num_states = self._num_states
        return result

    def __repr__(self) -> str:
        return (
            f"VQD(num_states={self._num_states}, "
            f"num_qubits={self.num_qubits})"
        )
