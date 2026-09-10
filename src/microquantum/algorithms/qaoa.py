"""Quantum Approximate Optimization Algorithm (QAOA)."""

from __future__ import annotations

from typing import Optional, Union

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..core.parameter import Parameter
from ..optimizers.base import Optimizer
from .vqe import VQE, VQEResult


def _build_initial_state(num_qubits: int) -> QuantumCircuit:
    """Build uniform superposition |+>^n via H on all qubits."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    return qc


def _build_qaoa_ansatz(
    num_qubits: int,
    cost_hamiltonian: Operator,
    num_layers: int,
    gamma_params: list[Parameter],
    beta_params: list[Parameter],
) -> QuantumCircuit:
    """Construct the QAOA ansatz circuit.

    Each layer applies:
    1. Cost unitary using CNOT-Rz-CNOT decomposition
    2. Mixer unitary Rx on each qubit

    Note: Parameters are used directly as rotation angles for
    compatibility with the parameter-shift gradient rule.
    """
    qc = _build_initial_state(num_qubits)

    for layer in range(num_layers):
        gamma = gamma_params[layer]
        beta = beta_params[layer]

        qc = _apply_cost_unitary(qc, num_qubits, gamma)
        for i in range(num_qubits):
            qc.rx(beta, i)

    return qc


def _apply_cost_unitary(
    qc: QuantumCircuit,
    num_qubits: int,
    gamma: Union[float, Parameter],
) -> QuantumCircuit:
    """Apply cost unitary via CNOT-Rz-CNOT decomposition.

    For ZZ-type cost Hamiltonians, applies CNOT-Rz(gamma)-CNOT
    for each pair of adjacent qubits.
    """
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(gamma, i + 1)
        qc.cx(i, i + 1)
    return qc


class QAOA:
    """Quantum Approximate Optimization Algorithm.

    Constructs a p-layer QAOA ansatz and optimizes the (gamma, beta)
    parameter pairs to minimize a cost Hamiltonian.

    Args:
        cost_hamiltonian: The problem cost Hamiltonian H_C.
        num_qubits: Number of qubits.
        num_layers: Number of QAOA layers (p).
        optimizer: Classical optimizer for parameter updates.
    """

    def __init__(
        self,
        cost_hamiltonian: Operator,
        num_qubits: int,
        num_layers: int = 1,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        self._cost_hamiltonian = cost_hamiltonian
        self._num_qubits = num_qubits
        self._num_layers = num_layers
        self._optimizer = optimizer

        # Create symbolic parameters
        self._gamma_params = [
            Parameter(f"gamma_{layer}") for layer in range(num_layers)
        ]
        self._beta_params = [
            Parameter(f"beta_{layer}") for layer in range(num_layers)
        ]

    @property
    def num_layers(self) -> int:
        """Number of QAOA layers."""
        return self._num_layers

    @property
    def parameters(self) -> list[Parameter]:
        """All QAOA parameters (gammas then betas)."""
        return list(self._gamma_params) + list(self._beta_params)

    def build_ansatz(self) -> QuantumCircuit:
        """Build the QAOA ansatz circuit with symbolic parameters."""
        return _build_qaoa_ansatz(
            self._num_qubits,
            self._cost_hamiltonian,
            self._num_layers,
            self._gamma_params,
            self._beta_params,
        )

    def solve(
        self,
        initial_gamma: Optional[list[float]] = None,
        initial_beta: Optional[list[float]] = None,
    ) -> VQEResult:
        """Run QAOA optimization.

        Args:
            initial_gamma: Initial gamma values for each layer.
                Defaults to all zeros.
            initial_beta: Initial beta values for each layer.
                Defaults to all zeros.

        Returns:
            VQEResult with optimal energy and parameters.
        """
        from ..optimizers.gradient_descent import GradientDescent

        ansatz = self.build_ansatz()

        if self._optimizer is None:
            optimizer: Optimizer = GradientDescent(learning_rate=0.1, max_iter=200)
        else:
            optimizer = self._optimizer

        vqe = VQE(ansatz, self._cost_hamiltonian, optimizer)

        # Build initial param map
        init_params: dict[Union[str, Parameter], float] = {}
        if initial_gamma is None:
            initial_gamma = [0.0] * self._num_layers
        if initial_beta is None:
            initial_beta = [0.0] * self._num_layers

        for layer in range(self._num_layers):
            init_params[self._gamma_params[layer]] = initial_gamma[layer]
            init_params[self._beta_params[layer]] = initial_beta[layer]

        return vqe.compute_minimum_eigenvalue(initial_params=init_params)
