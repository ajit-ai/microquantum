"""Variational Quantum Eigensolver (VQE) algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from ..core.circuit import QuantumCircuit
from ..core.gradient import gradient as compute_gradient
from ..core.measurement import expectation_value
from ..core.operators import Operator
from ..core.parameter import Parameter
from ..optimizers.base import Optimizer, OptimizerResult


@dataclass
class VQEResult:
    """Result container for VQE execution.

    Attributes:
        eigenvalue: Estimated ground-state energy.
        eigenstate: Optimal parameter values.
        optimizer_result: Full optimizer output including history.
    """

    eigenvalue: float = float("inf")
    eigenstate: dict[Parameter, float] = field(default_factory=dict)
    optimizer_result: Optional[OptimizerResult] = None


class VQE:
    """Variational Quantum Eigensolver.

    Finds the minimum eigenvalue of a Hamiltonian by optimizing
    a parameterized quantum circuit (ansatz) to minimize
    <psi(theta)|H|psi(theta)>.

    Args:
        ansatz: Parameterized quantum circuit.
        hamiltonian: Hermitian operator whose ground-state energy
            is sought.
        optimizer: Classical optimizer for parameter updates.
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: Operator,
        optimizer: Optimizer,
    ) -> None:
        if not ansatz.is_parameterized:
            raise ValueError("Ansatz circuit must have at least one parameter")
        self._ansatz = ansatz
        self._hamiltonian = hamiltonian
        self._optimizer = optimizer

    @property
    def ansatz(self) -> QuantumCircuit:
        """The parameterized ansatz circuit."""
        return self._ansatz

    @property
    def hamiltonian(self) -> Operator:
        """The target Hamiltonian."""
        return self._hamiltonian

    def _cost_fn(self, param_values: dict[Parameter, float]) -> float:
        """Evaluate the energy expectation value."""
        bound = self._ansatz.bind_parameters(param_values)  # type: ignore[arg-type]
        state = bound.run()
        return expectation_value(state, self._hamiltonian)

    def _gradient_fn(
        self, param_values: dict[Parameter, float]
    ) -> dict[Parameter, float]:
        """Compute gradients of the energy w.r.t. all parameters."""
        return compute_gradient(
            self._ansatz, self._hamiltonian, param_values  # type: ignore[arg-type]
        )

    def compute_minimum_eigenvalue(
        self,
        initial_params: Optional[dict[Union[str, Parameter], float]] = None,
    ) -> VQEResult:
        """Run VQE to find the minimum eigenvalue.

        Args:
            initial_params: Starting parameter values. If None,
                all parameters initialized to 0.0.

        Returns:
            VQEResult with eigenvalue, optimal parameters, and history.
        """
        # Build initial parameter dict from circuit parameters
        circuit_params = sorted(self._ansatz.parameters, key=lambda p: p.name)
        init: dict[Parameter, float] = {}
        if initial_params is None:
            initial_params = {}
        for p in circuit_params:
            if p in initial_params:
                init[p] = float(initial_params[p])
            elif p.name in initial_params:
                init[p] = float(initial_params[p.name])
            else:
                init[p] = 0.0

        opt_result = self._optimizer.minimize(
            cost_fn=self._cost_fn,
            gradient_fn=self._gradient_fn,
            initial_params=init,
        )

        return VQEResult(
            eigenvalue=opt_result.optimal_value,
            eigenstate=opt_result.optimal_parameters,
            optimizer_result=opt_result,
        )
