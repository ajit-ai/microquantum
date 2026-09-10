"""Adaptive Variational Quantum Eigensolver (ADAPT-VQE)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit
from ..core.parameter import Parameter
from ..core.pauli import PauliString, PauliSum
from ..core.state import StateVector


@dataclass
class AdaptResult(JSONSerializable):
    """Result container for ADAPT-VQE execution.

    Attributes:
        circuit: Final ansatz circuit with optimal parameters bound.
        energy: Final ground-state energy estimate.
        state: Final state vector produced by the ansatz.
        num_layers: Number of operators added to the ansatz.
        energy_history: Energy at each iteration (including initial).
        operator_indices: Indices of operators added, in order.
        converged: Whether the gradient threshold was met.
    """

    circuit: QuantumCircuit
    energy: float
    state: StateVector
    num_layers: int
    energy_history: list[float] = field(default_factory=list)
    operator_indices: list[int] = field(default_factory=list)
    converged: bool = False


class AdaptVQE:
    """Adaptive Variational Quantum Eigensolver.

    Iteratively grows a parameterized ansatz by selecting operators
    from a pool that have the largest energy gradient, then optimizes
    all parameters after each addition.

    Args:
        hamiltonian: Target Hamiltonian as a PauliSum.
        pool: Operator pool to select from. If None, uses
            X, Y, Z on each qubit plus ZZ on adjacent pairs.
        initial_circuit: Optional pre-existing circuit to start from.
        max_layers: Maximum number of operators to add.
        gradient_threshold: Convergence threshold on max gradient.
        optimizer_max_iter: Gradient-descent iterations per layer.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        hamiltonian: PauliSum,
        pool: list[PauliString] | None = None,
        initial_circuit: QuantumCircuit | None = None,
        max_layers: int = 20,
        gradient_threshold: float = 1e-4,
        optimizer_max_iter: int = 100,
        seed: int | None = None,
    ) -> None:
        if not hamiltonian.terms:
            raise ValueError("Hamiltonian must have at least one term")
        self._hamiltonian = hamiltonian
        self._num_qubits = hamiltonian.num_qubits
        self._pool = pool if pool is not None else self.build_default_pool()
        self._initial_circuit = initial_circuit
        self._max_layers = max_layers
        self._gradient_threshold = gradient_threshold
        self._optimizer_max_iter = optimizer_max_iter
        self._seed = seed

    @property
    def hamiltonian(self) -> PauliSum:
        """The target Hamiltonian."""
        return self._hamiltonian

    @property
    def pool(self) -> list[PauliString]:
        """The operator pool."""
        return self._pool

    @property
    def max_layers(self) -> int:
        """Maximum number of adaptive layers."""
        return self._max_layers

    @property
    def gradient_threshold(self) -> float:
        """Convergence threshold on gradient magnitude."""
        return self._gradient_threshold

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the system."""
        return self._num_qubits

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def build_default_pool(self) -> list[PauliString]:
        """Build a default operator pool.

        Includes X, Y, Z on each qubit and ZZ on each pair of
        adjacent qubits.

        Returns:
            List of PauliString operators.
        """
        pool: list[PauliString] = []
        n = self._num_qubits
        paulis = ("X", "Y", "Z")

        for i in range(n):
            for p in paulis:
                label = "I" * i + p + "I" * (n - i - 1)
                pool.append(PauliString(label))

        for i in range(n - 1):
            label = "I" * i + "ZZ" + "I" * (n - i - 2)
            pool.append(PauliString(label))

        return pool

    # ------------------------------------------------------------------
    # Gate decomposition for exp(-i*param*P/2)
    # ------------------------------------------------------------------

    def _apply_exp_operator(
        self,
        qc: QuantumCircuit,
        label: str,
        param: Parameter,
        qubits: int,
    ) -> None:
        """Append gates for exp(-i * param * P / 2) to the circuit.

        Uses basis change, CNOT ladder for parity, and Rz rotation.

        Args:
            qc: Circuit to append gates to (mutated in place).
            label: Pauli label string (e.g. "XYZI").
            param: Parameter used as the rotation angle.
            qubits: Total number of qubits in the circuit.
        """
        del qubits  # number of qubits inferred from circuit

        active = [i for i, p in enumerate(label) if p != "I"]
        if not active:
            return

        # Forward basis change: map each Pauli to Z
        for i in active:
            pauli = label[i]
            if pauli == "X":
                qc.h(i)
            elif pauli == "Y":
                qc.sdg(i)
                qc.h(i)

        # CNOT ladder to compute parity into the last active qubit
        for j in range(len(active) - 1):
            qc.cx(active[j], active[j + 1])

        # Rotation on the parity qubit
        qc.rz(param, active[-1])

        # Reverse CNOT ladder
        for j in range(len(active) - 2, -1, -1):
            qc.cx(active[j], active[j + 1])

        # Reverse basis change
        for i in active:
            pauli = label[i]
            if pauli == "X":
                qc.h(i)
            elif pauli == "Y":
                qc.h(i)
                qc.s(i)

    # ------------------------------------------------------------------
    # Ansatz construction
    # ------------------------------------------------------------------

    def _build_ansatz(
        self,
        operator_indices: list[int],
        params: list[Parameter],
    ) -> QuantumCircuit:
        """Build parameterized ansatz from initial circuit plus exp gates.

        Args:
            operator_indices: Indices into the pool for each added operator.
            params: One Parameter per operator in operator_indices.

        Returns:
            A new parameterized QuantumCircuit.
        """
        n = self._num_qubits

        if self._initial_circuit is not None:
            qc = self._initial_circuit + QuantumCircuit(n)
        else:
            qc = QuantumCircuit(n)

        for idx, param in zip(operator_indices, params, strict=False):
            label = self._pool[idx].label
            self._apply_exp_operator(qc, label, param, n)

        return qc

    # ------------------------------------------------------------------
    # Energy and gradient computation
    # ------------------------------------------------------------------

    def _compute_energy(self, state: StateVector) -> float:
        """Compute the energy expectation value <psi|H|psi>.

        Args:
            state: Quantum state vector.

        Returns:
            Real-valued energy.
        """
        return self._hamiltonian.expectation(state)

    def _eval_ansatz(
        self,
        ansatz: QuantumCircuit,
        param_values: dict[Parameter, float],
    ) -> tuple[StateVector, float]:
        """Bind parameters, run circuit, and compute energy.

        Args:
            ansatz: Parameterized circuit.
            param_values: Mapping from Parameter to float.

        Returns:
            Tuple of (state, energy).
        """
        bound = ansatz.bind_parameters(param_values)  # type: ignore[arg-type]
        state = bound.run()
        energy = self._compute_energy(state)
        return state, energy

    def _compute_gradients(
        self,
        operator_indices: list[int],
        params: list[Parameter],
        param_values: dict[Parameter, float],
        delta: float = 0.01,
    ) -> list[float]:
        """Compute energy gradients for each pool operator via finite difference.

        For each operator A_i in the pool, the gradient measures how
        much the energy would change if A_i were appended with a
        small parameter value.

        Args:
            operator_indices: Currently selected operator indices.
            params: Currently active Parameters.
            param_values: Current parameter values.
            delta: Finite-difference step size.

        Returns:
            List of gradients, one per pool operator.
        """
        gradients: list[float] = []

        for i in range(len(self._pool)):
            param_delta = Parameter(f"_grad_delta_{i}")

            op_idx_plus = operator_indices + [i]
            params_plus = params + [param_delta]
            ansatz_plus = self._build_ansatz(op_idx_plus, params_plus)

            vals_plus: dict[Parameter, float] = dict(param_values)
            vals_plus[param_delta] = delta
            _, energy_plus = self._eval_ansatz(ansatz_plus, vals_plus)

            vals_minus: dict[Parameter, float] = dict(param_values)
            vals_minus[param_delta] = -delta
            _, energy_minus = self._eval_ansatz(ansatz_plus, vals_minus)

            gradients.append((energy_plus - energy_minus) / (2.0 * delta))

        return gradients

    # ------------------------------------------------------------------
    # Parameter optimization
    # ------------------------------------------------------------------

    def _optimize(
        self,
        ansatz: QuantumCircuit,
        params: list[Parameter],
        param_values: dict[Parameter, float],
    ) -> dict[Parameter, float]:
        """Optimize parameters via gradient descent with parameter shift.

        For Rz(theta), the parameter-shift rule gives:
            dE/dtheta = (E(theta + pi/2) - E(theta - pi/2)) / 2

        Args:
            ansatz: Parameterized circuit.
            params: Parameters to optimize.
            param_values: Current parameter values (mutated in place).

        Returns:
            Updated parameter values.
        """
        shift = np.pi / 2.0
        learning_rate = 0.1
        values = dict(param_values)

        for _ in range(self._optimizer_max_iter):
            for param in params:
                vals_plus = dict(values)
                vals_plus[param] = values[param] + shift
                _, energy_plus = self._eval_ansatz(ansatz, vals_plus)

                vals_minus = dict(values)
                vals_minus[param] = values[param] - shift
                _, energy_minus = self._eval_ansatz(ansatz, vals_minus)

                gradient = (energy_plus - energy_minus) / 2.0
                values[param] = values[param] - learning_rate * gradient

        return values

    # ------------------------------------------------------------------
    # Main ADAPT loop
    # ------------------------------------------------------------------

    def run(self) -> AdaptResult:
        """Execute the ADAPT-VQE algorithm.

        Returns:
            AdaptResult with the best ansatz, energy, and diagnostics.
        """
        selected_indices: list[int] = []
        params: list[Parameter] = []
        param_values: dict[Parameter, float] = {}
        energy_history: list[float] = []
        converged = False

        # Evaluate initial energy
        if self._initial_circuit is not None and not self._initial_circuit.is_parameterized or self._initial_circuit is not None and self._initial_circuit.is_parameterized:
            state = self._initial_circuit.run()
        else:
            state = StateVector(self._num_qubits)

        energy = self._compute_energy(state)
        energy_history.append(energy)

        for layer in range(self._max_layers):
            # Compute gradients for every pool operator
            gradients = self._compute_gradients(
                selected_indices, params, param_values
            )

            # Select operator with largest gradient magnitude
            abs_grads = [abs(g) for g in gradients]
            best_idx = int(np.argmax(abs_grads))
            best_grad = abs_grads[best_idx]

            if best_grad < self._gradient_threshold:
                converged = True
                break

            # Add operator to the ansatz
            new_param = Parameter(f"adapt_{layer}")
            selected_indices.append(best_idx)
            params.append(new_param)
            param_values[new_param] = 0.0

            # Build ansatz and optimize all parameters
            ansatz = self._build_ansatz(selected_indices, params)
            param_values = self._optimize(ansatz, params, param_values)

            # Record energy after optimization
            state, energy = self._eval_ansatz(ansatz, param_values)
            energy_history.append(energy)

        # Build final ansatz with bound parameters
        final_ansatz = self._build_ansatz(selected_indices, params)
        final_bound = final_ansatz.bind_parameters(param_values)  # type: ignore[arg-type]
        final_state = final_bound.run()

        return AdaptResult(
            circuit=final_bound,
            energy=energy_history[-1],
            state=final_state,
            num_layers=len(selected_indices),
            energy_history=energy_history,
            operator_indices=selected_indices,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        pool_size = len(self._pool)
        return (
            f"AdaptVQE(num_qubits={self._num_qubits}, "
            f"pool_size={pool_size}, "
            f"max_layers={self._max_layers}, "
            f"gradient_threshold={self._gradient_threshold})"
        )
