"""Quantum Approximate Optimization Algorithm (QAOA)."""

from __future__ import annotations

from typing import Any, Optional, Union

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..core.parameter import Parameter, ParameterExpression
from ..core.pauli import PauliSum
from ..optimizers.base import Optimizer
from ..problems.optimization import OptimizationProblem
from .vqe import VQE, VQEResult

CostHamiltonian = Union[Operator, PauliSum]


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
    """Apply the legacy adjacent-pair ZZ cost unitary.

    Uses CNOT-Rz-CNOT decomposition for each pair of adjacent qubits.
    """
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(gamma, i + 1)
        qc.cx(i, i + 1)
    return qc


def _apply_cost_rotation(
    qc: QuantumCircuit,
    z_positions: list[int],
    angle: Union[float, Parameter, ParameterExpression],
) -> QuantumCircuit:
    """Apply ``exp(-i * angle/2 * prod(Z_i))`` over ``z_positions``.

    Builds the parity qubit with a CNOT chain, applies a single-qubit
    ``Rz`` rotation on it, and unwinds the chain.  This realizes the
    multi-qubit Z-product rotation used in the QAOA cost unitary.
    """
    order = sorted(z_positions)
    chain = list(zip(order, order[1:], strict=False))
    for a, b in chain:
        qc.cx(a, b)
    qc.rz(angle, order[-1])
    for a, b in reversed(chain):
        qc.cx(a, b)
    return qc


def _build_qaoa_ansatz_generic(
    num_qubits: int,
    cost_hamiltonian: PauliSum,
    num_layers: int,
    gamma_params: list[Parameter],
    beta_params: list[Parameter],
) -> QuantumCircuit:
    """Build a QAOA ansatz for an arbitrary Ising (:class:`PauliSum`) cost.

    The cost unitary is ``prod_terms exp(-i * gamma * c_t * P_t)`` where
    ``P_t`` are the Pauli strings of the cost Hamiltonian.  Each term is
    decomposed with :func:`_apply_cost_rotation`; identity terms contribute
    only a constant phase and are skipped.
    """
    qc = _build_initial_state(num_qubits)

    for layer in range(num_layers):
        gamma = gamma_params[layer]
        beta = beta_params[layer]

        for term in cost_hamiltonian.terms:
            coeff = float(term.coefficient.real)
            if coeff == 0.0:
                continue
            z_positions = [k for k, ch in enumerate(term.label) if ch == "Z"]
            if not z_positions:
                continue
            # Rz(angle) = exp(-i * angle * Z / 2); we need exp(-i * gamma * coeff * P)
            _apply_cost_rotation(qc, z_positions, 2.0 * gamma * coeff)

        for i in range(num_qubits):
            qc.rx(beta, i)

    return qc


class QAOA:
    """Quantum Approximate Optimization Algorithm.

    Constructs a p-layer QAOA ansatz and optimizes the (gamma, beta)
    parameter pairs to minimize a cost Hamiltonian.

    ``cost_hamiltonian`` may be:

    * an :class:`Operator` -> legacy adjacent-pair ZZ decomposition, or
    * a :class:`PauliSum` (Ising) -> generic per-term decomposition used by
      the problem-driven API (:meth:`from_problem` / ``solve(problem)``).

    Args:
        cost_hamiltonian: The problem cost Hamiltonian H_C.
        num_qubits: Number of qubits.
        num_layers: Number of QAOA layers (p).
        optimizer: Classical optimizer for parameter updates.
        runtime: Optional execution runtime routed for all evaluations.
        shots: Shots per runtime evaluation (statevector is exact).
        seed: Optional RNG seed for reproducible runtime evaluations.
    """

    def __init__(
        self,
        cost_hamiltonian: CostHamiltonian,
        num_qubits: int,
        num_layers: int = 1,
        optimizer: Optional[Optimizer] = None,
        *,
        runtime: Any = None,
        shots: int = 4096,
        seed: Optional[int] = None,
    ) -> None:
        self._cost_hamiltonian = cost_hamiltonian
        self._num_qubits = num_qubits
        self._num_layers = num_layers
        self._optimizer = optimizer
        self._runtime = runtime
        self._shots = shots
        self._seed = seed
        self._is_generic = isinstance(cost_hamiltonian, PauliSum)

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

    @property
    def cost_hamiltonian(self) -> CostHamiltonian:
        """The cost Hamiltonian defining the objective."""
        return self._cost_hamiltonian

    def build_ansatz(self) -> QuantumCircuit:
        """Build the QAOA ansatz circuit with symbolic parameters."""
        if self._is_generic:
            return _build_qaoa_ansatz_generic(
                self._num_qubits,
                self._cost_hamiltonian,  # type: ignore[arg-type]
                self._num_layers,
                self._gamma_params,
                self._beta_params,
            )
        return _build_qaoa_ansatz(
            self._num_qubits,
            self._cost_hamiltonian,  # type: ignore[arg-type]
            self._num_layers,
            self._gamma_params,
            self._beta_params,
        )

    # ------------------------------------------------------------------
    # generic problem-driven entry points (MQ-05)
    # ------------------------------------------------------------------

    @classmethod
    def from_problem(
        cls,
        problem: OptimizationProblem,
        num_layers: int = 1,
        optimizer: Optional[Optimizer] = None,
        *,
        runtime: Any = None,
        shots: int = 4096,
        seed: Optional[int] = None,
    ) -> "QAOA":
        """Construct a QAOA instance from an :class:`OptimizationProblem`.

        Uses the problem's Ising cost Hamiltonian; raises ``ValueError`` if
        the problem carries no Ising view (build it with
        :meth:`OptimizationProblem.from_qubo` / ``from_ising``).

        Args:
            problem: The optimization problem to solve.
            num_layers: Number of QAOA layers (p).
            optimizer: Classical optimizer; defaults to gradient descent
                with a small step.

        Raises:
            TypeError: If ``problem`` is not an OptimizationProblem.
            ValueError: If the problem is invalid or has no Ising view.
        """
        if not isinstance(problem, OptimizationProblem):
            raise TypeError(
                f"QAOA solves OptimizationProblem, got {type(problem).__name__}"
            )
        issues = problem.validate()
        if issues:
            raise ValueError("Problem validation failed: " + "; ".join(issues))
        ising = problem.cost_hamiltonian()
        return cls(
            ising,
            problem.num_variables,
            num_layers=num_layers,
            optimizer=optimizer,
            runtime=runtime,
            shots=shots,
            seed=seed,
        )

    def validate(self, problem: Any) -> list[str]:
        """Check ``problem`` compatibility (empty list => valid)."""
        if not isinstance(problem, OptimizationProblem):
            return [
                f"QAOA expects an OptimizationProblem, "
                f"got {type(problem).__name__}"
            ]
        issues = list(problem.validate())
        if problem.ising is None:
            issues.append("problem has no Ising cost Hamiltonian (ising=None)")
        elif int(problem.ising.num_qubits) != self._num_qubits:
            issues.append("problem and QAOA qubit counts differ")
        return issues

    def solve(
        self,
        problem: Optional[OptimizationProblem] = None,
        initial_gamma: Optional[list[float]] = None,
        initial_beta: Optional[list[float]] = None,
        *,
        runtime: Any = None,
        initial_params: Optional[dict[Union[str, Parameter], float]] = None,
    ) -> VQEResult:
        """Run QAOA optimization.

        Two calling styles are supported:

        * Problem-driven: ``solve(problem, runtime=rt)`` — uses the problem's
          Ising cost Hamiltonian (and its generic ansatz).
        * Legacy: ``solve(initial_gamma=[...], initial_beta=[...])`` — uses
          the Hamiltonian fixed at construction time.

        Args:
            problem: Optional optimization problem to solve.
            initial_gamma: Initial gamma values per layer (legacy path),
                defaults to zeros.
            initial_beta: Initial beta values per layer (legacy path),
                defaults to zeros.
            runtime: Optional execution runtime override.
            initial_params: Optional starting parameter values.

        Returns:
            VQEResult with optimal energy and parameters.
        """
        if problem is None or isinstance(problem, OptimizationProblem):
            return self._solve_with_options(
                problem, runtime, initial_gamma, initial_beta, initial_params
            )
        raise TypeError(
            "first argument to solve() must be an OptimizationProblem "
            f"or None, got {type(problem).__name__}"
        )

    def _solve_with_options(
        self,
        problem: Optional[OptimizationProblem],
        runtime: Any,
        initial_gamma: Optional[list[float]],
        initial_beta: Optional[list[float]],
        initial_params: Optional[dict[Union[str, Parameter], float]],
    ) -> VQEResult:
        from ..optimizers.gradient_descent import GradientDescent

        if problem is not None:
            validation = self.validate(problem)
            if validation:
                raise ValueError(
                    "QAOA problem validation failed:\n  - " + "\n  - ".join(validation)
                )
            qaoa = self
            if qaoa._cost_hamiltonian is not problem.cost_hamiltonian():
                qaoa = QAOA.from_problem(
                    problem,
                    num_layers=self._num_layers,
                    optimizer=self._optimizer,
                    runtime=runtime if runtime is not None else self._runtime,
                    shots=self._shots,
                    seed=self._seed,
                )
        else:
            qaoa = self

        ansatz = qaoa.build_ansatz()

        if qaoa._optimizer is None:
            optimizer: Optimizer = GradientDescent(learning_rate=0.1, max_iter=200)
        else:
            optimizer = qaoa._optimizer

        vqe = VQE(
            ansatz,
            qaoa._cost_hamiltonian,
            optimizer,
            runtime=runtime if runtime is not None else qaoa._runtime,
            shots=qaoa._shots,
            seed=qaoa._seed,
        )

        init_params: dict[Union[str, Parameter], float] = {}
        if initial_params is not None:
            init_params.update(initial_params)
        elif initial_gamma is not None or initial_beta is not None:
            if initial_gamma is None:
                initial_gamma = [0.0] * self._num_layers
            if initial_beta is None:
                initial_beta = [0.0] * self._num_layers
            for layer in range(self._num_layers):
                init_params[self._gamma_params[layer]] = initial_gamma[layer]
                init_params[self._beta_params[layer]] = initial_beta[layer]
        else:
            # Default starts: (gamma, beta) = (0.7, 0.7) per layer.  The
            # uniform state |+>^n is a fixed point of the QAOA dynamics at
            # (0, 0), where every gradient vanishes; a non-zero tabula-rasa
            # start escapes it and keeps the algorithm deterministic.
            for layer in range(self._num_layers):
                init_params[self._gamma_params[layer]] = 0.7
                init_params[self._beta_params[layer]] = 0.7

        return vqe.compute_minimum_eigenvalue(initial_params=init_params)

    def __repr__(self) -> str:
        return (
            f"QAOA(qubits={self._num_qubits}, layers={self._num_layers}, "
            f"optimizer={type(self._optimizer).__name__ if self._optimizer else 'default'}, "
            f"generic={self._is_generic})"
        )