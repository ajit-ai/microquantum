"""Variational Quantum Eigensolver (VQE) algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit
from ..core.gradient import gradient as compute_gradient
from ..core.measurement import expectation_value
from ..core.operators import Operator
from ..core.parameter import Parameter
from ..core.pauli import PauliSum
from ..core.state import StateVector
from ..optimizers.base import Optimizer, OptimizerResult
from ..problems.eigenvalue import EigenvalueProblem, HamiltonianProblem

Hamiltonian = Union[Operator, PauliSum]


@dataclass
class VQEResult(JSONSerializable):
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
        hamiltonian: Hermitian operator or Pauli sum whose ground-state
            energy is sought.
        optimizer: Classical optimizer for parameter updates.
        runtime: Optional :class:`~microquantum.runtime.ExecutionRuntime`.
            When provided, every circuit evaluation is executed through the
            MQ-04 runtime pipeline (plan -> backend -> job -> result) instead
            of the internal state-vector engine.
        shots: Shots used per runtime evaluation (statevector is exact).
        seed: Optional RNG seed for reproducible runtime evaluations.
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: Hamiltonian,
        optimizer: Optimizer,
        *,
        runtime: Any = None,
        shots: int = 4096,
        seed: Optional[int] = None,
    ) -> None:
        if not ansatz.is_parameterized:
            raise ValueError("Ansatz circuit must have at least one parameter")
        self._ansatz = ansatz
        self._hamiltonian = hamiltonian
        self._optimizer = optimizer
        self._runtime = runtime
        self._shots = shots
        self._seed = seed

    @property
    def ansatz(self) -> QuantumCircuit:
        """The parameterized ansatz circuit."""
        return self._ansatz

    @property
    def hamiltonian(self) -> Hamiltonian:
        """The target Hamiltonian."""
        return self._hamiltonian

    @property
    def optimizer(self) -> Optimizer:
        """The configured classical optimizer."""
        return self._optimizer

    @property
    def runtime(self) -> Any:
        """Execution runtime used for circuit evaluations (or None)."""
        return self._runtime

    # ------------------------------------------------------------------
    # observables
    # ------------------------------------------------------------------

    def _expectation(self, state: StateVector) -> float:
        if isinstance(self._hamiltonian, PauliSum):
            return self._hamiltonian.expectation(state)
        return expectation_value(state, self._hamiltonian)

    def _state_for(self, param_values: dict[Parameter, float]) -> StateVector:
        """Produce the ansatz state at ``param_values``.

        Uses the MQ-04 runtime when configured, otherwise the internal
        state-vector engine.  The runtime returns an exact statevector
        regardless of ``shots``, so expectation values are deterministic.
        """
        bound = self._ansatz.bind_parameters(param_values)  # type: ignore[arg-type]
        if self._runtime is None:
            return bound.run()
        result = self._runtime.execute(bound, shots=self._shots, seed=self._seed)
        amps = np.asarray(result.statevector, dtype=np.complex128)
        return StateVector(num_qubits=self._ansatz.num_qubits, amplitudes=amps)

    def _cost_fn(self, param_values: dict[Parameter, float]) -> float:
        """Evaluate the energy expectation value."""
        return self._expectation(self._state_for(param_values))

    def _gradient_fn(
        self, param_values: dict[Parameter, float]
    ) -> dict[Parameter, float]:
        """Compute gradients of the energy w.r.t. all parameters."""
        if isinstance(self._hamiltonian, PauliSum):
            total: dict[Parameter, float] = {}
            for term in self._hamiltonian.terms:
                op = term.to_operator()
                grad = compute_gradient(
                    self._ansatz, op, param_values  # type: ignore[arg-type]
                )
                for p, value in grad.items():
                    total[p] = total.get(p, 0.0) + value
            return total
        return compute_gradient(
            self._ansatz, self._hamiltonian, param_values  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # legacy entry point
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # generic problem-driven entry points (MQ-05)
    # ------------------------------------------------------------------

    @classmethod
    def from_problem(
        cls,
        problem: Union[EigenvalueProblem, HamiltonianProblem],
        ansatz: QuantumCircuit,
        optimizer: Optional[Optimizer] = None,
        *,
        runtime: Any = None,
        shots: int = 4096,
        seed: Optional[int] = None,
    ) -> "VQE":
        """Construct a VQE for a Hamiltonian/Eigenvalue problem.

        Args:
            problem: The problem to solve (its ``hamiltonian`` is used).
            ansatz: Parameterized and dimension-compatible ansatz.
            optimizer: Classical optimizer (required to actually solve).

        Raises:
            TypeError: If ``problem`` is not a Hamiltonian/Eigenvalue problem.
            ValueError: If the problem or ansatz dimensions mismatch.
        """
        if not isinstance(problem, (EigenvalueProblem, HamiltonianProblem)):
            raise TypeError(
                f"VQE solves HamiltonianProblem/EigenvalueProblem, "
                f"got {type(problem).__name__}"
            )
        problems = problem.validate()
        if problems:
            raise ValueError("Problem validation failed: " + "; ".join(problems))
        hamiltonian = problem.hamiltonian
        if hamiltonian is None:
            raise ValueError("problem has no Hamiltonian")
        if int(hamiltonian.num_qubits) != ansatz.num_qubits:
            raise ValueError(
                f"ansatz uses {ansatz.num_qubits} qubits but problem "
                f"Hamiltonian acts on {hamiltonian.num_qubits}"
            )
        from ..optimizers.gradient_descent import GradientDescent

        if optimizer is None:
            optimizer = GradientDescent(learning_rate=0.1, max_iter=200)
        return cls(
            ansatz,
            hamiltonian,
            optimizer,
            runtime=runtime,
            shots=shots,
            seed=seed,
        )

    def validate(self, problem: Any) -> list[str]:
        """Check ``problem`` compatibility (empty list => valid)."""
        if not isinstance(problem, (EigenvalueProblem, HamiltonianProblem)):
            return [
                f"VQE expects an EigenvalueProblem/HamiltonianProblem, "
                f"got {type(problem).__name__}"
            ]
        problems = list(problem.validate())
        hamiltonian = problem.hamiltonian
        if hamiltonian is not None and int(hamiltonian.num_qubits) != self._ansatz.num_qubits:
            problems.append("ansatz and problem Hamiltonian qubit counts differ")
        return problems

    def solve(
        self,
        problem: Union[EigenvalueProblem, HamiltonianProblem],
        runtime: Any = None,
        initial_params: Optional[dict[Union[str, Parameter], float]] = None,
    ) -> VQEResult:
        """Solve a Hamiltonian/Eigenvalue problem with this ansatz strategy.

        Recreates the solver with the problem's Hamiltonian so the same
        ansatz/optimizer configuration can be applied to any compatible
        problem (problem/algorithm separation).

        Args:
            problem: The problem to solve.
            runtime: Optional execution runtime override.
            initial_params: Optional starting parameter values.

        Returns:
            VQEResult for the solved problem.
        """
        validation = self.validate(problem)
        if validation:
            raise ValueError("VQE problem validation failed:\n  - " + "\n  - ".join(validation))
        from ..optimizers.gradient_descent import GradientDescent

        optimizer = self._optimizer if self._optimizer is not None else GradientDescent(learning_rate=0.1, max_iter=200)
        vqe = VQE(
            self._ansatz,
            problem.hamiltonian,  # type: ignore[arg-type]
            optimizer,
            runtime=runtime if runtime is not None else self._runtime,
            shots=self._shots,
            seed=self._seed,
        )
        return vqe.compute_minimum_eigenvalue(initial_params=initial_params)

    def __repr__(self) -> str:
        return (
            f"VQE(ansatz_qubits={self._ansatz.num_qubits}, "
            f"params={len(self._ansatz.parameters)}, "
            f"optimizer={type(self._optimizer).__name__}, "
            f"runtime={'yes' if self._runtime is not None else 'no'})"
        )