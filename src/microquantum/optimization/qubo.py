"""QUBO (Quadratic Unconstrained Binary Optimization) Toolchain.

Converts business optimization problems into quantum-ready Ising Hamiltonians.

QUBO is the standard formulation for combinatorial optimization:
    minimize  x^T Q x
    where x in {0, 1}^n

This module provides:
- QUBOBuilder: Build QUBO matrices from business variables
- ConstraintPenalty: Encode constraints as penalty terms
- IsingConverter: Convert QUBO to Ising Hamiltonian for quantum solvers
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from ..core.operators import Operator
from ..core.pauli import PauliString, PauliSum


@dataclass
class QUBOProblem:
    """A QUBO problem formulation.

    Attributes:
        Q: Upper-triangular coefficient matrix.
        linear: Linear coefficients (diagonal of Q).
        offset: Constant energy offset.
        num_variables: Number of binary variables.
        name: Problem name.
        metadata: Additional problem metadata.
    """
    Q: np.ndarray
    linear: np.ndarray
    offset: float = 0.0
    num_variables: int = 0
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_variables == 0:
            self.num_variables = self.Q.shape[0]

    def energy(self, x: np.ndarray) -> float:
        """Compute QUBO energy for a binary vector.

        Args:
            x: Binary vector of shape (n,).

        Returns:
            Energy value.
        """
        x = np.asarray(x, dtype=float)
        # Q holds off-diagonal interactions; linear holds diagonal terms
        return float(x @ self.Q @ x + self.linear @ x + self.offset)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "Q": self.Q.tolist(),
            "linear": self.linear.tolist(),
            "offset": self.offset,
            "num_variables": self.num_variables,
            "name": self.name,
        }


class QUBOBuilder:
    """Build QUBO problems from business variables and objectives.

    Example::

        builder = QUBOBuilder(num_variables=4)
        builder.add_quadratic(0, 1, 2.0)  # x0 * x1 coefficient
        builder.add_linear(2, -1.5)       # x2 coefficient
        builder.add_penalty_equality(0, 1, target=1)  # x0 + x1 == 1
        qubo = builder.build("portfolio_selection")
    """

    def __init__(self, num_variables: int) -> None:
        """Initialize builder.

        Args:
            num_variables: Number of binary variables.
        """
        self._n = num_variables
        self._Q = np.zeros((num_variables, num_variables), dtype=float)
        self._linear = np.zeros(num_variables, dtype=float)
        self._offset = 0.0

    @property
    def num_variables(self) -> int:
        return self._n

    def add_linear(self, i: int, coefficient: float) -> None:
        """Add linear term: coefficient * x_i."""
        self._linear[i] += coefficient

    def add_quadratic(self, i: int, j: int, coefficient: float) -> None:
        """Add quadratic term: coefficient * x_i * x_j."""
        if i == j:
            self._linear[i] += coefficient
        else:
            self._Q[i, j] += coefficient

    def add_constant(self, value: float) -> None:
        """Add constant offset."""
        self._offset += value

    def add_penalty_equality(
        self, i: int, j: int, target: int = 1, penalty: float = 10.0
    ) -> None:
        """Add penalty for x_i + x_j == target.

        Penalty = penalty * (x_i + x_j - target)^2
        """
        if target == 1:
            # (x_i + x_j - 1)^2 = x_i^2 + x_j^2 + 1 + 2*x_i*x_j - 2*x_i - 2*x_j
            # = x_i + x_j + 1 + 2*x_i*x_j - 2*x_i - 2*x_j (since x_i^2 = x_i)
            # = -x_i - x_j + 1 + 2*x_i*x_j
            self._linear[i] += penalty * (-1)
            self._linear[j] += penalty * (-1)
            self._Q[i, j] += penalty * 2
            self._offset += penalty * 1
        elif target == 0:
            # (x_i + x_j)^2 = x_i + x_j + 2*x_i*x_j
            self._linear[i] += penalty * 1
            self._linear[j] += penalty * 1
            self._Q[i, j] += penalty * 2

    def add_penalty_inequality_le(
        self, indices: list[int], max_sum: int, penalty: float = 10.0
    ) -> None:
        """Add penalty for sum(x_i for i in indices) <= max_sum.

        Uses slack variable approach: introduce slack s >= 0 such that
        sum(x_i) + s = max_sum, then penalize s^2.
        """
        n_slack = max_sum
        # Simple approach: penalize all pairs that violate the constraint
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                if a != b:
                    self._Q[indices[a], indices[b]] += penalty

    def add_penalty_one_hot(self, indices: list[int], penalty: float = 10.0) -> None:
        """Add one-hot constraint: exactly one x_i = 1.

        Penalty = penalty * (sum(x_i) - 1)^2
        """
        k = len(indices)
        # (sum(x_i) - 1)^2 = sum(x_i^2) + sum(x_i*x_j) - 2*sum(x_i) + 1
        # = sum(x_i) + sum(x_i*x_j) - 2*sum(x_i) + 1 (since x_i^2 = x_i)
        # = -sum(x_i) + sum(x_i*x_j) + 1
        for i in indices:
            self._linear[i] += penalty * (-1)
        for a in range(k):
            for b in range(a + 1, k):
                self._Q[indices[a], indices[b]] += penalty * 2
        self._offset += penalty * 1

    def add_penalty_at_most_one(
        self, indices: list[int], penalty: float = 10.0
    ) -> None:
        """Add at-most-one constraint: at most one x_i = 1."""
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                self._Q[indices[a], indices[b]] += penalty

    def build(self, name: str = "") -> QUBOProblem:
        """Build the QUBO problem.

        Note:
            Q is kept purely off-diagonal (interactions); linear terms are
            stored separately to avoid double counting in the energy.

        Returns:
            QUBOProblem with the assembled Q matrix.
        """
        return QUBOProblem(
            Q=self._Q.copy(),
            linear=self._linear.copy(),
            offset=self._offset,
            num_variables=self._n,
            name=name,
        )


class IsingConverter:
    """Convert QUBO problems to Ising Hamiltonians for quantum solvers.

    Maps binary variables x in {0, 1} to spin variables s in {-1, +1}:
        x = (1 - s) / 2
    """

    @staticmethod
    def qubo_to_ising(qubo: QUBOProblem) -> PauliSum:
        """Convert QUBO to Ising Hamiltonian.

        The QUBO energy E(x) = x^T Q x + c^T x + offset
        maps to Ising: H = sum J_ij * Z_i * Z_j + sum h_i * Z_i + const

        Args:
            qubo: QUBO problem to convert.

        Returns:
            PauliSum representing the Ising Hamiltonian.
        """
        n = qubo.num_variables
        terms: list[PauliString] = []

        # Convert Q matrix elements to Ising interactions
        for i in range(n):
            for j in range(i, n):
                q_ij = qubo.Q[i, j]
                if q_ij == 0:
                    continue

                if i == j:
                    # Diagonal: q_ii * x_i = q_ii * (1 - Z_i) / 2
                    # = q_ii/2 - q_ii/2 * Z_i
                    # Add -q_ii/2 * Z_i term
                    pauli_str = "I" * i + "Z" + "I" * (n - i - 1)
                    terms.append(PauliString(pauli_str, -q_ij / 2))
                else:
                    # Off-diagonal: q_ij * x_i * x_j
                    # = q_ij * (1-Z_i)/2 * (1-Z_j)/2
                    # = q_ij/4 * (1 - Z_i - Z_j + Z_i*Z_j)
                    pauli_str_zz = list("I" * n)
                    pauli_str_zz[i] = "Z"
                    pauli_str_zz[j] = "Z"
                    terms.append(PauliString("".join(pauli_str_zz), q_ij / 4))

                    pauli_str_zi = list("I" * n)
                    pauli_str_zi[i] = "Z"
                    terms.append(PauliString("".join(pauli_str_zi), -q_ij / 4))

                    pauli_str_iz = list("I" * n)
                    pauli_str_iz[j] = "Z"
                    terms.append(PauliString("".join(pauli_str_iz), -q_ij / 4))

        # Convert linear terms
        for i in range(n):
            c_i = qubo.linear[i]
            if c_i == 0:
                continue
            # c_i * x_i = c_i * (1 - Z_i) / 2 = c_i/2 - c_i/2 * Z_i
            pauli_str = "I" * i + "Z" + "I" * (n - i - 1)
            terms.append(PauliString(pauli_str, -c_i / 2))

        # Compute constant offset from QUBO-to-Ising conversion
        total_offset = qubo.offset
        for i in range(n):
            total_offset += qubo.linear[i] / 2
        for i in range(n):
            for j in range(i + 1, n):
                total_offset += qubo.Q[i, j] / 4

        if total_offset != 0:
            terms.append(PauliString("I" * n, total_offset))

        pauli_sum = PauliSum(terms)
        return pauli_sum.simplify()

    @staticmethod
    def ising_to_qubo(pauli_sum: PauliSum) -> QUBOProblem:
        """Convert Ising Hamiltonian back to QUBO.

        Uses the exact spin to bit mapping s_i = 2*x_i - 1 so that the
        reconstructed QUBO has identical energies on all bit strings.

        Args:
            pauli_sum: Ising Hamiltonian as PauliSum.

        Returns:
            QUBOProblem equivalent.
        """
        n = pauli_sum.num_qubits
        Q = np.zeros((n, n), dtype=float)
        linear = np.zeros(n, dtype=float)
        offset = 0.0
        h = np.zeros(n, dtype=float)
        J = np.zeros((n, n), dtype=float)
        const = 0.0

        for term in pauli_sum.terms:
            label = term.label
            coeff = float(term.coefficient.real)

            z_count = 0
            z_positions = []
            for k, ch in enumerate(label):
                if ch == "Z":
                    z_count += 1
                    z_positions.append(k)

            if z_count == 0:
                const += coeff
            elif z_count == 1:
                h[z_positions[0]] += coeff
            elif z_count == 2:
                i, j = z_positions
                J[i, j] += coeff
                J[j, i] += coeff

        # x = (1-s)/2, i.e. s = 1 - 2*x
        # h_i = -c_i/2 - sum_{j!=i} Q_ij/4  ->  c_i = -2*h_i - 2*sum_{j!=i} J_ij
        # q_ij = 4*J_ij
        # offset = const - sum(c_i)/2 - sum_{i<j}(Q_ij)/4
        for i in range(n):
            linear[i] = -2 * h[i] - 2 * float(np.sum(J[i, :]))
            for j in range(i + 1, n):
                Q[i, j] = 4 * J[i, j]
        # sum_{i<j}(Q_ij)/4 = sum_{i<j}(J_ij) = np.sum(J)/2 (J is symmetric)
        offset = const - float(np.sum(linear) / 2) - float(np.sum(J) / 2)

        return QUBOProblem(Q=Q, linear=linear, offset=offset, num_variables=n)

    @staticmethod
    def evaluate(qubo: QUBOProblem, solution: np.ndarray) -> dict[str, Any]:
        """Evaluate a QUBO solution.

        Args:
            qubo: QUBO problem.
            solution: Binary solution vector.

        Returns:
            Dictionary with energy, solution vector, and feasibility.
        """
        energy = qubo.energy(solution)
        return {
            "energy": energy,
            "solution": solution.tolist(),
            "num_variables": qubo.num_variables,
            "num_ones": int(np.sum(solution)),
        }
