"""Density matrix representation for quantum states."""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .state import StateVector


class DensityMatrix:
    """A quantum state represented as a density matrix.

    Density matrices generalize state vectors to represent mixed states
    and are essential for noise modeling. For a pure state |psi>,
    rho = |psi><psi|.

    Attributes:
        num_qubits: Number of qubits in the system.
        dim: Dimension of the matrix (2^num_qubits).
        matrix: The density matrix in complex128 dtype.
    """

    def __init__(
        self,
        num_qubits: int,
        matrix: Optional[NDArray[np.complex128]] = None,
    ) -> None:
        """Initialize a density matrix.

        Args:
            num_qubits: Number of qubits. Must be >= 1.
            matrix: Optional density matrix of shape (dim, dim). If None,
                defaults to |0><0| (ground state).

        Raises:
            ValueError: If num_qubits < 1, matrix has wrong shape,
                or matrix is not positive semidefinite with trace 1.
        """
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")

        self._num_qubits = num_qubits
        self._dim = 2 ** num_qubits

        if matrix is not None:
            arr = np.asarray(matrix, dtype=np.complex128)
            if arr.shape != (self._dim, self._dim):
                raise ValueError(
                    f"Matrix must have shape ({self._dim}, {self._dim}), "
                    f"got {arr.shape}"
                )
            self._matrix = arr
        else:
            self._matrix = np.zeros((self._dim, self._dim), dtype=np.complex128)
            self._matrix[0, 0] = 1.0

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return self._num_qubits

    @property
    def dim(self) -> int:
        """Matrix dimension (2^num_qubits)."""
        return int(self._dim)

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """The density matrix."""
        return self._matrix

    @property
    def trace(self) -> complex:
        """Trace of the density matrix (should be 1 for physical states)."""
        return complex(np.trace(self._matrix))

    @property
    def is_pure(self) -> bool:
        """Check if the density matrix represents a pure state.

        A pure state satisfies Tr(rho^2) = 1.
        """
        rho_sq = self._matrix @ self._matrix
        tr_rho_sq = np.trace(rho_sq)
        return bool(abs(tr_rho_sq - 1.0) < 1e-9)

    @property
    def fidelity_with_ground(self) -> float:
        """Fidelity with the ground state |0...0>."""
        return float(np.real(self._matrix[0, 0]))

    @classmethod
    def from_statevector(cls, state: StateVector) -> DensityMatrix:
        """Create a pure-state density matrix from a state vector.

        Args:
            state: The quantum state vector.

        Returns:
            DensityMatrix equal to |psi><psi|.
        """
        amps = state.amplitudes.reshape(-1, 1)
        rho = DensityMatrix(
            num_qubits=state.num_qubits,
            matrix=amps @ amps.conj().T,
        )
        return rho

    @classmethod
    def from_label(cls, label: str) -> DensityMatrix:
        """Create a density matrix from a computational basis label.

        Args:
            label: Bitstring like "0", "1", "00", "101", etc.

        Returns:
            DensityMatrix for that basis state.
        """
        num_qubits = len(label)
        idx = int(label, 2)
        dim = 2 ** num_qubits
        rho = cls(num_qubits)
        rho._matrix = np.zeros((dim, dim), dtype=np.complex128)
        rho._matrix[idx, idx] = 1.0
        return rho

    def apply_unitary(self, unitary: NDArray[np.complex128]) -> DensityMatrix:
        """Apply a unitary transformation: rho' = U rho U^dagger.

        Args:
            unitary: Unitary gate matrix of shape (dim, dim).

        Returns:
            New DensityMatrix with the unitary applied.
        """
        u = np.asarray(unitary, dtype=np.complex128)
        new_rho = DensityMatrix(
            num_qubits=self._num_qubits,
            matrix=u @ self._matrix @ u.conj().T,
        )
        return new_rho

    def apply_kraus(self, operators: list[NDArray[np.complex128]]) -> DensityMatrix:
        """Apply a quantum channel via Kraus operators.

        rho' = sum_k E_k rho E_k^dagger

        Args:
            operators: List of Kraus operator matrices.

        Returns:
            New DensityMatrix after the channel.
        """
        new_rho = np.zeros_like(self._matrix)
        for e_k in operators:
            e = np.asarray(e_k, dtype=np.complex128)
            new_rho += e @ self._matrix @ e.conj().T
        return DensityMatrix(num_qubits=self._num_qubits, matrix=new_rho)

    def expectation(self, observable: NDArray[np.complex128]) -> complex:
        """Compute Tr(rho * H).

        Args:
            observable: Hermitian observable matrix.

        Returns:
            Expectation value <H>.
        """
        h = np.asarray(observable, dtype=np.complex128)
        return complex(np.trace(self._matrix @ h))

    def partial_trace(self, trace_over: list[int]) -> DensityMatrix:
        """Trace out specified qubits.

        Args:
            trace_over: List of qubit indices to trace out.

        Returns:
            Reduced density matrix.
        """
        n = self._num_qubits
        rho_tensor = self._matrix.reshape((2,) * n + (2,) * n)

        # Sort indices to trace over in descending order
        sorted_over = sorted(trace_over, reverse=True)
        for q in sorted_over:
            dim_left = 2 ** q
            dim_right = 2 ** (n - q - 1)
            # Trace over axis q (physical) and axis n+q (bra)
            rho_tensor = np.trace(rho_tensor, axis1=q, axis2=n + q)

        remaining = n - len(trace_over)
        if remaining == 0:
            # Fully traced out — return scalar as 1x1 matrix
            return DensityMatrix(1, matrix=rho_tensor.reshape(1, 1))

        new_dim = 2 ** remaining
        return DensityMatrix(
            num_qubits=remaining,
            matrix=rho_tensor.reshape(new_dim, new_dim),
        )

    def copy(self) -> DensityMatrix:
        """Return a deep copy."""
        return DensityMatrix(
            num_qubits=self._num_qubits,
            matrix=copy.deepcopy(self._matrix),
        )

    def __repr__(self) -> str:
        purity = "pure" if self.is_pure else "mixed"
        return (
            f"DensityMatrix(num_qubits={self._num_qubits}, "
            f"trace={self.trace:.4f}, {purity})"
        )

    def __str__(self) -> str:
        lines = [
            f"DensityMatrix ({self._num_qubits} qubits, "
            f"{self._dim}x{self._dim})",
            f"Trace: {self.trace:.6f}",
            f"Pure: {self.is_pure}",
        ]
        return "\n".join(lines)
