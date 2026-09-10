"""Randomized benchmarking for noise characterization.

Estimates the average error rate of quantum gates by applying random
sequences of Clifford group elements and measuring survival probability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product as iterproduct

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..core.state import StateVector


@dataclass
class RandomizedBenchmarkingResult:
    """Result from a randomized benchmarking experiment.

    Attributes:
        num_cliffords: Sequence lengths tested.
        survival_probabilities: Average survival probability per length.
        depolarizing_parameter: Estimated depolarizing parameter p.
        average_gate_fidelity: Estimated average gate fidelity F.
        error_per_gate: Estimated error per gate 1 - F.
        fit_params: Tuple (A, p) from exponential fit A * p^m + B.
        seed: Random seed used, or None.
    """

    num_cliffords: list[int]
    survival_probabilities: list[float]
    depolarizing_parameter: float
    average_gate_fidelity: float
    error_per_gate: float
    fit_params: tuple[float, float]
    seed: int | None


class RandomizedBenchmarking:
    """Randomized benchmarking for noise characterization.

    Estimates the average error rate of a gate set by measuring the
    decay of survival probability as random Clifford sequences grow
    longer.  The depolarizing parameter p is extracted from an
    exponential fit and converted to average gate fidelity.

    Args:
        num_qubits: Number of qubits.  For 1 qubit the full 24-element
            Clifford group is generated.  For more qubits, tensor
            products of single-qubit Cliffords are used.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, num_qubits: int = 1, seed: int | None = None) -> None:
        if num_qubits < 1:
            raise ValueError(f"Need >= 1 qubit, got {num_qubits}")
        self._num_qubits = num_qubits
        self._seed = seed
        self._rng = random.Random(seed)
        self._clifford_group: list[QuantumCircuit] | None = None

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return self._num_qubits

    @property
    def seed(self) -> int | None:
        """Random seed."""
        return self._seed

    # ------------------------------------------------------------------
    # Clifford group generation
    # ------------------------------------------------------------------

    def generate_clifford_group(self) -> list[QuantumCircuit]:
        """Generate the Clifford group for this qubit count.

        For a single qubit the full 24-element group is returned.
        For multiple qubits the group consists of tensor products of
        single-qubit Cliffords (24^n elements).

        Returns:
            List of QuantumCircuit instances, each implementing one
            Clifford element.
        """
        if self._clifford_group is not None:
            return self._clifford_group

        single = self._single_qubit_cliffords()

        if self._num_qubits == 1:
            self._clifford_group = single
        else:
            self._clifford_group = self._multi_qubit_cliffords(single)

        return self._clifford_group

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _single_qubit_cliffords(self) -> list[QuantumCircuit]:
        """Compute the 24 single-qubit Clifford elements by closure."""
        H_mat = Operator.H().matrix
        S_mat = Operator.S().matrix
        generators = [H_mat, S_mat]

        group: list[NDArray[np.complex128]] = [
            np.eye(2, dtype=np.complex128)
        ]

        changed = True
        while changed:
            changed = False
            new_elements: list[NDArray[np.complex128]] = []
            for g in group:
                for gen in generators:
                    candidate = gen @ g
                    if not self._is_same_gate(candidate, group):
                        if not self._is_same_gate(candidate, new_elements):
                            new_elements.append(candidate)
                            changed = True
            group.extend(new_elements)

        cliffords: list[QuantumCircuit] = []
        for mat in group:
            op = Operator(mat, name="clifford")
            qc = QuantumCircuit(1)
            qc.append(op, [0])
            cliffords.append(qc)
        return cliffords

    @staticmethod
    def _is_same_gate(
        mat: NDArray[np.complex128],
        group: list[NDArray[np.complex128]],
    ) -> bool:
        """Check whether *mat* equals a group element up to global phase."""
        for g in group:
            product = mat @ g.conj().T
            flat = product.flatten()
            idx = int(np.argmax(np.abs(flat)))
            phase = flat[idx]
            if abs(phase) < 1e-10:
                continue
            if np.allclose(
                product,
                phase * np.eye(2, dtype=np.complex128),
                atol=1e-8,
            ):
                return True
        return False

    def _multi_qubit_cliffords(
        self, single: list[QuantumCircuit]
    ) -> list[QuantumCircuit]:
        """Build multi-qubit Cliffords as tensor products."""
        single_ops: list[Operator] = [qc.gates[0][0] for qc in single]
        n = self._num_qubits
        num_single = len(single_ops)

        multi: list[QuantumCircuit] = []
        for indices in iterproduct(range(num_single), repeat=n):
            qc = QuantumCircuit(n)
            for qubit, ci in enumerate(indices):
                qc.append(single_ops[ci], [qubit])
            multi.append(qc)
        return multi

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def generate_sequence(self, length: int) -> QuantumCircuit:
        """Generate a random Clifford sequence of given length.

        The circuit applies *length* random Cliffords followed by the
        inverse of their composition so that the overall unitary is
        the identity (in the absence of noise).

        Args:
            length: Number of random Cliffords in the sequence.

        Returns:
            Combined QuantumCircuit (sequence + recovery).
        """
        cliffords = self.generate_clifford_group()
        num_cliffords = len(cliffords)

        qc = QuantumCircuit(self._num_qubits)
        for _ in range(length):
            idx = self._rng.randint(0, num_cliffords - 1)
            qc = qc + cliffords[idx]

        recovery = qc.inverse()
        return qc + recovery

    # ------------------------------------------------------------------
    # Benchmark execution
    # ------------------------------------------------------------------

    def run(
        self,
        sequence_lengths: list[int] | None = None,
        num_samples: int = 30,
    ) -> RandomizedBenchmarkingResult:
        """Run the randomized benchmarking experiment.

        For each sequence length *m* in *sequence_lengths*, generate
        *num_samples* random sequences, simulate them, and compute
        the average survival probability (probability of |0…0⟩).

        The survival probabilities are fit to
        ``P(m) = A * p^m + B`` with ``B = 1/d`` (``d = 2^n``), and
        the depolarizing parameter *p* is converted to average gate
        fidelity ``F = (d*p + 1) / (d + 1)``.

        Args:
            sequence_lengths: Sequence lengths to test.  Defaults to
                ``[1, 2, 4, 8, 16, 32, 64]``.
            num_samples: Random sequences per length.

        Returns:
            RandomizedBenchmarkingResult with fitted parameters.
        """
        if sequence_lengths is None:
            sequence_lengths = [1, 2, 4, 8, 16, 32, 64]
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        survival_probs: list[float] = []

        for m in sequence_lengths:
            probs: list[float] = []
            for _ in range(num_samples):
                circuit = self.generate_sequence(m)
                state = circuit.run()
                survival_prob = float(np.abs(state.amplitudes[0]) ** 2)
                probs.append(survival_prob)
            survival_probs.append(float(np.mean(probs)))

        A, p = self._fit_exponential(sequence_lengths, survival_probs)

        d = 2 ** self._num_qubits
        fidelity = (d * p + 1.0) / (d + 1.0)

        return RandomizedBenchmarkingResult(
            num_cliffords=list(sequence_lengths),
            survival_probabilities=survival_probs,
            depolarizing_parameter=p,
            average_gate_fidelity=fidelity,
            error_per_gate=1.0 - fidelity,
            fit_params=(A, p),
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_exponential(
        self,
        lengths: list[int],
        probs: list[float],
    ) -> tuple[float, float]:
        """Fit survival probabilities to A * p^m + B with B = 1/d.

        Uses a linearised least-squares fit on the log-transformed
        residuals after subtracting the asymptotic value B.

        Returns:
            Tuple (A, p).
        """
        d = 2 ** self._num_qubits
        B = 1.0 / d

        valid: list[tuple[int, float]] = []
        for m, prob in zip(lengths, probs):
            if prob > B + 1e-12:
                valid.append((m, prob))

        if len(valid) < 2:
            return 1.0, 1.0

        ms = np.array([v[0] for v in valid], dtype=np.float64)
        ps = np.array([v[1] for v in valid], dtype=np.float64)

        log_vals = np.log(ps - B)
        coeffs = np.polyfit(ms, log_vals, 1)

        slope = float(np.real(coeffs[0]))
        intercept = float(np.real(coeffs[1]))

        p = min(max(float(np.exp(slope)), 0.0), 1.0)
        A = float(np.exp(intercept))

        return A, p

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RandomizedBenchmarking(num_qubits={self._num_qubits}, "
            f"seed={self._seed})"
        )
