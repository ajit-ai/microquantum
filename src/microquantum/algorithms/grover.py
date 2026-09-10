"""Grover's search algorithm.

Implements Grover's quantum search algorithm for finding marked items
in an unstructured database. Achieves quadratic speedup over classical
search: O(sqrt(N)) vs O(N) queries.

The algorithm consists of:
1. Initialize uniform superposition over all basis states
2. Repeat O(sqrt(N)) times:
   a. Apply oracle: flip phase of marked state
   b. Apply diffusion: reflect amplitudes about the mean
3. Measure to obtain the marked state with high probability
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit
from ..core.operators import Operator


@dataclass
class GroverResult(JSONSerializable):
    """Result from Grover's search algorithm.

    Attributes:
        circuit: The quantum circuit used for the search.
        num_qubits: Number of qubits in the search register.
        target: Target state(s) being searched for.
        num_iterations: Number of Grover iterations used.
        success_probability: Probability of measuring the target state.
        most_probable: The most probable measurement outcome.
        probabilities: Full probability distribution over all states.
    """
    circuit: QuantumCircuit
    num_qubits: int
    target: Union[int, list[int]]
    num_iterations: int
    success_probability: float
    most_probable: int
    probabilities: dict[str, float] = field(default_factory=dict)


class GroverSearch:
    """Grover's search algorithm.

    Finds marked items in an unstructured database using quantum
    amplitude amplification. Requires O(sqrt(N)) queries instead
    of the classical O(N).

    The oracle can be specified in two ways:
    - As an integer target state (single target search)
    - As a callable oracle function that returns a QuantumCircuit

    Args:
        num_qubits: Number of qubits (database has 2^num_qubits items).
        target: Target state index, list of target states, or a callable
            that builds the oracle circuit. If None, num_targets must
            be set and random targets are used.
        num_iterations: Number of Grover iterations. If None, uses
            the optimal floor(pi/4 * sqrt(N / num_targets)).
        num_targets: Number of marked items (used when target is None).
    """

    def __init__(
        self,
        num_qubits: int,
        target: Optional[Union[int, list[int], Callable[[int], QuantumCircuit]]] = None,
        num_iterations: Optional[int] = None,
        num_targets: int = 1,
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"Need >= 1 qubit, got {num_qubits}")

        self._num_qubits = num_qubits
        self._N = 2**num_qubits
        self._num_targets = num_targets

        if target is None:
            self._targets = [0]
            self._oracle_fn: Optional[Callable[[int], QuantumCircuit]] = None
        elif callable(target):
            self._targets = list(range(num_targets)) if num_targets > 1 else [0]
            self._oracle_fn = target
        elif isinstance(target, int):
            if target < 0 or target >= self._N:
                raise ValueError(
                    f"Target {target} out of range for {num_qubits}-qubit system "
                    f"(valid: 0..{self._N - 1})"
                )
            self._targets = [target]
            self._oracle_fn = None
        else:
            for t in target:
                if t < 0 or t >= self._N:
                    raise ValueError(
                        f"Target {t} out of range for {num_qubits}-qubit system"
                    )
            self._targets = list(target)
            self._num_targets = len(target)
            self._oracle_fn = None

        if num_iterations is None:
            self._num_iterations = self._optimal_iterations(
                self._N, self._num_targets
            )
        else:
            if num_iterations < 1:
                raise ValueError(f"Need >= 1 iteration, got {num_iterations}")
            self._num_iterations = num_iterations

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def targets(self) -> list[int]:
        return list(self._targets)

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    @staticmethod
    def _optimal_iterations(N: int, num_targets: int) -> int:
        """Compute optimal number of Grover iterations."""
        if num_targets >= N:
            return 1
        return max(1, int(math.floor(math.pi / 4 * math.sqrt(N / num_targets))))

    def build_circuit(self) -> QuantumCircuit:
        """Build the Grover search circuit.

        Returns:
            QuantumCircuit implementing Grover's algorithm.
        """
        n = self._num_qubits
        qc = QuantumCircuit(n)

        # Step 1: Initialize uniform superposition
        for i in range(n):
            qc.h(i)

        # Step 2: Grover iterations
        for _ in range(self._num_iterations):
            # Oracle
            qc = self._apply_oracle(qc)
            # Diffusion
            qc = self._apply_diffusion(qc)

        return qc

    def _apply_oracle(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Apply the oracle that marks the target state(s)."""
        if self._oracle_fn is not None:
            oracle_qc = self._oracle_fn(self._num_qubits)
            return qc + oracle_qc

        # Build diagonal oracle: -1 at target indices, +1 elsewhere
        n = self._num_qubits
        dim = 2**n
        matrix = np.eye(dim, dtype=np.complex128)
        for t in self._targets:
            matrix[t, t] = -1.0

        oracle_op = Operator(matrix)
        qc.append(oracle_op, list(range(n)))
        return qc

    def _apply_diffusion(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Apply the diffusion operator: 2|s><s| - I."""
        n = self._num_qubits
        dim = 2**n
        matrix = np.full((dim, dim), 2.0 / dim, dtype=np.complex128)
        matrix -= np.eye(dim, dtype=np.complex128)

        diffusion_op = Operator(matrix)
        qc.append(diffusion_op, list(range(n)))
        return qc

    def run(self) -> GroverResult:
        """Execute Grover's search algorithm.

        Returns:
            GroverResult with the search outcome.
        """
        qc = self.build_circuit()
        state = qc.run()

        probs_array = np.abs(state.amplitudes) ** 2

        # Build probability dict
        probabilities: dict[str, float] = {}
        for i in range(2**self._num_qubits):
            bitstring = format(i, f"0{self._num_qubits}b")
            probabilities[bitstring] = float(probs_array[i])

        most_probable = int(np.argmax(probs_array))
        success_prob = float(sum(probs_array[t] for t in self._targets))

        return GroverResult(
            circuit=qc,
            num_qubits=self._num_qubits,
            target=self._targets if len(self._targets) > 1 else self._targets[0],
            num_iterations=self._num_iterations,
            success_probability=success_prob,
            most_probable=most_probable,
            probabilities=probabilities,
        )

    def __repr__(self) -> str:
        return (
            f"GroverSearch(qubits={self._num_qubits}, "
            f"targets={self._targets}, "
            f"iterations={self._num_iterations})"
        )
