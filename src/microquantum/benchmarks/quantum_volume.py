"""Quantum volume benchmark.

Measures the effective quantum volume of a quantum computer by
testing random square circuits of increasing depth.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator


@dataclass
class BenchmarkResult:
    """Result from a quantum benchmark.

    Attributes:
        metric_name: Name of the benchmark metric.
        value: Measured metric value.
        num_qubits: Number of qubits used.
        depth: Circuit depth.
        success_rate: Fraction of successful circuits.
        confidence: Confidence level (0 to 1).
        raw_data: Additional raw measurement data.
        timestamp: Time of benchmark execution.
    """
    metric_name: str
    value: float
    num_qubits: int = 0
    depth: int = 0
    success_rate: float = 0.0
    confidence: float = 0.0
    raw_data: dict = field(default_factory=dict)
    timestamp: float = 0.0


class QuantumVolumeBenchmark:
    """Quantum volume benchmark.

    Tests random circuits of size d x d (d qubits, depth d) with
    random permutations. The quantum volume is the largest d for
    which the heavy output probability exceeds 2/3.

    Args:
        max_qubits: Maximum number of qubits to test.
        num_shots: Number of shots per circuit.
        seed: Random seed for reproducibility.
        target_success: Target success probability (default 2/3).
    """

    def __init__(
        self,
        max_qubits: int = 10,
        num_shots: int = 1024,
        seed: Optional[int] = None,
        target_success: float = 2.0 / 3.0,
    ) -> None:
        if max_qubits < 1:
            raise ValueError(f"Need >= 1 qubit, got {max_qubits}")
        self._max_qubits = max_qubits
        self._num_shots = num_shots
        self._rng = random.Random(seed)
        self._target_success = target_success

    @property
    def max_qubits(self) -> int:
        return self._max_qubits

    def generate_random_circuit(self, depth: int) -> QuantumCircuit:
        """Generate a random d x d circuit for quantum volume testing.

        Each layer consists of random single-qubit gates followed by
        a random permutation of qubits with random two-qubit gates.

        Args:
            depth: Circuit depth (also number of qubits).

        Returns:
            Random QuantumCircuit.
        """
        n = depth
        qc = QuantumCircuit(n)

        for _ in range(depth):
            # Random single-qubit gates (Ry, Rz)
            for i in range(n):
                gate_type = self._rng.choice(["ry", "rz"])
                angle = self._rng.uniform(0, 2 * np.pi)
                if gate_type == "ry":
                    qc.ry(float(angle), int(i))
                else:
                    qc.rz(float(angle), int(i))

            # Random permutation
            perm = list(range(n))
            self._rng.shuffle(perm)

            # Two-qubit gates between permuted pairs
            for i in range(0, n - 1, 2):
                q1, q2 = int(perm[i]), int(perm[i + 1])
                qc.cx(q1, q2)

        return qc

    def compute_heavy_output(
        self, circuit: QuantumCircuit, state_probs: dict[str, float]
    ) -> tuple[float, list[str]]:
        """Compute the heavy output probability for a circuit.

        The heavy output is the set of bitstrings whose probability
        exceeds the median probability.

        Args:
            circuit: The quantum circuit.
            state_probs: Probability distribution from simulation.

        Returns:
            Tuple of (heavy_output_probability, list_of_heavy_outputs).
        """
        if not state_probs:
            return 0.0, []

        probs = sorted(state_probs.values())
        median_idx = len(probs) // 2
        median_prob = probs[median_idx]

        heavy_outputs = [
            bitstring for bitstring, prob in state_probs.items()
            if prob > median_prob
        ]

        total_prob = sum(
            state_probs.get(b, 0.0) for b in heavy_outputs
        )

        return total_prob, heavy_outputs

    def run_single(self, depth: int) -> BenchmarkResult:
        """Run a single quantum volume test at given depth.

        Args:
            depth: Circuit depth to test.

        Returns:
            BenchmarkResult with success probability.
        """
        qc = self.generate_random_circuit(depth)

        # Simulate
        state = qc.run()
        probs = {format(i, f"0{qc.num_qubits}b"): float(abs(state.amplitudes[i]) ** 2)
                 for i in range(2**qc.num_qubits)}

        hop, heavy = self.compute_heavy_output(qc, probs)
        success = hop > self._target_success

        return BenchmarkResult(
            metric_name="quantum_volume",
            value=hop,
            num_qubits=depth,
            depth=depth,
            success_rate=hop,
            confidence=1.0 if success else 0.0,
            raw_data={
                "heavy_outputs": len(heavy),
                "total_outputs": len(probs),
                "target": self._target_success,
            },
            timestamp=time.time(),
        )

    def run(self) -> BenchmarkResult:
        """Run the full quantum volume benchmark.

        Tests increasing depths until failure or max_qubits.

        Returns:
            BenchmarkResult with the quantum volume.
        """
        quantum_volume = 0
        all_results = {}

        for depth in range(1, self._max_qubits + 1):
            result = self.run_single(depth)
            all_results[depth] = result

            if result.success_rate > self._target_success:
                quantum_volume = 2**depth
            else:
                break

        return BenchmarkResult(
            metric_name="quantum_volume",
            value=float(quantum_volume),
            num_qubits=int(np.log2(quantum_volume)) if quantum_volume > 0 else 0,
            depth=int(np.log2(quantum_volume)) if quantum_volume > 0 else 0,
            success_rate=all_results.get(1, BenchmarkResult("", 0)).success_rate,
            raw_data={"all_depths": all_results},
            timestamp=time.time(),
        )

    def __repr__(self) -> str:
        return (
            f"QuantumVolumeBenchmark(max_qubits={self._max_qubits}, "
            f"shots={self._num_shots})"
        )
