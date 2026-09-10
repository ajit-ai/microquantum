"""CLOPS (Circuit Layer Operations Per Second) benchmark.

Measures the speed of quantum circuit execution by timing
repeated execution of randomized circuits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.circuit import QuantumCircuit
from .quantum_volume import BenchmarkResult


class CLOPSBenchmark:
    """CLOPS benchmark for circuit execution speed.

    Measures how many circuit layer operations per second (CLOPS)
    the quantum system can process. Useful for comparing execution
    backends.

    Args:
        num_qubits: Number of qubits per circuit.
        num_layers: Number of layers per circuit.
        num_circuits: Number of random circuits to execute.
        seed: Random seed.
    """

    def __init__(
        self,
        num_qubits: int = 5,
        num_layers: int = 10,
        num_circuits: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"Need >= 1 qubit, got {num_qubits}")
        if num_layers < 1:
            raise ValueError(f"Need >= 1 layer, got {num_layers}")
        if num_circuits < 1:
            raise ValueError(f"Need >= 1 circuit, got {num_circuits}")

        self._num_qubits = num_qubits
        self._num_layers = num_layers
        self._num_circuits = num_circuits
        self._rng = np.random.RandomState(seed)

    def generate_circuit(self) -> QuantumCircuit:
        """Generate a random benchmark circuit.

        Returns:
            Random QuantumCircuit with the specified depth.
        """
        n = self._num_qubits
        qc = QuantumCircuit(n)

        for _ in range(self._num_layers):
            # Random single-qubit rotations
            for i in range(n):
                gate = self._rng.choice(["ry", "rz"])
                angle = self._rng.uniform(0, 2 * np.pi)
                if gate == "ry":
                    qc.ry(float(angle), int(i))
                else:
                    qc.rz(float(angle), int(i))

            # Random CNOT pattern
            for i in range(n - 1):
                if self._rng.random() < 0.5:
                    qc.cx(i, i + 1)

        return qc

    def run(self) -> BenchmarkResult:
        """Run the CLOPS benchmark.

        Executes random circuits and measures total time and throughput.

        Returns:
            BenchmarkResult with CLOPS metric.
        """
        total_layers = 0
        start_time = time.perf_counter()

        for _ in range(self._num_circuits):
            qc = self.generate_circuit()
            qc.run()
            total_layers += self._num_layers

        elapsed = time.perf_counter() - start_time
        clops = total_layers / elapsed if elapsed > 0 else 0.0

        return BenchmarkResult(
            metric_name="clops",
            value=clops,
            num_qubits=self._num_qubits,
            depth=self._num_layers,
            raw_data={
                "total_circuits": self._num_circuits,
                "total_layers": total_layers,
                "elapsed_seconds": elapsed,
            },
            timestamp=time.time(),
        )

    def __repr__(self) -> str:
        return (
            f"CLOPSBenchmark(qubits={self._num_qubits}, "
            f"layers={self._num_layers}, circuits={self._num_circuits})"
        )
