"""Mirror-circuit benchmarking: scalable fidelity estimation.

A mirror circuit runs a random Clifford-ish layer followed by its exact
inverse, so the ideal outcome is always ``|0...0>``.  The adjusted
all-zero probability (polarization) estimates the layer fidelity and
scales to widths where full tomography is infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..backends.statevector import StatevectorBackend
from ..core.circuit import QuantumCircuit
from .quantum_volume import BenchmarkResult

__all__ = [
    "MirrorBenchmarking",
]


@dataclass
class _MirrorConfig:
    """Resolved mirror-benchmark configuration."""

    num_qubits: int
    depths: tuple[int, ...]
    num_circuits: int
    num_shots: int
    seed: Optional[int]


class MirrorBenchmarking:
    """Mirror-circuit fidelity benchmark.

    Args:
        num_qubits: Circuit width (must be >= 1).
        depths: Forward-layer depths to probe.
        num_circuits: Random mirrors per depth.
        num_shots: Shots per circuit execution.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depths: Sequence[int] = (1, 2, 4),
        num_circuits: int = 10,
        num_shots: int = 1024,
        seed: Optional[int] = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        cleaned = tuple(int(depth) for depth in depths)
        if not cleaned or any(depth < 1 for depth in cleaned):
            raise ValueError("depths must be non-empty positive integers")
        if num_circuits < 1:
            raise ValueError("num_circuits must be >= 1")
        if num_shots < 1:
            raise ValueError("num_shots must be >= 1")
        self._config = _MirrorConfig(
            num_qubits=num_qubits,
            depths=cleaned,
            num_circuits=num_circuits,
            num_shots=num_shots,
            seed=seed,
        )

    @property
    def num_qubits(self) -> int:
        """Circuit width."""
        return self._config.num_qubits

    def mirror_circuit(self, depth: int, seed: Optional[int] = None) -> QuantumCircuit:
        """Build one mirror circuit: random layer + exact inverse.

        The random layer uses H / X / CX gates; the inverse is obtained
        from :meth:`QuantumCircuit.inverse`, so the ideal output is the
        all-zero state by construction.
        """
        if depth < 1:
            raise ValueError("depth must be >= 1")
        rng = np.random.default_rng(seed if seed is not None else self._config.seed)
        forward = QuantumCircuit(self._config.num_qubits)
        for _ in range(depth):
            for qubit in range(self._config.num_qubits):
                choice = int(rng.integers(0, 3))
                if choice == 0:
                    forward.h(qubit)
                elif choice == 1:
                    forward.x(qubit)
                else:
                    forward.s(qubit)
            for qubit in range(self._config.num_qubits - 1):
                if bool(rng.integers(0, 2)):
                    forward.cx(qubit, qubit + 1)
        mirrored = forward + forward.inverse()
        mirrored.measure_all()
        return mirrored

    @staticmethod
    def polarization(counts: Mapping[str, int], num_qubits: int) -> float:
        """Adjusted all-zero probability in ``[0, 1]``.

        ``(P(0) - 1/D) / (1 - 1/D)`` with ``D = 2**num_qubits``: 1.0 for
        a perfect mirror, 0.0 for a fully depolarized output.
        """
        total = sum(counts.values())
        if total == 0:
            raise ValueError("counts must be non-empty")
        dim = 2**num_qubits
        ideal = counts.get("0" * num_qubits, 0) / total
        if dim == 1:
            return float(ideal)
        return float((ideal - 1.0 / dim) / (1.0 - 1.0 / dim))

    def run(self, backend: Any = None) -> BenchmarkResult:
        """Run the mirror suite and return the mean polarization.

        Args:
            backend: Backend exposing ``run(circuit, shots, seed)``;
                defaults to :class:`StatevectorBackend`.
        """
        runner = backend if backend is not None else StatevectorBackend()
        polarizations: list[float] = []
        per_depth: dict[str, float] = {}
        for depth in self._config.depths:
            values: list[float] = []
            for index in range(self._config.num_circuits):
                circuit = self.mirror_circuit(depth, seed=index)
                result = runner.run(
                    circuit, shots=self._config.num_shots, seed=self._config.seed
                )
                values.append(self.polarization(result.get_counts(), self._config.num_qubits))
            per_depth[str(depth)] = float(sum(values) / len(values))
            polarizations.extend(values)
        mean_value = float(sum(polarizations) / len(polarizations))
        return BenchmarkResult(
            metric_name="mirror_fidelity",
            value=mean_value,
            num_qubits=self._config.num_qubits,
            depth=max(self._config.depths),
            success_rate=mean_value,
            confidence=float(np.std(polarizations)) if len(polarizations) > 1 else 0.0,
            raw_data={"per_depth": per_depth, "num_circuits": self._config.num_circuits},
        )

    def __repr__(self) -> str:
        return (
            f"MirrorBenchmarking(qubits={self._config.num_qubits}, "
            f"depths={list(self._config.depths)})"
        )
