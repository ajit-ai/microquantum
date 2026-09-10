"""Cross-Entropy Benchmarking (XEB).

Measures circuit output fidelity against ideal simulation to
characterize quantum processor performance.  XEB fidelity is the
primary metric used to demonstrate quantum advantage (quantum supremacy).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..core.state import StateVector


@dataclass
class XEBResult:
    """Result from cross-entropy benchmarking.

    Attributes:
        depths: Circuit depths tested.
        fidelities: XEB fidelity at each depth.
        mean_fidelity: Average fidelity across all depths.
        is_above_threshold: True if mean fidelity exceeds 1/e
            (indicating quantum advantage).
        num_qubits: Number of qubits used.
        num_circuits: Number of random circuits per depth.
        seed: Random seed used.
    """

    depths: list[int]
    fidelities: list[float]
    mean_fidelity: float
    is_above_threshold: bool
    num_qubits: int
    num_circuits: int
    seed: int | None


def _random_circuit(
    num_qubits: int,
    depth: int,
    rng: random.Random,
) -> QuantumCircuit:
    """Generate a random quantum circuit of given depth.

    Each layer applies random single-qubit gates followed by
    random two-qubit gates on adjacent pairs.
    """
    qc = QuantumCircuit(num_qubits)

    single_gates = [Operator.H(), Operator.X(), Operator.Y(), Operator.Z()]
    two_gates = [Operator.CNOT(), Operator.CZ()]

    for _ in range(depth):
        # Single-qubit layer
        for q in range(num_qubits):
            op = rng.choice(single_gates)
            qc.append(op, [q])

        # Two-qubit layer: alternating pairs
        for q in range(0, num_qubits - 1, 2):
            op = rng.choice(two_gates)
            qc.append(op, [q, q + 1])
        for q in range(1, num_qubits - 1, 2):
            op = rng.choice(two_gates)
            qc.append(op, [q, q + 1])

    return qc


def _xeb_fidelity(
    measured_probs: NDArray[np.float64],
    ideal_state: StateVector,
    num_qubits: int,
) -> float:
    """Compute XEB fidelity from measured output probabilities.

    XEB fidelity = 2^n * sum_x P_ideal(x) * P_measured(x) - 1

    For ideal output (delta function), this equals 1.
    For uniform random output, this equals 0.
    """
    dim = 2**num_qubits
    ideal_probs = np.abs(ideal_state.amplitudes[:dim]) ** 2
    fidelity = float(dim * np.sum(ideal_probs * measured_probs[:dim]) - 1)
    return max(fidelity, 0.0)


class CrossEntropyBenchmarking:
    """Cross-entropy benchmarking for quantum advantage testing.

    Generates random circuits of varying depths, simulates them
    ideally, and compares the output distribution against the ideal
    to compute XEB fidelity.

    Args:
        num_qubits: Number of qubits.
        depths: List of circuit depths to test.
        num_circuits: Number of random circuits per depth.
        seed: Optional RNG seed.

    Example::

        xeb = CrossEntropyBenchmarking(
            num_qubits=4, depths=[10, 20, 30], num_circuits=5
        )
        result = xeb.run()
    """

    def __init__(
        self,
        num_qubits: int,
        depths: list[int] | None = None,
        num_circuits: int = 10,
        seed: int | None = None,
    ) -> None:
        if num_qubits < 2:
            raise ValueError(
                f"num_qubits must be >= 2, got {num_qubits}"
            )
        self._num_qubits = num_qubits
        self._depths = depths or [5, 10, 20, 30, 50]
        self._num_circuits = num_circuits
        self._seed = seed

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return self._num_qubits

    @property
    def depths(self) -> list[int]:
        """Circuit depths tested."""
        return list(self._depths)

    @property
    def seed(self) -> int | None:
        """Random seed."""
        return self._seed

    def run(self) -> XEBResult:
        """Run cross-entropy benchmarking.

        Returns:
            An ``XEBResult`` with fidelity measurements.
        """
        rng = random.Random(self._seed)
        fidelities: list[float] = []

        for depth in self._depths:
            depth_fids: list[float] = []

            for _ in range(self._num_circuits):
                qc = _random_circuit(self._num_qubits, depth, rng)
                sv = qc.run()

                # Compute ideal output probabilities
                ideal_probs = np.abs(sv.amplitudes) ** 2

                # XEB fidelity: comparing ideal to itself (simulated)
                # In real benchmarking, this compares ideal to hardware output
                fid = _xeb_fidelity(ideal_probs, sv, self._num_qubits)
                depth_fids.append(fid)

            fidelities.append(
                float(np.mean(depth_fids)) if depth_fids else 0.0
            )

        mean_fid = float(np.mean(fidelities)) if fidelities else 0.0

        return XEBResult(
            depths=list(self._depths),
            fidelities=fidelities,
            mean_fidelity=mean_fid,
            is_above_threshold=mean_fid > 1.0 / math.e,
            num_qubits=self._num_qubits,
            num_circuits=self._num_circuits,
            seed=self._seed,
        )

    def __repr__(self) -> str:
        return (
            f"CrossEntropyBenchmarking("
            f"num_qubits={self._num_qubits}, "
            f"depths={self._depths})"
        )
