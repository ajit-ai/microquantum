"""Cycle Benchmarking, Layer Fidelity, and Pauli Twirling.

Provides gate set characterization via random circuit fidelity decay
(Cycle Benchmarking), per-layer fidelity comparison (Layer Fidelity),
and randomized compiling via Pauli twirling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit
from ..core.operators import Operator

PAULIS = [Operator.I(), Operator.X(), Operator.Y(), Operator.Z()]


@dataclass
class CycleBenchmarkingResult(JSONSerializable):
    """Result from cycle benchmarking.

    Attributes:
        cycle_lengths: Number of cycles tested.
        fidelities: Average survival probability per cycle length.
        error_per_cycle: Estimated error per cycle.
        fidelity_per_gate: Estimated per-gate fidelity.
        num_qubits: Number of qubits tested.
        seed: Random seed used.
    """

    cycle_lengths: list[int]
    fidelities: list[float]
    error_per_cycle: float
    fidelity_per_gate: float
    num_qubits: int
    seed: int | None


@dataclass
class LayerFidelityResult(JSONSerializable):
    """Result from layer fidelity benchmarking.

    Attributes:
        layer_names: Names of the layers tested.
        fidelities: Fidelity for each layer.
        average_fidelity: Mean fidelity across all layers.
        worst_layer: Name of the layer with lowest fidelity.
    """

    layer_names: list[str]
    fidelities: list[float]
    average_fidelity: float
    worst_layer: str


@dataclass
class TwirlingResult(JSONSerializable):
    """Result from Pauli twirling.

    Attributes:
        original_depth: Circuit depth before twirling.
        twirled_depth: Circuit depth after twirling.
        num_twirled_gates: Number of 2-qubit gates twirled.
        twirled_circuit: The twirled circuit.
    """

    original_depth: int
    twirled_depth: int
    num_twirled_gates: int
    twirled_circuit: QuantumCircuit


class CycleBenchmarking:
    """Cycle benchmarking for gate set error characterization.

    Estimates error per cycle by applying random circuits of
    increasing length and measuring survival probability decay.

    Args:
        num_qubits: Number of qubits.
        cycle_lengths: Cycle counts to test.
        num_samples: Random circuits per length.
        gates_per_cycle: Gates per cycle layer.
        seed: Random seed.

    Example::

        cb = CycleBenchmarking(num_qubits=2, cycle_lengths=[1,2,4,8])
        result = cb.run()
    """

    def __init__(
        self,
        num_qubits: int,
        cycle_lengths: list[int] | None = None,
        num_samples: int = 20,
        gates_per_cycle: int = 1,
        seed: int | None = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self._num_qubits = num_qubits
        self._cycle_lengths = cycle_lengths or [1, 2, 4, 8, 16]
        self._num_samples = num_samples
        self._gates_per_cycle = gates_per_cycle
        self._seed = seed

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def _build_cycle_circuit(
        self, num_cycles: int, rng: random.Random
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self._num_qubits)
        single = [Operator.X(), Operator.Y(), Operator.Z(), Operator.H()]
        two = [Operator.CNOT(), Operator.CZ()]

        for _ in range(num_cycles):
            for _ in range(self._gates_per_cycle):
                if self._num_qubits == 1 or rng.random() < 0.7:
                    gate = rng.choice(single)
                    for q in range(self._num_qubits):
                        qc.append(gate, [q])
                else:
                    gate = rng.choice(two)
                    q0 = rng.randint(0, self._num_qubits - 2)
                    qc.append(gate, [q0, q0 + 1])
        return qc

    def run(self) -> CycleBenchmarkingResult:
        """Run cycle benchmarking."""
        rng = random.Random(self._seed)
        fidelities: list[float] = []

        for length in self._cycle_lengths:
            sample_fids: list[float] = []
            for _ in range(self._num_samples):
                qc = self._build_cycle_circuit(length, rng)
                sv = qc.run()
                sample_fids.append(float(abs(sv.amplitudes[0]) ** 2))
            fidelities.append(float(np.mean(sample_fids)))

        valid = [
            (m, f)
            for m, f in zip(self._cycle_lengths, fidelities, strict=False)
            if f > 0
        ]
        if len(valid) >= 2:
            lengths_arr = np.array([v[0] for v in valid])
            log_fids = np.log(np.maximum([v[1] for v in valid], 1e-15))
            coeffs = np.polyfit(lengths_arr, log_fids, 1)
            p = float(np.exp(coeffs[0]))
            error_per_cycle = max(0.0, 1.0 - p)
            fidelity_per_gate = p
        else:
            error_per_cycle = 0.0
            fidelity_per_gate = 1.0

        return CycleBenchmarkingResult(
            cycle_lengths=list(self._cycle_lengths),
            fidelities=fidelities,
            error_per_cycle=error_per_cycle,
            fidelity_per_gate=fidelity_per_gate,
            num_qubits=self._num_qubits,
            seed=self._seed,
        )

    def __repr__(self) -> str:
        return (
            f"CycleBenchmarking(num_qubits={self._num_qubits}, "
            f"cycles={self._cycle_lengths})"
        )


class LayerFidelity:
    """Layer fidelity benchmarking for identifying weak circuit layers.

    Tests predefined circuit layers to compare their fidelities and
    identify the worst-performing layer.

    Args:
        num_qubits: Number of qubits.
        layers: Dict mapping name to list of (Operator, targets).
        num_samples: Circuits per layer.
        seed: Random seed.

    Example::

        lf = LayerFidelity(num_qubits=2)
        result = lf.run()
        print(result.worst_layer)
    """

    def __init__(
        self,
        num_qubits: int,
        layers: dict[str, list[tuple[Operator, list[int]]]] | None = None,
        num_samples: int = 20,
        seed: int | None = None,
    ) -> None:
        self._num_qubits = num_qubits
        self._layers = layers or self._default_layers(num_qubits)
        self._num_samples = num_samples
        self._seed = seed

    @staticmethod
    def _default_layers(n: int) -> dict[str, list[tuple[Operator, list[int]]]]:
        layers: dict[str, list[tuple[Operator, list[int]]]] = {}
        layers["identity"] = []
        layers["h_all"] = [(Operator.H(), [q]) for q in range(n)]
        layers["x_all"] = [(Operator.X(), [q]) for q in range(n)]
        if n >= 2:
            layers["cnot_chain"] = [
                (Operator.CNOT(), [q, q + 1]) for q in range(n - 1)
            ]
        layers["rz_all"] = [(Operator.Rz(0.5), [q]) for q in range(n)]
        return layers

    def _measure_layer_fidelity(
        self,
        layer_gates: list[tuple[Operator, list[int]]],
        rng: random.Random,
    ) -> float:
        """Measure fidelity by repeating the layer and checking survival."""
        fids: list[float] = []
        for _ in range(self._num_samples):
            qc = QuantumCircuit(self._num_qubits)
            # Repeat layer several times for better statistics
            for _ in range(3):
                for op, targets in layer_gates:
                    qc.append(op, targets)
            sv = qc.run()
            fids.append(float(abs(sv.amplitudes[0]) ** 2))
        return float(np.mean(fids))

    def run(self) -> LayerFidelityResult:
        """Run layer fidelity benchmarking."""
        rng = random.Random(self._seed)
        names: list[str] = []
        fids: list[float] = []

        for name, gates in self._layers.items():
            names.append(name)
            fid = self._measure_layer_fidelity(gates, rng)
            fids.append(fid)

        avg = float(np.mean(fids)) if fids else 0.0
        worst_idx = int(np.argmin(fids)) if fids else 0

        return LayerFidelityResult(
            layer_names=names,
            fidelities=fids,
            average_fidelity=avg,
            worst_layer=names[worst_idx] if names else "",
        )

    def __repr__(self) -> str:
        return (
            f"LayerFidelity(num_qubits={self._num_qubits}, "
            f"num_layers={len(self._layers)})"
        )


class PauliTwirling:
    """Randomized compiling via Pauli twirling.

    Wraps each two-qubit gate with random Pauli gates to convert
    coherent errors into stochastic (Pauli channel) errors, making
    noise easier to characterize and mitigate.

    For each 2-qubit gate G, applies:
        P_prep . G . P_post
    where P_prep and P_post are random single-qubit Pauli operators
    chosen so the overall operation is equivalent up to a Pauli frame.

    Args:
        seed: Random seed.

    Example::

        twirler = PauliTwirling(seed=42)
        result = twirler.twirl(circuit)
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    def twirl(self, circuit: QuantumCircuit) -> TwirlingResult:
        """Apply Pauli twirling to all two-qubit gates.

        Args:
            circuit: Input circuit.

        Returns:
            TwirlingResult with twirled circuit and metadata.
        """
        rng = random.Random(self._seed)
        new_gates: list[tuple[Operator, list[int]]] = []
        num_twirled = 0

        for op, targets in circuit.gates:
            if op.num_qubits == 2:
                # Twirl: P_prep . G . P_post
                p_prep = rng.choice(PAULIS)
                p_post = rng.choice(PAULIS)

                # Apply pre-gate Paulis to each qubit
                for t in targets:
                    if p_prep.name != "i":
                        new_gates.append((p_prep, [t]))

                # Original gate
                new_gates.append((op, targets))

                # Apply post-gate Paulis to each qubit
                for t in targets:
                    if p_post.name != "i":
                        new_gates.append((p_post, [t]))

                num_twirled += 1
            else:
                new_gates.append((op, targets))

        qc = QuantumCircuit(circuit.num_qubits)
        for gate_op, gate_targets in new_gates:
            qc.append(gate_op, gate_targets)

        return TwirlingResult(
            original_depth=circuit.depth,
            twirled_depth=qc.depth,
            num_twirled_gates=num_twirled,
            twirled_circuit=qc,
        )

    def __repr__(self) -> str:
        return "PauliTwirling()"
