"""Data encoding circuits for quantum machine learning.

Implements original encoding strategies for mapping classical feature
vectors into quantum states. Each encoder builds a circuit that, when
applied to |0...0>, produces a quantum state encoding the input data.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..core.state import StateVector


class BaseEncoder(ABC):
    """Base class for data encoding circuits."""

    @abstractmethod
    def encode(self, features: list[float]) -> QuantumCircuit:
        """Build a circuit that encodes the feature vector.

        Args:
            features: Classical feature vector to encode.

        Returns:
            QuantumCircuit that produces the encoded state.
        """
        ...

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits used by this encoder."""
        ...

    @abstractmethod
    def __repr__(self) -> str: ...


class AngleEncoding(BaseEncoder):
    """Encode features as rotation angles on individual qubits.

    Each feature x_i is encoded as R_y(x_i) on qubit i. Requires
    num_features <= num_qubits.

    Circuit for 3 features on 3 qubits:
        |0> -- Ry(x_0) --
        |0> -- Ry(x_1) --
        |0> -- Ry(x_2) --

    Args:
        num_features: Number of input features.
        gate: Rotation gate type ('rx', 'ry', 'rz').
    """

    def __init__(self, num_features: int, gate: str = "ry") -> None:
        if num_features < 1:
            raise ValueError(f"Need >= 1 feature, got {num_features}")
        valid_gates = {"rx", "ry", "rz"}
        if gate not in valid_gates:
            raise ValueError(f"gate must be one of {valid_gates}, got {gate!r}")
        self._num_features = num_features
        self._gate = gate

    @property
    def num_qubits(self) -> int:
        return self._num_features

    def encode(self, features: list[float]) -> QuantumCircuit:
        if len(features) != self._num_features:
            raise ValueError(
                f"Expected {self._num_features} features, got {len(features)}"
            )

        qc = QuantumCircuit(self._num_features)
        gate_fn = getattr(qc, self._gate)
        for i, x in enumerate(features):
            gate_fn(x, i)
        return qc

    def __repr__(self) -> str:
        return f"AngleEncoding(features={self._num_features}, gate={self._gate!r})"


class AmplitudeEncoding(BaseEncoder):
    """Encode features as amplitudes of a quantum state.

    Normalizes the feature vector and stores it directly as the
    statevector. Requires 2^n_qubits >= num_features.

    Args:
        num_features: Number of input features (will be zero-padded
            to next power of 2 if needed).
    """

    def __init__(self, num_features: int) -> None:
        if num_features < 1:
            raise ValueError(f"Need >= 1 feature, got {num_features}")
        self._num_features = num_features
        self._n_qubits = max(1, math.ceil(math.log2(num_features)))
        self._dim = 2**self._n_qubits

    @property
    def num_qubits(self) -> int:
        return self._n_qubits

    def encode(self, features: list[float]) -> QuantumCircuit:
        if len(features) != self._num_features:
            raise ValueError(
                f"Expected {self._num_features} features, got {len(features)}"
            )

        padded = features + [0.0] * (self._dim - len(features))
        norm = math.sqrt(sum(x * x for x in padded))
        if norm < 1e-12:
            raise ValueError("Feature vector is zero; cannot normalize.")

        amp_array = np.array([complex(x / norm) for x in padded], dtype=np.complex128)
        sv = StateVector(self._n_qubits, amp_array)

        # Build a circuit that prepares this state via rotation gates.
        # For each basis state with non-zero amplitude, apply the
        # appropriate controlled rotations.
        qc = QuantumCircuit(self._n_qubits)
        self._prepare_state(sv, qc, list(range(self._n_qubits)))
        return qc

    def _prepare_state(
        self, sv: 'StateVector', qc: QuantumCircuit, qubits: list[int]
    ) -> None:
        """Recursively prepare a state vector using multiplexed rotations."""
        amps = sv.amplitudes
        n = len(qubits)

        if n == 1:
            # Single qubit: compute rotation angle from amplitudes
            r = abs(amps[0])
            if r < 1e-12:
                qc.x(qubits[0])
            else:
                theta = 2.0 * math.atan2(abs(amps[1]), abs(amps[0]))
                if abs(theta) > 1e-10:
                    qc.ry(theta, qubits[0])
            return

        # Multi-qubit: use Ry rotations on first qubit conditioned on amplitudes
        half = len(amps) // 2
        upper_norm = math.sqrt(sum(abs(a) ** 2 for a in amps[:half]))
        lower_norm = math.sqrt(sum(abs(a) ** 2 for a in amps[half:]))

        total_norm = math.sqrt(upper_norm ** 2 + lower_norm ** 2)
        if total_norm < 1e-12:
            return

        theta = 2.0 * math.atan2(lower_norm, upper_norm)
        if abs(theta) > 1e-10:
            qc.ry(theta, qubits[0])

        # Recurse on sub-registers
        if upper_norm > 1e-12:
            upper_amps = np.array([a / upper_norm for a in amps[:half]], dtype=np.complex128)
            upper_sv = StateVector(len(qubits) - 1, upper_amps)
            self._prepare_state(upper_sv, qc, qubits[1:])

        if lower_norm > 1e-12:
            lower_amps = np.array([a / lower_norm for a in amps[half:]], dtype=np.complex128)
            lower_sv = StateVector(len(qubits) - 1, lower_amps)
            # Flip first qubit, prepare lower half, flip back
            qc.x(qubits[0])
            self._prepare_state(lower_sv, qc, qubits[1:])
            qc.x(qubits[0])

    def __repr__(self) -> str:
        return f"AmplitudeEncoding(features={self._num_features}, qubits={self._n_qubits})"


class IQPEncoding(BaseEncoder):
    """Interleaved Quantum Processing encoding with entangling layers.

    Encodes features via Rz rotations interleaved with a fixed
    entangling structure (CNOT chain). Applies `reps` repetitions
    of the encoding-entangling pattern.

    Circuit structure per repetition:
        Rz(x_0) -- Rz(x_1) -- ... -- Rz(x_{n-1})
        CNOT(0,1) -- CNOT(1,2) -- ... -- CNOT(n-2,n-1)

    Args:
        num_features: Number of input features.
        reps: Number of encoding repetitions.
    """

    def __init__(self, num_features: int, reps: int = 2) -> None:
        if num_features < 1:
            raise ValueError(f"Need >= 1 feature, got {num_features}")
        if reps < 1:
            raise ValueError(f"Need >= 1 rep, got {reps}")
        self._num_features = num_features
        self._n_qubits = num_features
        self._reps = reps

    @property
    def num_qubits(self) -> int:
        return self._n_qubits

    def encode(self, features: list[float]) -> QuantumCircuit:
        if len(features) != self._num_features:
            raise ValueError(
                f"Expected {self._num_features} features, got {len(features)}"
            )

        n = self._n_qubits
        qc = QuantumCircuit(n)

        for _ in range(self._reps):
            for i, x in enumerate(features):
                qc.rz(x, i)
            for i in range(n - 1):
                qc.cx(i, i + 1)

        return qc

    def __repr__(self) -> str:
        return (
            f"IQPEncoding(features={self._num_features}, reps={self._reps})"
        )


class ZFeatureMap(BaseEncoder):
    """Z-rotation feature map with CNOT entangler.

    Encodes features as Z-axis rotations with a full CNOT entangling
    layer between each rotation layer. This creates an IQP-style
    encoding with tunable entangling power.

    Circuit structure per repetition:
        H -- Rz(x_0) -- H -- Rz(x_1) -- ...
        CNOT(all pairs)
        ...

    Args:
        num_features: Number of input features.
        reps: Number of repetitions.
    """

    def __init__(self, num_features: int, reps: int = 2) -> None:
        if num_features < 1:
            raise ValueError(f"Need >= 1 feature, got {num_features}")
        if reps < 1:
            raise ValueError(f"Need >= 1 rep, got {reps}")
        self._num_features = num_features
        self._n_qubits = num_features
        self._reps = reps

    @property
    def num_qubits(self) -> int:
        return self._n_qubits

    def encode(self, features: list[float]) -> QuantumCircuit:
        if len(features) != self._num_features:
            raise ValueError(
                f"Expected {self._num_features} features, got {len(features)}"
            )

        n = self._n_qubits
        qc = QuantumCircuit(n)

        for _ in range(self._reps):
            for i in range(n):
                qc.h(i)
            for i, x in enumerate(features):
                qc.rz(2.0 * x, i)
            for i in range(n):
                for j in range(i + 1, n):
                    qc.cx(i, j)

        return qc

    def __repr__(self) -> str:
        return (
            f"ZFeatureMap(features={self._num_features}, reps={self._reps})"
        )
