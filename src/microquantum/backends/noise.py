"""Noise models for realistic quantum simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.density_matrix import DensityMatrix


@dataclass
class NoiseChannel:
    """A single noise channel described by Kraus operators.

    Attributes:
        name: Channel name (e.g., "depolarizing").
        qubits: Qubit indices this channel acts on.
        probability: Noise probability parameter.
        kraus_ops: List of Kraus operator matrices.
    """

    name: str
    qubits: list[int]
    probability: float
    kraus_ops: list[NDArray[np.complex128]] = field(default_factory=list)


class NoiseModel:
    """A noise model composed of quantum channels.

    Channels are applied in order after each gate execution.
    Supports common noise channels: depolarizing, amplitude damping,
    phase damping, and bit flip.

    Attributes:
        channels: List of noise channels to apply.
    """

    def __init__(self) -> None:
        self.channels: list[NoiseChannel] = []

    def depolarizing(self, probability: float, qubits: Optional[list[int]] = None) -> NoiseModel:
        """Add a depolarizing channel.

        With probability p, replaces the state with the maximally mixed
        state I/2^n. Kraus operators:
            E_0 = sqrt(1-p) * I
            E_1 = sqrt(p/3) * X
            E_2 = sqrt(p/3) * Y
            E_3 = sqrt(p/3) * Z

        Args:
            probability: Noise probability in [0, 1].
            qubits: Qubit indices to apply to. None = all qubits.

        Returns:
            self, for chaining.
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"probability must be in [0,1], got {probability}")

        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        p = probability
        kraus = [
            np.sqrt(1 - p) * I,
            np.sqrt(p / 3) * X,
            np.sqrt(p / 3) * Y,
            np.sqrt(p / 3) * Z,
        ]
        self.channels.append(NoiseChannel(
            name="depolarizing",
            qubits=qubits or [],
            probability=probability,
            kraus_ops=kraus,
        ))
        return self

    def amplitude_damping(self, probability: float, qubits: Optional[list[int]] = None) -> NoiseModel:
        """Add an amplitude damping channel.

        Models energy relaxation (T1 decay). Kraus operators:
            E_0 = [[1, 0], [0, sqrt(1-gamma)]]
            E_1 = [[0, sqrt(gamma)], [0, 0]]

        Args:
            probability: Decay probability gamma in [0, 1].
            qubits: Qubit indices. None = all.

        Returns:
            self, for chaining.
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"probability must be in [0,1], got {probability}")

        g = probability
        e0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=np.complex128)
        e1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=np.complex128)

        self.channels.append(NoiseChannel(
            name="amplitude_damping",
            qubits=qubits or [],
            probability=probability,
            kraus_ops=[e0, e1],
        ))
        return self

    def phase_damping(self, probability: float, qubits: Optional[list[int]] = None) -> NoiseModel:
        """Add a phase damping channel.

        Models dephasing (T2 decay). Kraus operators:
            E_0 = [[1, 0], [0, sqrt(1-lambda)]]
            E_1 = [[0, 0], [0, sqrt(lambda)]]

        Args:
            probability: Dephasing probability lambda in [0, 1].
            qubits: Qubit indices. None = all.

        Returns:
            self, for chaining.
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"probability must be in [0,1], got {probability}")

        lam = probability
        e0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=np.complex128)
        e1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=np.complex128)

        self.channels.append(NoiseChannel(
            name="phase_damping",
            qubits=qubits or [],
            probability=probability,
            kraus_ops=[e0, e1],
        ))
        return self

    def bit_flip(self, probability: float, qubits: Optional[list[int]] = None) -> NoiseModel:
        """Add a bit flip channel.

        With probability p, flips the qubit (applies X).
        Kraus operators:
            E_0 = sqrt(1-p) * I
            E_1 = sqrt(p) * X

        Args:
            probability: Flip probability in [0, 1].
            qubits: Qubit indices. None = all.

        Returns:
            self, for chaining.
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"probability must be in [0,1], got {probability}")

        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

        p = probability
        self.channels.append(NoiseChannel(
            name="bit_flip",
            qubits=qubits or [],
            probability=probability,
            kraus_ops=[np.sqrt(1 - p) * I, np.sqrt(p) * X],
        ))
        return self

    def apply(self, rho: DensityMatrix) -> DensityMatrix:
        """Apply all noise channels to a density matrix.

        Each channel acts on its specified qubits by expanding
        the single-qubit Kraus operators to the full Hilbert space.

        Args:
            rho: The current density matrix.

        Returns:
            New DensityMatrix after all channels are applied.
        """
        result = rho
        for channel in self.channels:
            result = self._apply_channel(result, channel)
        return result

    @staticmethod
    def _apply_channel(rho: DensityMatrix, channel: NoiseChannel) -> DensityMatrix:
        """Apply a single noise channel to the density matrix.

        For channels targeting specific qubits, expands the Kraus operators
        to the full Hilbert space using tensor products.
        """
        n = rho.num_qubits

        if not channel.qubits:
            # Apply to all qubits independently
            result = rho
            for q in range(n):
                result = NoiseModel._apply_single_qubit_channel(result, channel.kraus_ops, q)
            return result

        # Apply to specific qubits
        result = rho
        for q in channel.qubits:
            result = NoiseModel._apply_single_qubit_channel(result, channel.kraus_ops, q)
        return result

    @staticmethod
    def _apply_single_qubit_channel(
        rho: DensityMatrix,
        kraus_ops: list[NDArray[np.complex128]],
        target_qubit: int,
    ) -> DensityMatrix:
        """Apply a single-qubit Kraus channel to a specific qubit."""
        n = rho.num_qubits
        dim = rho.dim

        # Build full-space Kraus operators
        full_kraus = []
        for e_k in kraus_ops:
            # Expand: I_q0 ⊗ ... ⊗ E_k ⊗ ... ⊗ I_qN
            ops = []
            for q in range(n):
                if q == target_qubit:
                    ops.append(np.asarray(e_k, dtype=np.complex128))
                else:
                    ops.append(np.eye(2, dtype=np.complex128))
            full: NDArray[np.complex128] = ops[0]
            for op in ops[1:]:
                full = np.asarray(np.kron(full, op), dtype=np.complex128)
            full_kraus.append(np.asarray(full, dtype=np.complex128))

        return rho.apply_kraus(full_kraus)

    def __repr__(self) -> str:
        return f"NoiseModel(channels={len(self.channels)})"

    def __str__(self) -> str:
        lines = [f"NoiseModel ({len(self.channels)} channels)"]
        for ch in self.channels:
            qubits_str = ",".join(str(q) for q in ch.qubits) if ch.qubits else "all"
            lines.append(f"  {ch.name}(p={ch.probability}, qubits=[{qubits_str}])")
        return "\n".join(lines)

    def phase_flip(self, probability: float, qubits: Optional[list[int]] = None) -> NoiseModel:
        """Add a phase flip channel.

        With probability p, applies Z (phase flip).
        Kraus operators:
            E_0 = sqrt(1-p) * I
            E_1 = sqrt(p) * Z

        Args:
            probability: Flip probability in [0, 1].
            qubits: Qubit indices. None = all.

        Returns:
            self, for chaining.
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"probability must be in [0,1], got {probability}")

        I = np.eye(2, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        p = probability
        self.channels.append(NoiseChannel(
            name="phase_flip",
            qubits=qubits or [],
            probability=probability,
            kraus_ops=[np.sqrt(1 - p) * I, np.sqrt(p) * Z],
        ))
        return self

    def thermal_relaxation(
        self,
        t1: float,
        t2: float,
        gate_time: float,
        qubits: Optional[list[int]] = None,
    ) -> NoiseModel:
        """Add a thermal relaxation channel (T1/T2 decay).

        Models energy relaxation (T1) and dephasing (T2) during a gate.
        Requires T2 <= 2*T1 and gate_time <= T1.

        Args:
            t1: Relaxation time (amplitude damping time constant).
            t2: Dephasing time (T2 <= 2*T1).
            gate_time: Duration of the gate operation.
            qubits: Qubit indices. None = all.

        Returns:
            self, for chaining.

        Raises:
            ValueError: If parameters violate physical constraints.
        """
        if t1 <= 0 or t2 <= 0 or gate_time <= 0:
            raise ValueError("t1, t2, gate_time must be > 0")
        if t2 > 2 * t1:
            raise ValueError(f"T2 ({t2}) must be <= 2*T1 ({2*t1})")
        if gate_time > t1:
            raise ValueError(f"gate_time ({gate_time}) must be <= T1 ({t1})")

        # Amplitude damping parameter
        gamma = 1 - np.exp(-gate_time / t1)
        # Dephasing parameter
        lam = 0.5 * (1 - np.exp(-gate_time / t2))

        # Combined channel: amplitude damping + phase damping
        # Kraus operators for combined channel
        sqrt_1g = np.sqrt(1 - gamma)
        sqrt_g = np.sqrt(gamma)
        sqrt_1l = np.sqrt(1 - lam)
        sqrt_l = np.sqrt(lam)

        e0 = np.array([[sqrt_1l, 0], [0, sqrt_1g * sqrt_1l]], dtype=np.complex128)
        e1 = np.array([[0, sqrt_g * sqrt_1l], [0, 0]], dtype=np.complex128)
        e2 = np.array([[0, 0], [0, sqrt_1g * sqrt_l]], dtype=np.complex128)
        e3 = np.array([[0, sqrt_g * sqrt_l], [0, 0]], dtype=np.complex128)

        self.channels.append(NoiseChannel(
            name="thermal_relaxation",
            qubits=qubits or [],
            probability=gamma,
            kraus_ops=[e0, e1, e2, e3],
        ))
        return self

    def readout_error(
        self,
        prob_measured_0_when_actual_1: float,
        prob_measured_1_when_actual_0: float,
        qubits: Optional[list[int]] = None,
    ) -> NoiseModel:
        """Add a readout (measurement) error channel.

        Models classical measurement confusion where qubit state is
        read incorrectly with some probability.

        Args:
            prob_measured_0_when_actual_1: P(read 0 | state is 1).
            prob_measured_1_when_actual_0: P(read 1 | state is 0).
            qubits: Qubit indices. None = all.

        Returns:
            self, for chaining.

        Raises:
            ValueError: If probabilities are outside [0, 1].
        """
        p01 = prob_measured_0_when_actual_1
        p10 = prob_measured_1_when_actual_0
        if not (0 <= p01 <= 1 and 0 <= p10 <= 1):
            raise ValueError("Readout error probabilities must be in [0, 1]")

        # CPTP readout error channel:
        # E0 = diag(sqrt(1-p10), sqrt(1-p01))
        # E1 = [[0, sqrt(p01)], [0, 0]]   (read 0 when actually 1)
        # E2 = [[0, 0], [sqrt(p10), 0]]   (read 1 when actually 0)
        # Verify: E0†E0 + E1†E1 + E2†E2 = I
        e0 = np.array(
            [[np.sqrt(1 - p10), 0], [0, np.sqrt(1 - p01)]],
            dtype=np.complex128,
        )
        e1 = np.array(
            [[0, np.sqrt(p01)], [0, 0]],
            dtype=np.complex128,
        )
        e2 = np.array(
            [[0, 0], [np.sqrt(p10), 0]],
            dtype=np.complex128,
        )

        self.channels.append(NoiseChannel(
            name="readout_error",
            qubits=qubits or [],
            probability=p01 + p10,
            kraus_ops=[e0, e1, e2],
        ))
        return self

    def pauli_channel(
        self,
        px: float = 0.0,
        py: float = 0.0,
        pz: float = 0.0,
        qubits: Optional[list[int]] = None,
    ) -> NoiseModel:
        """Add a general Pauli error channel.

        With probability px, py, pz applies X, Y, Z respectively.
        Kraus operators:
            E_0 = sqrt(1-px-py-pz) * I
            E_1 = sqrt(px) * X
            E_2 = sqrt(py) * Y
            E_3 = sqrt(pz) * Z

        Args:
            px: Probability of X error.
            py: Probability of Y error.
            pz: Probability of Z error.
            qubits: Qubit indices. None = all.

        Returns:
            self, for chaining.

        Raises:
            ValueError: If total probability exceeds 1.
        """
        if px < 0 or py < 0 or pz < 0:
            raise ValueError("Pauli probabilities must be >= 0")
        if px + py + pz > 1:
            raise ValueError(f"Total Pauli probability ({px+py+pz}) must be <= 1")

        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        p_total = px + py + pz
        self.channels.append(NoiseChannel(
            name="pauli_channel",
            qubits=qubits or [],
            probability=p_total,
            kraus_ops=[
                np.sqrt(1 - p_total) * I,
                np.sqrt(px) * X if px > 0 else np.zeros((2, 2), dtype=np.complex128),
                np.sqrt(py) * Y if py > 0 else np.zeros((2, 2), dtype=np.complex128),
                np.sqrt(pz) * Z if pz > 0 else np.zeros((2, 2), dtype=np.complex128),
            ],
        ))
        return self
