"""Measurement error mitigation for quantum computing.

Corrects readout errors by calibrating the measurement process:
1. Prepare all computational basis states
2. Measure each to build a confusion matrix
3. Invert the matrix to correct subsequent measurements
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.density_matrix import DensityMatrix


@dataclass
class MitigationMatrix:
    """Calibrated measurement mitigation matrix.

    Attributes:
        confusion: The measured confusion matrix C[i][j] = P(measure j | state i).
        inverse: The inverted mitigation matrix for correction.
        num_qubits: Number of qubits calibrated.
        fidelity: Average readout fidelity of the original confusion matrix.
    """
    confusion: np.ndarray
    inverse: np.ndarray
    num_qubits: int
    fidelity: float


class MeasurementErrorMitigation:
    """Measurement error mitigation via confusion matrix inversion.

    Calibrates readout errors by preparing basis states and measuring
    to build a confusion matrix, then inverts it to correct future
    measurements.

    Args:
        num_qubits: Number of qubits to calibrate.
        shots: Number of shots per calibration basis state.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_qubits: int,
        shots: int = 4096,
        seed: Optional[int] = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"Need >= 1 qubit, got {num_qubits}")
        self._num_qubits = num_qubits
        self._shots = shots
        self._rng = np.random.RandomState(seed)
        self._mitigation: Optional[MitigationMatrix] = None

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def is_calibrated(self) -> bool:
        return self._mitigation is not None

    def calibrate(
        self,
        noisy_circuit_fn: Optional[Callable[[QuantumCircuit], QuantumCircuit]] = None,
    ) -> MitigationMatrix:
        """Calibrate measurement errors.

        Prepares each computational basis state, applies optional
        noise, and measures to build the confusion matrix.

        Args:
            noisy_circuit_fn: Optional function that takes a circuit
                and adds noise before measurement. If None, measures directly.

        Returns:
            MitigationMatrix with the calibration results.
        """
        dim = 2**self._num_qubits
        confusion = np.zeros((dim, dim))

        for state_idx in range(dim):
            # Prepare basis state
            qc = QuantumCircuit(self._num_qubits)
            for bit in range(self._num_qubits):
                if (state_idx >> bit) & 1:
                    qc.x(bit)

            # Apply noise if provided
            if noisy_circuit_fn is not None:
                qc = noisy_circuit_fn(qc)

            # Measure
            rho = DensityMatrix.from_statevector(qc.run())
            probs = np.real(np.diag(rho.matrix))

            # Record confusion matrix row
            confusion[state_idx, :] = probs

        # Compute fidelity (average diagonal)
        fidelity = float(np.trace(confusion) / dim)

        # Invert for mitigation
        try:
            inv = np.linalg.inv(confusion)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(confusion)

        self._mitigation = MitigationMatrix(
            confusion=confusion,
            inverse=inv,
            num_qubits=self._num_qubits,
            fidelity=fidelity,
        )

        return self._mitigation

    def mitigate_counts(
        self, raw_counts: dict[str, int]
    ) -> dict[str, float]:
        """Mitigate a measurement outcome distribution.

        Args:
            raw_counts: Raw measurement counts as {bitstring: count}.

        Returns:
            Mitigated probabilities as {bitstring: probability}.
        """
        if self._mitigation is None:
            raise RuntimeError("Must calibrate first. Call calibrate().")

        dim = 2**self._num_qubits
        total = sum(raw_counts.values())
        if total == 0:
            return {}

        # Convert counts to probability vector
        raw_probs = np.zeros(dim)
        for bitstring, count in raw_counts.items():
            idx = int(bitstring, 2)
            if idx < dim:
                raw_probs[idx] = count / total

        # Apply mitigation matrix
        mitigated_probs = self._mitigation.inverse @ raw_probs

        # Ensure non-negative and normalize
        mitigated_probs = np.maximum(mitigated_probs, 0.0)
        total_p = np.sum(mitigated_probs)
        if total_p > 1e-12:
            mitigated_probs /= total_p

        # Convert back to dict
        result: dict[str, float] = {}
        for i in range(dim):
            if mitigated_probs[i] > 1e-10:
                bitstring = format(i, f"0{self._num_qubits}b")
                result[bitstring] = float(mitigated_probs[i])

        return result

    def mitigate_expectation(
        self,
        raw_counts: dict[str, int],
        observable: np.ndarray,
    ) -> float:
        """Mitigate an expectation value from raw counts.

        Args:
            raw_counts: Raw measurement counts.
            observable: Observable matrix (2^n x 2^n).

        Returns:
            Mitigated expectation value.
        """
        mitigated = self.mitigate_counts(raw_counts)
        dim = 2**self._num_qubits

        expectation = 0.0
        for bitstring, prob in mitigated.items():
            idx = int(bitstring, 2)
            expectation += prob * observable[idx, idx].real

        return expectation

    def __repr__(self) -> str:
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return (
            f"MeasurementErrorMitigation(qubits={self._num_qubits}, "
            f"{status})"
        )
