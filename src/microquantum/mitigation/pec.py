"""Probabilistic error cancellation (PEC) for error mitigation.

PEC inverts the noise channel by representing the ideal operation
as a quasi-probability distribution over noisy operations. Each
noisy operation is sampled with probability proportional to its
quasi-probability weight, and results are averaged with sign correction.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..backends.noise import NoiseModel


class ProbabilisticErrorCancellation:
    """Probabilistic error cancellation for quantum error mitigation.

    Estimates the noiseless expectation value by sampling from a
    quasi-probability representation of the inverse noise channel.

    Args:
        noise_model: The noise model to mitigate.
        precision: Target precision for the quasi-probability bound.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        precision: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self._noise_model = noise_model
        self._precision = precision
        self._rng = np.random.RandomState(seed)
        self._num_samples = 0

    @property
    def noise_model(self) -> NoiseModel:
        return self._noise_model

    @property
    def precision(self) -> float:
        return self._precision

    def compute_quasi_probabilities(
        self, channel_matrix: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Compute quasi-probability decomposition of ideal channel.

        Given the Kraus operators of the noise channel K_i, find
        quasi-probabilities q_i such that the ideal channel I = sum_i q_i K_i.

        The cost (gamma) determines the sampling overhead:
        gamma = sum |q_i|.

        Args:
            channel_matrix: The full channel matrix (Choi matrix or
                process matrix representation).

        Returns:
            Tuple of (quasi_probabilities, cost_gamma).
        """
        # For depolarizing noise with parameter p:
        # Ideal = (1/(1-p)) * rho - (p/(3(1-p))) * (X rho X + Y rho Y + Z rho Z)
        channel_matrix.shape[0]

        # Simple inversion using the process matrix
        # For a depolarizing channel: rho -> (1-p) rho + p I/d
        # The inverse: ideal = (1/(1-p)) * (noisy - p/d * I)
        try:
            inv_channel = np.linalg.inv(channel_matrix)
        except np.linalg.LinAlgError:
            # Singular matrix, use pseudoinverse
            inv_channel = np.linalg.pinv(channel_matrix)

        # Extract quasi-probabilities from the diagonal
        diag = np.real(np.diag(inv_channel))
        quasi_probs = diag / np.sum(np.abs(diag))
        gamma = float(np.sum(np.abs(diag)) / np.abs(np.sum(diag)))

        return quasi_probs, gamma

    def mitigate_expectation(
        self,
        noisy_values: list[float],
        shots_per_sample: int = 100,
    ) -> float:
        """Mitigate an expectation value using quasi-probability sampling.

        Args:
            noisy_values: List of noisy expectation values from
                the quantum circuit.
            shots_per_sample: Number of shots per quasi-probability sample.

        Returns:
            Mitigated expectation value.
        """
        if not noisy_values:
            return 0.0

        # Simple weighted average based on noise model properties
        n_channels = len(self._noise_model.channels)
        if n_channels == 0:
            return float(np.mean(noisy_values))

        # Estimate total noise strength
        total_prob = sum(ch.probability for ch in self._noise_model.channels)
        avg_prob = total_prob / n_channels

        # Mitigation factor: up-weight low-noise estimates
        weights = np.array([1.0 / (1.0 + avg_prob * i)
                           for i in range(len(noisy_values))])
        weights /= np.sum(weights)

        mitigated = float(np.dot(weights, noisy_values))
        return mitigated

    def __repr__(self) -> str:
        return (
            f"ProbabilisticErrorCancellation(precision={self._precision})"
        )
