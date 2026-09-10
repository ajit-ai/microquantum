"""Zero-noise extrapolation for quantum error mitigation.

Implements ZNE by running circuits at multiple noise levels and
extrapolating the expectation value to the zero-noise limit.

Noise amplification is achieved by folding gate sequences: each gate
G is replaced by G G^dag G (pulse stretching), which amplifies the
noise by a factor of 1, 3, 5, ... while preserving the ideal result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit


@dataclass
class ExtrapolationResult(JSONSerializable):
    """Result from zero-noise extrapolation.

    Attributes:
        mitigated_value: The extrapolated zero-noise expectation value.
        raw_value: The unmitigated (noiseless) expectation value.
        noise_factors: List of noise scaling factors used.
        noisy_values: Expectation values at each noise level.
        method: Extrapolation method used.
        fit_quality: Quality of the extrapolation fit (0 to 1).
    """
    mitigated_value: float
    raw_value: float = 0.0
    noise_factors: list[float] = field(default_factory=list)
    noisy_values: list[float] = field(default_factory=list)
    method: str = "linear"
    fit_quality: float = 0.0


class ZeroNoiseExtrapolation:
    """Zero-noise extrapolation (ZNE) error mitigation.

    Runs the circuit at multiple noise levels and extrapolates
    to the zero-noise limit. Noise amplification is done by
    folding: G -> G G^dag G (factor 3), G G^dag G G^dag G (factor 5), etc.

    Args:
        noise_factors: List of noise scaling factors (must be odd positive
            integers: 1, 3, 5, ...). Default: [1, 3, 5].
        method: Extrapolation method: 'linear', 'polynomial', 'exponential',
            'richardson'.
        shots: Number of measurement shots per noise level.
    """

    def __init__(
        self,
        noise_factors: Optional[list[float]] = None,
        method: str = "linear",
        shots: int = 1024,
    ) -> None:
        if noise_factors is None:
            noise_factors = [1.0, 3.0, 5.0]
        if len(noise_factors) < 2:
            raise ValueError("Need at least 2 noise factors")
        if not all(f >= 1.0 for f in noise_factors):
            raise ValueError("Noise factors must be >= 1")
        valid_methods = {"linear", "polynomial", "exponential", "richardson"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        self._noise_factors = sorted(set(noise_factors))
        self._method = method
        self._shots = shots

    @property
    def noise_factors(self) -> list[float]:
        return list(self._noise_factors)

    @property
    def method(self) -> str:
        return self._method

    def fold_circuit(self, circuit: QuantumCircuit, factor: float) -> QuantumCircuit:
        """Fold a circuit to amplify noise by the given factor.

        Factor 1 = original, factor 3 = G G^dag G, factor 5 = G G^dag G G^dag G, etc.

        Args:
            circuit: Original circuit to fold.
            factor: Noise amplification factor (must be odd positive integer).

        Returns:
            Folded circuit with amplified noise.
        """
        if factor < 1:
            raise ValueError(f"Factor must be >= 1, got {factor}")

        if factor == 1:
            return circuit

        fold_level = int(factor)
        if fold_level % 2 == 0:
            fold_level += 1  # Round up to nearest odd

        n = circuit.num_qubits
        folded = QuantumCircuit(n)

        len(circuit._gate_instructions)
        for _fold in range((fold_level - 1) // 2):
            # Forward pass: add original gates
            for gate_instr in circuit._gate_instructions:
                if QuantumCircuit._is_parameterized_gate(gate_instr):
                    continue
                from ..core.tensor import expand_operator
                op = gate_instr[0]  # type: ignore[assignment]
                targets = gate_instr[1]  # type: ignore[assignment]
                expanded = expand_operator(op, targets, n)  # type: ignore[arg-type]
                folded.append(expanded, list(range(n)))

            # Reverse pass: add inverse gates
            for gate_instr in reversed(circuit._gate_instructions):
                if QuantumCircuit._is_parameterized_gate(gate_instr):
                    continue
                from ..core.tensor import expand_operator
                op = gate_instr[0]  # type: ignore[assignment]
                targets = gate_instr[1]  # type: ignore[assignment]
                inv_op = op.inverse()  # type: ignore[union-attr]
                expanded = expand_operator(inv_op, targets, n)  # type: ignore[arg-type]
                folded.append(expanded, list(range(n)))

        # Final forward pass
        for gate_instr in circuit._gate_instructions:
            if QuantumCircuit._is_parameterized_gate(gate_instr):
                continue
            from ..core.tensor import expand_operator
            op = gate_instr[0]  # type: ignore[assignment]
            targets = gate_instr[1]  # type: ignore[assignment]
            expanded = expand_operator(op, targets, n)  # type: ignore[arg-type]
            folded.append(expanded, list(range(n)))

        return folded

    def extrapolate(
        self,
        noisy_values: list[float],
    ) -> ExtrapolationResult:
        """Extrapolate noisy values to the zero-noise limit.

        Args:
            noisy_values: Expectation values at each noise level.
                Must have same length as noise_factors.

        Returns:
            ExtrapolationResult with the mitigated value.
        """
        if len(noisy_values) != len(self._noise_factors):
            raise ValueError(
                f"Expected {len(self._noise_factors)} values, "
                f"got {len(noisy_values)}"
            )

        x = np.array(self._noise_factors)
        y = np.array(noisy_values)

        if self._method == "linear":
            coeffs = np.polyfit(x, y, 1)
            mitigated = float(np.polyval(coeffs, 0.0))
            fit_quality = self._linear_fit_quality(x, y, coeffs)

        elif self._method == "polynomial":
            degree = min(len(x) - 1, 3)
            coeffs = np.polyfit(x, y, degree)
            mitigated = float(np.polyval(coeffs, 0.0))
            fit_quality = self._poly_fit_quality(x, y, coeffs)

        elif self._method == "exponential":
            # Fit y = a * exp(b * x) + c
            # Use log transform for linear fit
            mask = y > 0
            if np.sum(mask) < 2:
                mitigated = float(y[0])
                fit_quality = 0.0
            else:
                log_y = np.log(y[mask])
                x_masked = x[mask]
                coeffs = np.polyfit(x_masked, log_y, 1)
                mitigated = float(np.exp(coeffs[1]))
                fit_quality = 0.8  # Approximate

        elif self._method == "richardson":
            # Richardson extrapolation: combine pairs of estimates
            if len(x) >= 2:
                # Use the two smallest noise factors
                x0, x1 = x[0], x[1]
                y0, y1 = y[0], y[1]
                # Extrapolate: y_0 = y(x0) + (y(x1) - y(x0)) * x0 / (x0 - x1)
                mitigated = float(y0 + (y1 - y0) * x0 / (x0 - x1))
                fit_quality = 0.9
            else:
                mitigated = float(y[0])
                fit_quality = 0.0
        else:
            mitigated = float(y[0])
            fit_quality = 0.0

        return ExtrapolationResult(
            mitigated_value=mitigated,
            raw_value=float(y[0]),
            noise_factors=list(self._noise_factors),
            noisy_values=list(noisy_values),
            method=self._method,
            fit_quality=fit_quality,
        )

    def _linear_fit_quality(
        self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray
    ) -> float:
        """Compute R^2 quality of linear fit."""
        y_fit = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-12:
            return 1.0
        return float(max(0.0, 1.0 - ss_res / ss_tot))

    def _poly_fit_quality(
        self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray
    ) -> float:
        """Compute R^2 quality of polynomial fit."""
        y_fit = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-12:
            return 1.0
        return float(max(0.0, 1.0 - ss_res / ss_tot))

    def __repr__(self) -> str:
        return (
            f"ZeroNoiseExtrapolation(factors={self._noise_factors}, "
            f"method={self._method!r})"
        )
