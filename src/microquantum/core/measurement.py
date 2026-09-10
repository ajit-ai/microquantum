"""Measurement engine, sampling, and observable expectation values."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .._json import json_string
from .operators import Operator
from .state import StateVector
from .tensor import expand_operator


class MeasurementResult:
    """Container for shot-based measurement outputs.

    Attributes:
        counts: Dictionary mapping bitstrings to shot counts.
        shots: Total number of measurement shots.
        qubits: List of qubit indices that were measured.
    """

    def __init__(
        self,
        counts: dict[str, int],
        shots: int,
        qubits: list[int],
    ) -> None:
        self._counts = dict(counts)
        self._shots = shots
        self._qubits = list(qubits)

    @property
    def counts(self) -> dict[str, int]:
        """Raw bitstring count dictionary."""
        return dict(self._counts)

    @property
    def shots(self) -> int:
        """Total number of measurement shots."""
        return self._shots

    @property
    def qubits(self) -> list[int]:
        """List of measured qubit indices."""
        return list(self._qubits)

    def get_counts(self) -> dict[str, int]:
        """Return raw bitstring count dictionary (Big-Endian ordering)."""
        return dict(self._counts)

    def get_probabilities(self) -> dict[str, float]:
        """Return normalized probability distribution from counts."""
        if self._shots == 0:
            return {}
        return {k: v / self._shots for k, v in self._counts.items()}

    def most_frequent(self) -> str:
        """Return the bitstring with the highest count."""
        if not self._counts:
            return ""
        return max(self._counts, key=self._counts.get)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "counts": self.get_counts(),
            "shots": int(self._shots),
            "qubits": list(self._qubits),
            "probabilities": self.get_probabilities(),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"MeasurementResult(shots={self._shots}, "
            f"qubits={self._qubits}, "
            f"outcomes={len(self._counts)})"
        )

    def __str__(self) -> str:
        lines = [f"MeasurementResult ({self._shots} shots, "
                 f"qubits {self._qubits})"]
        for bitstring, count in sorted(self._counts.items()):
            lines.append(f"  '{bitstring}': {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------

def _format_bitstring(index: int, num_qubits: int) -> str:
    """Format an integer index as a big-endian bitstring."""
    return format(index, f"0{num_qubits}b")


def sample_state(
    state: StateVector,
    shots: int = 1000,
    seed: Optional[int] = None,
) -> MeasurementResult:
    """Sample all qubits of a state vector using Born's rule.

    Args:
        state: Quantum state to measure.
        shots: Number of measurement shots.
        seed: Optional RNG seed for reproducibility.

    Returns:
        MeasurementResult with bitstring counts.
    """
    rng = np.random.default_rng(seed)
    probs = np.abs(state.amplitudes) ** 2
    probs = probs / np.sum(probs)

    outcomes = rng.choice(state.dim, size=shots, p=probs)

    counts: dict[str, int] = {}
    for outcome in outcomes:
        bs = _format_bitstring(int(outcome), state.num_qubits)
        counts[bs] = counts.get(bs, 0) + 1

    return MeasurementResult(
        counts=counts,
        shots=shots,
        qubits=list(range(state.num_qubits)),
    )


def measure_qubits(
    state: StateVector,
    targets: list[int],
    shots: int = 1000,
    seed: Optional[int] = None,
) -> MeasurementResult:
    """Measure a subset of target qubits, marginalizing over the rest.

    Args:
        state: Quantum state to measure.
        targets: Qubit indices to measure.
        shots: Number of measurement shots.
        seed: Optional RNG seed for reproducibility.

    Returns:
        MeasurementResult with bitstrings for the target qubits only.

    Raises:
        ValueError: If any target qubit index is out of range.
    """
    n = state.num_qubits
    for t in targets:
        if t < 0 or t >= n:
            raise ValueError(
                f"Target qubit {t} out of range for "
                f"{n}-qubit state (valid: 0..{n - 1})"
            )

    rng = np.random.default_rng(seed)
    probs = np.abs(state.amplitudes) ** 2
    probs = probs / np.sum(probs)

    num_target = len(targets)
    target_mask = 0
    for t in targets:
        target_mask |= 1 << (n - 1 - t)

    collapsed_probs: dict[int, float] = {}
    for idx in range(2**n):
        target_val = 0
        for t in targets:
            bit = (idx >> (n - 1 - t)) & 1
            target_val = (target_val << 1) | bit
        collapsed_probs[target_val] = (
            collapsed_probs.get(target_val, 0.0) + probs[idx]
        )

    target_indices = list(collapsed_probs.keys())
    target_probabilities = np.array(
        [collapsed_probs[i] for i in target_indices]
    )
    target_probabilities = target_probabilities / target_probabilities.sum()

    outcomes = rng.choice(target_indices, size=shots, p=target_probabilities)

    counts: dict[str, int] = {}
    for outcome in outcomes:
        bs = _format_bitstring(int(outcome), num_target)
        counts[bs] = counts.get(bs, 0) + 1

    return MeasurementResult(
        counts=counts,
        shots=shots,
        qubits=sorted(targets),
    )


def measure_and_collapse(
    state: StateVector,
    targets: list[int],
    seed: Optional[int] = None,
) -> tuple[str, StateVector]:
    """Perform a single projective measurement on target qubits.

    Args:
        state: Quantum state to measure.
        targets: Qubit indices to measure.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A tuple of (measured_bitstring, post_measurement_state).

    Raises:
        ValueError: If any target qubit index is out of range.
    """
    n = state.num_qubits
    for t in targets:
        if t < 0 or t >= n:
            raise ValueError(
                f"Target qubit {t} out of range for "
                f"{n}-qubit state (valid: 0..{n - 1})"
            )

    rng = np.random.default_rng(seed)
    probs = np.abs(state.amplitudes) ** 2
    probs = probs / np.sum(probs)

    outcome_idx = rng.choice(2**n, p=probs)
    outcome_bs = _format_bitstring(int(outcome_idx), n)

    measured_bits = ""
    for t in sorted(targets):
        measured_bits += outcome_bs[t]

    projector = np.zeros((2**n, 2**n), dtype=np.complex128)
    for idx in range(2**n):
        match = True
        for t in targets:
            bit = (idx >> (n - 1 - t)) & 1
            expected_bit = int(outcome_bs[t])
            if bit != expected_bit:
                match = False
                break
        if match:
            projector[idx, idx] = 1.0

    collapsed_amps = projector @ state.amplitudes
    norm = np.linalg.norm(collapsed_amps)
    if norm > 0:
        collapsed_amps = collapsed_amps / norm

    collapsed_state = StateVector(
        num_qubits=n,
        amplitudes=collapsed_amps.astype(np.complex128),
    )

    return measured_bits, collapsed_state


# ---------------------------------------------------------------------------
# Expectation values
# ---------------------------------------------------------------------------

def expectation_value(
    state: StateVector,
    observable: Operator,
    targets: Optional[list[int]] = None,
) -> float:
    """Compute the expectation value <psi|H|psi> of an observable.

    Args:
        state: Quantum state.
        observable: Hermitian observable operator.
        targets: Optional qubit indices the observable acts on.
            If provided, the observable is automatically expanded.

    Returns:
        The real part of the expectation value.

    Raises:
        ValueError: If the imaginary part is non-negligible (> 1e-7),
            indicating the observable is not Hermitian.
    """
    if targets is not None:
        expanded = expand_operator(observable, targets, state.num_qubits)
    else:
        if observable.num_qubits != state.num_qubits:
            raise ValueError(
                f"Observable acts on {observable.num_qubits} qubit(s) "
                f"but state has {state.num_qubits}"
            )
        expanded = observable

    psi = state.amplitudes
    expectation = np.vdot(psi, expanded.matrix @ psi)

    imag_part = abs(expectation.imag)
    if imag_part > 1e-7:
        raise ValueError(
            f"Expectation value has non-negligible imaginary part "
            f"({imag_part:.2e}). Observable may not be Hermitian."
        )

    return float(expectation.real)
