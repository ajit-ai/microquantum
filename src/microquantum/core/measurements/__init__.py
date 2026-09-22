"""Complete measurement abstraction.

Works conceptually across state-vector and density-matrix execution:
computational-basis measurement, projective measurements, general POVMs,
sampling, post-measurement states and expectation helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from microquantum.core.measurement import MeasurementResult

__all__ = [
    "MeasurementResult",
    "Measurement",
    "ProjectiveMeasurement",
    "POVM",
    "MeasurementOutcome",
    "computational_basis_measurement",
    "sample_counts",
]


@dataclass(frozen=True)
class MeasurementOutcome:
    """A single measurement outcome."""

    label: str
    probability: float
    post_state: NDArray[np.complex128] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0 + 1e-12:
            raise ValueError(f"probability out of range: {self.probability}")


class Measurement(ABC):
    """Abstract measurement with outcome probabilities."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits measured."""

    @abstractmethod
    def probabilities(self, state: Any) -> list[MeasurementOutcome]:
        """Outcome probabilities for *state*."""

    @abstractmethod
    def sample(self, state: Any, shots: int, seed: int | None = None) -> MeasurementResult:
        """Sample *shots* outcomes."""

    def expectation(self, state: Any, values: dict[str, float]) -> float:
        """Expectation of a classical value assignment over outcomes."""
        total = 0.0
        for outcome in self.probabilities(state):
            if outcome.label not in values:
                raise ValueError(f"No value assigned for outcome '{outcome.label}'")
            total += outcome.probability * values[outcome.label]
        return total


def _as_vector(state: Any) -> NDArray[np.complex128]:
    if isinstance(state, np.ndarray) and state.ndim == 1:
        return np.asarray(state, dtype=np.complex128)
    amps = getattr(state, "amplitudes", None)
    if amps is not None:
        return np.asarray(amps, dtype=np.complex128)
    raise TypeError(f"Cannot interpret {type(state).__name__} as a state vector")


def _as_density(state: Any) -> NDArray[np.complex128]:
    if isinstance(state, np.ndarray) and state.ndim == 2:
        return np.asarray(state, dtype=np.complex128)
    for attr in ("matrix", "rho", "density_matrix"):
        mat = getattr(state, attr, None)
        if isinstance(mat, np.ndarray) and mat.ndim == 2:
            return np.asarray(mat, dtype=np.complex128)
    vec = _as_vector(state)
    return np.outer(vec, vec.conj())


class ProjectiveMeasurement(Measurement):
    """Projective measurement defined by orthogonal projectors."""

    def __init__(self, projectors: list[Any], labels: Sequence[str] | None = None) -> None:
        if not projectors:
            raise ValueError("ProjectiveMeasurement requires at least one projector")
        mats = [np.asarray(p, dtype=np.complex128) for p in projectors]
        dim = mats[0].shape[0]
        for m in mats:
            if m.shape != (dim, dim):
                raise ValueError("All projectors must share the same dimension")
            if not np.allclose(m, m.conj().T, atol=1e-9):
                raise ValueError("Projectors must be Hermitian")
            if not np.allclose(m @ m, m, atol=1e-9):
                raise ValueError("Projectors must be idempotent")
        total = sum(mats)
        if not np.allclose(total, np.eye(dim), atol=1e-9):
            raise ValueError("Projectors must sum to identity (completeness)")
        if dim == 0 or (dim & (dim - 1)) != 0:
            raise ValueError(f"Dimension must be a power of 2, got {dim}")
        self._projectors = mats
        self._dim = dim
        if labels is None:
            self._labels = [format(i, f"0{len(mats) - 1}b") if len(mats) > 1 else "0" for i in range(len(mats))]
        else:
            if len(labels) != len(mats):
                raise ValueError("labels length must match projectors length")
            self._labels = list(labels)

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        import math as _math

        return int(_math.log2(self._dim))

    @property
    def labels(self) -> list[str]:
        """Outcome labels."""
        return list(self._labels)

    def probabilities(self, state: Any) -> list[MeasurementOutcome]:
        rho = _as_density(state)
        if rho.shape != (self._dim, self._dim):
            raise ValueError("State dimension does not match measurement")
        outcomes: list[MeasurementOutcome] = []
        for proj, label in zip(self._projectors, self._labels, strict=False):
            p = float(np.real(np.trace(proj @ rho)))
            p = min(max(p, 0.0), 1.0)
            outcomes.append(MeasurementOutcome(label, p))
        return outcomes

    def sample(self, state: Any, shots: int, seed: int | None = None) -> MeasurementResult:
        if shots < 1:
            raise ValueError("shots must be >= 1")
        outcomes = self.probabilities(state)
        probs = [o.probability for o in outcomes]
        total = sum(probs)
        if total <= 0:
            raise ValueError("Measurement has no nonzero outcomes")
        probs = [p / total for p in probs]
        rng = np.random.default_rng(seed)
        draws = rng.choice(len(outcomes), size=shots, p=np.asarray(probs))
        counts: dict[str, int] = {}
        for d in draws:
            label = outcomes[int(d)].label
            counts[label] = counts.get(label, 0) + 1
        return MeasurementResult(counts, shots, list(range(self.num_qubits)))

    def post_measurement_state(self, state: Any, outcome: str) -> NDArray[np.complex128]:
        """Collapsed (unnormalized-normalized) state-vector after *outcome*."""
        vec = _as_vector(state)
        idx = self._labels.index(outcome) if outcome in self._labels else (_ for _ in ()).throw(ValueError(f"Unknown outcome '{outcome}'"))
        proj = self._projectors[idx]
        collapsed = proj @ vec
        norm = float(np.linalg.norm(collapsed))
        if norm < 1e-12:
            raise ValueError(f"Outcome '{outcome}' has zero probability")
        return (collapsed / norm).astype(np.complex128)


class POVM(Measurement):
    """General positive operator-valued measure."""

    def __init__(self, effects: list[Any], labels: Sequence[str] | None = None) -> None:
        if not effects:
            raise ValueError("POVM requires at least one effect")
        mats = [np.asarray(e, dtype=np.complex128) for e in effects]
        dim = mats[0].shape[0]
        for m in mats:
            if m.shape != (dim, dim):
                raise ValueError("All POVM effects must share the same dimension")
            eigs = np.linalg.eigvalsh((m + m.conj().T) / 2)
            if np.min(eigs) < -1e-9:
                raise ValueError("POVM effects must be positive semidefinite")
        total = sum(mats)
        if not np.allclose(total, np.eye(dim), atol=1e-9):
            raise ValueError("POVM effects must sum to identity")
        self._effects = mats
        self._dim = dim
        self._labels = list(labels) if labels is not None else [str(i) for i in range(len(mats))]
        if len(self._labels) != len(mats):
            raise ValueError("labels length must match effects length")

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        import math as _math

        return int(_math.log2(self._dim))

    def probabilities(self, state: Any) -> list[MeasurementOutcome]:
        rho = _as_density(state)
        if rho.shape != (self._dim, self._dim):
            raise ValueError("State dimension does not match POVM")
        outcomes: list[MeasurementOutcome] = []
        for eff, label in zip(self._effects, self._labels, strict=False):
            p = float(np.real(np.trace(eff @ rho)))
            outcomes.append(MeasurementOutcome(label, min(max(p, 0.0), 1.0)))
        return outcomes

    def sample(self, state: Any, shots: int, seed: int | None = None) -> MeasurementResult:
        if shots < 1:
            raise ValueError("shots must be >= 1")
        outcomes = self.probabilities(state)
        probs = np.asarray([o.probability for o in outcomes], dtype=float)
        probs = probs / probs.sum()
        rng = np.random.default_rng(seed)
        draws = rng.choice(len(outcomes), size=shots, p=probs)
        counts: dict[str, int] = {}
        for d in draws:
            label = outcomes[int(d)].label
            counts[label] = counts.get(label, 0) + 1
        return MeasurementResult(counts, shots, list(range(self.num_qubits)))


def computational_basis_measurement(num_qubits: int) -> ProjectiveMeasurement:
    """Z-basis measurement on *num_qubits* qubits."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    dim = 2**num_qubits
    projectors: list[NDArray[np.complex128]] = []
    labels: list[str] = []
    for i in range(dim):
        mat = np.zeros((dim, dim), dtype=np.complex128)
        mat[i, i] = 1.0
        projectors.append(mat)
        labels.append(format(i, f"0{num_qubits}b"))
    return ProjectiveMeasurement(projectors, labels)


def sample_counts(
    amplitudes: Any, shots: int, seed: int | None = None
) -> dict[str, int]:
    """Sample computational-basis counts from amplitudes."""
    vec = np.asarray(amplitudes, dtype=np.complex128)
    if vec.ndim != 1:
        raise ValueError("amplitudes must be a 1D array")
    probs = (np.abs(vec) ** 2).astype(float)
    total = probs.sum()
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"State is not normalized (sum p = {total})")
    probs = probs / total
    n = int(np.log2(vec.shape[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(vec.shape[0], size=shots, p=probs)
    counts: dict[str, int] = {}
    for d in draws:
        key = format(int(d), f"0{n}b")
        counts[key] = counts.get(key, 0) + 1
    return counts
