"""Unified mitigation protocol: one interface over ZNE, PEC and MEM.

The existing mitigators keep their behavior byte-identical; this module
adds :class:`MitigationData` (uniform input), :class:`MitigationOutcome`
(uniform output) and thin :class:`MitigationProtocol` wrappers that
delegate to the tested implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..backends.noise import NoiseModel
from .mem import MeasurementErrorMitigation, MitigationMatrix
from .pec import ProbabilisticErrorCancellation
from .zne import ZeroNoiseExtrapolation

__all__ = [
    "MitigationData",
    "MitigationOutcome",
    "MitigationProtocol",
    "ZNEProtocol",
    "MEMProtocol",
    "PECProtocol",
]


@dataclass
class MitigationData:
    """Uniform input for any mitigation protocol."""

    expectation: Optional[float] = None
    counts: Optional[dict[str, int]] = None
    noisy_values: list[float] = field(default_factory=list)
    noise_factors: list[float] = field(default_factory=list)
    channel_matrix: Optional[NDArray[np.complex128]] = None
    confusion: Optional[NDArray[np.float64]] = None
    observable: Optional[NDArray[np.complex128]] = None
    num_qubits: Optional[int] = None


@dataclass(frozen=True)
class MitigationOutcome:
    """Uniform output of any mitigation protocol."""

    mitigated_value: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MitigationProtocol(ABC):
    """Abstract error-mitigation protocol."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol identifier."""

    @abstractmethod
    def mitigate(self, data: MitigationData) -> MitigationOutcome:
        """Mitigate the error described by *data*."""


class ZNEProtocol(MitigationProtocol):
    """Zero-noise extrapolation behind the protocol interface."""

    def __init__(self, method: str = "linear", shots: int = 1024) -> None:
        self._zne = ZeroNoiseExtrapolation(method=method, shots=shots)

    @property
    def name(self) -> str:
        return "zne"

    def mitigate(self, data: MitigationData) -> MitigationOutcome:
        """Extrapolate ``data.noisy_values`` to zero noise."""
        if not data.noisy_values:
            raise ValueError("ZNEProtocol requires data.noisy_values")
        factors = data.noise_factors or [1.0, 3.0, 5.0]
        zne = ZeroNoiseExtrapolation(noise_factors=list(factors), method=self._zne.method)
        result = zne.extrapolate(list(data.noisy_values))
        return MitigationOutcome(
            mitigated_value=result.mitigated_value,
            method=f"zne-{result.method}",
            metadata={"fit_quality": result.fit_quality, "raw_value": result.raw_value},
        )


class MEMProtocol(MitigationProtocol):
    """Measurement-error mitigation behind the protocol interface."""

    def __init__(self, shots: int = 4096, seed: Optional[int] = None) -> None:
        self._shots = shots
        self._seed = seed

    @property
    def name(self) -> str:
        return "mem"

    def mitigate(self, data: MitigationData) -> MitigationOutcome:
        """Correct ``data.counts`` with ``data.confusion`` for ``data.observable``."""
        if data.counts is None or data.observable is None or data.confusion is None:
            raise ValueError("MEMProtocol requires counts, observable and confusion")
        confusion = np.asarray(data.confusion, dtype=float)
        inverse = np.linalg.inv(confusion)
        nq = data.num_qubits if data.num_qubits is not None else int(np.log2(confusion.shape[0]))
        matrix = MitigationMatrix(
            confusion=confusion,
            inverse=inverse,
            num_qubits=nq,
            fidelity=float(np.mean(np.diag(confusion))),
        )
        mitigator = MeasurementErrorMitigation(num_qubits=nq, shots=self._shots, seed=self._seed)
        mitigator.set_calibration(matrix)
        value = mitigator.mitigate_expectation(
            dict(data.counts), np.asarray(data.observable, dtype=np.complex128)
        )
        return MitigationOutcome(
            mitigated_value=float(value),
            method="mem",
            metadata={"matrix_fidelity": matrix.fidelity},
        )


class PECProtocol(MitigationProtocol):
    """Probabilistic error cancellation behind the protocol interface."""

    def __init__(self, precision: float = 0.01, seed: Optional[int] = None) -> None:
        self._precision = precision
        self._seed = seed

    @property
    def name(self) -> str:
        return "pec"

    def mitigate(self, data: MitigationData) -> MitigationOutcome:
        """Cancel ``data.channel_matrix`` noise over ``data.noisy_values``."""
        if data.channel_matrix is None or not data.noisy_values:
            raise ValueError("PECProtocol requires channel_matrix and noisy_values")
        pec = ProbabilisticErrorCancellation(
            noise_model=NoiseModel(),
            precision=self._precision,
            seed=self._seed,
        )
        quasiprobs, overhead = pec.compute_quasi_probabilities(
            np.asarray(data.channel_matrix, dtype=np.complex128)
        )
        value = pec.mitigate_expectation(list(data.noisy_values))
        return MitigationOutcome(
            mitigated_value=float(value),
            method="pec",
            metadata={"sampling_overhead": float(overhead), "num_quasiprobs": len(quasiprobs)},
        )
