"""Statevector / density-matrix analysis (MQ-07).

:class:`StateAnalysis` provides lightweight, mathematically valid inspection
of state-level backend output:

**Statevector** — normalization check, probabilities over basis states,
amplitudes and the most probable basis state (plus an expectation helper
restricted to *diagonal observables*).

**Density matrix** — trace, normalization check, diagonal (measurement)
probabilities and purity.

This is deliberately *not* a quantum-information research library: the focus
is practical result inspection with JSON-safe serialization.  The analyzer
accepts a :class:`~microquantum.backends.base.BackendResult`, an
:class:`ExecutionRecord`, a :class:`StateVector`, a raw NumPy array, or the
serialized ``to_dict()`` form of a result.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .._json import JSONSerializable, json_safe, json_string


def _decode_complex(value: Any) -> Any:
    """Recursively restore ``{"real": .., "imag": ..}`` encoded arrays."""
    if isinstance(value, dict):
        if set(value) == {"real", "imag"} and all(
            isinstance(v, (int, float)) for v in value.values()
        ):
            return complex(float(value["real"]), float(value["imag"]))
        return {k: _decode_complex(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_complex(v) for v in value]
    return value


def _array(value: Any) -> Optional[NDArray[np.complex128]]:
    if value is None:
        return None
    return np.asarray(_decode_complex(value), dtype=np.complex128)


class StateAnalysis(JSONSerializable):
    """Lightweight analysis of a statevector or density matrix.

    Args:
        source: A :class:`BackendResult`, :class:`ExecutionRecord`,
            :class:`StateVector`, a raw complex :class:`numpy.ndarray`
            (1-D = statevector, 2-D = density matrix), or a serialized dict.

    Raises:
        TypeError: If the source exposes no state-level data.
    """

    def __init__(self, source: Any) -> None:
        statevector: Optional[NDArray[np.complex128]] = None
        density: Optional[NDArray[np.complex128]] = None

        if isinstance(source, dict):
            statevector = _array(source.get("statevector"))
            density = _array(source.get("density_matrix"))
        elif isinstance(source, np.ndarray):
            if source.ndim == 1:
                statevector = np.asarray(source, dtype=np.complex128)
            elif source.ndim == 2:
                density = np.asarray(source, dtype=np.complex128)
        else:
            sv = getattr(source, "statevector", None)
            if sv is None:
                sv = getattr(source, "amplitudes", None)
            if sv is not None:
                statevector = (
                    sv.amplitudes
                    if hasattr(sv, "amplitudes")
                    else np.asarray(sv, dtype=np.complex128)
                )
            dm = getattr(source, "density_matrix", None)
            if dm is not None:
                density = np.asarray(
                    dm.amplitudes if hasattr(dm, "amplitudes") else dm,
                    dtype=np.complex128,
                )

        self._statevector = statevector
        self._density = density

        if statevector is None and density is None:
            raise TypeError(
                "StateAnalysis needs a result carrying a statevector or a "
                f"density matrix; got {type(source).__name__}"
            )
        if statevector is not None and density is not None:
            self._statevector = statevector
            self._density = None

    @property
    def kind(self) -> str:
        """``"statevector"`` or ``"density_matrix"``."""
        return "statevector" if self._statevector is not None else "density_matrix"

    @property
    def dim(self) -> int:
        """Hilbert-space dimension (2^num_qubits)."""
        if self._statevector is not None:
            return int(self._statevector.shape[0])
        return int(self._density.shape[0])  # type: ignore[union-attr]

    # -- statevector ----------------------------------------------------------

    def amplitudes(self) -> NDArray[np.complex128]:
        """Complex amplitudes (statevector only)."""
        self._require_statevector()
        return self._statevector.copy()  # type: ignore[union-attr]

    def norm_squared(self) -> float:
        """Sum of squared amplitude magnitudes (should be 1.0 when normalized)."""
        self._require_statevector()
        psi = self._statevector
        assert psi is not None
        return float(np.sum(np.abs(psi) ** 2))

    def is_normalized(self, tol: float = 1e-9) -> bool:
        """Whether the statevector norm is 1 within ``tol``."""
        return abs(self.norm_squared() - 1.0) <= tol

    def probabilities(self) -> dict[str, float]:
        """``{basis_bitstring: probability}`` over all basis states."""
        self._require_statevector()
        psi = self._statevector
        assert psi is not None
        probs = np.abs(psi) ** 2
        num_qubits = int(round(math.log2(max(2, self.dim))))
        return {
            format(index, f"0{num_qubits}b"): float(probs[index])
            for index in range(self.dim)
        }

    def most_probable_state(self) -> int:
        """Basis index with the largest probability (ties: lowest index)."""
        self._require_statevector()
        psi = self._statevector
        assert psi is not None
        probs = np.abs(psi) ** 2
        return int(np.argmax(probs))

    def most_probable_bitstring(self) -> str:
        """Most probable basis state as a bitstring."""
        self._require_statevector()
        num_qubits = int(round(math.log2(max(2, self.dim))))
        return format(self.most_probable_state(), f"0{num_qubits}b")

    def expectation(self, observable: Any) -> float:
        """Expectation of a *diagonal* observable.

        Args:
            observable: Either a 1-D array/sequence of length ``dim`` (the
                 diagonal entries of the observable matrix) or a callable
                 ``basis_index -> real``.

        Returns:
            ``sum_b |psi_b|^2 * obs[b]`` (the real part).
        """
        self._require_statevector()
        psi = self._statevector
        assert psi is not None
        if callable(observable):
            values = np.asarray(
                [float(observable(i)) for i in range(self.dim)], dtype=np.float64
            )
        else:
            try:
                values = np.asarray(
                    [float(v) for v in observable], dtype=np.float64
                )
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "observable must be a callable or an iterable of "
                    f"{self.dim} real numbers, got {type(observable).__name__}"
                ) from exc
        if values.ndim != 1 or len(values) != self.dim:
            raise ValueError(
                f"diagonal observable must have {self.dim} entries, "
                f"got {values.shape}"
            )
        probs = np.abs(psi) ** 2
        return float(np.real(np.sum(probs * values)))

    # -- density matrix -------------------------------------------------------

    def _require_statevector(self) -> None:
        if self._statevector is None:
            raise ValueError(
                "this operation requires a statevector, but the source "
                f"carries a density matrix ({type(self._density).__name__})"
            )

    def _require_density(self) -> NDArray[np.complex128]:
        if self._density is None:
            raise ValueError(
                "this operation requires a density matrix, but the source "
                "carries a statevector"
            )
        return self._density

    def trace(self) -> float:
        """Trace of the density matrix (should be 1.0 when valid)."""
        return float(np.real(np.trace(self._require_density())))

    def purity(self) -> float:
        """Purity ``tr(rho^2)`` (1.0 for a pure state, <= 1 in general)."""
        rho = self._require_density()
        return float(np.real(np.trace(rho @ rho)))

    def is_pure(self, tol: float = 1e-9) -> bool:
        """Whether the density matrix describes a (near-)pure state."""
        return abs(self.purity() - 1.0) <= tol

    def diagonal_probabilities(self) -> dict[str, float]:
        """Measurement probabilities from the density-matrix diagonal."""
        rho = self._require_density()
        diagonal = np.real(np.diag(rho))
        if any(diagonal < -1e-12):
            raise ValueError("density matrix has negative diagonal elements")
        diagonal = np.clip(diagonal, 0.0, None)
        total = float(np.sum(diagonal))
        if total <= 0:
            raise ValueError("density matrix diagonal sums to zero")
        num_qubits = int(round(math.log2(max(2, rho.shape[0]))))
        return {
            format(index, f"0{num_qubits}b"): float(diagonal[index] / total)
            for index in range(rho.shape[0])
        }

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize a JSON-safe analysis snapshot."""
        if self._statevector is not None:
            prob = self.probabilities()
            return {
                "type": "statevector",
                "dim": self.dim,
                "normalized": self.is_normalized(),
                "norm_squared": self.norm_squared(),
                "most_probable": self.most_probable_bitstring(),
                "probabilities": json_safe(prob),
            }
        rho = self._require_density()
        return {
            "type": "density_matrix",
            "dim": int(rho.shape[0]),
            "trace": self.trace(),
            "purity": self.purity(),
            "is_pure": self.is_pure(),
            "diagonal_probabilities": json_safe(self.diagonal_probabilities()),
        }

    def to_json(self) -> str:
        """Serialize the analysis snapshot to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return f"StateAnalysis(kind={self.kind!r}, dim={self.dim})"


__all__ = ["StateAnalysis"]