"""Sampling analysis (MQ-07).

:class:`SamplingAnalysis` derives mathematically well-defined summaries from
measurement **counts/samples** without assuming any particular meaning for the
bitstrings: total shots, unique outcomes, probabilities, the most frequent
outcome, entropy and marginal distributions.

For ``mean``/``variance``/``standard_deviation`` a numeric *observable* is
required — by convention the default interprets a bitstring as an unsigned
binary integer (MSB first), but an explicit ``value_of`` callable can map
outcomes to any real number.  Bitstring semantics are never silently assumed
beyond that documented convention.

The analyzer accepts a :class:`BackendResult`, an
:class:`ExecutionRecord`, an :class:`ExperimentResult`, a mapping of counts or
the ``to_dict()`` forms of any of these.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional

from .._json import JSONSerializable, json_string
from . import statistics as _stats

Outcome = str
ValueOf = Callable[[str], float]


def _int_value(bitstring: str) -> float:
    """Default observable: unsigned binary interpretation (MSB first)."""
    return float(int(bitstring, 2))


def _accepts_counts(obj: Any) -> bool:
    counts = getattr(obj, "counts", None)
    if isinstance(counts, dict):
        return counts is not None
    return False


def _normalize_counts(source: Any) -> dict[str, int]:
    """Extract ``{bitstring: count}`` from a supported source."""
    counts: Optional[Dict[str, int]] = None
    if isinstance(source, dict):
        if "counts" in source and isinstance(source["counts"], dict):
            counts = source["counts"]
        elif all(isinstance(k, str) for k in source):
            counts = source
    elif hasattr(source, "counts") and isinstance(source.counts, dict):
        counts = source.counts
    if counts is None:
        raise TypeError(
            "SamplingAnalysis needs a BackendResult/ExecutionRecord/"
            "ExperimentResult, a counts mapping or a serialized dict; "
            f"got {type(source).__name__}"
        )
    out: dict[str, int] = {}
    for bitstring, value in counts.items():
        if not isinstance(bitstring, str):
            raise ValueError(f"sampling outcomes must be bitstrings, got {bitstring!r}")
        try:
            out[bitstring] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sampling count for '{bitstring}' must be an integer, "
                f"got {value!r}"
            ) from exc
    if any(v < 0 for v in out.values()):
        raise ValueError("sampling counts must be non-negative")
    return out


class SamplingAnalysis(JSONSerializable):
    """Backend-independent sampling/measurement analysis.

    Args:
        source: A :class:`BackendResult`, :class:`ExecutionRecord`,
            :class:`ExperimentResult`, a ``{bitstring: count}`` mapping, or a
            serialized ``to_dict()`` dict carrying ``counts``.
    """

    def __init__(self, source: Any) -> None:
        self._counts = _normalize_counts(source)
        self._length = _bitstring_length(self._counts)

    # -- basics ---------------------------------------------------------------

    @property
    def counts(self) -> dict[str, int]:
        """Raw measurement counts (read-only view)."""
        return dict(self._counts)

    def total_shots(self) -> int:
        """Total number of shots represented by the counts."""
        return sum(self._counts.values())

    def unique_outcomes(self) -> list[str]:
        """Distinct measured bitstrings, sorted for determinism."""
        return sorted(self._counts)

    def probabilities(self) -> dict[str, float]:
        """Probability per outcome (``count / total``), sorted by outcome."""
        total = self.total_shots()
        if total == 0:
            return {}
        return {k: v / total for k, v in sorted(self._counts.items())}

    # -- extremes -------------------------------------------------------------

    def most_likely(self) -> str:
        """Most frequently measured bitstring.

        Raises:
            ValueError: If there are no counts.
        """
        if not self._counts:
            raise ValueError("no sampling outcomes available")
        return max(sorted(self._counts), key=lambda k: self._counts[k])

    def most_likely_probability(self) -> float:
        """Probability of the most frequent outcome."""
        return self.probabilities().get(self.most_likely(), 0.0)

    def least_likely(self) -> str:
        """Least frequently measured (non-zero) bitstring."""
        if not self._counts:
            raise ValueError("no sampling outcomes available")
        return min(sorted(self._counts), key=lambda k: self._counts[k])

    # -- information ----------------------------------------------------------

    def entropy(self, base: float = 2.0) -> float:
        """Shannon entropy of the outcome distribution (default: *bits*).

        An empty counts dict yields ``0.0``; a deterministic single outcome
        yields ``0.0``; a uniform distribution over ``n`` outcomes yields
        ``log_base(n)``.
        """
        if base <= 1:
            raise ValueError(f"entropy base must be > 1, got {base}")
        total = self.total_shots()
        if total == 0:
            return 0.0
        result = 0.0
        for prob in self.probabilities().values():
            if prob > 0:
                result -= prob * math.log(prob) / math.log(base)
        return result

    # -- numeric observable ---------------------------------------------------

    def _scores(self, value_of: Optional[ValueOf]) -> list[float]:
        if value_of is None:
            value_of = _int_value
        if not callable(value_of):
            raise TypeError("value_of must be a callable mapping bitstring -> float")
        scores: list[float] = []
        for bitstring, count in self._counts.items():
            value = float(value_of(bitstring))
            if not math.isfinite(value):
                raise ValueError(
                    f"observable produced a non-finite value for '{bitstring}'"
                )
            scores.extend([value] * count)
        return scores

    def mean(self, value_of: Optional[ValueOf] = None) -> float:
        """Mean of the observable over all shots.

        Args:
            value_of: Optional ``bitstring -> number`` mapping.  Defaults to
                the unsigned-binary integer value of the outcome (MSB first).
        """
        return _stats.mean(self._scores(value_of))

    def variance(self, value_of: Optional[ValueOf] = None, *, ddof: int = 0) -> float:
        """Variance of the observable (``ddof=0`` population, ``1`` sample)."""
        return _stats.variance(self._scores(value_of), ddof=ddof)

    def standard_deviation(
        self, value_of: Optional[ValueOf] = None, *, ddof: int = 0
    ) -> float:
        """Standard deviation of the observable."""
        return _stats.standard_deviation(self._scores(value_of), ddof=ddof)

    def expected_value(self, observable: Callable[[str], float]) -> float:
        """Expectation of an arbitrary bitstring-observable over the empirical
        distribution: ``E[O] = sum_b p(b) * O(b)``."""
        if not callable(observable):
            raise TypeError("observable must be a callable mapping bitstring -> float")
        total = self.total_shots()
        if total == 0:
            raise ValueError("no sampling outcomes available")
        return sum(
            count * float(observable(bitstring))
            for bitstring, count in self._counts.items()
        ) / total

    # -- marginals ------------------------------------------------------------

    def marginal(self, qubits: Iterable[int]) -> dict[str, float]:
        """Probability distribution over a set of qubits.

        Args:
            qubits: Physical qubit indices (within the bitstring length) to
                keep; the marginal drops the remaining qubits.

        Returns:
            A ``{reduced_bitstring: probability}`` mapping.  Reduced bitstrings
            preserve the physical qubit order of the given indices.
        """
        indices = sorted(set(int(q) for q in qubits))
        if not indices:
            raise ValueError("marginal() requires at least one qubit")
        length = self._length
        if length is None:
            raise ValueError("sampling outcomes have no bitstrings to marginalize")
        if any(i < 0 or i >= length for i in indices):
            raise ValueError(
                f"marginal qubit indices must be within 0..{length - 1}, "
                f"got {sorted(indices)}"
            )
        marginal_counts: dict[str, int] = {}
        for bitstring, count in self._counts.items():
            reduced = "".join(bitstring[i] for i in indices)
            marginal_counts[reduced] = marginal_counts.get(reduced, 0) + count
        total = sum(marginal_counts.values())
        return {k: v / total for k, v in sorted(marginal_counts.items())}

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the analysis snapshot (JSON-safe, deterministic)."""
        try:
            most_likely = self.most_likely()
        except ValueError:
            most_likely = None
        return {
            "type": "sampling",
            "counts": dict(self._counts),
            "total_shots": self.total_shots(),
            "probabilities": self.probabilities(),
            "entropy": self.entropy(),
            "most_likely": most_likely,
            "unique_outcomes": self.unique_outcomes(),
        }

    def to_json(self) -> str:
        """Serialize the analysis snapshot to a JSON string."""
        return json_string(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"SamplingAnalysis(shots={self.total_shots()}, "
            f"outcomes={len(self._counts)})"
        )


def _bitstring_length(counts: dict[str, int]) -> Optional[int]:
    """Common bitstring length across outcomes (None when ambiguous)."""
    lengths = {len(k) for k in counts}
    return lengths.pop() if len(lengths) == 1 else None


__all__ = ["SamplingAnalysis"]