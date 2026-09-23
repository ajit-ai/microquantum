"""Quantum counting from the Grover-operator spectrum.

Estimates the number of marked items ``M`` in an ``N``-item search space
from the eigenvalues of the Grover operator ``Q``: ``Q`` has eigenvalues
``e^{±2iβ}`` with ``sin²β = M/N`` (plus a trivial ``+1`` eigenspace), so
the count follows directly from the spectrum.  On a simulator this is
exact; on hardware the same quantity is what QPE-based counting (see
:class:`AmplitudeEstimation`) samples.  Oracle construction reuses the
diagonal phase-oracle pattern from :class:`GroverSearch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .._json import JSONSerializable
from ..core.circuit import QuantumCircuit
from ..core.operators import Operator
from ..problems.search import SearchProblem
from .amplitude_estimation import AmplitudeEstimation

__all__ = [
    "QuantumCountingResult",
    "QuantumCounting",
]


@dataclass
class QuantumCountingResult(JSONSerializable):
    """Outcome of a quantum counting run."""

    estimated_count: int = 0
    fraction: float = 0.0
    num_qubits: int = 0
    confidence: float = 0.0
    evaluations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "estimated_count": self.estimated_count,
            "fraction": self.fraction,
            "num_qubits": self.num_qubits,
            "confidence": self.confidence,
            "evaluations": self.evaluations,
            "metadata": dict(self.metadata),
        }


class QuantumCounting:
    """Count marked items from the Grover-operator spectrum.

    Implements the ``validate`` / ``solve`` algorithm interface
    (duck-typed, following :class:`VQE` and :class:`GroverSearch`).

    Args:
        num_evaluation_qubits: Precision qubits for amplitude estimation.
        shots: Shots per amplitude-estimation evaluation.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_evaluation_qubits: int = 3,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> None:
        if num_evaluation_qubits < 1:
            raise ValueError("num_evaluation_qubits must be >= 1")
        if shots < 1:
            raise ValueError("shots must be >= 1")
        self._num_evaluation_qubits = num_evaluation_qubits
        self._shots = shots
        self._seed = seed

    @property
    def name(self) -> str:
        """Algorithm identifier."""
        return "quantum_counting"

    def validate(self, problem: Any) -> list[str]:
        """Return defects of *problem* for counting (empty when valid)."""
        errors: list[str] = []
        if not isinstance(problem, SearchProblem):
            return [f"QuantumCounting solves SearchProblem, got {type(problem).__name__}"]
        errors.extend(problem.validate())
        if not problem.target_indices():
            errors.append("SearchProblem has no marked targets to count")
        return errors

    def build_oracle(self, num_qubits: int, targets: Sequence[int | str]) -> QuantumCircuit:
        """Build the diagonal phase oracle marking *targets*.

        Applies a ``-1`` phase to marked computational-basis states and
        ``+1`` elsewhere, following the :class:`GroverSearch` pattern.
        Targets may be integer indices or bitstrings (``"01"``).
        """
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        dim = 2**num_qubits
        indices: list[int] = []
        for target in targets:
            index = int(target, 2) if isinstance(target, str) else int(target)
            if not 0 <= index < dim:
                raise ValueError(f"Target {target!r} out of range for {num_qubits} qubits")
            indices.append(index)
        diagonal = np.ones(dim, dtype=np.complex128)
        for index in indices:
            diagonal[index] = -1.0
        oracle = QuantumCircuit(num_qubits)
        oracle.append(Operator(np.diag(diagonal), name="oracle"), list(range(num_qubits)))
        return oracle

    def build_state_preparation(self, num_qubits: int) -> QuantumCircuit:
        """Build the uniform-superposition state preparation."""
        preparation = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            preparation.h(qubit)
        return preparation

    def count(
        self,
        num_qubits: int,
        targets: Sequence[int | str],
        num_shots: Optional[int] = None,
    ) -> QuantumCountingResult:
        """Count *targets* in a ``num_qubits`` search space.

        Builds the Grover operator for the uniform superposition and
        reads the marked fraction from measured operator data: the
        oracle trace resolves the ``M = 0`` / ``M = N`` ambiguity
        (identical Grover spectra), otherwise ``M = N·sin²β`` with
        ``±2β`` the non-trivial Grover eigenphases.  Exact on a
        simulator up to floating-point error.
        """
        oracle = self.build_oracle(num_qubits, targets)
        preparation = self.build_state_preparation(num_qubits)
        estimator = AmplitudeEstimation(
            num_evaluation_qubits=self._num_evaluation_qubits,
            state_preparation=preparation,
            oracle=oracle,
        )
        size = 2**num_qubits
        oracle_trace = round(float(np.real(np.trace(oracle.get_unitary().matrix))))
        if oracle_trace == size:
            count = 0
        elif oracle_trace == -size:
            count = size
        else:
            grover = estimator.build_grover_operator(num_qubits, preparation, oracle)
            eigenvalues = np.linalg.eigvals(grover.get_unitary().matrix)
            pair: list[float] = []
            for eigenvalue in eigenvalues:
                raw = abs(float(np.angle(eigenvalue)))
                if raw <= 1e-6 or abs(raw - np.pi) <= 1e-6:
                    continue
                pair.append(min(raw, 2.0 * np.pi - raw))
            beta = (max(pair) / 2.0) if pair else 0.0
            count = int(round(size * float(np.sin(beta) ** 2)))
        fraction = count / size
        return QuantumCountingResult(
            estimated_count=count,
            fraction=fraction,
            num_qubits=num_qubits,
            confidence=1.0,
            evaluations=size,
            metadata={"shots": num_shots or self._shots, "seed": self._seed},
        )

    def estimate_count(
        self,
        num_qubits: int,
        targets: Sequence[int | str],
        *,
        num_evaluation_qubits: Optional[int] = None,
        num_shots: Optional[int] = None,
    ) -> QuantumCountingResult:
        """Count via QPE sampling (:class:`AmplitudeEstimation`).

        Runs the hardware-faithful QPE path instead of the exact
        spectral analysis: the estimate lands on the nearest QPE bin,
        so results carry genuine sampling/precision behavior.  Agrees
        with :meth:`count` whenever the marked fraction is exactly
        representable with the evaluation-qubit precision.
        """
        oracle = self.build_oracle(num_qubits, targets)
        preparation = self.build_state_preparation(num_qubits)
        precision = num_evaluation_qubits or self._num_evaluation_qubits
        estimator = AmplitudeEstimation(
            num_evaluation_qubits=precision,
            state_preparation=preparation,
            oracle=oracle,
        )
        estimate = estimator.estimate(
            num_qubits,
            state_prep=preparation,
            oracle=oracle,
            num_shots=num_shots or self._shots,
        )
        size = 2**num_qubits
        fraction = min(max(estimate.estimated_amplitude, 0.0), 1.0)
        return QuantumCountingResult(
            estimated_count=int(round(size * fraction)),
            fraction=fraction,
            num_qubits=num_qubits,
            confidence=estimate.confidence_interval,
            evaluations=estimate.num_evaluations,
            metadata={
                "shots": num_shots or self._shots,
                "seed": self._seed,
                "method": "qpe-sampling",
            },
        )

    def solve(self, problem: Any, runtime: Any = None) -> QuantumCountingResult:
        """Count the marked targets of a :class:`SearchProblem`."""
        errors = self.validate(problem)
        if errors:
            raise ValueError("Problem validation failed: " + "; ".join(errors))
        assert isinstance(problem, SearchProblem)
        targets = problem.target_indices()
        if problem.target is None and not targets:
            raise ValueError("SearchProblem has no marked targets to count")
        return self.count(problem.num_qubits, targets)
