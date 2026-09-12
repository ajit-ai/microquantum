"""Generic search problem abstraction.

A :class:`SearchProblem` describes an unstructured database search: find one
(or more) of the ``2^num_qubits`` computational basis states that satisfy a
predicate.  The predicate can be supplied three different ways (at most one):

* ``target`` — an integer index (or list of indices) of the sought state(s);
* ``oracle`` — a callable ``oracle(num_qubits) -> QuantumCircuit`` marking the
  solution subspace;
* ``predicate`` — a callable ``predicate(index) -> bool`` over the integer
  index of a computational basis state; used to classically expand the
  marked set.

The problem stays decoupled from the algorithm (e.g. Grover).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from .base import Problem


@dataclass
class SearchProblem(Problem):
    """Search the ``2^num_qubits``-element database for marked state(s).

    Exactly one of ``target`` / ``oracle`` / ``predicate`` should be given.

    Attributes:
        num_qubits: Size of the search register.
        target: Integer or list of integers identifying the solution basis
            state(s).
        oracle: Callable returning a circuit that marks the solutions.
        predicate: Callable ``predicate(bitstring) -> bool``.
        num_targets: Number of marked states (only used for the optimal
            iteration estimate when ``num_qubits`` is set without a target).
    """

    num_qubits: int = 1
    target: Optional[Union[int, list[int]]] = None
    oracle: Optional[Callable[[int], Any]] = None
    predicate: Optional[Callable[[int], bool]] = None
    num_targets: int = 1

    def __post_init__(self) -> None:
        if self.num_qubits is None or self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        n = 2**self.num_qubits
        if isinstance(self.target, int):
            if self.target < 0 or self.target >= n:
                raise ValueError(
                    f"target {self.target} out of range for "
                    f"{self.num_qubits}-qubit system"
                )
        elif isinstance(self.target, (list, tuple)):
            for t in self.target:
                if t < 0 or t >= n:
                    raise ValueError(
                        f"target {t} out of range for {self.num_qubits}-qubit system"
                    )
        if self.num_targets < 1:
            raise ValueError(f"num_targets must be >= 1, got {self.num_targets}")

    def validate(self) -> list[str]:
        problems = super().validate()
        if self.num_qubits is None or self.num_qubits < 1:
            problems.append("num_qubits must be >= 1")
        providers = sum(
            p is not None for p in (self.target, self.oracle, self.predicate)
        )
        if providers == 0:
            problems.append(
                "search problem requires one of target / oracle / predicate"
            )
        elif providers > 1:
            problems.append(
                "search problem must specify at most one of target / oracle / predicate"
            )
        return problems

    def target_indices(self) -> list[int]:
        """Return the integer indices of the marked states (if known)."""
        if self.target is not None:
            return [self.target] if isinstance(self.target, int) else list(self.target)
        seen: list[int] = []
        for i in range(2**self.num_qubits):
            if self.predicate is not None and self.predicate(i):
                seen.append(i)
        return seen

    def is_marked(self, bitstring: str) -> bool:
        """True if ``bitstring`` (of length ``num_qubits``) is a solution."""
        index = int(bitstring, 2)
        if self.predicate is not None:
            return bool(self.predicate(index))
        if self.target is not None:
            targets = [self.target] if isinstance(self.target, int) else list(self.target)
            return index in targets
        raise ValueError("search problem has no predicate or target to evaluate")

    def num_solutions(self) -> int:
        """Number of marked states (best-effort estimate)."""
        if self.target is not None:
            return 1 if isinstance(self.target, int) else len(self.target)
        if self.predicate is not None:
            return len(self.target_indices())
        return self.num_targets

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["num_qubits"] = self.num_qubits
        data["target"] = self.target if self.target is None else (
            self.target if isinstance(self.target, int) else list(self.target)
        )
        data["oracle"] = None
        data["predicate"] = None
        data["num_targets"] = self.num_targets
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchProblem":
        """Reconstruct a SearchProblem from its serialized dictionary.

        The ``target``-based form round-trips exactly; oracle/predicate
        callables are not serialized and default to ``None``.
        """
        return cls(
            num_qubits=int(data["num_qubits"]),
            target=data.get("target"),
            num_targets=int(data.get("num_targets", 1)),
            name=data["name"],
            metadata=dict(data.get("metadata") or {}),
        )


__all__ = ["SearchProblem"]