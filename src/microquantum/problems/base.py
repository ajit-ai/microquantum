"""Generic problem abstractions.

A :class:`Problem` describes *what* needs to be solved, independently of
*how* it is solved.  Problems are plain, serializable data: they never
execute anything and never reference a particular algorithm.  The
algorithm-processable hierarchy is intentionally small:

``Problem -> OptimizationProblem``
``Problem -> EigenvalueProblem -> HamiltonianProblem``
``Problem -> SearchProblem``
``Problem -> SamplingProblem``

Subclassing :class:`Problem` is the documented extension point for custom
problem families (see the ``extension`` examples).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, cast

from .._json import JSONSerializable, json_safe, json_string


@dataclass
class Problem(JSONSerializable):
    """Base class for all generic problem descriptions.

    Attributes:
        name: Human-readable problem identifier.
        num_qubits: Optional qubit count associated with the problem.
        metadata: Free-form problem metadata (never interpreted by the SDK).
    """

    name: str = "problem"
    num_qubits: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_qubits is not None and self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")

    def validate(self) -> list[str]:
        """Return a list of validation problems (empty means valid)."""
        problems: list[str] = []
        if not self.name or not isinstance(self.name, str):
            problems.append("name must be a non-empty string")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        data = cast(dict[str, Any], json_safe(vars(self)))
        type_name = type(self).__name__
        problem_type = ""
        # strip the trailing "Problem" marker for the discriminator
        if type_name.endswith("Problem"):
            problem_type = type_name[: -len("Problem")]
        return {"type": problem_type or type_name, **data}

    def to_json(self) -> str:
        """Serialize to a pretty-printed JSON string."""
        return json_string(self.to_dict())

    def __str__(self) -> str:
        qubits = f", qubits={self.num_qubits}" if self.num_qubits else ""
        return f"{type(self).__name__}(name='{self.name}'{qubits})"


@dataclass
class SamplingProblem(Problem):
    """Ask for samples of a distribution described by a circuit.

    Attributes:
        circuit: Optional circuit whose output distribution is sampled.
            Either ``circuit`` or ``num_qubits`` must be set.
        num_samples: Number of samples/shots requested.
    """

    circuit: Optional[Any] = None
    num_samples: int = 1024

    def __init__(
        self,
        circuit: Optional[Any] = None,
        *,
        num_samples: int = 1024,
        num_qubits: Optional[int] = None,
        name: str = "problem",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.num_qubits = num_qubits
        self.metadata = dict(metadata or {})
        self.circuit = circuit
        self.num_samples = num_samples
        self.__post_init__()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.circuit is not None:
            circuit_num_qubits = cast(Any, self.circuit).num_qubits
            if self.num_qubits is None:
                self.num_qubits = int(circuit_num_qubits)
            elif self.num_qubits != circuit_num_qubits:
                raise ValueError(
                    f"circuit has {circuit_num_qubits} qubits but problem "
                    f"declares {self.num_qubits}"
                )
        elif self.num_qubits is None:
            raise ValueError("SamplingProblem requires a circuit or num_qubits")

    def validate(self) -> list[str]:
        problems = super().validate()
        if self.circuit is None and self.num_qubits is None:
            problems.append("sampling problem requires a circuit or num_qubits")
        return problems

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SamplingProblem":
        """Reconstruct a SamplingProblem from its serialized dictionary."""
        from ..core.serialization import from_dict as circuit_from_dict

        circuit_data = data.get("circuit")
        circuit = circuit_from_dict(circuit_data) if circuit_data else None
        return cls(
            circuit,
            num_samples=int(data.get("num_samples", 1024)),
            num_qubits=data.get("num_qubits"),
            name=data["name"],
            metadata=data.get("metadata") or {},
        )


__all__ = ["Problem", "SamplingProblem"]