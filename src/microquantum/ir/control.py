"""Control-flow IR nodes: bounded loops and classical switches.

:class:`Loop` repeats a body a fixed (or symbolic) number of times;
:class:`Switch` selects a body by classical-bit value.  Both are frozen
dataclasses following the :mod:`ir.nodes` conventions (structural
identity, ``to_dict`` support) and are validated by
:mod:`ir.validation`, traversed by :meth:`IRCircuit.walk` and counted
by the gate/depth analyses like any nested block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from ..core.parameter import Parameter
from .nodes import Condition, IRNode, IRParam

__all__ = [
    "Loop",
    "Switch",
    "loop_from_dict",
    "switch_from_dict",
]


@dataclass(frozen=True)
class Loop(IRNode):
    """Repeat ``body`` ``trip_count`` times.

    Attributes:
        body: Ordered nodes forming the loop body.
        trip_count: Non-negative repetition count, or a symbolic
            :class:`Parameter` bound before execution.
    """

    body: tuple[IRNode, ...] = ()
    trip_count: Union[int, Parameter] = 1

    def __post_init__(self) -> None:
        if isinstance(self.trip_count, bool) or (
            isinstance(self.trip_count, int) and self.trip_count < 0
        ):
            raise ValueError(f"trip_count must be >= 0, got {self.trip_count!r}")
        if not isinstance(self.trip_count, (int, Parameter)):
            raise TypeError(
                "trip_count must be an int or Parameter, "
                f"got {type(self.trip_count).__name__}"
            )

    @property
    def kind(self) -> str:
        """Node kind identifier."""
        return "loop"

    @property
    def qubits(self) -> tuple[int, ...]:
        """Sorted union of body qubit indices."""
        used: set[int] = set()
        for op in self.body:
            used.update(getattr(op, "qubits", ()))
        return tuple(sorted(used))

    @property
    def cbits(self) -> tuple[int, ...]:
        """Sorted union of body classical-bit indices."""
        used: set[int] = set()
        for op in self.body:
            used.update(op.cbits)
        return tuple(sorted(used))

    @property
    def parameters(self) -> tuple[IRParam, ...]:
        """Body parameters plus a symbolic trip count, if any."""
        found: list[IRParam] = []
        for op in self.body:
            found.extend(op.parameters)
        if isinstance(self.trip_count, Parameter):
            found.append(self.trip_count)
        return tuple(found)

    def unrolled(self) -> tuple[IRNode, ...]:
        """Return the body repeated for a concrete trip count.

        Raises:
            ValueError: If the trip count is still symbolic.
        """
        if isinstance(self.trip_count, Parameter):
            raise ValueError("Cannot unroll a Loop with a symbolic trip count")
        return tuple(op for _ in range(self.trip_count) for op in self.body)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        count = (
            {"symbol": self.trip_count.name}
            if isinstance(self.trip_count, Parameter)
            else {"value": int(self.trip_count)}
        )
        return {
            "kind": self.kind,
            "qubits": list(self.qubits),
            "trip_count": count,
            "body": [op.to_dict() for op in self.body],
        }


@dataclass(frozen=True)
class Switch(IRNode):
    """Select a body by classical-bit value.

    Attributes:
        condition: Classical condition selecting the case.
        cases: ``((value, body), ...)`` pairs; ``value`` is 0 or 1.
        default: Body executed when no case matches.
    """

    condition: Condition = field(default_factory=lambda: Condition(bit=0))
    cases: tuple[tuple[int, tuple[IRNode, ...]], ...] = ()
    default: tuple[IRNode, ...] = ()

    def __post_init__(self) -> None:
        for value, _ in self.cases:
            if value not in (0, 1):
                raise ValueError(f"Switch case value must be 0 or 1, got {value}")

    @property
    def kind(self) -> str:
        """Node kind identifier."""
        return "switch"

    @property
    def qubits(self) -> tuple[int, ...]:
        """Sorted union of case/default qubit indices."""
        used: set[int] = set()
        for _, body in self.cases:
            for op in body:
                used.update(getattr(op, "qubits", ()))
        for op in self.default:
            used.update(getattr(op, "qubits", ()))
        return tuple(sorted(used))

    @property
    def cbits(self) -> tuple[int, ...]:
        """Condition bit plus sorted union of body classical bits."""
        used = {self.condition.bit}
        for _, body in self.cases:
            for op in body:
                used.update(op.cbits)
        for op in self.default:
            used.update(op.cbits)
        return tuple(sorted(used))

    @property
    def parameters(self) -> tuple[IRParam, ...]:
        """Union of case/default parameters."""
        found: list[IRParam] = []
        for _, body in self.cases:
            for op in body:
                found.extend(op.parameters)
        for op in self.default:
            found.extend(op.parameters)
        return tuple(found)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "kind": self.kind,
            "qubits": list(self.qubits),
            "condition": self.condition.to_dict(),
            "cases": [
                {"value": value, "body": [op.to_dict() for op in body]}
                for value, body in self.cases
            ],
            "default": [op.to_dict() for op in self.default],
        }


def _decode_body(items: list[Any]) -> tuple[IRNode, ...]:
    """Decode nested node dictionaries (base kinds plus control nodes)."""
    from .nodes import Barrier, ConditionalBlock, Gate, Measurement, Reset  # noqa: PLC0415

    decoded: list[IRNode] = []
    for item in items:
        kind = item.get("kind")
        if kind == "loop":
            decoded.append(loop_from_dict(item))
        elif kind == "switch":
            decoded.append(switch_from_dict(item))
        elif kind == "gate":
            decoded.append(
                Gate(
                    name=str(item["name"]),
                    qubits=tuple(int(q) for q in item.get("qubits", [])),
                    params=tuple(_decode_param(p) for p in item.get("params", [])),
                )
            )
        elif kind == "measure":
            decoded.append(
                Measurement(qubit=int(item["qubit"]), classical=item.get("classical"))
            )
        elif kind == "reset":
            decoded.append(Reset(qubit=int(item["qubit"])))
        elif kind == "barrier":
            decoded.append(Barrier(qubits=tuple(int(q) for q in item.get("qubits", []))))
        elif kind == "conditional":
            decoded.append(
                ConditionalBlock(
                    condition=Condition(
                        bit=int(item["condition"]["bit"]),
                        value=int(item["condition"].get("value", 1)),
                    ),
                    operations=_decode_body(item.get("operations", [])),
                )
            )
        else:
            raise ValueError(f"Unknown IR node kind '{kind}'")
    return tuple(decoded)


def _decode_param(spec: Any) -> IRParam:
    """Decode a parameter dictionary produced by ``param_to_dict``."""
    if isinstance(spec, dict) and "symbol" in spec and "coefficient" not in spec:
        return Parameter(str(spec["symbol"]))
    return float(spec["value"]) if isinstance(spec, dict) else float(spec)


def loop_from_dict(data: dict[str, Any]) -> Loop:
    """Rebuild a :class:`Loop` from :meth:`Loop.to_dict` output."""
    if data.get("kind") != "loop":
        raise ValueError(f"Expected a loop node, got kind '{data.get('kind')}'")
    count = data.get("trip_count", {"value": 1})
    trip: Union[int, Parameter]
    if isinstance(count, dict) and "symbol" in count:
        trip = Parameter(str(count["symbol"]))
    else:
        trip = int(count.get("value", 1))
    return Loop(body=_decode_body(data.get("body", [])), trip_count=trip)


def switch_from_dict(data: dict[str, Any]) -> Switch:
    """Rebuild a :class:`Switch` from :meth:`Switch.to_dict` output."""
    if data.get("kind") != "switch":
        raise ValueError(f"Expected a switch node, got kind '{data.get('kind')}'")
    raw_condition = data.get("condition", {"bit": 0})
    condition = Condition(bit=int(raw_condition["bit"]), value=int(raw_condition.get("value", 1)))
    cases = tuple(
        (int(case["value"]), _decode_body(case.get("body", []))) for case in data.get("cases", [])
    )
    return Switch(
        condition=condition,
        cases=cases,
        default=_decode_body(data.get("default", [])),
    )
