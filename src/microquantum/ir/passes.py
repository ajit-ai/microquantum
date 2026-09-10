"""IR transformation passes.

Provides the :class:`IRPass` / :class:`IRPassManager` abstraction and a
small initial set of correct structural transformations:

* :class:`RemoveIdentityGates` — drops gates that are structurally identity
  (identity gates, zero-angle rotations, empty conditional blocks).
* :class:`CancelAdjacentInverse` — cancels adjacent U, U-dagger pairs on the
  same qubits (self-inverse gates, known inverse pairs, opposite rotations).
* :class:`CombineRotations` — merges adjacent same-axis rotation gates on one
  qubit into a single rotation (R(a) R(b) -> R(a + b)).
* :class:`BindParameters` — substitutes symbolic parameters with numeric
  values, producing an executable (bound) IR.
* :class:`GateDecomposition` — expands a gate into a supported basis gate set
  (exact, standard identities only).

All passes are NumPy-free and produce new :class:`IRCircuit` objects,
leaving the input untouched.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Iterable, Mapping, Optional, Union

from ..core.device import Target
from ..core.parameter import Parameter, ParameterExpression
from .circuit_ir import IRCircuit
from .nodes import (
    ConditionalBlock,
    Gate,
    IRNode,
    IRParam,
)

_SELF_INVERSE = frozenset({"h", "x", "y", "z", "cnot", "cz", "swap"})
_INVERSE_PAIRS = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
_ROTATIONS = frozenset({"rx", "ry", "rz"})
_TWO_PI = 2.0 * math.pi


def _is_numeric(value: IRParam) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric(value: IRParam) -> float:
    if isinstance(value, complex):
        return float(value.real)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0  # pragma: no cover - guarded by callers


def _angle_is_identity(angle: float) -> bool:
    """True when a rotation by *angle* equals the identity (mod 2*pi)."""
    return abs(angle % _TWO_PI) < 1e-9 or abs(angle % _TWO_PI) > _TWO_PI - 1e-9


def _are_inverse(left: Gate, right: Gate) -> bool:
    """Structural check: does right == left-dagger on the same qubits?"""
    if left.qubits != right.qubits:
        return False
    if left.name in _ROTATIONS:
        if right.name != left.name:
            return False
        if len(left.params) != 1 or len(right.params) != 1:
            return False
        if not (_is_numeric(left.params[0]) and _is_numeric(right.params[0])):
            return False
        a, b = _numeric(left.params[0]), _numeric(right.params[0])
        if left.name in _ROTATIONS:
            # R(a) and R(b) cancel to first order; also true when (a+b) is
            # a full multiple of 2*pi (which also folds to the small-angle case).
            return abs((a + b) % _TWO_PI) < 1e-9 or abs((a + b) % _TWO_PI) > _TWO_PI - 1e-9
        return abs(a + b) < 1e-9
    if left.params or right.params:
        return False
    if left.name in _SELF_INVERSE and right.name == left.name:
        return True
    return right.name == _INVERSE_PAIRS.get(left.name, "")


class IRPass(ABC):
    """Abstract base class for IR transformation passes.

    A pass inspects and transforms an :class:`IRCircuit` into a new
    :class:`IRCircuit`.  Passes never mutate their input.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pass name."""

    @abstractmethod
    def run(self, ir: IRCircuit) -> IRCircuit:
        """Return the transformed IR circuit."""


class IRPassManager:
    """Runs an ordered pipeline of IR passes.

    Example::

        pm = IRPassManager([RemoveIdentityGates(), CancelAdjacentInverse()])
        optimized = pm.run(ir)
    """

    def __init__(self, passes: Optional[list[IRPass]] = None) -> None:
        self._passes: list[IRPass] = list(passes or [])

    @property
    def passes(self) -> list[IRPass]:
        """The registered passes."""
        return list(self._passes)

    def append_pass(self, pass_: IRPass) -> "IRPassManager":
        """Append a single pass and return self."""
        self._passes.append(pass_)
        return self

    def run(self, ir: IRCircuit) -> IRCircuit:
        """Run all passes sequentially, feeding each output into the next.

        Args:
            ir: The input IR circuit.

        Returns:
            The IR after all passes have been applied.
        """
        result = ir
        for pass_ in self._passes:
            result = pass_.run(result)
        return result


def _rebuilt(
    ir: IRCircuit,
    operations: list[IRNode],
) -> IRCircuit:
    """Build a new IRCircuit preserving metadata and classical bits."""
    return IRCircuit(
        num_qubits=ir.num_qubits,
        num_classical_bits=ir.num_classical_bits,
        name=ir.name,
        operations=operations,
        metadata=dict(ir.metadata),
    )


class RemoveIdentityGates(IRPass):
    """Remove structurally-identity operations.

    Drops ``id`` gates, zero-angle rotation gates, and empty conditional
    blocks (recursively).
    """

    @property
    def name(self) -> str:
        return "remove-identity-gates"

    def _filter(self, operations: list[IRNode]) -> list[IRNode]:
        filtered: list[IRNode] = []
        for op in operations:
            if isinstance(op, Gate):
                if self._is_identity(op):
                    continue
            elif isinstance(op, ConditionalBlock):
                nested = self._filter(list(op.operations))
                if not nested:
                    continue
                op = replace(op, operations=tuple(nested))
            filtered.append(op)
        return filtered

    @staticmethod
    def _is_identity(gate: Gate) -> bool:
        if gate.name == "id":
            return True
        if gate.name in _ROTATIONS and len(gate.params) == 1:
            p = gate.params[0]
            if _is_numeric(p) and _angle_is_identity(_numeric(p)):
                return True
        return False

    def run(self, ir: IRCircuit) -> IRCircuit:
        return _rebuilt(ir, self._filter(list(ir.operations)))


class CancelAdjacentInverse(IRPass):
    """Cancel adjacent inverse gate pairs on the same qubits.

    Recognizes self-inverse gates (H, X, Y, Z, CNOT, CZ, SWAP), the
    standard inverse pairs (S/Sdg, T/Tdg) and opposite rotations
    (R(a) R(-a)).  Barriers and non-gate nodes act as separators.
    """

    @property
    def name(self) -> str:
        return "cancel-adjacent-inverse"

    def _cancel(self, operations: list[IRNode]) -> list[IRNode]:
        result: list[IRNode] = []
        for op in operations:
            if isinstance(op, Gate):
                if result and isinstance(result[-1], Gate) and _are_inverse(
                    result[-1], op
                ):
                    result.pop()
                    continue
                result.append(op)
            elif isinstance(op, ConditionalBlock):
                nested = self._cancel(list(op.operations))
                result.append(replace(op, operations=tuple(nested)))
            else:
                result.append(op)
        return result

    def run(self, ir: IRCircuit) -> IRCircuit:
        return _rebuilt(ir, self._cancel(list(ir.operations)))


class CombineRotations(IRPass):
    """Fuse adjacent same-axis rotations on one qubit: R(a) R(b) -> R(a+b)."""

    @property
    def name(self) -> str:
        return "combine-rotations"

    def _combine(self, operations: list[IRNode]) -> list[IRNode]:
        result: list[IRNode] = []
        for op in operations:
            if (
                isinstance(op, Gate)
                and op.name in _ROTATIONS
                and len(op.qubits) == 1
                and len(op.params) == 1
                and op.condition is None
                and result
                and isinstance(result[-1], Gate)
            ):
                prev = result[-1]
                if (
                    prev.name == op.name
                    and prev.qubits == op.qubits
                    and prev.condition is None
                    and len(prev.params) == 1
                    and _is_numeric(prev.params[0])
                    and _is_numeric(op.params[0])
                ):
                    result[-1] = replace(
                        prev, params=(_numeric(prev.params[0]) + _numeric(op.params[0]),)
                    )
                    continue
            elif isinstance(op, ConditionalBlock):
                result.append(replace(op, operations=tuple(self._combine(list(op.operations)))))
                continue
            result.append(op)
        return result

    def run(self, ir: IRCircuit) -> IRCircuit:
        return _rebuilt(ir, self._combine(list(ir.operations)))


class BindParameters(IRPass):
    """Bind symbolic parameters to numeric values.

    Args:
        param_map: Mapping from :class:`Parameter` (or parameter name) to a
            numeric value.  Parameters absent from the mapping remain
            symbolic in the output IR.
    """

    def __init__(self, param_map: Mapping[Union[str, Parameter], float]) -> None:
        self._param_map = dict(param_map)

    @property
    def name(self) -> str:
        return "bind-parameters"

    def _resolve(self, param: IRParam) -> IRParam:
        if isinstance(param, Parameter):
            if param in self._param_map:
                return float(self._param_map[param])
            if param.name in self._param_map:
                return float(self._param_map[param.name])
            return param
        if isinstance(param, ParameterExpression):
            try:
                return float(param.evaluate(self._param_map))
            except KeyError:
                return param
        return param

    def _bind(self, operations: list[IRNode]) -> list[IRNode]:
        bound: list[IRNode] = []
        for op in operations:
            if isinstance(op, Gate) and op.params:
                op = replace(op, params=tuple(self._resolve(p) for p in op.params))
            elif isinstance(op, ConditionalBlock):
                op = replace(op, operations=tuple(self._bind(list(op.operations))))
            bound.append(op)
        return bound

    def run(self, ir: IRCircuit) -> IRCircuit:
        return _rebuilt(ir, self._bind(list(ir.operations)))


class GateDecomposition(IRPass):
    """Expand gates not in the target basis.

    Only standard, exact identities are used:

    * ``cz(c, t) -> h(t), cnot(c, t), h(t)``   (basis has h and cnot)
    * ``swap(a, b) -> cnot(a,b), cnot(b,a), cnot(a,b)``  (basis has cnot)

    Gates that cannot be decomposed are left in place; use the compiler's
    diagnostics to detect them.

    Args:
        basis_gates: The supported gate-name set (e.g. a
            :class:`~microquantum.core.device.Target`'s ``native_gates``).
    """

    def __init__(
        self,
        basis_gates: Union[Target, Iterable[str]],
    ) -> None:
        if isinstance(basis_gates, Target):
            names: Iterable[str] = basis_gates.native_gates
        else:
            names = basis_gates
        self._basis = set(names)

    @property
    def name(self) -> str:
        return "gate-decomposition"

    def _decompose(self, operations: list[IRNode]) -> list[IRNode]:
        expanded: list[IRNode] = []
        for op in operations:
            if isinstance(op, ConditionalBlock):
                expanded.append(
                    replace(op, operations=tuple(self._decompose(list(op.operations))))
                )
                continue
            if isinstance(op, Gate) and op.name not in self._basis:
                replacement = self._expand(op)
                if replacement is not None:
                    expanded.extend(replacement)
                    continue
            expanded.append(op)
        return expanded

    def _expand(self, gate: Gate) -> Optional[list[Gate]]:
        if len(gate.qubits) != 2 or gate.name not in ("cz", "swap"):
            return None
        c, t = gate.qubits[0], gate.qubits[1]
        if gate.name == "cz" and {"h", "cnot"} <= self._basis:
            return [
                Gate(name="h", qubits=(t,)),
                Gate(name="cnot", qubits=(c, t)),
                Gate(name="h", qubits=(t,)),
            ]
        if gate.name == "swap" and "cnot" in self._basis:
            return [
                Gate(name="cnot", qubits=(c, t)),
                Gate(name="cnot", qubits=(t, c)),
                Gate(name="cnot", qubits=(c, t)),
            ]
        return None

    def run(self, ir: IRCircuit) -> IRCircuit:
        return _rebuilt(ir, self._decompose(list(ir.operations)))

    def to_dict(self) -> dict[str, Any]:
        """Pass configuration for diagnostics."""
        return {"name": self.name, "basis_gates": sorted(self._basis)}


def optimize(ir: IRCircuit, level: int = 1) -> IRCircuit:
    """Run the standard optimization pipeline on *ir*.

    Args:
        ir: The IR circuit to optimize.
        level: 0 = returns input unchanged, 1 = identity removal + inverse
            cancellation, 2 = additionally fuses same-axis rotations.

    Returns:
        The optimized IR circuit.
    """
    if level <= 0:
        return ir
    if level > 2:
        raise ValueError(f"optimization level must be 0..2, got {level}")
    pipeline = [RemoveIdentityGates(), CancelAdjacentInverse()]
    if level >= 2:
        pipeline.append(CombineRotations())
    return IRPassManager(pipeline).run(ir)


__all__ = [
    "IRPass",
    "IRPassManager",
    "RemoveIdentityGates",
    "CancelAdjacentInverse",
    "CombineRotations",
    "BindParameters",
    "GateDecomposition",
    "optimize",
]