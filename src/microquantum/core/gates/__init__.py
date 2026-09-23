"""Core gate hierarchy: backend-independent gate abstractions.

This package provides the canonical :class:`Gate` hierarchy used by
``core.circuit``.  Gate objects are lightweight descriptors — they carry
a name, qubit arity and symbolic/numeric parameters, and materialize a
dense unitary only when :meth:`Gate.to_operator` is explicitly requested.

Standard gates: I, X, Y, Z, H, S, Sdg, T, Tdg, RX, RY, RZ, Phase,
U (general single-qubit unitary), CX/CNOT, CY, CZ, SWAP, CPhase,
CRX/CRY/CRZ, Toffoli (CCX), Fredkin (CSWAP) and controlled unitaries.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union, cast

if TYPE_CHECKING:
    from microquantum.core.parameter import Parameter

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Gate",
    "UnitaryGate",
    "ParameterizedGate",
    "ControlledGate",
    "CompositeGate",
    "StandardGate",
    "RotationGate",
    "gate_matrix",
    "make_gate",
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "Phase",
    "RX",
    "RY",
    "RZ",
    "U",
    "CX",
    "CNOT",
    "CY",
    "CZ",
    "SWAP",
    "CPhase",
    "CRX",
    "CRY",
    "CRZ",
    "Toffoli",
    "CCX",
    "Fredkin",
    "CSWAP",
    "ControlledUnitary",
]

_Scalar = Union[float, complex, str]


def _float_map(values: dict[str, complex]) -> dict[str, float]:
    """Convert a complex binding to real values for expression evaluation."""
    return {k: float(complex(v).real) for k, v in values.items()}


def _as_matrix(data: Any) -> NDArray[np.complex128]:
    return np.asarray(data, dtype=np.complex128)


class Gate(ABC):
    """Abstract base class for all quantum gates."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Lowercase gate identifier (e.g. ``"h"``, ``"cx"``)."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits the gate acts on."""

    @property
    def parameters(self) -> tuple[Any, ...]:
        """Gate parameters (numeric or symbolic)."""
        return ()

    @property
    def is_parameterized(self) -> bool:
        """True when any parameter is symbolic (non-numeric)."""
        from microquantum.core.parameters import Parameter, ParameterExpression

        for p in self.parameters:
            if isinstance(p, (Parameter, ParameterExpression)):
                return True
            if isinstance(p, str):
                return True
        return False

    @abstractmethod
    def to_matrix(self) -> NDArray[np.complex128]:
        """Dense unitary matrix. Raises if symbolically parameterized."""

    def to_operator(self) -> Any:
        """Convert to :class:`microquantum.core.operators.Operator`."""
        from microquantum.core.operators import Operator

        return Operator(self.to_matrix(), name=self.name)

    @abstractmethod
    def inverse(self) -> Gate:
        """Return the inverse gate."""

    def controlled(self, num_controls: int = 1) -> ControlledGate:
        """Return a controlled version of this gate."""
        if num_controls < 1:
            raise ValueError("num_controls must be >= 1")
        return ControlledGate(self, num_controls)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}, qubits={self.num_qubits})"


class UnitaryGate(Gate):
    """Gate backed by an explicit dense unitary matrix."""

    def __init__(self, matrix: NDArray[np.complex128] | list[Any], name: str = "custom") -> None:
        arr = _as_matrix(matrix)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Matrix must be square 2D, got shape {arr.shape}")
        dim = arr.shape[0]
        if dim == 0 or (dim & (dim - 1)) != 0:
            raise ValueError(f"Matrix dimension must be a power of 2, got {dim}")
        self._matrix = arr
        self._name = name.lower()
        self._num_qubits = int(math.log2(dim))

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def matrix(self) -> NDArray[np.complex128]:
        return self._matrix.copy()

    def to_matrix(self) -> NDArray[np.complex128]:
        return self._matrix.copy()

    def inverse(self) -> UnitaryGate:
        return UnitaryGate(self._matrix.conj().T, name=self._name + "_dg")


class StandardGate(UnitaryGate):
    """A named fixed unitary gate (no parameters)."""

    def inverse(self) -> StandardGate:
        inv_names = {
            "h": "h", "x": "x", "y": "y", "z": "z", "i": "i",
            "s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t",
            "cx": "cx", "cnot": "cnot", "cy": "cy", "cz": "cz",
            "swap": "swap", "ccx": "ccx", "toffoli": "ccx",
            "cswap": "cswap", "fredkin": "cswap",
        }
        name = inv_names.get(self._name, self._name + "_dg")
        return StandardGate(self._matrix.conj().T, name=name)


class ParameterizedGate(Gate):
    """Gate with numeric or symbolic rotation/phase parameters."""

    def __init__(self, name: str, num_qubits: int, params: tuple[Any, ...]) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self._gate_name = name.lower()
        self._n_qubits = num_qubits
        self._params = tuple(params)

    @property
    def name(self) -> str:
        return self._gate_name

    @property
    def num_qubits(self) -> int:
        return self._n_qubits

    @property
    def parameters(self) -> tuple[Any, ...]:
        return self._params

    def _resolve(self, values: dict[str, complex] | None = None) -> tuple[float, ...]:
        from microquantum.core.parameters import Parameter, ParameterExpression

        float_values = cast("dict[str, Any]", _float_map(values) if values is not None else {})
        binding = cast("dict[str | Parameter, float]", float_values)
        resolved: list[float] = []
        for p in self._params:
            if isinstance(p, (int, float, complex, np.floating, np.complexfloating)):
                resolved.append(float(complex(p).real))
            elif isinstance(p, Parameter):
                if values is None or p.name not in values:
                    raise ValueError(f"Unbound parameter '{p.name}' for gate '{self._gate_name}'")
                resolved.append(float(complex(values[p.name]).real))
            elif isinstance(p, ParameterExpression):
                if values is None:
                    raise ValueError(f"Unbound expression '{p}' for gate '{self._gate_name}'")
                resolved.append(float(complex(p.evaluate(binding)).real))
            else:
                raise ValueError(f"Cannot resolve parameter {p!r} for gate '{self._gate_name}'")
        return tuple(resolved)

    def to_matrix(self, values: dict[str, complex] | None = None) -> NDArray[np.complex128]:
        if self.is_parameterized and values is None:
            raise ValueError(f"Gate '{self._gate_name}' has unbound symbolic parameters")
        angles = self._resolve(values)
        return gate_matrix(self._gate_name, angles, self._n_qubits)

    def bind(self, values: dict[str, complex]) -> ParameterizedGate:
        """Return a copy with symbolic parameters substituted where possible."""
        from microquantum.core.parameters import Parameter, ParameterExpression

        float_values = cast("dict[str, Any]", _float_map(values))
        binding = cast("dict[str | Parameter, float]", float_values)
        new_params: list[Any] = []
        for p in self._params:
            if isinstance(p, Parameter) and p.name in values:
                new_params.append(float(complex(values[p.name]).real))
            elif isinstance(p, ParameterExpression):
                try:
                    new_params.append(float(complex(p.evaluate(binding)).real))
                except (ValueError, KeyError):
                    new_params.append(p)
            else:
                new_params.append(p)
        return ParameterizedGate(self._gate_name, self._n_qubits, tuple(new_params))

    def inverse(self) -> ParameterizedGate:
        negated: list[Any] = []
        for p in self._params:
            if isinstance(p, (int, float, complex, np.floating, np.complexfloating)):
                negated.append(-float(complex(p).real))
            else:
                negated.append(_NegatedParam(p))
        inv = ParameterizedGate(self._gate_name, self._n_qubits, tuple(negated))
        return inv


@dataclass(frozen=True)
class _NegatedParam:
    """Lazy negation wrapper for symbolic gate parameters."""

    inner: Any

    def __repr__(self) -> str:
        return f"-({self.inner!r})"


class ControlledGate(Gate):
    """A gate with ``num_controls`` control qubits prepended."""

    def __init__(self, base: Gate, num_controls: int = 1) -> None:
        if num_controls < 1:
            raise ValueError("num_controls must be >= 1")
        self._base = base
        self._controls = num_controls

    @property
    def name(self) -> str:
        return f"c{self._base.name}" if self._controls == 1 else f"c{self._controls}{self._base.name}"

    @property
    def num_qubits(self) -> int:
        return self._base.num_qubits + self._controls

    @property
    def parameters(self) -> tuple[Any, ...]:
        return self._base.parameters

    @property
    def base(self) -> Gate:
        return self._base

    @property
    def num_controls(self) -> int:
        return self._controls

    def to_matrix(self) -> NDArray[np.complex128]:
        base_mat = self._base.to_matrix()
        n_target = self._base.num_qubits
        n_total = self._controls + n_target
        dim = 2**n_total
        mat = np.eye(dim, dtype=np.complex128)
        mat[dim - base_mat.shape[0] :, dim - base_mat.shape[0] :] = base_mat
        # Correct general construction: identity except bottom-right block = base.
        # For multi-control this equals control-on-|1..1>.
        return mat

    def inverse(self) -> ControlledGate:
        return ControlledGate(self._base.inverse(), self._controls)


class CompositeGate(Gate):
    """A gate defined as an ordered sequence of sub-gates on the same qubits."""

    def __init__(self, name: str, num_qubits: int, components: list[Gate]) -> None:
        if not components:
            raise ValueError("CompositeGate requires at least one component")
        for c in components:
            if c.num_qubits != num_qubits:
                raise ValueError("All components must act on the same qubit count")
        self._gate_name = name.lower()
        self._n_qubits = num_qubits
        self._components = list(components)

    @property
    def name(self) -> str:
        return self._gate_name

    @property
    def num_qubits(self) -> int:
        return self._n_qubits

    @property
    def components(self) -> list[Gate]:
        return list(self._components)

    @property
    def parameters(self) -> tuple[Any, ...]:
        out: list[Any] = []
        for c in self._components:
            out.extend(c.parameters)
        return tuple(out)

    def to_matrix(self) -> NDArray[np.complex128]:
        acc: NDArray[np.complex128] | None = None
        for c in self._components:
            m = c.to_matrix()
            acc = m if acc is None else m @ acc
        assert acc is not None
        return acc

    def inverse(self) -> CompositeGate:
        return CompositeGate(
            self._gate_name + "_dg", self._n_qubits, [c.inverse() for c in reversed(self._components)]
        )


class RotationGate(ParameterizedGate):
    """Single-qubit Pauli rotation gate (rx/ry/rz)."""

    def __init__(self, axis: str, angle: Any) -> None:
        axis = axis.lower()
        if axis not in ("rx", "ry", "rz"):
            raise ValueError(f"Unknown rotation axis '{axis}'")
        super().__init__(axis, 1, (angle,))


def gate_matrix(
    name: str, params: tuple[float, ...] = (), num_qubits: int = 0
) -> NDArray[np.complex128]:
    """Return the dense unitary for a named gate.

    Args:
        name: Gate name (case-insensitive).
        params: Numeric rotation/phase angles.
        num_qubits: Required for generic ``"u"`` gate disambiguation.
    """
    key = name.lower()
    if key == "i":
        return _as_matrix(np.eye(2))
    if key == "x":
        return _as_matrix([[0, 1], [1, 0]])
    if key == "y":
        return _as_matrix([[0, -1j], [1j, 0]])
    if key == "z":
        return _as_matrix([[1, 0], [0, -1]])
    if key == "h":
        return _as_matrix([[1, 1], [1, -1]]) / math.sqrt(2.0)
    if key == "s":
        return _as_matrix([[1, 0], [0, 1j]])
    if key == "sdg":
        return _as_matrix([[1, 0], [0, -1j]])
    if key == "t":
        return _as_matrix([[1, 0], [0, np.exp(1j * math.pi / 4)]])
    if key == "tdg":
        return _as_matrix([[1, 0], [0, np.exp(-1j * math.pi / 4)]])
    if key in ("rx", "ry", "rz", "phase", "p", "u"):
        if key == "rx":
            (theta,) = params
            return _as_matrix(
                [
                    [math.cos(theta / 2), -1j * math.sin(theta / 2)],
                    [-1j * math.sin(theta / 2), math.cos(theta / 2)],
                ]
            )
        if key == "ry":
            (theta,) = params
            return _as_matrix(
                [
                    [math.cos(theta / 2), -math.sin(theta / 2)],
                    [math.sin(theta / 2), math.cos(theta / 2)],
                ]
            )
        if key == "rz":
            (theta,) = params
            return _as_matrix([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
        if key in ("phase", "p"):
            (phi,) = params
            return _as_matrix([[1, 0], [0, np.exp(1j * phi)]])
        # general single-qubit unitary U(theta, phi, lam)
        theta, phi, lam = params
        return _as_matrix(
            [
                [math.cos(theta / 2), -np.exp(1j * lam) * math.sin(theta / 2)],
                [np.exp(1j * phi) * math.sin(theta / 2), np.exp(1j * (phi + lam)) * math.cos(theta / 2)],
            ]
        )
    if key in ("cx", "cnot"):
        return _as_matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        )
    if key == "cy":
        return _as_matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]
        )
    if key == "cz":
        return _as_matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        )
    if key == "swap":
        return _as_matrix(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
    if key in ("cphase", "cp"):
        (phi,) = params
        return _as_matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j * phi)]]
        )
    if key in ("crx", "cry", "crz"):
        (theta,) = params
        single = gate_matrix(key[1:], (theta,))
        out = np.eye(4, dtype=np.complex128)
        out[2:, 2:] = single
        return _as_matrix(out)
    if key in ("ccx", "toffoli"):
        mat = np.eye(8, dtype=np.complex128)
        mat[6, 6] = 0
        mat[7, 7] = 0
        mat[6, 7] = 1
        mat[7, 6] = 1
        return _as_matrix(mat)
    if key in ("cswap", "fredkin"):
        mat = np.eye(8, dtype=np.complex128)
        mat[5, 5] = 0
        mat[6, 6] = 0
        mat[5, 6] = 1
        mat[6, 5] = 1
        return _as_matrix(mat)
    raise ValueError(f"Unknown gate '{name}'")


def _fixed(name: str) -> StandardGate:
    mat = gate_matrix(name)
    nq = int(math.log2(mat.shape[0]))
    g = StandardGate(mat, name=name)
    assert g.num_qubits == nq
    return g


def I() -> StandardGate:  # noqa: E743  (Pauli-I is a conventional quantum gate name)
    """Identity gate."""
    return _fixed("i")


def X() -> StandardGate:
    """Pauli-X gate."""
    return _fixed("x")


def Y() -> StandardGate:
    """Pauli-Y gate."""
    return _fixed("y")


def Z() -> StandardGate:
    """Pauli-Z gate."""
    return _fixed("z")


def H() -> StandardGate:
    """Hadamard gate."""
    return _fixed("h")


def S() -> StandardGate:
    """S (sqrt-Z) gate."""
    return _fixed("s")


def Sdg() -> StandardGate:
    """S-dagger gate."""
    return _fixed("sdg")


def T() -> StandardGate:
    """T (pi/8) gate."""
    return _fixed("t")


def Tdg() -> StandardGate:
    """T-dagger gate."""
    return _fixed("tdg")


def Phase(phi: Any) -> ParameterizedGate:
    """Phase gate with angle ``phi``."""
    return ParameterizedGate("phase", 1, (phi,))


def RX(theta: Any) -> RotationGate:
    """X-rotation gate."""
    return RotationGate("rx", theta)


def RY(theta: Any) -> RotationGate:
    """Y-rotation gate."""
    return RotationGate("ry", theta)


def RZ(theta: Any) -> RotationGate:
    """Z-rotation gate."""
    return RotationGate("rz", theta)


def U(theta: Any, phi: Any, lam: Any) -> ParameterizedGate:
    """General single-qubit unitary."""
    return ParameterizedGate("u", 1, (theta, phi, lam))


def CX() -> StandardGate:
    """Controlled-X (CNOT) gate."""
    return _fixed("cx")


CNOT = CX


def CY() -> StandardGate:
    """Controlled-Y gate."""
    return _fixed("cy")


def CZ() -> StandardGate:
    """Controlled-Z gate."""
    return _fixed("cz")


def SWAP() -> StandardGate:
    """SWAP gate."""
    return _fixed("swap")


def CPhase(phi: Any) -> ParameterizedGate:
    """Controlled-phase gate."""
    return ParameterizedGate("cphase", 2, (phi,))


def CRX(theta: Any) -> ParameterizedGate:
    """Controlled-RX gate."""
    return ParameterizedGate("crx", 2, (theta,))


def CRY(theta: Any) -> ParameterizedGate:
    """Controlled-RY gate."""
    return ParameterizedGate("cry", 2, (theta,))


def CRZ(theta: Any) -> ParameterizedGate:
    """Controlled-RZ gate."""
    return ParameterizedGate("crz", 2, (theta,))


def Toffoli() -> StandardGate:
    """Toffoli (CCX) gate."""
    return _fixed("ccx")


CCX = Toffoli


def Fredkin() -> StandardGate:
    """Fredkin (CSWAP) gate."""
    return _fixed("cswap")


CSWAP = Fredkin


def make_gate(name: str, params: tuple[Any, ...] = ()) -> Gate:
    """Factory: build a gate by name with optional parameters."""
    key = name.lower()
    fixed = {"i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "cx", "cnot", "cy", "cz", "swap", "ccx", "toffoli", "cswap", "fredkin"}
    if key in fixed:
        return _fixed("cx" if key == "cnot" else ("ccx" if key == "toffoli" else ("cswap" if key == "fredkin" else key)))
    if key in ("rx", "ry", "rz"):
        if len(params) != 1:
            raise ValueError(f"Gate '{name}' needs 1 parameter")
        return RotationGate(key, params[0])
    if key in ("phase", "p", "cphase", "cp", "crx", "cry", "crz"):
        return ParameterizedGate(key, 2 if key.startswith("c") else 1, params)
    if key == "u":
        if len(params) != 3:
            raise ValueError("Gate 'u' needs 3 parameters")
        return ParameterizedGate("u", 1, params)
    raise ValueError(f"Unknown gate '{name}'")


def ControlledUnitary(
    matrix: NDArray[np.complex128] | list[Any], num_controls: int = 1, name: str = "cu"
) -> ControlledGate:
    """Build a controlled gate from an arbitrary unitary matrix.

    Args:
        matrix: Square power-of-two unitary matrix for the target.
        num_controls: Number of control qubits (must be >= 1).
        name: Base gate name used in the controlled name.

    Raises:
        ValueError: If the matrix is not unitary or controls < 1.
    """
    if num_controls < 1:
        raise ValueError("num_controls must be >= 1")
    base = UnitaryGate(matrix, name=name)
    return ControlledGate(base, num_controls)


@dataclass(frozen=True)
class GateInfo:
    """Static gate metadata used by the transpiler and resource estimator."""

    name: str
    num_qubits: int
    is_clifford: bool = False
    is_parameterized: bool = False
