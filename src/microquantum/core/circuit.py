"""Quantum circuit builder and state-vector simulator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from .operators import Operator
from .parameter import Parameter, ParameterExpression
from .state import StateVector
from .tensor import expand_operator

if TYPE_CHECKING:
    pass


# Internal tuple types for parameterized gate storage.
# A gate instruction is either a concrete (Operator, targets) pair,
# or a parameterized descriptor tuple.
_ConcreteGate = tuple[Operator, list[int]]
_ParameterizedGate = tuple[str, Union[Parameter, "ParameterExpression"], int]
_GateInstruction = Union[_ConcreteGate, _ParameterizedGate]


def _gate_name(op: Operator) -> str:
    """Get the gate name from an Operator instance."""
    return op.name


class QuantumCircuit:
    """A sequential quantum circuit with named gate instructions.

    Supports method-chaining for fluent circuit construction and
    can compute the full unitary or simulate the circuit on a state.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        gates: Ordered list of gate instructions.
        measurements: Qubit indices marked for measurement (in order).
        depth: Circuit depth (sequential gate layers).
        num_gates: Total number of gate instructions.
        is_parameterized: Whether the circuit contains unbound parameters.
        parameters: Unbound parameters, in deterministic name order.
    """

    def __init__(self, num_qubits: int) -> None:
        """Initialize an empty circuit.

        Args:
            num_qubits: Number of qubits. Must be >= 1.

        Raises:
            ValueError: If num_qubits < 1.
        """
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self._num_qubits = num_qubits
        self._gate_instructions: list[_GateInstruction] = []
        self._measurements: list[int] = []

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._num_qubits

    @property
    def gates(self) -> list[tuple[Operator, list[int]]]:
        """List of (operator, targets) gate instructions.

        Only returns concrete (non-parameterized) gates.
        """
        result: list[tuple[Operator, list[int]]] = []
        for instr in self._gate_instructions:
            if len(instr) == 2 and isinstance(instr[0], Operator):
                result.append((instr[0], instr[1]))
        return result

    @property
    def num_gates(self) -> int:
        """Total number of gate instructions."""
        return len(self._gate_instructions)

    def depth(self) -> int:
        """Circuit depth computed via longest-path scheduling."""
        if not self._gate_instructions:
            return 0

        qubit_finish: dict[int, int] = {}
        for instr in self._gate_instructions:
            targets = self._get_targets(instr)
            layer = max((qubit_finish.get(t, 0) for t in targets), default=0)
            new_layer = layer + 1
            for t in targets:
                qubit_finish[t] = new_layer
        return max(qubit_finish.values())

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Unbound parameters of the circuit, in deterministic name order.

        Parameters are identified by name; the same-name parameter used in
        several gates appears exactly once.  The returned tuple is
        read-only from the caller's perspective and sorted alphabetically
        by name.
        """
        seen: dict[str, Parameter] = {}
        for instr in self._gate_instructions:
            if not self._is_parameterized_gate(instr):
                continue
            param = instr[1]
            symbol = (
                param.parameter if isinstance(param, ParameterExpression) else param
            )
            if symbol.name not in seen:
                seen[symbol.name] = symbol
        return tuple(sorted(seen.values(), key=lambda p: p.name))

    @property
    def is_parameterized(self) -> bool:
        """Whether the circuit contains unbound parameters."""
        return len(self.parameters) > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_parameterized_gate(instr: _GateInstruction) -> bool:
        """Check if a gate instruction is parameterized."""
        return len(instr) == 3 and isinstance(instr[0], str)

    @staticmethod
    def _get_targets(instr: _GateInstruction) -> list[int]:
        """Extract target qubits from any gate instruction."""
        if QuantumCircuit._is_parameterized_gate(instr):
            target = instr[2]
            return [int(target)]
        concrete = instr
        return list(concrete[1])

    def _ensure_bound(self) -> None:
        """Raise ValueError if circuit has unbound parameters."""
        if self.is_parameterized:
            params = self.parameters
            names = ", ".join(p.name for p in sorted(params, key=lambda p: p.name))
            raise ValueError(
                f"Circuit has unbound parameters: [{names}]. "
                f"Call bind_parameters() before executing."
            )

    def _resolve_angle(
        self,
        theta: Union[float, int, complex, Parameter],
    ) -> float:
        """Resolve a rotation angle to a float.

        If theta is numeric, return it directly.
        If theta is a Parameter, look it up in bound values.
        """
        if isinstance(theta, (int, float, complex)):
            return float(theta)  # type: ignore[arg-type]
        if isinstance(theta, Parameter):
            raise ValueError(
                f"Parameter '{theta.name}' is not bound. "
                f"Call bind_parameters() first."
            )
        raise TypeError(f"Expected numeric angle or Parameter, got {type(theta)}")

    # ------------------------------------------------------------------
    # Gate application methods (mutating, return self for chaining)
    # ------------------------------------------------------------------

    def _validate_targets(self, targets: list[int]) -> None:
        for t in targets:
            if t < 0 or t >= self._num_qubits:
                raise ValueError(
                    f"Qubit index {t} out of range for "
                    f"{self._num_qubits}-qubit circuit "
                    f"(valid: 0..{self._num_qubits - 1})"
                )

    def append(self, op: Operator, targets: list[int]) -> QuantumCircuit:
        """Append a generic gate instruction.

        Args:
            op: The operator to apply.
            targets: Target qubit indices.

        Returns:
            self, for method chaining.
        """
        self._validate_targets(targets)
        if op.num_qubits != len(targets):
            raise ValueError(
                f"Operator acts on {op.num_qubits} qubit(s) but "
                f"{len(targets)} target(s) given"
            )
        self._gate_instructions.append((op, list(targets)))
        return self

    def append_parameterized(
        self,
        gate_type: str,
        param: Union[float, int, complex, Parameter, ParameterExpression],
        target: int,
    ) -> QuantumCircuit:
        """Append a parameterized rotation gate.

        Args:
            gate_type: One of "rx", "ry", "rz".
            param: Rotation angle (numeric, Parameter or
                ParameterExpression).
            target: Target qubit index.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If gate_type is not a supported rotation gate.
            TypeError: If param is not numeric, a Parameter or a
                ParameterExpression.
        """
        self._validate_targets([target])
        if gate_type not in ("rx", "ry", "rz"):
            raise ValueError(
                f"parameterized gate type '{gate_type}' is not supported; "
                f"only 'rx', 'ry', 'rz' accept symbolic angles"
            )
        if not isinstance(
            param, (Parameter, ParameterExpression, int, float, complex)
        ):
            raise TypeError(
                f"param must be a numeric angle, Parameter or "
                f"ParameterExpression, got {type(param).__name__}"
            )
        if isinstance(param, (int, float, complex)):
            # Immediate numeric evaluation — store concrete operator
            gate_map = {"rx": Operator.Rx, "ry": Operator.Ry, "rz": Operator.Rz}
            op = gate_map[gate_type](float(param))  # type: ignore[arg-type]
            self._gate_instructions.append((op, [target]))
        else:
            self._gate_instructions.append((gate_type, param, target))
        return self

    def h(self, q: int) -> QuantumCircuit:
        """Apply Hadamard gate to qubit q."""
        return self.append(Operator.H(), [q])

    def x(self, q: int) -> QuantumCircuit:
        """Apply Pauli-X gate to qubit q."""
        return self.append(Operator.X(), [q])

    def y(self, q: int) -> QuantumCircuit:
        """Apply Pauli-Y gate to qubit q."""
        return self.append(Operator.Y(), [q])

    def z(self, q: int) -> QuantumCircuit:
        """Apply Pauli-Z gate to qubit q."""
        return self.append(Operator.Z(), [q])

    def s(self, q: int) -> QuantumCircuit:
        """Apply S (phase) gate to qubit q."""
        return self.append(Operator.S(), [q])

    def sdg(self, q: int) -> QuantumCircuit:
        """Apply S-dagger (S†) gate to qubit q."""
        return self.append(Operator.Sdg(), [q])

    def t(self, q: int) -> QuantumCircuit:
        """Apply T (pi/8) gate to qubit q."""
        return self.append(Operator.T(), [q])

    def tdg(self, q: int) -> QuantumCircuit:
        """Apply T-dagger (T†) gate to qubit q."""
        return self.append(Operator.Tdg(), [q])

    def rx(
        self, theta: Union[float, int, complex, Parameter, ParameterExpression], q: int
    ) -> QuantumCircuit:
        """Apply Rx(theta) rotation to qubit q.

        Args:
            theta: Rotation angle in radians, a Parameter or a
                ParameterExpression.
            q: Target qubit index.
        """
        return self.append_parameterized("rx", theta, q)

    def ry(
        self, theta: Union[float, int, complex, Parameter, ParameterExpression], q: int
    ) -> QuantumCircuit:
        """Apply Ry(theta) rotation to qubit q.

        Args:
            theta: Rotation angle in radians, a Parameter or a
                ParameterExpression.
            q: Target qubit index.
        """
        return self.append_parameterized("ry", theta, q)

    def rz(
        self, theta: Union[float, int, complex, Parameter, ParameterExpression], q: int
    ) -> QuantumCircuit:
        """Apply Rz(theta) rotation to qubit q.

        Args:
            theta: Rotation angle in radians, or a (expression of a) Parameter.
            q: Target qubit index.
        """
        return self.append_parameterized("rz", theta, q)

    def cx(self, control: int, target: int) -> QuantumCircuit:
        """Apply CNOT (CX) gate."""
        return self.append(Operator.CNOT(), [control, target])

    def cnot(self, control: int, target: int) -> QuantumCircuit:
        """Apply CNOT gate (alias for cx)."""
        return self.cx(control, target)

    def cz(self, control: int, target: int) -> QuantumCircuit:
        """Apply Controlled-Z gate."""
        return self.append(Operator.CZ(), [control, target])

    def swap(self, q1: int, q2: int) -> QuantumCircuit:
        """Apply SWAP gate."""
        return self.append(Operator.SWAP(), [q1, q2])

    # ------------------------------------------------------------------
    # Circuit properties
    # ------------------------------------------------------------------

    @property
    def qubits(self) -> list[int]:
        """List of qubit indices in this circuit."""
        return list(range(self._num_qubits))

    def gate_count(self, gate_type: str | None = None) -> int:
        """Count gates, optionally filtered by type.

        Args:
            gate_type: Gate type name to filter by (e.g., "rx", "h", "cnot").
                       If None, counts all gates.

        Returns:
            Number of gates matching the filter.
        """
        if gate_type is None:
            return self.num_gates
        gt = gate_type.lower()
        count = 0
        for instr in self._gate_instructions:
            if self._is_parameterized_gate(instr):
                if instr[0] == gt:
                    count += 1
            else:
                concrete_op = instr[0]
                if _gate_name(concrete_op) == gt:
                    count += 1
        return count

    def contains_gate(self, gate_type: str) -> bool:
        """Check if circuit contains a gate of the given type.

        Args:
            gate_type: Gate type name (e.g., "rx", "h", "cnot").

        Returns:
            True if at least one gate of that type exists.
        """
        return self.gate_count(gate_type) > 0

    def inverse(self) -> QuantumCircuit:
        """Return the inverse circuit (gates in reverse order, each inverted).

        For unitary gates, the inverse is U^dagger.
        For parameterized rotation gates, the inverse is rotation by -theta.

        Returns:
            New QuantumCircuit such that circuit + circuit.inverse() = identity.
        """
        result = QuantumCircuit(self._num_qubits)
        for instr in reversed(self._gate_instructions):
            if self._is_parameterized_gate(instr):
                gate_type = instr[0]
                param = instr[1]
                target = instr[2]
                from .parameter import Parameter, ParameterExpression
                if isinstance(param, ParameterExpression):
                    angle = param.evaluate({})
                elif isinstance(param, Parameter):
                    raise ValueError(
                        f"Cannot invert circuit with unbound parameter '{param.name}'. "
                        f"Call bind_parameters() first."
                    )
                else:
                    angle = -float(param)
                gate_map = {"rx": Operator.Rx, "ry": Operator.Ry, "rz": Operator.Rz}
                op = gate_map[gate_type](angle)
                result._gate_instructions.append((op, [target]))
            else:
                op = instr[0]
                targets = instr[1]
                result._gate_instructions.append((op.inverse(), list(targets)))
        return result

    # ------------------------------------------------------------------
    # Parameter binding
    # ------------------------------------------------------------------

    def bind_parameters(
        self,
        param_map: Mapping[Union[str, Parameter], Union[int, float, complex]],
    ) -> QuantumCircuit:
        """Return a new circuit with parameters bound to numeric values.

        Every parameter whose name appears in ``param_map`` is replaced
        by the corresponding numeric value; all other parameters remain
        symbolic in the returned circuit.  The original circuit is never
        modified.

        Validation is strict (MQ-12):

        * Unknown parameters that match no circuit parameter raise
          ``ValueError``.
        * Non-numeric values raise ``TypeError``.
        * Complex binding values with a non-zero imaginary part raise
          ``ValueError``.
        * Ambiguous definitions (the same logical parameter specified
          twice) raise ``ValueError``.

        Args:
            param_map: Mapping from ``Parameter`` (or parameter name) to
                a numeric value.

        Returns:
            A new :class:`QuantumCircuit` with matched parameters resolved
            and any unbound parameters kept symbolic.

        Raises:
            TypeError: If ``param_map`` is not a mapping, a key is not a
                ``Parameter``/``str``, or a value is not numeric.
            ValueError: If the mapping references unknown parameters, uses
                a complex value, or binds the same parameter twice.
        """
        bindings = self._validate_bindings(param_map)
        gate_map = {"rx": Operator.Rx, "ry": Operator.Ry, "rz": Operator.Rz}
        resolved = QuantumCircuit(self._num_qubits)
        for instr in self._gate_instructions:
            if not self._is_parameterized_gate(instr):
                resolved._gate_instructions.append(instr)
                continue
            gate_type, param, target = instr
            name = (
                param.parameter.name
                if isinstance(param, ParameterExpression)
                else param.name
            )
            if name in bindings:
                if isinstance(param, ParameterExpression):
                    angle = param.evaluate(bindings)
                else:
                    angle = bindings[name]
                op = gate_map[gate_type](angle)
                resolved._gate_instructions.append((op, [target]))
            else:
                resolved._gate_instructions.append(instr)
        resolved._measurements = list(self._measurements)
        return resolved

    def _validate_bindings(
        self,
        param_map: object,
    ) -> dict[str, float]:
        """Validate a binding mapping and return canonical ``{name: float}``.

        Enforces the MQ-12 strict binding contract: mapping type, key
        types, numeric-only values, no complex angles, no duplicate
        definitions and no unknown parameters.
        """
        if not isinstance(param_map, Mapping):
            raise TypeError(
                f"bind_parameters expects a Mapping of parameters to "
                f"values, got {type(param_map).__name__}"
            )
        known: set[str] = {p.name for p in self.parameters}
        bindings: dict[str, float] = {}
        for key, value in param_map.items():
            if isinstance(key, Parameter):
                name = key.name
            elif isinstance(key, str):
                name = key
            else:
                raise TypeError(
                    f"binding keys must be Parameter instances or "
                    f"parameter names, got {type(key).__name__}"
                )
            if name in bindings:
                raise ValueError(
                    f"parameter '{name}' is bound more than once; "
                    f"provide each parameter exactly once"
                )
            if not _is_numeric_binding(value):
                raise TypeError(
                    f"value for parameter '{name}' must be numeric, "
                    f"got {type(value).__name__}"
                )
            bindings[name] = _coerce_binding_value(value, name)
        unknown = sorted(n for n in bindings if n not in known)
        if unknown:
            available = sorted(known)
            raise ValueError(
                f"bind_parameters received unknown parameter(s): {unknown}; "
                f"circuit parameters are: {available}"
            )
        return bindings

    # ------------------------------------------------------------------
    # Simulation & unitary
    # ------------------------------------------------------------------

    def _resolve_gates(self) -> list[tuple[Operator, list[int]]]:
        """Resolve all gates to concrete operators (requires binding)."""
        self._ensure_bound()
        result: list[tuple[Operator, list[int]]] = []
        for instr in self._gate_instructions:
            if self._is_parameterized_gate(instr):
                raise ValueError(
                    "Circuit has unbound parameters. Call bind_parameters() first."
                )
            result.append((instr[0], instr[1]))  # type: ignore[misc]
        return result

    def get_unitary(self) -> Operator:
        """Compute the full 2^N x 2^N circuit unitary matrix.

        Returns:
            Operator representing U = U_m ... U_2 U_1.

        Raises:
            ValueError: If circuit contains unbound parameters.
        """
        self._ensure_bound()
        n = self._num_qubits
        total_dim = 2**n
        result = np.eye(total_dim, dtype=np.complex128)

        for instr in self._gate_instructions:
            if self._is_parameterized_gate(instr):
                raise ValueError(
                    "Circuit has unbound parameters. Call bind_parameters() first."
                )
            op = instr[0]
            targets = instr[1]
            expanded = expand_operator(op, targets, n)
            result = expanded.matrix @ result

        return Operator(result)

    def run(
        self, initial_state: Optional[StateVector] = None
    ) -> StateVector:
        """Execute the circuit on a state vector.

        Uses efficient tensor contraction (einsum) via engine.apply_gate()
        instead of constructing full 2^N matrices.

        Args:
            initial_state: Starting state. Defaults to |0...0>.

        Returns:
            Final state after all gates are applied.

        Raises:
            ValueError: If circuit contains unbound parameters.
        """
        from .engine import apply_gate

        self._ensure_bound()
        if initial_state is None:
            state = StateVector(self._num_qubits)
        else:
            if initial_state.num_qubits != self._num_qubits:
                raise ValueError(
                    f"State has {initial_state.num_qubits} qubits "
                    f"but circuit has {self._num_qubits}"
                )
            state = initial_state.copy()

        for instr in self._gate_instructions:
            if self._is_parameterized_gate(instr):
                raise ValueError(
                    "Circuit has unbound parameters. Call bind_parameters() first."
                )
            op = instr[0]
            targets = instr[1]
            state = apply_gate(state, op.matrix, targets)

        return state

    # ------------------------------------------------------------------
    # Measurement & expectation values
    # ------------------------------------------------------------------

    def measure(self, qubit: int) -> QuantumCircuit:
        """Mark a qubit for measurement in the computational basis.

        Measurements are represented explicitly on the circuit: they do
        not alter the gates or execute anything.  The backend samples the
        measured qubits during execution.

        Args:
            qubit: Qubit index to measure.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If the qubit index is out of range or already
                measured.
        """
        self._validate_targets([qubit])
        if qubit in self._measurements:
            raise ValueError(
                f"Qubit {qubit} is already marked for measurement"
            )
        self._measurements.append(qubit)
        return self

    def measure_all(self) -> QuantumCircuit:
        """Mark every qubit for measurement in the computational basis.

        Unlike earlier revisions, this method only records the measurement
        instructions on the circuit; it does not execute anything.  Pass the
        circuit to a backend to run and sample:

        .. code-block:: python

           qc = QuantumCircuit(2).h(0).cx(0, 1).measure_all()
           result = StatevectorBackend().run(qc, shots=1000, seed=42)
           print(result.counts())   # {'00': ~500, '11': ~500}

        Returns:
            self, for method chaining.
        """
        for qubit in range(self._num_qubits):
            if qubit not in self._measurements:
                self._measurements.append(qubit)
        return self

    @property
    def measurements(self) -> list[int]:
        """Qubit indices marked for measurement (in `.measure()` order).

        The list is ordered by insertion; measurement output bitstrings
        follow this same order (left to right) for deterministic, documented
        classical output ordering.
        """
        return list(self._measurements)

    def expectation_value(
        self,
        observable: Operator,
        targets: Optional[list[int]] = None,
    ) -> float:
        """Execute the circuit and compute observable expectation value.

        Args:
            observable: Hermitian observable operator.
            targets: Optional qubit indices the observable acts on.

        Returns:
            The real part of <psi|H|psi>.

        Raises:
            ValueError: If circuit contains unbound parameters.
        """
        from .measurement import expectation_value as _expectation_value

        self._ensure_bound()
        state = self.run()
        return _expectation_value(state, observable, targets=targets)

    # ------------------------------------------------------------------
    # I/O & display
    # ------------------------------------------------------------------

    def to_ir(self, include_terminal_measurements: bool = False) -> "object":
        """Convert this circuit into MicroQuantum IR.

        Args:
            include_terminal_measurements: If True, append a Measurement
                node per qubit so the IR describes a complete sampling
                program.

        Returns:
            An :class:`~microquantum.ir.IRCircuit` representing the gates.
        """
        from ..ir import to_ir as _ir_to_ir
        return _ir_to_ir(self, include_terminal_measurements=include_terminal_measurements)

    @staticmethod
    def from_ir(ir: "object") -> "QuantumCircuit":
        """Rebuild a QuantumCircuit from gate-level MicroQuantum IR.

        Args:
            ir: An IRCircuit produced by :meth:`to_ir` or a compiler.

        Returns:
            A QuantumCircuit executing the IR's gates.
        """
        from ..ir import from_ir as _ir_from_ir
        return _ir_from_ir(ir)

    def qasm(self, header: bool = True) -> str:
        """Export circuit to OpenQASM 2.0 format.

        Args:
            header: Whether to include the OPENQASM header and qreg/creg.

        Returns:
            OpenQASM 2.0 string.
        """
        from .qasm import to_qasm
        return to_qasm(self, header=header)

    @staticmethod
    def from_qasm(qasm_str: str) -> QuantumCircuit:
        """Import a circuit from OpenQASM 2.0 format.

        Args:
            qasm_str: OpenQASM 2.0 string.

        Returns:
            QuantumCircuit constructed from the QASM.
        """
        from .qasm import from_qasm
        return from_qasm(qasm_str)

    def draw(self, title: Optional[str] = None) -> str:
        """Generate an ASCII circuit diagram.

        Args:
            title: Optional title for the diagram.

        Returns:
            Multi-line string with the ASCII circuit.
        """
        from .visualization import draw
        return draw(self, title=title)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize circuit to a JSON string.

        Args:
            indent: JSON indentation level (None for compact).

        Returns:
            JSON string.
        """
        from .serialization import to_json
        return to_json(self, indent=indent)

    @staticmethod
    def from_json(json_str: str) -> QuantumCircuit:
        """Deserialize a circuit from a JSON string.

        Args:
            json_str: JSON string from to_json().

        Returns:
            Reconstructed QuantumCircuit.
        """
        from .serialization import from_json
        return from_json(json_str)

    def save(self, path: Union[str, "Path"]) -> None:
        """Save circuit to a JSON file.

        Args:
            path: File path to write to.
        """
        from .serialization import save
        save(self, path)

    @staticmethod
    def load(path: Union[str, "Path"]) -> QuantumCircuit:
        """Load a circuit from a JSON file.

        Args:
            path: File path to read from.

        Returns:
            Loaded QuantumCircuit.
        """
        from .serialization import load
        return load(path)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __add__(self, other: QuantumCircuit) -> QuantumCircuit:
        """Concatenate two circuits (sequentially) with matching qubit counts."""
        if not isinstance(other, QuantumCircuit):
            return NotImplemented
        if self._num_qubits != other._num_qubits:
            raise ValueError(
                f"Cannot concatenate {self._num_qubits}-qubit circuit "
                f"with {other._num_qubits}-qubit circuit"
            )
        result = QuantumCircuit(self._num_qubits)
        result._gate_instructions = (
            list(self._gate_instructions) + list(other._gate_instructions)
        )
        result._measurements = list(self._measurements) + list(other._measurements)
        return result

    def __repr__(self) -> str:
        param_info = ""
        if self.is_parameterized:
            param_names = ", ".join(
                p.name for p in sorted(self.parameters, key=lambda p: p.name)
            )
            param_info = f", parameters=[{param_names}]"
        measure_info = ""
        if self._measurements:
            measure_info = f", measurements={self._measurements}"
        return (
            f"QuantumCircuit(num_qubits={self._num_qubits}, "
            f"num_gates={self.num_gates}, depth={self.depth()}"
            f"{param_info}{measure_info})"
        )

    def __str__(self) -> str:
        param_info = ""
        if self.is_parameterized:
            param_names = ", ".join(
                p.name for p in sorted(self.parameters, key=lambda p: p.name)
            )
            param_info = f", params=[{param_names}]"
        measure_info = ""
        if self._measurements:
            measure_info = f", measurements={self._measurements}"
        lines = [
            f"QuantumCircuit ({self._num_qubits} qubits, "
            f"{self.num_gates} gates, depth {self.depth()}{param_info}"
            f"{measure_info})"
        ]
        for instr in self._gate_instructions:
            if self._is_parameterized_gate(instr):
                gate_type = instr[0]
                param = instr[1]
                target = instr[2]
                name = gate_type.upper()
                param_str = str(param)
                lines.append(f"  {name}({param_str}, {target})")
            else:
                op = instr[0]
                targets = instr[1]
                gate_names = {
                    Operator.H: "H", Operator.X: "X", Operator.Y: "Y",
                    Operator.Z: "Z", Operator.S: "S", Operator.Sdg: "S†",
                    Operator.T: "T", Operator.Tdg: "T†",
                    Operator.CNOT: "CNOT", Operator.CZ: "CZ",
                    Operator.SWAP: "SWAP",
                }
                name = gate_names.get(type(op), "U")
                target_str = ",".join(str(t) for t in targets)
                lines.append(f"  {name}({target_str})")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Binding validation helpers (module-level, used by QuantumCircuit)
# ------------------------------------------------------------------


def _is_numeric_binding(value: object) -> bool:
    """Return True if *value* is a numeric type suitable for a binding."""
    if isinstance(value, bool):
        return False
    return isinstance(
        value,
        (int, float, complex, np.integer, np.floating, np.complexfloating),
    )


def _coerce_binding_value(value: object, name: str) -> float:
    """Coerce a numeric value to a float, rejecting non-real angles.

    Args:
        value: The raw value supplied by the caller.
        name: Parameter name (used in error messages).

    Returns:
        Real part of the numeric value.

    Raises:
        ValueError: If the value has a non-zero imaginary part.
    """
    number: complex = complex(value)  # type: ignore[arg-type, call-overload]
    if abs(number.imag) > 1e-12:
        raise ValueError(
            f"value for parameter '{name}' must be real, "
            f"got {value!r} (imaginary part {number.imag})"
        )
    return number.real
