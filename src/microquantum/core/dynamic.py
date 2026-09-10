"""Dynamic circuit support with mid-circuit measurements and classical control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .circuit import QuantumCircuit
from .engine import apply_gate
from .operators import Operator
from .parameter import Parameter
from .state import StateVector


class ClassicalRegister:
    """Simple register for classical bits.

    Attributes:
        size: Number of classical bits.
        bits: Current bit values.
    """

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        self._bits: list[int] = [0] * size

    @property
    def bits(self) -> list[int]:
        """Current bit values."""
        return list(self._bits)

    def read(self, bit_index: int) -> int:
        """Read the value of a classical bit.

        Args:
            bit_index: Index of the classical bit.

        Returns:
            The bit value (0 or 1).

        Raises:
            IndexError: If bit_index is out of range.
        """
        if bit_index < 0 or bit_index >= len(self._bits):
            raise IndexError(
                f"Classical bit index {bit_index} out of range "
                f"(valid: 0..{len(self._bits) - 1})"
            )
        return self._bits[bit_index]

    def write(self, bit_index: int, value: int) -> None:
        """Write a value to a classical bit.

        Args:
            bit_index: Index of the classical bit.
            value: Value to write (0 or 1).

        Raises:
            IndexError: If bit_index is out of range.
            ValueError: If value is not 0 or 1.
        """
        if bit_index < 0 or bit_index >= len(self._bits):
            raise IndexError(
                f"Classical bit index {bit_index} out of range "
                f"(valid: 0..{len(self._bits) - 1})"
            )
        if value not in (0, 1):
            raise ValueError(f"Classical bit value must be 0 or 1, got {value}")
        self._bits[bit_index] = value

    def __len__(self) -> int:
        return len(self._bits)

    def __repr__(self) -> str:
        return f"ClassicalRegister(size={len(self._bits)}, bits={self._bits})"


@dataclass
class DynamicCircuitResult:
    """Result of simulating a dynamic circuit.

    Attributes:
        final_state: Final quantum state vector.
        classical_memory: Final classical bit values keyed by bit index.
        measurement_results: List of (qubit, classical_bit, value) for each
            mid-circuit measurement.
        intermediate_measurements: Detailed dicts of each mid-circuit
            measurement with keys 'qubit', 'classical_bit', 'value', and
            'step'.
    """

    final_state: StateVector
    classical_memory: dict[int, int]
    measurement_results: list[tuple[int, int, int]]
    intermediate_measurements: list[dict]


def _measure_single_qubit(
    state: StateVector,
    qubit: int,
    rng: np.random.Generator,
) -> tuple[int, StateVector]:
    """Perform a projective measurement on a single qubit, collapsing the state.

    Args:
        state: The current quantum state.
        qubit: Qubit index to measure.
        rng: Numpy random generator.

    Returns:
        Tuple of (measurement_outcome, collapsed_state).

    Raises:
        ValueError: If qubit index is out of range.
    """
    n = state.num_qubits
    if qubit < 0 or qubit >= n:
        raise ValueError(
            f"Qubit index {qubit} out of range for "
            f"{n}-qubit state (valid: 0..{n - 1})"
        )

    amps = state.amplitudes
    dim = 2**n
    mask = 1 << (n - 1 - qubit)

    indices = np.arange(dim)
    is_zero = (indices & mask) == 0

    prob_0 = float(np.sum(np.abs(amps[is_zero]) ** 2))

    outcome = 0 if rng.random() < prob_0 else 1

    new_amps = amps.copy()
    if outcome == 0:
        new_amps[~is_zero] = 0.0
    else:
        new_amps[is_zero] = 0.0

    norm = np.linalg.norm(new_amps)
    if norm > 0:
        new_amps = new_amps / norm

    return outcome, StateVector(num_qubits=n, amplitudes=new_amps.astype(np.complex128))


class DynamicCircuit:
    """A quantum circuit with mid-circuit measurements and classical control.

    Extends standard quantum circuits with the ability to perform measurements
    mid-circuit and use classical bits to conditionally apply gates.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        num_classical_bits: Number of classical bits available.
        num_gates: Total number of quantum gate instructions.
        depth: Circuit depth (longest path through gate layers).
    """

    def __init__(self, num_qubits: int, num_classical_bits: int = 0) -> None:
        """Initialize a dynamic circuit.

        Args:
            num_qubits: Number of qubits. Must be >= 1.
            num_classical_bits: Number of classical bits. Must be >= 0.

        Raises:
            ValueError: If num_qubits < 1 or num_classical_bits < 0.
        """
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        if num_classical_bits < 0:
            raise ValueError(
                f"num_classical_bits must be >= 0, got {num_classical_bits}"
            )
        self._num_qubits = num_qubits
        self._num_classical_bits = num_classical_bits
        self._ops: list[tuple] = []

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._num_qubits

    @property
    def num_classical_bits(self) -> int:
        """Number of classical bits."""
        return self._num_classical_bits

    @property
    def num_gates(self) -> int:
        """Total number of quantum gate instructions."""
        return sum(1 for op in self._ops if op[0] == "gate")

    @property
    def depth(self) -> int:
        """Circuit depth computed via longest-path scheduling over gates."""
        qubit_finish: dict[int, int] = {}
        for op in self._ops:
            if op[0] == "gate":
                targets: list[int] = op[2]
                layer = max(
                    (qubit_finish.get(t, 0) for t in targets), default=0
                )
                new_layer = layer + 1
                for t in targets:
                    qubit_finish[t] = new_layer
        return max(qubit_finish.values()) if qubit_finish else 0

    # ------------------------------------------------------------------
    # Gate application methods (delegated to Operator library)
    # ------------------------------------------------------------------

    def _add_gate(self, op: Operator, targets: list[int]) -> DynamicCircuit:
        """Append a gate instruction."""
        if op.num_qubits != len(targets):
            raise ValueError(
                f"Operator acts on {op.num_qubits} qubit(s) but "
                f"{len(targets)} target(s) given"
            )
        for t in targets:
            if t < 0 or t >= self._num_qubits:
                raise ValueError(
                    f"Qubit index {t} out of range for "
                    f"{self._num_qubits}-qubit circuit "
                    f"(valid: 0..{self._num_qubits - 1})"
                )
        self._ops.append(("gate", op, list(targets)))
        return self

    def append(self, op: Operator, targets: list[int]) -> DynamicCircuit:
        """Append a generic gate instruction.

        Args:
            op: The operator to apply.
            targets: Target qubit indices.

        Returns:
            self, for method chaining.
        """
        return self._add_gate(op, targets)

    def h(self, q: int) -> DynamicCircuit:
        """Apply Hadamard gate to qubit q."""
        return self._add_gate(Operator.H(), [q])

    def x(self, q: int) -> DynamicCircuit:
        """Apply Pauli-X gate to qubit q."""
        return self._add_gate(Operator.X(), [q])

    def y(self, q: int) -> DynamicCircuit:
        """Apply Pauli-Y gate to qubit q."""
        return self._add_gate(Operator.Y(), [q])

    def z(self, q: int) -> DynamicCircuit:
        """Apply Pauli-Z gate to qubit q."""
        return self._add_gate(Operator.Z(), [q])

    def s(self, q: int) -> DynamicCircuit:
        """Apply S (phase) gate to qubit q."""
        return self._add_gate(Operator.S(), [q])

    def t(self, q: int) -> DynamicCircuit:
        """Apply T (pi/8) gate to qubit q."""
        return self._add_gate(Operator.T(), [q])

    def rx(
        self, theta: Union[float, int, complex, Parameter], q: int
    ) -> DynamicCircuit:
        """Apply Rx(theta) rotation to qubit q.

        Args:
            theta: Rotation angle in radians.
            q: Target qubit index.
        """
        if isinstance(theta, Parameter):
            raise ValueError(
                f"Parameter '{theta.name}' not supported in dynamic circuits. "
                f"Use a numeric angle."
            )
        return self._add_gate(Operator.Rx(float(theta)), [q])  # type: ignore[arg-type]

    def ry(
        self, theta: Union[float, int, complex, Parameter], q: int
    ) -> DynamicCircuit:
        """Apply Ry(theta) rotation to qubit q.

        Args:
            theta: Rotation angle in radians.
            q: Target qubit index.
        """
        if isinstance(theta, Parameter):
            raise ValueError(
                f"Parameter '{theta.name}' not supported in dynamic circuits. "
                f"Use a numeric angle."
            )
        return self._add_gate(Operator.Ry(float(theta)), [q])  # type: ignore[arg-type]

    def rz(
        self, theta: Union[float, int, complex, Parameter], q: int
    ) -> DynamicCircuit:
        """Apply Rz(theta) rotation to qubit q.

        Args:
            theta: Rotation angle in radians.
            q: Target qubit index.
        """
        if isinstance(theta, Parameter):
            raise ValueError(
                f"Parameter '{theta.name}' not supported in dynamic circuits. "
                f"Use a numeric angle."
            )
        return self._add_gate(Operator.Rz(float(theta)), [q])  # type: ignore[arg-type]

    def cx(self, control: int, target: int) -> DynamicCircuit:
        """Apply CNOT (CX) gate."""
        return self._add_gate(Operator.CNOT(), [control, target])

    def cnot(self, control: int, target: int) -> DynamicCircuit:
        """Apply CNOT gate (alias for cx)."""
        return self.cx(control, target)

    def cz(self, control: int, target: int) -> DynamicCircuit:
        """Apply Controlled-Z gate."""
        return self._add_gate(Operator.CZ(), [control, target])

    def swap(self, q1: int, q2: int) -> DynamicCircuit:
        """Apply SWAP gate."""
        return self._add_gate(Operator.SWAP(), [q1, q2])

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self, qubit: int, classical_bit: int) -> DynamicCircuit:
        """Record a mid-circuit measurement of a qubit into a classical bit.

        Args:
            qubit: Qubit index to measure.
            classical_bit: Classical bit index to store the result.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If qubit or classical_bit index is out of range.
        """
        if qubit < 0 or qubit >= self._num_qubits:
            raise ValueError(
                f"Qubit index {qubit} out of range for "
                f"{self._num_qubits}-qubit circuit "
                f"(valid: 0..{self._num_qubits - 1})"
            )
        if classical_bit < 0 or classical_bit >= self._num_classical_bits:
            raise ValueError(
                f"Classical bit index {classical_bit} out of range for "
                f"{self._num_classical_bits}-bit register "
                f"(valid: 0..{self._num_classical_bits - 1})"
            )
        self._ops.append(("measure", qubit, classical_bit))
        return self

    def measure_all(self) -> DynamicCircuit:
        """Measure all qubits to classical bits.

        Qubit i maps to classical bit i % num_classical_bits.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If num_classical_bits is 0.
        """
        if self._num_classical_bits == 0:
            raise ValueError(
                "Cannot measure_all with 0 classical bits. "
                "Set num_classical_bits >= 1."
            )
        for q in range(self._num_qubits):
            cb = q % self._num_classical_bits
            self._ops.append(("measure", q, cb))
        return self

    # ------------------------------------------------------------------
    # Classical control
    # ------------------------------------------------------------------

    def classical_if(
        self,
        classical_bit: int,
        gate_fn: Callable[[DynamicCircuit], DynamicCircuit],
    ) -> DynamicCircuit:
        """Add a classically conditioned block.

        The gate_fn receives a temporary DynamicCircuit and should add gates
        to it. During simulation, these gates are applied only if the
        specified classical bit is 1.

        Args:
            classical_bit: Classical bit index to check.
            gate_fn: Callable that adds gates to a DynamicCircuit.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If classical_bit index is out of range.
        """
        if classical_bit < 0 or classical_bit >= self._num_classical_bits:
            raise ValueError(
                f"Classical bit index {classical_bit} out of range for "
                f"{self._num_classical_bits}-bit register "
                f"(valid: 0..{self._num_classical_bits - 1})"
            )
        self._ops.append(("classical_if", classical_bit, gate_fn))
        return self

    def c_if(
        self,
        classical_bit: int,
        gate_fn: Callable[[DynamicCircuit], DynamicCircuit],
    ) -> DynamicCircuit:
        """Alias for classical_if.

        Args:
            classical_bit: Classical bit index to check.
            gate_fn: Callable that adds gates to a DynamicCircuit.

        Returns:
            self, for method chaining.
        """
        return self.classical_if(classical_bit, gate_fn)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(
        self,
        initial_state: Optional[StateVector] = None,
        seed: Optional[int] = None,
    ) -> DynamicCircuitResult:
        """Simulate the dynamic circuit with mid-circuit measurements and
        classical feedforward.

        Processes instructions sequentially. Mid-circuit measurements
        collapse the state via projective measurement. Classically conditioned
        gates are applied only when their controlling classical bit is 1.

        Args:
            initial_state: Starting state. Defaults to |0...0>.
            seed: Optional RNG seed for reproducibility.

        Returns:
            DynamicCircuitResult with final state, classical memory, and
            measurement records.

        Raises:
            ValueError: If initial_state has wrong number of qubits.
        """
        rng = np.random.default_rng(seed)

        if initial_state is None:
            state = StateVector(self._num_qubits)
        else:
            if initial_state.num_qubits != self._num_qubits:
                raise ValueError(
                    f"State has {initial_state.num_qubits} qubits "
                    f"but circuit has {self._num_qubits}"
                )
            state = initial_state.copy()

        classical_reg = ClassicalRegister(self._num_classical_bits)
        measurement_results: list[tuple[int, int, int]] = []
        intermediate_measurements: list[dict] = []
        step = 0

        for op in self._ops:
            if op[0] == "gate":
                _, gate_op, targets = op
                state = apply_gate(state, gate_op.matrix, targets)
                step += 1

            elif op[0] == "measure":
                _, qubit, classical_bit = op
                outcome, state = _measure_single_qubit(state, qubit, rng)
                classical_reg.write(classical_bit, outcome)
                measurement_results.append((qubit, classical_bit, outcome))
                intermediate_measurements.append(
                    {
                        "step": step,
                        "qubit": qubit,
                        "classical_bit": classical_bit,
                        "value": outcome,
                    }
                )
                step += 1

            elif op[0] == "classical_if":
                _, classical_bit, gate_fn = op
                if classical_reg.read(classical_bit) == 1:
                    temp_dc = DynamicCircuit(
                        self._num_qubits, self._num_classical_bits
                    )
                    gate_fn(temp_dc)
                    for sub_op in temp_dc._ops:
                        if sub_op[0] == "gate":
                            state = apply_gate(
                                state, sub_op[1].matrix, sub_op[2]
                            )
                step += 1

        classical_memory = {
            i: classical_reg.read(i) for i in range(self._num_classical_bits)
        }

        return DynamicCircuitResult(
            final_state=state,
            classical_memory=classical_memory,
            measurement_results=measurement_results,
            intermediate_measurements=intermediate_measurements,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DynamicCircuit(num_qubits={self._num_qubits}, "
            f"num_classical_bits={self._num_classical_bits}, "
            f"num_gates={self.num_gates}, depth={self.depth})"
        )

    def __str__(self) -> str:
        lines = [
            f"DynamicCircuit ({self._num_qubits} qubits, "
            f"{self._num_classical_bits} classical bits, "
            f"{self.num_gates} gates, depth {self.depth})"
        ]
        gate_names = {
            "h": "H", "x": "X", "y": "Y", "z": "Z",
            "s": "S", "t": "T", "cnot": "CX", "cz": "CZ",
            "swap": "SWAP", "rx": "RX", "ry": "RY", "rz": "RZ",
        }
        for op in self._ops:
            if op[0] == "gate":
                gate_op: Operator = op[1]
                targets: list[int] = op[2]
                name = gate_names.get(gate_op.name, gate_op.name.upper())
                target_str = ",".join(str(t) for t in targets)
                lines.append(f"  {name}({target_str})")
            elif op[0] == "measure":
                _, qubit, cb = op
                lines.append(f"  M(q{qubit}->c{cb})")
            elif op[0] == "classical_if":
                _, cb, _ = op
                lines.append(f"  IF(c{cb}) {{ ... }}")
        return "\n".join(lines)
