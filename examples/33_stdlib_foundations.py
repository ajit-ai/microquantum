"""Example 33: The System Standard Library (``microquantum.stdlib``).

Demonstrates the Phase-117 standard library layer — a stable, dependency-light
foundation shared by user programs, the runtime and the compiler:

- ``stdlib.bits`` — MSB-first bitstring/integer conversions and Hamming
  measures, using the same convention as measurement outcomes.
- ``stdlib.numbers`` — canonical angle reduction (``mod_2pi`` / ``wrap_angle``)
  and modulo-``2*pi`` rotation comparisons.
- ``stdlib.states`` — common state factories (Bell / GHZ / W / basis /
  uniform) built on :class:`StateVector`, cross-checked against circuits.
"""

from __future__ import annotations

import math

import numpy as np

from microquantum import (
    QuantumCircuit,
    StateVector,
    StatevectorBackend,
    apply_gate,
    basis_state,
    bell_state,
    bitstring_to_int,
    ghz_state,
    hamming_distance,
    hamming_weight,
    int_to_bitstring,
    is_angle_close,
    mod_2pi,
    uniform_superposition,
    w_state,
)
from microquantum.stdlib import bits_to_int, int_to_bits, is_identity_angle, wrap_angle


def run(circuit: QuantumCircuit) -> np.ndarray:
    """Run a circuit on the statevector backend and return the amplitudes."""
    backend = StatevectorBackend()
    result = backend.run(circuit, shots=None)
    assert result.statevector is not None
    return np.asarray(result.statevector, dtype=np.complex128)


def state_after_gates(num_qubits: int, circuit: QuantumCircuit) -> np.ndarray:
    """Reference state via the core engine (no backend round-trip)."""
    state = StateVector(num_qubits)
    for gate, targets in circuit._gate_instructions:
        state = apply_gate(
            state, np.asarray(gate.matrix, dtype=np.complex128), list(targets)
        )
    return state.amplitudes


def bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def main() -> None:
    print("=== 33 Standard Library foundations ===\n")

    # -- bits: measurement outcomes use MSB-first bitstrings ---------------
    outcome = "1011"
    value = bitstring_to_int(outcome)
    print(f"bitstring '{outcome}' -> int {value} -> back '{int_to_bitstring(value)}'")
    assert int_to_bitstring(value) == outcome

    bits = int_to_bits(11)  # (1, 0, 1, 1)
    print(f"int 11 -> bit tuple   {bits} -> int {bits_to_int(bits)}")
    assert bits_to_int(bits) == 11

    print(f"hamming_weight(11) = {hamming_weight(11)}  "
          f"hamming_distance(0b1011, 0b1111) = {hamming_distance(11, 15)}")
    assert hamming_weight(11) == 3
    assert hamming_distance(11, 15) == 1

    # -- numbers: rotations are equal modulo 2*pi --------------------------
    angle = 4 * math.pi + 0.25
    print(f"mod_2pi({angle:.3f}) = {mod_2pi(angle):.3f}  "
          f"wrap_angle({9 * math.pi / 4:.3f}) = {wrap_angle(9 * math.pi / 4):.3f}")
    assert is_identity_angle(4 * math.pi)       # full turns are the identity
    assert is_angle_close(math.pi / 2, -3 * math.pi / 2)  # equivalent rotations

    # -- states: factories agree with the circuit engine -------------------
    ghz = ghz_state(4)
    fidelity = float(abs(np.vdot(state_after_gates(4, ghz_circuit()), ghz.amplitudes)) ** 2)
    print(f"ghz_state(4) matches an H+CX circuit: fidelity {fidelity:.10f}")
    assert fidelity > 1 - 1e-9

    bell = bell_state(0)
    fidelity = float(abs(np.vdot(run(bell_circuit()), bell.amplitudes)) ** 2)
    print(f"bell_state(0) matches |00>+|11> circuit: fidelity {fidelity:.10f}")
    assert fidelity > 1 - 1e-9

    uniform = uniform_superposition(3)
    print(f"uniform_superposition(3) normalized: {uniform.is_normalized}")
    assert uniform.is_normalized

    w = w_state(3)
    print(f"w_state(3): any amplitude = {abs(w.amplitudes[1]):.4f}, "
          f"non-zero entries = {np.count_nonzero(w.amplitudes)}")
    assert abs(np.linalg.norm(w.amplitudes) - 1.0) < 1e-12

    base = basis_state(3, 5)
    assert base.amplitudes[5] == 1.0
    print(f"basis_state(3, 5) = |{int_to_bitstring(5, width=3)}>")

    print("\nExample 33 completed!")


def ghz_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    return circuit


if __name__ == "__main__":
    main()