"""Tests for Phase 117: the System Standard Library (`microquantum.stdlib`).

Covers the three foundational modules:

* ``stdlib.bits`` — MSB-first bitstring/integer conversions and Hamming
  measures (conventions identical to the SDK's measurement bitstrings).
* ``stdlib.numbers`` — angle normalization and modulo-``2*pi`` comparisons.
* ``stdlib.states`` — common state-vector factories on top of ``StateVector``,
  validated against the SDK's own circuit engine.

The stdlib adds no new dependencies and is verified to be mypy-strict clean
via the conformance gate.
"""
import math

import numpy as np
import pytest

from microquantum import (
    QuantumCircuit,
    StateVector,
    basis_state,
    bell_state,
    bits_to_int,
    bitstring_to_int,
    ghz_state,
    hamming_distance,
    hamming_weight,
    int_to_bits,
    int_to_bitstring,
    is_angle_close,
    is_identity_angle,
    mod_2pi,
    stdlib,
    uniform_superposition,
    w_state,
    wrap_angle,
)


# ==============================================================================
# Namespace surface
# ==============================================================================
class TestNamespace:
    def test_stdlib_importable_from_top_level(self) -> None:
        assert stdlib.__name__ == "microquantum.stdlib"

    def test_top_level_aggregates_stdlib_names(self) -> None:
        imported = {
            "int_to_bits",
            "bits_to_int",
            "int_to_bitstring",
            "bitstring_to_int",
            "hamming_weight",
            "hamming_distance",
            "mod_2pi",
            "wrap_angle",
            "is_angle_close",
            "is_identity_angle",
            "basis_state",
            "uniform_superposition",
            "bell_state",
            "ghz_state",
            "w_state",
        }
        assert imported <= set(stdlib.__all__)
        for name in imported:
            assert hasattr(stdlib, name)

    def test_stdlib_names_declared_public(self) -> None:
        for name in stdlib.__all__:
            assert name in __import__("microquantum").__all__


# ==============================================================================
# microquantum.stdlib.bits
# ==============================================================================
class TestIntToBitstring:
    def test_simple_conversions(self) -> None:
        assert int_to_bitstring(0) == "0"
        assert int_to_bitstring(5) == "101"
        assert int_to_bitstring(255) == "11111111"

    def test_width_zero_pads(self) -> None:
        assert int_to_bitstring(5, width=8) == "00000101"
        assert int_to_bitstring(0, width=3) == "000"

    def test_roundtrip(self) -> None:
        for value in (0, 1, 7, 42, 2**30, 2**40 - 1):
            assert bitstring_to_int(int_to_bitstring(value)) == value

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            int_to_bitstring(-1)

    def test_width_too_small_rejected(self) -> None:
        with pytest.raises(ValueError):
            int_to_bitstring(16, width=4)

    def test_invalid_types_rejected(self) -> None:
        with pytest.raises(TypeError):
            int_to_bitstring(3.5)
        with pytest.raises(TypeError):
            int_to_bitstring(3, width=2.0)


class TestBitstringToInt:
    def test_simple_conversions(self) -> None:
        assert bitstring_to_int("0") == 0
        assert bitstring_to_int("101") == 5
        assert bitstring_to_int("11111111") == 255

    def test_matches_msb_first_sdk_convention(self) -> None:
        # The SDK interprets a measurement outcome as int(bitstring, 2),
        # i.e. the leftmost character is the most significant bit.
        assert bitstring_to_int("10") == 2
        assert bitstring_to_int("01") == 1

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            bitstring_to_int("")

    def test_non_binary_rejected(self) -> None:
        with pytest.raises(ValueError):
            bitstring_to_int("102")
        with pytest.raises(ValueError):
            bitstring_to_int("10a1")

    def test_non_str_rejected(self) -> None:
        with pytest.raises(TypeError):
            bitstring_to_int(101)


class TestIntToBits:
    def test_msb_first_tuple(self) -> None:
        assert int_to_bits(5) == (1, 0, 1)
        assert int_to_bits(0) == (0,)
        assert int_to_bits(8) == (1, 0, 0, 0)

    def test_width_pads_most_significant_side(self) -> None:
        assert int_to_bits(5, width=5) == (0, 0, 1, 0, 1)

    def test_roundtrip(self) -> None:
        for value in (0, 1, 3, 64, 2**32):
            assert bits_to_int(int_to_bits(value)) == value

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            int_to_bits(-2)
        with pytest.raises(ValueError):
            int_to_bits(16, width=3)


class TestBitsToInt:
    def test_simple_conversions(self) -> None:
        assert bits_to_int((1, 0, 1)) == 5
        assert bits_to_int((1,)) == 1
        assert bits_to_int((1, 0, 0, 0)) == 8

    def test_accepts_any_iterable(self) -> None:
        assert bits_to_int([1, 1]) == 3
        assert bits_to_int(range(1, 2)) == 1

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            bits_to_int([])

    def test_non_binary_values_rejected(self) -> None:
        with pytest.raises(ValueError):
            bits_to_int((1, 2))
        with pytest.raises(ValueError):
            bits_to_int((1, -1))

    def test_non_int_elements_rejected(self) -> None:
        with pytest.raises(TypeError):
            bits_to_int((0.0, 1))


class TestHammingWeight:
    def test_integer_input(self) -> None:
        assert hamming_weight(0) == 0
        assert hamming_weight(5) == 2  # 101
        assert hamming_weight(255) == 8
        assert hamming_weight(2**100) == 1

    def test_bitstring_input(self) -> None:
        assert hamming_weight("0000") == 0
        assert hamming_weight("101") == 2
        assert hamming_weight("111") == 3

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            hamming_weight(-1)
        with pytest.raises(ValueError):
            hamming_weight("")
        with pytest.raises(ValueError):
            hamming_weight("102")
        with pytest.raises(TypeError):
            hamming_weight(1.0)


class TestHammingDistance:
    def test_integer_input(self) -> None:
        assert hamming_distance(0, 0) == 0
        assert hamming_distance(5, 3) == 2  # 101 vs 011
        assert hamming_distance(255, 0) == 8
        assert hamming_distance(7, 7) == 0

    def test_bitstring_input(self) -> None:
        assert hamming_distance("00", "00") == 0
        assert hamming_distance("101", "111") == 1
        assert hamming_distance("000", "111") == 3

    def test_mixed_kinds_rejected(self) -> None:
        with pytest.raises(TypeError):
            hamming_distance(5, "101")

    def test_unequal_length_bitstrings_rejected(self) -> None:
        with pytest.raises(ValueError):
            hamming_distance("00", "000")

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            hamming_distance(-1, 0)
        with pytest.raises(ValueError):
            hamming_distance("", "0")


# ==============================================================================
# microquantum.stdlib.numbers
# ==============================================================================
class TestModTwoPi:
    def test_principal_range(self) -> None:
        assert mod_2pi(0.0) == 0.0
        assert mod_2pi(2 * math.pi) == pytest.approx(0.0)
        assert mod_2pi(math.pi) == pytest.approx(math.pi)
        assert mod_2pi(4 * math.pi + 0.5) == pytest.approx(0.5)

    def test_negative_angles_are_canonical(self) -> None:
        assert mod_2pi(-math.pi / 2) == pytest.approx(3 * math.pi / 2)
        assert mod_2pi(-2 * math.pi) == pytest.approx(0.0)

    def test_result_within_half_open_interval(self) -> None:
        for value in np.linspace(-40, 40, 1001):
            reduced = mod_2pi(float(value))
            assert 0.0 <= reduced < 2 * math.pi

    def test_invalid_inputs(self) -> None:
        with pytest.raises(TypeError):
            mod_2pi(1 + 2j)
        with pytest.raises(TypeError):
            mod_2pi("1.5")
        with pytest.raises(ValueError):
            mod_2pi(math.nan)
        with pytest.raises(ValueError):
            mod_2pi(math.inf)


class TestWrapAngle:
    def test_principal_value(self) -> None:
        assert wrap_angle(0.0) == 0.0
        assert wrap_angle(math.pi) == pytest.approx(math.pi)
        assert wrap_angle(-math.pi) == pytest.approx(-math.pi)
        assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
        assert wrap_angle(-math.pi / 2) == pytest.approx(-math.pi / 2)

    def test_result_within_symmetric_interval(self) -> None:
        for value in np.linspace(-40, 40, 1001):
            wrapped = wrap_angle(float(value))
            assert -math.pi <= wrapped <= math.pi

    def test_invalid_inputs(self) -> None:
        with pytest.raises(TypeError):
            wrap_angle(1 + 2j)
        with pytest.raises(ValueError):
            wrap_angle(math.nan)


class TestIsAngleClose:
    def test_equal_angles(self) -> None:
        assert is_angle_close(0.0, 0.0)
        assert is_angle_close(math.pi, math.pi)

    def test_equivalent_by_full_turns(self) -> None:
        assert is_angle_close(0.0, 2 * math.pi)
        assert is_angle_close(math.pi / 2, -3 * math.pi / 2)
        assert is_angle_close(0.0, 4 * math.pi)

    def test_different_rotations_not_close(self) -> None:
        assert not is_angle_close(math.pi, math.pi / 2)
        assert not is_angle_close(0.0, 0.5)

    def test_tolerance_respected(self) -> None:
        assert is_angle_close(0.0, 1e-6, tol=1e-5)
        assert not is_angle_close(0.0, 1e-6, tol=1e-9)
        assert is_angle_close(0.0, 2 * math.pi + 1e-6, tol=1e-5)

    def test_matches_sdk_transpiler_rule(self) -> None:
        # mirror of core/transpiler._angle_close
        for a, b in [(1.1, 1.1 + 8 * math.pi), (0.2, -2 * math.pi + 0.2)]:
            assert is_angle_close(a, b) is True

    def test_invalid_inputs(self) -> None:
        with pytest.raises(TypeError):
            is_angle_close(1j, 0.0)
        with pytest.raises(ValueError):
            is_angle_close(math.nan, 0.0)
        with pytest.raises(ValueError):
            is_angle_close(0.0, 0.0, tol=-1.0)


class TestIsIdentityAngle:
    def test_identity_angles(self) -> None:
        assert is_identity_angle(0.0)
        assert is_identity_angle(2 * math.pi)
        assert is_identity_angle(4 * math.pi)
        assert is_identity_angle(-2 * math.pi)

    def test_non_identity_angles(self) -> None:
        assert not is_identity_angle(math.pi)
        assert not is_identity_angle(math.pi / 2)

    def test_tolerance_respected(self) -> None:
        assert is_identity_angle(1e-8, tol=1e-6)
        assert not is_identity_angle(1e-6, tol=1e-9)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(TypeError):
            is_identity_angle("0")
        with pytest.raises(ValueError):
            is_identity_angle(math.inf)


# ==============================================================================
# microquantum.stdlib.states
# ==============================================================================
class TestBasisState:
    def test_shape_and_single_amplitude(self) -> None:
        state = basis_state(3, 5)  # |101>
        assert state.num_qubits == 3
        assert state.dim == 8
        assert state.amplitudes[5] == pytest.approx(1.0)
        assert np.count_nonzero(state.amplitudes) == 1

    def test_zero_index_default_like_constructor(self) -> None:
        assert np.array_equal(
            basis_state(2, 0).amplitudes, StateVector(2).amplitudes
        )

    def test_normalized(self) -> None:
        assert basis_state(4, 10).is_normalized

    def test_index_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            basis_state(2, 4)
        with pytest.raises(ValueError):
            basis_state(2, -1)

    def test_beyond_dense_budget_rejected(self) -> None:
        with pytest.raises(ValueError):
            basis_state(65, 0)


class TestUniformSuperposition:
    def test_equal_amplitudes(self) -> None:
        state = uniform_superposition(3)
        expected = 1.0 / math.sqrt(8)
        assert np.allclose(state.amplitudes, expected)

    def test_normalized(self) -> None:
        assert uniform_superposition(5).is_normalized

    def test_matches_hadamard_circuit(self) -> None:
        qc = QuantumCircuit(3)
        for qubit in range(3):
            qc.h(qubit)
        sv = StateVector(3)
        from microquantum import apply_gate

        for ins in qc._gate_instructions:
            g, t = ins
            sv = apply_gate(
                sv, np.asarray(g.matrix, dtype=np.complex128), list(t)
            )
        assert np.allclose(sv.amplitudes, uniform_superposition(3).amplitudes)


class TestBellState:
    def test_phi_plus(self) -> None:
        state = bell_state(0)
        expected = np.zeros(4, dtype=complex)
        expected[0] = expected[3] = 1.0 / math.sqrt(2)
        assert np.allclose(state.amplitudes, expected)

    def test_phi_minus(self) -> None:
        state = bell_state(1)
        assert state.amplitudes[0] == pytest.approx(1 / math.sqrt(2))
        assert state.amplitudes[3] == pytest.approx(-1 / math.sqrt(2))
        assert state.amplitudes[1] == pytest.approx(0.0)

    def test_psi_plus(self) -> None:
        state = bell_state(2)
        assert state.amplitudes[1] == pytest.approx(1 / math.sqrt(2))
        assert state.amplitudes[2] == pytest.approx(1 / math.sqrt(2))
        assert state.amplitudes[0] == pytest.approx(0.0)

    def test_psi_minus(self) -> None:
        state = bell_state(3)
        assert state.amplitudes[1] == pytest.approx(1 / math.sqrt(2))
        assert state.amplitudes[2] == pytest.approx(-1 / math.sqrt(2))

    def test_default_is_phi_plus(self) -> None:
        assert np.array_equal(bell_state().amplitudes, bell_state(0).amplitudes)

    def test_all_bell_states_normalized_and_orthogonal(self) -> None:
        states = [bell_state(i) for i in range(4)]
        for state in states:
            assert state.is_normalized
        for i in range(4):
            for j in range(i + 1, 4):
                overlap = abs(np.vdot(states[i].amplitudes, states[j].amplitudes))
                assert overlap < 1e-12

    def test_index_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            bell_state(4)
        with pytest.raises(ValueError):
            bell_state(-1)


class TestGHZState:
    def test_two_extremes(self) -> None:
        state = ghz_state(4)
        assert state.amplitudes[0] == pytest.approx(1 / math.sqrt(2))
        assert state.amplitudes[15] == pytest.approx(1 / math.sqrt(2))
        assert np.count_nonzero(state.amplitudes) == 2

    def test_normalized(self) -> None:
        assert ghz_state(5).is_normalized

    def test_single_qubit_is_plus(self) -> None:
        state = ghz_state(1)
        assert state.amplitudes[0] == pytest.approx(1 / math.sqrt(2))
        assert state.amplitudes[1] == pytest.approx(1 / math.sqrt(2))

    def test_matches_circuit_construction(self) -> None:
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        from microquantum import apply_gate

        sv = StateVector(3)
        for ins in qc._gate_instructions:
            g, t = ins
            sv = apply_gate(
                sv, np.asarray(g.matrix, dtype=np.complex128), list(t)
            )
        assert np.allclose(sv.amplitudes, ghz_state(3).amplitudes)

    def test_beyond_dense_budget_rejected(self) -> None:
        with pytest.raises(ValueError):
            ghz_state(65)


class TestWState:
    def test_single_excitation_superposition(self) -> None:
        state = w_state(3)
        expected = np.zeros(8, dtype=complex)
        for qubit in range(3):
            expected[1 << qubit] = 1.0 / math.sqrt(3)
        assert np.allclose(state.amplitudes, expected)

    def test_normalized(self) -> None:
        assert w_state(6).is_normalized

    def test_single_qubit_is_one(self) -> None:
        state = w_state(1)
        assert state.amplitudes[1] == pytest.approx(1.0)
        assert np.count_nonzero(state.amplitudes) == 1


class TestStateValidation:
    _FACTORIES = [
        lambda num_qubits: basis_state(num_qubits, 0),
        uniform_superposition,
        ghz_state,
        w_state,
    ]

    @pytest.mark.parametrize("factory", _FACTORIES)
    def test_zero_qubits_rejected(self, factory) -> None:
        with pytest.raises(ValueError):
            factory(0)

    @pytest.mark.parametrize("factory", _FACTORIES)
    def test_non_int_qubits_rejected(self, factory) -> None:
        with pytest.raises(ValueError):
            factory(2.0)

    def test_non_int_index_rejected(self) -> None:
        with pytest.raises(TypeError):
            basis_state(2, 1.0)
        with pytest.raises(TypeError):
            bell_state(True)