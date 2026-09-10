"""Unit tests for microquantum.core.state.StateVector."""

import copy

import numpy as np
import pytest

from microquantum.core import StateVector


class TestDefaultInitialization:
    """Tests for default |0...0> initialization across qubit counts."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_default_state(self, n: int) -> None:
        sv = StateVector(n)
        assert sv.num_qubits == n
        assert sv.dim == 2**n
        assert sv.amplitudes.dtype == np.complex128
        assert sv.amplitudes.shape == (2**n,)
        assert sv.is_normalized

    def test_one_qubit_default(self) -> None:
        sv = StateVector(1)
        expected = np.array([1.0, 0.0], dtype=np.complex128)
        np.testing.assert_array_equal(sv.amplitudes, expected)

    def test_two_qubit_default(self) -> None:
        sv = StateVector(2)
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        np.testing.assert_array_equal(sv.amplitudes, expected)

    def test_three_qubit_default(self) -> None:
        sv = StateVector(3)
        assert sv.dim == 8
        assert sv.amplitudes[0] == 1.0
        np.testing.assert_array_equal(sv.amplitudes[1:], np.zeros(7, dtype=np.complex128))


class TestCustomInitialization:
    """Tests for custom amplitude initialization."""

    def test_valid_custom_amplitudes(self) -> None:
        amps = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)
        sv = StateVector(1, amplitudes=amps)
        np.testing.assert_array_equal(sv.amplitudes, amps)

    def test_bell_state_initialization(self) -> None:
        amps = np.array([0, 0, 0, 1], dtype=np.complex128)
        sv = StateVector(2, amplitudes=amps)
        assert sv.amplitudes[3] == 1.0
        assert sv.is_normalized

    def test_wrong_length_amplitudes_raises(self) -> None:
        amps = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        with pytest.raises(ValueError, match="amplitudes must have length 4"):
            StateVector(2, amplitudes=amps)

    def test_num_qubits_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            StateVector(0)

    def test_num_qubits_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            StateVector(-1)


class TestNormalization:
    """Tests for normalization validation and normalization method."""

    def test_normalized_state(self) -> None:
        amps = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)
        sv = StateVector(1, amplitudes=amps)
        assert sv.is_normalized

    def test_unnormalized_state(self) -> None:
        amps = np.array([1.0, 1.0], dtype=np.complex128)
        sv = StateVector(1, amplitudes=amps)
        assert not sv.is_normalized

    def test_normalize_method(self) -> None:
        amps = np.array([3.0, 4.0], dtype=np.complex128)
        sv = StateVector(1, amplitudes=amps)
        result = sv.normalize()
        assert sv.is_normalized
        assert result is sv
        np.testing.assert_almost_equal(np.sum(np.abs(sv.amplitudes) ** 2), 1.0)

    def test_normalize_zero_vector(self) -> None:
        amps = np.array([0.0, 0.0], dtype=np.complex128)
        sv = StateVector(1, amplitudes=amps)
        sv.normalize()
        np.testing.assert_array_equal(sv.amplitudes, amps)


class TestInnerProduct:
    """Tests for inner product calculations."""

    def test_inner_product_ket_ket(self) -> None:
        ket0 = StateVector(1)
        ket1 = StateVector(1, amplitudes=np.array([0.0, 1.0], dtype=np.complex128))
        assert ket0.inner_product(ket0) == pytest.approx(1.0)
        assert ket0.inner_product(ket1) == pytest.approx(0.0)
        assert ket1.inner_product(ket1) == pytest.approx(1.0)

    def test_superposition_inner_product(self) -> None:
        plus = 1 / np.sqrt(2)
        ket_plus = StateVector(
            1,
            amplitudes=np.array([plus, plus], dtype=np.complex128),
        )
        ket0 = StateVector(1)
        assert ket0.inner_product(ket_plus) == pytest.approx(complex(plus))
        assert ket_plus.inner_product(ket_plus) == pytest.approx(1.0)

    def test_bell_state_inner_product(self) -> None:
        bell = StateVector(
            2,
            amplitudes=np.array(
                [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
                dtype=np.complex128,
            ),
        )
        ket00 = StateVector(2)
        assert ket00.inner_product(bell) == pytest.approx(complex(1 / np.sqrt(2)))

    def test_inner_product_conjugate(self) -> None:
        s1 = StateVector(
            1,
            amplitudes=np.array([1j, 0.0], dtype=np.complex128),
        )
        s2 = StateVector(1)
        result = s2.inner_product(s1)
        assert result == pytest.approx(-1j)

    def test_inner_product_dimension_mismatch(self) -> None:
        s1 = StateVector(1)
        s2 = StateVector(2)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            s1.inner_product(s2)


class TestFidelity:
    """Tests for fidelity calculations."""

    def test_identical_states_fidelity(self) -> None:
        s1 = StateVector(1)
        s2 = StateVector(1)
        assert s1.fidelity(s2) == pytest.approx(1.0)

    def test_orthogonal_states_fidelity(self) -> None:
        ket0 = StateVector(1)
        ket1 = StateVector(1, amplitudes=np.array([0.0, 1.0], dtype=np.complex128))
        assert ket0.fidelity(ket1) == pytest.approx(0.0)

    def test_partial_overlap_fidelity(self) -> None:
        ket0 = StateVector(1)
        plus = 1 / np.sqrt(2)
        ket_plus = StateVector(
            1,
            amplitudes=np.array([plus, plus], dtype=np.complex128),
        )
        assert ket0.fidelity(ket_plus) == pytest.approx(0.5)

    def test_fidelity_symmetry(self) -> None:
        s1 = StateVector(
            1,
            amplitudes=np.array([0.6, 0.8], dtype=np.complex128),
        )
        s2 = StateVector(
            1,
            amplitudes=np.array([0.8, 0.6], dtype=np.complex128),
        )
        assert s1.fidelity(s2) == pytest.approx(s2.fidelity(s1))

    def test_fidelity_dimension_mismatch(self) -> None:
        s1 = StateVector(1)
        s2 = StateVector(2)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            s1.fidelity(s2)


class TestCopy:
    """Tests for deep copy functionality."""

    def test_copy_independence(self) -> None:
        sv = StateVector(1, amplitudes=np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128))
        sv_copy = sv.copy()
        np.testing.assert_array_equal(sv.amplitudes, sv_copy.amplitudes)
        sv_copy._amplitudes[0] = 999.0
        assert sv.amplitudes[0] != 999.0

    def test_copy_preserves_metadata(self) -> None:
        sv = StateVector(3)
        sv_copy = sv.copy()
        assert sv_copy.num_qubits == sv.num_qubits
        assert sv_copy.dim == sv.dim

    def test_copy_amplitude_equality(self) -> None:
        amps = np.array([0.5 + 0.5j, 0.5 - 0.5j, 0.0, 0.0], dtype=np.complex128)
        sv = StateVector(2, amplitudes=amps)
        sv_copy = sv.copy()
        np.testing.assert_array_equal(sv.amplitudes, sv_copy.amplitudes)


class TestStringRepresentation:
    """Tests for __repr__ and __str__ formatting."""

    def test_repr_contains_class_name(self) -> None:
        sv = StateVector(1)
        r = repr(sv)
        assert "StateVector" in r
        assert "num_qubits=1" in r

    def test_str_default_state(self) -> None:
        sv = StateVector(1)
        s = str(sv)
        assert "|0>" in s
        assert "(1+0j)" in s

    def test_str_superposition(self) -> None:
        plus = 1 / np.sqrt(2)
        sv = StateVector(1, amplitudes=np.array([plus, plus], dtype=np.complex128))
        s = str(sv)
        assert "|0>" in s
        assert "|1>" in s
        assert " + " in s

    def test_str_two_qubits(self) -> None:
        sv = StateVector(2)
        s = str(sv)
        assert "|00>" in s

    def test_str_zero_state(self) -> None:
        sv = StateVector(1, amplitudes=np.array([0.0, 0.0], dtype=np.complex128))
        s = str(sv)
        assert s == "0"


class TestEdgeCases:
    """Tests for edge cases and additional validation."""

    def test_phase_state(self) -> None:
        sv = StateVector(1, amplitudes=np.array([1j, 0.0], dtype=np.complex128))
        assert sv.is_normalized
        assert sv.fidelity(StateVector(1)) == pytest.approx(1.0)

    def test_negative_amplitudes(self) -> None:
        amps = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)
        sv = StateVector(1, amplitudes=amps)
        assert sv.is_normalized

    def test_four_qubit_maximally_entangled(self) -> None:
        amps = np.zeros(16, dtype=np.complex128)
        amps[0] = 1 / np.sqrt(2)
        amps[15] = 1 / np.sqrt(2)
        sv = StateVector(4, amplitudes=amps)
        assert sv.is_normalized
        assert sv.num_qubits == 4
        assert sv.dim == 16
