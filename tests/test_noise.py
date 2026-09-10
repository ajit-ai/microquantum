"""Tests for NoiseModel."""

import numpy as np
import pytest

from microquantum.backends.noise import NoiseModel
from microquantum.core.density_matrix import DensityMatrix


class TestNoiseModelCreation:
    def test_empty_model(self) -> None:
        nm = NoiseModel()
        assert len(nm.channels) == 0

    def test_repr(self) -> None:
        nm = NoiseModel().depolarizing(0.1)
        assert "NoiseModel(channels=1)" in repr(nm)


class TestDepolarizing:
    def test_zero_probability(self) -> None:
        nm = NoiseModel().depolarizing(0.0)
        rho = DensityMatrix.from_label("0")
        rho2 = nm.apply(rho)
        np.testing.assert_allclose(rho2.matrix, rho.matrix, atol=1e-10)

    def test_full_depolarizing(self) -> None:
        # At p=3/4, the Pauli channel produces maximally mixed state I/2
        nm = NoiseModel().depolarizing(0.75)
        rho = DensityMatrix.from_label("0")
        rho2 = nm.apply(rho)
        expected = np.eye(2, dtype=np.complex128) / 2
        np.testing.assert_allclose(rho2.matrix, expected, atol=1e-10)

    def test_partial_depolarizing_preserves_trace(self) -> None:
        nm = NoiseModel().depolarizing(0.3)
        sv = DensityMatrix.from_statevector(
            StateVector(1, amplitudes=np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        )
        rho2 = nm.apply(sv)
        assert rho2.trace == pytest.approx(1.0)

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            NoiseModel().depolarizing(1.5)
        with pytest.raises(ValueError, match="probability"):
            NoiseModel().depolarizing(-0.1)


class TestAmplitudeDamping:
    def test_zero_probability(self) -> None:
        nm = NoiseModel().amplitude_damping(0.0)
        rho = DensityMatrix.from_label("0")
        rho2 = nm.apply(rho)
        np.testing.assert_allclose(rho2.matrix, rho.matrix, atol=1e-10)

    def test_full_decay_to_ground(self) -> None:
        nm = NoiseModel().amplitude_damping(1.0)
        rho = DensityMatrix.from_label("1")
        rho2 = nm.apply(rho)
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        np.testing.assert_allclose(rho2.matrix, expected, atol=1e-10)

    def test_preserves_ground_state(self) -> None:
        nm = NoiseModel().amplitude_damping(0.5)
        rho = DensityMatrix.from_label("0")
        rho2 = nm.apply(rho)
        np.testing.assert_allclose(rho2.matrix, rho.matrix, atol=1e-10)

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            NoiseModel().amplitude_damping(2.0)


class TestPhaseDamping:
    def test_zero_probability(self) -> None:
        nm = NoiseModel().phase_damping(0.0)
        rho = DensityMatrix.from_label("0")
        rho2 = nm.apply(rho)
        np.testing.assert_allclose(rho2.matrix, rho.matrix, atol=1e-10)

    def test_preserves_ground_state(self) -> None:
        nm = NoiseModel().phase_damping(0.5)
        rho = DensityMatrix.from_label("0")
        rho2 = nm.apply(rho)
        np.testing.assert_allclose(rho2.matrix, rho.matrix, atol=1e-10)

    def test_dephases_superposition(self) -> None:
        nm = NoiseModel().phase_damping(1.0)
        sv = StateVector(1, amplitudes=np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        rho = DensityMatrix.from_statevector(sv)
        rho2 = nm.apply(rho)
        # Full dephasing kills off-diagonals
        assert abs(rho2.matrix[0, 1]) < 1e-10
        assert abs(rho2.matrix[1, 0]) < 1e-10


class TestBitFlip:
    def test_zero_probability(self) -> None:
        nm = NoiseModel().bit_flip(0.0)
        rho = DensityMatrix.from_label("0")
        rho2 = nm.apply(rho)
        np.testing.assert_allclose(rho2.matrix, rho.matrix, atol=1e-10)

    def test_full_bit_flip(self) -> None:
        nm = NoiseModel().bit_flip(1.0)
        rho = DensityMatrix.from_label("0")
        rho2 = nm.apply(rho)
        expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        np.testing.assert_allclose(rho2.matrix, expected, atol=1e-10)

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            NoiseModel().bit_flip(1.5)


class TestNoiseModelChaining:
    def test_multiple_channels(self) -> None:
        nm = (
            NoiseModel()
            .depolarizing(0.1)
            .amplitude_damping(0.05)
        )
        assert len(nm.channels) == 2
        assert nm.channels[0].name == "depolarizing"
        assert nm.channels[1].name == "amplitude_damping"


class TestNoiseModelStr:
    def test_str(self) -> None:
        nm = NoiseModel().depolarizing(0.1).amplitude_damping(0.05)
        s = str(nm)
        assert "depolarizing" in s
        assert "amplitude_damping" in s


# Need to import StateVector here
from microquantum.core.state import StateVector  # noqa: E402
