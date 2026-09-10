"""Unit tests for microquantum.core.measurement."""

import numpy as np
import pytest

from microquantum.core import (
    MeasurementResult,
    Operator,
    QuantumCircuit,
    StateVector,
    expectation_value,
    measure_and_collapse,
    measure_qubits,
    sample_state,
    tensor,
)

# ---------------------------------------------------------------------------
# Deterministic state sampling
# ---------------------------------------------------------------------------

class TestDeterministicSampling:
    """Test sampling of computational basis states."""

    def test_ket00_yields_all_00(self) -> None:
        state = StateVector(2)
        result = sample_state(state, shots=100, seed=42)
        assert result.most_frequent() == "00"
        assert result.get_counts() == {"00": 100}

    def test_ket11_yields_all_11(self) -> None:
        amps = np.array([0, 0, 0, 1], dtype=np.complex128)
        state = StateVector(2, amplitudes=amps)
        result = sample_state(state, shots=100, seed=42)
        assert result.most_frequent() == "11"
        assert result.get_counts() == {"11": 100}

    def test_ket0_yields_all_0(self) -> None:
        state = StateVector(1)
        result = sample_state(state, shots=50, seed=0)
        assert result.get_counts() == {"0": 50}

    def test_ket1_yields_all_1(self) -> None:
        amps = np.array([0, 1], dtype=np.complex128)
        state = StateVector(1, amplitudes=amps)
        result = sample_state(state, shots=50, seed=0)
        assert result.get_counts() == {"1": 50}


# ---------------------------------------------------------------------------
# Bell state sampling
# ---------------------------------------------------------------------------

class TestBellStateSampling:
    """Bell state over many shots yields ~50/50 of |00> and |11>."""

    def test_bell_state_distribution(self) -> None:
        inv_sqrt2 = 1 / np.sqrt(2)
        amps = np.array(
            [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
        )
        state = StateVector(2, amplitudes=amps)
        result = sample_state(state, shots=10000, seed=42)

        counts = result.get_counts()
        assert "01" not in counts
        assert "10" not in counts
        assert "00" in counts
        assert "11" in counts

        ratio = counts["00"] / 10000
        assert 0.45 < ratio < 0.55


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Same seed yields identical shot counts."""

    def test_same_seed_same_counts(self) -> None:
        inv_sqrt2 = 1 / np.sqrt(2)
        amps = np.array(
            [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
        )
        state = StateVector(2, amplitudes=amps)

        r1 = sample_state(state, shots=1000, seed=123)
        r2 = sample_state(state, shots=1000, seed=123)
        assert r1.get_counts() == r2.get_counts()

    def test_different_seed_different_counts(self) -> None:
        inv_sqrt2 = 1 / np.sqrt(2)
        amps = np.array(
            [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
        )
        state = StateVector(2, amplitudes=amps)

        r1 = sample_state(state, shots=1000, seed=1)
        r2 = sample_state(state, shots=1000, seed=2)
        assert r1.get_counts() != r2.get_counts()


# ---------------------------------------------------------------------------
# MeasurementResult properties
# ---------------------------------------------------------------------------

class TestMeasurementResult:
    """Test MeasurementResult container methods."""

    def test_get_probabilities(self) -> None:
        mr = MeasurementResult(counts={"00": 300, "11": 700}, shots=1000, qubits=[0, 1])
        probs = mr.get_probabilities()
        assert probs["00"] == pytest.approx(0.3)
        assert probs["11"] == pytest.approx(0.7)

    def test_most_frequent(self) -> None:
        mr = MeasurementResult(counts={"0": 10, "1": 90}, shots=100, qubits=[0])
        assert mr.most_frequent() == "1"

    def test_repr(self) -> None:
        mr = MeasurementResult(counts={"0": 1}, shots=1, qubits=[0])
        r = repr(mr)
        assert "MeasurementResult" in r
        assert "shots=1" in r

    def test_str(self) -> None:
        mr = MeasurementResult(counts={"0": 50, "1": 50}, shots=100, qubits=[0])
        s = str(mr)
        assert "100 shots" in s


# ---------------------------------------------------------------------------
# Projective collapse
# ---------------------------------------------------------------------------

class TestMeasureAndCollapse:
    """Test measure_and_collapse on Bell state."""

    def test_bell_collapse_to_matching_state(self) -> None:
        inv_sqrt2 = 1 / np.sqrt(2)
        amps = np.array(
            [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
        )
        bell = StateVector(2, amplitudes=amps)

        measured, collapsed = measure_and_collapse(bell, [0, 1], seed=42)

        if measured == "00":
            expected = StateVector(2, amplitudes=np.array([1, 0, 0, 0], dtype=np.complex128))
        else:
            expected = StateVector(2, amplitudes=np.array([0, 0, 0, 1], dtype=np.complex128))

        np.testing.assert_array_almost_equal(collapsed.amplitudes, expected.amplitudes)

    def test_collapse_single_qubit(self) -> None:
        """Measuring qubit 0 of Bell state collapses qubit 1."""
        inv_sqrt2 = 1 / np.sqrt(2)
        amps = np.array(
            [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
        )
        bell = StateVector(2, amplitudes=amps)

        measured, collapsed = measure_and_collapse(bell, [0], seed=0)
        assert measured in ("0", "1")
        assert collapsed.is_normalized

    def test_collapse_bitstring_length(self) -> None:
        inv_sqrt2 = 1 / np.sqrt(2)
        amps = np.array(
            [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
        )
        bell = StateVector(2, amplitudes=amps)
        measured, _ = measure_and_collapse(bell, [0, 1], seed=0)
        assert len(measured) == 2


# ---------------------------------------------------------------------------
# Expectation values
# ---------------------------------------------------------------------------

class TestExpectationValues:
    """Test expectation values of standard observables."""

    def test_z_on_ket0(self) -> None:
        """<0|Z|0> = +1.0"""
        state = StateVector(1)
        result = expectation_value(state, Operator.Z())
        assert result == pytest.approx(1.0)

    def test_z_on_ket1(self) -> None:
        """<1|Z|1> = -1.0"""
        amps = np.array([0, 1], dtype=np.complex128)
        state = StateVector(1, amplitudes=amps)
        result = expectation_value(state, Operator.Z())
        assert result == pytest.approx(-1.0)

    def test_x_on_plus(self) -> None:
        """<+|X|+> = +1.0"""
        inv_sqrt2 = 1 / np.sqrt(2)
        amps = np.array([inv_sqrt2, inv_sqrt2], dtype=np.complex128)
        state = StateVector(1, amplitudes=amps)
        result = expectation_value(state, Operator.X())
        assert result == pytest.approx(1.0)

    def test_zz_on_bell(self) -> None:
        """<Phi+|ZZ|Phi+> = +1.0"""
        inv_sqrt2 = 1 / np.sqrt(2)
        amps = np.array(
            [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
        )
        state = StateVector(2, amplitudes=amps)
        zz = tensor(Operator.Z(), Operator.Z())
        result = expectation_value(state, zz)
        assert result == pytest.approx(1.0)

    def test_zz_on_bell_with_targets(self) -> None:
        """ZZ on Bell state using target expansion."""
        inv_sqrt2 = 1 / np.sqrt(2)
        amps = np.array(
            [inv_sqrt2, 0.0, 0.0, inv_sqrt2], dtype=np.complex128
        )
        state = StateVector(2, amplitudes=amps)
        zz = tensor(Operator.Z(), Operator.Z())
        result = expectation_value(state, zz, targets=[0, 1])
        assert result == pytest.approx(1.0)

    def test_x_on_ket0(self) -> None:
        """<0|X|0> = 0.0"""
        state = StateVector(1)
        result = expectation_value(state, Operator.X())
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Circuit integration
# ---------------------------------------------------------------------------

class TestCircuitMeasurement:
    """Test QuantumCircuit.measure_all and expectation_value."""

    def test_circuit_measure_all(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        result = qc.measure_all(shots=10000, seed=42)
        counts = result.get_counts()
        assert "01" not in counts
        assert "10" not in counts

    def test_circuit_expectation_value(self) -> None:
        qc = QuantumCircuit(1)
        result = qc.expectation_value(Operator.Z())
        assert result == pytest.approx(1.0)

    def test_circuit_expectation_with_targets(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        zz = tensor(Operator.Z(), Operator.Z())
        result = qc.expectation_value(zz, targets=[0, 1])
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge cases & validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Test validation error handling."""

    def test_measure_qubits_out_of_range(self) -> None:
        state = StateVector(2)
        with pytest.raises(ValueError, match="out of range"):
            measure_qubits(state, [2], shots=10)

    def test_measure_and_collapse_out_of_range(self) -> None:
        state = StateVector(2)
        with pytest.raises(ValueError, match="out of range"):
            measure_and_collapse(state, [-1])

    def test_expectation_value_state_size_mismatch(self) -> None:
        state = StateVector(1)
        with pytest.raises(ValueError, match="qubit"):
            expectation_value(state, Operator.CNOT())
