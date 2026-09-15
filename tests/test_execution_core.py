"""MQ-11 tests: quantum execution and measurement core.

Covers the native execution contract ``Circuit -> Backend.run(shots, seed) ->
ExecutionResult``, circuit-level measurement annotations, shot-based sampling,
seeded reproducibility and result-object behaviour.
"""

import numpy as np
import pytest

from microquantum import (
    BackendResult,
    QuantumCircuit,
    StatevectorBackend,
    execute,
)
from microquantum.backends.mock import MockBackend


def _bell() -> QuantumCircuit:
    return QuantumCircuit(2).h(0).cx(0, 1)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class TestMeasurement:
    def test_measure_all_marks_all_qubits(self) -> None:
        qc = _bell().measure_all()
        assert qc.measurements == [0, 1]

    def test_measure_single_qubit(self) -> None:
        qc = _bell().measure(0)
        assert qc.measurements == [0]

    def test_measure_is_explicit_not_executed(self) -> None:
        qc = _bell().measure_all()
        # measuring must not alter the gate sequence
        assert qc.num_gates == 2
        assert [g.name for g, _t in qc.gates] == ["h", "cnot"]

    def test_measurement_ordering_is_documented(self) -> None:
        qc = _bell().measure(1).measure(0)
        assert qc.measurements == [1, 0]

    def test_measure_out_of_range_raises(self) -> None:
        qc = _bell()
        with pytest.raises(ValueError, match="out of range"):
            qc.measure(2)

    def test_measure_duplicate_raises(self) -> None:
        qc = _bell().measure(0)
        with pytest.raises(ValueError, match="already marked"):
            qc.measure(0)

    def test_measure_all_idempotent(self) -> None:
        qc1 = _bell().measure_all()
        qc2 = _bell().measure_all().measure_all()
        assert qc1.measurements == qc2.measurements == [0, 1]

    def test_measurements_defensive_copy(self) -> None:
        qc = _bell().measure_all()
        measurements = qc.measurements
        measurements.append(999)
        assert qc.measurements == [0, 1]


class TestMeasurementExecution:
    def setup_method(self) -> None:
        self.backend = StatevectorBackend()

    def test_single_qubit_ket0_measurement(self) -> None:
        qc = QuantumCircuit(1).measure_all()
        result = self.backend.run(qc, shots=100, seed=42)
        assert result.counts == {"0": 100}

    def test_single_qubit_ket1_measurement(self) -> None:
        qc = QuantumCircuit(1).x(0).measure_all()
        result = self.backend.run(qc, shots=100, seed=42)
        assert result.counts == {"1": 100}

    def test_deterministic_basis_state_measurement(self) -> None:
        # |01> on a 2-qubit circuit -> deterministic '01'
        qc = QuantumCircuit(2).x(1).measure_all()
        result = self.backend.run(qc, shots=200, seed=7)
        assert result.counts == {"01": 200}

    def test_bell_state_measurement(self) -> None:
        qc = _bell().measure_all()
        result = self.backend.run(qc, shots=10000, seed=42)
        counts = result.counts
        assert "01" not in counts
        assert "10" not in counts
        assert counts.get("00", 0) + counts.get("11", 0) == 10000

    def test_partial_measurement(self) -> None:
        # Measure only qubit 0 of a Bell state -> single-bit outcomes.
        qc = _bell().measure(0)
        result = self.backend.run(qc, shots=1000, seed=42)
        assert all(len(k) == 1 for k in result.counts)
        assert sum(result.counts.values()) == 1000
        assert "0" in result.counts and "1" in result.counts


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def setup_method(self) -> None:
        self.backend = StatevectorBackend()

    def test_shots_count_equals_requested(self) -> None:
        qc = _bell().measure_all()
        for shots in (1, 7, 512, 2048):
            result = self.backend.run(qc, shots=shots, seed=1)
            assert sum(result.counts.values()) == shots

    def test_counts_only_valid_bitstrings(self) -> None:
        qc = QuantumCircuit(3).measure_all()
        result = self.backend.run(qc, shots=500, seed=3)
        assert result.counts == {"000": 500}

    def test_bell_distribution_statistically_sensible(self) -> None:
        qc = _bell().measure_all()
        result = self.backend.run(qc, shots=20000, seed=99)
        total = sum(result.counts.values())
        p00 = result.counts.get("00", 0) / total
        assert 0.48 < p00 < 0.52

    def test_seeded_execution_is_reproducible(self) -> None:
        qc = _bell().measure_all()
        r1 = self.backend.run(qc, shots=1000, seed=42)
        r2 = self.backend.run(qc, shots=1000, seed=42)
        assert r1.counts == r2.counts
        assert r1.samples == r2.samples

    def test_different_seeds_differ(self) -> None:
        qc = _bell().measure_all()
        r1 = self.backend.run(qc, shots=1000, seed=1)
        r2 = self.backend.run(qc, shots=1000, seed=2)
        assert r1.counts != r2.counts

    def test_invalid_shot_count_rejected(self) -> None:
        qc = QuantumCircuit(1).measure_all()
        with pytest.raises(ValueError):
            self.backend.run(qc, shots=0)
        with pytest.raises(ValueError):
            self.backend.run(qc, shots=-5)

    def test_seed_recorded_on_result(self) -> None:
        qc = _bell().measure_all()
        result = self.backend.run(qc, shots=100, seed=1234)
        assert result.seed == 1234
        assert result.shots == 100


# ---------------------------------------------------------------------------
# Execution & result object
# ---------------------------------------------------------------------------


class TestExecution:
    def setup_method(self) -> None:
        self.backend = StatevectorBackend()

    def test_basic_circuit_execution(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1).measure_all()
        result = self.backend.run(qc, shots=100, seed=0)
        assert isinstance(result, BackendResult)
        assert result.num_qubits == 2
        assert result.backend_name == "statevector"

    def test_empty_circuit_behavior(self) -> None:
        result = self.backend.run(QuantumCircuit(1).measure_all(), shots=10)
        assert result.counts == {"0": 10}
        state = result.state
        assert state is not None
        np.testing.assert_allclose(np.abs(state), np.array([1.0, 0.0]))

    def test_unmeasured_circuit_samples_all_qubits(self) -> None:
        # Backward compatible: without measurement annotations the backend
        # samples the full register.
        result = self.backend.run(QuantumCircuit(2), shots=10, seed=0)
        assert all(len(k) == 2 for k in result.counts)

    def test_backend_integration_through_execute(self) -> None:
        qc = _bell().measure_all()
        result = execute(qc, shots=1000, seed=42)
        assert isinstance(result, BackendResult)
        assert set(result.counts.keys()) <= {"00", "11"}
        assert sum(result.counts.values()) == 1000
        assert result.metadata["backend"] == "statevector"

    def test_result_state_property(self) -> None:
        qc = _bell().measure_all()
        result = self.backend.run(qc, shots=10, seed=0)
        assert result.state is result.statevector
        assert result.state is not None
        assert result.state.dtype == np.complex128

    def test_result_counts_accessor(self) -> None:
        qc = QuantumCircuit(1).x(0).measure_all()
        result = self.backend.run(qc, shots=3, seed=0)
        assert result.get_counts() == {"1": 3}
        assert result.counts == result.get_counts()

    def test_result_metadata_distinguishes_state_vs_samples(self) -> None:
        qc = QuantumCircuit(1).x(0).measure_all()
        result = self.backend.run(qc, shots=5, seed=1)
        assert result.state is not None          # state information
        assert result.counts == {"1": 5}         # sampled measurement info
        assert "shots" in result.metadata        # execution metadata

    def test_result_serialization_roundtrip(self) -> None:
        qc = _bell().measure_all()
        result = self.backend.run(qc, shots=500, seed=3)
        restored = BackendResult.from_dict(result.to_dict())
        assert restored.counts == result.counts
        assert restored.seed == result.seed
        assert restored.shots == result.shots
        if result.state is not None:
            np.testing.assert_allclose(restored.state, result.state)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


class TestRegression:
    def test_circuit_run_still_returns_statevector(self) -> None:
        qc = _bell()
        state = qc.run()
        assert hasattr(state, "amplitudes")
        assert state.num_qubits == 2

    def test_sample_state_still_accessible(self) -> None:
        from microquantum.core import sample_state

        qc = _bell()
        state = qc.run()
        mr = sample_state(state, shots=100, seed=42)
        assert mr.shots == 100
        assert sum(mr.counts.values()) == 100

    def test_gates_unchanged_by_measurement_annotations(self) -> None:
        qc_gates = [g.name for g, _t in _bell().gates]
        measured = _bell().measure_all()
        assert [g.name for g, _t in measured.gates] == qc_gates

    def test_circuit_concatenation_merges_measurements(self) -> None:
        a = QuantumCircuit(2).h(0).measure(0)
        b = QuantumCircuit(2).x(1).measure(1)
        combined = a + b
        assert combined.measurements == [0, 1]

    def test_bind_parameters_preserves_measurements(self) -> None:
        from microquantum import Parameter

        theta = Parameter("theta")
        qc = QuantumCircuit(1).rx(theta, 0).measure_all()
        bound = qc.bind_parameters({theta: 0.5})
        assert bound.measurements == [0]

    def test_json_roundtrip_preserves_measurements(self) -> None:
        qc = _bell().measure(0)
        restored = QuantumCircuit.from_json(qc.to_json())
        assert restored.measurements == [0]
        assert [g.name for g, _t in restored.gates] == ["h", "cnot"]

    def test_qasm_roundtrip_preserves_measurements(self) -> None:
        qc = _bell().measure_all()
        restored = QuantumCircuit.from_qasm(qc.qasm())
        assert restored.measurements == [0, 1]

    def test_mock_backend_measurement_restriction_consistent(self) -> None:
        qc = QuantumCircuit(3).measure(0).measure(2)
        mock = MockBackend()
        result = mock.run(qc, shots=100, seed=10)
        assert all(len(k) == 2 for k in result.counts)
        assert sum(result.counts.values()) == 100