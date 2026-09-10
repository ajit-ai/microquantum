"""Tests for executor, QFT, phase estimation, dynamic circuits, and resources."""

import numpy as np
import pytest

from microquantum.algorithms.phase_estimation import PhaseEstimation, PhaseEstimationResult
from microquantum.algorithms.qft import QFT, inverse_qft_circuit, qft_circuit
from microquantum.backends.executor import Executor, ExecutorResult
from microquantum.core import Operator, QuantumCircuit, StateVector
from microquantum.core.dynamic import ClassicalRegister, DynamicCircuit, DynamicCircuitResult
from microquantum.core.resources import ResourceEstimator

# ── Executor ──────────────────────────────────────────────────────────

class TestExecutor:
    def test_run_noiseless(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        ex = Executor(seed=42)
        result = ex.run(qc, shots=1000)
        assert isinstance(result, ExecutorResult)
        assert result.shots == 1000
        assert result.num_qubits == 2

    def test_result_probabilities_sum_to_one(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        ex = Executor(seed=42)
        result = ex.run(qc, shots=10000)
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_result_most_frequent(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        ex = Executor(seed=42)
        result = ex.run(qc, shots=100)
        assert result.most_frequent() == "1"

    def test_result_expectation(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        ex = Executor(seed=42)
        result = ex.run(qc, shots=1000)
        exp_val = result.expectation(Operator.Z())
        assert exp_val == pytest.approx(-1.0, abs=0.05)

    def test_run_batch(self):
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        ex = Executor(seed=42)
        results = ex.run_batch([qc1, qc2], shots=100)
        assert len(results) == 2
        assert results[1].most_frequent() == "1"

    def test_run_and_average(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        ex = Executor(seed=42)
        result = ex.run_and_average([qc, qc], shots=1000)
        assert isinstance(result, ExecutorResult)


# ── Resource Estimation ───────────────────────────────────────────────

class TestResourceEstimator:
    def test_empty_circuit(self):
        qc = QuantumCircuit(2)
        est = ResourceEstimator()
        r = est.estimate(qc)
        assert r.total_gates == 0
        assert r.num_qubits == 2

    def test_single_qubit_gates(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        est = ResourceEstimator()
        r = est.estimate(qc)
        assert r.single_qubit_gates == 2
        assert r.two_qubit_gates == 0

    def test_two_qubit_gates(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        est = ResourceEstimator()
        r = est.estimate(qc)
        assert r.two_qubit_gates == 1

    def test_cnot_count(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        est = ResourceEstimator()
        r = est.estimate(qc)
        assert r.cnot_count == 3

    def test_circuit_volume(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        est = ResourceEstimator()
        r = est.estimate(qc)
        assert r.circuit_volume == r.num_qubits * r.depth

    def test_gate_type_counts(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        est = ResourceEstimator()
        r = est.estimate(qc)
        assert r.gate_type_counts.get("h", 0) == 2
        assert r.gate_type_counts.get("cnot", 0) == 1

    def test_execution_time(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        est = ResourceEstimator()
        r = est.estimate(qc)
        assert r.estimated_time_ns > 0


# ── QFT ──────────────────────────────────────────────────────────────

class TestQFT:
    def test_qft_circuit_gates(self):
        qc = qft_circuit(3)
        assert qc.num_qubits == 3
        assert qc.num_gates > 0

    def test_qft_forward_inverse_roundtrip(self):
        qft = QFT(3)
        qc_fwd = qft.build_circuit()
        qc_inv = qft.build_inverse_circuit()
        qc_both = qc_fwd + qc_inv
        state = qc_both.run()
        expected = StateVector(3)
        assert np.allclose(state.amplitudes, expected.amplitudes, atol=1e-10)

    def test_qft_class_roundtrip(self):
        qft = QFT(2)
        sv = StateVector(2)
        transformed = qft.run_forward(sv)
        recovered = qft.run_inverse(transformed)
        assert np.allclose(sv.amplitudes, recovered.amplitudes, atol=1e-10)

    def test_qft_no_swaps(self):
        qc = qft_circuit(2, do_swaps=False)
        assert qc.num_qubits == 2

    def test_inverse_qft_circuit(self):
        qc = inverse_qft_circuit(3)
        assert qc.num_qubits == 3

    def test_qft_repr(self):
        qft = QFT(3)
        assert "QFT" in repr(qft)


# ── Phase Estimation ──────────────────────────────────────────────────

class TestPhaseEstimation:
    def test_estimate_z_gate(self):
        pe = PhaseEstimation(Operator.Z(), num_counting_qubits=6)
        result = pe.estimate()
        assert isinstance(result, PhaseEstimationResult)
        assert result.phase == pytest.approx(0.5)
        assert result.success_probability == pytest.approx(1.0)

    def test_estimate_s_gate(self):
        pe = PhaseEstimation(Operator.S(), num_counting_qubits=8)
        result = pe.estimate()
        assert result.phase == pytest.approx(0.25)
        assert result.success_probability == pytest.approx(1.0)

    def test_eigenvalue(self):
        pe = PhaseEstimation(Operator.Z(), num_counting_qubits=6)
        result = pe.estimate()
        expected_eigenvalue = np.exp(2j * np.pi * 0.5)
        assert abs(result.eigenvalue - expected_eigenvalue) < 1e-10

    def test_phase_radians(self):
        pe = PhaseEstimation(Operator.Z(), num_counting_qubits=6)
        result = pe.estimate()
        assert result.phase_radians == pytest.approx(np.pi)

    def test_repr(self):
        pe = PhaseEstimation(Operator.Z(), num_counting_qubits=4)
        assert "PhaseEstimation" in repr(pe)


# ── Dynamic Circuits ──────────────────────────────────────────────────

class TestDynamicCircuit:
    def test_creation(self):
        dc = DynamicCircuit(2, 2)
        assert dc.num_qubits == 2
        assert dc.num_classical_bits == 2

    def test_gates(self):
        dc = DynamicCircuit(2)
        dc.h(0)
        dc.cx(0, 1)
        assert dc.num_gates == 2

    def test_mid_circuit_measurement(self):
        dc = DynamicCircuit(1, 1)
        dc.h(0)
        dc.measure(0, 0)
        result = dc.run(seed=42)
        assert isinstance(result, DynamicCircuitResult)
        assert 0 in result.classical_memory

    def test_classical_register(self):
        cr = ClassicalRegister(4)
        assert len(cr) == 4
        cr.write(0, 1)
        cr.write(3, 1)
        assert cr.read(0) == 1
        assert cr.read(3) == 1
        assert cr.read(1) == 0

    def test_measure_all(self):
        dc = DynamicCircuit(2, 2)
        dc.x(0)
        dc.x(1)
        dc.measure_all()
        result = dc.run(seed=42)
        assert result.classical_memory[0] == 1
        assert result.classical_memory[1] == 1

    def test_dynamic_result_fields(self):
        dc = DynamicCircuit(1, 1)
        dc.h(0)
        dc.measure(0, 0)
        result = dc.run(seed=42)
        assert result.final_state.num_qubits == 1
        assert isinstance(result.classical_memory, dict)
