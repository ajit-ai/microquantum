"""Tests for Randomized Benchmarking."""

import numpy as np

from microquantum.benchmarks.randomized_benchmarking import (
    RandomizedBenchmarking,
    RandomizedBenchmarkingResult,
)


class TestRandomizedBenchmarking:
    def test_clifford_group_generation(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        group = rb.generate_clifford_group()
        assert len(group) > 0
        for qc in group:
            assert qc.num_qubits == 1
            assert qc.num_gates > 0

    def test_clifford_group_size(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        group = rb.generate_clifford_group()
        assert len(group) == 24

    def test_generate_sequence(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        qc = rb.generate_sequence(length=5)
        assert qc.num_qubits == 1
        assert qc.num_gates > 0

    def test_identity_sequence(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        qc = rb.generate_sequence(length=10)
        sv = qc.run()
        sv0 = np.zeros(2, dtype=np.complex128)
        sv0[0] = 1.0
        assert np.allclose(sv.amplitudes, sv0, atol=1e-6)

    def test_run(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        result = rb.run(sequence_lengths=[1, 2, 4], num_samples=10)
        assert isinstance(result, RandomizedBenchmarkingResult)
        assert len(result.num_cliffords) == 3
        assert len(result.survival_probabilities) == 3

    def test_run_fidelity(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        result = rb.run(sequence_lengths=[1, 2, 4, 8], num_samples=10)
        assert 0.0 <= result.average_gate_fidelity <= 1.0
        assert 0.0 <= result.error_per_gate <= 1.0

    def test_perfect_gates_high_fidelity(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        result = rb.run(sequence_lengths=[1, 2, 4, 8, 16], num_samples=15)
        assert result.average_gate_fidelity > 0.9

    def test_depolarizing_parameter(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        result = rb.run(sequence_lengths=[1, 2, 4, 8], num_samples=10)
        assert 0.0 <= result.depolarizing_parameter <= 1.0

    def test_repr(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        assert "RandomizedBenchmarking" in repr(rb)

    def test_properties(self):
        rb = RandomizedBenchmarking(num_qubits=1, seed=42)
        assert rb.num_qubits == 1
        assert rb.seed == 42
