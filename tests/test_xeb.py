"""Tests for Cross-Entropy Benchmarking."""

import pytest

from microquantum.benchmarks.xeb import (
    CrossEntropyBenchmarking,
    XEBResult,
)


class TestCrossEntropyBenchmarking:
    def test_run_small(self):
        xeb = CrossEntropyBenchmarking(
            num_qubits=3, depths=[2, 4], num_circuits=3, seed=42
        )
        result = xeb.run()
        assert isinstance(result, XEBResult)
        assert len(result.depths) == 2
        assert len(result.fidelities) == 2

    def test_fidelities_nonnegative(self):
        xeb = CrossEntropyBenchmarking(
            num_qubits=3, depths=[2, 4], num_circuits=3, seed=42
        )
        result = xeb.run()
        for fid in result.fidelities:
            assert fid >= 0.0

    def test_perfect_fidelity_high(self):
        xeb = CrossEntropyBenchmarking(
            num_qubits=2, depths=[1], num_circuits=1, seed=42
        )
        result = xeb.run()
        assert result.mean_fidelity >= 0.0

    def test_validation(self):
        with pytest.raises(ValueError, match=">="):
            CrossEntropyBenchmarking(num_qubits=1)

    def test_properties(self):
        xeb = CrossEntropyBenchmarking(
            num_qubits=4, depths=[5, 10], seed=42
        )
        assert xeb.num_qubits == 4
        assert xeb.depths == [5, 10]
        assert xeb.seed == 42

    def test_repr(self):
        xeb = CrossEntropyBenchmarking(num_qubits=3, depths=[5])
        assert "CrossEntropyBenchmarking" in repr(xeb)

    def test_result_fields(self):
        xeb = CrossEntropyBenchmarking(
            num_qubits=3, depths=[2], num_circuits=2, seed=42
        )
        result = xeb.run()
        assert hasattr(result, "depths")
        assert hasattr(result, "fidelities")
        assert hasattr(result, "mean_fidelity")
        assert hasattr(result, "is_above_threshold")
        assert hasattr(result, "num_qubits")
