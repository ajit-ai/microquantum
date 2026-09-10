"""Tests for benchmarks module."""
import pytest

from microquantum.benchmarks.clops import CLOPSBenchmark
from microquantum.benchmarks.quantum_volume import (
    BenchmarkResult,
    QuantumVolumeBenchmark,
)


class TestQuantumVolumeBenchmark:
    def test_creation(self):
        qvb = QuantumVolumeBenchmark(max_qubits=5)
        assert qvb.max_qubits == 5

    def test_invalid_qubits(self):
        with pytest.raises(ValueError, match="Need >= 1 qubit"):
            QuantumVolumeBenchmark(max_qubits=0)

    def test_generate_circuit(self):
        qvb = QuantumVolumeBenchmark(max_qubits=5, seed=42)
        qc = qvb.generate_random_circuit(3)
        assert qc.num_qubits == 3

    def test_heavy_output(self):
        qvb = QuantumVolumeBenchmark(max_qubits=5, seed=42)
        qc = qvb.generate_random_circuit(2)
        state = qc.run()
        probs = {format(i, f"0{qc.num_qubits}b"): float(abs(state.amplitudes[i]) ** 2)
                 for i in range(2**qc.num_qubits)}
        hop, heavy = qvb.compute_heavy_output(qc, probs)
        assert hop > 0
        assert len(heavy) > 0

    def test_run_single(self):
        qvb = QuantumVolumeBenchmark(max_qubits=5, seed=42)
        result = qvb.run_single(2)
        assert isinstance(result, BenchmarkResult)
        assert result.metric_name == "quantum_volume"

    def test_run(self):
        qvb = QuantumVolumeBenchmark(max_qubits=4, seed=42)
        result = qvb.run()
        assert isinstance(result, BenchmarkResult)
        assert result.value >= 0

    def test_repr(self):
        qvb = QuantumVolumeBenchmark(max_qubits=5)
        assert "QuantumVolumeBenchmark" in repr(qvb)


class TestCLOPSBenchmark:
    def test_creation(self):
        bench = CLOPSBenchmark(num_qubits=3, num_layers=5, num_circuits=10)
        assert bench._num_qubits == 3
        assert bench._num_layers == 5
        assert bench._num_circuits == 10

    def test_invalid_qubits(self):
        with pytest.raises(ValueError, match="Need >= 1 qubit"):
            CLOPSBenchmark(num_qubits=0)

    def test_invalid_layers(self):
        with pytest.raises(ValueError, match="Need >= 1 layer"):
            CLOPSBenchmark(num_qubits=3, num_layers=0)

    def test_invalid_circuits(self):
        with pytest.raises(ValueError, match="Need >= 1 circuit"):
            CLOPSBenchmark(num_qubits=3, num_layers=5, num_circuits=0)

    def test_generate_circuit(self):
        bench = CLOPSBenchmark(num_qubits=3, num_layers=5, seed=42)
        qc = bench.generate_circuit()
        assert qc.num_qubits == 3

    def test_run(self):
        bench = CLOPSBenchmark(num_qubits=2, num_layers=3, num_circuits=5, seed=42)
        result = bench.run()
        assert isinstance(result, BenchmarkResult)
        assert result.metric_name == "clops"
        assert result.value > 0

    def test_repr(self):
        bench = CLOPSBenchmark(num_qubits=3, num_layers=5)
        assert "CLOPSBenchmark" in repr(bench)


class TestBenchmarkResult:
    def test_default(self):
        r = BenchmarkResult(metric_name="test", value=1.0)
        assert r.metric_name == "test"
        assert r.value == 1.0
        assert r.raw_data == {}
