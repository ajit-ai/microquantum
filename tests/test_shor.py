"""Tests for Shor's Algorithm."""

import pytest

from microquantum.algorithms.shor import ShorResult, ShorsAlgorithm


class TestShorsAlgorithm:
    def test_factor_15(self):
        algo = ShorsAlgorithm(15, seed=42)
        result = algo.run()
        assert isinstance(result, ShorResult)
        assert result.success
        p, q = result.factors
        assert p * q == 15
        assert p > 1 and q > 1

    def test_factor_21(self):
        algo = ShorsAlgorithm(21, seed=42)
        result = algo.run()
        p, q = result.factors
        assert p * q == 21

    def test_factor_9(self):
        algo = ShorsAlgorithm(9, seed=42)
        result = algo.run()
        p, q = result.factors
        assert p * q == 9

    def test_factor_35(self):
        algo = ShorsAlgorithm(35, seed=42)
        result = algo.run()
        p, q = result.factors
        assert p * q == 35

    def test_build_circuit(self):
        algo = ShorsAlgorithm(15, seed=42)
        qc = algo.build_circuit(a=7)
        assert qc.num_qubits > 0
        assert qc.num_gates > 0

    def test_validation_even(self):
        with pytest.raises(ValueError, match="odd"):
            ShorsAlgorithm(4)

    def test_validation_small(self):
        with pytest.raises(ValueError, match=">="):
            ShorsAlgorithm(1)

    def test_properties(self):
        algo = ShorsAlgorithm(15)
        assert algo.n == 15

    def test_repr(self):
        algo = ShorsAlgorithm(15)
        assert "ShorsAlgorithm" in repr(algo)

    def test_result_fields(self):
        algo = ShorsAlgorithm(15, seed=42)
        result = algo.run()
        assert hasattr(result, "n")
        assert hasattr(result, "factors")
        assert hasattr(result, "period")
        assert hasattr(result, "circuit")
        assert hasattr(result, "num_qubits")
