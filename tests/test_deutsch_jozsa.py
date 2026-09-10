"""Tests for Deutsch-Jozsa Algorithm."""

import pytest

from microquantum.algorithms.deutsch_jozsa import DJResult, DeutschJozsa


class TestDeutschJozsa:
    def test_balanced_2q(self):
        dj = DeutschJozsa(n_qubits=2, balanced=True)
        result = dj.run()
        assert isinstance(result, DJResult)
        assert result.is_constant is False

    def test_constant_2q(self):
        dj = DeutschJozsa(n_qubits=2, balanced=False)
        result = dj.run()
        assert result.is_constant is True

    def test_balanced_3q(self):
        dj = DeutschJozsa(n_qubits=3, balanced=True)
        result = dj.run()
        assert result.is_constant is False

    def test_constant_3q(self):
        dj = DeutschJozsa(n_qubits=3, balanced=False)
        result = dj.run()
        assert result.is_constant is True

    def test_build_oracle(self):
        dj = DeutschJozsa(n_qubits=2, balanced=True)
        oracle = dj.build_oracle()
        assert oracle.num_qubits == 3
        # Balanced oracle should have CNOTs
        assert oracle.num_gates > 0

    def test_build_circuit(self):
        dj = DeutschJozsa(n_qubits=3, balanced=True)
        qc = dj.build_circuit()
        assert qc.num_qubits == 4

    def test_validation(self):
        with pytest.raises(ValueError, match=">="):
            DeutschJozsa(n_qubits=0)

    def test_properties(self):
        dj = DeutschJozsa(n_qubits=3, balanced=True)
        assert dj.num_qubits == 3

    def test_repr(self):
        dj = DeutschJozsa(n_qubits=2, balanced=True)
        assert "DeutschJozsa" in repr(dj)
