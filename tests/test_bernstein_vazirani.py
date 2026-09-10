"""Tests for Bernstein-Vazirani Algorithm."""

import pytest

from microquantum.algorithms.bernstein_vazirani import (
    BernsteinVazirani,
    BVResult,
)


class TestBernsteinVazirani:
    def test_secret_1011(self):
        bv = BernsteinVazirani("1011")
        result = bv.run()
        assert isinstance(result, BVResult)
        assert result.measured == [1, 0, 1, 1]
        assert result.correct

    def test_secret_111(self):
        bv = BernsteinVazirani("111")
        result = bv.run()
        assert result.measured == [1, 1, 1]
        assert result.correct

    def test_secret_001(self):
        bv = BernsteinVazirani("001")
        result = bv.run()
        assert result.measured == [0, 0, 1]
        assert result.correct

    def test_secret_single_bit(self):
        bv = BernsteinVazirani("1")
        result = bv.run()
        assert result.measured == [1]
        assert result.correct

    def test_list_input(self):
        bv = BernsteinVazirani([1, 0, 1])
        result = bv.run()
        assert result.measured == [1, 0, 1]
        assert result.correct

    def test_build_circuit(self):
        bv = BernsteinVazirani("101")
        qc = bv.build_circuit()
        assert qc.num_qubits == 4  # 3 + 1 ancilla

    def test_validation_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            BernsteinVazirani("")

    def test_validation_bad_chars(self):
        with pytest.raises(ValueError, match="0s and 1s"):
            BernsteinVazirani("102")

    def test_properties(self):
        bv = BernsteinVazirani("1011")
        assert bv.secret_string == [1, 0, 1, 1]
        assert bv.num_qubits == 5

    def test_repr(self):
        bv = BernsteinVazirani("1011")
        assert "BernsteinVazirani" in repr(bv)
