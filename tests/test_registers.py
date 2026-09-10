"""Tests for QuantumRegister and ClassicalRegister."""

import pytest

from microquantum.core.registers import ClassicalRegister, QuantumRegister


class TestQuantumRegister:
    """Test QuantumRegister initialization and properties."""

    def test_creation(self):
        """Create a quantum register."""
        qr = QuantumRegister("q", 3)
        assert qr.name == "q"
        assert qr.size == 3

    def test_qubit_indices(self):
        """Qubit indices are 0 to size-1."""
        qr = QuantumRegister("q", 4)
        assert qr.qubit_indices == [0, 1, 2, 3]

    def test_getitem(self):
        """Get qubit index by position."""
        qr = QuantumRegister("q", 3)
        assert qr[0] == 0
        assert qr[1] == 1
        assert qr[2] == 2

    def test_getitem_negative(self):
        """Negative index raises IndexError."""
        qr = QuantumRegister("q", 3)
        with pytest.raises(IndexError):
            qr[-1]

    def test_getitem_out_of_range(self):
        """Out of range index raises IndexError."""
        qr = QuantumRegister("q", 3)
        with pytest.raises(IndexError):
            qr[3]

    def test_iter(self):
        """Iterate over qubit indices."""
        qr = QuantumRegister("q", 3)
        assert list(qr) == [0, 1, 2]

    def test_len(self):
        """Length equals size."""
        qr = QuantumRegister("q", 5)
        assert len(qr) == 5

    def test_zero_size_raises(self):
        """Zero size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            QuantumRegister("q", 0)

    def test_negative_size_raises(self):
        """Negative size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            QuantumRegister("q", -1)

    def test_repr(self):
        """repr contains name and size."""
        qr = QuantumRegister("q", 3)
        assert "q" in repr(qr)
        assert "3" in repr(qr)

    def test_str(self):
        """str contains name and size."""
        qr = QuantumRegister("q", 3)
        assert "q" in str(qr)
        assert "3" in str(qr)


class TestClassicalRegister:
    """Test ClassicalRegister initialization and properties."""

    def test_creation(self):
        """Create a classical register."""
        cr = ClassicalRegister("c", 4)
        assert cr.name == "c"
        assert cr.size == 4

    def test_getitem(self):
        """Get bit index by position."""
        cr = ClassicalRegister("c", 3)
        assert cr[0] == 0
        assert cr[1] == 1
        assert cr[2] == 2

    def test_getitem_negative(self):
        """Negative index raises IndexError."""
        cr = ClassicalRegister("c", 3)
        with pytest.raises(IndexError):
            cr[-1]

    def test_getitem_out_of_range(self):
        """Out of range index raises IndexError."""
        cr = ClassicalRegister("c", 3)
        with pytest.raises(IndexError):
            cr[3]

    def test_iter(self):
        """Iterate over bit indices."""
        cr = ClassicalRegister("c", 3)
        assert list(cr) == [0, 1, 2]

    def test_len(self):
        """Length equals size."""
        cr = ClassicalRegister("c", 5)
        assert len(cr) == 5

    def test_zero_size_raises(self):
        """Zero size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ClassicalRegister("c", 0)

    def test_negative_size_raises(self):
        """Negative size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ClassicalRegister("c", -1)

    def test_repr(self):
        """repr contains name and size."""
        cr = ClassicalRegister("c", 4)
        assert "c" in repr(cr)
        assert "4" in repr(cr)

    def test_str(self):
        """str contains name and size."""
        cr = ClassicalRegister("c", 4)
        assert "c" in str(cr)
        assert "4" in str(cr)


class TestRegistersTogether:
    """Test registers used together."""

    def test_multiple_registers(self):
        """Create multiple registers with different names."""
        qr1 = QuantumRegister("q", 3)
        qr2 = QuantumRegister("r", 2)
        assert qr1.name != qr2.name
        assert qr1.size != qr2.size

    def test_register_indices_independent(self):
        """Registers have independent indices."""
        qr1 = QuantumRegister("q", 3)
        qr2 = QuantumRegister("r", 2)
        assert qr1[0] == 0
        assert qr2[0] == 0
