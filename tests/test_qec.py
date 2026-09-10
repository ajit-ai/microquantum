"""Tests for quantum error correction codes."""

import pytest

from microquantum.qec import BitFlipCode, PhaseFlipCode, RepetitionCode, ShorCode


class TestRepetitionCode:
    def test_init(self):
        code = RepetitionCode(5)
        assert code.num_data_qubits == 5
        assert code.num_syndrome_qubits == 4
        assert code.total_qubits == 9
        assert code.distance == 5

    def test_odd_qubits_required(self):
        with pytest.raises(ValueError):
            RepetitionCode(4)

    def test_minimum_qubits(self):
        with pytest.raises(ValueError):
            RepetitionCode(2)

    def test_encode_circuit(self):
        code = RepetitionCode(3)
        qc = code.encode_circuit()
        assert qc.num_qubits == 3
        assert qc.num_gates == 2

    def test_syndrome_circuit(self):
        code = RepetitionCode(3)
        qc = code.syndrome_circuit()
        assert qc.num_qubits == 5

    def test_decode_syndrome_no_error(self):
        code = RepetitionCode(3)
        result = code.decode_syndrome([0, 0])
        assert result is None

    def test_decode_syndrome_error_qubit0(self):
        code = RepetitionCode(3)
        result = code.decode_syndrome([1, 0])
        assert result == 0

    def test_decode_syndrome_error_qubit1(self):
        code = RepetitionCode(3)
        result = code.decode_syndrome([1, 1])
        assert result == 1

    def test_decode_syndrome_error_qubit2(self):
        code = RepetitionCode(3)
        result = code.decode_syndrome([0, 1])
        assert result == 2

    def test_error_locations_none(self):
        code = RepetitionCode(3)
        locs = code.error_locations([0, 0])
        assert locs == []

    def test_error_locations_single(self):
        code = RepetitionCode(5)
        locs = code.error_locations([1, 0, 0, 0])
        assert 0 in locs

    def test_repr(self):
        code = RepetitionCode(3)
        assert "RepetitionCode" in repr(code)


class TestBitFlipCode:
    def test_init(self):
        code = BitFlipCode()
        assert code.num_data_qubits == 3
        assert code.num_syndrome_qubits == 2
        assert code.total_qubits == 5
        assert code.distance == 3

    def test_encode_circuit(self):
        code = BitFlipCode()
        qc = code.encode_circuit()
        assert qc.num_qubits == 3
        assert qc.num_gates == 2

    def test_decode_syndrome(self):
        code = BitFlipCode()
        assert code.decode_syndrome([0, 0]) is None
        assert code.decode_syndrome([1, 0]) == 0
        assert code.decode_syndrome([1, 1]) == 1
        assert code.decode_syndrome([0, 1]) == 2

    def test_repr(self):
        code = BitFlipCode()
        assert "BitFlipCode" in repr(code)


class TestPhaseFlipCode:
    def test_init(self):
        code = PhaseFlipCode()
        assert code.num_data_qubits == 3
        assert code.num_syndrome_qubits == 2
        assert code.total_qubits == 5
        assert code.distance == 3

    def test_encode_circuit(self):
        code = PhaseFlipCode()
        qc = code.encode_circuit()
        assert qc.num_qubits == 3
        assert qc.num_gates > 0

    def test_decode_syndrome(self):
        code = PhaseFlipCode()
        assert code.decode_syndrome([0, 0]) is None
        assert code.decode_syndrome([1, 0]) == 0
        assert code.decode_syndrome([1, 1]) == 1
        assert code.decode_syndrome([0, 1]) == 2

    def test_error_locations(self):
        code = PhaseFlipCode()
        assert code.error_locations([0, 0]) == []
        assert code.error_locations([1, 0]) == [0]

    def test_repr(self):
        code = PhaseFlipCode()
        assert "PhaseFlipCode" in repr(code)


class TestShorCode:
    def test_init(self):
        code = ShorCode()
        assert code.num_data_qubits == 9
        assert code.num_syndrome_qubits == 8
        assert code.total_qubits == 17
        assert code.distance == 3

    def test_encode_circuit(self):
        code = ShorCode()
        qc = code.encode_circuit()
        assert qc.num_qubits == 9
        assert qc.num_gates > 0

    def test_syndrome_circuit(self):
        code = ShorCode()
        qc = code.syndrome_circuit()
        assert qc.num_qubits == 17

    def test_decode_syndrome_no_error(self):
        code = ShorCode()
        result = code.decode_syndrome([0] * 8)
        assert result is None

    def test_decode_syndrome_x_error(self):
        code = ShorCode()
        # Bit-flip in block 1 at position 0
        syndrome = [1, 0, 0, 0, 0, 0, 0, 0]
        result = code.decode_syndrome(syndrome)
        assert result is not None
        qubit, error_type = result
        assert error_type == "X"
        assert qubit == 0

    def test_decode_syndrome_z_error(self):
        code = ShorCode()
        # Phase-flip on block 0
        syndrome = [0, 0, 0, 0, 0, 0, 1, 0]
        result = code.decode_syndrome(syndrome)
        assert result is not None
        qubit, error_type = result
        assert error_type == "Z"
        assert qubit == 0

    def test_decode_syndrome_y_error(self):
        code = ShorCode()
        # Y error on qubit 0 (both X and Z syndromes)
        syndrome = [1, 0, 0, 0, 0, 0, 1, 0]
        result = code.decode_syndrome(syndrome)
        assert result is not None
        qubit, error_type = result
        assert error_type == "Y"
        assert qubit == 0

    def test_repr(self):
        code = ShorCode()
        assert "ShorCode" in repr(code)
