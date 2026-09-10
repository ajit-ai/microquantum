"""Tests for QASM export/import, circuit visualization, and JSON serialization."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from microquantum.core.circuit import QuantumCircuit
from microquantum.core.operators import Operator


# ======================================================================
# QASM Tests
# ======================================================================

class TestQASMExport:
    def test_empty_circuit(self):
        from microquantum.core.qasm import to_qasm

        qc = QuantumCircuit(2)
        qasm = to_qasm(qc)
        assert "OPENQASM 2.0" in qasm
        assert "qreg q[2]" in qasm
        assert "creg c[2]" in qasm

    def test_h_gate(self):
        from microquantum.core.qasm import to_qasm

        qc = QuantumCircuit(1)
        qc.h(0)
        qasm = to_qasm(qc)
        assert "h q[0];" in qasm

    def test_x_gate(self):
        from microquantum.core.qasm import to_qasm

        qc = QuantumCircuit(1)
        qc.x(0)
        qasm = to_qasm(qc)
        assert "x q[0];" in qasm

    def test_cnot_gate(self):
        from microquantum.core.qasm import to_qasm

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qasm = to_qasm(qc)
        assert "cx q[0], q[1];" in qasm

    def test_rotation_gate(self):
        from microquantum.core.qasm import to_qasm

        qc = QuantumCircuit(1)
        qc.ry(math.pi / 4, 0)
        qasm = to_qasm(qc)
        assert "ry(" in qasm
        assert "q[0];" in qasm

    def test_no_header(self):
        from microquantum.core.qasm import to_qasm

        qc = QuantumCircuit(1)
        qc.h(0)
        qasm = to_qasm(qc, header=False)
        assert "OPENQASM" not in qasm
        assert "h q[0];" in qasm

    def test_bell_circuit(self):
        from microquantum.core.qasm import to_qasm

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qasm = to_qasm(qc)
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm


class TestQASMImport:
    def test_import_simple(self):
        from microquantum.core.qasm import from_qasm

        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
"""
        qc = from_qasm(qasm)
        assert qc.num_qubits == 1
        assert qc.num_gates == 1

    def test_import_bell(self):
        from microquantum.core.qasm import from_qasm

        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
"""
        qc = from_qasm(qasm)
        assert qc.num_qubits == 2
        assert qc.num_gates == 2

    def test_import_with_rotation(self):
        from microquantum.core.qasm import from_qasm

        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
ry(1.570796) q[0];
"""
        qc = from_qasm(qasm)
        assert qc.num_qubits == 1
        assert qc.num_gates == 1

    def test_import_no_header(self):
        from microquantum.core.qasm import from_qasm

        qasm = """qreg q[2];
h q[0];
cx q[0], q[1];
"""
        qc = from_qasm(qasm)
        assert qc.num_qubits == 2
        assert qc.num_gates == 2

    def test_import_with_comments(self):
        from microquantum.core.qasm import from_qasm

        qasm = """// This is a comment
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
// Another comment
h q[0];
"""
        qc = from_qasm(qasm)
        assert qc.num_qubits == 1
        assert qc.num_gates == 1


class TestQASMRoundtrip:
    def test_roundtrip_bell(self):
        from microquantum.core.qasm import from_qasm, to_qasm

        original = QuantumCircuit(2)
        original.h(0)
        original.cx(0, 1)

        qasm = to_qasm(original)
        reconstructed = from_qasm(qasm)

        assert reconstructed.num_qubits == original.num_qubits
        assert reconstructed.num_gates == original.num_gates

        state_orig = original.run()
        state_recon = reconstructed.run()
        assert np.allclose(state_orig.amplitudes, state_recon.amplitudes, atol=1e-10)

    def test_roundtrip_rotation(self):
        from microquantum.core.qasm import from_qasm, to_qasm

        original = QuantumCircuit(1)
        original.ry(math.pi / 3, 0)

        qasm = to_qasm(original)
        reconstructed = from_qasm(qasm)

        state_orig = original.run()
        state_recon = reconstructed.run()
        assert np.allclose(state_orig.amplitudes, state_recon.amplitudes, atol=1e-4)


# ======================================================================
# Visualization Tests
# ======================================================================

class TestVisualization:
    def test_empty_circuit(self):
        from microquantum.core.visualization import draw

        qc = QuantumCircuit(1)
        result = draw(qc)
        assert "q[0]" in result

    def test_single_gate(self):
        from microquantum.core.visualization import draw

        qc = QuantumCircuit(1)
        qc.h(0)
        result = draw(qc)
        assert "H" in result
        assert "q[0]" in result

    def test_bell_circuit(self):
        from microquantum.core.visualization import draw

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = draw(qc)
        assert "H" in result
        assert "q[0]" in result
        assert "q[1]" in result

    def test_with_title(self):
        from microquantum.core.visualization import draw

        qc = QuantumCircuit(1)
        qc.h(0)
        result = draw(qc, title="My Circuit")
        assert "My Circuit" in result

    def test_x_gate(self):
        from microquantum.core.visualization import draw

        qc = QuantumCircuit(1)
        qc.x(0)
        result = draw(qc)
        assert "X" in result

    def test_multiple_qubits(self):
        from microquantum.core.visualization import draw

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        result = draw(qc)
        for i in range(4):
            assert f"q[{i}]" in result

    def test_text_alias(self):
        from microquantum.core.visualization import text

        qc = QuantumCircuit(1)
        qc.h(0)
        result = text(qc)
        assert "H" in result


# ======================================================================
# Serialization Tests
# ======================================================================

class TestSerializationDict:
    def test_empty_circuit(self):
        from microquantum.core.serialization import to_dict, from_dict

        qc = QuantumCircuit(2)
        data = to_dict(qc)
        assert data["num_qubits"] == 2
        assert data["num_gates"] == 0
        assert data["gates"] == []

    def test_single_gate(self):
        from microquantum.core.serialization import to_dict, from_dict

        qc = QuantumCircuit(1)
        qc.h(0)
        data = to_dict(qc)
        assert len(data["gates"]) == 1
        assert data["gates"][0]["name"] == "h"

    def test_roundtrip(self):
        from microquantum.core.serialization import to_dict, from_dict

        original = QuantumCircuit(2)
        original.h(0)
        original.cx(0, 1)
        original.ry(math.pi / 4, 1)

        data = to_dict(original)
        reconstructed = from_dict(data)

        assert reconstructed.num_qubits == original.num_qubits
        assert reconstructed.num_gates == original.num_gates

        state_orig = original.run()
        state_recon = reconstructed.run()
        assert np.allclose(state_orig.amplitudes, state_recon.amplitudes, atol=1e-10)

    def test_format_version(self):
        from microquantum.core.serialization import to_dict, FORMAT_VERSION

        qc = QuantumCircuit(1)
        data = to_dict(qc)
        assert data["format_version"] == FORMAT_VERSION


class TestSerializationJSON:
    def test_json_roundtrip(self):
        from microquantum.core.serialization import to_json, from_json

        original = QuantumCircuit(2)
        original.h(0)
        original.cx(0, 1)

        json_str = to_json(original)
        data = json.loads(json_str)
        assert "gates" in data

        reconstructed = from_json(json_str)
        assert reconstructed.num_qubits == 2
        assert reconstructed.num_gates == 2

    def test_json_contains_all_fields(self):
        from microquantum.core.serialization import to_json

        qc = QuantumCircuit(1)
        qc.h(0)
        json_str = to_json(qc)
        data = json.loads(json_str)
        assert "format_version" in data
        assert "num_qubits" in data
        assert "num_gates" in data
        assert "depth" in data
        assert "gates" in data


class TestSerializationFile:
    def test_save_load(self):
        from microquantum.core.serialization import save, load

        original = QuantumCircuit(2)
        original.h(0)
        original.cx(0, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_circuit.json"
            save(original, path)
            assert path.exists()

            loaded = load(path)
            assert loaded.num_qubits == 2
            assert loaded.num_gates == 2

    def test_save_creates_directories(self):
        from microquantum.core.serialization import save, load

        qc = QuantumCircuit(1)
        qc.h(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "circuit.json"
            save(qc, path)
            assert path.exists()

    def test_load_nonexistent_raises(self):
        from microquantum.core.serialization import load

        with pytest.raises((FileNotFoundError, OSError)):
            load("/nonexistent/path/circuit.json")


class TestSerializationGates:
    def test_rotation_gates_preserve_angle(self):
        from microquantum.core.serialization import to_dict, from_dict

        qc = QuantumCircuit(1)
        qc.ry(math.pi / 3, 0)

        data = to_dict(qc)
        assert "angle" in data["gates"][0]
        assert abs(data["gates"][0]["angle"] - math.pi / 3) < 0.01

        reconstructed = from_dict(data)
        state_orig = qc.run()
        state_recon = reconstructed.run()
        assert np.allclose(state_orig.amplitudes, state_recon.amplitudes, atol=1e-10)

    def test_swap_gate(self):
        from microquantum.core.serialization import to_dict, from_dict

        qc = QuantumCircuit(2)
        qc.swap(0, 1)

        data = to_dict(qc)
        reconstructed = from_dict(data)
        assert reconstructed.num_gates == 1

    def test_all_standard_gates(self):
        from microquantum.core.serialization import to_dict, from_dict

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.cx(0, 1)
        qc.swap(1, 2)
        qc.ry(0.5, 2)

        data = to_dict(qc)
        reconstructed = from_dict(data)
        assert reconstructed.num_gates == 5
