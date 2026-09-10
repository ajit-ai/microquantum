"""Tests for QuantumCircuit convenience I/O/display methods."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from microquantum.core.circuit import QuantumCircuit


class TestQASMConvenience:
    def test_qasm_export(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qasm = qc.qasm()
        assert "OPENQASM 2.0" in qasm
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm

    def test_qasm_no_header(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qasm = qc.qasm(header=False)
        assert "OPENQASM" not in qasm
        assert "h q[0];" in qasm

    def test_from_qasm(self):
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""
        qc = QuantumCircuit.from_qasm(qasm)
        assert qc.num_qubits == 2
        assert qc.num_gates == 2

    def test_roundtrip(self):
        original = QuantumCircuit(2)
        original.h(0)
        original.cx(0, 1)
        original.ry(math.pi / 4, 1)

        qasm = original.qasm()
        reconstructed = QuantumCircuit.from_qasm(qasm)

        assert reconstructed.num_qubits == original.num_qubits
        assert reconstructed.num_gates == original.num_gates

        state_orig = original.run()
        state_recon = reconstructed.run()
        assert np.allclose(state_orig.amplitudes, state_recon.amplitudes, atol=1e-10)


class TestVisualizationConvenience:
    def test_draw(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = qc.draw()
        assert "H" in result
        assert "q[0]" in result

    def test_draw_with_title(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        result = qc.draw(title="Bell Prep")
        assert "Bell Prep" in result
        assert "H" in result


class TestSerializationConvenience:
    def test_to_json(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        json_str = qc.to_json()
        data = json.loads(json_str)
        assert data["num_qubits"] == 2
        assert data["num_gates"] == 2

    def test_from_json(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        json_str = qc.to_json()
        reconstructed = QuantumCircuit.from_json(json_str)
        assert reconstructed.num_qubits == 2
        assert reconstructed.num_gates == 2

    def test_json_roundtrip(self):
        original = QuantumCircuit(2)
        original.h(0)
        original.cx(0, 1)
        original.ry(math.pi / 3, 0)

        reconstructed = QuantumCircuit.from_json(original.to_json())
        state_orig = original.run()
        state_recon = reconstructed.run()
        assert np.allclose(state_orig.amplitudes, state_recon.amplitudes, atol=1e-10)

    def test_save_load(self):
        original = QuantumCircuit(2)
        original.h(0)
        original.cx(0, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "circuit.json"
            original.save(path)
            assert path.exists()

            loaded = QuantumCircuit.load(path)
            assert loaded.num_qubits == 2
            assert loaded.num_gates == 2

    def test_save_load_roundtrip(self):
        original = QuantumCircuit(2)
        original.h(0)
        original.cx(0, 1)
        original.ry(0.5, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_circuit.json"
            original.save(path)
            loaded = QuantumCircuit.load(path)

            state_orig = original.run()
            state_loaded = loaded.run()
            assert np.allclose(state_orig.amplitudes, state_loaded.amplitudes, atol=1e-10)
