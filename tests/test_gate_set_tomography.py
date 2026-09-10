"""Tests for Gate Set Tomography."""

import numpy as np
import pytest

from microquantum.benchmarks.gate_set_tomography import (
    GateSetTomography,
    GSTResult,
)
from microquantum.core import Operator


class TestGateSetTomography:
    def test_identity_gate(self):
        gst = GateSetTomography([Operator.I()])
        result = gst.run()
        assert isinstance(result, GSTResult)
        assert len(result.gate_names) == 1
        assert result.gate_fidelities[0] == pytest.approx(1.0, abs=0.01)

    def test_hadamard_gate(self):
        gst = GateSetTomography([Operator.H()])
        result = gst.run()
        assert len(result.chi_matrices) == 1
        assert result.chi_matrices[0].shape == (4, 4)

    def test_multiple_gates(self):
        gst = GateSetTomography([Operator.X(), Operator.Y(), Operator.Z()])
        result = gst.run()
        assert len(result.gate_names) == 3
        assert len(result.gate_fidelities) == 3

    def test_average_fidelity(self):
        gst = GateSetTomography([Operator.X(), Operator.H()])
        result = gst.run()
        assert 0.0 <= result.average_fidelity <= 1.0

    def test_chi_matrix_shape(self):
        gst = GateSetTomography([Operator.X()])
        result = gst.run()
        assert result.chi_matrices[0].shape == (4, 4)
        # Chi matrix should be 4x4 with finite values
        chi = result.chi_matrices[0]
        assert np.all(np.isfinite(chi))
        assert np.linalg.norm(chi) > 0.0

    def test_properties(self):
        gst = GateSetTomography([Operator.H(), Operator.X()])
        assert gst.num_gates == 2

    def test_repr(self):
        gst = GateSetTomography([Operator.H()])
        assert "GateSetTomography" in repr(gst)

    def test_result_fields(self):
        gst = GateSetTomography([Operator.X()])
        result = gst.run()
        assert hasattr(result, "gate_names")
        assert hasattr(result, "chi_matrices")
        assert hasattr(result, "gate_fidelities")
        assert hasattr(result, "average_fidelity")
