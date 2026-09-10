"""Tests for gradient computation via Parameter-Shift Rule."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.core import (
    Operator,
    Parameter,
    QuantumCircuit,
    gradient,
    parameter_shift_gradient,
)


class TestSingleParameterGradient:
    """Gradients on single-parameter circuits."""

    def test_ry_gradient_at_zero(self) -> None:
        """d/d theta <Z> = -sin(theta); at theta=0, derivative = 0."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        grad_val = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: 0.0}
        )
        assert grad_val == pytest.approx(-np.sin(0.0), abs=1e-10)

    def test_ry_gradient_at_pi4(self) -> None:
        """d/d theta <Z> = -sin(theta); at theta=pi/4."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        val = np.pi / 4
        grad_val = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: val}
        )
        expected = -np.sin(val)
        assert grad_val == pytest.approx(expected, abs=1e-10)

    def test_ry_gradient_at_pi(self) -> None:
        """d/d theta <Z> = -sin(theta); at theta=pi, derivative = 0."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        grad_val = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: np.pi}
        )
        assert grad_val == pytest.approx(0.0, abs=1e-10)

    def test_rx_gradient_at_pi4(self) -> None:
        """RX gradient: d/d theta <Z> = -sin(theta)."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).rx(theta, 0)
        val = np.pi / 4
        grad_val = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: val}
        )
        expected = -np.sin(val)
        assert grad_val == pytest.approx(expected, abs=1e-10)

    def test_rx_gradient_at_zero(self) -> None:
        """RX at theta=0: derivative = 0."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).rx(theta, 0)
        grad_val = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: 0.0}
        )
        assert grad_val == pytest.approx(0.0, abs=1e-10)

    def test_rz_gradient_expectation(self) -> None:
        """RZ on |0> produces Z expectation = 1 for all theta (global phase)."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).rz(theta, 0)
        grad_val = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: np.pi / 4}
        )
        assert grad_val == pytest.approx(0.0, abs=1e-10)


class TestMultiParameterGradient:
    """Gradients on multi-parameter circuits."""

    def test_two_parameter_gradient(self) -> None:
        """RY(theta1) on q0, RX(theta2) on q1, observable Z on q0."""
        theta1 = Parameter("theta1")
        theta2 = Parameter("theta2")
        qc = QuantumCircuit(2).ry(theta1, 0).rx(theta2, 1)
        obs = Operator.Z()
        params = {theta1: np.pi / 4, theta2: np.pi / 3}

        grad_vals = gradient(qc, obs, params, targets=[0])
        assert len(grad_vals) == 2

        # Z only acts on qubit 0, so gradient w.r.t. theta1 depends on RY
        # Ry(theta)|0> => <Z> = cos(theta), d/dtheta = -sin(theta)
        expected_theta1 = -np.sin(params[theta1])
        assert grad_vals[theta1] == pytest.approx(expected_theta1, abs=1e-10)

        # Z on q0 only, gradient w.r.t. theta2 (RX on q1) = 0
        assert grad_vals[theta2] == pytest.approx(0.0, abs=1e-10)


class TestShiftValues:
    """Custom shift values."""

    def test_custom_shift_pi6(self) -> None:
        """Parameter-shift with custom shift s = pi/6."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        val = np.pi / 4
        shift = np.pi / 6
        grad_val = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: val}, shift=shift
        )
        expected = -np.sin(val)
        assert grad_val == pytest.approx(expected, abs=1e-10)

    def test_shift_zero_raises(self) -> None:
        """Shift of 0 is invalid."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(ValueError, match="sin"):
            parameter_shift_gradient(
                qc, Operator.Z(), theta, {theta: 0.0}, shift=0.0
            )

    def test_shift_pi_raises(self) -> None:
        """Shift of pi is invalid (sin(pi) ~ 0)."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(ValueError, match="sin"):
            parameter_shift_gradient(
                qc, Operator.Z(), theta, {theta: 0.0}, shift=np.pi
            )


class TestAnalyticalAgreement:
    """Verify gradient matches exact mathematical derivatives."""

    @pytest.mark.parametrize("val", [0.0, 0.5, 1.0, np.pi / 4, np.pi / 2])
    def test_ry_z_expectation_gradient(self, val: float) -> None:
        """<Z> = cos(theta), d/dtheta = -sin(theta)."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        grad_val = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: val}
        )
        expected = -np.sin(val)
        assert grad_val == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("val", [0.0, 0.5, 1.0, np.pi / 4, np.pi / 2])
    def test_rx_z_expectation_gradient(self, val: float) -> None:
        """<Z> = cos(theta), d/dtheta = -sin(theta)."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).rx(theta, 0)
        grad_val = parameter_shift_gradient(
            qc, Operator.Z(), theta, {theta: val}
        )
        expected = -np.sin(val)
        assert grad_val == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("val", [0.0, 0.5, 1.0, np.pi / 4, np.pi / 2])
    def test_ry_x_expectation_gradient(self, val: float) -> None:
        """<X> = sin(theta), d/dtheta = cos(theta)."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        grad_val = parameter_shift_gradient(
            qc, Operator.X(), theta, {theta: val}
        )
        expected = np.cos(val)
        assert grad_val == pytest.approx(expected, abs=1e-10)


class TestEdgeCases:
    """Edge cases and validation."""

    def test_missing_param_in_values(self) -> None:
        """Parameter not in param_values should raise KeyError."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(KeyError):
            parameter_shift_gradient(qc, Operator.Z(), theta, {phi: 1.0})

    def test_circuit_without_parameter(self) -> None:
        """Gradient on non-parameterized circuit should return empty dict."""
        qc = QuantumCircuit(1).h(0)
        grad_vals = gradient(qc, Operator.Z(), {})
        assert len(grad_vals) == 0

    def test_gradient_returns_all_params(self) -> None:
        """gradient() should return entries for all parameters."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        grad_vals = gradient(
            qc, Operator.Z(), {theta: 0.5, phi: 0.3}, targets=[0]
        )
        assert len(grad_vals) == 2
        assert theta in grad_vals
        assert phi in grad_vals


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def tensor_z(num_qubits: int) -> Operator:
    """Build Z ⊗ Z ⊗ ... ⊗ Z for the given number of qubits."""
    from microquantum.core import tensor

    ops = [Operator.Z() for _ in range(num_qubits)]
    result = ops[0]
    for op in ops[1:]:
        result = tensor(result, op)
    return result
