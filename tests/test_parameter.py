"""Tests for parameterized quantum circuits."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.core import (
    Operator,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
)


class TestParameter:
    """Parameter class basics."""

    def test_creation(self) -> None:
        p = Parameter("theta")
        assert p.name == "theta"

    def test_repr(self) -> None:
        p = Parameter("alpha")
        assert "alpha" in repr(p)

    def test_str(self) -> None:
        p = Parameter("alpha")
        assert str(p) == "alpha"

    def test_equality(self) -> None:
        p1 = Parameter("theta")
        p2 = Parameter("theta")
        p3 = Parameter("phi")
        assert p1 == p2
        assert p1 != p3
        assert p1 != "theta"

    def test_hash(self) -> None:
        p1 = Parameter("theta")
        p2 = Parameter("theta")
        assert hash(p1) == hash(p2)
        s = {p1, p2}
        assert len(s) == 1

    def test_as_dict_key(self) -> None:
        p = Parameter("theta")
        d = {p: 1.5}
        assert d[Parameter("theta")] == 1.5
        assert d["theta"] if "theta" in d else d[p] == 1.5


class TestParameterExpression:
    """ParameterExpression arithmetic."""

    def test_default_coefficient(self) -> None:
        p = Parameter("theta")
        expr = ParameterExpression(p)
        assert expr.coefficient == 1.0
        assert expr.constant == 0.0

    def test_scaled_parameter(self) -> None:
        p = Parameter("theta")
        expr = 2 * p
        assert isinstance(expr, ParameterExpression)
        assert expr.coefficient == 2.0

    def test_negative_parameter(self) -> None:
        p = Parameter("theta")
        expr = -p
        assert isinstance(expr, ParameterExpression)
        assert expr.coefficient == -1.0

    def test_evaluate_simple(self) -> None:
        p = Parameter("theta")
        expr = ParameterExpression(p, coefficient=2.0)
        result = expr.evaluate({p: 1.5})
        assert result == pytest.approx(3.0)

    def test_evaluate_with_constant(self) -> None:
        p = Parameter("theta")
        expr = ParameterExpression(p, coefficient=1.0, constant=0.5)
        result = expr.evaluate({p: 2.0})
        assert result == pytest.approx(2.5)

    def test_evaluate_by_name(self) -> None:
        p = Parameter("theta")
        expr = ParameterExpression(p, coefficient=3.0)
        result = expr.evaluate({"theta": 1.0})
        assert result == pytest.approx(3.0)

    def test_add_expressions(self) -> None:
        p = Parameter("theta")
        e1 = ParameterExpression(p, coefficient=1.0)
        e2 = ParameterExpression(p, coefficient=2.0)
        e3 = e1 + e2
        assert e3.coefficient == pytest.approx(3.0)

    def test_add_scalar(self) -> None:
        p = Parameter("theta")
        expr = ParameterExpression(p, coefficient=1.0, constant=0.0)
        result = expr + 0.5
        assert isinstance(result, ParameterExpression)
        assert result.evaluate({p: 1.0}) == pytest.approx(1.5)


class TestParameterizedCircuit:
    """Creating parameterized circuits."""

    def test_single_parameter_rx(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).rx(theta, 0)
        assert qc.is_parameterized
        assert qc.parameters == {theta}

    def test_single_parameter_ry(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        assert qc.is_parameterized
        assert qc.parameters == {theta}

    def test_single_parameter_rz(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).rz(theta, 0)
        assert qc.is_parameterized
        assert qc.parameters == {theta}

    def test_multiple_unique_parameters(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        assert qc.is_parameterized
        assert qc.parameters == {theta, phi}

    def test_duplicate_parameter_same_name(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2).ry(theta, 0).rx(theta, 1)
        assert len(qc.parameters) == 1
        assert theta in qc.parameters

    def test_mixed_parameterized_and_fixed(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2).h(0).ry(theta, 1).cx(0, 1)
        assert qc.is_parameterized
        assert qc.parameters == {theta}

    def test_no_parameters(self) -> None:
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        assert not qc.is_parameterized
        assert len(qc.parameters) == 0

    def test_circuit_with_numeric_rx_not_parameterized(self) -> None:
        qc = QuantumCircuit(1).rx(0.5, 0)
        assert not qc.is_parameterized
        assert len(qc.parameters) == 0


class TestBindParameters:
    """Parameter binding and execution."""

    def test_bind_single_parameter(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        bound = qc.bind_parameters({theta: np.pi})
        assert not bound.is_parameterized
        state = bound.run()
        # Ry(pi)|0> = |1>
        assert abs(state.amplitudes[1]) == pytest.approx(1.0)

    def test_bind_by_name(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        bound = qc.bind_parameters({"theta": np.pi})
        state = bound.run()
        assert abs(state.amplitudes[1]) == pytest.approx(1.0)

    def test_bind_multiple_parameters(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        bound = qc.bind_parameters({theta: np.pi, phi: 0.0})
        state = bound.run()
        # Ry(pi)|0> = |1>, Rx(0)|0> = |0>, so |10>
        assert abs(state.amplitudes[2]) == pytest.approx(1.0)

    def test_bind_partial_and_rebind(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2).ry(theta, 0).rx(phi, 1)
        partial = qc.bind_parameters({theta: np.pi})
        assert partial.is_parameterized
        full = partial.bind_parameters({phi: 0.0})
        assert not full.is_parameterized
        state = full.run()
        assert abs(state.amplitudes[2]) == pytest.approx(1.0)

    def test_bind_with_expression(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        bound = qc.bind_parameters({theta: np.pi / 2})
        state = bound.run()
        # Ry(pi/2)|0> = (|0> + |1>)/sqrt(2)
        assert abs(state.amplitudes[0]) == pytest.approx(1 / np.sqrt(2))
        assert abs(state.amplitudes[1]) == pytest.approx(1 / np.sqrt(2))

    def test_bind_missing_parameter_returns_parameterized(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        bound = qc.bind_parameters({})
        assert bound.is_parameterized
        assert theta in bound.parameters

    def test_bind_extra_parameters_ok(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(1).ry(theta, 0)
        bound = qc.bind_parameters({theta: 0.5, phi: 0.3})
        assert not bound.is_parameterized

    def test_get_unitary_requires_binding(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(ValueError, match="unbound parameters"):
            qc.get_unitary()

    def test_run_requires_binding(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(ValueError, match="unbound parameters"):
            qc.run()

    def test_expectation_value_requires_binding(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        with pytest.raises(ValueError, match="unbound parameters"):
            qc.expectation_value(Operator.Z())


class TestCircuitComposition:
    """Parameterized circuits with composition."""

    def test_add_two_parameterized_circuits(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc1 = QuantumCircuit(2).ry(theta, 0)
        qc2 = QuantumCircuit(2).rx(phi, 1)
        combined = qc1 + qc2
        assert combined.is_parameterized
        assert combined.parameters == {theta, phi}

    def test_bind_concatenated_circuit(self) -> None:
        theta = Parameter("theta")
        qc1 = QuantumCircuit(1).ry(theta, 0)
        qc2 = QuantumCircuit(1).h(0)
        combined = qc1 + qc2
        bound = combined.bind_parameters({theta: np.pi})
        state = bound.run()
        # Ry(pi)|0> = |1>, H|1> = (|0> - |1>)/sqrt(2)
        assert abs(abs(state.amplitudes[0]) - 1 / np.sqrt(2)) < 1e-10
        assert abs(abs(state.amplitudes[1]) - 1 / np.sqrt(2)) < 1e-10


class TestReprStr:
    """String representations."""

    def test_repr_parameterized(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        r = repr(qc)
        assert "theta" in r

    def test_str_parameterized(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1).ry(theta, 0)
        s = str(qc)
        assert "RY" in s
        assert "theta" in s
