"""Tests for SPSA and QNSPSA optimizers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from microquantum.core import Parameter, QuantumCircuit, Operator
from microquantum.optimizers import SPSA, QNSPSA


class TestSPSA:
    """SPSA optimizer tests."""

    def test_minimize_parabola(self) -> None:
        """SPSA should minimize f(x) = x^2 towards zero."""
        p = Parameter("x")
        opt = SPSA(a=0.5, c=0.1, max_iter=300, tol=1e-8, seed=42)

        def cost(params: dict[Parameter, float]) -> float:
            return params[p] ** 2

        result = opt.minimize(cost, initial_params={p: 1.0})
        assert abs(result.optimal_parameters[p]) < 0.3
        assert result.optimal_value < 0.1
        assert result.iterations > 0

    def test_minimize_linear(self) -> None:
        """SPSA should minimize f(x) = (x - 3)^2."""
        p = Parameter("x")
        opt = SPSA(a=0.5, c=0.1, max_iter=400, seed=7)

        def cost(params: dict[Parameter, float]) -> float:
            return (params[p] - 3.0) ** 2

        result = opt.minimize(cost, initial_params={p: 0.0})
        assert abs(result.optimal_parameters[p] - 3.0) < 0.5

    def test_two_parameters(self) -> None:
        """SPSA with multiple parameters."""
        p1, p2 = Parameter("x"), Parameter("y")
        opt = SPSA(a=0.3, c=0.1, max_iter=300, seed=42)

        def cost(params: dict[Parameter, float]) -> float:
            return (params[p1] - 1.0) ** 2 + (params[p2] + 2.0) ** 2

        result = opt.minimize(cost, initial_params={p1: 0.0, p2: 0.0})
        assert abs(result.optimal_parameters[p1] - 1.0) < 0.5
        assert abs(result.optimal_parameters[p2] + 2.0) < 0.5

    def test_requires_cost_fn(self) -> None:
        """SPSA should raise if cost_fn is missing."""
        p = Parameter("x")
        opt = SPSA(max_iter=5, seed=0)
        with pytest.raises(ValueError, match="cost_fn"):
            opt._step({p: 0.0}, {}, 0)

    def test_repr(self) -> None:
        opt = SPSA(a=0.1, max_iter=100)
        assert "SPSA" in repr(opt)

    def test_max_iter_property(self) -> None:
        opt = SPSA(max_iter=250)
        assert opt.max_iter == 250

    def test_convergence(self) -> None:
        """SPSA should detect convergence."""
        p = Parameter("x")
        opt = SPSA(a=0.1, c=0.01, max_iter=500, tol=1e-4, seed=42)

        def cost(params: dict[Parameter, float]) -> float:
            return (params[p]) ** 2

        result = opt.minimize(cost, initial_params={p: 0.5})
        assert result.optimal_value < 0.5


class TestQNSPSA:
    """QNSPSA optimizer tests."""

    def test_minimize_parabola(self) -> None:
        """QNSPSA should minimize f(x) = x^2."""
        p = Parameter("x")
        opt = QNSPSA(a=0.5, c=0.1, max_iter=300, seed=42)

        def cost(params: dict[Parameter, float]) -> float:
            return params[p] ** 2

        result = opt.minimize(cost, initial_params={p: 1.0})
        assert abs(result.optimal_parameters[p]) < 0.3
        assert result.optimal_value < 0.1

    def test_with_fidelity_fn(self) -> None:
        """QNSPSA with custom fidelity function."""
        p = Parameter("x")
        opt = QNSPSA(
            a=0.5,
            c=0.1,
            max_iter=200,
            fidelity_fn=lambda a, b: 1.0 - abs(a.get(p, 0) - b.get(p, 0)) / math.pi,
            seed=42,
        )

        def cost(params: dict[Parameter, float]) -> float:
            return params[p] ** 2

        result = opt.minimize(cost, initial_params={p: 1.0})
        assert abs(result.optimal_parameters[p]) < 0.5

    def test_requires_cost_fn(self) -> None:
        """QNSPSA should raise if cost_fn is missing."""
        p = Parameter("x")
        opt = QNSPSA(max_iter=5, seed=0)
        with pytest.raises(ValueError, match="cost_fn"):
            opt._step({p: 0.0}, {}, 0)

    def test_repr(self) -> None:
        opt = QNSPSA(a=0.1, max_iter=100)
        assert "QNSPSA" in repr(opt)

    def test_max_iter_property(self) -> None:
        opt = QNSPSA(max_iter=250)
        assert opt.max_iter == 250
