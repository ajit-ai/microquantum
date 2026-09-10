"""Tests for COBYLA and Nelder-Mead gradient-free optimizers."""

import math

import pytest

from microquantum.core.parameter import Parameter
from microquantum.optimizers import COBYLA, NelderMead


class TestCOBYLA:
    def test_minimize_quadratic(self):
        """Minimize f(x) = x^2. Optimal at x=0."""
        theta = Parameter("theta")
        cost_fn = lambda p: p[theta] ** 2

        opt = COBYLA(max_iter=50, tol=1e-8, rhobeg=1.0)
        result = opt.minimize(cost_fn, initial_params={theta: 2.0})

        assert abs(result.optimal_parameters[theta]) < 0.1
        assert result.optimal_value < 0.01

    def test_minimize_rosenbrock_1d(self):
        """Minimize f(x) = (x-1)^2. Optimal at x=1."""
        theta = Parameter("theta")
        cost_fn = lambda p: (p[theta] - 1.0) ** 2

        opt = COBYLA(max_iter=100, tol=1e-8, rhobeg=0.5)
        result = opt.minimize(cost_fn, initial_params={theta: 0.0})

        assert abs(result.optimal_parameters[theta] - 1.0) < 0.1

    def test_two_parameters(self):
        """Minimize f(x, y) = (x-1)^2 + (y-2)^2."""
        x = Parameter("x")
        y = Parameter("y")
        cost_fn = lambda p: (p[x] - 1.0) ** 2 + (p[y] - 2.0) ** 2

        opt = COBYLA(max_iter=200, tol=1e-8, rhobeg=1.0)
        result = opt.minimize(cost_fn, initial_params={x: 0.0, y: 0.0})

        assert abs(result.optimal_parameters[x] - 1.0) < 0.2
        assert abs(result.optimal_parameters[y] - 2.0) < 0.2
        assert result.optimal_value < 0.05

    def test_no_gradient_needed(self):
        """COBYLA should work without gradient function."""
        theta = Parameter("theta")
        cost_fn = lambda p: p[theta] ** 2

        opt = COBYLA(max_iter=50)
        result = opt.minimize(cost_fn, initial_params={theta: 3.0})

        assert abs(result.optimal_parameters[theta]) < 0.2

    def test_history_recorded(self):
        theta = Parameter("theta")
        cost_fn = lambda p: p[theta] ** 2

        opt = COBYLA(max_iter=10)
        result = opt.minimize(cost_fn, initial_params={theta: 1.0})

        assert len(result.history) > 1
        assert result.iterations > 0


class TestNelderMead:
    def test_minimize_quadratic(self):
        """Minimize f(x) = x^2. Optimal at x=0."""
        theta = Parameter("theta")
        cost_fn = lambda p: p[theta] ** 2

        opt = NelderMead(max_iter=50, tol=1e-8)
        result = opt.minimize(cost_fn, initial_params={theta: 2.0})

        assert abs(result.optimal_parameters[theta]) < 0.1
        assert result.optimal_value < 0.01

    def test_minimize_rosenbrock_1d(self):
        """Minimize f(x) = (x-1)^2. Optimal at x=1."""
        theta = Parameter("theta")
        cost_fn = lambda p: (p[theta] - 1.0) ** 2

        opt = NelderMead(max_iter=100, tol=1e-8)
        result = opt.minimize(cost_fn, initial_params={theta: 0.0})

        assert abs(result.optimal_parameters[theta] - 1.0) < 0.1

    def test_two_parameters(self):
        """Minimize f(x, y) = (x-1)^2 + (y-2)^2."""
        x = Parameter("x")
        y = Parameter("y")
        cost_fn = lambda p: (p[x] - 1.0) ** 2 + (p[y] - 2.0) ** 2

        opt = NelderMead(max_iter=200, tol=1e-8)
        result = opt.minimize(cost_fn, initial_params={x: 0.0, y: 0.0})

        assert abs(result.optimal_parameters[x] - 1.0) < 0.2
        assert abs(result.optimal_parameters[y] - 2.0) < 0.2
        assert result.optimal_value < 0.05

    def test_no_gradient_needed(self):
        """Nelder-Mead should work without gradient function."""
        theta = Parameter("theta")
        cost_fn = lambda p: p[theta] ** 2

        opt = NelderMead(max_iter=50)
        result = opt.minimize(cost_fn, initial_params={theta: 3.0})

        assert abs(result.optimal_parameters[theta]) < 0.2

    def test_nonlinear_function(self):
        """Minimize a non-linear function with multiple local minima."""
        theta = Parameter("theta")
        cost_fn = lambda p: math.sin(p[theta]) ** 2 + 0.1 * (p[theta] - 1.0) ** 2

        opt = NelderMead(max_iter=200, tol=1e-8)
        result = opt.minimize(cost_fn, initial_params={theta: 0.5})

        # Near the local minimum around 0
        assert result.optimal_value < 0.1

    def test_history_recorded(self):
        theta = Parameter("theta")
        cost_fn = lambda p: p[theta] ** 2

        opt = NelderMead(max_iter=10)
        result = opt.minimize(cost_fn, initial_params={theta: 1.0})

        assert len(result.history) > 1
        assert result.iterations > 0
