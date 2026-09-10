"""Tests for the NumPy BFGS quasi-Newton optimizer (MQ-05)."""

from __future__ import annotations

import pytest

from microquantum.core import Parameter
from microquantum.optimizers import BFGS
from microquantum.optimizers.base import Optimizer


def _make_params(*names: str) -> dict[str, Parameter]:
    return {name: Parameter(name) for name in names}


class TestBFGS:
    def test_converges_scalar_parabola(self) -> None:
        (x,) = _make_params("x").values()
        result = BFGS(max_iter=50, tol=1e-10).minimize(
            cost_fn=lambda p: p[x] ** 2,
            gradient_fn=lambda p: {x: 2.0 * p[x]},
            initial_params={x: 1.0},
        )
        assert result.converged
        assert result.optimal_parameters[x] == pytest.approx(0.0, abs=1e-4)
        assert result.optimal_value == pytest.approx(0.0, abs=1e-4)

    def test_converges_quadratic(self) -> None:
        params = _make_params("x", "y")
        x, y = params["x"], params["y"]

        def cost(p):
            return (p[x] - 2.0) ** 2 + (p[y] + 1.0) ** 2

        def grad(p):
            return {x: 2.0 * (p[x] - 2.0), y: 2.0 * (p[y] + 1.0)}

        result = BFGS(max_iter=100, tol=1e-10).minimize(
            cost_fn=cost,
            gradient_fn=grad,
            initial_params={x: 0.0, y: 0.0},
        )
        assert result.converged
        assert result.optimal_parameters[x] == pytest.approx(2.0, abs=1e-3)
        assert result.optimal_parameters[y] == pytest.approx(-1.0, abs=1e-3)

    def test_history_recorded(self) -> None:
        (x,) = _make_params("x").values()
        result = BFGS(max_iter=50).minimize(
            cost_fn=lambda p: p[x] ** 2,
            gradient_fn=lambda p: {x: 2.0 * p[x]},
            initial_params={x: 1.0},
        )
        assert len(result.history) > 1
        assert result.history[0] >= result.history[-1]

    def test_contract_abstractmethods_unchanged(self) -> None:
        assert Optimizer.__abstractmethods__ == frozenset({"_step", "_converged", "max_iter"})

    def test_minimize_without_gradient(self) -> None:
        (x,) = _make_params("x").values()
        # BFGS falls back to finite differences when no gradient is given
        result = BFGS(max_iter=80, tol=1e-6).minimize(
            cost_fn=lambda p: p[x] ** 2,
            initial_params={x: 1.0},
        )
        assert result.converged
        assert result.optimal_parameters[x] == pytest.approx(0.0, abs=1e-3)

    def test_precision_option(self) -> None:
        (x,) = _make_params("x").values()
        result = BFGS(max_iter=50, tol=1e-6).minimize(
            cost_fn=lambda p: p[x] ** 2,
            gradient_fn=lambda p: {x: 2.0 * p[x]},
            initial_params={x: 0.5},
        )
        assert result.optimal_value == pytest.approx(0.0, abs=1e-6)


class TestLBFGSB:
    def test_scipy_missing_raises_import_error(self) -> None:
        # scipy is not an SDK dependency; constructing the wrapper must
        # fail clearly when the optional dependency is absent.
        from microquantum.optimizers import L_BFGS_B

        try:
            import scipy  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError):
                L_BFGS_B()
        else:
            assert L_BFGS_B(max_iter=10, tol=1e-4) is not None