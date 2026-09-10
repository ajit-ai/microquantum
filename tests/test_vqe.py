"""Tests for VQE and classical optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from microquantum.algorithms import VQE, VQEResult
from microquantum.core import (
    Operator,
    Parameter,
    QuantumCircuit,
    tensor,
)
from microquantum.optimizers import Adam, GradientDescent, OptimizerResult

# ------------------------------------------------------------------
# Optimizer tests
# ------------------------------------------------------------------


class TestGradientDescent:
    """Gradient descent optimizer."""

    def test_minimize_parabola(self) -> None:
        """Minimize f(x) = x^2 starting from x=1.0."""
        p = Parameter("x")
        optimizer = GradientDescent(learning_rate=0.5, max_iter=50, tol=1e-8)

        def cost(params: dict[Parameter, float]) -> float:
            return params[p] ** 2

        def grad(params: dict[Parameter, float]) -> dict[Parameter, float]:
            return {p: 2.0 * params[p]}

        result = optimizer.minimize(cost, grad, {p: 1.0})
        assert result.optimal_parameters[p] == pytest.approx(0.0, abs=1e-4)
        assert result.optimal_value == pytest.approx(0.0, abs=1e-6)
        assert result.converged
        assert result.iterations > 0
        assert len(result.history) > 1

    def test_history_decreasing(self) -> None:
        """Energy should generally decrease over iterations."""
        p = Parameter("x")
        optimizer = GradientDescent(learning_rate=0.3, max_iter=30)

        def cost(params: dict[Parameter, float]) -> float:
            return (params[p] - 2.0) ** 2

        def grad(params: dict[Parameter, float]) -> dict[Parameter, float]:
            return {p: 2.0 * (params[p] - 2.0)}

        result = optimizer.minimize(cost, grad, {p: 0.0})
        # Final value should be better than initial
        assert result.history[-1] < result.history[0]

    def test_result_dataclass(self) -> None:
        """OptimizerResult fields are populated."""
        result = OptimizerResult()
        assert result.iterations == 0
        assert result.history == []
        assert result.converged is False


class TestAdam:
    """Adam optimizer."""

    def test_minimize_parabola(self) -> None:
        """Minimize f(x) = x^2 starting from x=1.0."""
        p = Parameter("x")
        optimizer = Adam(learning_rate=0.5, max_iter=100, tol=1e-10)

        def cost(params: dict[Parameter, float]) -> float:
            return params[p] ** 2

        def grad(params: dict[Parameter, float]) -> dict[Parameter, float]:
            return {p: 2.0 * params[p]}

        result = optimizer.minimize(cost, grad, {p: 1.0})
        assert result.optimal_parameters[p] == pytest.approx(0.0, abs=1e-2)
        assert result.optimal_value == pytest.approx(0.0, abs=1e-4)

    def test_multidimensional(self) -> None:
        """Minimize f(x,y) = x^2 + y^2."""
        x = Parameter("x")
        y = Parameter("y")
        optimizer = Adam(learning_rate=0.1, max_iter=200, tol=1e-8)

        def cost(params: dict[Parameter, float]) -> float:
            return params[x] ** 2 + params[y] ** 2

        def grad(params: dict[Parameter, float]) -> dict[Parameter, float]:
            return {x: 2.0 * params[x], y: 2.0 * params[y]}

        result = optimizer.minimize(cost, grad, {x: 3.0, y: -2.0})
        assert result.optimal_parameters[x] == pytest.approx(0.0, abs=1e-2)
        assert result.optimal_parameters[y] == pytest.approx(0.0, abs=1e-2)


# ------------------------------------------------------------------
# VQE tests
# ------------------------------------------------------------------


class TestVQE:
    """VQE algorithm on small systems."""

    def test_single_qubit_z(self) -> None:
        """Minimize <Z> on a single qubit. Ground state = |1>, energy = -1.

        Ry(theta)|0> has <Z> = cos(theta). Start at theta=pi/4 so
        gradient (-sin(theta)) is non-zero.
        """
        H = Operator.Z()
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)

        optimizer = GradientDescent(learning_rate=0.3, max_iter=100, tol=1e-8)
        vqe = VQE(ansatz, H, optimizer)
        result = vqe.compute_minimum_eigenvalue(initial_params={theta: 0.5})

        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-3)

    def test_two_qubit_ising_zz_zi(self) -> None:
        """H = Z x Z + Z x I on 2 qubits.

        Eigenvalues of Z x Z: {+1, -1, -1, +1}
        Eigenvalues of Z x I: {+1, +1, -1, -1}
        Combined: {+2, 0, -2, 0}
        Ground state energy = -2.0

        Use a 2-parameter ansatz with sufficient expressivity and
        start from non-zero params so gradients are non-zero.
        """
        H = tensor(Operator.Z(), Operator.Z()) + tensor(Operator.Z(), Operator.I())

        exact_eigenvalues = np.linalg.eigvalsh(H.matrix)
        ground_energy = float(exact_eigenvalues[0])

        t0 = Parameter("t0")
        t1 = Parameter("t1")
        ansatz = QuantumCircuit(2).ry(t0, 0).ry(t1, 1)

        optimizer = GradientDescent(learning_rate=0.2, max_iter=300, tol=1e-7)
        vqe = VQE(ansatz, H, optimizer)
        result = vqe.compute_minimum_eigenvalue(initial_params={t0: 0.5, t1: 0.5})

        assert result.eigenvalue == pytest.approx(ground_energy, abs=1e-2)

    def test_vqe_result_fields(self) -> None:
        """VQEResult has all expected fields."""
        H = Operator.Z()
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)

        optimizer = GradientDescent(learning_rate=0.3, max_iter=50)
        vqe = VQE(ansatz, H, optimizer)
        result = vqe.compute_minimum_eigenvalue(initial_params={theta: 0.5})

        assert isinstance(result, VQEResult)
        assert result.optimizer_result is not None
        assert len(result.optimizer_result.history) > 0

    def test_vqe_with_adam(self) -> None:
        """VQE converges using Adam optimizer."""
        H = Operator.Z()
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)

        optimizer = Adam(learning_rate=0.1, max_iter=200, tol=1e-6)
        vqe = VQE(ansatz, H, optimizer)
        result = vqe.compute_minimum_eigenvalue(initial_params={theta: 0.5})

        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-2)

    def test_vqe_requires_parameterized_ansatz(self) -> None:
        """Non-parameterized ansatz raises ValueError."""
        ansatz = QuantumCircuit(1).h(0)
        H = Operator.Z()
        optimizer = GradientDescent(learning_rate=0.1, max_iter=10)

        with pytest.raises(ValueError, match="parameter"):
            VQE(ansatz, H, optimizer)

    def test_vqe_initial_params(self) -> None:
        """VQE with custom initial parameters."""
        H = Operator.Z()
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)

        optimizer = GradientDescent(learning_rate=0.3, max_iter=100, tol=1e-8)
        vqe = VQE(ansatz, H, optimizer)
        result = vqe.compute_minimum_eigenvalue(initial_params={theta: 1.0})

        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-3)

    def test_vqe_initial_params_by_name(self) -> None:
        """VQE with initial params given by string name."""
        H = Operator.Z()
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)

        optimizer = GradientDescent(learning_rate=0.3, max_iter=100, tol=1e-8)
        vqe = VQE(ansatz, H, optimizer)
        result = vqe.compute_minimum_eigenvalue(initial_params={"theta": 1.0})

        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-3)

    def test_vqe_history_monotonic(self) -> None:
        """Optimization history should generally improve."""
        H = Operator.Z()
        theta = Parameter("theta")
        ansatz = QuantumCircuit(1).ry(theta, 0)

        optimizer = GradientDescent(learning_rate=0.3, max_iter=100)
        vqe = VQE(ansatz, H, optimizer)
        result = vqe.compute_minimum_eigenvalue(initial_params={theta: 0.5})

        history = result.optimizer_result.history
        # Final energy should be better than initial
        assert history[-1] <= history[0]
