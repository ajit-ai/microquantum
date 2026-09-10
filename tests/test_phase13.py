"""Tests for Phase 13: QUBO Toolchain + Visualization.

Tests QUBO/Ising converter and visualization helpers.
"""
import numpy as np
import pytest

from microquantum.analytics.visualization import (
    format_analysis_report,
    plot_comparison_bar,
    plot_kernel_matrix,
    plot_optimization_history,
    plot_portfolio_allocation,
    plot_risk_return_scatter,
    plot_route_map,
    plot_var_distribution,
)
from microquantum.core.pauli import PauliSum
from microquantum.optimization.qubo import (
    IsingConverter,
    QUBOBuilder,
)


# =============================================================================
# QUBO Builder Tests
# =============================================================================
class TestQUBOBuilder:
    """Test QUBO problem construction."""

    def test_basic_builder(self) -> None:
        """Test basic QUBO building."""
        builder = QUBOBuilder(num_variables=3)
        builder.add_linear(0, -1.0)
        builder.add_quadratic(0, 1, 2.0)
        builder.add_constant(0.5)

        qubo = builder.build("test")
        assert qubo.num_variables == 3
        assert qubo.linear[0] == -1.0
        assert qubo.Q[0, 1] == 2.0
        assert qubo.offset == 0.5

    def test_energy_computation(self) -> None:
        """Test QUBO energy computation."""
        builder = QUBOBuilder(num_variables=2)
        builder.add_quadratic(0, 1, 2.0)
        builder.add_linear(0, -1.0)

        qubo = builder.build()

        # x = [0, 1]: 0*0 + 0*(-1) + 0*1*2... E = 0
        e0 = qubo.energy(np.array([0, 0]))
        assert e0 == 0.0

        # x = [1, 1]: 1*(-1) + 1*1*2 = 1
        e1 = qubo.energy(np.array([1, 1]))
        assert e1 == 1.0

    def test_penalty_equality(self) -> None:
        """Test equality constraint penalty."""
        builder = QUBOBuilder(num_variables=2)
        builder.add_penalty_equality(0, 1, target=1, penalty=10.0)

        qubo = builder.build()

        # x = [1, 0]: satisfies constraint, penalty = 0
        e = qubo.energy(np.array([1, 0]))
        assert e == pytest.approx(0.0, abs=1e-6)

        # x = [1, 1]: violates constraint, penalty = 10 * (2-1)^2 = 10
        e = qubo.energy(np.array([1, 1]))
        assert e == pytest.approx(10.0, abs=1e-6)

    def test_penalty_one_hot(self) -> None:
        """Test one-hot constraint penalty."""
        builder = QUBOBuilder(num_variables=3)
        builder.add_penalty_one_hot([0, 1, 2], penalty=10.0)

        qubo = builder.build()

        # x = [1, 0, 0]: exactly one, penalty = 0
        e = qubo.energy(np.array([1, 0, 0]))
        assert e == pytest.approx(0.0, abs=1e-6)

        # x = [1, 1, 0]: two ones, penalty = 10 * (2-1)^2 = 10
        e = qubo.energy(np.array([1, 1, 0]))
        assert e == pytest.approx(10.0, abs=1e-6)

    def test_to_dict(self) -> None:
        """Test QUBO dictionary serialization."""
        builder = QUBOBuilder(num_variables=2)
        builder.add_linear(0, 1.0)
        qubo = builder.build("test")

        data = qubo.to_dict()
        assert data["num_variables"] == 2
        assert data["name"] == "test"
        assert len(data["linear"]) == 2


# =============================================================================
# Ising Converter Tests
# =============================================================================
class TestIsingConverter:
    """Test QUBO to Ising Hamiltonian conversion."""

    def test_convert_basic(self) -> None:
        """Test conversion of simple QUBO to Ising."""
        builder = QUBOBuilder(num_variables=1)
        builder.add_linear(0, 1.0)
        qubo = builder.build()

        ising = IsingConverter.qubo_to_ising(qubo)
        assert isinstance(ising, PauliSum)

        # x=0: energy 0, x=1: energy 1
        # Ising: x=(1-Z)/2, so H = (1-Z)/2
        # Verify via state evaluation
        from microquantum.core.measurement import expectation_value
        from microquantum.core.state import StateVector

        s0 = StateVector(1, np.array([1.0, 0.0], dtype=complex))
        s1 = StateVector(1, np.array([0.0, 1.0], dtype=complex))
        ising_op = ising.to_operator()
        e0 = expectation_value(s0, ising_op)
        e1 = expectation_value(s1, ising_op)
        assert e0 == pytest.approx(0.0, abs=1e-6)
        assert e1 == pytest.approx(1.0, abs=1e-6)

    def test_convert_quadratic(self) -> None:
        """Test conversion with quadratic term."""
        builder = QUBOBuilder(num_variables=2)
        builder.add_quadratic(0, 1, 2.0)
        qubo = builder.build()

        ising = IsingConverter.qubo_to_ising(qubo)
        assert isinstance(ising, PauliSum)
        assert ising.num_qubits == 2

    def test_ising_qubo_roundtrip(self) -> None:
        """Test QUBO -> Ising -> QUBO roundtrip."""
        builder = QUBOBuilder(num_variables=2)
        builder.add_quadratic(0, 1, 1.5)
        builder.add_linear(0, -0.5)
        builder.add_linear(1, 0.5)
        qubo = builder.build()

        ising = IsingConverter.qubo_to_ising(qubo)
        qubo2 = IsingConverter.ising_to_qubo(ising)

        # Check agreement on all valid solutions
        for x_bits in range(4):
            x = np.array([(x_bits >> 1) & 1, x_bits & 1], dtype=float)
            e1 = qubo.energy(x)
            e2 = qubo2.energy(x)
            assert e1 == pytest.approx(e2, abs=1e-6)

    def test_evaluate(self) -> None:
        """Test QUBO solution evaluation."""
        builder = QUBOBuilder(num_variables=2)
        builder.add_penalty_one_hot([0, 1])
        qubo = builder.build()

        result = IsingConverter.evaluate(qubo, np.array([1, 0]))
        assert result["energy"] == pytest.approx(0.0, abs=1e-6)
        assert result["num_ones"] == 1
        assert result["num_variables"] == 2


# =============================================================================
# Visualization Tests
# =============================================================================
class TestVisualization:
    """Test visualization helpers."""

    def test_plot_kernel_matrix(self) -> None:
        """Test kernel matrix plot."""
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        plot = plot_kernel_matrix(K)
        assert plot["type"] == "heatmap"
        assert plot["data"] == [[1.0, 0.5], [0.5, 1.0]]

    def test_plot_optimization_history(self) -> None:
        """Test optimization history plot."""
        history = [5.0, 3.0, 1.5, 0.5]
        plot = plot_optimization_history(history)
        assert plot["type"] == "line"
        assert plot["y"] == history
        assert plot["x"] == [0, 1, 2, 3]

    def test_plot_portfolio_allocation(self) -> None:
        """Test portfolio allocation plot."""
        plot = plot_portfolio_allocation(
            weights=[0.4, 0.3, 0.3],
            assets=["AAPL", "GOOGL", "MSFT"],
        )
        assert plot["type"] == "pie"
        assert plot["labels"] == ["AAPL", "GOOGL", "MSFT"]
        assert plot["values"] == [0.4, 0.3, 0.3]

    def test_plot_risk_return(self) -> None:
        """Test risk-return scatter plot."""
        plot = plot_risk_return_scatter(risks=[0.1, 0.2], returns=[0.12, 0.18])
        assert plot["type"] == "scatter"
        assert plot["x"] == [0.1, 0.2]

    def test_plot_comparison(self) -> None:
        """Test comparison bar chart."""
        plot = plot_comparison_bar(
            quantum_values=[1.0, 2.0],
            classical_values=[0.8, 1.9],
            metric_names=["A", "B"],
        )
        assert plot["type"] == "bar"
        assert plot["categories"] == ["A", "B"]
        assert len(plot["series"]) == 2

    def test_plot_route_map(self) -> None:
        """Test route map plot."""
        cities = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)]
        plot = plot_route_map(cities=cities, tour=[0, 1, 2])
        assert plot["type"] == "line"
        assert len(plot["x"]) == 4  # Returns to start

    def test_plot_var_distribution(self) -> None:
        """Test VaR distribution plot."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 500)
        plot = plot_var_distribution(returns, var_value=-0.05, cvar_value=-0.08)
        assert plot["type"] == "histogram"
        assert len(plot["annotations"]) == 2

    def test_format_report(self) -> None:
        """Test report formatting."""
        result = {
            "solution": {"energy": -1.857, "name": "H2"},
            "quantum_metrics": {"algorithm": "VQE"},
            "comparison": {
                "metric_name": "test",
                "quantum_value": 1.0,
                "classical_value": 0.8,
                "improvement_pct": 25.0,
                "quantum_wins": True,
            },
        }
        report = format_analysis_report(result)
        assert "QUANTUM ANALYTICS REPORT" in report
        assert "energy" in report
        assert "25.00%" in report


# =============================================================================
# Integration Tests
# =============================================================================
class TestPhase13Integration:
    """Test Phase 13 module integration."""

    def test_full_qubo_to_optimization_pipeline(self) -> None:
        """Test QUBO -> Ising -> energy evaluation pipeline."""
        # Portfolio selection: pick exactly 1 of 3 assets to maximize return
        builder = QUBOBuilder(num_variables=3)

        # Objective: maximize expected return (minimize negative return)
        returns = np.array([0.12, 0.18, 0.15])
        for i in range(3):
            builder.add_linear(i, -returns[i])

        # Constraint: exactly 1 asset selected
        builder.add_penalty_one_hot([0, 1, 2], penalty=5.0)

        qubo = builder.build("portfolio_selection")

        # Convert to Ising Hamiltonian
        ising = IsingConverter.qubo_to_ising(qubo)
        assert isinstance(ising, PauliSum)

        # Brute-force find the minimum-energy solution
        best_energy = float("inf")
        best_ones = -1
        for x_bits in range(8):
            x = np.array(
                [(x_bits >> 2) & 1, (x_bits >> 1) & 1, x_bits & 1],
                dtype=float,
            )
            energy = qubo.energy(x)
            if energy < best_energy:
                best_energy = energy
                best_ones = int(np.sum(x))

        # The optimal solution must select exactly 1 asset
        assert best_ones == 1
        # ... and it must be the highest-return asset (index 1, return 0.18)
        assert best_energy == pytest.approx(-0.18, abs=1e-6)

        # Ising energies must agree with QUBO energies on all solutions
        from microquantum.core.measurement import expectation_value
        from microquantum.core.state import StateVector

        ising_op = ising.to_operator()
        for x_bits in range(8):
            x = np.array(
                [(x_bits >> 2) & 1, (x_bits >> 1) & 1, x_bits & 1],
                dtype=float,
            )
            qubo_energy = qubo.energy(x)
            # bit x -> spin state |x_bits>
            amp = np.zeros(8, dtype=complex)
            amp[x_bits] = 1.0
            st = StateVector(3, amp)
            ising_energy = float(expectation_value(st, ising_op))
            assert qubo_energy == pytest.approx(ising_energy, abs=1e-6)