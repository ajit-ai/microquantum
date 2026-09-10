"""Visualization module for quantum analytics results.

Generates charts and plots for business presentations.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def plot_kernel_matrix(
    kernel_matrix: np.ndarray,
    title: str = "Quantum Kernel Matrix",
    labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Generate kernel matrix visualization data.

    Args:
        kernel_matrix: 2D kernel matrix.
        title: Plot title.
        labels: Axis labels.

    Returns:
        Dictionary with plot data for rendering.
    """
    return {
        "type": "heatmap",
        "title": title,
        "data": kernel_matrix.tolist(),
        "labels": labels or [f"Sample {i}" for i in range(kernel_matrix.shape[0])],
        "colorscale": "Viridis",
    }


def plot_optimization_history(
    history: list[float],
    title: str = "Optimization Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Energy",
) -> dict[str, Any]:
    """Generate optimization history plot data.

    Args:
        history: Energy values over iterations.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Dictionary with plot data for rendering.
    """
    return {
        "type": "line",
        "title": title,
        "x": list(range(len(history))),
        "y": history,
        "xlabel": xlabel,
        "ylabel": ylabel,
    }


def plot_portfolio_allocation(
    weights: list[float],
    assets: Optional[list[str]] = None,
    title: str = "Portfolio Allocation",
) -> dict[str, Any]:
    """Generate portfolio allocation pie chart data.

    Args:
        weights: Asset weights.
        assets: Asset names.
        title: Plot title.

    Returns:
        Dictionary with plot data for rendering.
    """
    if assets is None:
        assets = [f"Asset {i}" for i in range(len(weights))]

    return {
        "type": "pie",
        "title": title,
        "labels": assets,
        "values": weights,
    }


def plot_risk_return_scatter(
    risks: list[float],
    returns: list[float],
    labels: Optional[list[str]] = None,
    title: str = "Risk-Return Profile",
) -> dict[str, Any]:
    """Generate risk-return scatter plot data.

    Args:
        risks: Risk values (volatility).
        returns: Return values.
        labels: Point labels.
        title: Plot title.

    Returns:
        Dictionary with plot data for rendering.
    """
    if labels is None:
        labels = [f"Portfolio {i}" for i in range(len(risks))]

    return {
        "type": "scatter",
        "title": title,
        "x": risks,
        "y": returns,
        "labels": labels,
        "xlabel": "Risk (Volatility)",
        "ylabel": "Return",
    }


def plot_comparison_bar(
    quantum_values: list[float],
    classical_values: list[float],
    metric_names: list[str],
    title: str = "Quantum vs Classical Comparison",
) -> dict[str, Any]:
    """Generate comparison bar chart data.

    Args:
        quantum_values: Quantum algorithm results.
        classical_values: Classical algorithm results.
        metric_names: Names of metrics being compared.
        title: Plot title.

    Returns:
        Dictionary with plot data for rendering.
    """
    return {
        "type": "bar",
        "title": title,
        "categories": metric_names,
        "series": [
            {"name": "Quantum", "values": quantum_values},
            {"name": "Classical", "values": classical_values},
        ],
        "xlabel": "Metric",
        "ylabel": "Value",
    }


def plot_route_map(
    cities: list[tuple[float, float]],
    tour: list[int],
    title: str = "Optimized Route",
) -> dict[str, Any]:
    """Generate route visualization data.

    Args:
        cities: List of (x, y) city coordinates.
        tour: Ordered city indices.
        title: Plot title.

    Returns:
        Dictionary with plot data for rendering.
    """
    route_x = [cities[i][0] for i in tour]
    route_y = [cities[i][1] for i in tour]
    route_x.append(route_x[0])  # Return to start
    route_y.append(route_y[0])

    return {
        "type": "line",
        "title": title,
        "x": route_x,
        "y": route_y,
        "markers": {
            "x": [c[0] for c in cities],
            "y": [c[1] for c in cities],
            "labels": [f"City {i}" for i in range(len(cities))],
        },
        "xlabel": "X",
        "ylabel": "Y",
    }


def plot_var_distribution(
    returns: np.ndarray,
    var_value: float,
    cvar_value: float,
    confidence_level: float = 0.95,
    title: str = "Return Distribution with VaR",
) -> dict[str, Any]:
    """Generate VaR distribution plot data.

    Args:
        returns: Historical returns.
        var_value: Value at Risk.
        cvar_value: Conditional VaR.
        confidence_level: Confidence level.
        title: Plot title.

    Returns:
        Dictionary with plot data for rendering.
    """
    hist, bin_edges = np.histogram(returns, bins=50)

    return {
        "type": "histogram",
        "title": title,
        "x": bin_edges[:-1].tolist(),
        "y": hist.tolist(),
        "annotations": [
            {"x": var_value, "label": f"VaR ({confidence_level:.0%})", "color": "red"},
            {"x": cvar_value, "label": "CVaR", "color": "darkred"},
        ],
        "xlabel": "Return",
        "ylabel": "Frequency",
    }


def format_analysis_report(result: dict[str, Any]) -> str:
    """Format analysis result as a readable text report.

    Args:
        result: AnalysisResult.to_json() output.

    Returns:
        Formatted text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("QUANTUM ANALYTICS REPORT")
    lines.append("=" * 60)

    if result.get("solution"):
        lines.append("\nSolution:")
        for key, value in result["solution"].items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            elif isinstance(value, list) and len(value) > 10:
                lines.append(f"  {key}: [{len(value)} items]")
            else:
                lines.append(f"  {key}: {value}")

    if result.get("quantum_metrics"):
        lines.append("\nQuantum Metrics:")
        for key, value in result["quantum_metrics"].items():
            lines.append(f"  {key}: {value}")

    if result.get("comparison"):
        comp = result["comparison"]
        lines.append("\nComparison:")
        lines.append(f"  Metric: {comp.get('metric_name', 'N/A')}")
        lines.append(f"  Quantum: {comp.get('quantum_value', 'N/A')}")
        lines.append(f"  Classical: {comp.get('classical_value', 'N/A')}")
        lines.append(f"  Improvement: {comp.get('improvement_pct', 0):.2f}%")
        lines.append(f"  Quantum wins: {comp.get('quantum_wins', False)}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
