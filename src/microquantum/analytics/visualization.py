"""Visualization module for quantum analytics results.

Generates chart data for quantum/classical analysis reports.
"""
from __future__ import annotations

import warnings
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


def plot_allocation(
    weights: list[float],
    labels: Optional[list[str]] = None,
    title: str = "Allocation",
) -> dict[str, Any]:
    """Generate allocation pie chart data.

    Args:
        weights: Weight of each category (should sum to 1 for a share).
        labels: Category names.
        title: Plot title.

    Returns:
        Dictionary with plot data for rendering.
    """
    if labels is None:
        labels = [f"Category {i}" for i in range(len(weights))]

    return {
        "type": "pie",
        "title": title,
        "labels": labels,
        "values": weights,
    }


def plot_scatter(
    x_values: list[float],
    y_values: list[float],
    labels: Optional[list[str]] = None,
    title: str = "Scatter Plot",
    xlabel: str = "Metric X",
    ylabel: str = "Metric Y",
) -> dict[str, Any]:
    """Generate scatter plot data.

    Args:
        x_values: Values for the horizontal axis.
        y_values: Values for the vertical axis.
        labels: Point labels.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Dictionary with plot data for rendering.
    """
    if labels is None:
        labels = [f"Point {i}" for i in range(len(x_values))]

    return {
        "type": "scatter",
        "title": title,
        "x": x_values,
        "y": y_values,
        "labels": labels,
        "xlabel": xlabel,
        "ylabel": ylabel,
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
        cities: List of (x, y) coordinates.
        tour: Ordered point indices.
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
            "labels": [f"Point {i}" for i in range(len(cities))],
        },
        "xlabel": "X",
        "ylabel": "Y",
    }


def plot_distribution(
    values: np.ndarray,
    thresholds: Optional[list[tuple[float, str, str]]] = None,
    confidence_level: float = 0.95,
    title: str = "Distribution",
) -> dict[str, Any]:
    """Generate distribution histogram data with optional threshold lines.

    Args:
        values: The values to histogram.
        thresholds: Optional list of (value, label, color) threshold markers.
        confidence_level: Confidence level shown in threshold labels.
        title: Plot title.

    Returns:
        Dictionary with plot data for rendering.
    """
    hist, bin_edges = np.histogram(values, bins=50)
    annotations = [
        {"x": value, "label": label, "color": color}
        for value, label, color in (thresholds or [])
    ]

    return {
        "type": "histogram",
        "title": title,
        "x": bin_edges[:-1].tolist(),
        "y": hist.tolist(),
        "annotations": annotations,
        "xlabel": "Value",
        "ylabel": "Frequency",
    }


def format_analysis_report(result: dict[str, Any]) -> str:
    """Format analysis result as a readable text report.

    Args:
        result: ``AnalysisResult.to_dict()`` output.

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


# ------------------------------------------------------------------
# Deprecated aliases (pre-MQ-01 names, kept for backward compatibility)
# ------------------------------------------------------------------

def plot_portfolio_allocation(
    weights: list[float],
    assets: Optional[list[str]] = None,
    title: str = "Allocation",
) -> dict[str, Any]:
    """Deprecated alias for :func:`plot_allocation`."""
    warnings.warn(
        "plot_portfolio_allocation is deprecated; use plot_allocation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_allocation(weights, labels=assets, title=title)


def plot_risk_return_scatter(
    risks: list[float],
    returns: list[float],
    labels: Optional[list[str]] = None,
    title: str = "Scatter Plot",
) -> dict[str, Any]:
    """Deprecated alias for :func:`plot_scatter`."""
    warnings.warn(
        "plot_risk_return_scatter is deprecated; use plot_scatter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_scatter(risks, returns, labels=labels, title=title)


def plot_var_distribution(
    returns: np.ndarray,
    var_value: float,
    cvar_value: float,
    confidence_level: float = 0.95,
    title: str = "Distribution",
) -> dict[str, Any]:
    """Deprecated alias for :func:`plot_distribution`."""
    warnings.warn(
        "plot_var_distribution is deprecated; use plot_distribution instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    thresholds = [
        (var_value, f"Threshold ({confidence_level:.0%})", "red"),
        (cvar_value, "Threshold (extreme)", "darkred"),
    ]
    return plot_distribution(returns, thresholds=thresholds, title=title)