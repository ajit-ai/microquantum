"""Analytics toolkit for microquantum.

Provides the standardized, domain-agnostic result contract plus generic
data-loading and visualization helpers that depend only on NumPy.
"""
from .base import AnalysisResult, BaseAnalytics
from .csv_loader import load_csv, load_csv_column, load_distance_matrix
from .result import Result
from .visualization import (
    format_analysis_report,
    plot_allocation,
    plot_comparison_bar,
    plot_distribution,
    plot_kernel_matrix,
    plot_optimization_history,
    plot_portfolio_allocation,
    plot_risk_return_scatter,
    plot_route_map,
    plot_scatter,
    plot_var_distribution,
)

__all__ = [
    "AnalysisResult",
    "BaseAnalytics",
    "Result",
    "load_csv",
    "load_csv_column",
    "load_distance_matrix",
    "plot_allocation",
    "plot_scatter",
    "plot_distribution",
    "plot_kernel_matrix",
    "plot_optimization_history",
    "plot_comparison_bar",
    "plot_route_map",
    "format_analysis_report",
    # Deprecated aliases (pre-MQ-01) — kept for backward compatibility
    "plot_portfolio_allocation",
    "plot_risk_return_scatter",
    "plot_var_distribution",
]