"""Analytics toolkit for microquantum.

Provides the standardized result contract used by downstream
proprietary solvers (published separately) plus generic
data-loading and visualization helpers that depend only on NumPy.
"""
from .base import AnalysisResult, BaseAnalytics
from .csv_loader import load_csv, load_csv_column, load_distance_matrix
from .result import Result
from .visualization import (
    plot_kernel_matrix,
    plot_optimization_history,
    plot_portfolio_allocation,
    plot_risk_return_scatter,
    plot_comparison_bar,
    plot_route_map,
    plot_var_distribution,
    format_analysis_report,
)

__all__ = [
    "AnalysisResult",
    "BaseAnalytics",
    "Result",
    "load_csv",
    "load_csv_column",
    "load_distance_matrix",
    "plot_kernel_matrix",
    "plot_optimization_history",
    "plot_portfolio_allocation",
    "plot_risk_return_scatter",
    "plot_comparison_bar",
    "plot_route_map",
    "plot_var_distribution",
    "format_analysis_report",
]