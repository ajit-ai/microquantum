"""CSV and DataFrame ingestion for analytics modules.

Provides utilities to load data from CSV files or numpy arrays.
"""
from __future__ import annotations

import csv
import os

import numpy as np


def load_csv(
    filepath: str,
    delimiter: str = ",",
    has_header: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Load data from a CSV file.

    Args:
        filepath: Path to the CSV file.
        delimiter: Column delimiter.
        has_header: If True, first row is treated as header.

    Returns:
        Tuple of (column_names, data_array).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty")

    if has_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        headers = [f"col_{i}" for i in range(len(rows[0]))]
        data_rows = rows

    if not data_rows:
        raise ValueError("CSV file has no data rows")

    try:
        data = np.array(data_rows, dtype=float)
    except ValueError as e:
        raise ValueError(f"Cannot convert CSV data to numeric array: {e}") from e

    return headers, data


def load_csv_column(
    filepath: str,
    column: str,
    delimiter: str = ",",
) -> np.ndarray:
    """Load a single column from a CSV file.

    Args:
        filepath: Path to the CSV file.
        column: Column name to load.
        delimiter: Column delimiter.

    Returns:
        1D numpy array of the column values.
    """
    headers, data = load_csv(filepath, delimiter)

    if column not in headers:
        raise ValueError(f"Column '{column}' not found. Available: {headers}")

    col_idx = headers.index(column)
    return data[:, col_idx]


def load_distance_matrix(filepath: str, delimiter: str = ",") -> np.ndarray:
    """Load a distance matrix from a CSV file.

    The CSV should have cities as rows and columns, with distances.
    First row and column can be city names (optional).

    Args:
        filepath: Path to the CSV file.
        delimiter: Column delimiter.

    Returns:
        2D numpy array of distances.
    """
    headers, data = load_csv(filepath, delimiter, has_header=False)

    if data.shape[0] != data.shape[1]:
        raise ValueError(
            f"Distance matrix must be square, got {data.shape[0]}x{data.shape[1]}"
        )

    return data


def arrays_to_csv(
    filepath: str,
    headers: list[str],
    data: np.ndarray,
    delimiter: str = ",",
) -> None:
    """Save numpy arrays to a CSV file.

    Args:
        filepath: Path to save the CSV file.
        headers: Column names.
        data: 2D numpy array of data.
        delimiter: Column delimiter.
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(data.tolist())
