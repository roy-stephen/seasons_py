"""Detrending utilities for preprocessing time series.

The rest of seasons_py assumes the input is already detrended. This module provides
light, deterministic detrenders so users can build a complete raw → detrended →
seasonal pipeline in one place.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def linear_detrend(series: np.ndarray) -> np.ndarray:
    """
    Remove a linear trend from a 1-D series using ordinary least squares.

    Parameters
    ----------
    series : array-like, shape (n,)
        The raw time series.

    Returns
    -------
    np.ndarray, shape (n,)
        The input series minus the fitted linear trend.

    Examples
    --------
    >>> import numpy as np
    >>> from seasons_py.detrend import linear_detrend
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> linear_detrend(y)
    array([0., 0., 0., 0., 0.])
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError("linear_detrend expects a 1-D array")
    n = len(series)
    if n < 2:
        return series.copy()

    x = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, series, rcond=None)
    trend = X @ beta
    return series - trend


def mean_detrend(series: np.ndarray) -> np.ndarray:
    """Subtract the sample mean (the trivial detrender)."""
    return np.asarray(series, dtype=float) - np.mean(series)


def detrend(
    series: np.ndarray,
    method: str = "linear",
    inplace: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detrend a series and optionally return the removed trend.

    Parameters
    ----------
    series : array-like, shape (n,)
        Raw time series.
    method : str
        "linear" (default) or "mean".
    inplace : bool
        If True, the returned array is a new array regardless; the parameter is
        kept for API symmetry.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (detrended_series, trend)
    """
    series = np.asarray(series, dtype=float)
    if method == "linear":
        n = len(series)
        x = np.arange(n, dtype=float)
        X = np.column_stack([np.ones(n), x])
        beta, *_ = np.linalg.lstsq(X, series, rcond=None)
        trend = X @ beta
    elif method == "mean":
        trend = np.full_like(series, np.mean(series))
    else:
        raise ValueError(f"Unknown detrend method '{method}'. Use 'linear' or 'mean'.")

    detrended = series - trend
    return detrended, trend


__all__ = ["linear_detrend", "mean_detrend", "detrend"]
