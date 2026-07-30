"""
Extract the seasonal effect from a detrended series once a period is known.

This module provides the building block for iterative multi-seasonality
detection: given a candidate period s, compute the seasonal profile,
reconstruct the fitted seasonal signal, and return the residual.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SeasonalityResult:
    """The extracted effect of a single integer seasonal period."""

    period: int
    profile: np.ndarray       # per-phase mean, shape (period,)
    std: np.ndarray           # per-phase std, shape (period,)
    count: np.ndarray         # number of observations per phase, shape (period,)
    fitted: np.ndarray        # tiled profile to full length, shape (n,)
    residual: np.ndarray      # input - fitted, shape (n,)
    explained_var: float      # 1 - Var(residual) / Var(input), capped at (-inf, 1]

    def __repr__(self) -> str:
        return (
            f"SeasonalityResult(period={self.period}, "
            f"explained_var={self.explained_var:.3f})"
        )


def extract_seasonality(series: np.ndarray, period: int) -> SeasonalityResult:
    """
    Extract the seasonal effect of a given integer period from a series.

    The input series is assumed to be detrended. The fitted seasonal signal is
    the per-phase mean (the profile) tiled to the original series length. The
    residual is simply series - fitted.

    Parameters
    ----------
    series : array-like, shape (n,)
        The (assumed detrended) time series.
    period : int
        Seasonal period to extract.

    Returns
    -------
    SeasonalityResult
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    # We only have complete cycles if n divides evenly. For the profile we use
    # complete cycles (same convention as detect.py); the fitted signal extends
    # the profile over the full length, including any trailing partial cycle.
    n_complete = (n // period) * period
    complete = series[:n_complete].reshape(-1, period)

    profile = complete.mean(axis=0)
    std = complete.std(axis=0, ddof=1)
    count = np.full(period, complete.shape[0], dtype=int)

    # Tile profile to full length n.
    fitted = np.tile(profile, (n // period) + 1)[:n]
    residual = series - fitted

    var_input = np.var(series, ddof=1)
    var_residual = np.var(residual, ddof=1)
    if var_input == 0:
        explained_var = 0.0
    else:
        explained_var = 1.0 - var_residual / var_input

    return SeasonalityResult(
        period=period,
        profile=profile,
        std=std,
        count=count,
        fitted=fitted,
        residual=residual,
        explained_var=explained_var,
    )


def iterative_detect(
    series: np.ndarray,
    n_seasons: int = 2,
    alpha: float = 0.05,
) -> list[SeasonalityResult]:
    """
    Greedy multi-seasonality detection: detect, extract, remove, repeat.

    Parameters
    ----------
    series : array-like, shape (n,)
        The (assumed detrended) time series.
    n_seasons : int
        Maximum number of seasonalities to look for.
    alpha : float
        Significance threshold (passed to detect_seasonality).

    Returns
    -------
    list of SeasonalityResult
        Detected seasonalities in the order they were found.
    """
    from seasons_py.detect import detect_seasonality

    residual = np.asarray(series, dtype=float).copy()
    found: list[SeasonalityResult] = []

    for _ in range(n_seasons):
        best = detect_seasonality(residual, alpha=alpha)
        if best is None:
            break
        extracted = extract_seasonality(residual, best.period)
        found.append(extracted)
        residual = extracted.residual

    return found