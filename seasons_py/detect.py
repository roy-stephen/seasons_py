"""
Core detection logic for seasons_py.

Given a detrended series (only seasonality + noise), we brute-force test every
candidate integer period s from 2 to n//2 by "folding" the series into a matrix
of shape (rows x s) and running a one-way ANOVA on the columns. A low p-value
indicates that the column means differ significantly, which we interpret as
evidence that s is a seasonal period.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class PeriodResult:
    """Result of testing a single candidate period."""

    period: int
    p_value: float
    f_statistic: float
    n_rows: int  # number of complete rows after folding
    column_means: np.ndarray  # mean of each column (seasonal profile)

    def __repr__(self) -> str:
        return (
            f"PeriodResult(period={self.period}, p_value={self.p_value:.2e}, "
            f"f_stat={self.f_statistic:.2f}, n_rows={self.n_rows})"
        )


def fold_series(series: np.ndarray, period: int) -> np.ndarray:
    """
    Fold a 1-D series into a 2-D matrix of shape (rows x period).

    Trailing elements that don't fill a complete row are dropped.

    Parameters
    ----------
    series : array-like, shape (n,)
        The detrended time series.
    period : int
        Candidate seasonal period (number of columns).

    Returns
    -------
    np.ndarray, shape (n // period, period)
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    n_complete = (n // period) * period
    trimmed = series[:n_complete]
    return trimmed.reshape(-1, period)


def anova_pvalue(folded: np.ndarray) -> tuple[float, float]:
    """
    Run one-way ANOVA on the columns of a folded matrix.

    Each column is treated as a group. Returns (f_statistic, p_value).

    Parameters
    ----------
    folded : np.ndarray, shape (rows, period)
        Matrix obtained from fold_series().

    Returns
    -------
    (f_statistic, p_value)
    """
    # Each column is a group: pass as unpacked arguments to f_oneway
    groups = [folded[:, col] for col in range(folded.shape[1])]
    f_stat, p_value = stats.f_oneway(*groups)
    return float(f_stat), float(p_value)


def scan_periods(
    series: np.ndarray,
    min_period: int = 2,
    max_period: Optional[int] = None,
    alpha: float = 0.05,
) -> list[PeriodResult]:
    """
    Scan all candidate periods and return results sorted by p-value (ascending).

    Parameters
    ----------
    series : array-like, shape (n,)
        The detrended time series.
    min_period : int
        Minimum candidate period to test (default 2).
    max_period : int or None
        Maximum candidate period to test. If None, defaults to n // 2.
    alpha : float
        Significance level for flagging a period as significant.

    Returns
    -------
    list of PeriodResult, sorted by p_value ascending.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    if max_period is None:
        max_period = n // 2

    # Need at least 2 rows to run ANOVA (otherwise variance is undefined)
    results: list[PeriodResult] = []
    for s in range(min_period, max_period + 1):
        n_rows = n // s
        if n_rows < 2:
            continue
        folded = fold_series(series, s)
        f_stat, p_val = anova_pvalue(folded)
        col_means = folded.mean(axis=0)
        results.append(
            PeriodResult(
                period=s,
                p_value=p_val,
                f_statistic=f_stat,
                n_rows=n_rows,
                column_means=col_means,
            )
        )

    results.sort(key=lambda r: r.p_value)
    return results


def detect_seasonality(
    series: np.ndarray,
    min_period: int = 2,
    max_period: Optional[int] = None,
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> Optional[PeriodResult]:
    """
    Detect the most significant seasonal period in a detrended series.

    Returns the PeriodResult with the lowest p-value if it passes the
    significance threshold (p < adjusted_alpha), otherwise None.

    Since we test many candidate periods simultaneously, a multiple-comparison
    correction is applied to control the family-wise error rate:

    - "bonferroni": adjusted_alpha = alpha / n_tests  (conservative, default)
    - "none": adjusted_alpha = alpha  (no correction, more false positives)

    Parameters
    ----------
    series : array-like, shape (n,)
        The detrended time series.
    min_period : int
        Minimum candidate period (default 2).
    max_period : int or None
        Maximum candidate period. If None, defaults to n // 2.
    alpha : float
        Desired family-wise significance level (default 0.05).
    correction : str
        Multiple-comparison correction method: "bonferroni" or "none".

    Returns
    -------
    PeriodResult or None
    """
    results = scan_periods(series, min_period, max_period, alpha)
    if not results:
        return None

    best = results[0]

    if correction == "bonferroni":
        n_tests = len(results)
        adjusted_alpha = alpha / n_tests
    else:
        adjusted_alpha = alpha

    if best.p_value < adjusted_alpha:
        return best
    return None