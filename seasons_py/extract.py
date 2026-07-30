"""
Extract multiple known seasonal effects from a detrended series via joint OLS.

When the seasonal periods are known, we do not need to search or iterate. We
fit all seasonal profiles simultaneously with ordinary least squares. This is
order-independent and computationally light.

For identifiability, each seasonal component is constrained to have a zero mean
over one full cycle (sum-to-zero constraint). This removes the global level and
keeps the per-period subspaces cleanly separated.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SeasonalityResult:
    """The extracted effect of a single integer seasonal period."""

    period: int
    profile: np.ndarray       # per-phase coefficient, shape (period,)
    fitted: np.ndarray        # tiled profile to full length, shape (n,)
    explained_var: float      # this component's share of total fitted variance

    def __repr__(self) -> str:
        return (
            f"SeasonalityResult(period={self.period}, "
            f"explained_var={self.explained_var:.3f})"
        )


@dataclass
class MultiSeasonalityResult:
    """Result of fitting multiple seasonalities jointly."""

    periods: list[int]
    components: dict[int, SeasonalityResult]
    fitted: np.ndarray        # sum of all fitted components, shape (n,)
    residual: np.ndarray      # input - fitted, shape (n,)
    total_explained_var: float

    def __repr__(self) -> str:
        return (
            f"MultiSeasonalityResult(periods={self.periods}, "
            f"total_explained_var={self.total_explained_var:.3f})"
        )


def _build_design_matrix(n: int, periods: list[int]) -> tuple[np.ndarray, list[int]]:
    """
    Build the design matrix for a joint regression of multiple seasonalities.

    Each period contributes `period - 1` columns with a sum-to-zero constraint:
    the coefficient for the last phase is minus the sum of the others. The j-th
    column is +1 when phase == j and -1 when phase == last, so that the fitted
    profile tiles correctly.
    """
    blocks: list[np.ndarray] = []
    offsets: list[int] = []
    offset = 0

    for s in periods:
        offsets.append(offset)
        phase = np.arange(n) % s
        onehot = np.zeros((n, s), dtype=float)
        onehot[np.arange(n), phase] = 1.0
        # Sum-to-zero: columns 0..s-2 relative to the dropped last column.
        block = onehot[:, :-1] - onehot[:, -1:]
        blocks.append(block)
        offset += s - 1

    X = np.column_stack(blocks) if blocks else np.zeros((n, 0))
    return X, offsets

def _profile_from_coeffs(coeffs: np.ndarray, period: int) -> np.ndarray:
    """Recover the full period-length zero-sum profile from free coefficients."""
    if period == 1:
        return np.zeros(1)
    free = coeffs[: period - 1]
    return np.append(free, -np.sum(free))


def _model_rss_and_k(series: np.ndarray, periods: list[int]) -> tuple[float, float]:
    """Return (rss, k) for the model with periods p."""
    if not periods:
        rss = float(np.sum((series - series.mean()) ** 2))
        return rss, 0.0
    fit = extract_multiple_seasonalities(series, periods)
    rss = float(np.sum(fit.residual ** 2))
    k = sum(max(s - 1, 1) for s in periods)
    return rss, k


def extract_multiple_seasonalities(
    series: np.ndarray,
    periods: list[int],
) -> MultiSeasonalityResult:
    """
    Fit multiple integer seasonalities jointly via OLS.

    Parameters
    ----------
    series : array-like, shape (n,)
        The (assumed detrended) time series.
    periods : list[int]
        Integer periods to fit jointly.

    Returns
    -------
    MultiSeasonalityResult
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    periods = [int(s) for s in periods]

    if not periods:
        return MultiSeasonalityResult(
            periods=[],
            components={},
            fitted=np.zeros(n),
            residual=series.copy(),
            total_explained_var=0.0,
        )

    y = series - series.mean()
    X, offsets = _build_design_matrix(n, periods)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)

    components: dict[int, SeasonalityResult] = {}
    fitted_total = np.zeros(n)
    for i, s in enumerate(periods):
        start = offsets[i]
        period_coeffs = coeffs[start : start + s - 1] if s > 1 else np.array([])
        profile = _profile_from_coeffs(period_coeffs, s)
        fitted = np.tile(profile, (n // s) + 1)[:n]
        fitted_total += fitted
        components[s] = SeasonalityResult(
            period=s,
            profile=profile,
            fitted=fitted,
            explained_var=0.0,
        )

    total_fitted_var = np.var(fitted_total, ddof=1)
    for s in periods:
        comp_var = np.var(components[s].fitted, ddof=1)
        components[s].explained_var = (
            comp_var / total_fitted_var if total_fitted_var > 0 else 0.0
        )

    residual = series - fitted_total
    total_explained_var = (
        1.0 - np.var(residual, ddof=1) / np.var(series, ddof=1)
        if np.var(series, ddof=1) > 0
        else 0.0
    )

    return MultiSeasonalityResult(
        periods=periods,
        components=components,
        fitted=fitted_total,
        residual=residual,
        total_explained_var=total_explained_var,
    )


def _find_divisor_replacement(
    series: np.ndarray,
    current_periods: list[int],
    candidate_pool: set[int],
    criterion: str = "bic",
) -> tuple[float, list[int]]:
    """
    Check whether replacing any selected period by a set of its divisors
    (from candidate_pool) improves the model score.

    Returns (best_score, best_periods). If no improvement, returns the current
    score and periods.
    """
    n = len(series)

    def score(periods: list[int]) -> float:
        rss, k = _model_rss_and_k(series, periods)
        if criterion == "bic":
            return n * np.log(max(rss / n, 1e-300)) + k * np.log(n)
        return n * np.log(max(rss / n, 1e-300)) + 2 * k

    current_score = score(current_periods)
    best_score = current_score
    best_periods = list(current_periods)

    for s in current_periods:
        # Find divisors of s that are available in the candidate pool.
        divisors = [d for d in candidate_pool if d < s and s % d == 0]
        if not divisors:
            continue

        base = [p for p in current_periods if p != s]

        # Try replacing s with single divisor, then all divisors together.
        for r in range(1, min(len(divisors), 3) + 1):
            from itertools import combinations
            for combo in combinations(divisors, r):
                trial = sorted(set(base + list(combo)))
                trial_score = score(trial)
                if trial_score < best_score:
                    best_score = trial_score
                    best_periods = trial

    return best_score, best_periods


def select_seasonalities(
    series: np.ndarray,
    candidate_periods: list[int],
    criterion: str = "bic",
    p_values: Optional[dict[int, float]] = None,
    prefer_short: bool = True,
    max_periods: Optional[int] = None,
) -> list[int]:
    """
    Forward-select seasonal periods using a model-selection criterion, followed
    by a factorize-and-improve pass that replaces selected periods by their
    divisors whenever that improves BIC/AIC.

    This handles the LCM trap: {21} is replaced by {3, 7} because the latter
    explains the same signal with fewer parameters.

    Parameters
    ----------
    series : array-like, shape (n,)
        The (assumed detrended) time series.
    candidate_periods : list[int]
        Candidate periods, typically from detect.scan_periods.
    criterion : str
        "bic" (default, stronger penalty) or "aic".
    p_values : dict[int, float] or None
        Optional p-values used for length-penalized ranking when prefer_short=True.
    prefer_short : bool
        If True, rank candidates to prefer shorter fundamentals.
    max_periods : int or None
        Hard cap on the number of selected periods.

    Returns
    -------
    list[int]
        Selected periods in the order they were added (then reordered by value).
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    candidates = sorted(set(candidate_periods))
    candidate_pool = set(candidates)

    def bic(rss: float, k: float) -> float:
        return n * np.log(max(rss / n, 1e-300)) + k * np.log(n)

    def aic(rss: float, k: float) -> float:
        return n * np.log(max(rss / n, 1e-300)) + 2 * k

    score_fn = bic if criterion == "bic" else aic
    best_score = score_fn(*_model_rss_and_k(series, []))

    selected: list[int] = []
    remaining = set(candidates)

    while remaining:
        if max_periods is not None and len(selected) >= max_periods:
            break

        if prefer_short and p_values:
            ranked = sorted(
                remaining,
                key=lambda s: (-np.log10(p_values.get(s, 1e-300)) / s, s),
                reverse=True,
            )
        elif prefer_short:
            ranked = sorted(remaining, key=lambda s: (1.0 / s, s), reverse=True)
        else:
            ranked = sorted(remaining)

        best_new_score = best_score
        best_period: Optional[int] = None
        for s in ranked:
            score = score_fn(*_model_rss_and_k(series, selected + [s]))
            if score < best_new_score:
                best_new_score = score
                best_period = s

        if best_period is None:
            break

        selected.append(best_period)
        best_score = best_new_score
        remaining.remove(best_period)

    # Factorize-and-improve pass: replace LCMs with their divisors.
    improved = True
    while improved and len(selected) > 0:
        improved = False
        new_score, new_periods = _find_divisor_replacement(
            series, selected, candidate_pool, criterion
        )
        if new_score < best_score:
            selected = new_periods
            best_score = new_score
            improved = True

    return sorted(selected)


def prune_seasonalities(
    series: np.ndarray,
    candidate_periods: list[int],
    criterion: str = "bic",
    p_values: Optional[dict[int, float]] = None,
    prefer_short: bool = True,
) -> list[int]:
    """Wrapper around select_seasonalities over the full candidate set."""
    return select_seasonalities(
        series,
        candidate_periods,
        criterion=criterion,
        p_values=p_values,
        prefer_short=prefer_short,
    )


def extract_seasonality(series: np.ndarray, period: int) -> SeasonalityResult:
    """Backwards-compatible single-period extraction (delegates to joint fit)."""
    multi = extract_multiple_seasonalities(series, [period])
    comp = multi.components[period]
    # For single-period extraction, explained_var means total explained variance.
    comp.explained_var = multi.total_explained_var
    return comp


def iterative_detect(
    series: np.ndarray,
    n_seasons: int = 2,
    alpha: float = 0.05,
    max_period: Optional[int] = None,
    suppress_multiples: bool = True,
) -> list[SeasonalityResult]:
    """
    Greedy multi-seasonality detection with optional harmonic suppression.

    Parameters
    ----------
    series : array-like, shape (n,)
        The (assumed detrended) time series.
    n_seasons : int
        Maximum number of seasonalities to look for.
    alpha : float
        Significance threshold (passed to detect_seasonality).
    max_period : int or None
        Maximum candidate period to consider.
    suppress_multiples : bool
        If True, once a period s is selected, 2s, 3s, ... are skipped.

    Returns
    -------
    list of SeasonalityResult
        Detected seasonalities in the order they were found.
    """
    from seasons_py.detect import detect_seasonality

    residual = np.asarray(series, dtype=float).copy()
    found: list[SeasonalityResult] = []
    excluded: set[int] = set()

    for _ in range(n_seasons):
        best = detect_seasonality(residual, max_period=max_period, alpha=alpha)
        if best is None:
            break
        if suppress_multiples:
            for s in [f.period for f in found]:
                for k in range(2, (best.period // s) + 1):
                    excluded.add(s * k)
            if best.period in excluded:
                continue
        extracted = extract_seasonality(residual, best.period)
        found.append(extracted)
        # Residual comes from the joint single-period fit.
        multi = extract_multiple_seasonalities(residual, [best.period])
        residual = multi.residual

    return found