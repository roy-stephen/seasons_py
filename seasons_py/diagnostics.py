"""
Diagnostic utilities for inspecting learned seasonal profiles.

The main idea: a learned profile (e.g. a 365-day day-of-year effect) may itself
contain shorter periodic structure. This module provides tools to inspect that
internal structure without automatically changing the fitted model.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from seasons_py.detect import scan_periods, PeriodResult
from seasons_py.extract import extract_multiple_seasonalities


def profile_periodogram(
    profile: np.ndarray,
    max_period: Optional[int] = None,
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> list[tuple[int, float]]:
    """
    Scan a single learned seasonal profile for internal periodic structure.

    Parameters
    ----------
    profile : array-like, shape (period,)
        The learned profile for one seasonal component (e.g. from
        SeasonalityResult.profile).
    max_period : int or None
        Maximum sub-period to test. Defaults to len(profile) // 2.
    alpha : float
        Family-wise significance level.
    correction : str
        Multiple-comparison correction: "bonferroni" or "none".

    Returns
    -------
    list[tuple[int, float]]
        Significant sub-periods and the share of the profile variance they
        explain, sorted by explained share descending.

    Notes
    -----
    This is a diagnostic, not a model-selection step. The returned sub-periods
    describe the internal shape of the profile. Whether to add them to the
    main model is a separate decision.
    """
    profile = np.asarray(profile, dtype=float)
    n = len(profile)
    if n < 4:
        return []

    results = scan_periods(profile, min_period=2, max_period=max_period, alpha=alpha)
    if not results:
        return []

    n_tests = len(results)
    adjusted_alpha = alpha / n_tests if correction == "bonferroni" else alpha
    significant = [r for r in results if r.p_value < adjusted_alpha]

    if not significant:
        return []

    # Centered profile for variance accounting.
    y = profile - profile.mean()
    total_var = float(np.var(y, ddof=1))

    out: list[tuple[int, float]] = []
    for r in significant:
        s = r.period
        # Fit a sub-period seasonality to the profile itself.
        fit = extract_multiple_seasonalities(y, [s])
        explained = (
            1.0 - np.var(fit.residual, ddof=1) / total_var
            if total_var > 0
            else 0.0
        )
        out.append((s, float(max(0.0, explained))))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


def result_profile_periodogram(
    result,
    max_period: Optional[int] = None,
    alpha: float = 0.05,
    correction: str = "bonferroni",
    exclude_selected: bool = True,
) -> dict[int | str, list[tuple[int, float]]]:
    """
    Run profile_periodogram on every component of a MultiSeasonalityResult.

    Parameters
    ----------
    result : MultiSeasonalityResult or calendar dict
        The fitted result. For calendar results, components_by_rule is used.
    max_period, alpha, correction
        Passed to profile_periodogram.
    exclude_selected : bool
        If True, sub-periods that are already selected periods/rules are omitted.

    Returns
    -------
    dict
        Mapping from component identifier (period or rule name) to a list of
        (sub_period, explained_share) tuples.
    """
    from seasons_py.extract import MultiSeasonalityResult

    # If a calendar result dict was passed, unwrap the inner MultiSeasonalityResult.
    if isinstance(result, dict) and "result" in result and hasattr(result["result"], "components_by_rule"):
        inner = result["result"]
        components = {rule: comp.profile for rule, comp in inner.components_by_rule.items()}
        selected_periods = set(inner.periods)
    elif hasattr(result, "components_by_rule"):
        components = {rule: comp.profile for rule, comp in result.components_by_rule.items()}
        selected_periods = set()
    elif isinstance(result, MultiSeasonalityResult):
        components = {p: comp.profile for p, comp in result.components.items()}
        selected_periods = set(result.periods)
    else:
        raise TypeError("result must be a MultiSeasonalityResult or a calendar result dict")

    diagnostics: dict[int | str, list[tuple[int, float]]] = {}
    for key, profile in components.items():
        raw = profile_periodogram(profile, max_period=max_period, alpha=alpha, correction=correction)
        if exclude_selected:
            raw = [(s, share) for s, share in raw if s not in selected_periods]
        diagnostics[key] = raw
    return diagnostics
