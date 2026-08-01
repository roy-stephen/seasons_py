"""
Calendar seasonality support.

Maps a pandas DatetimeIndex to integer phase arrays for named calendar rules.
The existing joint-OLS core (`extract_multiple_seasonalities`) can then fit those
phases directly, because from the estimator's point of view a calendar season is
just another categorical variable repeated every K observations.

Supported rules:
- "dow"   : day of week    (Mon=0 ... Sun=6)
- "month" : month of year  (Jan=1 ... Dec=12)
- "dom"   : day of month   (1..31)
- "doy"   : day of year    (1..366)
- "quarter": quarter of year (1..4)
- "hour"  : hour of day    (0..23)
- "week_of_year"         : ISO week of year (1..53)
- "week_of_month"        : 7-day chunks within month (1..5)
- "week_of_month_monday": ISO-like week of month, starting on Monday (1..5)

The phase values are kept as small non-negative integers suitable for use as
period lengths in the integer-period extractor.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


_RRULES = {
    "dow": lambda dt: dt.dayofweek,
    "month": lambda dt: dt.month,
    "dom": lambda dt: dt.day,
    "doy": lambda dt: dt.dayofyear,
    "quarter": lambda dt: dt.quarter,
    "hour": lambda dt: dt.hour,
    "week_of_year": lambda dt: dt.isocalendar().week,
    "week_of_month": lambda dt: (dt.day - 1) // 7 + 1,
    "week_of_month_monday": lambda dt: _week_of_month_monday(dt),
}


def _week_of_month_monday(dt: pd.DatetimeIndex) -> np.ndarray:
    """ISO-like week-of-month where weeks start on Monday.

    Days before the first Monday of the month are assigned to week 1.
    """
    import pandas as pd

    first_day_of_month = dt.to_period("M").to_timestamp()
    first_dow = first_day_of_month.dayofweek  # Monday=0
    days_before_first_monday = ((7 - first_dow) % 7)
    days_since_first_monday = (dt.day - 1) - days_before_first_monday
    week = (days_since_first_monday // 7) + 1
    return np.asarray(np.maximum(week, 1), dtype=int)


def calendar_phases(index: pd.DatetimeIndex, rules: list[str]) -> dict[str, np.ndarray]:
    """
    Map a DatetimeIndex to integer phase arrays for named calendar rules.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The datetime index of the series.
    rules : list[str]
        Calendar rules, e.g., ["dow", "month", "dom"].

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from rule name to phase array of shape (len(index),).
        Phase values are non-negative integers.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("calendar_phases requires pandas") from exc

    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    phases: dict[str, np.ndarray] = {}
    for rule in rules:
        if rule not in _RRULES:
            raise ValueError(
                f"Unknown calendar rule '{rule}'. Supported: {list(_RRULES.keys())}"
            )
        phases[rule] = np.asarray(_RRULES[rule](index), dtype=int)
    return phases


def rule_to_period(rule: str, phases: np.ndarray) -> int:
    """
    Return the effective one-hot matrix width for a calendar rule.

    The width must be large enough to index every observed phase value. For
    0-indexed rules (dow, hour) this is max(phase) + 1. For 1-indexed rules
    (month, dom, doy, quarter, week_of_year, week_of_month,
    week_of_month_monday) this is also max(phase) + 1, because the array
    needs an index for the maximum phase; index 0 simply goes unused.
    """
    return int(np.max(phases) + 1)


def extract_calendar_seasonality(
    series: np.ndarray,
    index: pd.DatetimeIndex,
    rules: list[str],
) -> dict[str, object]:
    """
    Convenience wrapper: extract calendar seasonal effects jointly.

    Parameters
    ----------
    series : array-like, shape (n,)
        The (assumed detrended) time series.
    index : pd.DatetimeIndex
        Datetime index aligned with series.
    rules : list[str]
        Calendar rules to fit, e.g., ["dow", "month"].

    Returns
    -------
    dict[str, object]
        {
            "phases": dict of phase arrays,
            "periods": list of integer period lengths used by the estimator,
            "result": MultiSeasonalityResult from extract_multiple_seasonalities,
            "rule_periods": mapping from rule name to its integer period length,
            "rules": rules used,
        }
    """
    from seasons_py.extract import extract_multiple_seasonalities

    phases = calendar_phases(index, rules)
    period_by_rule = {rule: rule_to_period(rule, phase) for rule, phase in phases.items()}
    periods = [period_by_rule[rule] for rule in rules]
    phase_arrays = [phases[rule] for rule in rules]
    result = extract_multiple_seasonalities(series, periods, phase_arrays=phase_arrays)

    # Map raw period back to rule name for interpretability.
    components_by_rule = {
        rule: result.components[period_by_rule[rule]] for rule in rules
    }
    result.components_by_rule = components_by_rule  # type: ignore[attr-defined]

    return {
        "phases": phases,
        "periods": periods,
        "result": result,
        "rule_periods": period_by_rule,
        "rules": rules,
    }


def select_calendar_seasonality(
    series: np.ndarray,
    index: pd.DatetimeIndex,
    rules: list[str] | None = None,
    criterion: str = "bic",
    max_rules: int | None = None,
    min_obs_per_phase: int = 5,
) -> dict[str, object]:
    """
    Automatically select a parsimonious subset of calendar rules.

    Each candidate rule is evaluated individually, then a forward step adds the
    rule that most improves the chosen information criterion (BIC/AIC). The
    process stops when no remaining rule improves the score or `max_rules` is
    reached.

    Ultra-high-cardinality rules (e.g. doy with only ~2 obs/phase) are excluded
    by the `min_obs_per_phase` guard, not by a hard-coded list, so genuinely
    informative rules like `dom` are still considered.

    Parameters
    ----------
    series : array-like, shape (n,)
        The (assumed detrended) time series.
    index : pd.DatetimeIndex
        Datetime index aligned with series.
    rules : list[str] or None
        Candidate rules. If None, all supported rules are tried.
    criterion : str
        "bic" (default) or "aic".
    max_rules : int or None
        Hard cap on the number of selected rules.
    min_obs_per_phase : int
        Minimum average observations per phase for a rule to be considered.
        Rules below this are too sparse / likely to overfit.

    Returns
    -------
    dict[str, object]
        Same output shape as `extract_calendar_seasonality`, plus the key
        "selected_rules" listing the rules that were chosen.
    """
    from seasons_py.extract import extract_multiple_seasonalities

    if rules is None:
        rules = list(_RRULES.keys())

    series = np.asarray(series, dtype=float)
    n = len(series)

    def _score(rule_subset: list[str]) -> float:
        if not rule_subset:
            rss = float(np.sum((series - series.mean()) ** 2))
            k = 0
        else:
            phases = calendar_phases(index, rule_subset)
            periods = [rule_to_period(rule, phases[rule]) for rule in rule_subset]
            phase_arrays = [phases[rule] for rule in rule_subset]
            result = extract_multiple_seasonalities(series, periods, phase_arrays=phase_arrays)
            rss = float(np.sum(result.residual ** 2))
            k = sum(p - 1 for p in periods)
        penalty = np.log(n) if criterion == "bic" else 2.0
        return n * np.log(max(rss / n, 1e-300)) + k * penalty

    # Compute obs-per-phase for every candidate rule and drop sparse ones.
    phases_all = calendar_phases(index, [r for r in rules if r in _RRULES])
    usable_rules = []
    for rule in rules:
        if rule not in _RRULES:
            continue
        n_phase = int(np.max(phases_all[rule]) + 1)
        if n / n_phase >= min_obs_per_phase:
            usable_rules.append(rule)

    selected: list[str] = []
    remaining = list(usable_rules)
    best_score = _score([])

    while remaining:
        if max_rules is not None and len(selected) >= max_rules:
            break
        best_new_score = best_score
        best_rule: str | None = None
        for rule in remaining:
            trial_score = _score(selected + [rule])
            if trial_score < best_new_score:
                best_new_score = trial_score
                best_rule = rule
        if best_rule is None:
            break
        selected.append(best_rule)
        remaining.remove(best_rule)
        best_score = best_new_score

    out = extract_calendar_seasonality(series, index, selected)
    out["selected_rules"] = selected
    return out
