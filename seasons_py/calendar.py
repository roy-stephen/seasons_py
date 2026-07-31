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

The phase values are kept as small non-negative integers suitable for use as
period lengths in the integer-period extractor.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


_RULES = {
    "dow": lambda dt: dt.dayofweek,
    "month": lambda dt: dt.month,
    "dom": lambda dt: dt.day,
    "doy": lambda dt: dt.dayofyear,
    "quarter": lambda dt: dt.quarter,
    "hour": lambda dt: dt.hour,
}


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

    Examples
    --------
    >>> import pandas as pd
    >>> from seasons_py.calendar import calendar_phases
    >>> idx = pd.date_range("2020-01-01", periods=10, freq="D")
    >>> calendar_phases(idx, ["dow", "dom"])
    {"dow": array([2, 3, 4, 5, 6, 0, 1, 2, 3, 4]), ...}
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("calendar_phases requires pandas") from exc

    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    phases: dict[str, np.ndarray] = {}
    for rule in rules:
        if rule not in _RULES:
            raise ValueError(
                f"Unknown calendar rule '{rule}'. Supported: {list(_RULES.keys())}"
            )
        phases[rule] = np.asarray(_RULES[rule](index), dtype=int)
    return phases


def rule_to_period(rule: str, phases: np.ndarray) -> int:
    """
    Return the effective one-hot matrix width for a calendar rule.

    The width must be large enough to index every observed phase value. For
    0-indexed rules (dow, hour) this is max(phase) + 1. For 1-indexed rules
    (month, dom, doy, quarter) this is also max(phase) + 1, because the array
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