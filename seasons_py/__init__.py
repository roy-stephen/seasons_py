"""seasons_py: detect integer and calendar seasonal periods in detrended time series."""

from seasons_py.calendar import (
    calendar_phases,
    extract_calendar_seasonality,
    select_calendar_seasonality,
)
from seasons_py.detect import detect_seasonality, fold_series, anova_pvalue, scan_periods
from seasons_py.extract import (
    extract_seasonality,
    extract_multiple_seasonalities,
    select_seasonalities,
    prune_seasonalities,
    iterative_detect,
    SeasonalityResult,
    MultiSeasonalityResult,
)

__all__ = [
    "calendar_phases",
    "detect_seasonality",
    "extract_calendar_seasonality",
    "select_calendar_seasonality",
    "extract_multiple_seasonalities",
    "extract_seasonality",
    "fold_series",
    "anova_pvalue",
    "iterative_detect",
    "prune_seasonalities",
    "scan_periods",
    "select_seasonalities",
    "SeasonalityResult",
    "MultiSeasonalityResult",
]