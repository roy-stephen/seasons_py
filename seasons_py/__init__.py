"""seasons_py: detect integer seasonal periods in detrended time series via ANOVA."""

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
    "detect_seasonality",
    "fold_series",
    "anova_pvalue",
    "scan_periods",
    "extract_seasonality",
    "extract_multiple_seasonalities",
    "select_seasonalities",
    "prune_seasonalities",
    "iterative_detect",
    "SeasonalityResult",
    "MultiSeasonalityResult",
]