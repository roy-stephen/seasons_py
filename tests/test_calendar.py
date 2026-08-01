"""Tests for calendar seasonality support."""
import numpy as np
import pandas as pd
from seasons_py.calendar import (
    calendar_phases,
    extract_calendar_seasonality,
    select_calendar_seasonality,
)


def test_calendar_phases_daily():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    phases = calendar_phases(idx, ["dow", "dom", "month"])
    # 2024-01-01 is a Monday -> dow=0
    np.testing.assert_array_equal(phases["dow"], np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2]))
    np.testing.assert_array_equal(phases["dom"], np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    np.testing.assert_array_equal(phases["month"], np.array([1] * 10))
    print("PASS: calendar_phases for daily index")


def test_extract_calendar_dow():
    """Series with a pure day-of-week effect should be recovered."""
    np.random.seed(30)
    idx = pd.date_range("2024-01-01", periods=70, freq="D")  # 10 full weeks
    # Monday=0 high, Friday=4 low, weekend medium.
    profile = np.array([3.0, 1.0, 1.0, 1.0, -3.0, 0.0, 0.0])
    series = profile[idx.dayofweek] + np.random.normal(0, 0.3, len(idx))

    out = extract_calendar_seasonality(series, idx, ["dow"])
    result = out["result"]
    assert set(out["rules"]) == {"dow"}
    assert out["rule_periods"]["dow"] == 7
    assert result.total_explained_var > 0.85
    # Profile should be close up to ordering/centering (zero-sum constraint shifts levels).
    np.testing.assert_allclose(result.components_by_rule["dow"].profile, profile, atol=0.8)
    print("PASS: extract_calendar_seasonality recovers DOW effect")


def test_extract_calendar_month_and_dow():
    """Joint extraction of month-of-year and day-of-week effects."""
    np.random.seed(31)
    idx = pd.date_range("2023-01-01", periods=365 * 2, freq="D")
    dow_profile = np.array([2.0, 0.5, 0.5, 0.5, -2.0, 0.0, 0.0])
    month_profile = np.array([5.0, 4.0, 2.0, 0.0, -2.0, -4.0, -5.0, -4.0, -2.0, 0.0, 2.0, 4.0])
    series = (
        dow_profile[idx.dayofweek]
        + month_profile[idx.month - 1]
        + np.random.normal(0, 0.5, len(idx))
    )

    out = extract_calendar_seasonality(series, idx, ["dow", "month"])
    result = out["result"]
    assert set(out["rules"]) == {"dow", "month"}
    assert result.total_explained_var > 0.85
    assert out["rule_periods"]["dow"] == 7
    assert out["rule_periods"]["month"] == 13  # width includes unused index 0
    print("PASS: joint calendar extraction recovers DOW + month effects")


def test_extract_calendar_dom_sparse():
    """Day-of-month on a full daily series: profile length should cover all observed days."""
    np.random.seed(32)
    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    # Boost the 1st of every month.
    dom_profile = np.zeros(31)
    dom_profile[0] = 4.0
    dom_profile[14] = -2.0
    series = dom_profile[idx.day - 1] + np.random.normal(0, 0.5, len(idx))

    out = extract_calendar_seasonality(series, idx, ["dom"])
    result = out["result"]
    assert out["rule_periods"]["dom"] == 32  # width includes unused index 0
    assert result.total_explained_var > 0.30
    assert result.components_by_rule["dom"].profile[1] > 3.0
    print("PASS: DOM extraction recovers 1st-of-month spike")


def test_extract_calendar_gapped_index():
    """A business-daily index (Mon-Fri only) still gives a valid DOW profile."""
    np.random.seed(33)
    idx = pd.bdate_range("2024-01-01", periods=50)
    # Saturday/Sunday phases never appear, but dow still produces 0..4.
    profile = np.array([2.0, 1.0, 0.0, -1.0, -2.0, 0.0, 0.0])
    series = profile[idx.dayofweek] + np.random.normal(0, 0.3, len(idx))

    out = extract_calendar_seasonality(series, idx, ["dow"])
    result = out["result"]
    assert out["rule_periods"]["dow"] == 5  # max observed phase + 1
    assert result.total_explained_var > 0.80
    print("PASS: gapped business-daily index handled")


def test_week_of_year_phases():
    """week_of_year should produce ISO week numbers 1..53."""
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    phases = calendar_phases(idx, ["week_of_year"])
    assert phases["week_of_year"].min() >= 1
    assert phases["week_of_year"].max() <= 53
    print("PASS: week_of_year phases")


def test_week_of_month_chunks():
    """week_of_month splits each month into 7-day chunks."""
    idx = pd.date_range("2024-01-01", periods=31, freq="D")
    phases = calendar_phases(idx, ["week_of_month"])
    expected = np.array([1] * 7 + [2] * 7 + [3] * 7 + [4] * 7 + [5] * 3)
    np.testing.assert_array_equal(phases["week_of_month"], expected)
    print("PASS: week_of_month 7-day chunks")


def test_week_of_month_monday():
    """week_of_month_monday starts weeks on Monday. Jan 2024 starts on Monday."""
    idx = pd.date_range("2024-01-01", periods=31, freq="D")
    phases = calendar_phases(idx, ["week_of_month_monday"])
    # Jan 2024 starts Monday, so weeks align with 7-day chunks.
    expected = np.array([1] * 7 + [2] * 7 + [3] * 7 + [4] * 7 + [5] * 3)
    np.testing.assert_array_equal(phases["week_of_month_monday"], expected)
    print("PASS: week_of_month_monday starts on Monday")


def test_select_calendar_seasonality_auto():
    """Auto-selection should pick relevant calendar rules and ignore irrelevant ones."""
    np.random.seed(77)
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    # Build a series with only month and dow effects.
    month_profile = np.array([4.0, 3.0, 2.0, 0.0, -1.0, -3.0, -4.0, -3.0, -2.0, 0.0, 1.0, 3.0])
    month_profile = month_profile - month_profile.mean()
    dow_profile = np.array([2.0, 1.0, 0.0, 0.0, -1.0, -1.5, -0.5])
    dow_profile = dow_profile - dow_profile.mean()
    series = month_profile[idx.month - 1] + dow_profile[idx.dayofweek] + np.random.normal(0, 0.5, len(idx))

    out = select_calendar_seasonality(series, idx, rules=None, criterion="bic", max_rules=5)
    selected = out["selected_rules"]
    print(f"Auto-selected calendar rules: {selected}")
    assert "month" in selected, "month should be selected"
    assert "dow" in selected, "dow should be selected"
    assert out["result"].total_explained_var > 0.80
    print("PASS: auto calendar selection picks relevant rules")


if __name__ == "__main__":
    test_calendar_phases_daily()
    test_extract_calendar_dow()
    test_extract_calendar_month_and_dow()
    test_extract_calendar_dom_sparse()
    test_extract_calendar_gapped_index()
    test_week_of_year_phases()
    test_week_of_month_chunks()
    test_week_of_month_monday()
    test_select_calendar_seasonality_auto()
    print("\nAll calendar tests passed!")