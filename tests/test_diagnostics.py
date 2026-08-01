"""Tests for diagnostic profile-periodogram utilities."""
import numpy as np
from seasons_py.diagnostics import profile_periodogram, result_profile_periodogram
from seasons_py import extract_multiple_seasonalities, extract_calendar_seasonality
import pandas as pd


def test_profile_periodogram_finds_sub_period():
    """A 12-phase profile built from periods 3 and 4 should reveal at least one sub-period."""
    np.random.seed(0)
    p3 = np.tile(np.array([2.0, -1.0, -1.0]), 4)
    p4 = np.tile(np.array([1.0, 0.5, -0.5, -1.0]), 3)
    profile = p3 + p4
    profile = profile - profile.mean()

    subs = profile_periodogram(profile, alpha=0.05)
    periods = {s for s, _ in subs}
    assert 3 in periods or 4 in periods, f"expected period 3 or 4 in {subs}"
    print("PASS: profile_periodogram finds internal sub-periods")


def test_result_profile_periodogram_excludes_selected():
    """Excludes sub-periods that are already selected in the main model."""
    np.random.seed(1)
    _cycle = 21
    base = np.zeros(_cycle)
    base += np.tile(np.array([2.0, -1.0, -1.0]), _cycle // 3)
    base += np.tile(np.array([1.0, 0.5, -0.5, -1.0, 0.0, -1.0, 1.0]), _cycle // 7)
    series = np.tile(base, 50)[:500] + np.random.normal(0, 0.3, 500)

    multi = extract_multiple_seasonalities(series, [3, 7])
    diag = result_profile_periodogram(multi, exclude_selected=True)

    # The 3-period component should not report 7 as internal because 7 is selected.
    # The 7-period component should not report 3 as internal because 3 is selected.
    for p, subs in diag.items():
        selected_subs = {s for s, _ in subs}
        assert selected_subs.isdisjoint({3, 7}), f"component {p} returned selected sub-periods: {selected_subs}"
    print("PASS: result_profile_periodogram excludes already-selected periods")


def test_result_profile_periodogram_calendar():
    """Works on calendar results keyed by rule name."""
    np.random.seed(2)
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    month_profile = np.array([4.0, 3.0, 2.0, 0.0, -1.0, -3.0, -4.0, -3.0, -2.0, 0.0, 1.0, 3.0])
    month_profile = month_profile - month_profile.mean()
    series = month_profile[idx.month - 1] + np.random.normal(0, 0.5, len(idx))

    cal = extract_calendar_seasonality(series, idx, ["month"])
    diag = result_profile_periodogram(cal, exclude_selected=False)
    assert "month" in diag
    # A 12-month profile often has sub-structure; accept no crash and reasonable output.
    assert isinstance(diag["month"], list)
    print("PASS: result_profile_periodogram handles calendar result dict")


if __name__ == "__main__":
    test_profile_periodogram_finds_sub_period()
    test_result_profile_periodogram_excludes_selected()
    test_result_profile_periodogram_calendar()
    print("\nAll diagnostics tests passed!")
