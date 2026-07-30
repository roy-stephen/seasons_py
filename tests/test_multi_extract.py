"""Tests for joint multi-seasonality extraction and forward selection."""
import numpy as np
from seasons_py import (
    extract_multiple_seasonalities,
    select_seasonalities,
    extract_seasonality,
)


def generate_series(n, patterns, noise_std=0.3, seed=None):
    """Build a series by summing repeated patterns + noise."""
    if seed is not None:
        np.random.seed(seed)
    series = np.zeros(n)
    for period, pattern in patterns.items():
        tiled = np.tile(pattern, (n // period) + 1)[:n]
        series += tiled
    series += np.random.normal(0, noise_std, n)
    return series


def test_joint_extract_two_periods():
    n = 420  # lcm(7, 6) = 42, 420 = 10 * 42
    patterns = {
        6: np.array([3.0, -3.0, 1.0, -1.0, 0.0, 0.0]),
        7: np.array([5.0, -5.0, 2.0, -2.0, 1.0, -1.0, 0.0]),
    }
    series = generate_series(n, patterns, noise_std=0.3, seed=20)
    result = extract_multiple_seasonalities(series, [6, 7])

    assert set(result.periods) == {6, 7}
    assert result.total_explained_var > 0.80
    np.testing.assert_allclose(result.components[6].profile, patterns[6], atol=0.3)
    np.testing.assert_allclose(result.components[7].profile, patterns[7], atol=0.3)
    print("PASS: joint extraction recovers two known periods")


def test_redundant_multiples_dropped():
    """Given 3, 7, 14, 21, selection should keep only 3 and 7."""
    n = 420
    patterns = {
        3: np.array([4.0, -2.0, -2.0]),
        7: np.array([3.0, -3.0, 1.0, -1.0, 0.5, -0.5, 0.0]),
    }
    series = generate_series(n, patterns, noise_std=0.4, seed=21)
    candidates = [3, 7, 14, 21]
    selected = select_seasonalities(series, candidates, criterion="bic", prefer_short=True)
    print(f"Redundant multiples pruned to: {selected}")
    assert set(selected) == {3, 7}, f"Expected {{3, 7}}, got {set(selected)}"
    print("PASS: redundant multiples are pruned")


def test_genuine_larger_period_kept():
    """
    A series with a true 12-period pattern that is NOT a pure 6-period pattern.
    When both 6 and 12 are candidates, BIC should prefer 12 alone because it
    uses fewer parameters and captures all the structure.
    """
    n = 240  # 20 full 12-cycles
    # 12-period pattern with substantial cycle-to-cycle variation.
    pattern_12 = np.array([5.0, -5.0, 3.0, -3.0, 1.0, -1.0,
                           2.0, -2.0, 6.0, -6.0, 0.0, 4.0])
    series = generate_series(n, {12: pattern_12}, noise_std=0.3, seed=22)
    candidates = [6, 12]
    selected = select_seasonalities(series, candidates, criterion="bic")
    print(f"Genuine 12-period structure selected: {selected}")
    assert set(selected) == {12}, f"Expected {{12}}, got {set(selected)}"
    print("PASS: 12 kept as the minimal general model")


def test_pure_divisor_period_preferred():
    """
    A series with a pure 6-period pattern. When 6 and 12 are both candidates,
    BIC should prefer 6 (the smaller fundamental) over 12.
    """
    n = 240
    pattern_6 = np.array([4.0, -4.0, 2.0, -2.0, 0.0, 0.0])
    series = generate_series(n, {6: pattern_6}, noise_std=0.3, seed=23)
    candidates = [6, 12]
    selected = select_seasonalities(series, candidates, criterion="bic")
    print(f"Pure 6-period selected: {selected}")
    assert set(selected) == {6}, f"Expected {{6}}, got {set(selected)}"
    print("PASS: smaller fundamental preferred for pure divisor pattern")


def test_select_seasonalities_on_noise():
    np.random.seed(23)
    series = np.random.normal(0, 1.0, 300)
    candidates = list(range(2, 25))
    selected = select_seasonalities(series, candidates, criterion="bic")
    print(f"Noise selected: {selected}")
    assert len(selected) == 0, f"Expected none, got {selected}"
    print("PASS: no periods selected for pure noise")


def test_extract_seasonality_single_backwards_compatible():
    period = 5
    pattern = np.array([3.0, -1.0, 4.0, 0.0, -6.0])
    series = np.tile(pattern, 20)
    result = extract_seasonality(series, period)
    np.testing.assert_allclose(result.profile, pattern, atol=1e-10)
    assert result.explained_var > 0.99
    print("PASS: single-period extraction remains backwards-compatible")


if __name__ == "__main__":
    test_joint_extract_two_periods()
    test_redundant_multiples_dropped()
    test_genuine_larger_period_kept()
    test_pure_divisor_period_preferred()
    test_select_seasonalities_on_noise()
    test_extract_seasonality_single_backwards_compatible()
    print("\nAll multi-seasonality tests passed!")