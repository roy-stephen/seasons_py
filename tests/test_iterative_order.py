"""Stress-test iterative greedy detection and document order/harmonic issues."""
import numpy as np
from seasons_py import iterative_detect, select_seasonalities


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


def test_non_harmonic_pair():
    """Two far-apart, non-harmonic periods should both be found when max_period is capped."""
    n = 910
    patterns = {
        7: np.array([3.0, -3.0, 1.0, 2.0, -2.0, 0.0, -1.0]),
        13: np.array([5.0, -5.0, 0.0, 0.0, 4.0, -4.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 0.0]),
    }
    series = generate_series(n, patterns, noise_std=0.3, seed=10)
    found = iterative_detect(series, n_seasons=2, alpha=0.05, max_period=50)
    detected = {f.period for f in found}
    print(f"Non-harmonic pair (true 7, 13): detected {detected}")
    assert detected == {7, 13}, f"Expected {{7, 13}}, got {detected}"
    print("  PASS")


def test_harmonic_fundamental_first():
    """True period is 12; greedy may pick 12 or a divisor depending on the pattern."""
    n = 240
    pattern = np.array([6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0])
    series = generate_series(n, {12: pattern}, noise_std=0.5, seed=11)
    found = iterative_detect(series, n_seasons=1, alpha=0.05, max_period=24)
    first = found[0].period if found else None
    print(f"Harmonic fundamental (true 12): first detected = {first}")
    # With harmonic suppression, we expect the fundamental if it's strongest.
    assert first in {6, 12, 4, 3, 2}, f"Unexpected first period {first}"


def test_close_periods():
    """Two close periods (11 and 12) are hard to separate. Record behavior."""
    n = 660
    patterns = {
        11: np.array([5.0, -5.0, 3.0, -3.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        12: np.array([4.0, 0.0, -4.0, 0.0, 4.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    series = generate_series(n, patterns, noise_std=0.4, seed=12)
    found = iterative_detect(series, n_seasons=2, alpha=0.05, max_period=50)
    detected = {f.period for f in found}
    print(f"Close periods (true 11, 12): detected {detected}")
    # Greedy may pick one or the other, or their near-LCM. Just record.


def test_strong_and_weak():
    """Strong seasonality should be found first; weak second."""
    n = 210
    patterns = {
        7: np.array([1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0]),
        14: np.array([10.0, -10.0, 5.0, -5.0, 2.0, -2.0, 0.0,
                      -10.0, 10.0, -5.0, 5.0, -2.0, 2.0, 0.0]),
    }
    series = generate_series(n, patterns, noise_std=1.0, seed=13)
    found = iterative_detect(series, n_seasons=2, alpha=0.05, max_period=50)
    detected = [f.period for f in found]
    print(f"Strong + weak (true 14 strong, 7 weak): detected in order {detected}")
    # 14 is a harmonic of 7; greedy may only see 14.


def test_joint_selection_better_than_greedy():
    """Where greedy fails due to LCMs, joint BIC selection recovers fundamentals."""
    n = 420
    patterns = {
        3: np.array([4.0, -2.0, -2.0]),
        7: np.array([3.0, -3.0, 1.0, -1.0, 0.5, -0.5, 0.0]),
    }
    series = generate_series(n, patterns, noise_std=0.4, seed=21)

    # Greedy without cap may pick LCM 21.
    greedy = iterative_detect(series, n_seasons=2, alpha=0.05, max_period=50)
    greedy_periods = {f.period for f in greedy}
    print(f"Greedy (max_period=50): {greedy_periods}")

    # Joint BIC selection over candidates from scan_periods should find {3, 7}.
    from seasons_py import scan_periods
    candidates = [r.period for r in scan_periods(series, max_period=50)]
    selected = select_seasonalities(series, candidates, criterion="bic", max_periods=3)
    print(f"Joint BIC selection: {selected}")
    assert set(selected) == {3, 7}, f"Expected {{3, 7}}, got {set(selected)}"
    print("  PASS: joint selection recovers fundamentals")


if __name__ == "__main__":
    test_non_harmonic_pair()
    test_harmonic_fundamental_first()
    test_close_periods()
    test_strong_and_weak()
    test_joint_selection_better_than_greedy()
    print("\nOrder-sensitivity diagnostics completed.")