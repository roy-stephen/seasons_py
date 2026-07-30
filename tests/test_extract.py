"""Tests for single-seasonality extraction."""
import numpy as np
from seasons_py import extract_seasonality, iterative_detect


def test_extract_profile_matches_true_pattern():
    period = 5
    pattern = np.array([3.0, -1.0, 4.0, 0.0, -6.0])
    series = np.tile(pattern, 20)  # 100 points
    result = extract_seasonality(series, period)

    np.testing.assert_allclose(result.profile, pattern, atol=1e-10)
    assert result.fitted.shape == series.shape
    assert result.residual.shape == series.shape
    assert np.allclose(result.residual, 0.0, atol=1e-10)
    assert result.explained_var > 0.999
    print("PASS: extract profile matches true pattern")


def test_extract_with_noise_has_high_explained_variance():
    np.random.seed(1)
    period = 7
    pattern = np.linspace(5.0, -5.0, period)
    series = np.tile(pattern, 30) + np.random.normal(0, 0.5, 210)
    result = extract_seasonality(series, period)

    # The fitted signal should explain most of the predictable structure.
    assert result.explained_var > 0.70, f"explained_var={result.explained_var:.3f}"
    # Profile should be close to the true pattern (allowing a global offset
    # because pattern is not zero-mean and extraction centers on phases).
    np.testing.assert_allclose(result.profile, pattern, atol=1.0)
    print("PASS: noisy extraction has high explained variance")


def test_extract_on_noise_has_low_explained_variance():
    np.random.seed(2)
    series = np.random.normal(0, 1.0, 200)
    # Even if we force a period, the explained variance should be near zero.
    result = extract_seasonality(series, 12)
    assert abs(result.explained_var) < 0.05, f"explained_var={result.explained_var:.3f}"
    print("PASS: noise has near-zero explained variance")


def test_iterative_detect_finds_two_seasonalities():
    np.random.seed(3)
    n = 420  # must be divisible by 12 and 14 (lcm = 84); 420 = 5 * 84
    s1 = np.array([2.0, -2.0, 1.0, -1.0])  # period 4
    s2 = np.array([5.0, 0.0, -5.0, 0.0, 3.0, -3.0])  # period 6
    series = np.tile(s1, n // 4) + np.tile(s2, n // 6) + np.random.normal(0, 0.3, n)

    found = iterative_detect(series, n_seasons=2, alpha=0.05)
    detected_periods = [f.period for f in found]
    print(f"Iterative detected periods: {detected_periods}")

    assert len(found) == 2, f"Expected 2, got {len(found)}"
    assert set(detected_periods) == {4, 6}, f"Expected {{4, 6}}, got {set(detected_periods)}"

    # Both should explain substantial variance in their residuals.
    for f in found:
        assert f.explained_var > 0.20, f"period {f.period}: explained_var={f.explained_var:.3f}"
    print("PASS: iterative detection finds two seasonalities")


if __name__ == "__main__":
    test_extract_profile_matches_true_pattern()
    test_extract_with_noise_has_high_explained_variance()
    test_extract_on_noise_has_low_explained_variance()
    test_iterative_detect_finds_two_seasonalities()
    print("\nAll extraction tests passed!")