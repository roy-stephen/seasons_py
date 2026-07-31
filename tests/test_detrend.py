"""Tests for detrending utilities."""
import numpy as np
from seasons_py.detrend import linear_detrend, mean_detrend, detrend


def test_linear_detrend_removes_trend():
    n = 100
    trend = 0.5 * np.arange(n) + 2.0
    noise = np.random.normal(0, 0.1, n)
    raw = trend + noise
    detrended = linear_detrend(raw)

    # The slope and intercept should be almost gone.
    x = np.arange(n, dtype=float)
    beta = np.linalg.lstsq(np.column_stack([np.ones(n), x]), detrended, rcond=None)[0]
    assert abs(beta[1]) < 1e-6, f"remaining slope {beta[1]}"
    assert abs(beta[0]) < 1e-6, f"remaining intercept {beta[0]}"
    print("PASS: linear_detrend removes slope and intercept")


def test_linear_detrend_constant_is_zero():
    raw = np.array([3.0, 3.0, 3.0, 3.0])
    detrended = linear_detrend(raw)
    np.testing.assert_allclose(detrended, 0.0, atol=1e-12)
    print("PASS: linear_detrend on constant returns zeros")


def test_mean_detrend():
    raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    detrended = mean_detrend(raw)
    np.testing.assert_allclose(detrended, [-2, -1, 0, 1, 2])
    print("PASS: mean_detrend subtracts the mean")


def test_detrend_returns_trend():
    n = 50
    trend = -0.3 * np.arange(n) + 10.0
    noise = np.random.normal(0, 0.2, n)
    raw = trend + noise
    detrended, returned_trend = detrend(raw, method="linear")
    np.testing.assert_allclose(detrended + returned_trend, raw, atol=1e-12)
    # The fitted trend should be close to the true deterministic trend up to noise.
    np.testing.assert_allclose(returned_trend, trend, atol=0.2)
    # Residual must have near-zero slope.
    x = np.arange(n, dtype=float)
    beta = np.linalg.lstsq(np.column_stack([np.ones(n), x]), detrended, rcond=None)[0]
    assert abs(beta[1]) < 1e-6
    print("PASS: detrend returns additive trend")


def test_detrend_method_mean():
    raw = np.arange(10, dtype=float)
    detrended, trend = detrend(raw, method="mean")
    np.testing.assert_allclose(trend, 4.5, atol=1e-12)
    assert np.allclose(detrended.mean(), 0.0, atol=1e-12)
    print("PASS: detrend with method='mean'")


if __name__ == "__main__":
    np.random.seed(0)
    test_linear_detrend_removes_trend()
    test_linear_detrend_constant_is_zero()
    test_mean_detrend()
    test_detrend_returns_trend()
    test_detrend_method_mean()
    print("\nAll detrend tests passed!")
