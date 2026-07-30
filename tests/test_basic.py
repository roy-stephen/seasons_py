"""Quick smoke test for seasons_py core detection."""
import numpy as np
from seasons_py import detect_seasonality, scan_periods, fold_series

# Test 1: Perfect seasonal series with period 3, no noise
series = np.tile([1.0, 2.0, 3.0], 12)  # length 36, period 3
best = detect_seasonality(series)
print(f"Test 1 (perfect s=3): detected={best.period}, p={best.p_value:.2e}, f={best.f_statistic:.2f}")
assert best.period == 3, f"Expected 3, got {best.period}"
assert best.p_value < 0.05
print("  PASS")

# Test 2: Seasonal series with noise
np.random.seed(42)
series2 = np.tile([10.0, -5.0, 3.0, -8.0], 50) + np.random.normal(0, 1.0, 200)  # period 4
best2 = detect_seasonality(series2)
print(f"Test 2 (noisy s=4): detected={best2.period}, p={best2.p_value:.2e}")
assert best2.period == 4, f"Expected 4, got {best2.period}"
print("  PASS")

# Test 3: Pure noise (should not detect)
np.random.seed(123)
series3 = np.random.normal(0, 1.0, 200)
best3 = detect_seasonality(series3)
print(f"Test 3 (pure noise): detected={best3}")
assert best3 is None, f"Should be None, got {best3}"
print("  PASS")

# Test 4: Fold shape correctness
folded = fold_series(np.arange(10), 3)
print(f"Test 4 (fold 10 by 3): shape={folded.shape}, data=\n{folded}")
assert folded.shape == (3, 3), f"Expected (3,3), got {folded.shape}"
print("  PASS")

# Test 5: Larger period (s=7)
np.random.seed(7)
pattern7 = np.array([3.0, 1.0, -2.0, 0.5, 4.0, -1.0, 2.0])
series5 = np.tile(pattern7, 30) + np.random.normal(0, 0.5, 210)
best5 = detect_seasonality(series5)
print(f"Test 5 (noisy s=7): detected={best5.period}, p={best5.p_value:.2e}")
assert best5.period == 7, f"Expected 7, got {best5.period}"
print("  PASS")

print("\nAll tests passed!")