# seasons_py — Basic Implementation Plan

## Goal (basic version)
Detect a single integer seasonal period in a detrended time series using brute-force ANOVA.

## Assumptions
- Input series is already detrended (only seasonality + noise remain)
- Seasonal periods are integers
- Only one seasonal period is present (multi-period detection deferred)
- No seasonal effect extraction yet (deferred)

## Algorithm
1. Given a series of length n, iterate candidate period s from 2 to n//2
2. For each s, "fold" the series into a matrix of shape (rows × s)
   - Truncate the series so it divides evenly by s (drop trailing remainder)
3. Run one-way ANOVA on the s columns (each column = one group)
4. Record the p-value for each s
5. Return the candidate with the lowest p-value (or all below a significance threshold)

## Project structure
```
seasons_py/
├── pyproject.toml          # uv-managed project
├── seasons_py/
│   ├── __init__.py
│   └── detect.py           # core ANOVA detection logic
└── notebooks/
    └── demo.py             # marimo notebook (interactive UI)
```

## Dependencies
- numpy
- scipy (for ANOVA / f_oneway)
- marimo (notebook UI)
- matplotlib (plots in marimo)

## Steps
1. Create venv with uv (Python 3.12)
2. Install dependencies
3. Write `seasons_py/detect.py` — the core `detect_seasonality` function
4. Write `marimo` notebook — synthetic data + interactive detection
5. Test the core function with a known synthetic series
6. Verify marimo notebook runs