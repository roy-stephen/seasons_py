import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from seasons_py import detect_seasonality, scan_periods, fold_series, extract_seasonality

    return detect_seasonality, extract_seasonality, fold_series, mo, np, plt, scan_periods


@app.cell
def _(mo):
    # --- Controls: build a synthetic detrended series ---
    mo.md("## seasons_py — ANOVA Seasonality Detector")
    mo.md("_Assumption: detrended series, integer period, single seasonality._")

    true_period = mo.ui.slider(2, 24, value=7, label="True seasonal period")
    noise_std = mo.ui.slider(0.0, 2.0, step=0.1, value=0.5, label="Noise std")
    n_points = mo.ui.slider(50, 2000, step=50, value=350, label="Series length")

    [true_period, noise_std, n_points]
    return n_points, noise_std, true_period


@app.cell
def _(n_points, noise_std, np, true_period):
    # Generate a *discrete* repeating pattern of the chosen period + noise.
    # This matches the "integer seasons" assumption better than a smooth sine.
    _s = true_period.value
    _n = n_points.value
    _sigma = noise_std.value

    np.random.seed(0)
    # A clearly non-sinusoidal repeating shape with _s distinct values.
    seasonal_pattern = np.linspace(5.0, -5.0, _s) + np.random.normal(0, 1.0, _s)
    seasonal_pattern = seasonal_pattern - seasonal_pattern.mean()  # zero-mean
    series = np.tile(seasonal_pattern, _n // _s + 1)[:_n]
    series = series + np.random.normal(0, _sigma, _n)
    return (series,)


@app.cell
def _(mo, plt, series):
    # Plot the series
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series, lw=0.8)
    ax.set_title("Detrended Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    mo.mpl.interactive(fig)
    return


@app.cell
def _(detect_seasonality, mo, scan_periods, series, true_period):
    # --- Run detection ---
    results = scan_periods(series)
    best = detect_seasonality(series)

    _detected = best.period if best else 'None'
    _pval = f"{best.p_value:.2e}" if best else "N/A"
    _n_tests = len(results)
    _alpha_corr = 0.05 / _n_tests if _n_tests > 0 else 0.05

    summary = mo.md(
        f"""
        **Detected period:** {_detected}

        **p-value:** {_pval}

        **True period:** {true_period.value}

        **Tests run:** {_n_tests} (Bonferroni-corrected α = {_alpha_corr:.2e})
        """
    )

    summary
    return best, results


@app.cell
def _(best, mo, np, plt, results, true_period):
    # Evidence plot: show each candidate period as a stem/bar, capped so the
    # huge harmonic peaks don't squash the noise floor.
    n_tests = len(results)
    raw_alpha = 0.05
    corrected_alpha = raw_alpha / n_tests if n_tests > 0 else raw_alpha

    periods = np.array([r.period for r in results])
    pvals = np.array([r.p_value for r in results])

    # -log10(p), capped at 25 so strong peaks don't blow out the plot.
    logp = -np.log10(np.maximum(pvals, 1e-300))
    cap = 25.0
    capped_logp = np.minimum(logp, cap)
    n_capped = np.sum(logp >= cap)

    sig_line = -np.log10(corrected_alpha)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.stem(periods, capped_logp, basefmt=" ", linefmt="C0-", markerfmt="C0o", label="-log10(p)")
    ax2.axhline(sig_line, color="red", ls="--", lw=1, label=f"α_corr = {corrected_alpha:.2e}")
    ax2.axvline(true_period.value, color="green", ls="--", lw=1.5, alpha=0.7, label="True period")
    if best:
        ax2.axvline(best.period, color="orange", ls="-", lw=2, alpha=0.7, label="Detected period")

    ax2.set_xlabel("Candidate period")
    ax2.set_ylabel("-log10(p-value)  (capped at {})".format(int(cap)))
    ax2.set_title("ANOVA evidence across candidate periods (higher = more significant)")
    ax2.legend(loc="upper right")
    ax2.set_ylim(bottom=-0.5, top=cap + 2)

    # Annotation explaining what is visible.
    note = (
        f"**How to read this:** each bar is one candidate period. "
        f"Bars above the red line pass the corrected significance threshold. "
        f"Values capped at {int(cap)}: {n_capped} period(s) had p-values so small they hit the cap."
    )

    mo.vstack([mo.md(note), mo.mpl.interactive(fig2)])
    return


@app.cell
def _(best, extract_seasonality, mo, plt, series):
    # Extract the seasonal effect for the detected period and visualize
    # original / fitted / residual side by side.
    if best:
        extracted = extract_seasonality(series, best.period)

        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(series, lw=0.7, alpha=0.8, label="Original")
        ax3.plot(extracted.fitted, lw=1.2, label="Fitted seasonal")
        ax3.set_title(f"Original vs fitted seasonal (period = {best.period})")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Value")
        ax3.legend()
        fitted_plot = mo.mpl.interactive(fig3)

        fig4, ax4 = plt.subplots(figsize=(10, 3))
        ax4.plot(extracted.residual, lw=0.7, color="gray")
        ax4.axhline(0, color="black", ls="--", lw=0.5)
        ax4.set_title(f"Residual (explained variance = {extracted.explained_var:.2%})")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Value")
        residual_plot = mo.mpl.interactive(fig4)

        # Seasonal profile
        fig5, ax5 = plt.subplots(figsize=(8, 3))
        ax5.bar(range(1, best.period + 1), extracted.profile, color="steelblue", edgecolor="black")
        ax5.set_xlabel("Phase")
        ax5.set_ylabel("Mean value")
        ax5.set_title(f"Seasonal profile (period = {best.period})")
        ax5.set_xticks(range(1, best.period + 1))
        profile_plot = mo.mpl.interactive(fig5)

        extraction_section = mo.vstack([
            mo.md(f"**Explained variance:** {extracted.explained_var:.2%}"),
            fitted_plot,
            residual_plot,
            profile_plot,
        ])
    else:
        extraction_section = mo.md("No significant period detected; cannot extract seasonality.")

    return (extraction_section,)


if __name__ == "__main__":
    app.run()