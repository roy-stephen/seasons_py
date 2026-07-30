import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from seasons_py import (
        detect_seasonality,
        scan_periods,
        extract_seasonality,
        extract_multiple_seasonalities,
        select_seasonalities,
    )

    return (
        detect_seasonality,
        extract_seasonality,
        extract_multiple_seasonalities,
        mo,
        np,
        plt,
        scan_periods,
        select_seasonalities,
    )


@app.cell
def _(mo):
    # --- Controls: build a synthetic detrended series ---
    mo.md("## seasons_py — ANOVA Seasonality Detector")
    mo.md("_Assumption: detrended series, integer periods._")

    true_period = mo.ui.slider(2, 24, value=7, label="Primary seasonal period")
    second_period = mo.ui.slider(2, 24, value=13, label="Secondary period (set = primary to disable)")
    noise_std = mo.ui.slider(0.0, 2.0, step=0.1, value=0.5, label="Noise std")
    n_points = mo.ui.slider(50, 2000, step=50, value=350, label="Series length")
    max_selected = mo.ui.slider(1, 5, value=3, label="Max selected periods")

    [true_period, second_period, noise_std, n_points, max_selected]
    return n_points, noise_std, second_period, true_period, max_selected


@app.cell
def _(n_points, noise_std, np, second_period, true_period):
    # Generate one or two discrete repeating patterns + noise.
    _s1 = true_period.value
    _s2 = second_period.value
    _n = n_points.value
    _sigma = noise_std.value

    np.random.seed(0)

    def make_pattern(s):
        p = np.linspace(5.0, -5.0, s) + np.random.normal(0, 1.0, s)
        return p - p.mean()

    # Use a common cycle length so two patterns of different periods tile cleanly.
    _cycle = (_s1 * _s2) if _s1 != _s2 else _s1
    base = np.zeros(_cycle)
    base += np.tile(make_pattern(_s1), _cycle // _s1)
    if _s2 != _s1:
        base += np.tile(make_pattern(_s2), _cycle // _s2)

    series = np.tile(base, _n // _cycle + 1)[:_n]
    series = series + np.random.normal(0, _sigma, _n)
    return (series,)


@app.cell
def _(mo, plt, series):
    # Plot the series
    fig_series, ax_series = plt.subplots(figsize=(10, 3))
    ax_series.plot(series, lw=0.8)
    ax_series.set_title("Detrended Series")
    ax_series.set_xlabel("Time")
    ax_series.set_ylabel("Value")
    mo.mpl.interactive(fig_series)
    return


@app.cell
def _(detect_seasonality, mo, scan_periods, series, true_period):
    # --- Run single detection ---
    results = scan_periods(series)
    best = detect_seasonality(series)

    _detected = best.period if best else 'None'
    _pval = f"{best.p_value:.2e}" if best else "N/A"
    _n_tests = len(results)
    _alpha_corr = 0.05 / _n_tests if _n_tests > 0 else 0.05

    summary = mo.md(
        f"""
        **Single-detected period:** {_detected}

        **p-value:** {_pval}

        **True period:** {true_period.value}

        **Tests run:** {_n_tests} (Bonferroni-corrected α = {_alpha_corr:.2e})
        """
    )

    summary
    return best, results


@app.cell
def _(best, mo, np, plt, results, true_period):
    # Evidence plot
    n_tests = len(results)
    raw_alpha = 0.05
    corrected_alpha = raw_alpha / n_tests if n_tests > 0 else raw_alpha

    periods = np.array([r.period for r in results])
    pvals = np.array([r.p_value for r in results])

    logp = -np.log10(np.maximum(pvals, 1e-300))
    cap = 25.0
    capped_logp = np.minimum(logp, cap)
    n_capped = np.sum(logp >= cap)

    sig_line = -np.log10(corrected_alpha)

    fig_evidence, ax_evidence = plt.subplots(figsize=(10, 4))
    ax_evidence.stem(periods, capped_logp, basefmt=" ", linefmt="C0-", markerfmt="C0o", label="-log10(p)")
    ax_evidence.axhline(sig_line, color="red", ls="--", lw=1, label=f"α_corr = {corrected_alpha:.2e}")
    ax_evidence.axvline(true_period.value, color="green", ls="--", lw=1.5, alpha=0.7, label="True period")
    if best:
        ax_evidence.axvline(best.period, color="orange", ls="-", lw=2, alpha=0.7, label="Detected period")

    ax_evidence.set_xlabel("Candidate period")
    ax_evidence.set_ylabel("-log10(p-value)  (capped at {})".format(int(cap)))
    ax_evidence.set_title("ANOVA evidence across candidate periods (higher = more significant)")
    ax_evidence.legend(loc="upper right")
    ax_evidence.set_ylim(bottom=-0.5, top=cap + 2)

    note = (
        f"**How to read this:** each bar is one candidate period. "
        f"Bars above the red line pass the corrected significance threshold. "
        f"Values capped at {int(cap)}: {n_capped} period(s) had p-values so small they hit the cap."
    )

    mo.vstack([mo.md(note), mo.mpl.interactive(fig_evidence)])
    return


@app.cell
def _(best, extract_seasonality, mo, plt, series):
    # Single-seasonality extraction
    if best:
        extracted = extract_seasonality(series, best.period)

        fig_single_fit, ax_single_fit = plt.subplots(figsize=(10, 3))
        ax_single_fit.plot(series, lw=0.7, alpha=0.8, label="Original")
        ax_single_fit.plot(extracted.fitted, lw=1.2, label="Fitted seasonal")
        ax_single_fit.set_title(f"Original vs fitted seasonal (period = {best.period})")
        ax_single_fit.set_xlabel("Time")
        ax_single_fit.set_ylabel("Value")
        ax_single_fit.legend()
        fitted_plot = mo.mpl.interactive(fig_single_fit)

        fig_single_res, ax_single_res = plt.subplots(figsize=(10, 3))
        ax_single_res.plot(series - extracted.fitted, lw=0.7, color="gray")
        ax_single_res.axhline(0, color="black", ls="--", lw=0.5)
        ax_single_res.set_title(f"Residual (explained variance = {extracted.explained_var:.2%})")
        ax_single_res.set_xlabel("Time")
        ax_single_res.set_ylabel("Value")
        residual_plot = mo.mpl.interactive(fig_single_res)

        fig_single_prof, ax_single_prof = plt.subplots(figsize=(8, 3))
        ax_single_prof.bar(range(1, best.period + 1), extracted.profile, color="steelblue", edgecolor="black")
        ax_single_prof.set_xlabel("Phase")
        ax_single_prof.set_ylabel("Mean value")
        ax_single_prof.set_title(f"Seasonal profile (period = {best.period})")
        ax_single_prof.set_xticks(range(1, best.period + 1))
        profile_plot = mo.mpl.interactive(fig_single_prof)

        extraction_section = mo.vstack([
            mo.md("### Single seasonality extraction"),
            mo.md(f"**Explained variance:** {extracted.explained_var:.2%}"),
            fitted_plot,
            residual_plot,
            profile_plot,
        ])
    else:
        extraction_section = mo.md("No significant period detected; cannot extract seasonality.")

    extraction_section
    return


@app.cell
def _(max_selected, mo, plt, results, select_seasonalities, series, true_period, second_period):
    # Multi-seasonality selection via BIC forward selection + divisor factorization
    candidate_periods = [r.period for r in results if r.p_value < 0.05]
    # Fallback: use top candidates by p-value if none pass threshold
    if not candidate_periods:
        candidate_periods = [r.period for r in results[:10]]

    selected = select_seasonalities(
        series,
        candidate_periods,
        criterion="bic",
        max_periods=max_selected.value,
    )

    multi = extract_multiple_seasonalities(series, selected)

    multi_text = mo.md(
        f"""
        ### Multi-seasonality selection
        **Selected periods:** {selected}

        **Total explained variance:** {multi.total_explained_var:.2%}

        **True periods:** {true_period.value}{f', {second_period.value}' if second_period.value != true_period.value else ''}
        """
    )

    # Plot original + total fitted + residual
    fig_joint, ax_joint = plt.subplots(figsize=(10, 3))
    ax_joint.plot(series, lw=0.7, alpha=0.8, label="Original")
    ax_joint.plot(multi.fitted, lw=1.2, label="Joint fitted seasonal")
    ax_joint.set_title(f"Joint fit (periods = {selected})")
    ax_joint.set_xlabel("Time")
    ax_joint.set_ylabel("Value")
    ax_joint.legend()
    joint_plot = mo.mpl.interactive(fig_joint)

    fig_joint_res, ax_joint_res = plt.subplots(figsize=(10, 3))
    ax_joint_res.plot(multi.residual, lw=0.7, color="gray")
    ax_joint_res.axhline(0, color="black", ls="--", lw=0.5)
    ax_joint_res.set_title("Residual after joint extraction")
    ax_joint_res.set_xlabel("Time")
    ax_joint_res.set_ylabel("Value")
    joint_residual_plot = mo.mpl.interactive(fig_joint_res)

    # Per-component profiles
    profile_plots = []
    for s in multi.periods:
        fig_prof, ax_prof = plt.subplots(figsize=(6, 2.5))
        ax_prof.bar(range(1, s + 1), multi.components[s].profile, color="steelblue", edgecolor="black")
        ax_prof.set_xlabel("Phase")
        ax_prof.set_ylabel("Value")
        ax_prof.set_title(f"Profile (period = {s}, share = {multi.components[s].explained_var:.1%})")
        ax_prof.set_xticks(range(1, s + 1))
        profile_plots.append(mo.mpl.interactive(fig_prof))

    multi_section = mo.vstack([
        multi_text,
        joint_plot,
        joint_residual_plot,
        mo.hstack(profile_plots) if len(profile_plots) > 1 else profile_plots[0] if profile_plots else mo.md(""),
    ])

    return multi_section


if __name__ == "__main__":
    app.run()