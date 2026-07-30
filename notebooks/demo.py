import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    """Setup: imports and public API."""
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
    """Setup: user-facing controls."""
    mo.md("""
    # seasons_py demo

    **Assumptions:** the input series is already detrended, and seasonal periods are integer-valued.

    This notebook walks through two ideas:
    1. **Single-period extraction** — detect one seasonality and extract its profile.
    2. **Multiple-period extraction** — the main point: select and estimate several seasonalities *jointly*,
       so that each effect is conditioned on the others.
    """)

    mo.md("### Build a synthetic series")
    true_period = mo.ui.slider(2, 24, value=7, label="Primary period")
    second_period = mo.ui.slider(2, 24, value=13, label="Secondary period (set equal to primary to disable)")
    noise_std = mo.ui.slider(0.0, 2.0, step=0.1, value=0.5, label="Noise std")
    n_points = mo.ui.slider(50, 2000, step=50, value=350, label="Series length")
    max_selected = mo.ui.slider(1, 5, value=3, label="Max periods in joint model")

    controls = mo.hstack([
        mo.vstack([true_period, second_period, noise_std]),
        mo.vstack([n_points, max_selected]),
    ])
    controls
    return controls, max_selected, n_points, noise_std, second_period, true_period


@app.cell
def _(max_selected, n_points, noise_std, np, second_period, true_period):
    """Setup: generate the synthetic detrended series."""
    _s1 = true_period.value
    _s2 = second_period.value
    _n = n_points.value
    _sigma = noise_std.value
    _max_selected = max_selected.value

    np.random.seed(0)

    def make_pattern(s):
        p = np.linspace(5.0, -5.0, s) + np.random.normal(0, 1.0, s)
        return p - p.mean()

    # Use the LCM of the two periods as the repeating base so the combined
    # pattern tiles cleanly for any pair of integer periods.
    _cycle = (_s1 * _s2) if _s1 != _s2 else _s1
    base = np.zeros(_cycle)
    base += np.tile(make_pattern(_s1), _cycle // _s1)
    if _s2 != _s1:
        base += np.tile(make_pattern(_s2), _cycle // _s2)

    series = np.tile(base, _n // _cycle + 1)[:_n]
    series = series + np.random.normal(0, _sigma, _n)
    return (series,)


@app.cell
def _(mo, plt, series, true_period, second_period):
    """Setup: plot the raw series."""
    fig_raw, ax_raw = plt.subplots(figsize=(10, 3))
    ax_raw.plot(series, lw=0.8)
    ax_raw.set_title("Synthetic detrended series")
    ax_raw.set_xlabel("Time")
    ax_raw.set_ylabel("Value")
    active_periods = f"{true_period.value}" if second_period.value == true_period.value else f"{true_period.value}, {second_period.value}"
    mo.vstack([
        mo.md(f"**True periods in the generator:** {active_periods}"),
        mo.mpl.interactive(fig_raw),
    ])
    return


@app.cell
def _(detect_seasonality, mo, scan_periods, series):
    """Act 1 — Single-period detection."""
    mo.md("""
    ## 1. Single-period detection

    The simplest use case: scan candidate periods and return the most significant one.
    Each period is tested by folding the series into that many phase buckets and running
    a one-way ANOVA. The p-value is Bonferroni-corrected over all candidates.
    """)

    results = scan_periods(series)
    best = detect_seasonality(series)

    _detected = best.period if best else "None"
    _pval = f"{best.p_value:.2e}" if best else "N/A"
    _n_tests = len(results)
    _alpha_corr = 0.05 / _n_tests if _n_tests > 0 else 0.05

    mo.md(
        f"""
        **Detected period:** {_detected}

        **p-value:** {_pval}

        **Number of tests:** {_n_tests} (Bonferroni-corrected α = {_alpha_corr:.2e})
        """
    )
    return best, results


@app.cell
def _(best, mo, np, plt, results, true_period, second_period):
    """Act 1 — single-period evidence plot."""
    n_tests = len(results)
    corrected_alpha = 0.05 / n_tests if n_tests > 0 else 0.05

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

    true_periods_set = {true_period.value, second_period.value} if second_period.value != true_period.value else {true_period.value}
    for tp in true_periods_set:
        ax_evidence.axvline(tp, color="green", ls="--", lw=1.5, alpha=0.7)
    if best:
        ax_evidence.axvline(best.period, color="orange", ls="-", lw=2, alpha=0.7, label="Detected")

    ax_evidence.set_xlabel("Candidate period")
    ax_evidence.set_ylabel(f"-log10(p-value)  (capped at {int(cap)})")
    ax_evidence.set_title("ANOVA evidence per candidate period")
    ax_evidence.legend(loc="upper right")
    ax_evidence.set_ylim(bottom=-0.5, top=cap + 2)

    note = (
        f"Bars above the red line pass the corrected significance threshold. "
        f"Values capped at {int(cap)}: {n_capped} period(s) had extremely small p-values."
    )
    mo.vstack([mo.md(f"*{note}*"), mo.mpl.interactive(fig_evidence)])
    return None


@app.cell
def _(best, extract_seasonality, mo, plt, series):
    """Act 1 — single-period extraction (fit / residual / profile)."""
    mo.md("### Extract the single detected seasonality")

    extracted = extract_seasonality(series, best.period) if best else None

    if extracted is None:
        single_section = mo.md("No significant period was detected, so single extraction is skipped.")
    else:
        fig_single_fit, ax_single_fit = plt.subplots(figsize=(10, 3))
        ax_single_fit.plot(series, lw=0.7, alpha=0.8, label="Original")
        ax_single_fit.plot(extracted.fitted, lw=1.2, label="Fitted seasonal")
        ax_single_fit.set_title(f"Original vs fitted seasonal (period = {best.period})")
        ax_single_fit.set_xlabel("Time")
        ax_single_fit.set_ylabel("Value")
        ax_single_fit.legend()

        fig_single_res, ax_single_res = plt.subplots(figsize=(10, 3))
        residual = series - extracted.fitted
        ax_single_res.plot(residual, lw=0.7, color="gray")
        ax_single_res.axhline(0, color="black", ls="--", lw=0.5)
        ax_single_res.set_title(f"Residual (single-fit explained variance = {extracted.explained_var:.2%})")
        ax_single_res.set_xlabel("Time")
        ax_single_res.set_ylabel("Value")

        fig_single_prof, ax_single_prof = plt.subplots(figsize=(8, 3))
        ax_single_prof.bar(range(1, best.period + 1), extracted.profile, color="steelblue", edgecolor="black")
        ax_single_prof.set_xlabel("Phase")
        ax_single_prof.set_ylabel("Effect")
        ax_single_prof.set_title(f"Learned seasonal profile (period = {best.period})")
        ax_single_prof.set_xticks(range(1, best.period + 1))

        single_section = mo.vstack([
            mo.mpl.interactive(fig_single_fit),
            mo.mpl.interactive(fig_single_res),
            mo.mpl.interactive(fig_single_prof),
        ])

    single_section
    return None


@app.cell
def _(max_selected, mo, results, select_seasonalities, series, true_period, second_period):
    """Act 2 — multiple-period selection (the main feature)."""
    mo.md("""
    ## 2. Multiple-period extraction (joint OLS)

    This is the main idea: instead of detecting periods one at a time and subtracting them,
    we **select a set of candidate periods and fit all of their seasonal profiles simultaneously**
    with one linear model. Each profile is estimated conditioning on the others, so the result is
    order-independent.

    Selection uses a forward step: at each step we add the candidate whose inclusion most improves
    the BIC, with a preference for shorter fundamental periods. After selection, a divisor check
    replaces a larger period by its divisors when that improves BIC — this prevents LCMs such as
    21 hiding the pair `{3, 7}`.
    """)

    # Candidates are the periods that pass the uncorrected significance screen.
    candidate_periods = [r.period for r in results if r.p_value < 0.05]
    if not candidate_periods:
        candidate_periods = [r.period for r in results[:10]]

    selected = select_seasonalities(
        series,
        candidate_periods,
        criterion="bic",
        max_periods=max_selected.value,
    )

    true_periods_str = f"{true_period.value}" if second_period.value == true_period.value else f"{true_period.value}, {second_period.value}"
    mo.md(
        f"""
        **Candidate pool:** {len(candidate_periods)} periods

        **Selected by BIC:** {selected}

        **True generator periods:** {true_periods_str}
        """
    )
    return (selected,)


@app.cell
def _(extract_multiple_seasonalities, mo, plt, selected, series):
    """Act 2 — joint fit, residual, and per-component profiles."""
    multi = extract_multiple_seasonalities(series, selected) if selected else None

    if multi is None:
        multi_section = mo.md("No periods were selected, so the joint model is empty.")
    else:
        fig_joint, ax_joint = plt.subplots(figsize=(10, 3))
        ax_joint.plot(series, lw=0.7, alpha=0.8, label="Original")
        ax_joint.plot(multi.fitted, lw=1.2, label="Joint fitted seasonal")
        ax_joint.set_title(f"Joint fit (periods = {selected})")
        ax_joint.set_xlabel("Time")
        ax_joint.set_ylabel("Value")
        ax_joint.legend()

        fig_joint_res, ax_joint_res = plt.subplots(figsize=(10, 3))
        ax_joint_res.plot(multi.residual, lw=0.7, color="gray")
        ax_joint_res.axhline(0, color="black", ls="--", lw=0.5)
        ax_joint_res.set_title(f"Residual after joint extraction (total explained variance = {multi.total_explained_var:.2%})")
        ax_joint_res.set_xlabel("Time")
        ax_joint_res.set_ylabel("Value")

        profile_plots = []
        for s in multi.periods:
            fig_prof, ax_prof = plt.subplots(figsize=(5, 2.5))
            ax_prof.bar(range(1, s + 1), multi.components[s].profile, color="steelblue", edgecolor="black")
            ax_prof.set_xlabel("Phase")
            ax_prof.set_ylabel("Effect")
            ax_prof.set_title(f"Period {s} (share {multi.components[s].explained_var:.1%})")
            ax_prof.set_xticks(range(1, s + 1))
            profile_plots.append(mo.mpl.interactive(fig_prof))

        multi_section = mo.vstack([
            mo.mpl.interactive(fig_joint),
            mo.mpl.interactive(fig_joint_res),
            mo.hstack(profile_plots) if len(profile_plots) > 1 else profile_plots[0],
        ])

    multi_section
    return None


@app.cell
def _(mo):
    """Closing: what to remember."""
    mo.md("""
    ### Takeaway

    - **Single detection** is a good first look, but it is greedy and order-sensitive.
    - **Joint extraction** is the recommended workflow when you suspect more than one seasonality:
      give it a candidate list (or the whole `scan_periods` output) and let BIC decide
      which periods earn their keep.
    - The fitted signal is a deterministic tiling of each learned profile from index 0, and the
      residual is simply `series - fitted`.
    """)
    return


if __name__ == "__main__":
    app.run()