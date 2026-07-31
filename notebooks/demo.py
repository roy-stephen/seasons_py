import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    """Setup: imports and public API."""
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from seasons_py import (
        detect_seasonality,
        scan_periods,
        extract_seasonality,
        extract_multiple_seasonalities,
        extract_calendar_seasonality,
        select_seasonalities,
    )

    return (
        detect_seasonality,
        extract_multiple_seasonalities,
        extract_seasonality,
        extract_calendar_seasonality,
        mo,
        np,
        pd,
        plt,
        scan_periods,
        select_seasonalities,
    )


@app.cell
def _(mo):
    """Setup: user-facing controls."""
    mo.md("""
    # seasons_py demo

    **Assumptions:** the input series is already detrended.

    This notebook walks through three ideas:
    1. **Single-period extraction** — detect one integer seasonality and extract its profile.
    2. **Multiple-period extraction** — select and estimate several integer seasonalities *jointly*,
       so that each effect is conditioned on the others.
    3. **Calendar seasonality** — same joint estimator, but phases come from a `DatetimeIndex`
       (day-of-week, month-of-year, day-of-month, etc.).
    """)

    mo.md("### Build a synthetic series")
    demo_mode = mo.ui.dropdown(
        options=["integer periods", "calendar rules", "5-seasonality stress test"],
        value="integer periods",
        label="Demo mode",
    )
    true_period = mo.ui.slider(2, 24, value=7, label="Primary period")
    second_period = mo.ui.slider(2, 24, value=13, label="Secondary period (set equal to primary to disable)")
    noise_std = mo.ui.slider(0.0, 2.0, step=0.1, value=0.5, label="Noise std")
    n_points = mo.ui.slider(50, 5000, step=50, value=350, label="Series length")
    max_selected = mo.ui.slider(1, 8, value=5, label="Max periods in joint model")
    calendar_rules = mo.ui.dropdown(
        options=["dow,month", "dow,dom", "dow,month,dom", "month,quarter"],
        value="dow,month",
        label="Calendar rules",
    )

    controls = mo.vstack([
        demo_mode,
        mo.hstack([
            mo.vstack([true_period, second_period, noise_std]),
            mo.vstack([n_points, max_selected, calendar_rules]),
        ]),
    ])
    controls
    return calendar_rules, demo_mode, max_selected, n_points, noise_std, second_period, true_period


@app.cell
def _(
    calendar_rules,
    max_selected,
    n_points,
    noise_std,
    np,
    pd,
    second_period,
    true_period,
    demo_mode,
):
    """Setup: generate the synthetic detrended series."""
    _mode = demo_mode.value
    _s1 = true_period.value
    _s2 = second_period.value
    _n = n_points.value
    _sigma = noise_std.value
    _max_selected = max_selected.value
    _rules = [r.strip() for r in calendar_rules.value.split(",")]

    np.random.seed(0)

    def make_pattern(s):
        p = np.linspace(5.0, -5.0, s) + np.random.normal(0, 1.0, s)
        return p - p.mean()

    if _mode == "5-seasonality stress test":
        # High-dimension stress test: five small co-prime periods.
        preset_periods = [3, 5, 7, 11, 13]
        _cycle = int(np.lcm.reduce(preset_periods))
        base = np.zeros(_cycle)
        for p in preset_periods:
            base += np.tile(make_pattern(p), _cycle // p)
        series = np.tile(base, _n // _cycle + 1)[:_n]
        generator_periods = preset_periods
        index = None
    elif _mode == "calendar rules":
        # Daily calendar seasonality.
        _n_obs = max(_n, 365 * 2)
        index = pd.date_range("2020-01-01", periods=_n_obs, freq="D")
        series = np.zeros(_n_obs)
        for rule in _rules:
            if rule == "dow":
                dow_profile = np.array([3.0, 1.0, 0.0, 0.0, -2.0, -1.0, 1.0])
                dow_profile = dow_profile - dow_profile.mean()
                series += dow_profile[index.dayofweek]
            elif rule == "month":
                month_profile = np.array([4.0, 3.0, 2.0, 0.0, -1.0, -3.0, -4.0, -3.0, -2.0, 0.0, 1.0, 3.0])
                month_profile = month_profile - month_profile.mean()
                series += month_profile[index.month - 1]
            elif rule == "dom":
                dom_profile = np.zeros(31)
                dom_profile[0] = 3.0
                dom_profile[14] = -2.0
                dom_profile = dom_profile - dom_profile.mean()
                series += dom_profile[index.day - 1]
            elif rule == "quarter":
                quarter_profile = np.array([-2.0, 1.0, 2.0, -1.0])
                quarter_profile = quarter_profile - quarter_profile.mean()
                series += quarter_profile[index.quarter - 1]
        series = series - series.mean()  # ensure zero-mean detrended signal
        series = series[:_n]
        index = index[:_n]
        generator_periods = _rules
    else:
        # Use the LCM of the two periods as the repeating base so the combined
        # pattern tiles cleanly for any pair of integer periods.
        _cycle = (_s1 * _s2) if _s1 != _s2 else _s1
        base = np.zeros(_cycle)
        base += np.tile(make_pattern(_s1), _cycle // _s1)
        if _s2 != _s1:
            base += np.tile(make_pattern(_s2), _cycle // _s2)
        series = np.tile(base, _n // _cycle + 1)[:_n]
        generator_periods = [_s1, _s2] if _s2 != _s1 else [_s1]
        index = None

    series = series + np.random.normal(0, _sigma, len(series))
    return series, generator_periods, index, _rules


@app.cell
def _(generator_periods, index, mo, np, plt, series):
    """Setup: plot the detrended series and verify mean/slope."""
    _mean = float(np.mean(series))
    _slope = float(np.linalg.lstsq(
        np.column_stack([np.ones(len(series)), np.arange(len(series))]),
        series,
        rcond=None,
    )[0][1])

    fig_raw, ax_raw = plt.subplots(figsize=(10, 3))
    ax_raw.plot(series, lw=0.8)
    ax_raw.set_title("Synthetic detrended series")
    ax_raw.set_xlabel("Time")
    ax_raw.set_ylabel("Value")
    active_periods = ", ".join(str(p) for p in generator_periods)
    index_note = f"Index: {index[0].date()} to {index[-1].date()}" if index is not None else "Index: positional integer index"
    mo.vstack([
        mo.md(f"**True periods / rules in the generator:** {active_periods}"),
        mo.md(f"*{index_note}*"),
        mo.md(
            f"**Detrended check:** mean = `{_mean:.3f}`, residual linear slope = `{_slope:.2e}`. "
            f"These should be essentially zero."
        ),
        mo.mpl.interactive(fig_raw),
    ])
    return


@app.cell
def _(mo):
    """Act 1 — Single-period detection."""
    mo.md("""
    ## 1. Single-period detection

    The simplest use case: scan candidate periods and return the most significant one.
    Each period is tested by folding the series into that many phase buckets and running
    a one-way ANOVA. The p-value is Bonferroni-corrected over all candidates.
    """)
    return


@app.cell
def _(detect_seasonality, mo, scan_periods, series):
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
def _(best, generator_periods, mo, np, plt, results):
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

    for tp in generator_periods:
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
    return


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
    return


@app.cell
def _(
    demo_mode,
    extract_calendar_seasonality,
    max_selected,
    mo,
    results,
    select_seasonalities,
    series,
    index,
    generator_periods,
):
    """Act 2 — multiple-period selection (the main feature)."""
    mo.md("""
    ## 2. Multiple-period / calendar extraction (joint OLS)

    This is the main idea: instead of detecting periods one at a time and subtracting them,
    we **select a set of candidate periods and fit all of their seasonal profiles simultaneously**
    with one linear model. Each profile is estimated conditioning on the others, so the result is
    order-independent.

    For integer periods, selection uses a forward step with BIC and a divisor factorization pass.
    For calendar rules, we fit the chosen rules jointly directly.
    """)

    if demo_mode.value == "5-seasonality stress test":
        mo.md("*Stress-test mode active: the generator contains 5 co-prime seasonalities. "
               "Make sure **Max periods in joint model** is set to 5 or more to see all of them.*")

    if demo_mode.value == "calendar rules":
        # Calendar mode: directly fit the selected rules.
        cal_out = extract_calendar_seasonality(series, index, generator_periods)
        selected = cal_out["rules"]
        multi_result = cal_out["result"]
        mode_note = "calendar"
    else:
        # Integer mode: select from ANOVA candidates.
        candidate_periods = [r.period for r in results if r.p_value < 0.05]
        if not candidate_periods:
            candidate_periods = [r.period for r in results[:10]]
        selected = select_seasonalities(
            series,
            candidate_periods,
            criterion="bic",
            max_periods=max_selected.value,
        )
        multi_result = extract_multiple_seasonalities(series, selected)
        mode_note = "integer periods"

    true_periods_str = ", ".join(str(p) for p in generator_periods)
    mo.md(
        f"""
        **Mode:** {mode_note}

        **Selected / fitted:** {selected}

        **True generator periods / rules:** {true_periods_str}
        """
    )
    return (multi_result,)


@app.cell
def _(demo_mode, mo, plt, multi_result, series):
    """Act 2 — joint fit, residual, and per-component profiles."""
    multi = multi_result

    if multi is None or not multi.periods:
        multi_section = mo.md("No periods / rules were selected, so the joint model is empty.")
    else:
        fig_joint, ax_joint = plt.subplots(figsize=(10, 3))
        ax_joint.plot(series, lw=0.7, alpha=0.8, label="Original")
        ax_joint.plot(multi.fitted, lw=1.2, label="Joint fitted seasonal")
        ax_joint.set_title(f"Joint fit ({multi.periods})")
        ax_joint.set_xlabel("Time")
        ax_joint.set_ylabel("Value")
        ax_joint.legend()

        fig_joint_res, ax_joint_res = plt.subplots(figsize=(10, 3))
        ax_joint_res.plot(multi.residual, lw=0.7, color="gray")
        ax_joint_res.axhline(0, color="black", ls="--", lw=0.5)
        ax_joint_res.set_title(f"Residual after joint extraction (total explained variance = {multi.total_explained_var:.2%})")
        ax_joint_res.set_xlabel("Time")
        ax_joint_res.set_ylabel("Value")

        # Use rule names for labels when in calendar mode.
        label_by_period = {}
        if demo_mode.value == "calendar rules" and hasattr(multi_result, "components_by_rule"):
            label_by_period = {comp.period: rule for rule, comp in multi_result.components_by_rule.items()}

        profile_plots = []
        for period in multi.periods:
            fig_prof, ax_prof = plt.subplots(figsize=(5, 2.5))
            label = label_by_period.get(period, f"Period {period}")
            profile = multi.components[period].profile
            # For calendar rules, phase 0 may be unused; drop leading zeros from the plot.
            if demo_mode.value == "calendar rules":
                first_used = next((i for i, v in enumerate(profile) if abs(v) > 1e-12), 0)
                x_positions = list(range(first_used, len(profile)))
                profile_plot = profile[first_used:]
            else:
                x_positions = list(range(1, period + 1))
                profile_plot = profile
            ax_prof.bar(x_positions, profile_plot, color="steelblue", edgecolor="black")
            ax_prof.set_xlabel("Phase")
            ax_prof.set_ylabel("Effect")
            ax_prof.set_title(f"{label} (share {multi.components[period].explained_var:.1%})")
            ax_prof.set_xticks(x_positions)
            profile_plots.append(mo.mpl.interactive(fig_prof))

        multi_section = mo.vstack([
            mo.mpl.interactive(fig_joint),
            mo.mpl.interactive(fig_joint_res),
            mo.hstack(profile_plots) if len(profile_plots) > 1 else profile_plots[0],
        ])

    multi_section
    return


@app.cell
def _(mo):
    """Closing: what to remember."""
    mo.md("""
    ### Takeaway

    - **Single detection** is a good first look, but it is greedy and order-sensitive.
    - **Joint extraction** is the recommended workflow when you suspect more than one seasonality.
    - For **integer periods**, let BIC select from `scan_periods` candidates.
    - For **calendar rules**, pass a `DatetimeIndex` and the rule names directly.
    - The fitted signal is a deterministic tiling of each learned profile, and the residual is
      simply `series - fitted`.
    """)
    return


if __name__ == "__main__":
    app.run()
