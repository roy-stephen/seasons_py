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
        extract_calendar_seasonality,
        extract_multiple_seasonalities,
        extract_seasonality,
        mo,
        np,
        pd,
        plt,
        scan_periods,
        select_seasonalities,
    )


@app.cell
def _():
    """Aesthetic helpers for clean, single-color seasonal plots."""
    _ACCENT = "#2c3e50"
    _ACCENT_LIGHT = "#5d7a99"
    _MUTED = "#95a5a6"

    def _label_for_period(period, rule):
        if rule == "dow":
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if rule == "month":
            return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if rule == "quarter":
            return ["Q1", "Q2", "Q3", "Q4"]
        if rule == "dom":
            return [str(i) for i in range(1, period + 1)]
        return [str(i) for i in range(1, period + 1)]

    def _calendar_first_used(rule):
        """Rules that are naturally 1-indexed have an unused zero slot in our arrays."""
        return 1 if rule in {"month", "dom", "quarter", "doy"} else 0

    def plot_fit_and_residual(plt, series, fitted, residual, total_explained, title=""):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            ax0, ax1 = axes

            ax0.plot(series, lw=0.7, color=_MUTED, alpha=0.7, label="Original")
            ax0.plot(fitted, lw=1.6, color=_ACCENT, label="Fitted seasonal")
            ax0.set_title(f"{title}\nTotal explained variance: {total_explained:.1%}", loc="left", fontsize=11)
            ax0.set_ylabel("Value", fontsize=9)
            ax0.legend(frameon=True, fancybox=False, edgecolor="white", loc="upper right")

            ax1.plot(residual, lw=0.7, color=_MUTED)
            ax1.axhline(0, color=_ACCENT, ls="--", lw=1.0)
            ax1.set_title("Residual", loc="left", fontsize=11)
            ax1.set_xlabel("Time", fontsize=9)
            ax1.set_ylabel("Value", fontsize=9)

            fig.tight_layout()
        return fig

    def plot_profile_grid(plt, np, periods, components, demo_mode, rule_map):
        n = len(periods)
        if n == 0:
            return None
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig = plt.figure(figsize=(3.2 * cols + 2.0, 3.2 * rows))
            fig.patch.set_facecolor("white")
            gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [0.35])

            for idx, period in enumerate(periods):
                comp = components[period]
                profile = comp.profile
                rule = rule_map.get(period)
                title = rule.capitalize() if rule else f"Period {period}"
                labels = _label_for_period(period, rule)

                # Calendar profiles may have an unused index 0; skip it for display.
                if demo_mode == "calendar rules":
                    first_used = _calendar_first_used(rule)
                else:
                    first_used = 0
                display_profile = profile[first_used:]
                display_labels = labels[first_used:]

                # Ensure label count matches data count (defensive, in case widths differ).
                display_labels = display_labels[: len(display_profile)]
                display_profile = display_profile[: len(display_labels)]
                n_phases = len(display_profile)
                theta = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)

                ax = fig.add_subplot(gs[idx // cols, idx % cols], projection="polar" if n_phases <= 12 else None)
                if n_phases <= 12:
                    # Seasonal clock: line + markers + filled area + zero reference circle.
                    ax.set_theta_zero_location("N")
                    ax.set_theta_direction(-1)
                    # Close the loop for the line plot.
                    theta_closed = np.append(theta, theta[0])
                    values_closed = np.append(display_profile, display_profile[0])

                    # Draw zero-reference circle.
                    ax.plot(theta_closed, np.zeros_like(theta_closed), color=_MUTED, ls="--", lw=1.0, label="zero")
                    # Fill between profile and zero.
                    ax.fill_between(theta_closed, 0, values_closed, color=_ACCENT_LIGHT, alpha=0.4)
                    # Plot the profile line with markers.
                    ax.plot(theta_closed, values_closed, color=_ACCENT, lw=1.5, marker="o", markersize=5)

                    ax.set_xticks(theta)
                    ax.set_xticklabels(display_labels, fontsize=8)
                    # Minimal radial ticks.
                    vmax = np.max(np.abs(display_profile))
                    if vmax > 0:
                        rticks = np.linspace(0, vmax, num=3)
                        ax.set_yticks(rticks)
                        ax.set_yticklabels([f"{v:.1f}" for v in rticks], fontsize=7, color="#555555")
                    else:
                        ax.set_yticks([])
                    ax.set_title(f"{title}\n(share {comp.explained_var:.1%})", fontsize=10, pad=10)
                else:
                    # Horizontal bar plot for long cycles.
                    y_positions = np.arange(n_phases)
                    colors = [_ACCENT if v >= 0 else _ACCENT_LIGHT for v in display_profile]
                    ax.barh(y_positions, display_profile, color=colors, edgecolor=_ACCENT, lw=0.5)
                    ax.axvline(0, color=_ACCENT, lw=0.8)
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels(display_labels, fontsize=7)
                    ax.set_xlabel("Effect", fontsize=9)
                    ax.set_title(f"{title}\n(share {comp.explained_var:.1%})", fontsize=10, loc="left")

            # Variance decomposition subplot on the far right.
            ax_var = fig.add_subplot(gs[:, -1])
            shares = [components[p].explained_var for p in periods]
            y = np.arange(n)
            ax_var.barh(y, shares, color=_ACCENT, edgecolor=_ACCENT, height=0.6)
            ax_var.set_yticks(y)
            ax_var.set_yticklabels([rule_map.get(p, f"P{p}").capitalize() if rule_map.get(p) else f"Period {p}" for p in periods], fontsize=8)
            ax_var.set_xlabel("Explained share", fontsize=8)
            ax_var.set_title("Variance\ndecomposition", fontsize=9, loc="left")
            ax_var.set_xlim(0, 1)

            fig.tight_layout()
        return fig

    return (
        plot_fit_and_residual,
        plot_profile_grid,
    )


@app.cell
def _(mo):
    """Setup: user-facing controls."""
    headr = mo.md("""
    # seasons_py demo

    **Assumptions:** the input series is already detrended.

    This notebook walks through three ideas:
    1. **Single-period extraction** — detect one integer seasonality and extract its profile.
    2. **Multiple-period extraction** — select and estimate several integer seasonalities *jointly*,
       so that each effect is conditioned on the others.
    3. **Calendar seasonality** — same joint estimator, but phases come from a `DatetimeIndex`
       (day-of-week, month-of-year, day-of-month, etc.).
    """)

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
    mo.vstack([headr, controls])
    return (
        calendar_rules,
        demo_mode,
        max_selected,
        n_points,
        noise_std,
        second_period,
        true_period,
    )


@app.cell
def _(
    calendar_rules,
    demo_mode,
    max_selected,
    n_points,
    noise_std,
    np,
    pd,
    second_period,
    true_period,
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
    return generator_periods, index, series


@app.cell
def _(generator_periods, index, mo, np, pd, plt, series):
    """Setup: plot the detrended series, show a data preview, and verify mean/slope."""
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

    # Build a small data preview as a pandas DataFrame.
    if index is not None:
        preview_df = pd.DataFrame({"value": series[:12]}, index=index[:12])
    else:
        preview_df = pd.DataFrame({"time": np.arange(12), "value": series[:12]})

    mo.vstack([
        mo.md(f"**True periods / rules in the generator:** {active_periods}"),
        mo.md(f"*{index_note}*"),
        mo.md(
            f"**Detrended check:** mean = `{_mean:.3f}`, residual linear slope = `{_slope:.2e}`. "
            f"These should be essentially zero."
        ),
        mo.md("**First 12 observations:**"),
        mo.ui.table(data=preview_df, selection=None),
        mo.mpl.interactive(fig_raw),
    ])
    return


@app.cell
def _(detect_seasonality, mo, scan_periods, series):
    """Act 1 — Single-period detection."""
    s1_detect = mo.md("""
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

    res1 = mo.md(
        f"""
        **Detected period:** {_detected}

        **p-value:** {_pval}

        **Number of tests:** {_n_tests} (Bonferroni-corrected α = {_alpha_corr:.2e})
        """
    )

    mo.vstack([s1_detect, res1])
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
def _(best, extract_seasonality, mo, np, plt, series, plot_fit_and_residual, plot_profile_grid):
    """Act 1 — single-period extraction (fit / residual / profile)."""
    mo.md("### Extract the single detected seasonality")

    extracted = extract_seasonality(series, best.period) if best else None

    if extracted is None:
        single_section = mo.md("No significant period was detected, so single extraction is skipped.")
    else:
        single_fit_fig = plot_fit_and_residual(
            plt,
            series,
            extracted.fitted,
            series - extracted.fitted,
            extracted.explained_var,
            title=f"Single-period fit (period = {best.period})",
        )

        single_profile_fig = plot_profile_grid(
            plt,
            np,
            [best.period],
            {best.period: extracted},
            demo_mode="integer periods",
            rule_map={best.period: None},
        )

        single_section = mo.vstack([
            mo.mpl.interactive(single_fit_fig),
            mo.mpl.interactive(single_profile_fig),
        ])

    single_section
    return


@app.cell
def _(
    demo_mode,
    extract_calendar_seasonality,
    extract_multiple_seasonalities,
    generator_periods,
    index,
    max_selected,
    mo,
    results,
    select_seasonalities,
    series,
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
def _(demo_mode, mo, multi_result, np, plt, series, plot_fit_and_residual, plot_profile_grid):
    """Act 2 — joint fit, residual, and per-component profiles."""
    multi = multi_result

    if multi is None or not multi.periods:
        multi_section = mo.md("No periods / rules were selected, so the joint model is empty.")
    else:
        # Map raw period back to rule name for calendar mode labels.
        rule_map = {}
        if demo_mode.value == "calendar rules" and hasattr(multi, "components_by_rule"):
            rule_map = {comp.period: rule for rule, comp in multi.components_by_rule.items()}

        multi_fit_fig = plot_fit_and_residual(
            plt,
            series,
            multi.fitted,
            multi.residual,
            multi.total_explained_var,
            title=f"Joint fit ({multi.periods})",
        )

        multi_profile_fig = plot_profile_grid(
            plt,
            np,
            multi.periods,
            multi.components,
            demo_mode=demo_mode.value,
            rule_map=rule_map,
        )

        multi_section = mo.vstack([
            mo.mpl.interactive(multi_fit_fig),
            mo.mpl.interactive(multi_profile_fig),
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
