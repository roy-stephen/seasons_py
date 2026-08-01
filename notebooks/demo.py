import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    """Setup: imports and public API."""
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from seasons_py import (
        detect_seasonality,
        scan_periods,
        extract_seasonality,
        extract_multiple_seasonalities,
        extract_calendar_seasonality,
        select_calendar_seasonality,
        select_seasonalities,
    )

    return (
        alt,
        detect_seasonality,
        extract_calendar_seasonality,
        extract_multiple_seasonalities,
        extract_seasonality,
        mo,
        np,
        pd,
        scan_periods,
        select_calendar_seasonality,
        select_seasonalities,
    )


@app.cell
def _(np):
    """Aesthetic helpers for clean, interactive Altair plots using default colors."""
    def _label_for_period(period, rule):
        if rule == "dow":
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if rule == "month":
            return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if rule == "quarter":
            return ["Q1", "Q2", "Q3", "Q4"]
        if rule == "week_of_year":
            return [str(i) for i in range(1, period + 1)]
        if rule in {"week_of_month", "week_of_month_monday"}:
            return [f"W{i}" for i in range(1, period + 1)]
        if rule == "dom":
            return [str(i) for i in range(1, period + 1)]
        return [str(i) for i in range(1, period + 1)]

    def _calendar_first_used(rule):
        """Rules that are naturally 1-indexed have an unused zero slot in our arrays."""
        return 1 if rule in {"month", "dom", "quarter", "doy", "week_of_year", "week_of_month", "week_of_month_monday"} else 0

    def chart_fit_and_residual(alt, pd, series, fitted, residual, total_explained, title=""):
        """Interactive original + fitted and residual time-series charts."""
        n = len(series)
        df = pd.DataFrame({
            "time": np.arange(n),
            "original": series,
            "fitted": fitted,
            "residual": residual,
        })

        base = alt.Chart(df).encode(x=alt.X("time:Q", title="Time"))

        # Long-form data so Altair can produce a real color legend.
        df_melt = df.melt("time", var_name="series", value_name="value")
        # Filter to only the series we want in the top panel.
        df_top = df_melt[df_melt["series"].isin(["original", "fitted"])]
        top_chart = alt.Chart(df_top).mark_line().encode(
            x=alt.X("time:Q", title="Time"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["original", "fitted"], range=["steelblue", "coral"]),
                legend=alt.Legend(title=""),
            ),
            tooltip=["time", "series", "value"],
        ).properties(
            title=f"{title} — explained variance {total_explained:.1%}",
            width=900,
            height=180,
        )

        residual_chart = base.mark_line(size=1).encode(
            y=alt.Y("residual:Q", title="Residual"),
            tooltip=["time", "residual"],
        ) + base.mark_rule(strokeDash=[4, 4]).encode(y=alt.datum(0)).properties(
            width=900,
            height=140,
        )

        return alt.vconcat(top_chart, residual_chart).configure_view(stroke=None)

    def chart_evidence(alt, pd, np, results, best, selected_periods, generator_periods):
        """Interactive ANOVA evidence stem chart with significance highlighting."""
        n_tests = len(results)
        corrected_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
        threshold_logp = -np.log10(corrected_alpha)

        periods = np.array([r.period for r in results])
        pvals = np.array([r.p_value for r in results])
        logp = -np.log10(np.maximum(pvals, 1e-300))
        cap = 25.0
        capped_logp = np.minimum(logp, cap)

        selected_set = set(int(p) for p in selected_periods)
        generator_set = set(int(p) for p in generator_periods if str(p).isdigit())

        df = pd.DataFrame({
            "period": periods,
            "logp": capped_logp,
            "raw_logp": np.minimum(logp, 999.0),
            "kind": np.where(
                capped_logp >= threshold_logp,
                np.where(
                    np.isin(periods, list(selected_set)),
                    "selected",
                    "significant",
                ),
                "not significant",
            ),
        })

        # Faint background points for all candidates.
        background = alt.Chart(df).mark_point(filled=True, size=20, opacity=0.25).encode(
            x=alt.X("period:Q", title="Candidate period"),
            y=alt.Y("logp:Q", title="-log10(p-value)", scale=alt.Scale(domainMin=0)),
        )

        # Prominent points colored by kind.
        points = alt.Chart(df).mark_point(filled=True, size=70).encode(
            x=alt.X("period:Q", title="Candidate period"),
            y=alt.Y("logp:Q", title="-log10(p-value)", scale=alt.Scale(domainMin=0)),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["not significant", "significant", "selected"],
                    range=["#bbbbbb", "#1f77b4", "#d62728"],
                ),
                legend=alt.Legend(title="Candidate status"),
            ),
            tooltip=["period", "raw_logp", "kind"],
        )

        # Stems only for significant candidates.
        sig_df = df[df["kind"] != "not significant"]
        stems = alt.Chart(sig_df).mark_rule(opacity=0.5).encode(
            x="period:Q",
            y="logp:Q",
            y2=alt.datum(0),
        )

        threshold = alt.Chart(pd.DataFrame({"y": [threshold_logp]})).mark_rule(
            strokeDash=[4, 4]
        ).encode(y="y:Q")

        # Vertical reference lines for true generator periods.
        true_periods_df = pd.DataFrame({"x": list(generator_set)})
        true_lines = alt.Chart(true_periods_df).mark_rule(strokeDash=[4, 4], opacity=0.7).encode(
            x="x:Q"
        )

        # Vertical line for the single best period.
        detected_x = best.period if best else None
        layers = [background, stems, threshold, true_lines]
        if detected_x is not None:
            detected_line = alt.Chart(pd.DataFrame({"x": [detected_x]})).mark_rule(
                size=2, opacity=0.7
            ).encode(x="x:Q")
            layers.append(detected_line)
        layers.append(points)

        chart = alt.layer(*layers).properties(
            title="ANOVA evidence per candidate period",
            width=900,
            height=220,
        ).configure_view(stroke=None)
        return chart

    def chart_profile_grid(alt, pd, np, periods, components, demo_mode, rule_map):
        """Grid of interactive baseline line charts for each seasonal profile."""
        n = len(periods)
        if n == 0:
            return None

        cols = min(n, 2)
        # Altair adds ~40 px of spacing between hconcat charts.
        spacing = 40
        subplot_width = min(450, max(280, (900 - spacing * (cols - 1)) // cols))
        if cols == 1:
            subplot_width = 900
        charts = []

        for idx, period in enumerate(periods):
            comp = components[period]
            profile = comp.profile
            rule = rule_map.get(period)
            title = rule.capitalize() if rule else f"Period {period}"
            labels = _label_for_period(period, rule)

            if demo_mode == "calendar rules":
                first_used = _calendar_first_used(rule)
            else:
                first_used = 0
            display_profile = profile[first_used:]
            # Labels for 1-indexed rules are already aligned with phase 1..N;
            # only the profile has an unused index-0 slot to drop.
            display_labels = labels if first_used == 1 else labels[first_used:]
            display_labels = display_labels[: len(display_profile)]
            display_profile = display_profile[: len(display_labels)]
            n_phases = len(display_profile)

            df = pd.DataFrame({
                "phase": np.arange(n_phases),
                "effect": display_profile,
                "label": display_labels,
                "sign": np.where(display_profile >= 0, "positive", "negative"),
            })

            axis_kwargs = {"labels": n_phases <= 24}
            if n_phases > 24:
                axis_kwargs["values"] = list(range(0, n_phases, max(1, n_phases // 6)))
            x_encode = alt.X(
                "phase:O",
                title="Phase",
                axis=alt.Axis(**axis_kwargs),
            )

            # Baseline rule at y=0.
            baseline = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")

            # Neutral light fill under the whole line.
            df_area = pd.DataFrame({
                "phase": np.arange(n_phases),
                "effect": display_profile,
                "label": display_labels,
            })
            area = alt.Chart(df_area).mark_area(opacity=0.15, interpolate="linear").encode(
                x=x_encode,
                y=alt.Y("effect:Q", title="Effect"),
            )

            line = alt.Chart(df).mark_line(size=2).encode(
                x=x_encode,
                y="effect:Q",
            )
            points = alt.Chart(df).mark_circle(size=60).encode(
                x=x_encode,
                y="effect:Q",
                color=alt.Color(
                    "sign:N",
                    scale=alt.Scale(domain=["positive", "negative"], range=["#1f77b4", "#d62728"]),
                    legend=alt.Legend(title="Effect direction") if idx == 0 else None,
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Phase"),
                    alt.Tooltip("effect:Q", title="Effect", format=".2f"),
                    alt.Tooltip("sign:N", title="Direction"),
                ],
            )

            chart = (baseline + area + line + points).properties(
                title=f"{title}  |  share {comp.explained_var:.1%}",
                width=subplot_width,
                height=200,
            )
            charts.append(chart)

        rows = []
        for i in range(0, len(charts), cols):
            rows.append(alt.hconcat(*charts[i:i + cols]))
        return alt.vconcat(*rows).configure_view(stroke=None)

    return chart_evidence, chart_fit_and_residual, chart_profile_grid


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
    max_selected = mo.ui.slider(1, 8, value=5, label="Max periods / rules in joint model")
    calendar_rules = mo.ui.dropdown(
        options=[
            "dow,month",
            "dow,dom",
            "dow,month,dom",
            "month,quarter",
            "dow,week_of_month",
            "week_of_year",
            "week_of_month,week_of_month_monday",
            "auto",
        ],
        value="dow,month",
        label="Calendar rules (choose 'auto' to search all)",
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
    _rules = [r.strip() for r in calendar_rules.value.split(",")]

    np.random.seed(0)

    def make_pattern(s):
        p = np.linspace(5.0, -5.0, s) + np.random.normal(0, 1.0, s)
        return p - p.mean()

    if _mode == "5-seasonality stress test":
        preset_periods = [3, 5, 7, 11, 13]
        _cycle = int(np.lcm.reduce(preset_periods))
        base = np.zeros(_cycle)
        for p in preset_periods:
            base += np.tile(make_pattern(p), _cycle // p)
        series = np.tile(base, _n // _cycle + 1)[:_n]
        generator_periods = preset_periods
        index = None
    elif _mode == "calendar rules":
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
            elif rule == "week_of_year":
                woy_profile = np.zeros(53)
                woy_profile[0] = 2.0
                woy_profile[25] = -3.0
                woy_profile[51] = 2.5
                woy_profile = woy_profile - woy_profile.mean()
                series += woy_profile[index.isocalendar().week - 1]
            elif rule == "week_of_month":
                wom_profile = np.array([1.5, 0.5, -1.0, -2.0, 0.0])
                wom_profile = wom_profile - wom_profile.mean()
                series += wom_profile[(index.day - 1) // 7]
            elif rule == "week_of_month_monday":
                first_day = index - pd.offsets.MonthBegin()
                first_dow = first_day.dayofweek
                days_since_first_monday = (index.day - 1) - ((7 - first_dow) % 7)
                week = (days_since_first_monday // 7).astype(int)
                wom_mon_profile = np.array([1.0, 0.0, -1.5, -2.5, 0.5])
                wom_mon_profile = wom_mon_profile - wom_mon_profile.mean()
                series += wom_mon_profile[week]
        series = series - series.mean()
        series = series[:_n]
        index = index[:_n]
        generator_periods = _rules
    else:
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
def _(alt, generator_periods, index, mo, np, pd, series):
    """Setup: plot the detrended series, show a data preview, and verify mean/slope."""
    _mean = float(np.mean(series))
    _slope = float(np.linalg.lstsq(
        np.column_stack([np.ones(len(series)), np.arange(len(series))]),
        series,
        rcond=None,
    )[0][1])

    active_periods = ", ".join(str(p) for p in generator_periods)
    index_note = f"Index: {index[0].date()} to {index[-1].date()}" if index is not None else "Index: positional integer index"

    if index is not None:
        preview_df = pd.DataFrame({"value": series[:12]}, index=index[:12])
    else:
        preview_df = pd.DataFrame({"time": np.arange(12), "value": series[:12]})

    df_plot = pd.DataFrame({"time": np.arange(len(series)), "value": series})
    raw_chart = alt.Chart(df_plot).mark_line(size=1.5).encode(
        x=alt.X("time:Q", title="Time"),
        y=alt.Y("value:Q", title="Value"),
        tooltip=["time", "value"],
    ).properties(title="Synthetic detrended series", width=900, height=160).configure_view(stroke=None)

    mo.vstack([
        mo.md(f"**True periods / rules in the generator:** {active_periods}"),
        mo.md(f"*{index_note}*"),
        raw_chart,
    ])
    return


@app.cell
def _(detect_seasonality, mo, scan_periods, series):
    """Act 1 — Single-period detection."""
    s1_detect = mo.md("""
    ## Single-period detection

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
def _(alt, best, chart_evidence, generator_periods, mo, np, pd, results, selected):
    """Act 1 — single-period evidence plot."""
    evidence_chart = chart_evidence(alt, pd, np, results, best, selected, generator_periods)
    mo.vstack([mo.md("""
    *Hover for raw -log10(p-value). Dashed line = Bonferroni-corrected threshold.
    **Significant** candidates pass the threshold; **selected** candidates are the ones BIC kept.
    Dashed vertical lines mark true generator periods.*"""), evidence_chart])
    return


@app.cell
def _(
    alt,
    best,
    chart_fit_and_residual,
    chart_profile_grid,
    extract_seasonality,
    mo,
    np,
    pd,
    series,
):
    """Act 1 — single-period extraction (fit / residual / profile)."""
    mo.md("### Extract the single detected seasonality")

    extracted = extract_seasonality(series, best.period) if best else None

    if extracted is None:
        single_section = mo.md("No significant period was detected, so single extraction is skipped.")
    else:
        single_fit_chart = chart_fit_and_residual(
            alt,
            pd,
            series,
            extracted.fitted,
            series - extracted.fitted,
            extracted.explained_var,
            title=f"Single-period fit (period = {best.period})",
        )

        single_profile_chart = chart_profile_grid(
            alt,
            pd,
            np,
            [best.period],
            {best.period: extracted},
            demo_mode="integer periods",
            rule_map={best.period: None},
        )

        single_section = mo.vstack([
            single_fit_chart,
            single_profile_chart,
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
    select_calendar_seasonality,
    select_seasonalities,
    series,
):
    """Act 2 — multiple-period / calendar selection (the main feature)."""
    mo.md("""
    ## 2. Multiple-period / calendar extraction (joint OLS)

    Instead of detecting periods one at a time and subtracting them, we **select a
    set of candidates and fit all of their seasonal profiles simultaneously** with
    one linear model. Each profile is estimated conditioning on the others, so the
    result is order-independent.

    For integer periods, selection uses a forward step with BIC and a divisor
    factorization pass. For calendar rules, choose specific rules or use **auto**
    to let BIC search across all sensible calendar rules.
    """)

    if demo_mode.value == "5-seasonality stress test":
        mo.md("*Stress-test mode active: the generator contains 5 co-prime seasonalities. "
               "Make sure **Max periods in joint model** is set to 5 or more to see all of them.*")

    if demo_mode.value == "calendar rules":
        if "auto" in generator_periods:
            cal_out = select_calendar_seasonality(
                series,
                index,
                rules=None,
                criterion="bic",
                max_rules=max_selected.value,
            )
            selected = cal_out["selected_rules"]
            multi_result = cal_out["result"]
        else:
            cal_out = extract_calendar_seasonality(series, index, generator_periods)
            selected = cal_out["rules"]
            multi_result = cal_out["result"]
        mode_note = "calendar"
    else:
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
def _(
    alt,
    chart_fit_and_residual,
    chart_profile_grid,
    demo_mode,
    mo,
    multi_result,
    np,
    pd,
    series,
):
    """Act 2 — joint fit, residual, and per-component profiles."""
    multi = multi_result

    if multi is None or not multi.periods:
        multi_section = mo.md("No periods / rules were selected, so the joint model is empty.")
    else:
        rule_map = {}
        if demo_mode.value == "calendar rules" and hasattr(multi, "components_by_rule"):
            rule_map = {comp.period: rule for rule, comp in multi.components_by_rule.items()}

        multi_fit_chart = chart_fit_and_residual(
            alt,
            pd,
            series,
            multi.fitted,
            multi.residual,
            multi.total_explained_var,
            title=f"Joint fit ({multi.periods})",
        )

        multi_profile_chart = chart_profile_grid(
            alt,
            pd,
            np,
            multi.periods,
            multi.components,
            demo_mode=demo_mode.value,
            rule_map=rule_map,
        )

        multi_section = mo.vstack([
            multi_fit_chart,
            multi_profile_chart,
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
    - For **calendar rules**, pass a `DatetimeIndex` and the rule names directly, or use `select_calendar_seasonality` to search.
    - The fitted signal is a deterministic tiling of each learned profile, and the residual is
      simply `series - fitted`.
    """)
    return


@app.cell
def _(
    alt,
    chart_evidence,
    chart_fit_and_residual,
    chart_profile_grid,
    detect_seasonality,
    extract_multiple_seasonalities,
    mo,
    np,
    pd,
    scan_periods,
    select_calendar_seasonality,
    select_seasonalities,
):
    """Real-data test: load series_test.txt and apply the whole pipeline."""
    mo.md("""
    ## 3. Real-data test: `series_test.txt`

    This section loads an external tab-separated series (`date\tseries`) and runs
    the same detection + extraction pipeline. The series is business-daily and
    assumed already detrended. Calendar selection uses `select_calendar_seasonality`
    so we do not have to pre-specify which rules matter.
    """)

    df_test = pd.read_csv("series_test.txt", sep="\t", parse_dates=["date"])
    series_test = df_test["series"].values
    index_test = pd.DatetimeIndex(df_test["date"])

    _mean = float(np.mean(series_test))
    _slope = float(np.linalg.lstsq(
        np.column_stack([np.ones(len(series_test)), np.arange(len(series_test))]),
        series_test,
        rcond=None,
    )[0][1])

    test_df_plot = pd.DataFrame({"time": np.arange(len(series_test)), "value": series_test})
    test_raw_chart = alt.Chart(test_df_plot).mark_line(size=1).encode(
        x=alt.X("time:Q", title="Time"),
        y=alt.Y("value:Q", title="Value"),
        tooltip=["time", "value"],
    ).properties(title="series_test.txt raw series", width=900, height=160).configure_view(stroke=None)

    results_test = scan_periods(series_test)
    best_test = detect_seasonality(series_test)

    test_candidate_periods = [r.period for r in results_test if r.p_value < 0.05]
    selected_test = select_seasonalities(series_test, test_candidate_periods, criterion="bic", max_periods=5)
    integer_result = extract_multiple_seasonalities(series_test, selected_test)

    test_cal_out = select_calendar_seasonality(series_test, index_test, rules=None, max_rules=4)
    calendar_result = test_cal_out["result"]
    calendar_rules_test = test_cal_out["selected_rules"]

    integer_evidence_chart = chart_evidence(alt, pd, np, results_test, best_test, selected_test, [])

    integer_fit_chart = chart_fit_and_residual(
        alt,
        pd,
        series_test,
        integer_result.fitted,
        integer_result.residual,
        integer_result.total_explained_var,
        title=f"Integer joint fit — selected {selected_test}",
    )

    cal_rule_map = {comp.period: rule for rule, comp in test_cal_out["result"].components_by_rule.items()}
    calendar_profile_chart = chart_profile_grid(
        alt,
        pd,
        np,
        calendar_result.periods,
        calendar_result.components,
        demo_mode="calendar rules",
        rule_map=cal_rule_map,
    )
    calendar_fit_chart = chart_fit_and_residual(
        alt,
        pd,
        series_test,
        calendar_result.fitted,
        calendar_result.residual,
        calendar_result.total_explained_var,
        title=f"Calendar joint fit — selected rules {calendar_rules_test}",
    )

    cal_shares = "\n".join(
        f"- `{rule}`: {comp.explained_var:.1%}"
        for rule, comp in test_cal_out["result"].components_by_rule.items()
    )
    summary = mo.md(
        f"""
        **Series length:** {len(series_test)} observations

        **Detrended check:** mean = `{_mean:.3f}`, residual linear slope = `{_slope:.2e}`

        **Single detection:** period = `{best_test.period if best_test else None}`

        **BIC integer selection:** `{selected_test}` (explained = {integer_result.total_explained_var:.1%})

        **Auto-selected calendar rules:** `{calendar_rules_test}` (explained = {calendar_result.total_explained_var:.1%})

        **Calendar component shares:**
        {cal_shares}
        """
    )

    mo.vstack([
        summary,
        test_raw_chart,
        mo.md("### Integer-period evidence and fit"),
        integer_evidence_chart,
        integer_fit_chart,
        mo.md("### Calendar-seasonality fit and profiles"),
        calendar_fit_chart,
        calendar_profile_chart,
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
