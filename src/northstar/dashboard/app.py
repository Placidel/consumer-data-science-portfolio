"""Northstar Consumer executive sales and marketing dashboard (section 07).

Launch with ``northstar dashboard`` (or ``streamlit run src/northstar/dashboard/app.py``).
Everything shown is read from the saved outputs of the section pipelines; filters slice those
outputs and never refit a model. See ``projects/07_dashboard/README.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from northstar.dashboard import views
from northstar.dashboard.artifacts import (
    SectionArtifacts,
    fingerprint,
    load_all,
    projects_dir,
    provenance,
)
from northstar.dashboard.kpis import (
    OVERVIEW,
    THEMES,
    KPIValue,
    evaluate_all,
    fmt_value,
    kpi_catalog,
)

# Reference categorical palette (same slots and order as the section figures).
S1, S2, S3, S4, S5, S6, S7 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                              "#4a3aa7")
REFERENCE = "#7a7973"  # neutral ink for "random" and other reference series
STATE_COLORS = {"new": S1, "active": S2, "loyal": S3, "at_risk": S4, "churned": S5}
# Colour follows the entity, never its rank, so filters never repaint surviving series.
ENTITY_COLORS = {
    "logistic_regression": S1, "risk_ranked": S1, "bgnbd_gamma_gamma": S1, "harmonic": S1,
    "value_ranked": S2, "gradient_boosting": S2, "harmonic_no_promo": S2,
    "recent_activity": S3, "recency_rule": S3, "run_rate": S3, "seasonal_naive_yoy": S3,
    "channel_rate": S4, "rfm_cell_rate": S4, "rfm_cell_mean": S4, "naive_4wk": S4,
    "random": REFERENCE,
}
CHART_HEIGHT = 280

PAGES = ("Executive overview", "Acquisition", "Conversion", "Retention", "Revenue growth",
         "Forecast", "Lifecycle", "Definitions & sources")


# --- data ----------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load(root: str, stamp: tuple) -> dict[str, SectionArtifacts]:
    """Cached per outputs folder and file fingerprint, so rerunning a pipeline refreshes it."""
    del stamp  # only part of the cache key
    return load_all(Path(root))


def _require(artifacts: Mapping[str, SectionArtifacts], *keys: str) -> bool:
    ok = True
    for key in keys:
        art = artifacts[key]
        if not art.available:
            ok = False
            st.warning(f"Section {art.section.number} ({art.section.title}) outputs are missing "
                       f"({', '.join(art.missing)}). Run `{art.section.command}` and reload.",
                       icon=":material/warning:")
    return ok


def _md(text: str) -> str:
    """Escape ``$`` so Streamlit markdown shows dollar amounts instead of LaTeX."""
    return text.replace("$", "\\$")


def _source(art: SectionArtifacts, *tables: str) -> None:
    files = [art.source(t) for t in tables] if tables else [art.source()]
    st.caption("Source: " + ", ".join(f"`{f}`" for f in files))


# --- display helpers -----------------------------------------------------------------------------

def _metric(v: KPIValue) -> None:
    if not v.available:
        st.metric(v.kpi.label, "n/a", help=_md(v.error or ""))
        return
    kwargs: dict = {}
    if v.comparison:
        kwargs = {"delta": v.comparison,
                  "delta_color": {True: "normal", False: "inverse", None: "off"}[
                      v.kpi.higher_is_better]}
    elif v.interval:
        kwargs = {"delta": v.interval, "delta_color": "off", "delta_arrow": "off"}
    st.metric(v.kpi.label, v.display, help=_md(v.tooltip), **kwargs)


def _metrics_row(values: Mapping[str, KPIValue], keys: Sequence[str]) -> None:
    for col, key in zip(st.columns(len(keys)), keys, strict=True):
        with col:
            _metric(values[key])


def _table(df: pd.DataFrame, columns: Mapping[str, tuple[str, str]]) -> None:
    """Show ``df`` with columns renamed and formatted: ``{column: (header, fmt)}``."""
    def cell(x, fmt: str) -> str:
        if pd.isna(x):
            return ""
        return views.label(x) if fmt == "label" else fmt_value(x, fmt)

    out = pd.DataFrame({header: [cell(x, fmt) for x in df[col]]
                        for col, (header, fmt) in columns.items()})
    st.dataframe(out, hide_index=True, width="stretch")


def _color(domain: Sequence[str], colors: Mapping[str, str], field: str = "series",
           title: str | None = None) -> alt.Color:
    labels = [views.label(d) for d in domain]
    return alt.Color(f"{field}:N", title=title,
                     scale=alt.Scale(domain=labels, range=[colors[d] for d in domain]),
                     legend=alt.Legend(orient="top", labelLimit=260))


def _chart(chart: alt.Chart) -> None:
    st.altair_chart(chart.properties(height=CHART_HEIGHT), width="stretch")


def _bar(df: pd.DataFrame, x: str, y: str, x_title: str, y_format: str, y_title: str,
         sort: Sequence[str] | None = None, color: str = S1) -> alt.Chart:
    """Single-series horizontal bars (categories on the y axis)."""
    return alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format=y_format)),
        y=alt.Y(f"{y}:N", title=y_title, sort=list(sort) if sort is not None else "-x"),
        tooltip=[alt.Tooltip(f"{y}:N", title=y_title or "Level"),
                 alt.Tooltip(f"{x}:Q", title=x_title, format=y_format)],
    )


def _policy_curve(curve: pd.DataFrame, policies: Sequence[str], value: str, depth: float,
                  value_title: str) -> alt.Chart:
    df = curve.loc[curve["policy"].isin(policies)].copy()
    df["series"] = df["policy"].map(views.label)
    ordered = [p for p in ENTITY_COLORS if p in policies]
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("depth:Q", title="Share of customers targeted", axis=alt.Axis(format=".0%")),
        y=alt.Y(f"{value}:Q", title=value_title, axis=alt.Axis(format="$,.0f")),
        color=_color(ordered, ENTITY_COLORS),
        tooltip=[alt.Tooltip("series:N", title="Policy"),
                 alt.Tooltip("depth:Q", title="Depth", format=".0%"),
                 alt.Tooltip(f"{value}:Q", title=value_title, format="$,.0f")],
    )
    rule = alt.Chart(pd.DataFrame({"depth": [depth]})).mark_rule(
        color=REFERENCE, strokeDash=[4, 4]).encode(x="depth:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=REFERENCE, opacity=0.5).encode(
        y="y:Q")
    return lines + rule + zero


# --- pages ---------------------------------------------------------------------------------------

def page_overview(A: Mapping[str, SectionArtifacts], values: Mapping[str, KPIValue]) -> None:
    st.title("Executive overview")
    st.caption("Headline results from every section, read from saved pipeline outputs. Hover "
               "? icon next to a KPI for its definition and source field.")
    themes = list(OVERVIEW)
    for row in (themes[0:2], themes[2:4], themes[4:6]):
        for col, theme in zip(st.columns(2), row, strict=True):
            with col.container(border=True):
                st.markdown(f"**{theme}**")
                _metrics_row(values, OVERVIEW[theme])

    st.subheader("Monthly trend")
    if _require(A, "foundation"):
        f = A["foundation"]
        monthly = f.tables["monthly_kpis"]
        months = monthly["month"].astype(str).tolist()
        start, end = st.select_slider("Months shown", options=months,
                                      value=(months[0], months[-1]), key="overview_months")
        window = views.month_window(monthly, start, end)
        t = views.window_totals(window)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net revenue in window", fmt_value(t["net_revenue"], "usd"))
        c2.metric("Orders in window", fmt_value(t["orders"], "int"))
        c3.metric("New customers in window", fmt_value(t["new_customers"], "int"))
        c4.metric("Average order value in window", fmt_value(t["average_order_value"], "usd"))
        left, right = st.columns(2)
        base = alt.Chart(window).encode(x=alt.X("month:O", title=None,
                                                axis=alt.Axis(labelAngle=-45)))
        with left:
            st.markdown("Net revenue by month")
            _chart(base.mark_bar(color=S1, cornerRadiusEnd=4).encode(
                y=alt.Y("net_revenue:Q", title=None, axis=alt.Axis(format="$,.0f")),
                tooltip=[alt.Tooltip("month:O", title="Month"),
                         alt.Tooltip("net_revenue:Q", title="Net revenue", format="$,.0f"),
                         alt.Tooltip("orders:Q", title="Orders", format=",")]))
        with right:
            st.markdown("New customers by month")
            _chart(base.mark_bar(color=S1, cornerRadiusEnd=4).encode(
                y=alt.Y("new_customers:Q", title=None, axis=alt.Axis(format=",.0f")),
                tooltip=[alt.Tooltip("month:O", title="Month"),
                         alt.Tooltip("new_customers:Q", title="New customers", format=","),
                         alt.Tooltip("new_leads:Q", title="New leads", format=",")]))
        _source(f, "monthly_kpis")

    st.subheader("Where to act")
    bullets = _action_summary(A)
    st.markdown("\n".join(f"- {_md(b)}" for b in bullets) if bullets else
                "Section outputs are missing; see *Definitions & sources*.")
    st.caption("Each line is assembled from the section's saved metrics. The section READMEs "
               "give the reasoning, validation and caveats.")


def _action_summary(A: Mapping[str, SectionArtifacts]) -> list[str]:
    out = []
    if A["acquisition"].available:
        m, budget = A["acquisition"].metrics, A["acquisition"].tables["budget_simulation"]
        cap = float(budget["capacity_share"].min())
        row = views.budget_at_capacity(budget, cap)
        champ = row.loc[row["policy"] == m["champion"]].iloc[0]
        out.append(f"**Acquisition (01):** calling the top {cap:.0%} of open leads by lead score "
                   f"reaches {champ['conversions_reached_per_run']:.0f} eventual buyers per run, "
                   f"{champ['lift_vs_random']:.2f}× random outreach.")
    if A["retention"].available:
        e = A["retention"].metrics["simulation"]["expected_net_positive"]
        out.append(f"**Retention (02):** offering vouchers only where the expected value is "
                   f"positive ({e['depth']:.0%} of the active base) nets "
                   f"{fmt_value(e['net_value_per_run'], 'usd')} per monthly run under the stated "
                   f"assumptions; it breaks even at a {e['break_even_save_rate']:.1%} save rate.")
    if A["conversion"].available:
        d = A["conversion"].metrics["experiment"]["decision"]
        out.append(f"**Conversion (03):** one-page checkout: {d['recommendation_text']}")
    if A["revenue"].available:
        e = A["revenue"].metrics["scenario"]["expected_net_positive"]
        out.append(f"**Revenue (04):** a growth perk for the {e['customers_targeted']:,.0f} "
                   f"customers whose predicted margin covers its cost nets "
                   f"{fmt_value(e['net_value'], 'usd')}; it breaks even at "
                   f"{e['break_even_uplift']:.1%} uplift.")
    if A["forecast"].available:
        t = A["forecast"].metrics["forecast"]["total"]
        out.append(f"**Forecast (05):** plan on {fmt_value(t['forecast'], 'usd')} net revenue "
                   f"over the next 13 weeks (80% interval {fmt_value(t['lower_80'], 'usd')} to "
                   f"{fmt_value(t['upper_80'], 'usd')}).")
    if A["lifecycle"].available:
        dp = A["lifecycle"].tables["decision_points"]
        top = dp.loc[dp["revenue_gap_per_year"].idxmax()]
        out.append(f"**Lifecycle (06):** the largest lifecycle gap is *{top['label'].lower()}*: "
                   f"{fmt_value(top['revenue_gap_per_year'], 'usd')} a year between the good and "
                   f"bad path (descriptive upper bound, not a program effect).")
    return out


def page_acquisition(A, values) -> None:
    st.title("Acquisition")
    st.caption("Which channels bring customers efficiently, and how much a lead score improves "
               "outreach. Sections 00 and 01.")
    _metrics_row(values, ("leads", "lead_conversion_60d", "lead_score_auc",
                          "lead_score_lift_top10"))
    if _require(A, "foundation"):
        f = A["foundation"]
        channels = f.tables["channel_summary"]
        all_channels = channels["acquisition_channel"].tolist()
        chosen = st.multiselect("Acquisition channels", all_channels, default=all_channels,
                                format_func=views.label, key="acq_channels")
        shown = views.filter_channels(channels, chosen)
        shown = shown.assign(channel=shown["acquisition_channel"].map(views.label))
        if shown.empty:
            st.info("Select at least one channel.")
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("Leads converting within 60 days")
                _chart(_bar(shown, "conversion_rate_60d", "channel", "Conversion rate", ".0%",
                            ""))
            with right:
                st.markdown("Sourcing cost per acquired customer")
                _chart(_bar(shown, "cost_per_customer", "channel", "Cost per customer",
                            "$,.0f", ""))
            with st.expander("Channel table"):
                _table(shown, {"acquisition_channel": ("Channel", "label"),
                               "leads": ("Leads", "int"), "customers": ("Customers", "int"),
                               "conversion_rate_60d": ("60-day conversion", "pct"),
                               "sourcing_spend": ("Sourcing spend", "usd"),
                               "cost_per_lead": ("Cost per lead", "usd"),
                               "cost_per_customer": ("Cost per customer", "usd")})
            _source(f, "channel_summary")

    st.subheader("Outreach capacity: who to call first")
    if _require(A, "acquisition"):
        a = A["acquisition"]
        budget = a.tables["budget_simulation"]
        caps = sorted(budget["capacity_share"].unique())
        cap = st.select_slider("Share of open leads the sales team can contact per run", caps,
                               value=caps[0], format_func=lambda c: f"{c:.0%}",
                               key="acq_capacity")
        rows = views.budget_at_capacity(budget, cap)
        champ = rows.loc[rows["policy"] == a.metrics["champion"]].iloc[0]
        st.markdown(f"At **{cap:.0%}** capacity, ranking by the {views.label(champ['policy'])} "
                    f"lead score reaches **{champ['conversions_reached_per_run']:.0f}** eventual "
                    f"buyers per run, **{champ['lift_vs_random']:.2f}×** random outreach and "
                    f"{champ['share_of_conversions_captured']:.0%} of all conversions.")
        rows = rows.assign(series=rows["policy"].map(views.label))
        order = [p for p in ENTITY_COLORS if p in set(rows["policy"])]
        _chart(alt.Chart(rows).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
            x=alt.X("conversions_reached_per_run:Q", title="Buyers reached per run"),
            y=alt.Y("series:N", title=None, sort="-x", axis=alt.Axis(labelLimit=240)),
            color=_color(order, ENTITY_COLORS).legend(None),
            tooltip=[alt.Tooltip("series:N", title="Policy"),
                     alt.Tooltip("conversions_reached_per_run:Q", title="Buyers per run",
                                 format=",.1f"),
                     alt.Tooltip("precision:Q", title="Precision", format=".1%"),
                     alt.Tooltip("lift_vs_random:Q", title="Lift vs random", format=".2f")]))
        with st.expander("Lead score deciles (holdout)"):
            _table(a.tables["decile_lift"], {
                "decile": ("Decile", "int"), "leads": ("Leads", "int"),
                "conversions": ("Conversions", "int"), "conversion_rate": ("Rate", "pct"),
                "lift": ("Lift", "x"), "cumulative_capture": ("Cumulative capture", "pct")})
        _source(a, "budget_simulation", "decile_lift")


def page_conversion(A, values) -> None:
    st.title("Conversion")
    st.caption("Where prospects drop out of the funnel, and what the checkout experiment "
               "showed. Section 03.")
    _metrics_row(values, ("lead_purchase_rate_30d", "checkout_lift", "checkout_order_value"))
    if not _require(A, "conversion"):
        return
    c = A["conversion"]
    left, right = st.columns(2)
    with left:
        st.subheader("Lead funnel (30 days)")
        loss = values["largest_loss_step"]
        st.caption(f"Largest session-funnel loss: **{loss.display}**." if loss.available else "")
        funnel = c.tables["funnel_prospect_cohort"]
        funnel = funnel.assign(stage_label=funnel["stage"].map(views.label))
        _chart(_bar(funnel, "share_of_start", "stage_label", "Share of leads reaching stage",
                    ".0%", "", sort=funnel["stage_label"].tolist()))
    with right:
        st.subheader("Session funnel by segment")
        dims = sorted(c.tables["funnel_segments"]["dimension"].unique())
        dim = st.selectbox("Segment by", dims, format_func=views.label, key="conv_dimension")
        seg = views.segment_funnel(c.tables["funnel_segments"], dim)
        seg = seg.assign(level_label=seg["level"].astype(str))
        _chart(_bar(seg, "session_to_purchase", "level_label", "Session → purchase", ".1%",
                    "").properties(height=CHART_HEIGHT - 70))
    with st.expander("Step conversion by segment"):
        _table(seg, {"level": ("Level", "label"), "sessions": ("Sessions", "int"),
                     **{col: (views.label(col.removeprefix("rate_")), "pct")
                        for col in views.STEP_COLUMNS},
                     "session_to_purchase": ("Session → purchase", "pct")})
    _source(c, "funnel_prospect_cohort", "funnel_segments")

    st.subheader("One-page checkout experiment")
    decision = c.metrics["experiment"]["decision"]
    box = st.success if decision["recommendation"] == "ship" else st.warning
    box(decision["recommendation_text"])
    results = c.tables["experiment_results"]
    _table(results, {"role": ("Role", "label"), "label": ("Metric", "text"),
                     "control": ("Control", "num"), "treatment": ("Treatment", "num"),
                     "diff": ("Difference", "num"), "ci_low": ("95% CI low", "num"),
                     "ci_high": ("95% CI high", "num"), "p_value": ("p-value", "num"),
                     "status": ("Guardrail", "text")})
    cum = c.tables["cumulative_effect"]
    st.markdown("Cumulative conversion lift by week with 95% CI (why the test ran to its "
                "planned end rather than stopping at the first significant week)")
    band = alt.Chart(cum).mark_area(color=S1, opacity=0.18).encode(
        x=alt.X("week:O", title="Week of experiment", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("ci_low:Q", title="Treatment − control", axis=alt.Axis(format="+.0%")),
        y2="ci_high:Q")
    line = alt.Chart(cum).mark_line(color=S1, strokeWidth=2, point=alt.OverlayMarkDef(size=40)
                                    ).encode(
        x="week:O", y="diff:Q",
        tooltip=[alt.Tooltip("data_through:N", title="Data through"),
                 alt.Tooltip("units:Q", title="Prospects", format=","),
                 alt.Tooltip("diff:Q", title="Lift", format="+.2%"),
                 alt.Tooltip("ci_low:Q", title="CI low", format="+.2%"),
                 alt.Tooltip("ci_high:Q", title="CI high", format="+.2%"),
                 alt.Tooltip("p_value:Q", title="p-value", format=".3f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=REFERENCE).encode(y="y:Q")
    _chart(band + zero + line)
    _source(c, "experiment_results", "cumulative_effect")


def page_retention(A, values) -> None:
    st.title("Retention")
    st.caption("Which active customers are likely to lapse, and how deep a retention offer "
               "should go. Section 02.")
    _metrics_row(values, ("churn_rate_90d", "churn_auc", "retention_net_value"))
    if not _require(A, "retention"):
        return
    r = A["retention"]
    sim = r.metrics["simulation"]
    curve = r.tables["retention_value_curve"]
    st.subheader("How deep should the retention offer go?")
    a = sim["assumptions"]
    st.caption(_md(f"Assumptions from section 02: save rate {a['save_rate']:.0%} of contacted "
               f"churners, contact cost {fmt_value(a['contact_cost'], 'usd')}, incentive "
               f"{fmt_value(a['incentive_cost'], 'usd')} (redeemed by "
               f"{a['nonchurner_redemption']:.0%} of non-churners). Net value = incremental "
               f"margin − program cost, per monthly run, on holdout customers."))
    policies = [p for p in ENTITY_COLORS if p in set(curve["policy"])]
    c1, c2 = st.columns([2, 3])
    options = views.depths(curve)
    depth = c1.select_slider("Share of active customers targeted", options,
                             value=min(options, key=lambda d: abs(d - 0.2)),
                             format_func=lambda d: f"{d:.0%}", key="ret_depth")
    chosen = c2.multiselect("Policies", policies,
                            default=["risk_ranked", "value_ranked", "random"],
                            format_func=views.label, key="ret_policies")
    if chosen:
        _chart(_policy_curve(curve, chosen, "net_value_per_run", depth, "Net value per run"))
        at = views.curve_at_depth(curve, depth, chosen)
        _table(at, {"policy": ("Policy", "label"),
                    "customers_targeted_per_run": ("Customers / run", "int"),
                    "expected_saves_per_run": ("Expected saves / run", "num"),
                    "program_cost_per_run": ("Cost / run", "usd"),
                    "net_value_per_run": ("Net value / run", "usd"),
                    "roi": ("ROI", "x"), "break_even_save_rate": ("Break-even save rate", "pct")})
    else:
        st.info("Select at least one policy.")
    _source(r, "retention_value_curve")

    st.subheader("Who churns")
    drivers = r.tables["segment_drivers"]
    seg = st.selectbox("Customer attribute", sorted(drivers["segment"].unique()),
                       format_func=views.label, key="ret_segment")
    rows = drivers.loc[drivers["segment"] == seg]
    rows = rows.assign(level_label=rows["level"].map(views.label))
    base = values["churn_rate_90d"].value
    bars = _bar(rows, "churn_rate", "level_label", "90-day churn rate", ".0%", "",
                sort=rows["level_label"].tolist())
    rule = alt.Chart(pd.DataFrame({"x": [base]})).mark_rule(color=REFERENCE,
                                                            strokeDash=[4, 4]).encode(x="x:Q")
    _chart((bars + rule).properties(height=max(120, 42 * len(rows))))
    st.caption(f"Dashed line: holdout churn rate across all active customers ({base:.1%}). "
               "Associations, not causes.")
    _source(r, "segment_drivers")


def page_revenue(A, values) -> None:
    st.title("Revenue growth")
    st.caption("Where future revenue is concentrated and which growth actions pay back. "
               "Sections 00 and 04.")
    _metrics_row(values, ("net_revenue", "average_order_value", "repeat_revenue_share",
                          "revenue_capture_top10", "growth_net_value"))
    if not _require(A, "revenue"):
        return
    rv = A["revenue"]
    left, right = st.columns([5, 6])
    with left:
        st.subheader("Value tiers")
        tiers = rv.tables["value_tiers"]
        long = tiers.melt(id_vars="group", value_vars=["share_of_customers", "share_of_actual"],
                          var_name="measure", value_name="share")
        names = {"share_of_customers": "Share of customers",
                 "share_of_actual": "Share of next-180-day revenue"}
        long["series"] = long["measure"].map(names)
        _chart(alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("share:Q", title=None, axis=alt.Axis(format=".0%")),
            y=alt.Y("group:N", title=None, sort=tiers["group"].tolist()),
            yOffset=alt.YOffset("series:N", sort=list(names.values())),
            color=alt.Color("series:N", title=None, sort=list(names.values()),
                            scale=alt.Scale(domain=list(names.values()), range=[REFERENCE, S1]),
                            legend=alt.Legend(orient="top", labelLimit=260)),
            tooltip=[alt.Tooltip("group:N", title="Tier"), alt.Tooltip("series:N", title=""),
                     alt.Tooltip("share:Q", title="Share", format=".1%")]))
    with right:
        st.subheader("Next best action")
        _table(rv.tables["next_best_action"],
               {"group": ("Action", "label"), "customers": ("Customers", "int"),
                "share_of_predicted": ("Predicted revenue", "pct"),
                "mean_predicted": ("Per customer", "usd"),
                "buyer_rate": ("Bought", "pct")})
        st.caption("Predicted revenue: share of the next 180 days' predicted revenue; bought: "
                   "share who ordered in the following 180 days (holdout run).")
    _source(rv, "value_tiers", "next_best_action")

    st.subheader("Growth program depth")
    sc = rv.metrics["scenario"]
    a = sc["assumptions"]
    st.caption(_md(f"Assumptions from section 04: {a['uplift']:.0%} revenue uplift among targeted "
               f"customers, contact cost {fmt_value(a['contact_cost'], 'usd')}, perk "
               f"{fmt_value(a['perk_cost'], 'usd')} redeemed by {a['redemption_rate']:.0%}, "
               f"margin rate {sc['observed_margin_rate']:.1%}. Net value realized on the "
               f"holdout run."))
    curve = rv.tables["growth_value_curve"]
    policies = [p for p in ENTITY_COLORS if p in set(curve["policy"])]
    c1, c2 = st.columns([2, 3])
    options = views.depths(curve)
    depth = c1.select_slider("Share of customers targeted", options,
                             value=min(options, key=lambda d: abs(d - 0.1)),
                             format_func=lambda d: f"{d:.0%}", key="rev_depth")
    default = [rv.metrics["champion"], "run_rate", "random"]
    chosen = c2.multiselect("Ranking", policies, default=[p for p in default if p in policies],
                            format_func=views.label, key="rev_policies")
    if chosen:
        _chart(_policy_curve(curve, chosen, "net_value", depth, "Net value"))
        _table(views.curve_at_depth(curve, depth, chosen),
               {"policy": ("Ranking", "label"), "customers_targeted": ("Customers", "int"),
                "incremental_revenue": ("Incremental revenue", "usd"),
                "program_cost": ("Program cost", "usd"), "net_value": ("Net value", "usd"),
                "roi": ("ROI", "x"), "break_even_uplift": ("Break-even uplift", "pct")})
    else:
        st.info("Select at least one ranking.")
    _source(rv, "growth_value_curve")


def page_forecast(A, values) -> None:
    st.title("Revenue forecast")
    st.caption("What revenue to plan for over the next quarter, and how much to trust it. "
               "Section 05.")
    _metrics_row(values, ("forecast_13w", "forecast_wape"))
    if not _require(A, "forecast"):
        return
    fc = A["forecast"]
    c1, c2, c3 = st.columns([1, 2, 2])
    level = c1.segmented_control("Interval", [80, 50], default=80, required=True,
                                 format_func=lambda v: f"{v}%", key="fc_level")
    weeks = c2.slider("History shown (weeks)", 13, len(fc.tables["weekly_revenue"]), 52,
                      step=13, key="fc_history")
    baselines = c3.multiselect("Compare with baselines", ["naive_4wk", "seasonal_naive_yoy"],
                               default=[], format_func=views.label, key="fc_baselines")
    champion = fc.metrics["config"]["champion"]
    history, ahead = views.forecast_view(fc.tables["weekly_revenue"], fc.tables["forecast"],
                                         champion, int(level), weeks)
    series = {"actual": "Actual", champion: f"Forecast ({views.label(champion)})",
              **{b: views.label(b) for b in baselines}}
    long = pd.concat([
        history.assign(series=series["actual"], value=history["net_revenue"]),
        ahead.assign(series=series[champion], value=ahead[champion]),
        *[ahead.assign(series=series[b], value=ahead[b]) for b in baselines],
    ])[["week_start", "series", "value"]]
    colors = {"Actual": S1, series[champion]: S2, views.label("naive_4wk"): S4,
              views.label("seasonal_naive_yoy"): S3}
    domain = list(series.values())
    band = alt.Chart(ahead).mark_area(color=S2, opacity=0.2).encode(
        x="week_start:T", y="lower:Q", y2="upper:Q",
        tooltip=[alt.Tooltip("week_start:T", title="Week"),
                 alt.Tooltip("lower:Q", title=f"{level}% lower", format="$,.0f"),
                 alt.Tooltip("upper:Q", title=f"{level}% upper", format="$,.0f")])
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("week_start:T", title=None),
        y=alt.Y("value:Q", title="Weekly net revenue", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=domain, range=[colors[d] for d in domain]),
                        legend=alt.Legend(orient="top", labelLimit=260)),
        strokeDash=alt.condition(alt.FieldOneOfPredicate("series", domain[2:] or ["-"]),
                                 alt.value([5, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("week_start:T", title="Week"), alt.Tooltip("series:N", title=""),
                 alt.Tooltip("value:Q", title="Net revenue", format="$,.0f")])
    _chart((band + lines).properties(height=CHART_HEIGHT + 40))
    total = ahead[champion].sum()
    st.markdown(_md(f"Next {len(ahead)} weeks: **{fmt_value(total, 'usd')}**, {level}% interval "
                f"{fmt_value(fc.metrics['forecast']['total'][f'lower_{level}'], 'usd')} to "
                f"{fmt_value(fc.metrics['forecast']['total'][f'upper_{level}'], 'usd')} "
                f"(intervals for the 13-week total come from past total errors, so they are not "
                f"the sum of the weekly bands)."))
    with st.expander("Backtest accuracy by lead time (WAPE, lower is better)"):
        acc = fc.tables["accuracy_by_horizon"]
        wide = acc.pivot_table(index="model", columns="bucket", values="wape",
                               aggfunc="first").reset_index()
        order = {m: i for i, m in enumerate([champion, "harmonic_no_promo",
                                             "seasonal_naive_yoy", "naive_4wk"])}
        wide = wide.sort_values("model", key=lambda s: s.map(order))
        _table(wide, {"model": ("Model", "label"),
                      **{b: (b.capitalize(), "pct") for b in wide.columns if b != "model"}})
    _source(fc, "weekly_revenue", "forecast", "accuracy_by_horizon")


def page_lifecycle(A, values) -> None:
    st.title("Customer lifecycle")
    st.caption("How customers move between lifecycle states and where value leaks. Section 06.")
    _metrics_row(values, ("second_purchase_rate", "at_risk_recovery", "loyal_share",
                          "churned_share"))
    if not _require(A, "lifecycle"):
        return
    lc = A["lifecycle"]
    st.subheader("Customers by state at each month end")
    as_share = st.toggle("Show as share of customers", value=True, key="lc_share")
    mix = views.state_mix(lc.tables["state_counts_by_month"], as_share)
    mix["series"] = mix["state"].map(views.label)
    mix["month"] = mix["period"].astype(str).str[:7]
    fmt = ".0%" if as_share else ","
    _chart(alt.Chart(mix).mark_bar().encode(
        x=alt.X("month:O", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("value:Q", title="Share of customers" if as_share else "Customers",
                axis=alt.Axis(format=fmt), stack="zero"),
        color=_color(list(views.CUSTOMER_STATES), STATE_COLORS),
        order=alt.Order("order:Q", sort="ascending"),
        stroke=alt.value("white"), strokeWidth=alt.value(1),
        tooltip=[alt.Tooltip("month:O", title="Month"),
                 alt.Tooltip("series:N", title="State"),
                 alt.Tooltip("count:Q", title="Customers", format=","),
                 alt.Tooltip("share:Q", title="Share", format=".1%")]))
    _source(lc, "state_counts_by_month")

    st.subheader("Decision points")
    _table(lc.tables["decision_points"],
           {"label": ("Decision point", "text"), "resolved": ("Resolved", "int"),
            "favourable_rate": ("Good-path rate", "pct"), "rate_ci_low": ("95% CI low", "pct"),
            "rate_ci_high": ("95% CI high", "pct"),
            "revenue_gap_per_year": ("Revenue gap / year (upper bound)", "usd")})
    st.caption("Revenue gap: extra revenue of good-path customers × customers taking the bad "
               "path each year. Descriptive, not the effect of a program.")
    _source(lc, "decision_points")

    st.subheader("Cohort retention by acquisition channel")
    by_channel = lc.tables["cohort_retention_by_channel"]
    all_channels = sorted(by_channel["acquisition_channel"].unique())
    channel_colors = dict(zip(all_channels, (S1, S2, S3, S4, S5, S6, S7), strict=False))
    c1, c2 = st.columns([3, 2])
    chosen = c1.multiselect("Channels (up to 4)", all_channels,
                            default=sorted(views.largest_channels(by_channel, 3)),
                            max_selections=4, format_func=views.label, key="lc_channels")
    metric = c2.radio("Measure", ["retention", "cumulative_revenue_per_customer"],
                      format_func={"retention": "Share buying in month",
                                   "cumulative_revenue_per_customer":
                                       "Cumulative revenue per customer"}.get,
                      horizontal=True, key="lc_metric")
    if chosen:
        df = views.filter_channels(by_channel, chosen)
        if metric == "retention":  # month 0 is 100% by definition; start at month 1
            df = df.loc[df["months_since_acquisition"] >= 1]
        df["series"] = df["acquisition_channel"].map(views.label)
        fmt, tip = (".0%", ".1%") if metric == "retention" else ("$,.0f", "$,.2f")
        _chart(alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36)).encode(
            x=alt.X("months_since_acquisition:Q", title="Months since first order",
                    axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y(f"{metric}:Q", title=None, axis=alt.Axis(format=fmt)),
            color=_color([c for c in all_channels if c in chosen], channel_colors),
            tooltip=[alt.Tooltip("series:N", title="Channel"),
                     alt.Tooltip("months_since_acquisition:Q", title="Month"),
                     alt.Tooltip(f"{metric}:Q", title="Value", format=tip),
                     alt.Tooltip("customers:Q", title="Cohort customers", format=",")]))
    else:
        st.info("Select at least one channel.")
    _source(lc, "cohort_retention_by_channel")


def page_definitions(A, values) -> None:
    st.title("Definitions & sources")
    st.caption("Every KPI on this dashboard, its definition, and the saved output field it is "
               "read from. Written to `projects/07_dashboard/outputs/kpi_catalog.csv` by "
               "`northstar dashboard-kpis`.")
    catalog = kpi_catalog(A)
    themes = st.multiselect("Themes", list(THEMES), default=list(THEMES), key="defs_themes")
    shown = catalog.loc[catalog["theme"].isin(themes)]
    table = pd.DataFrame({
        "Theme": shown["theme"], "KPI": shown["kpi"], "Value": shown["display"],
        "Interval / comparison": ["; ".join(x for x in pair if x)
                                  for pair in zip(shown["interval"], shown["comparison"],
                                                  strict=True)],
        "Definition": shown["definition"],
        "Source": shown["source"] + " → " + shown["field"],
    })
    st.table(table.set_index("Theme").map(_md))
    st.subheader("Section outputs")
    prov = provenance(A)
    st.dataframe(prov.drop(columns="key").rename(columns={
        "section": "Section", "available": "Available", "seed": "Seed",
        "n_prospects": "Prospects", "consistent": "Same data as section 00",
        "missing_files": "Missing files", "regenerate_with": "Regenerate with",
        "outputs": "Outputs folder"}), hide_index=True, width="stretch")
    st.markdown("Refresh everything with `make results` (or `northstar generate-data` followed "
                "by each section command), then `northstar dashboard-kpis`. The dashboard "
                "picks up regenerated files on the next interaction.")


PAGE_FUNCS = {
    "Executive overview": page_overview, "Acquisition": page_acquisition,
    "Conversion": page_conversion, "Retention": page_retention,
    "Revenue growth": page_revenue, "Forecast": page_forecast, "Lifecycle": page_lifecycle,
    "Definitions & sources": page_definitions,
}


def main() -> None:
    st.set_page_config(page_title="Northstar executive dashboard",
                       page_icon=":material/insights:", layout="wide")
    root = projects_dir()
    artifacts = _load(str(root), fingerprint(root))
    values = evaluate_all(artifacts)
    with st.sidebar:
        st.markdown("### Northstar Consumer")
        st.caption("Executive sales & marketing dashboard (synthetic data)")
        page = st.radio("Page", PAGES, key="page", label_visibility="collapsed")
        prov = provenance(artifacts)
        available = prov.loc[prov["available"]]
        if prov["available"].all() and prov["consistent"].all():
            seed, n = available.iloc[0][["seed", "n_prospects"]]
            st.caption(f"All {len(prov)} section outputs present, built from data seed "
                       f"{int(seed)} ({int(n):,} prospects).")
        else:
            st.warning("Some section outputs are missing or were built from different data. "
                       "See *Definitions & sources*.", icon=":material/warning:")
        st.caption("Filters slice saved outputs; no model is re-fit in the dashboard.")
    PAGE_FUNCS[page](artifacts, values)


main()
