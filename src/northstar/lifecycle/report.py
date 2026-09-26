"""Section 06 pipeline: assign lifecycle states, measure transitions and cohort retention, segment
customers by RFM, audit the calculations, and write tables, figures and the README results block.

Everything reported in ``projects/06_lifecycle/README.md`` between the generated-block markers is
rendered from ``outputs/metrics.json`` by :func:`render_markdown`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from northstar.lifecycle import cohorts as co
from northstar.lifecycle import segments as sg
from northstar.lifecycle import transitions as tr
from northstar.lifecycle.states import (
    CUSTOMER_STATES,
    NOT_YET,
    PRECEDENCE,
    STATE_LABELS,
    STATES,
    LifecyclePanel,
    LifecycleRules,
    assign_states,
    build_panel,
)
from northstar.profile import (
    GRID,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _style,
    update_generated_block,
)
from northstar.timeline import DEFAULT_CUTOFF, snapshot

__all__ = ["IntegrityError", "LifecycleConfig", "render_markdown", "run_analysis",
           "write_outputs"]

BEGIN_MARKER = "<!-- BEGIN GENERATED: lifecycle-results -->"
END_MARKER = "<!-- END GENERATED: lifecycle-results -->"
COHORT_MONTHS_SHOWN = (1, 2, 3, 6, 9, 12)
CHANNEL_MONTHS = (3, 6, 12)

# Reference categorical palette in fixed slot order (one slot per customer state).
STATE_COLORS = {"new": "#2a78d6", "active": "#eb6834", "loyal": "#1baf7a",
                "at_risk": "#eda100", "churned": "#e87ba4"}
SERIES = "#2a78d6"
SERIES_2 = "#eb6834"
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
BAND = "#f0efec"


class IntegrityError(RuntimeError):
    """Raised when a lifecycle calculation fails its consistency audit."""


@dataclass(frozen=True)
class LifecycleConfig:
    rules: LifecycleRules = field(default_factory=LifecycleRules)
    report_months: int = 12
    """Transitions are pooled over the most recent ``report_months`` month-to-month steps."""
    follow_up_months: int = 6
    """Revenue after a decision point is summed over this many months."""
    cohort_follow_up_months: int = 12
    """Pooled cohort curves use every cohort observed for at least this many months."""
    segment_as_of: str = str(DEFAULT_CUTOFF.date())
    outcome_days: int = 90

    def as_dict(self) -> dict:
        return {"rules": self.rules.as_dict(), "report_months": self.report_months,
                "follow_up_months": self.follow_up_months,
                "cohort_follow_up_months": self.cohort_follow_up_months,
                "segment_as_of": self.segment_as_of, "outcome_days": self.outcome_days}


# ---------------------------------------------------------------- helpers
def _clean(obj):
    """JSON-safe copy with floats rounded (stable diffs; NaN -> None)."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer | bool | np.bool_):
        return obj.item() if isinstance(obj, np.generic) else obj
    if isinstance(obj, float | np.floating):
        return None if np.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, pd.Timestamp | np.datetime64):
        return str(pd.Timestamp(obj).date())
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


def _month(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def _period_index(panel: LifecyclePanel, as_of: pd.Timestamp) -> int:
    """Panel column whose month-end snapshot is ``as_of``."""
    matches = np.flatnonzero(panel.as_of == pd.Timestamp(as_of))
    if len(matches) != 1:
        raise ValueError(f"{as_of} is not a monthly as-of date of the panel")
    return int(matches[0])


# ---------------------------------------------------------------- integrity audit
def integrity_audit(tables: Mapping[str, pd.DataFrame], panel: LifecyclePanel,
                    cohort_table: pd.DataFrame, pooled: pd.DataFrame, profile: pd.DataFrame,
                    config: LifecycleConfig) -> dict:
    """Runtime checks that states, transitions, cohorts and segments are internally consistent."""
    people = panel.people
    created = people["created_at"].to_numpy()[:, None] < panel.as_of.to_numpy()[None, :]
    present = panel.state != NOT_YET
    valid_codes = bool(np.all((panel.state[present] >= 0)
                              & (panel.state[present] < len(STATES))))

    # Every person present in month t is present in t+1, and transitions out of each state sum to
    # the people in that state (checked month by month).
    monotone = bool(np.all(present[:, :-1] <= present[:, 1:]))
    counts = panel.state_counts()
    row_mismatch = 0
    for t in range(panel.n_periods - 1):
        rows = tr.transition_counts(panel, [t]).sum(axis=1)
        row_mismatch += int((rows != counts.iloc[t]).sum())

    impossible = tr.impossible_transitions(panel.rules)
    all_counts = tr.transition_counts(panel)
    observed_impossible = {f"{a}->{b}": int(all_counts.loc[a, b]) for a, b in sorted(impossible)
                           if all_counts.loc[a, b] > 0}

    # States at the segmentation date are identical when every later order is removed, and the
    # segment profile uses exactly the panel's states.
    as_of = pd.Timestamp(config.segment_as_of)
    t_probe = _period_index(panel, as_of)
    truncated = assign_states(people, tables["orders"].loc[tables["orders"]["order_ts"] < as_of],
                              as_of, panel.rules)["state"].to_numpy()
    point_in_time = pd.Series(truncated).equals(pd.Series(panel.labels(t_probe)))
    panel_states = pd.Series(panel.labels(t_probe), index=people["customer_id"])
    profile_match = bool((profile["state"].to_numpy()
                          == panel_states.loc[profile["customer_id"]].to_numpy()).all())
    in_panel_customers = int(pd.Series(panel.labels(t_probe)).isin(CUSTOMER_STATES).sum())

    # Segment scores do not change when every row after the as-of date is removed.
    view = snapshot(tables, as_of)
    score_cols = ["customer_id", "state", "r_score", "f_score", "m_score", "rfm_segment"]
    rescored = sg.customer_profile(view, as_of, panel.rules, config.outcome_days)
    scores_invariant = rescored[score_cols].equals(profile[score_cols])

    # Cohort denominators: one size per cohort, equal to the customers acquired that month.
    acquired = tables["customers"]["customer_since"].dt.to_period("M").dt.to_timestamp()
    expected = acquired.value_counts().to_dict()
    sizes = cohort_table.groupby("cohort")["cohort_size"].agg(["nunique", "first"])
    denominators_ok = bool((sizes["nunique"] == 1).all()
                           and sizes["first"].to_dict() == expected)
    censor_ok = bool((cohort_table["retention"].isna() == ~cohort_table["observed"]).all())
    pooled_fixed = bool((pooled[["customers", "cohorts"]] == pooled[["customers", "cohorts"]]
                         .iloc[0]).all().all())

    checks = {
        "every_person_has_one_valid_state_once_created":
            valid_codes and bool(np.array_equal(present, created)),
        "people_never_leave_the_panel": monotone,
        "transitions_out_of_each_state_sum_to_its_population": row_mismatch == 0,
        "no_transition_the_state_rules_forbid": not observed_impossible,
        "states_unchanged_when_later_orders_removed": point_in_time,
        "segment_profile_uses_panel_states": profile_match and len(profile) == in_panel_customers,
        "segment_scores_unchanged_when_later_rows_removed": bool(scores_invariant),
        "cohort_denominator_fixed_at_acquisition_size": denominators_ok,
        "unobserved_cohort_months_missing_not_zero": censor_ok,
        "pooled_curve_uses_one_fixed_cohort_set": pooled_fixed,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "person_months": int(present.sum()),
            "transition_row_mismatches": row_mismatch,
            "impossible_transitions_checked": len(impossible),
            "impossible_transitions_observed": observed_impossible,
            "point_in_time_probe": str(as_of.date()),
            "cohorts": len(sizes),
        },
    }


# ---------------------------------------------------------------- analysis
def _matrix_dict(df: pd.DataFrame) -> dict:
    return {a: {b: df.loc[a, b] for b in df.columns} for a in df.index}


def run_analysis(tables: Mapping[str, pd.DataFrame], config: LifecycleConfig | None = None
                 ) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Run the full section 06 analysis; returns (metrics dict, output tables)."""
    config = config or LifecycleConfig()
    panel = build_panel(tables, config.rules)
    window = tr.recent_origins(panel, config.report_months)

    # ---- states over time
    counts_by_month = panel.state_counts()
    last, year_ago = panel.n_periods - 1, panel.n_periods - 13

    # ---- transitions
    counts = tr.transition_counts(panel, window)
    matrix = tr.transition_matrix(counts)
    by_month = tr.transitions_by_period(panel)
    entry = tr.entries(panel)
    entry_window = entry.loc[entry["period"].isin(panel.periods[[t + 1 for t in window]])]
    entry_share = entry_window.groupby("state")["people"].sum()
    value = tr.state_value(panel, window)
    decisions = tr.decision_point_table(panel, window, config.follow_up_months)
    by_channel = tr.decision_points_by_group(panel, window, "acquisition_channel")
    recency = tr.repurchase_by_recency(panel, window)
    lead_age = tr.conversion_by_lead_age(panel, window)

    # ---- cohorts
    cohort_table = co.cohort_retention(tables["customers"], tables["orders"])
    pooled = co.pooled_retention(cohort_table, config.cohort_follow_up_months)
    channel_table = co.cohort_retention(tables["customers"], tables["orders"],
                                        by=["acquisition_channel"])
    pooled_channel = co.pooled_retention(channel_table, config.cohort_follow_up_months,
                                         by=["acquisition_channel"])

    # ---- segments
    profile = sg.customer_profile(tables, config.segment_as_of, config.rules,
                                  config.outcome_days)
    rfm = sg.segment_summary(profile, "rfm_segment", sg.RFM_SEGMENTS)
    by_state = sg.segment_summary(profile, "state", CUSTOMER_STATES)
    cross = pd.crosstab(profile["state"], profile["rfm_segment"]).reindex(
        index=list(CUSTOMER_STATES), columns=list(sg.RFM_SEGMENTS), fill_value=0)

    audit = integrity_audit(tables, panel, cohort_table, pooled, profile, config)
    if not audit["passed"]:
        failed = [k for k, ok in audit["checks"].items() if not ok]
        raise IntegrityError(f"Lifecycle integrity audit failed: {failed}; "
                             f"details: {audit['details']}")

    def state_snapshot(t: int) -> dict:
        row = counts_by_month.iloc[t]
        customers = row[list(CUSTOMER_STATES)].sum()
        return {"period": _month(panel.periods[t]), "as_of": panel.as_of[t],
                "people": int(row.sum()), "customers": int(customers),
                "counts": {s: int(row[s]) for s in STATES},
                "customer_shares": {s: row[s] / customers for s in CUSTOMER_STATES}}

    as_of = pd.Timestamp(config.segment_as_of)
    ch_rows = pooled_channel.loc[pooled_channel["months_since_acquisition"].isin(CHANNEL_MONTHS)]
    metrics = {
        "config": {**config.as_dict(), "states": list(STATES), "precedence": list(PRECEDENCE),
                   "definitions": config.rules.definitions(),
                   "impossible_transitions": [f"{a}->{b}" for a, b in sorted(
                       tr.impossible_transitions(config.rules))],
                   "decision_points": [p.as_dict() for p in tr.DECISION_POINTS],
                   "rfm_rules": sg.RFM_RULES},
        "population": {"people": len(panel.people),
                       "customers": int(panel.people["customer_id"].notna().sum()),
                       "orders": len(tables["orders"]),
                       "first_period": _month(panel.periods[0]),
                       "last_period": _month(panel.periods[-1]), "periods": panel.n_periods},
        "states": {"current": state_snapshot(last), "year_ago": state_snapshot(year_ago)},
        "transitions": {
            "first_origin": _month(panel.periods[window[0]]),
            "last_destination": _month(panel.periods[window[-1] + 1]),
            "months": len(window),
            "counts": _matrix_dict(counts), "probabilities": _matrix_dict(matrix),
            "new_leads": int(entry_share.sum()),
            "new_leads_entering_as_new": int(entry_share.get("new", 0)),
        },
        "state_value": _records(value),
        "decision_points": _records(decisions),
        "decision_points_by_channel": _records(by_channel),
        "repurchase_by_recency": _records(recency),
        "conversion_by_lead_age": _records(lead_age),
        "cohorts": {
            "count": int(cohort_table["cohort"].nunique()),
            "first": _month(cohort_table["cohort"].min()),
            "last": _month(cohort_table["cohort"].max()),
            "pooled": _records(pooled.drop(columns=["first_cohort", "last_cohort"])),
            "pooled_first_cohort": _month(pooled["first_cohort"].iloc[0]),
            "pooled_last_cohort": _month(pooled["last_cohort"].iloc[0]),
            "by_channel": _records(ch_rows[["acquisition_channel", "months_since_acquisition",
                                            "customers", "retention",
                                            "cumulative_revenue_per_customer"]]),
        },
        "segments": {
            "as_of": as_of, "outcome_end": as_of + pd.Timedelta(days=config.outcome_days - 1),
            "customers": len(profile),
            "rfm": _records(rfm), "states": _records(by_state),
            "crosstab": _matrix_dict(cross),
        },
        "audit": audit,
    }
    counts_out = counts_by_month.reset_index()
    counts_out["customers"] = counts_by_month[list(CUSTOMER_STATES)].sum(axis=1).to_numpy()
    out = {
        "state_counts_by_month": counts_out,
        "transition_counts": counts.reset_index(),
        "transition_matrix": matrix.reset_index(),
        "transitions_by_month": by_month,
        "entries_by_month": entry,
        "state_value": value,
        "decision_points": decisions,
        "decision_points_by_channel": by_channel,
        "repurchase_by_recency": recency,
        "conversion_by_lead_age": lead_age,
        "cohort_retention": cohort_table,
        "cohort_retention_pooled": pooled,
        "cohort_retention_by_channel": pooled_channel,
        "rfm_segments": rfm,
        "lifecycle_state_summary": by_state,
        "rfm_by_lifecycle": cross.rename_axis(index="state", columns=None).reset_index(),
    }
    return _clean(metrics), out


# ---------------------------------------------------------------- figures
def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def _axes(ax: plt.Axes, x: str, y: str, grid: str = "y") -> None:
    ax.set_xlabel(x, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(y, color=TEXT_SECONDARY, fontsize=9)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(ax: plt.Axes, **kw) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, **kw)


def _heatmap(ax: plt.Axes, values: np.ndarray, fmt, vmax: float | None = None,
             blank_zeros: bool = False) -> None:
    """Sequential one-hue heatmap; NaN (and optionally zero) cells stay on the surface with a
    ``·`` marker; ink flips to white on dark cells."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL).with_extremes(
        bad=SURFACE)
    shown = np.where(values == 0, np.nan, values) if blank_zeros else values
    vmax = np.nanmax(shown) if vmax is None else vmax
    ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(shown):
        if np.isfinite(v):
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * vmax else TEXT_PRIMARY)
        elif blank_zeros:
            ax.text(j, i, "·", ha="center", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.tick_params(length=0, colors=TEXT_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figures(metrics: Mapping, out: Mapping[str, pd.DataFrame], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    rules = metrics["config"]["rules"]

    # 1. Customer state mix over time (stacked in fixed slot order).
    counts = out["state_counts_by_month"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=120)
    ax.stackplot(counts["period"], *[counts[s] for s in CUSTOMER_STATES],
                 colors=[STATE_COLORS[s] for s in CUSTOMER_STATES],
                 labels=[STATE_LABELS[s] for s in CUSTOMER_STATES], edgecolor=SURFACE,
                 linewidth=1.5)
    _axes(ax, "Month end", "Customers")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY,
              loc="upper left")
    _style(ax, "Customers by lifecycle state at each month end",
           "Every customer is in exactly one state; prospects (not yet customers) not shown")
    paths.append(_finish(fig, fig_dir / "state_mix.png"))

    # 2. Transition matrix over the reporting window.
    t = metrics["transitions"]
    probs = np.array([[t["probabilities"][a][b] for b in STATES] for a in STATES], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=120)
    _heatmap(ax, probs * 100, lambda v: f"{v:.1f}%", vmax=100, blank_zeros=True)
    ax.set_xticks(range(len(STATES)), [STATE_LABELS[s] for s in STATES])
    ax.set_yticks(range(len(STATES)), [STATE_LABELS[s] for s in STATES])
    ax.set_xlabel("State at the next month end", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("State at month end", color=TEXT_SECONDARY, fontsize=9)
    _style(ax, "Month-to-month lifecycle transitions",
           f"Share of each row's people, {t['first_origin']} to {t['last_destination']}; "
           "· = none")
    paths.append(_finish(fig, fig_dir / "transition_matrix.png"))

    # 3. Cohort retention triangle and the pooled curve by acquisition channel.
    tri = out["cohort_retention"]
    tri = tri.loc[tri["months_since_acquisition"] >= 1]
    grid = tri.pivot_table(index="cohort", columns="months_since_acquisition",
                           values="retention", dropna=False)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=120)
    _heatmap(ax, grid.to_numpy() * 100, lambda v: f"{v:.0f}", vmax=np.nanmax(grid.to_numpy())
             * 100)
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), [_month(c) for c in grid.index])
    ax.set_xlabel("Months since first order", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Acquisition cohort (first-order month)", color=TEXT_SECONDARY, fontsize=9)
    _style(ax, "Cohort retention: % of each cohort ordering in month k",
           "Denominator = cohort size at acquisition; blank = not yet observed")
    paths.append(_finish(fig, fig_dir / "cohort_retention.png"))

    pooled, ch = out["cohort_retention_pooled"], out["cohort_retention_by_channel"]
    months = metrics["config"]["cohort_follow_up_months"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=120,
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    p1 = pooled.loc[pooled["months_since_acquisition"] >= 1]
    ax.plot(p1["months_since_acquisition"], p1["retention"] * 100, color=SERIES, linewidth=2,
            marker="o", markersize=5)
    ax.set_ylim(0, max(40, p1["retention"].max() * 110))
    ax.set_xticks(range(1, months + 1))
    _axes(ax, "Months since first order", "Cohort ordering in the month (%)")
    _style(ax, "Pooled retention curve",
           f"{int(pooled['cohorts'].iloc[0])} cohorts, {int(pooled['customers'].iloc[0]):,} "
           f"customers, fixed for every month")
    ax = axes[1]
    wide = ch.loc[ch["months_since_acquisition"].isin([3, months])].pivot_table(
        index="acquisition_channel", columns="months_since_acquisition", values="retention")
    wide = wide.sort_values(months)
    y = np.arange(len(wide))
    for offset, (k, color) in zip((-0.2, 0.2), ((3, SERIES), (months, SERIES_2)), strict=True):
        ax.barh(y + offset, wide[k] * 100, height=0.36, color=color, label=f"Month {k}",
                edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, wide.index.str.replace("_", " "))
    _axes(ax, "Cohort ordering in the month (%)", "", grid="x")
    _legend(ax, loc="lower right")
    _style(ax, "By acquisition channel", "Same fixed cohorts")
    paths.append(_finish(fig, fig_dir / "retention_curves.png"))

    # 4. Repurchase probability by days since last order, with the state bands.
    rec = out["repurchase_by_recency"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=120)
    x = np.arange(len(rec))
    edges = [int(str(b).split("-")[0].rstrip("+")) for b in rec["days_since_last_order"]]
    bands = ((0, rules["at_risk_days"], "Engaged (new / active / loyal)"),
             (rules["at_risk_days"], rules["churn_days"], "At risk"),
             (rules["churn_days"], np.inf, "Churned"))
    for lo, hi, label in bands:
        idx = [i for i, e in enumerate(edges) if lo <= e < hi]
        if idx and label == "At risk":
            ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=BAND, zorder=0)
        if idx:
            ax.text((idx[0] + idx[-1]) / 2, 1.0, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=TEXT_SECONDARY)
    ax.bar(x, rec["repurchase_rate"] * 100, color=SERIES, width=0.7, edgecolor=SURFACE)
    ax.set_ylim(0, rec["repurchase_rate"].max() * 100 * 1.15)
    ax.set_xticks(x, rec["days_since_last_order"], rotation=30)
    _axes(ax, "Days since last order at month end", "Ordered during the next month (%)")
    _style(ax, "The chance of another order falls steeply with idle time",
           f"Customer-months {metrics['transitions']['first_origin']} to "
           f"{metrics['transitions']['last_destination']}")
    paths.append(_finish(fig, fig_dir / "repurchase_by_recency.png"))

    # 5. Decision points: favourable rate and the annual revenue gap.
    dp = out["decision_points"].iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=120, sharey=True)
    y = np.arange(len(dp))
    ax = axes[0]
    ax.barh(y, dp["favourable_rate"] * 100, color=SERIES, height=0.6)
    ax.errorbar(dp["favourable_rate"] * 100, y,
                xerr=[(dp["favourable_rate"] - dp["rate_ci_low"]) * 100,
                      (dp["rate_ci_high"] - dp["favourable_rate"]) * 100],
                fmt="none", ecolor=TEXT_PRIMARY, elinewidth=1, capsize=2)
    ax.set_yticks(y, dp["label"])
    ax.set_xlim(0, 100)
    _axes(ax, "Favourable outcome (%)", "", grid="x")
    _style(ax, "How often the good path is taken", "Resolved transitions, 95% Wilson interval")
    ax = axes[1]
    ax.barh(y, dp["revenue_gap_per_year"] / 1000, color=SERIES, height=0.6)
    for yi, v in zip(y, dp["revenue_gap_per_year"], strict=True):
        ax.text(v / 1000, yi, f"  ${v / 1000:,.0f}K", va="center", fontsize=8,
                color=TEXT_PRIMARY)
    _axes(ax, "Annual revenue gap ($K)", "", grid="x")
    ax.set_xlim(0, dp["revenue_gap_per_year"].max() / 1000 * 1.25)
    _style(ax, "Revenue gap behind the bad path",
           f"Unfavourable per year × {metrics['config']['follow_up_months']}-month gap; "
           "not causal")
    paths.append(_finish(fig, fig_dir / "decision_points.png"))

    # 6. RFM segments against lifecycle states; value concentration.
    cross = out["rfm_by_lifecycle"].set_index("state")
    rfm = out["rfm_segments"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=120,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    _heatmap(ax, cross.to_numpy(dtype=float), lambda v: f"{v:,.0f}", blank_zeros=True)
    ax.set_xticks(range(cross.shape[1]), [sg.RFM_LABELS[s] for s in cross.columns],
                  rotation=30, ha="right")
    ax.set_yticks(range(cross.shape[0]), [STATE_LABELS[s] for s in cross.index])
    _style(ax, "Customers by lifecycle state and RFM segment",
           f"As of {metrics['segments']['as_of']}")
    ax = axes[1]
    y = np.arange(len(rfm))[::-1]
    ax.barh(y + 0.2, rfm["share_of_customers"] * 100, height=0.38, color=SERIES,
            label="Share of customers", edgecolor=SURFACE)
    ax.barh(y - 0.2, rfm["share_of_next_revenue"] * 100, height=0.38, color=SERIES_2,
            label=f"Share of next {metrics['config']['outcome_days']}-day revenue",
            edgecolor=SURFACE)
    ax.set_yticks(y, [sg.RFM_LABELS[s] for s in rfm["rfm_segment"]])
    _axes(ax, "%", "", grid="x")
    _legend(ax, loc="lower right")
    _style(ax, "Size vs. subsequent revenue", "Revenue measured after segmentation")
    paths.append(_finish(fig, fig_dir / "rfm_lifecycle.png"))
    return paths


# ---------------------------------------------------------------- README rendering
def _pct(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def _usd(v: float | None) -> str:
    if v is None:
        return "-"
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def _n(v: float | None) -> str:
    return "-" if v is None else f"{v:,.0f}"


def render_markdown(metrics: Mapping) -> str:
    cfg = metrics["config"]
    pop = metrics["population"]
    data = metrics.get("data")
    source = f"Data seed `{data['seed']}`, {data['n_prospects']:,} prospects. " if data else ""
    t = metrics["transitions"]
    cur, prev = metrics["states"]["current"], metrics["states"]["year_ago"]
    lines = [
        f"_{source}Rendered from `outputs/metrics.json`. {pop['people']:,} people "
        f"({pop['customers']:,} became customers) assigned a state at each of {pop['periods']} "
        f"month ends, {pop['first_period']} to {pop['last_period']}._",
        "",
        "**State definitions** (first matching rule in this order wins, so every person has "
        "exactly one state per month end):",
        "",
        "| Precedence | State | Definition at month end `d` (orders before `d` only) |",
        "|---:|---|---|",
    ]
    for i, s in enumerate(cfg["precedence"], start=1):
        lines.append(f"| {i} | {STATE_LABELS[s]} | {cfg['definitions'][s]} |")
    lines += [
        "",
        f"**State mix.** Customers by state at the end of {cur['period']} and a year earlier:",
        "",
        f"| State | {cur['period']} | Share of customers | {prev['period']} | Share of customers |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in CUSTOMER_STATES:
        lines.append(f"| {STATE_LABELS[s]} | {_n(cur['counts'][s])} | "
                     f"{_pct(cur['customer_shares'][s])} | {_n(prev['counts'][s])} | "
                     f"{_pct(prev['customer_shares'][s])} |")
    lines += [
        f"| **All customers** | **{_n(cur['customers'])}** | | **{_n(prev['customers'])}** | |",
        f"| Prospects (not yet customers) | {_n(cur['counts']['prospect'])} | | "
        f"{_n(prev['counts']['prospect'])} | |",
        "",
        f"**Transition matrix**, pooled over the {t['months']} month-to-month steps from "
        f"{t['first_origin']} to {t['last_destination']} (row = state at month end, column = "
        "state one month later; each row sums to 100% of the people in that state):",
        "",
        "| From \\ to | " + " | ".join(STATE_LABELS[s] for s in STATES) + " | People-months |",
        "|---|" + "---:|" * (len(STATES) + 1),
    ]
    for a in STATES:
        cells = " | ".join("·" if not t["counts"][a][b] else _pct(t["probabilities"][a][b])
                           for b in STATES)
        lines.append(f"| {STATE_LABELS[a]} | {cells} | {_n(sum(t['counts'][a].values()))} |")
    lines += [
        "",
        f"`·` = no transitions. {len(cfg['impossible_transitions'])} cells are impossible under "
        "the rules (for example Active -> Churned needs more than one month of extra idle "
        "time) and are required to be zero; they are. Of "
        f"{_n(t['new_leads'])} leads created in the window, {_n(t['new_leads_entering_as_new'])}"
        f" ({_pct(t['new_leads_entering_as_new'] / t['new_leads'])}) ordered in their first "
        "month and entered the panel directly as New.",
        "",
        "**What each state is worth next month** (same window):",
        "",
        "| State at month end | Customer-months | Share | Ordered next month | Revenue next "
        "month per customer | Share of next-month revenue |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["state_value"]:
        lines.append(f"| {STATE_LABELS[r['state']]} | {_n(r['customer_months'])} | "
                     f"{_pct(r['share_of_customer_months'])} | "
                     f"{_pct(r['next_month_purchase_rate'])} | "
                     f"{_usd(r['next_month_revenue_per_customer'])} | "
                     f"{_pct(r['share_of_next_month_revenue'])} |")
    lines += [
        "",
        f"**Decision points** (rates over the window above; revenue = net revenue in the "
        f"{cfg['follow_up_months']} months starting with the transition month, over every "
        "origin month with complete follow-up; descriptive, not causal):",
        "",
        "| Decision point | Favourable / unfavourable next state | Resolved | Favourable rate "
        "(95% CI) | Unfavourable per month | Revenue after: favourable | Revenue after: "
        "unfavourable | Annual revenue gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["decision_points"]:
        lines.append(
            f"| {r['label']} | {r['favourable']} / {r['unfavourable']} | {_n(r['resolved'])} | "
            f"{_pct(r['favourable_rate'])} ({_pct(r['rate_ci_low'])}-{_pct(r['rate_ci_high'])})"
            f" | {_n(r['unfavourable_per_month'])} | {_usd(r['revenue_after_favourable'])} | "
            f"{_usd(r['revenue_after_unfavourable'])} | {_usd(r['revenue_gap_per_year'])} |")
    points = [p["key"] for p in cfg["decision_points"]]
    labels = {p["key"]: p["label"] for p in cfg["decision_points"]}
    by_ch = {(r["decision_point"], r["acquisition_channel"]): r
             for r in metrics["decision_points_by_channel"]}
    channels = sorted({r["acquisition_channel"] for r in metrics["decision_points_by_channel"]})
    lines += [
        "",
        "Favourable rate by acquisition channel (resolved transitions in parentheses):",
        "",
        "| Channel | " + " | ".join(labels[k] for k in points) + " |",
        "|---|" + "---:|" * len(points),
    ]
    for c in channels:
        cells = " | ".join(f"{_pct(by_ch[(k, c)]['favourable_rate'])} "
                           f"({_n(by_ch[(k, c)]['resolved'])})" if (k, c) in by_ch else "-"
                           for k in points)
        lines.append(f"| {c.replace('_', ' ')} | {cells} |")
    lines += [
        "",
        "**Repurchase by idle time**: share of customers ordering during the next month, by "
        "days since their last order at month end:",
        "",
        "| Days since last order | " + " | ".join(r["days_since_last_order"]
                                                 for r in metrics["repurchase_by_recency"]) + " |",
        "|---|" + "---:|" * len(metrics["repurchase_by_recency"]),
        "| Ordered next month | " + " | ".join(_pct(r["repurchase_rate"])
                                              for r in metrics["repurchase_by_recency"]) + " |",
        "| Customer-months | " + " | ".join(_n(r["customer_months"])
                                           for r in metrics["repurchase_by_recency"]) + " |",
        "",
        "**Lead conversion by lead age**: share of not-yet-converted leads placing a first "
        "order during the next month:",
        "",
        "| Lead age (days) | " + " | ".join(r["lead_age_days"]
                                           for r in metrics["conversion_by_lead_age"]) + " |",
        "|---|" + "---:|" * len(metrics["conversion_by_lead_age"]),
        "| Converted next month | " + " | ".join(_pct(r["conversion_rate"], 2)
                                                for r in metrics["conversion_by_lead_age"]) + " |",
        "| Prospect-months | " + " | ".join(_n(r["prospect_months"])
                                           for r in metrics["conversion_by_lead_age"]) + " |",
    ]
    c = metrics["cohorts"]
    pooled = {r["months_since_acquisition"]: r for r in c["pooled"]}
    k_max = cfg["cohort_follow_up_months"]
    shown = [k for k in COHORT_MONTHS_SHOWN if k in pooled]
    lines += [
        "",
        f"**Cohort retention.** {c['count']} monthly acquisition cohorts ({c['first']} to "
        f"{c['last']}); the full triangle is in `outputs/cohort_retention.csv` and the figure "
        f"below. Pooled over the {pooled[0]['cohorts']} cohorts with at least {k_max} months "
        f"observed ({c['pooled_first_cohort']} to {c['pooled_last_cohort']}); the denominator is "
        f"the same {_n(pooled[0]['customers'])} customers in every column:",
        "",
        "| Months since first order | " + " | ".join(str(k) for k in shown) + " |",
        "|---|" + "---:|" * len(shown),
        "| Customers (denominator) | " + " | ".join(_n(pooled[k]["customers"]) for k in shown)
        + " |",
        "| Ordered in the month | " + " | ".join(_pct(pooled[k]["retention"]) for k in shown)
        + " |",
        "| Cumulative revenue per acquired customer | " + " | ".join(
            _usd(pooled[k]["cumulative_revenue_per_customer"]) for k in shown) + " |",
        "",
        "Same fixed cohorts by acquisition channel:",
        "",
        "| Channel | Customers | " + " | ".join(f"Month {k}" for k in CHANNEL_MONTHS)
        + f" | Revenue per customer, months 0-{k_max} |",
        "|---|---:|" + "---:|" * (len(CHANNEL_MONTHS) + 1),
    ]
    by = {(r["acquisition_channel"], r["months_since_acquisition"]): r for r in c["by_channel"]}
    for ch in sorted({r["acquisition_channel"] for r in c["by_channel"]},
                     key=lambda x: -by[(x, k_max)]["retention"]):
        lines.append(f"| {ch.replace('_', ' ')} | {_n(by[(ch, k_max)]['customers'])} | "
                     + " | ".join(_pct(by[(ch, k)]["retention"]) for k in CHANNEL_MONTHS)
                     + f" | {_usd(by[(ch, k_max)]['cumulative_revenue_per_customer'])} |")
    s = metrics["segments"]
    lines += [
        "",
        f"**RFM segments** as of {s['as_of']} ({_n(s['customers'])} customers with an order "
        f"before that date; outcomes are orders from {s['as_of']} to {s['outcome_end']}, "
        "measured after segmentation):",
        "",
        "| Segment | Rule | Customers | Share | Revenue, last 365 days (share) | Orders, last 365 "
        f"days | Plus members | Email open rate | Ordered in next {cfg['outcome_days']} days | "
        "Share of next-period revenue |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["rfm"]:
        seg = r["rfm_segment"]
        lines.append(
            f"| {sg.RFM_LABELS[seg]} | {cfg['rfm_rules'][seg]} | {_n(r['customers'])} | "
            f"{_pct(r['share_of_customers'])} | {_usd(r['revenue_365d'])} "
            f"({_pct(r['share_of_revenue_365d'])}) | {r['orders_365d']:.1f} | "
            f"{_pct(r['plus_member_rate'])} | {_pct(r['email_open_rate_90d'])} | "
            f"{_pct(r['next_purchase_rate'])} | {_pct(r['share_of_next_revenue'])} |")
    lines += [
        "",
        "Lifecycle states at the same date:",
        "",
        "| State | Customers | Share | Revenue, last 365 days (share) | Orders, last 365 days | "
        f"Browse sessions, last 90 days | Plus members | Ordered in next {cfg['outcome_days']} "
        "days | Revenue per customer, next period | Share of next-period revenue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["states"]:
        lines.append(
            f"| {STATE_LABELS[r['state']]} | {_n(r['customers'])} | "
            f"{_pct(r['share_of_customers'])} | {_usd(r['revenue_365d'])} "
            f"({_pct(r['share_of_revenue_365d'])}) | {r['orders_365d']:.1f} | "
            f"{r['browse_sessions_90d']:.1f} | {_pct(r['plus_member_rate'])} | "
            f"{_pct(r['next_purchase_rate'])} | {_usd(r['next_revenue_per_customer'])} | "
            f"{_pct(r['share_of_next_revenue'])} |")
    lines += [
        "",
        "Customers by lifecycle state (rows) and RFM segment (columns):",
        "",
        "| State | " + " | ".join(sg.RFM_LABELS[g] for g in sg.RFM_SEGMENTS) + " |",
        "|---|" + "---:|" * len(sg.RFM_SEGMENTS),
    ]
    for st in CUSTOMER_STATES:
        lines.append(f"| {STATE_LABELS[st]} | " + " | ".join(
            _n(s["crosstab"][st][g]) if s["crosstab"][st][g] else "·" for g in sg.RFM_SEGMENTS)
            + " |")
    audit = metrics["audit"]
    lines += [
        "",
        "**Integrity audit** (the pipeline refuses to write results if any check fails):",
        "",
        "| Check | Result |",
        "|---|---|",
        *[f"| {name.replace('_', ' ')} | {'pass' if ok else 'FAIL'} |"
          for name, ok in audit["checks"].items()],
        "",
        f"{_n(audit['details']['person_months'])} person-months audited; "
        f"{audit['details']['impossible_transitions_checked']} rule-forbidden transitions "
        f"checked; point-in-time probe at {audit['details']['point_in_time_probe']}.",
    ]
    return "\n".join(lines)


def write_outputs(tables: Mapping[str, pd.DataFrame], out_dir: Path, readme: Path | None = None,
                  config: LifecycleConfig | None = None, data_manifest: Mapping | None = None
                  ) -> dict:
    """Run the analysis and write metrics JSON, CSV tables, figures and the README block."""
    config = config or LifecycleConfig()
    metrics, out = run_analysis(tables, config)
    if data_manifest is not None:
        metrics = {"data": {"seed": data_manifest["seed"],
                            "n_prospects": data_manifest["n_prospects"]}, **metrics}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name, df in out.items():
        numeric = df.select_dtypes("number").columns
        df.assign(**df[numeric].round(4)).to_csv(out_dir / f"{name}.csv", index=False,
                                                 date_format="%Y-%m-%d")
    save_figures(metrics, out, out_dir / "figures")
    if readme is not None and readme.exists():
        update_generated_block(readme, render_markdown(metrics), BEGIN_MARKER, END_MARKER)
    return metrics
