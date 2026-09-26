"""Section 03 pipeline: funnel diagnostics, then the pre-registered analysis of the one-page
checkout experiment, then exploratory subgroups; writes tables, figures and the README block.

Everything reported in ``projects/03_conversion/README.md`` between the generated-block markers
is rendered from ``outputs/metrics.json`` by :func:`render_markdown`.
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

from northstar.conversion import experiment as ex
from northstar.conversion import funnel as fn
from northstar.conversion import stats
from northstar.profile import GRID, TEXT_PRIMARY, TEXT_SECONDARY, _style, update_generated_block

__all__ = ["AssignmentError", "ConversionConfig", "render_markdown", "run_analysis",
           "write_outputs"]

AssignmentError = ex.AssignmentError
BEGIN_MARKER = "<!-- BEGIN GENERATED: conversion-results -->"
END_MARKER = "<!-- END GENERATED: conversion-results -->"

# Reference categorical palette (fixed slot order) plus a muted ink for reference marks.
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
LIGHT_BLUE = "#9ec5f0"

PRACTICAL = {
    "ci_above_mei": "the whole CI is above the minimum effect of interest",
    "estimate_above_mei": "the point estimate is above the minimum effect of interest, but the "
                          "CI extends below it",
    "estimate_below_mei": "the point estimate is below the minimum effect of interest, but the "
                          "CI extends above it",
    "ci_below_mei": "the whole CI is below the minimum effect of interest",
}
DIMENSION_LABELS = {"device_type": "Device", "platform": "Platform",
                    "traffic_source": "Traffic source", "visit_number": "Prospect visit",
                    "acquisition_channel": "Acquisition channel", "lead_type": "Lead type"}


@dataclass(frozen=True)
class ConversionConfig:
    plan: ex.ExperimentPlan = field(default_factory=ex.ExperimentPlan)
    funnel_weeks: int = 52  # descriptive funnel: the year before the experiment starts
    prospect_window_days: int = 30
    opportunity_dimension: str = "device_type"
    opportunity_reference: str = "desktop"
    min_entering_for_weakest: int = 1000


def _clean(obj):
    """JSON-safe copy with floats rounded (stable diffs; NaN -> None)."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


# ---------------------------------------------------------------- analysis
def funnel_analysis(tables: Mapping[str, pd.DataFrame], start: pd.Timestamp,
                    config: ConversionConfig) -> tuple[dict, dict[str, pd.DataFrame]]:
    period_start = start - pd.Timedelta(weeks=config.funnel_weeks)
    frame = fn.session_frame(tables, period_start, start)
    prospect = frame.loc[frame["visitor_type"] == "prospect"]
    by_visitor = {v: fn.stage_table(frame.loc[frame["visitor_type"] == v, "depth"], v)
                  for v in ("prospect", "customer")}
    overall = pd.concat([t.assign(visitor_type=v) for v, t in by_visitor.items()],
                        ignore_index=True)
    cohort_end = start - pd.Timedelta(days=config.prospect_window_days)
    leads = fn.prospect_funnel(tables, period_start, cohort_end, config.prospect_window_days)
    lead_table = fn.stage_table(leads["depth"], "prospect funnel")
    segments = fn.segment_funnel(prospect)
    opportunity = fn.dropoff_opportunity(segments, config.opportunity_dimension,
                                         config.opportunity_reference)

    main = by_visitor["prospect"]
    steps = main.iloc[1:]
    weakest = []
    for i, step in enumerate(fn.STEPS):
        entering = segments[f"reached_{fn.FUNNEL_STAGES[i]}"]
        eligible = segments.loc[entering >= config.min_entering_for_weakest]
        if eligible.empty:  # small datasets: fall back to every level
            eligible = segments.loc[entering > 0]
        row = eligible.loc[eligible[f"rate_{step}"].idxmin()]
        weakest.append({"step": step, "dimension": row["dimension"], "level": row["level"],
                        "rate": row[f"rate_{step}"],
                        "overall_rate": main["step_conversion"].iloc[i + 1]})
    metrics = {
        "period_start": period_start, "period_end_exclusive": start,
        "integrity": fn.funnel_integrity(tables["sessions"], tables["funnel_events"]),
        "session_funnel": {v: _records(t) for v, t in by_visitor.items()},
        "prospect_cohort": {"created_from": period_start, "created_before": cohort_end,
                            "window_days": config.prospect_window_days, "leads": len(leads)},
        "prospect_funnel": _records(lead_table),
        "largest_loss_step": fn.STEPS[int(steps["lost_at_step"].to_numpy().argmax())],
        "lowest_step_conversion": fn.STEPS[int(steps["step_conversion"].to_numpy().argmin())],
        "segments": _records(segments),
        "weakest_by_step": weakest,
        "opportunity": _records(opportunity),
    }
    out = {"funnel_overall": overall, "funnel_prospect_cohort": lead_table,
           "funnel_segments": segments, "dropoff_opportunity": opportunity}
    return metrics, out


def experiment_analysis(tables: Mapping[str, pd.DataFrame], plan: ex.ExperimentPlan
                        ) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Order matters and mirrors the plan: design checks, primary, guardrails, secondary,
    decision; only then robustness, mechanism and exploratory subgroups."""
    reg = ex.registry(tables, plan.experiment_id)
    units = ex.build_units(tables, plan)
    audit = ex.assignment_audit(tables, units, plan)
    if not audit["passed"]:
        failed = [k for k, ok in audit["checks"].items() if not ok]
        raise AssignmentError(f"Assignment audit failed: {failed}; details: {audit['details']}")
    balance = ex.covariate_balance(units)
    planned = ex.planning(tables, plan)

    primary = ex.estimate(units, plan.primary_metric, "proportion", plan.alpha)
    guardrails = {}
    for g in plan.guardrails:
        r = ex.estimate(units, g.metric, g.kind, plan.alpha)
        guardrails[g.metric] = {"label": g.label, "scale": g.scale, "margin": g.margin,
                                "status": g.status(r), **r}
    secondary = {m: {"label": label, **ex.estimate(units, m, "mean", plan.alpha)}
                 for m, label in plan.secondary_metrics}
    for s, p_adj in zip(secondary.values(),
                        stats.holm([s["p_value"] for s in secondary.values()]), strict=True):
        s["p_holm"] = float(p_adj)
    basket = ex.estimate(units, "first_order_items", "mean", plan.alpha)
    decision = ex.decide(primary, guardrails, plan)
    business = ex.business_translation(units, primary, secondary,
                                       guardrails["first_order_value"], reg, plan)

    y = units[plan.primary_metric].to_numpy(dtype=float)
    treated = units["treated"].to_numpy()
    robustness = {
        "p_value_pooled": primary["p_value_pooled"],
        "permutation_p_value": stats.permutation_p_value(y, treated, plan.n_permutations,
                                                         plan.seed),
        "n_permutations": plan.n_permutations,
        "regression_adjusted": stats.regression_adjusted_effect(
            y, treated, ex.covariate_matrix(units), plan.alpha),
        "aa_control_arm": stats.aa_simulation(y[~treated], plan.n_aa_simulations, plan.seed,
                                              plan.alpha, float(reg["treatment_share"])),
    }
    robustness["regression_adjusted"]["se_ratio_vs_unadjusted"] = (
        robustness["regression_adjusted"]["se"] / primary["se"])
    arm_funnel = ex.arm_funnel(units, plan.alpha)
    subgroups, heterogeneity = ex.subgroup_effects(units, plan)
    cumulative = ex.cumulative_effects(units, plan)
    curve = ex.power_curve(planned, primary, plan)

    metrics = {
        "registry": {k: reg[k] for k in ("experiment_id", "experiment_name", "hypothesis",
                                         "randomization_unit", "eligible_population",
                                         "start_date", "end_date", "treatment_share",
                                         "primary_metric", "guardrail_metrics")},
        "assignment_audit": audit,
        "balance": {"max_abs_smd": float(balance["smd"].abs().max()),
                    "limit": plan.balance_smd_limit,
                    "covariates_over_limit": int((balance["smd"].abs()
                                                  > plan.balance_smd_limit).sum()),
                    "covariates_checked": len(balance)},
        "planning": planned,
        "primary": primary,
        "guardrails": guardrails,
        "secondary": secondary,
        "basket_items": basket,
        "decision": decision,
        "business": business,
        "realized_design": ex.realized_design(primary, plan),
        "robustness": robustness,
        "arm_funnel": _records(arm_funnel),
        "subgroups": _records(subgroups),
        "heterogeneity": heterogeneity,
        "subgroup_tests": len(subgroups),
        "cumulative": _records(cumulative.assign(data_through=cumulative["data_through"]
                                                 .astype(str))),
    }
    out = {"experiment_units_summary": _unit_summary(units), "covariate_balance": balance,
           "experiment_results": _results_table(primary, guardrails, secondary, plan),
           "experiment_funnel_by_arm": arm_funnel, "subgroup_effects": subgroups,
           "power_curve": curve, "cumulative_effect": cumulative}
    return metrics, out


def _unit_summary(units: pd.DataFrame) -> pd.DataFrame:
    """Per-arm aggregates (no unit-level rows are written)."""
    cols = ["converted", "first_order_value", "first_order_items", "first_order_revenue",
            "revenue_90d",
            *[f"reached_{s}" for s in ex.REACH_STAGES], "prior_sessions"]
    g = units.groupby("variant")[cols]
    return g.mean().add_prefix("mean_").assign(units=g.size()).reset_index()


def _results_table(primary, guardrails, secondary, plan) -> pd.DataFrame:
    rows = [{"role": "primary", "metric": plan.primary_metric, "label": plan.primary_label,
             **_flat(primary)}]
    rows += [{"role": "guardrail", "metric": k, "label": v["label"], "status": v["status"],
              "margin": v["margin"], "margin_scale": v["scale"], **_flat(v)}
             for k, v in guardrails.items()]
    rows += [{"role": "secondary", "metric": k, "label": v["label"], "p_holm": v["p_holm"],
              **_flat(v)} for k, v in secondary.items()]
    return pd.DataFrame(rows)


def _flat(r: Mapping) -> dict:
    return {"control": r["control"], "treatment": r["treatment"], "n_control": r["n_control"],
            "n_treatment": r["n_treatment"], "diff": r["diff"], "ci_low": r["ci"][0],
            "ci_high": r["ci"][1], "relative": r["relative"],
            "relative_ci_low": (r["relative_ci"] or [None])[0],
            "relative_ci_high": (r["relative_ci"] or [None, None])[1], "p_value": r["p_value"]}


def run_analysis(tables: Mapping[str, pd.DataFrame], config: ConversionConfig | None = None
                 ) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Run the full section 03 analysis; returns (metrics dict, output tables)."""
    config = config or ConversionConfig()
    plan = config.plan
    reg = ex.registry(tables, plan.experiment_id)
    funnel_metrics, funnel_out = funnel_analysis(tables, reg["start_date"], config)
    exp_metrics, exp_out = experiment_analysis(tables, plan)
    metrics = {
        "config": {"funnel_weeks": config.funnel_weeks,
                   "prospect_window_days": config.prospect_window_days,
                   "opportunity_dimension": config.opportunity_dimension,
                   "opportunity_reference": config.opportunity_reference,
                   "min_entering_for_weakest": config.min_entering_for_weakest},
        # The plan is recorded before any result, and the decision never reads subgroups.
        "plan": plan.as_dict(),
        "funnel": funnel_metrics,
        "experiment": exp_metrics,
    }
    return _clean(metrics), {**funnel_out, **exp_out}


# ---------------------------------------------------------------- figures
def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def _labels(ax: plt.Axes, x: str, y: str) -> None:
    ax.set_xlabel(x, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(y, color=TEXT_SECONDARY, fontsize=9)
    ax.set_axisbelow(True)


def save_figures(metrics: Mapping, out: Mapping[str, pd.DataFrame], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    f, e, plan = metrics["funnel"], metrics["experiment"], metrics["plan"]
    period = f"{f['period_start']} to {f['period_end_exclusive']} (exclusive)"
    paths = []

    # 1. Session funnel: prospects vs existing customers.
    overall = out["funnel_overall"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=120)
    ys = np.arange(len(fn.FUNNEL_STAGES))[::-1]
    for offset, (visitor, color, name) in zip(
            (0.19, -0.19), (("prospect", BLUE, "Prospect sessions (pre-first-order)"),
                            ("customer", ORANGE, "Existing-customer sessions")), strict=True):
        t = overall.loc[overall["visitor_type"] == visitor]
        share = t["share_of_start"].to_numpy() * 100
        ax.barh(ys + offset, share, height=0.36, color=color, label=name)
        for k, (y, v, n, step) in enumerate(zip(ys, share, t["reached"], t["step_conversion"],
                                                strict=True)):
            text = f"{v:.1f}%" if visitor == "customer" else f"{n:,.0f} sessions" + (
                f", step {step:.0%}" if k else "")
            ax.text(v + 1, y + offset, text, va="center", fontsize=8, color=TEXT_PRIMARY)
    ax.set_yticks(ys)
    ax.set_yticklabels([fn.STAGE_LABELS[s] for s in fn.FUNNEL_STAGES])
    ax.set_xlim(0, 135)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _labels(ax, "Share of sessions reaching the stage (%)", "")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_PRIMARY)
    _style(ax, "Where sessions drop out of the funnel", f"Web and app sessions, {period}")
    paths.append(_finish(fig, fig_dir / "funnel_stages.png"))

    # 2. Step conversion by segment (small multiples, one hue).
    seg = out["funnel_segments"]
    prospect = overall.loc[overall["visitor_type"] == "prospect"]
    dims = ["device_type", "visit_number"]
    fig, axes = plt.subplots(len(dims), len(fn.STEPS), figsize=(12, 5.2), dpi=120,
                             sharex="col")
    for r, dim in enumerate(dims):
        sub = seg.loc[seg["dimension"] == dim]
        for c, step in enumerate(fn.STEPS):
            ax = axes[r, c]
            yy = np.arange(len(sub))[::-1]
            vals = sub[f"rate_{step}"].to_numpy() * 100
            ax.barh(yy, vals, height=0.6, color=BLUE)
            ax.axvline(prospect["step_conversion"].iloc[c + 1] * 100, color=MUTED,
                       linestyle="--", linewidth=1.2)
            for y, v in zip(yy, vals, strict=True):
                ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8, color=TEXT_PRIMARY)
            ax.set_yticks(yy)
            ax.set_yticklabels(sub["level"] if c == 0 else [""] * len(sub))
            ax.set_xlim(0, 105)
            ax.xaxis.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
            if r == 0:
                a, b = step.split("->")
                ax.set_title(f"{fn.STAGE_LABELS[a]} →\n{fn.STAGE_LABELS[b]}", fontsize=9,
                             color=TEXT_PRIMARY, loc="left")
            if c == 0:
                ax.set_ylabel(DIMENSION_LABELS[dim], color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle("Step conversion of prospect sessions by segment (dashed = all prospect "
                 "sessions)", x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.supxlabel("Step conversion (%)", fontsize=9, color=TEXT_SECONDARY)
    paths.append(_finish(fig, fig_dir / "step_conversion_by_segment.png"))

    # 3. Experiment effects: absolute (pp) and relative (%) panels.
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=120,
                                      gridspec_kw={"width_ratios": [1, 1.2]})
    prim, gr, sec = e["primary"], e["guardrails"], e["secondary"]
    abs_rows = [("First-purchase conversion\n(primary)", prim, plan["minimum_effect_of_interest"],
                 "Minimum effect of interest"),
                ("Checkout-start rate\n(guardrail)", gr["reached_checkout_start"],
                 -gr["reached_checkout_start"]["margin"], "Guardrail tolerance")]
    rel_rows = [("First-order value\n(guardrail)", gr["first_order_value"],
                 -gr["first_order_value"]["margin"], "Guardrail tolerance"),
                ("First-order revenue\nper prospect", sec["first_order_revenue"], None, None),
                ("90-day revenue\nper prospect", sec["revenue_90d"], None, None)]
    for ax, rows, key, scale, xlabel in (
            (left, abs_rows, "ci", 100, "Treatment − control (percentage points)"),
            (right, rel_rows, "relative_ci", 100, "Treatment vs control (% change)")):
        yy = np.arange(len(rows))[::-1]
        seen = set()
        for y, (_, r, ref, ref_label) in zip(yy, rows, strict=True):
            point = r["diff"] if key == "ci" else r["relative"]
            lo, hi = r[key]
            ax.plot([lo * scale, hi * scale], [y, y], color=BLUE, linewidth=2.5,
                    solid_capstyle="round")
            ax.plot([point * scale], [y], "o", color=BLUE, markersize=8,
                    markeredgecolor="white", markeredgewidth=2)
            ax.text(hi * scale, y + 0.18, f"{point * scale:+.1f} [{lo * scale:+.1f}, "
                    f"{hi * scale:+.1f}]", fontsize=8, color=TEXT_PRIMARY, ha="right")
            if ref is not None:
                mei = ref_label.startswith("Minimum")
                ax.plot([ref * scale] * 2, [y - 0.3, y + 0.3], color=ORANGE if mei else MUTED,
                        linestyle="-" if mei else "--", linewidth=2,
                        label=None if ref_label in seen else ref_label)
                seen.add(ref_label)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_yticks(yy)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        _labels(ax, xlabel, "")
        if seen:
            ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=TEXT_PRIMARY)
    _style(left, "One-page checkout: effects with 95% CIs", "Per assigned prospect")
    _style(right, "", "Value metrics (delta-method CIs on the ratio)")
    paths.append(_finish(fig, fig_dir / "experiment_effects.png"))

    # 4. Subgroup forest plot.
    sub = out["subgroup_effects"]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(sub) + 1.6), dpi=120)
    yy = np.arange(len(sub))[::-1]
    for i, (y, (_, r)) in enumerate(zip(yy, sub.iterrows(), strict=True)):
        ax.plot([r["ci_bonferroni_low"] * 100, r["ci_bonferroni_high"] * 100], [y, y],
                color=LIGHT_BLUE, linewidth=1.5,
                label=f"Bonferroni-adjusted CI ({len(sub)} tests)" if i == 0 else None)
        ax.plot([r["ci_low"] * 100, r["ci_high"] * 100], [y, y], color=BLUE, linewidth=3,
                solid_capstyle="round", label="Unadjusted 95% CI" if i == 0 else None)
        ax.plot([r["diff"] * 100], [y], "o", color=BLUE, markersize=7, markeredgecolor="white",
                markeredgewidth=1.5)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.axvline(prim["diff"] * 100, color=ORANGE, linestyle="--", linewidth=1.5,
               label="Overall effect")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{DIMENSION_LABELS[d]}: {lv}" for d, lv in
                        zip(sub["dimension"], sub["level"], strict=True)])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _labels(ax, "Lift in first-purchase conversion (percentage points)", "")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_PRIMARY)
    _style(ax, "Exploratory subgroup effects", "Not used for the decision; see Holm-adjusted "
           "p-values and heterogeneity tests")
    paths.append(_finish(fig, fig_dir / "subgroup_forest.png"))

    # 5. Power curve.
    curve = out["power_curve"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=120)
    ax.plot(curve["absolute_lift"] * 100, curve["power_realized"] * 100, color=BLUE,
            linewidth=2, label="As run (actual arms, observed control rate)")
    ax.plot(curve["absolute_lift"] * 100, curve["power_planned"] * 100, color=ORANGE,
            linewidth=2, linestyle="--", label="Planned (pre-period traffic and baseline)")
    ax.axhline(plan["target_power"] * 100, color=MUTED, linestyle="--", linewidth=1.2)
    ax.axvline(plan["minimum_effect_of_interest"] * 100, color=MUTED, linestyle=":",
               linewidth=1.5)
    ax.text(plan["minimum_effect_of_interest"] * 100 + 0.08, 4, "minimum effect\nof interest",
            fontsize=8, color=TEXT_SECONDARY)
    mde = e["realized_design"]["mde_absolute"] * 100
    ax.plot([mde], [plan["target_power"] * 100], "o", color=BLUE, markersize=8,
            markeredgecolor="white", markeredgewidth=2)
    ax.text(mde + 0.1, plan["target_power"] * 100 - 7, f"MDE {mde:.1f} pp", fontsize=8,
            color=TEXT_PRIMARY)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, curve["absolute_lift"].max() * 100)
    ax.grid(True, color=GRID, linewidth=0.8)
    _labels(ax, "True absolute lift in conversion (percentage points)", "Power (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_PRIMARY)
    _style(ax, "How large an effect could this test detect?",
           f"Two-sided alpha {plan['alpha']:g}; dashed line = {plan['target_power']:.0%} power")
    paths.append(_finish(fig, fig_dir / "power_curve.png"))

    # 6. Cumulative estimate by week (why the plan forbids peeking).
    cum = out["cumulative_effect"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=120)
    ax.fill_between(cum["week"], cum["ci_low"] * 100, cum["ci_high"] * 100, color=LIGHT_BLUE,
                    alpha=0.5, linewidth=0, label="Nominal 95% CI at that look")
    ax.plot(cum["week"], cum["diff"] * 100, color=BLUE, linewidth=2)
    sig = cum["nominally_significant"].to_numpy(dtype=bool)
    ax.plot(cum.loc[sig, "week"], cum.loc[sig, "diff"] * 100, "o", color=BLUE, markersize=8,
            markeredgecolor="white", markeredgewidth=2, label="p < 0.05 at that look")
    ax.plot(cum.loc[~sig, "week"], cum.loc[~sig, "diff"] * 100, "o", color="white",
            markersize=8, markeredgecolor=BLUE, markeredgewidth=2, label="p ≥ 0.05")
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xticks(cum["week"])
    ax.grid(True, color=GRID, linewidth=0.8)
    _labels(ax, "Week of the experiment", "Estimated lift (percentage points)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=TEXT_PRIMARY)
    _style(ax, "The estimate a weekly dashboard would have shown",
           "Only the final, pre-planned look is used for the decision")
    paths.append(_finish(fig, fig_dir / "cumulative_effect.png"))
    return paths


# ---------------------------------------------------------------- README rendering
def _pct(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def _pp(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v * 100:+.{digits}f} pp"


def _pp_ci(ci: list[float] | None, digits: int = 2) -> str:
    return "-" if ci is None else f"[{ci[0] * 100:+.{digits}f}, {ci[1] * 100:+.{digits}f}]"


def _rel(v: float | None, ci: list[float] | None) -> str:
    if v is None:
        return "-"
    return f"{v * 100:+.1f}% [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]"


def _usd(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "-"
    return f"-${-v:,.{digits}f}" if v < 0 else f"${v:,.{digits}f}"


def _usd_ci(ci: list[float], digits: int = 2) -> str:
    return f"[{_usd(ci[0], digits)}, {_usd(ci[1], digits)}]"


def _p(v: float) -> str:
    return "< 0.0001" if v < 0.0001 else f"{v:.4f}"


def _step_label(step: str) -> str:
    a, b = step.split("->")
    return f"{fn.STAGE_LABELS[a]} → {fn.STAGE_LABELS[b].lower()}"


def _render_funnel(f: Mapping, cfg: Mapping) -> list[str]:
    integ = f["integrity"]
    defects = {k: v for k, v in integ.items() if k not in ("sessions", "events")}
    customer = {r["stage"]: r for r in f["session_funnel"]["customer"]}
    lines = [
        f"**Funnel data quality.** {integ['events']:,} funnel events across "
        f"{integ['sessions']:,} sessions; "
        + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in defects.items())
        + ". Every table below is checked to be non-increasing downstream.",
        "",
        f"**Session funnel** ({f['period_start']} to {f['period_end_exclusive']}, exclusive: "
        f"the {cfg['funnel_weeks']} weeks before the experiment)",
        "",
        "| Stage | Prospect sessions | Share of sessions | Step conversion | Lost at this step | "
        "Share of all losses | Existing-customer step conversion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in f["session_funnel"]["prospect"]:
        step = "-" if r["stage_number"] == 1 else _pct(r["step_conversion"])
        cust = "-" if r["stage_number"] == 1 else _pct(customer[r["stage"]]["step_conversion"])
        lines.append(f"| {fn.STAGE_LABELS[r['stage']]} | {r['reached']:,} | "
                     f"{_pct(r['share_of_start'])} | {step} | {r['lost_at_step']:,} | "
                     f"{_pct(r['share_of_all_losses'])} | {cust} |")
    c = f["prospect_cohort"]
    lines += [
        "",
        f"Largest absolute loss: **{_step_label(f['largest_loss_step'])}**. Lowest step "
        f"conversion: **{_step_label(f['lowest_step_conversion'])}**.",
        "",
        f"**Person-level funnel** ({c['leads']:,} leads created {c['created_from']} to "
        f"{c['created_before']} (exclusive), deepest stage in any session within "
        f"{c['window_days']} days of lead creation)",
        "",
        "| Stage | Leads reaching | Share of leads | Step conversion |",
        "|---|---:|---:|---:|",
    ]
    for r in f["prospect_funnel"]:
        step = "-" if r["stage_number"] == 1 else _pct(r["step_conversion"])
        lines.append(f"| {fn.STAGE_LABELS[r['stage']]} | {r['reached']:,} | "
                     f"{_pct(r['share_of_start'])} | {step} |")
    lines += [
        "",
        "**Step conversion by segment** (prospect sessions; descriptive, not causal)",
        "",
        "| Segment | Level | Sessions | " + " | ".join(_step_label(s) for s in fn.STEPS)
        + " | Session → purchase |",
        "|---|---|---:|" + "---:|" * (len(fn.STEPS) + 1),
    ]
    for r in f["segments"]:
        rates = " | ".join(_pct(r[f"rate_{s}"]) for s in fn.STEPS)
        lines.append(f"| {DIMENSION_LABELS[r['dimension']]} | {r['level']} | "
                     f"{r['sessions']:,} | {rates} | {_pct(r['session_to_purchase'])} |")
    lines += [
        "",
        f"Weakest segment at each step (levels with at least "
        f"{cfg['min_entering_for_weakest']:,} sessions "
        "entering the step):",
        "",
        "| Step | Weakest segment | Its rate | All prospect sessions |",
        "|---|---|---:|---:|",
    ]
    for w in f["weakest_by_step"]:
        lines.append(f"| {_step_label(w['step'])} | {DIMENSION_LABELS[w['dimension']]}: "
                     f"{w['level']} | {_pct(w['rate'])} | {_pct(w['overall_rate'])} |")
    lines += [
        "",
        f"**Drop-off sizing: {DIMENSION_LABELS[cfg['opportunity_dimension']].lower()} gap to "
        f"{cfg['opportunity_reference']}.** Extra step completions in the period if each level "
        f"converted at the {cfg['opportunity_reference']} rate, and the purchases they would "
        "carry through at that level's own downstream rates. A benchmark gap, not a causal "
        "estimate.",
        "",
        "| Level | Step | Entering | Level rate | Reference rate | Extra completions | "
        "Purchase equivalents |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in f["opportunity"]:
        lines.append(f"| {r['level']} | {_step_label(r['step'])} | {r['entering']:,} | "
                     f"{_pct(r['level_rate'])} | {_pct(r['reference_rate'])} | "
                     f"{r['extra_step_completions']:+,.0f} | "
                     f"{r['purchase_equivalents']:+,.0f} |")
    return lines


def _render_experiment(e: Mapping, plan: Mapping) -> list[str]:
    reg, audit, bal, pl = e["registry"], e["assignment_audit"], e["balance"], e["planning"]
    prim, dec = e["primary"], e["decision"]
    mei = plan["minimum_effect_of_interest"]
    lines = [
        f"**Experiment registry.** `{reg['experiment_id']}` ({reg['experiment_name']}), "
        f"{reg['start_date']} to {reg['end_date']} inclusive, randomized by "
        f"{reg['randomization_unit']}, target treatment share {reg['treatment_share']:.0%}. "
        f"Hypothesis: _{reg['hypothesis']}_",
        "",
        "**Analysis plan (fixed before outcomes are computed; `ExperimentPlan`)**",
        "",
        "| Item | Plan |",
        "|---|---|",
        f"| Primary metric | {plan['primary_label']}: {plan['primary_definition']} |",
        f"| Registry wording | {reg['primary_metric']} |",
        f"| Test | {plan['test']}, alpha {plan['alpha']:g} |",
        f"| Minimum effect of interest (assumed) | {mei * 100:+.1f} pp absolute |",
        f"| Target power | {plan['target_power']:.0%} |",
    ]
    for g in plan["guardrails"]:
        tol = (f"{g['margin'] * 100:.0f}% relative" if g["scale"] == "relative"
               else f"{g['margin'] * 100:.0f} pp absolute")
        lines.append(f"| Guardrail (assumed tolerance) | {g['label']}: passes if the 95% CI "
                     f"rules out a decline worse than {tol} |")
    lines += [
        "| Secondary (decision support) | " + "; ".join(lbl for _, lbl in
                                                     plan["secondary_metrics"]) + " |",
        f"| Exploratory subgroups | {', '.join(plan['subgroup_dimensions'])}; Holm (FWER) "
        "and Benjamini-Hochberg (FDR) across all subgroup tests; not used for the decision |",
        "",
        "**Assignment audit** (the pipeline refuses to write results if any check fails)",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, ok in audit["checks"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'pass' if ok else 'FAIL'} |")
    d = audit["details"]
    lines += [
        "",
        f"Arms: {d['control']:,} control / {d['treatment']:,} treatment (observed treatment "
        f"share {d['observed_treatment_share']:.2%}; SRM chi-square p = "
        f"{_p(d['srm_p_value'])}, alarm below {d['srm_alpha']:g}). Covariate balance: largest "
        f"|standardized mean difference| {bal['max_abs_smd']:.3f} across "
        f"{bal['covariates_checked']} pre-treatment covariates "
        f"({bal['covariates_over_limit']} above {bal['limit']:g}).",
        "",
        f"**Power analysis.** Planning used the {pl['planned_weeks']:.0f} weeks before launch "
        f"({pl['pre_period_start']} to {pl['pre_period_end_exclusive']}, exclusive) with the "
        f"same eligibility rule: {pl['expected_eligible_prospects']:,} eligible prospects, "
        f"baseline conversion {_pct(pl['baseline_conversion'])}.",
        "",
        "| Design quantity | Planned | As run |",
        "|---|---:|---:|",
        f"| Units per arm | {pl['expected_per_arm']:,.0f} | {prim['n_control']:,} / "
        f"{prim['n_treatment']:,} |",
        f"| Control conversion | {_pct(pl['baseline_conversion'])} | {_pct(prim['control'])} |",
        f"| MDE at {plan['target_power']:.0%} power | {_pp(pl['mde_absolute'])} "
        f"({pl['mde_relative'] * 100:.1f}% relative) | "
        f"{_pp(e['realized_design']['mde_absolute'])} |",
        f"| Power at the minimum effect of interest ({_pp(mei, 1)}) | "
        f"{_pct(pl['power_at_mei'])} | {_pct(e['realized_design']['power_at_mei'])} |",
        "",
    ]
    rd = e["realized_design"]["retrodesign_at_mei"]
    lines += [
        f"Reaching {plan['target_power']:.0%} power at the minimum effect of interest would "
        f"need {pl['per_arm_needed_for_mei']:,} prospects per arm, about "
        f"{pl['weeks_needed_for_mei']:.0f} weeks at pre-period traffic. If the true lift were "
        f"exactly {_pp(mei, 1)}, a significant result from this design would overstate it by "
        f"{(rd['exaggeration_ratio'] - 1) * 100:.0f}% on average (exaggeration ratio "
        f"{rd['exaggeration_ratio']:.2f}; sign-error rate {rd['type_s_error']:.4f}).",
        "",
        "**Primary result (pre-registered)**",
        "",
        "| Metric | Control | Treatment | Absolute lift [95% CI] | Relative lift [95% CI] | "
        "p-value |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {plan['primary_label']} | {_pct(prim['control'], 2)} (n = {prim['n_control']:,})"
        f" | {_pct(prim['treatment'], 2)} (n = {prim['n_treatment']:,}) | {_pp(prim['diff'])} "
        f"{_pp_ci(prim['ci'])} | {_rel(prim['relative'], prim['relative_ci'])} | "
        f"{_p(prim['p_value'])} |",
        "",
        "**Guardrails**",
        "",
        "| Guardrail | Control | Treatment | Difference [95% CI] | Relative [95% CI] | "
        "p-value | Tolerance | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for g in e["guardrails"].values():
        money = g["scale"] == "relative"
        ctl = _usd(g["control"]) if money else _pct(g["control"], 2)
        trt = _usd(g["treatment"]) if money else _pct(g["treatment"], 2)
        diff = (f"{_usd(g['diff'])} {_usd_ci(g['ci'])}" if money
                else f"{_pp(g['diff'])} {_pp_ci(g['ci'])}")
        tol = (f"-{g['margin'] * 100:.0f}% relative" if money
               else f"-{g['margin'] * 100:.0f} pp")
        lines.append(f"| {g['label']} | {ctl} | {trt} | {diff} | "
                     f"{_rel(g['relative'], g['relative_ci'])} | {_p(g['p_value'])} | {tol} | "
                     f"**{g['status']}** |")
    fov, items = e["guardrails"]["first_order_value"], e["basket_items"]
    lines += [
        "",
        f"First-order value is measured on converters only ({fov['n_control']:,} control, "
        f"{fov['n_treatment']:,} treatment), a post-treatment subset. The secondary revenue "
        "metrics below include every assigned prospect (zeros for non-buyers), so they compare "
        "randomized groups. Basket diagnostic (same converters, descriptive): "
        f"{items['control']:.2f} vs {items['treatment']:.2f} items per first order "
        f"({_rel(items['relative'], items['relative_ci'])}, p = {_p(items['p_value'])}).",
        "",
        "**Secondary metrics** (decision support; Welch t-tests, Holm-adjusted within the "
        "family)",
        "",
        "| Metric | Control | Treatment | Difference [95% CI] | Relative [95% CI] | p-value | "
        "Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in e["secondary"].values():
        lines.append(f"| {s['label']} | {_usd(s['control'])} | {_usd(s['treatment'])} | "
                     f"{_usd(s['diff'])} {_usd_ci(s['ci'])} | "
                     f"{_rel(s['relative'], s['relative_ci'])} | {_p(s['p_value'])} | "
                     f"{_p(s['p_holm'])} |")
    b = e["business"]
    lines += [
        "",
        "**Decision (pre-specified rule in `experiment.decide`)**",
        "",
        "| Question | Answer |",
        "|---|---|",
        f"| Statistically significant at alpha {plan['alpha']:g}? | "
        f"{'yes' if dec['statistically_significant'] else 'no'} (p = {_p(prim['p_value'])}; "
        f"95% CI {'excludes' if dec['ci_excludes_zero'] else 'includes'} 0) |",
        f"| Practically significant? | {PRACTICAL[dec['practical_significance']]} "
        f"({_pp(mei, 1)}) |",
        f"| Guardrails pass? | {'yes' if dec['guardrails_pass'] else 'no'} |",
        f"| Recommendation | **{dec['recommendation_text']}** |",
        "",
        "**Business translation** (observed effects restated; eligible traffic annualized from "
        "the experiment window)",
        "",
        "| Quantity | Estimate [95% CI] |",
        "|---|---:|",
        f"| Extra first purchases per 1,000 eligible prospects | "
        f"{b['extra_first_purchases_per_1000']:+.1f} "
        f"[{b['extra_first_purchases_per_1000_ci'][0]:+.1f}, "
        f"{b['extra_first_purchases_per_1000_ci'][1]:+.1f}] |",
        f"| Extra first purchases per year (~{b['annual_eligible_prospects']:,.0f} eligible "
        f"prospects) | {b['extra_first_purchases_per_year']:+,.0f} "
        f"[{b['extra_first_purchases_per_year_ci'][0]:+,.0f}, "
        f"{b['extra_first_purchases_per_year_ci'][1]:+,.0f}] |",
        f"| First-order net revenue per 1,000 eligible prospects | "
        f"{_usd(b['first_order_revenue_per_1000'], 0)} "
        f"{_usd_ci(b['first_order_revenue_per_1000_ci'], 0)} |",
        f"| {plan['revenue_window_days']}-day net revenue per 1,000 eligible prospects | "
        f"{_usd(b['revenue_window_per_1000'], 0)} {_usd_ci(b['revenue_window_per_1000_ci'], 0)} "
        "|",
        f"| Conversion lift needed to hold first-order revenue flat at the treatment's "
        f"first-order value | {_pp(b['break_even_lift_given_observed_aov'])} |",
    ]
    r = e["robustness"]
    ra, aa = r["regression_adjusted"], r["aa_control_arm"]
    lines += [
        "",
        "**Robustness and validation of the method**",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Pooled-variance z-test (= chi-square) | p = {_p(r['p_value_pooled'])} |",
        f"| Randomization inference ({r['n_permutations']:,} re-randomizations) | "
        f"p = {_p(r['permutation_p_value'])} |",
        f"| Regression-adjusted lift (Lin estimator, {ra['n_covariates']} pre-treatment "
        f"covariates, HC2 SE) | {_pp(ra['diff'])} {_pp_ci(ra['ci'])}, p = {_p(ra['p_value'])};"
        f" SE {ra['se_ratio_vs_unadjusted']:.3f}x unadjusted |",
        f"| A/A: {aa['simulations']:,} random splits of the control arm "
        f"({aa['units']:,} units) | false-positive rate {_pct(aa['false_positive_rate'], 2)} "
        f"[{_pct(aa['false_positive_rate_ci'][0], 2)}, "
        f"{_pct(aa['false_positive_rate_ci'][1], 2)}] vs nominal "
        f"{_pct(aa['nominal_alpha'], 0)}; CI covers 0 in {_pct(aa['ci_coverage_of_zero'], 2)} |",
        "",
        "**Mechanism: funnel reach by arm** (share of assigned prospects reaching each stage in "
        "the window; the last row conditions on a post-treatment event and is descriptive)",
        "",
        "| Stage | Control | Treatment | Difference [95% CI] | p-value |",
        "|---|---:|---:|---:|---:|",
    ]
    for a in e["arm_funnel"]:
        label = ("Purchase given checkout start" if "|" in a["stage"]
                 else fn.STAGE_LABELS[a["stage"]])
        lines.append(f"| {label} | {_pct(a['control'])} | {_pct(a['treatment'])} | "
                     f"{_pp(a['diff'])} {_pp_ci([a['ci_low'], a['ci_high']])} | "
                     f"{_p(a['p_value'])} |")
    n_raw = sum(s["significant_raw"] for s in e["subgroups"])
    n_holm = sum(s["significant_holm"] for s in e["subgroups"])
    lines += [
        "",
        f"**Exploratory subgroups** ({e['subgroup_tests']} tests; {n_raw} nominally "
        f"significant, {n_holm} after Holm correction)",
        "",
        "| Segment | Level | n (C / T) | Control | Treatment | Lift [95% CI] | p | Holm p | "
        "BH q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in e["subgroups"]:
        lines.append(f"| {DIMENSION_LABELS[s['dimension']]} | {s['level']} | "
                     f"{s['n_control']:,} / {s['n_treatment']:,} | {_pct(s['control'])} | "
                     f"{_pct(s['treatment'])} | {_pp(s['diff'], 1)} "
                     f"{_pp_ci([s['ci_low'], s['ci_high']], 1)} | {_p(s['p_value'])} | "
                     f"{_p(s['p_holm'])} | {_p(s['q_bh'])} |")
    lines += [
        "",
        "Heterogeneity (Cochran's Q: do the levels share one effect?):",
        "",
        "| Segment | Levels | Q | df | p | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for h in e["heterogeneity"]:
        lines.append(f"| {DIMENSION_LABELS[h['dimension']]} | {h['levels']} | {h['q']:.2f} | "
                     f"{h['df']} | {_p(h['p_value'])} | {_p(h['p_holm'])} |")
    cum = e["cumulative"]
    n_sig = sum(c["nominally_significant"] for c in cum)
    lines += [
        "",
        f"**Weekly looks** (what a dashboard would have shown; {n_sig} of {len(cum)} looks "
        "had p < 0.05; only the final look is the planned analysis)",
        "",
        "| Week | Data through | Units | Lift [95% CI] | p-value |",
        "|---:|---|---:|---:|---:|",
    ]
    for c in cum:
        lines.append(f"| {c['week']} | {c['data_through']} | {c['units']:,} | "
                     f"{_pp(c['diff'])} {_pp_ci([c['ci_low'], c['ci_high']])} | "
                     f"{_p(c['p_value'])} |")
    return lines


def render_markdown(metrics: Mapping) -> str:
    data = metrics.get("data")
    source = f"Data seed `{data['seed']}`, {data['n_prospects']:,} prospects. " if data else ""
    cfg = metrics["config"]
    lines = [
        f"_{source}Rendered from `outputs/metrics.json`._",
        "",
        "### Part 1 - Funnel diagnostics",
        "",
        *_render_funnel(metrics["funnel"], cfg),
        "",
        "### Part 2 - One-page checkout experiment",
        "",
        *_render_experiment(metrics["experiment"], metrics["plan"]),
    ]
    return "\n".join(lines)


def write_outputs(tables: Mapping[str, pd.DataFrame], out_dir: Path, readme: Path | None = None,
                  config: ConversionConfig | None = None, data_manifest: Mapping | None = None
                  ) -> dict:
    """Run the analysis and write metrics JSON, CSV tables, figures and the README block."""
    metrics, out = run_analysis(tables, config)
    if data_manifest is not None:
        metrics = {"data": {"seed": data_manifest["seed"],
                            "n_prospects": data_manifest["n_prospects"]}, **metrics}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name, df in out.items():
        df.round(6).to_csv(out_dir / f"{name}.csv", index=False)
    save_figures(metrics, out, out_dir / "figures")
    if readme is not None and readme.exists():
        update_generated_block(readme, render_markdown(metrics), BEGIN_MARKER, END_MARKER)
    return metrics
