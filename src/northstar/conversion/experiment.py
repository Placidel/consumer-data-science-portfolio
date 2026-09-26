"""The one-page checkout experiment (EXP001): analysis plan, unit-level data and inference.

The randomization unit is the prospect, so every metric is computed per assigned prospect.
Session-level rates would treat correlated sessions from one person as independent. The
analysis plan (:class:`ExperimentPlan`) fixes the primary metric, test, alpha, minimum effect
of interest, guardrails and the subgroup family before any outcome is computed. The pipeline
records the plan first and never uses subgroup results for the ship decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from northstar.conversion import stats
from northstar.conversion.funnel import session_depth
from northstar.schema import FUNNEL_STAGES
from northstar.synthetic import experiment_variant
from northstar.timeline import DATA_END

REACH_STAGES = FUNNEL_STAGES[1:]  # every assigned prospect has a session by definition
COVARIATES = ("device_type", "acquisition_channel", "region", "age_band", "income_band",
              "email_opt_in", "lead_type")
NUMERIC_COVARIATES = ("prior_sessions", "days_since_lead")


@dataclass(frozen=True)
class Guardrail:
    metric: str
    label: str
    kind: str  # "proportion" (all assigned units) or "mean" (units where the metric is defined)
    scale: str  # margin applies to the "absolute" difference or the "relative" change
    margin: float  # largest tolerated degradation; higher is better for every guardrail

    def status(self, result: Mapping) -> str:
        """pass: the whole 95% CI is above -margin; fail: the whole CI is below it."""
        low, high = result["ci"] if self.scale == "absolute" else result["relative_ci"]
        if low >= -self.margin:
            return "pass"
        if high < -self.margin:
            return "fail"
        return "inconclusive"


@dataclass(frozen=True)
class ExperimentPlan:
    """Pre-registered analysis plan. The primary metric is the registry's; the minimum effect
    of interest and guardrail margins are business assumptions fixed before analysis."""

    experiment_id: str = "EXP001"
    primary_metric: str = "converted"
    primary_label: str = "First-purchase conversion"
    primary_definition: str = ("share of assigned prospects whose first order is placed between "
                               "assignment and the end of the experiment window")
    test: str = "two-sided two-proportion z-test (unpooled SE) with the matching Wald 95% CI"
    alpha: float = 0.05
    target_power: float = 0.80
    minimum_effect_of_interest: float = 0.02  # absolute lift in conversion (assumed)
    guardrails: tuple[Guardrail, ...] = (
        Guardrail("first_order_value", "Average first-order net value", "mean", "relative",
                  0.05),
        Guardrail("reached_checkout_start", "Checkout-start rate", "proportion", "absolute",
                  0.01),
    )
    secondary_metrics: tuple[tuple[str, str], ...] = (
        ("first_order_revenue", "First-order net revenue per assigned prospect"),
        ("revenue_90d", "Net revenue per assigned prospect, 90 days from assignment"),
    )
    revenue_window_days: int = 90
    subgroup_dimensions: tuple[str, ...] = ("device_type", "acquisition_channel", "lead_type")
    subgroup_min_units_per_arm: int = 30
    srm_alpha: float = 0.001
    balance_smd_limit: float = 0.1
    n_permutations: int = 10_000
    n_aa_simulations: int = 2_000
    seed: int = 20250303
    power_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.arange(0.0, 0.0625, 0.0025), 4)))

    def as_dict(self) -> dict:
        return asdict(self)


class AssignmentError(RuntimeError):
    """Raised when the assignment audit fails; results would not be trustworthy."""


# ---------------------------------------------------------------- data
def registry(tables: Mapping[str, pd.DataFrame], experiment_id: str) -> dict:
    exp = tables["experiments"]
    row = exp.loc[exp["experiment_id"] == experiment_id]
    if len(row) != 1:
        raise KeyError(f"experiment {experiment_id} not found in the registry")
    out = row.iloc[0].to_dict()
    out["window_end"] = out["end_date"] + pd.Timedelta(days=1)  # exclusive
    out["duration_days"] = (out["window_end"] - out["start_date"]).days
    return out


def eligible_population(tables: Mapping[str, pd.DataFrame], start: pd.Timestamp,
                        end: pd.Timestamp) -> pd.DataFrame:
    """Prospects with a pre-conversion web/app session in ``[start, end)`` and their first such
    session: the registry's eligibility rule, usable for any window (e.g. planning)."""
    s = tables["sessions"]
    s = s.loc[s["customer_id"].isna() & (s["session_start"] >= start) & (s["session_start"] < end)]
    return (s.groupby("prospect_id", as_index=False)["session_start"].min()
            .rename(columns={"session_start": "first_eligible_session"}))


def build_units(tables: Mapping[str, pd.DataFrame], plan: ExperimentPlan) -> pd.DataFrame:
    """One row per assigned prospect: arm, pre-treatment covariates and post-assignment outcomes.

    Covariates use only sessions *before* the assignment timestamp. Outcomes use only events at
    or after it: conversion and funnel reach inside the experiment window, revenue in a fixed
    window from assignment.
    """
    reg = registry(tables, plan.experiment_id)
    start, end = reg["start_date"], reg["window_end"]
    revenue_end = end + pd.Timedelta(days=plan.revenue_window_days)
    if revenue_end > DATA_END:
        raise ValueError(f"revenue window needs data until {revenue_end.date()}")
    a = tables["experiment_assignments"]
    units = a.loc[a["experiment_id"] == plan.experiment_id,
                  ["prospect_id", "variant", "assigned_at"]].merge(
        tables["prospects"].drop(columns=["campaign_id"]), on="prospect_id", how="left",
        validate="one_to_one")
    units = units.merge(tables["customers"][["prospect_id", "customer_id", "customer_since"]],
                        on="prospect_id", how="left", validate="one_to_one")
    units["treated"] = units["variant"] == "treatment"

    # Pre-treatment covariates.
    s = tables["sessions"][["session_id", "prospect_id", "customer_id", "session_start"]].merge(
        units[["prospect_id", "assigned_at"]], on="prospect_id")
    prior = s.loc[s["session_start"] < s["assigned_at"]].groupby("prospect_id").size()
    units["prior_sessions"] = units["prospect_id"].map(prior).fillna(0).astype(int)
    units["lead_type"] = np.where(units["prior_sessions"] == 0, "new lead", "returning lead")
    since_lead = units["assigned_at"] - units["created_at"]
    units["days_since_lead"] = since_lead.dt.total_seconds() / 86400

    # Primary outcome: first order inside [assigned_at, window end).
    units["converted"] = (units["customer_since"].notna()
                          & (units["customer_since"] >= units["assigned_at"])
                          & (units["customer_since"] < end))
    orders = tables["orders"][["customer_id", "order_ts", "net_amount", "item_count"]]
    first = (orders.sort_values(["order_ts", "customer_id"]).drop_duplicates("customer_id")
             .set_index("customer_id"))
    units["first_order_value"] = units["customer_id"].map(first["net_amount"]).where(
        units["converted"])
    units["first_order_items"] = units["customer_id"].map(first["item_count"]).where(
        units["converted"]).astype(float)
    units["first_order_revenue"] = units["first_order_value"].fillna(0.0)
    o = orders.merge(units[["customer_id", "assigned_at"]].dropna(subset=["customer_id"]),
                     on="customer_id")
    horizon = pd.Timedelta(days=plan.revenue_window_days)
    o = o.loc[(o["order_ts"] >= o["assigned_at"]) & (o["order_ts"] < o["assigned_at"] + horizon)]
    units[f"revenue_{plan.revenue_window_days}d"] = units["customer_id"].map(
        o.groupby("customer_id")["net_amount"].sum()).fillna(0.0)

    # Funnel reach across the prospect's pre-conversion sessions in the window.
    post = s.loc[s["customer_id"].isna() & (s["session_start"] >= s["assigned_at"])
                 & (s["session_start"] < end)]
    depth = post["session_id"].map(session_depth(tables["funnel_events"])).fillna(0)
    deepest = units["prospect_id"].map(depth.groupby(post["prospect_id"]).max()).fillna(0)
    for k, stage in enumerate(REACH_STAGES, start=2):
        units[f"reached_{stage}"] = deepest >= k
    units["start_date"], units["window_end"] = start, end
    return units.sort_values("prospect_id").reset_index(drop=True)


# ---------------------------------------------------------------- design checks
def assignment_audit(tables: Mapping[str, pd.DataFrame], units: pd.DataFrame,
                     plan: ExperimentPlan) -> dict:
    """Checks that make the comparison a valid randomized one; the pipeline stops on failure."""
    reg = registry(tables, plan.experiment_id)
    share = float(reg["treatment_share"])
    recomputed = np.array([experiment_variant(plan.experiment_id, pid, share)
                           for pid in units["prospect_id"]], dtype=object)
    eligible = eligible_population(tables, reg["start_date"], reg["window_end"])
    joined = eligible.merge(units[["prospect_id", "assigned_at"]], on="prospect_id", how="outer")
    counts = [int((~units["treated"]).sum()), int(units["treated"].sum())]
    srm = stats.sample_ratio_test(counts, [1 - share, share])
    checks = {
        "variant_matches_hash_rule": bool((units["variant"].to_numpy() == recomputed).all()),
        "one_assignment_per_prospect": bool(units["prospect_id"].is_unique),
        "assigned_inside_window": bool(((units["assigned_at"] >= reg["start_date"])
                                        & (units["assigned_at"] < reg["window_end"])).all()),
        "assigned_before_first_order": bool((units["customer_since"].isna()
                                             | (units["customer_since"] >= units["assigned_at"])
                                             ).all()),
        "every_eligible_prospect_assigned_at_first_exposure": bool(
            (joined["first_eligible_session"] == joined["assigned_at"]).all()),
        "no_sample_ratio_mismatch": bool(srm["p_value"] >= plan.srm_alpha),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "details": {"control": counts[0], "treatment": counts[1],
                        "designed_treatment_share": share,
                        "observed_treatment_share": srm["observed_share"][1],
                        "srm_chi2": srm["chi2"], "srm_p_value": srm["p_value"],
                        "srm_alpha": plan.srm_alpha,
                        "eligible_prospects": len(eligible)}}


def _covariate_frame(units: pd.DataFrame, drop_first: bool) -> pd.DataFrame:
    cats = pd.get_dummies(units[list(COVARIATES)].astype(str), prefix_sep="=",
                          drop_first=drop_first, dtype=float)
    nums = units[list(NUMERIC_COVARIATES)].astype(float)
    return pd.concat([cats, nums], axis=1)


def covariate_matrix(units: pd.DataFrame) -> np.ndarray:
    """Pre-treatment design matrix for regression adjustment (reference levels dropped)."""
    x = _covariate_frame(units, drop_first=True)
    x[list(NUMERIC_COVARIATES)] = np.log1p(x[list(NUMERIC_COVARIATES)])
    return x.to_numpy()


def covariate_balance(units: pd.DataFrame) -> pd.DataFrame:
    """Standardized mean difference of every pre-treatment covariate between arms."""
    x = _covariate_frame(units, drop_first=False)
    t = units["treated"].to_numpy()
    return pd.DataFrame([
        {"covariate": col, "control_mean": x.loc[~t, col].mean(),
         "treatment_mean": x.loc[t, col].mean(),
         "smd": stats.standardized_mean_difference(x.loc[~t, col], x.loc[t, col])}
        for col in x.columns])


# ---------------------------------------------------------------- inference
def estimate(units: pd.DataFrame, metric: str, kind: str, alpha: float) -> dict:
    t = units["treated"].to_numpy()
    if kind == "proportion":
        y = units[metric].to_numpy(dtype=bool)
        return stats.two_proportion_test(int(y[~t].sum()), int((~t).sum()), int(y[t].sum()),
                                         int(t.sum()), alpha)
    y = units[metric].to_numpy(dtype=float)
    defined = ~np.isnan(y)
    return stats.welch_test(y[~t & defined], y[t & defined], alpha)


def planning(tables: Mapping[str, pd.DataFrame], plan: ExperimentPlan) -> dict:
    """Pre-launch power analysis from the equally long window just before the experiment."""
    reg = registry(tables, plan.experiment_id)
    start = reg["start_date"]
    pre_start = start - pd.Timedelta(days=reg["duration_days"])
    pre = eligible_population(tables, pre_start, start).merge(
        tables["customers"][["prospect_id", "customer_since"]], on="prospect_id", how="left")
    converted = pre["customer_since"].notna() & (pre["customer_since"] < start)
    baseline = float(converted.mean())
    share = float(reg["treatment_share"])
    n_c, n_t = len(pre) * (1 - share), len(pre) * share
    mde = stats.minimum_detectable_effect(baseline, n_c, n_t, plan.alpha, plan.target_power)
    mei = plan.minimum_effect_of_interest
    needed = stats.sample_size_per_arm(baseline, mei, plan.alpha, plan.target_power)
    weeks = reg["duration_days"] / 7
    return {"pre_period_start": pre_start, "pre_period_end_exclusive": start,
            "baseline_conversion": baseline, "expected_eligible_prospects": len(pre),
            "expected_per_arm": n_c, "planned_weeks": weeks, "mde_absolute": mde,
            "mde_relative": mde / baseline,
            "power_at_mei": stats.power_two_proportions(baseline, mei, n_c, n_t, plan.alpha),
            "per_arm_needed_for_mei": needed,
            "weeks_needed_for_mei": needed / n_c * weeks}


def realized_design(primary: Mapping, plan: ExperimentPlan) -> dict:
    """Sensitivity of the test as run (actual arm sizes, observed control rate)."""
    p_c, n_c, n_t = primary["control"], primary["n_control"], primary["n_treatment"]
    mei = plan.minimum_effect_of_interest
    se_mei = np.sqrt(p_c * (1 - p_c) / n_c + (p_c + mei) * (1 - p_c - mei) / n_t)
    return {"mde_absolute": stats.minimum_detectable_effect(p_c, n_c, n_t, plan.alpha,
                                                            plan.target_power),
            "power_at_mei": stats.power_two_proportions(p_c, mei, n_c, n_t, plan.alpha),
            "retrodesign_at_mei": stats.retrodesign(mei, float(se_mei), plan.alpha)}


def power_curve(planned: Mapping, primary: Mapping, plan: ExperimentPlan) -> pd.DataFrame:
    rows = []
    for lift in plan.power_grid:
        rows.append({
            "absolute_lift": lift,
            "power_planned": stats.power_two_proportions(
                planned["baseline_conversion"], lift, planned["expected_per_arm"],
                planned["expected_eligible_prospects"] - planned["expected_per_arm"],
                plan.alpha),
            "power_realized": stats.power_two_proportions(
                primary["control"], lift, primary["n_control"], primary["n_treatment"],
                plan.alpha)})
    return pd.DataFrame(rows)


def practical_significance(primary: Mapping, mei: float) -> str:
    low, high = primary["ci"]
    if low >= mei:
        return "ci_above_mei"
    if high < mei:
        return "ci_below_mei"
    return "estimate_above_mei" if primary["diff"] >= mei else "estimate_below_mei"


DECISIONS = {
    "ship": "Ship: the lift is significant, its whole CI clears the minimum effect of interest "
            "and every guardrail passes.",
    "ship_and_monitor": "Ship and keep measuring: the lift is significant and every guardrail "
                        "passes, but the CI does not rule out a lift below the minimum effect of "
                        "interest.",
    "hold_guardrail": "Do not ship as tested: conversion improved, but at least one guardrail "
                      "did not pass. Fix the cause and re-test.",
    "hold_no_evidence": "Do not ship on this evidence: the lift is not statistically "
                        "distinguishable from zero.",
    "reject_harm": "Do not ship: conversion fell significantly.",
}


def decide(primary: Mapping, guardrail_results: Mapping[str, Mapping],
           plan: ExperimentPlan) -> dict:
    """Pre-specified decision rule; significance is read from the p-value and cross-checked
    against the CI (they come from the same standard error, so they must agree)."""
    significant = primary["p_value"] < plan.alpha
    excludes_zero = primary["ci"][0] > 0 or primary["ci"][1] < 0
    if significant != excludes_zero:
        raise AssertionError("p-value and confidence interval disagree")
    practical = practical_significance(primary, plan.minimum_effect_of_interest)
    guardrails_pass = all(g["status"] == "pass" for g in guardrail_results.values())
    if not significant:
        code = "hold_no_evidence"
    elif primary["diff"] < 0:
        code = "reject_harm"
    elif not guardrails_pass:
        code = "hold_guardrail"
    else:
        code = "ship" if practical == "ci_above_mei" else "ship_and_monitor"
    return {"statistically_significant": significant, "ci_excludes_zero": excludes_zero,
            "practical_significance": practical, "guardrails_pass": guardrails_pass,
            "recommendation": code, "recommendation_text": DECISIONS[code]}


def business_translation(units: pd.DataFrame, primary: Mapping, secondary: Mapping[str, Mapping],
                         aov: Mapping, reg: Mapping, plan: ExperimentPlan) -> dict:
    """Effects restated per 1,000 eligible prospects and per year of eligible traffic."""
    annual = len(units) * 365 / reg["duration_days"]
    p_c = primary["control"]
    rev_key = f"revenue_{plan.revenue_window_days}d"
    return {
        "annual_eligible_prospects": annual,
        "extra_first_purchases_per_1000": primary["diff"] * 1000,
        "extra_first_purchases_per_1000_ci": [v * 1000 for v in primary["ci"]],
        "extra_first_purchases_per_year": primary["diff"] * annual,
        "extra_first_purchases_per_year_ci": [v * annual for v in primary["ci"]],
        "first_order_revenue_per_1000": secondary["first_order_revenue"]["diff"] * 1000,
        "first_order_revenue_per_1000_ci": [
            v * 1000 for v in secondary["first_order_revenue"]["ci"]],
        "revenue_window_per_1000": secondary[rev_key]["diff"] * 1000,
        "revenue_window_per_1000_ci": [v * 1000 for v in secondary[rev_key]["ci"]],
        # Conversion lift at which first-order revenue per prospect breaks even, given the
        # treatment's observed first-order value.
        "break_even_lift_given_observed_aov": p_c * (aov["control"] / aov["treatment"] - 1),
    }


def arm_funnel(units: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Share of assigned prospects reaching each stage in the window, by arm (mechanism)."""
    rows = []
    for stage in REACH_STAGES:
        r = estimate(units, f"reached_{stage}", "proportion", alpha)
        rows.append({"stage": stage, "control": r["control"], "treatment": r["treatment"],
                     "diff": r["diff"], "ci_low": r["ci"][0], "ci_high": r["ci"][1],
                     "p_value": r["p_value"]})
    # Conditional step (descriptive; conditioning on a post-treatment event).
    started = units.loc[units["reached_checkout_start"]]
    r = estimate(started, "reached_purchase", "proportion", alpha)
    rows.append({"stage": "purchase | checkout_start", "control": r["control"],
                 "treatment": r["treatment"], "diff": r["diff"], "ci_low": r["ci"][0],
                 "ci_high": r["ci"][1], "p_value": r["p_value"]})
    return pd.DataFrame(rows)


def subgroup_effects(units: pd.DataFrame, plan: ExperimentPlan) -> tuple[pd.DataFrame, list]:
    """Exploratory per-segment effects with Holm (FWER) and Benjamini-Hochberg (FDR) adjustment
    across the whole family, Bonferroni-width intervals for display, and a Cochran's Q
    heterogeneity test per dimension (Holm-adjusted across dimensions)."""
    rows = []
    for dim in plan.subgroup_dimensions:
        for level in sorted(units[dim].astype(str).unique()):
            sub = units.loc[units[dim].astype(str) == level]
            n_t = int(sub["treated"].sum())
            if min(n_t, len(sub) - n_t) < plan.subgroup_min_units_per_arm:
                continue
            r = estimate(sub, plan.primary_metric, "proportion", plan.alpha)
            rows.append({"dimension": dim, "level": level, "n_control": r["n_control"],
                         "n_treatment": r["n_treatment"], "control": r["control"],
                         "treatment": r["treatment"], "diff": r["diff"], "se": r["se"],
                         "ci_low": r["ci"][0], "ci_high": r["ci"][1], "p_value": r["p_value"]})
    if not rows:
        raise ValueError("no subgroup has enough units per arm")
    table = pd.DataFrame(rows)
    m = len(table)
    table["p_holm"] = stats.holm(table["p_value"])
    table["q_bh"] = stats.benjamini_hochberg(table["p_value"])
    table["significant_raw"] = table["p_value"] < plan.alpha
    table["significant_holm"] = table["p_holm"] < plan.alpha
    z_bonf = stats.z_critical(plan.alpha / m)
    table["ci_bonferroni_low"] = table["diff"] - z_bonf * table["se"]
    table["ci_bonferroni_high"] = table["diff"] + z_bonf * table["se"]
    heterogeneity = []
    for dim, g in table.groupby("dimension", sort=False):
        if len(g) < 2:
            continue
        q = stats.cochran_q(g["diff"], g["se"])
        heterogeneity.append({"dimension": dim, "levels": len(g), **q})
    p_adj = stats.holm([h["p_value"] for h in heterogeneity])
    for h, p in zip(heterogeneity, p_adj, strict=True):
        h["p_holm"] = float(p)
    return table, heterogeneity


def cumulative_effects(units: pd.DataFrame, plan: ExperimentPlan) -> pd.DataFrame:
    """What a weekly dashboard would have shown: the primary estimate on data available at the
    end of each week (units assigned and conversions observed so far). Only the final,
    pre-planned look is used for the decision."""
    start, end = units["start_date"].iloc[0], units["window_end"].iloc[0]
    rows = []
    week = 1
    while True:
        cut = min(start + pd.Timedelta(weeks=week), end)
        seen = units.loc[units["assigned_at"] < cut]
        y = (seen["customer_since"] < cut) & seen["converted"]
        t = seen["treated"].to_numpy()
        r = stats.two_proportion_test(int(y[~t].sum()), int((~t).sum()), int(y[t].sum()),
                                      int(t.sum()), plan.alpha)
        rows.append({"week": week, "data_through": (cut - pd.Timedelta(days=1)).date(),
                     "units": len(seen), "control": r["control"], "treatment": r["treatment"],
                     "diff": r["diff"], "ci_low": r["ci"][0], "ci_high": r["ci"][1],
                     "p_value": r["p_value"], "nominally_significant": r["p_value"] < plan.alpha})
        if cut >= end:
            break
        week += 1
    return pd.DataFrame(rows)
