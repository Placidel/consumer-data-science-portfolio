"""Section 03 experiment: stable assignment, design audit, outcome windows and decision rules."""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from northstar.conversion import experiment as ex
from northstar.synthetic import assignment_bucket, experiment_variant

PLAN = ex.ExperimentPlan()


@pytest.fixture(scope="module")
def units(tables):
    return ex.build_units(tables, PLAN)


# ---------------------------------------------------------------- assignment
def test_assignment_is_deterministic_and_order_independent():
    ids = [f"P{i:06d}" for i in range(2000)]
    first = [experiment_variant("EXP001", pid, 0.5) for pid in ids]
    again = [experiment_variant("EXP001", pid, 0.5) for pid in reversed(ids)][::-1]
    assert first == again


def test_assignment_buckets_are_uniform_and_salted_per_experiment():
    ids = [f"P{i:06d}" for i in range(20_000)]
    b1 = np.array([assignment_bucket("EXP001", pid) for pid in ids])
    b2 = np.array([assignment_bucket("EXP002", pid) for pid in ids])
    assert ((b1 >= 0) & (b1 < 1)).all()
    counts, _ = np.histogram(b1, bins=10, range=(0, 1))
    assert (np.abs(counts - 2000) < 4 * np.sqrt(2000)).all()
    # A different experiment id re-randomizes: arms are uncorrelated across experiments.
    assert abs(np.corrcoef(b1 < 0.5, b2 < 0.5)[0, 1]) < 0.03


def test_ramping_up_treatment_share_never_moves_a_treated_unit_back_to_control():
    ids = [f"P{i:06d}" for i in range(5000)]
    shares = (0.05, 0.1, 0.25, 0.5, 0.9)
    arms = {s: np.array([experiment_variant("EXP001", pid, s) == "treatment" for pid in ids])
            for s in shares}
    for lo, hi in pairwise(shares):
        assert (arms[hi] | ~arms[lo]).all()  # treated at lo => treated at hi
        assert abs(arms[hi].mean() - hi) < 0.03


def test_eligible_population_reproduces_the_assignment_log(tables):
    reg = ex.registry(tables, PLAN.experiment_id)
    eligible = ex.eligible_population(tables, reg["start_date"], reg["window_end"])
    a = tables["experiment_assignments"]
    merged = eligible.merge(a, on="prospect_id", how="outer", indicator=True)
    assert (merged["_merge"] == "both").all()
    assert (merged["first_eligible_session"] == merged["assigned_at"]).all()


def test_assignment_audit_passes_on_generated_data(tables, units):
    audit = ex.assignment_audit(tables, units, PLAN)
    assert audit["passed"], audit
    assert audit["details"]["control"] + audit["details"]["treatment"] == len(units)


def test_assignment_audit_catches_a_flipped_arm(tables, units):
    broken = units.copy()
    broken.loc[0, "variant"] = "control" if broken.loc[0, "variant"] == "treatment" else \
        "treatment"
    broken["treated"] = broken["variant"] == "treatment"
    audit = ex.assignment_audit(tables, broken, PLAN)
    assert not audit["passed"]
    assert not audit["checks"]["variant_matches_hash_rule"]


def test_assignment_audit_catches_sample_ratio_mismatch_from_lost_units(tables, units):
    """Losing a third of the treatment arm (e.g. a logging bug) must trip the SRM alarm."""
    treated = units.index[units["treated"]]
    lost = units.drop(index=treated[: len(treated) // 3])
    audit = ex.assignment_audit(tables, lost, PLAN)
    assert not audit["checks"]["no_sample_ratio_mismatch"]
    assert not audit["checks"]["every_eligible_prospect_assigned_at_first_exposure"]
    assert not audit["passed"]


def test_arm_sizes_are_balanced_and_covariates_look_randomized(units):
    share = units["treated"].mean()
    assert abs(share - 0.5) < 4 * np.sqrt(0.25 / len(units))
    balance = ex.covariate_balance(units)
    assert set(balance["covariate"]) >= {"prior_sessions", "days_since_lead",
                                         "device_type=mobile", "lead_type=new lead"}
    assert balance["smd"].abs().max() < 0.2  # small sample; the full run is checked at 0.1


# ---------------------------------------------------------------- outcomes
def test_unit_outcomes_respect_assignment_and_window(tables, units):
    reg = ex.registry(tables, PLAN.experiment_id)
    conv = units.loc[units["converted"]]
    assert (conv["customer_since"] >= conv["assigned_at"]).all()
    assert (conv["customer_since"] < reg["window_end"]).all()
    later = units.loc[~units["converted"] & units["customer_since"].notna()]
    assert (later["customer_since"] >= reg["window_end"]).all()
    assert units.loc[~units["converted"], "first_order_value"].isna().all()
    assert (units.loc[units["converted"], "first_order_value"] > 0).all()
    assert (units["revenue_90d"] >= units["first_order_revenue"] - 1e-9).all()
    # Reach is nested and purchase reach equals conversion.
    reach = units[[f"reached_{s}" for s in ex.REACH_STAGES]].to_numpy()
    assert (np.diff(reach.astype(int), axis=1) <= 0).all()
    assert (units["reached_purchase"] == units["converted"]).mean() > 0.99


def test_covariates_use_only_pre_assignment_sessions(tables):
    """Deleting every session after assignment changes outcomes but not covariates."""
    reg = ex.registry(tables, PLAN.experiment_id)
    full = ex.build_units(tables, PLAN)
    t = dict(tables)
    first_assign = full.set_index("prospect_id")["assigned_at"]
    s = t["sessions"]
    cutoff = s["prospect_id"].map(first_assign)
    t["sessions"] = s.loc[cutoff.isna() | (s["session_start"] <= cutoff)]
    t["funnel_events"] = t["funnel_events"].loc[
        t["funnel_events"]["session_id"].isin(t["sessions"]["session_id"])]
    trimmed = ex.build_units(t, PLAN)
    cols = ["prior_sessions", "lead_type", "days_since_lead", "device_type"]
    pd.testing.assert_frame_equal(full[cols], trimmed[cols])
    # ... while in-window funnel reach (an outcome) can only shrink, and does.
    assert (trimmed["reached_add_to_cart"] <= full["reached_add_to_cart"]).all()
    assert (trimmed["reached_add_to_cart"] < full["reached_add_to_cart"]).any()
    assert reg["window_end"] > full["assigned_at"].max()


def test_conversion_after_the_window_is_not_counted(mutable_tables):
    t = mutable_tables
    reg = ex.registry(t, PLAN.experiment_id)
    base = ex.build_units(t, PLAN)
    pid = base.loc[base["converted"], "prospect_id"].iloc[0]
    c = t["customers"]
    c.loc[c["prospect_id"] == pid, "customer_since"] = reg["window_end"] + pd.Timedelta(hours=1)
    moved = ex.build_units(t, PLAN).set_index("prospect_id")
    assert not moved.loc[pid, "converted"]
    assert moved.loc[pid, "first_order_revenue"] == 0
    assert moved["converted"].sum() == base["converted"].sum() - 1


# ---------------------------------------------------------------- plan and decisions
def test_plan_matches_the_registry(tables):
    reg = ex.registry(tables, PLAN.experiment_id)
    assert reg["primary_metric"].startswith(PLAN.primary_label)
    for g in PLAN.guardrails:
        assert g.label.lower() in reg["guardrail_metrics"].lower(), g.metric
    assert reg["duration_days"] == 84


def test_primary_result_does_not_depend_on_the_subgroup_family(units):
    """The primary metric is fixed before subgroup exploration; changing subgroups is inert."""
    a = ex.estimate(units, PLAN.primary_metric, "proportion", PLAN.alpha)
    other = dataclasses.replace(PLAN, subgroup_dimensions=("region",))
    b = ex.estimate(units, other.primary_metric, "proportion", other.alpha)
    assert a == b
    table, het = ex.subgroup_effects(units, other)
    assert set(table["dimension"]) == {"region"} and len(het) == 1


def test_subgroup_family_is_multiplicity_adjusted(units):
    table, het = ex.subgroup_effects(units, PLAN)
    assert (table["p_holm"] >= table["p_value"] - 1e-12).all()
    assert (table["q_bh"] >= table["p_value"] - 1e-12).all()
    assert (table["q_bh"] <= table["p_holm"] + 1e-12).all()
    width = table["ci_bonferroni_high"] - table["ci_bonferroni_low"]
    assert (width > table["ci_high"] - table["ci_low"]).all()
    assert (table[["n_control", "n_treatment"]] >= PLAN.subgroup_min_units_per_arm).all().all()
    assert {h["dimension"] for h in het} <= set(PLAN.subgroup_dimensions)


def _primary(diff: float, lo: float, hi: float, p: float) -> dict:
    return {"diff": diff, "ci": [lo, hi], "p_value": p}


@pytest.mark.parametrize(("primary", "guardrail", "expected"), [
    (_primary(0.03, 0.025, 0.035, 0.001), "pass", "ship"),
    (_primary(0.03, 0.005, 0.055, 0.01), "pass", "ship_and_monitor"),
    (_primary(0.03, 0.005, 0.055, 0.01), "fail", "hold_guardrail"),
    (_primary(0.03, 0.005, 0.055, 0.01), "inconclusive", "hold_guardrail"),
    (_primary(0.01, -0.01, 0.03, 0.3), "pass", "hold_no_evidence"),
    (_primary(-0.03, -0.05, -0.01, 0.004), "pass", "reject_harm"),
])
def test_decision_rule(primary, guardrail, expected):
    result = ex.decide(primary, {"g": {"status": guardrail}}, PLAN)
    assert result["recommendation"] == expected
    assert result["statistically_significant"] == result["ci_excludes_zero"]


def test_decision_rule_rejects_inconsistent_inputs():
    with pytest.raises(AssertionError):
        ex.decide(_primary(0.03, -0.01, 0.07, 0.01), {}, PLAN)


def test_practical_significance_categories():
    mei = 0.02
    assert ex.practical_significance(_primary(0.03, 0.021, 0.04, 0), mei) == "ci_above_mei"
    assert ex.practical_significance(_primary(0.03, 0.01, 0.05, 0), mei) == "estimate_above_mei"
    assert ex.practical_significance(_primary(0.01, -0.01, 0.03, 0), mei) == "estimate_below_mei"
    assert ex.practical_significance(_primary(0.01, 0.0, 0.019, 0), mei) == "ci_below_mei"


@pytest.mark.parametrize(("ci", "status"), [
    ([-0.04, 0.02], "pass"), ([-0.08, -0.06], "fail"), ([-0.07, 0.01], "inconclusive")])
def test_guardrail_status_uses_the_whole_interval(ci, status):
    g = ex.Guardrail("first_order_value", "AOV", "mean", "relative", 0.05)
    assert g.status({"relative_ci": ci}) == status
    a = ex.Guardrail("reached_checkout_start", "CS", "proportion", "absolute", 0.05)
    assert a.status({"ci": ci}) == status


def test_planning_uses_only_the_pre_period(tables):
    """Power analysis must be computable at launch: truncating everything from the start date
    onwards leaves it unchanged."""
    reg = ex.registry(tables, PLAN.experiment_id)
    planned = ex.planning(tables, PLAN)
    t = dict(tables)
    t["sessions"] = t["sessions"].loc[t["sessions"]["session_start"] < reg["start_date"]]
    t["customers"] = t["customers"].loc[t["customers"]["customer_since"] < reg["start_date"]]
    assert ex.planning(t, PLAN) == planned
    assert planned["pre_period_end_exclusive"] == reg["start_date"]
    assert (reg["start_date"] - planned["pre_period_start"]).days == reg["duration_days"]
    assert 0 < planned["baseline_conversion"] < 1
    assert planned["mde_absolute"] > 0 and 0 < planned["power_at_mei"] < 1
