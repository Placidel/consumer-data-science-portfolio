"""Section 03 end to end: CLI interface, assignment gate and traceability of documented results."""

from __future__ import annotations

import json
import math

import pytest

from northstar.cli import main
from northstar.conversion import report
from northstar.conversion.experiment import ExperimentPlan
from northstar.conversion.funnel import STEPS
from northstar.paths import CONVERSION_DIR
from northstar.profile import extract_generated_block
from northstar.schema import FUNNEL_STAGES
from northstar.synthetic import generate
from northstar.synthetic import params as p

METRICS = CONVERSION_DIR / "outputs" / "metrics.json"
README = CONVERSION_DIR / "README.md"
FIGURES = ("funnel_stages", "step_conversion_by_segment", "experiment_effects", "power_curve",
           "subgroup_forest", "cumulative_effect")
TABLES = ("funnel_overall", "funnel_prospect_cohort", "funnel_segments", "dropoff_opportunity",
          "experiment_units_summary", "covariate_balance", "experiment_results",
          "experiment_funnel_by_arm", "subgroup_effects", "power_curve", "cumulative_effect")


@pytest.fixture(scope="module")
def small_data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("conv") / "raw"
    assert main(["generate-data", "--seed", "11", "--n-prospects", "4000", "--out", str(out),
                 "--skip-profile"]) == 0
    return out


def _readme(path):
    path.write_text(f"# Test\n\n{report.BEGIN_MARKER}\nstale\n{report.END_MARKER}\n\nend\n")
    return path


def test_conversion_command_writes_traceable_outputs(small_data_dir, tmp_path):
    out, readme = tmp_path / "outputs", _readme(tmp_path / "README.md")
    code = main(["conversion", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(readme), "--no-generate"])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data"] == {"seed": 11, "n_prospects": 4000}
    assert metrics["experiment"]["assignment_audit"]["passed"]
    for name in TABLES:
        assert (out / f"{name}.csv").stat().st_size > 0, name
    for name in FIGURES:
        assert (out / "figures" / f"{name}.png").stat().st_size > 0, name
    block = extract_generated_block(readme, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(metrics)
    assert "stale" not in readme.read_text() and readme.read_text().rstrip().endswith("end")
    # Every written funnel is non-increasing downstream.
    for rows in (*metrics["funnel"]["session_funnel"].values(),
                 metrics["funnel"]["prospect_funnel"]):
        reached = [r["reached"] for r in rows]
        assert reached == sorted(reached, reverse=True)
    for seg in metrics["funnel"]["segments"]:
        reached = [seg[f"reached_{s}"] for s in FUNNEL_STAGES]
        assert reached == sorted(reached, reverse=True)


def test_plan_is_recorded_before_results_and_decision_ignores_subgroups(small_data_dir,
                                                                        tmp_path):
    from northstar.io import load_tables

    tables = load_tables(small_data_dir)
    metrics, _ = report.run_analysis(tables)
    keys = list(metrics)
    assert keys.index("plan") < keys.index("funnel") < keys.index("experiment")
    exp_keys = list(metrics["experiment"])
    assert exp_keys.index("primary") < exp_keys.index("decision") < exp_keys.index("subgroups")
    assert metrics["plan"] == json.loads(json.dumps(report._clean(ExperimentPlan().as_dict())))
    other = report.ConversionConfig(plan=ExperimentPlan(subgroup_dimensions=("region",)))
    alt, _ = report.run_analysis(tables, other)
    for key in ("primary", "guardrails", "secondary", "decision"):
        assert alt["experiment"][key] == metrics["experiment"][key], key
    assert {s["dimension"] for s in alt["experiment"]["subgroups"]} == {"region"}


def test_conversion_command_refuses_to_write_when_assignment_audit_fails(
        small_data_dir, tmp_path, monkeypatch):
    def failing_audit(*args, **kwargs):
        return {"passed": False, "checks": {"no_sample_ratio_mismatch": False}, "details": {}}

    monkeypatch.setattr(report.ex, "assignment_audit", failing_audit)
    out = tmp_path / "outputs"
    code = main(["conversion", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(tmp_path / "missing.md"), "--no-generate"])
    assert code == 1
    assert not out.exists()


def test_conversion_command_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="northstar generate-data"):
        main(["conversion", "--data-dir", str(tmp_path / "none"), "--no-generate"])


# ---------------------------------------------------------------- committed results
@pytest.fixture(scope="module")
def committed():
    return json.loads(METRICS.read_text())


def test_readme_results_block_matches_committed_metrics(committed):
    block = extract_generated_block(README, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(committed)


def test_committed_run_uses_the_default_data_and_plan(committed):
    assert committed["data"] == {"seed": p.DEFAULT_SEED, "n_prospects": p.DEFAULT_N_PROSPECTS}
    assert committed["plan"] == json.loads(json.dumps(report._clean(ExperimentPlan().as_dict())))
    assert committed["experiment"]["assignment_audit"]["passed"]
    assert committed["experiment"]["registry"]["primary_metric"].startswith(
        committed["plan"]["primary_label"])


def test_statistical_conclusions_match_intervals_and_p_values(committed):
    e = committed["experiment"]
    alpha = committed["plan"]["alpha"]
    prim, dec = e["primary"], e["decision"]
    excludes = prim["ci"][0] > 0 or prim["ci"][1] < 0
    assert dec["statistically_significant"] == (prim["p_value"] < alpha) == excludes
    for g in e["guardrails"].values():
        low, high = g["ci"] if g["scale"] == "absolute" else g["relative_ci"]
        expected = ("pass" if low >= -g["margin"] else
                    "fail" if high < -g["margin"] else "inconclusive")
        assert g["status"] == expected
    for s in e["subgroups"]:
        assert s["significant_raw"] == (s["p_value"] < alpha)
        assert s["significant_holm"] == (s["p_holm"] < alpha)
        assert (s["ci_low"] > 0 or s["ci_high"] < 0) == s["significant_raw"]
    for c in e["cumulative"]:
        assert c["nominally_significant"] == (c["ci_low"] > 0 or c["ci_high"] < 0)
    assert e["cumulative"][-1]["diff"] == pytest.approx(prim["diff"])


def test_readme_separates_statistical_from_business_significance():
    text = README.read_text()
    stat = text.index("### Statistical significance")
    biz = text.index("### Business significance")
    assert stat < biz
    for phrase in ("not causal", "(**assumed**)",
                   "Statistical significance says the effect is probably not zero",
                   "single fixed-horizon look is the only valid test"):
        assert phrase in text, phrase


def test_prose_claims_in_the_readme_hold_for_the_committed_run(committed):
    """The narrative sections of the README make these claims; keep them true."""
    f, e = committed["funnel"], committed["experiment"]
    prospect = {r["stage"]: r for r in f["session_funnel"]["prospect"]}
    customer = {r["stage"]: r for r in f["session_funnel"]["customer"]}
    # About one prospect session in nine purchases; about three in four leads have not bought
    # within 30 days.
    assert 0.09 < prospect["purchase"]["share_of_start"] < 0.13
    assert 0.2 < f["prospect_funnel"][-1]["share_of_start"] < 0.3
    # Biggest leak: product view -> add to cart (largest loss, over half of losses, lowest rate).
    assert f["largest_loss_step"] == f["lowest_step_conversion"] == "product_view->add_to_cart"
    assert prospect["add_to_cart"]["share_of_all_losses"] > 0.5
    assert all(customer[s]["step_conversion"] > prospect[s]["step_conversion"]
               for s in FUNNEL_STAGES[1:])
    seg = {(r["dimension"], r["level"]): r for r in f["segments"]}
    first, third = seg[("visit_number", "1st visit")], seg[("visit_number", "3rd+ visit")]
    assert third["rate_product_view->add_to_cart"] > 1.5 * first["rate_product_view->add_to_cart"]
    weakest = {w["step"]: w for w in f["weakest_by_step"]}
    assert weakest["add_to_cart->checkout_start"]["level"] == "1st visit"
    assert weakest["checkout_start->purchase"]["level"] == "mobile"
    traffic = {k[1]: v for k, v in seg.items() if k[0] == "traffic_source"}
    assert max(traffic, key=lambda k: traffic[k]["session_to_purchase"]) == "email"
    mobile, desktop = seg[("device_type", "mobile")], seg[("device_type", "desktop")]
    for step in STEPS[:2]:
        assert abs(mobile[f"rate_{step}"] - desktop[f"rate_{step}"]) < 0.015, step
    for step in STEPS[2:]:
        assert mobile[f"rate_{step}"] < desktop[f"rate_{step}"] - 0.03, step
    opp = {(r["level"], r["step"]): r["purchase_equivalents"] for r in f["opportunity"]}
    assert opp[("mobile", "add_to_cart->checkout_start")] > opp[("mobile",
                                                                 "checkout_start->purchase")] > 0

    # Statistical significance and its caveats.
    prim, dec, rob = e["primary"], e["decision"], e["robustness"]
    assert dec["statistically_significant"] and prim["diff"] > 0
    assert rob["p_value_pooled"] < 0.05 and rob["permutation_p_value"] < 0.05
    assert rob["regression_adjusted"]["ci"][0] > 0
    assert rob["regression_adjusted"]["diff"] < prim["diff"]
    assert rob["regression_adjusted"]["se_ratio_vs_unadjusted"] > 0.95  # "a few percent"
    aa = rob["aa_control_arm"]
    assert aa["false_positive_rate_ci"][0] <= 0.05 <= aa["false_positive_rate_ci"][1]
    assert e["balance"]["covariates_over_limit"] == 0
    pl, mei = e["planning"], committed["plan"]["minimum_effect_of_interest"]
    assert 0.025 < pl["mde_absolute"] < 0.035 and 0.35 < pl["power_at_mei"] < 0.45
    assert pl["weeks_needed_for_mei"] > 2 * pl["planned_weeks"]
    assert round(pl["weeks_needed_for_mei"]) == 31
    assert 1.4 < e["realized_design"]["retrodesign_at_mei"]["exaggeration_ratio"] < 1.65
    cum = e["cumulative"]
    assert cum[0]["nominally_significant"] and cum[1]["nominally_significant"]
    assert cum[0]["diff"] > 1.5 * prim["diff"]
    assert sum(not c["nominally_significant"] for c in cum) > len(cum) / 2
    assert cum[-1]["nominally_significant"]
    arm = {a["stage"]: a for a in e["arm_funnel"]}
    assert abs(arm["checkout_start"]["diff"] - prim["diff"]) < 0.005
    cond = arm["purchase | checkout_start"]
    assert cond["diff"] > 0 and cond["p_value"] > 0.05

    # Business significance.
    assert dec["practical_significance"] == "estimate_above_mei"
    assert prim["ci"][0] < mei / 2
    assert e["guardrails"]["first_order_value"]["status"] == "fail"
    assert e["guardrails"]["reached_checkout_start"]["status"] == "pass"
    assert e["basket_items"]["ci"][1] < 0
    for s in e["secondary"].values():
        assert s["diff"] < 0 and s["p_value"] > 0.05
    b = e["business"]
    assert b["break_even_lift_given_observed_aov"] > 1.3 * prim["diff"]
    assert 100 < b["extra_first_purchases_per_year"] < 1000
    assert dec["recommendation"] == "hold_guardrail"
    raw = {(s["dimension"], s["level"]) for s in e["subgroups"] if s["significant_raw"]}
    assert raw == {("device_type", "tablet"), ("acquisition_channel", "referral"),
                   ("device_type", "mobile"), ("lead_type", "new lead")}
    assert not any(s["significant_holm"] for s in e["subgroups"])
    assert all(h["p_value"] > 0.05 for h in e["heterogeneity"])
    assert len(e["subgroups"]) == 12


def _assert_close(fresh, committed, path="metrics"):
    if isinstance(committed, dict):
        assert set(fresh) == set(committed), path
        for k in committed:
            _assert_close(fresh[k], committed[k], f"{path}.{k}")
    elif isinstance(committed, list):
        assert len(fresh) == len(committed), path
        for i, (x, y) in enumerate(zip(fresh, committed, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(committed, float) and not isinstance(committed, bool):
        assert math.isclose(fresh, committed, rel_tol=1e-6, abs_tol=1e-6), path
    else:
        assert fresh == committed, path


@pytest.mark.slow
def test_committed_metrics_are_reproduced_from_default_generation(committed):
    tables = generate(seed=p.DEFAULT_SEED, n_prospects=p.DEFAULT_N_PROSPECTS)
    fresh, _ = report.run_analysis(tables)
    expected = {k: v for k, v in committed.items() if k != "data"}
    _assert_close(json.loads(json.dumps(fresh)), expected)
