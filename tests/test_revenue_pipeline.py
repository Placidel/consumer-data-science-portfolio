"""Section 04 end to end: CLI interface, leakage gate and traceability of documented results."""

from __future__ import annotations

import json
import math

import pytest

from northstar.cli import main
from northstar.paths import REVENUE_DIR
from northstar.profile import extract_generated_block
from northstar.revenue import report
from northstar.revenue.actions import ACTIONS, ActionRules
from northstar.revenue.models import BASELINES, MODEL_NAMES
from northstar.revenue.scenarios import GrowthAssumptions
from northstar.synthetic import generate
from northstar.synthetic import params as p

METRICS = REVENUE_DIR / "outputs" / "metrics.json"
README = REVENUE_DIR / "README.md"
FIGURES = ("revenue_gains", "decile_calibration", "value_tiers", "growth_scenario",
           "feature_importance")
TABLES = ("model_comparison", "decile_table", "gains_curve", "revenue_capture",
          "feature_importance", "value_tiers", "value_migration", "next_best_action",
          "nba_priority_list", "growth_scenarios", "growth_value_curve", "scenario_sensitivity",
          "planned_vs_realized")


@pytest.fixture(scope="module")
def small_data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("rev") / "raw"
    assert main(["generate-data", "--seed", "11", "--n-prospects", "4000", "--out", str(out),
                 "--skip-profile"]) == 0
    return out


def _readme(path):
    path.write_text(f"# Test\n\n{report.BEGIN_MARKER}\nstale\n{report.END_MARKER}\n\nend\n")
    return path


def test_revenue_command_writes_traceable_outputs(small_data_dir, tmp_path):
    out, readme = tmp_path / "outputs", _readme(tmp_path / "README.md")
    code = main(["revenue", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(readme), "--no-generate"])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data"] == {"seed": 11, "n_prospects": 4000}
    assert metrics["leakage_audit"]["passed"]
    assert {m["model"] for m in metrics["models"]} == set(MODEL_NAMES)
    for name in TABLES:
        assert (out / f"{name}.csv").stat().st_size > 0, name
    for name in FIGURES:
        assert (out / "figures" / f"{name}.png").stat().st_size > 0, name
    block = extract_generated_block(readme, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(metrics)
    assert "stale" not in readme.read_text() and readme.read_text().rstrip().endswith("end")
    # Scenario assumptions are reported with the results and labelled as assumed.
    assert metrics["scenario"]["assumptions"] == GrowthAssumptions().as_dict()
    assert "(assumed)" in block and "not observed facts" in block and "(observed)" in block
    assert metrics["next_best_action"]["rules"] == ActionRules().as_dict()
    # Every holdout customer gets exactly one action and one value tier.
    holdout_rows = next(s["customer_runs"] for s in metrics["splits"] if s["split"] == "holdout")
    assert sum(r["customers"] for r in metrics["next_best_action"]["actions"]) == holdout_rows
    assert sum(r["customers"] for r in metrics["segmentation"]["tiers"]) == holdout_rows
    assert {r["group"] for r in metrics["next_best_action"]["actions"]} <= set(ACTIONS)


def test_revenue_command_refuses_to_write_when_leakage_audit_fails(
        small_data_dir, tmp_path, monkeypatch):
    def failing_audit(*args, **kwargs):
        return {"passed": False, "checks": {"historical_spend_excludes_target_period": False},
                "details": {}}

    monkeypatch.setattr(report, "leakage_audit", failing_audit)
    out = tmp_path / "outputs"
    code = main(["revenue", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(tmp_path / "missing.md"), "--no-generate"])
    assert code == 1
    assert not out.exists()


def test_revenue_command_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="northstar generate-data"):
        main(["revenue", "--data-dir", str(tmp_path / "none"), "--no-generate"])


# ---------------------------------------------------------------- committed results
@pytest.fixture(scope="module")
def committed():
    return json.loads(METRICS.read_text())


def test_readme_results_block_matches_committed_metrics(committed):
    block = extract_generated_block(README, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(committed)


def test_committed_run_uses_the_default_design_and_assumptions(committed):
    assert committed["data"] == {"seed": p.DEFAULT_SEED, "n_prospects": p.DEFAULT_N_PROSPECTS}
    assert committed["config"]["holdout_runs"] == ["2025-07-01"]
    assert committed["config"]["horizon_days"] == 180
    assert committed["leakage_audit"]["passed"]
    assert committed["leakage_audit"]["checks"]["historical_spend_excludes_target_period"]
    assert committed["scenario"]["assumptions"] == GrowthAssumptions().as_dict()
    assert committed["next_best_action"]["rules"] == ActionRules().as_dict()


def test_readme_frames_results_as_prioritization_not_causal_proof():
    text = README.read_text()
    for phrase in ("Prioritization ≠ incremental revenue",
                   "Nothing in this section is causal proof", "prioritization rules",
                   "assumptions, not observed facts", "not measured returns", "(assumed)",
                   "randomized"):
        assert phrase in text, phrase


def test_prose_claims_in_the_readme_hold_for_the_committed_run(committed):
    """The narrative sections of the README make these claims; keep them true."""
    models = {m["model"]: m for m in committed["models"]}
    champion = committed["champion"]
    assert champion == "bgnbd_gamma_gamma"
    deltas = committed["champion_vs_models"]
    # Champion beats both baselines on RMSE, Gini and top-10% capture; intervals exclude zero.
    for ref in BASELINES:
        d = deltas[f"{champion}_minus_{ref}"]
        assert d["rmse_ci"][1] < 0 and d["normalized_gini_ci"][0] > 0
        assert d["capture_top10_ci"][0] > 0, ref
    # ... also within the active base.
    active = committed["active_base"]
    for ref in BASELINES:
        assert active[champion]["normalized_gini"] > active[ref]["normalized_gini"]
        assert active[champion]["rmse"] < active[ref]["rmse"]
    # Gradient boosting ranks the holdout better; its RMSE edge is not significant.
    gbm = deltas[f"{champion}_minus_gradient_boosting"]
    assert gbm["normalized_gini_ci"][1] < 0
    assert gbm["rmse_ci"][0] < 0 < gbm["rmse_ci"][1]
    assert models["gradient_boosting"]["validation_rmse"] > models[champion]["validation_rmse"]
    fit = next(s for s in committed["splits"] if s["split"] == "fit")
    assert 45 <= fit["median_tenure_days"] <= 75 and fit["runs"] == 4
    top4 = {r["feature"] for r in committed["feature_importance"][:4]}
    assert top4 == {"clv_expected_revenue", "browse_sessions_90d", "days_since_last_order",
                    "mean_days_between_orders"}
    # Seasonality: second-half windows under-predicted, the post-holiday window over-predicted
    # by about a quarter; every model swings the same way; plans are conservative on holdout.
    for r in committed["clv_level_by_run"]:
        if r["split"] == "validation":
            assert 0.2 < r["bias"] < 0.3
        else:
            assert int(r["label_window_end"][5:7]) >= 9 and r["bias"] < 0, r["run"]
    for m in models.values():
        assert m["validation_bias"] > 0 and m["holdout_bias"] < m["validation_bias"] - 0.15
    plan = committed["scenario"]["planned_vs_realized"]
    assert all(r["planned_baseline_revenue"] < r["realized_baseline_revenue"] for r in plan)
    # Concentration.
    capture = {(r["share_targeted"], r["policy"]): r for r in committed["revenue_capture"]}
    assert capture[(0.1, "oracle")]["share_of_revenue"] > 0.5
    assert 3.5 < capture[(0.1, champion)]["lift"] < 4.5
    assert capture[(0.2, champion)]["share_of_revenue"] > 0.5
    holdout = next(s for s in committed["splits"] if s["split"] == "holdout")
    assert holdout["buyer_rate"] < 0.5
    assert all(0.4 < s["buyer_rate"] < 0.6 for s in committed["splits"])
    tiers = {r["group"]: r for r in committed["segmentation"]["tiers"]}
    assert tiers["Bottom 50%"]["share_of_actual"] < 0.1
    # Rising customers out-spend fading ones despite lower past spend.
    mig = {r["group"]: r for r in committed["segmentation"]["migration"]}
    assert mig["rising"]["mean_actual"] > mig["fading"]["mean_actual"]
    assert mig["rising"]["mean_revenue_180d"] < mig["fading"]["mean_revenue_180d"]
    # Action groups are ordered alike on predicted and realized revenue; rule checks hold.
    acts = committed["next_best_action"]["actions"]
    by_pred = [r["group"] for r in sorted(acts, key=lambda r: -r["mean_predicted"])]
    by_real = [r["group"] for r in sorted(acts, key=lambda r: -r["mean_actual"])]
    assert by_pred == by_real
    groups = {r["group"]: r for r in acts}
    assert groups["retention_save"]["buyer_rate"] < 0.5 * mig["core"]["buyer_rate"]
    cr = committed["next_best_action"]["category_rules"]
    assert cr["featured_category_hit_rate"] > cr["best_seller_hit_rate"] + 0.05
    assert cr["suggested_new_category_hit_rate"] > cr["random_new_category_hit_rate"] + 0.05
    # Scenarios under the stated assumptions.
    s = committed["scenario"]
    by_depth: dict[float, dict[str, dict]] = {}
    for r in s["policies"]:
        by_depth.setdefault(r["depth"], {})[r["policy"]] = r
    for depth, pol in by_depth.items():
        assert pol["random"]["net_value"] < 0, depth
        assert all(pol[champion]["net_value"] > pol[b]["net_value"] for b in BASELINES), depth
        assert pol[champion]["break_even_uplift"] < 0.6 * pol["random"]["break_even_uplift"]
        if depth <= 0.1:
            assert all(pol[m]["net_value"] > 0 for m in MODEL_NAMES), depth
    sens = [r for r in s["sensitivity"] if r["uplift"] == 0.02 and r["perk_cost"] == 15.0]
    assert sens and all(r["net_value"] < 0 for r in sens)


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
        assert math.isclose(fresh, committed, rel_tol=1e-3, abs_tol=2e-3), path
    else:
        assert fresh == committed, path


@pytest.mark.slow
def test_committed_metrics_are_reproduced_from_default_generation(committed):
    tables = generate(seed=p.DEFAULT_SEED, n_prospects=p.DEFAULT_N_PROSPECTS)
    fresh, _ = report.run_analysis(tables)
    expected = {k: v for k, v in committed.items() if k != "data"}
    _assert_close(json.loads(json.dumps(fresh)), expected)
