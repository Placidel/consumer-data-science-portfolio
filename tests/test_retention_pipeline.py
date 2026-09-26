"""Section 02 end to end: CLI interface, leakage gate and traceability of documented results."""

from __future__ import annotations

import json
import math

import pytest

from northstar.cli import main
from northstar.paths import RETENTION_DIR
from northstar.profile import extract_generated_block
from northstar.retention import report
from northstar.retention.models import BASELINES, LEARNED
from northstar.retention.simulation import RetentionAssumptions
from northstar.synthetic import generate
from northstar.synthetic import params as p

METRICS = RETENTION_DIR / "outputs" / "metrics.json"
README = RETENTION_DIR / "README.md"
FIGURES = ("gains_curve", "decile_churn", "calibration", "retention_value_curve",
           "shap_importance")
TABLES = ("model_comparison", "calibration", "targeting_depths", "decile_table", "gains_curve",
          "segment_drivers", "feature_importance", "logistic_coefficients",
          "retention_simulation", "retention_value_curve", "roi_sensitivity",
          "reason_codes_sample")


@pytest.fixture(scope="module")
def small_data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("ret") / "raw"
    assert main(["generate-data", "--seed", "11", "--n-prospects", "4000", "--out", str(out),
                 "--skip-profile"]) == 0
    return out


def _readme(path):
    path.write_text(f"# Test\n\n{report.BEGIN_MARKER}\nstale\n{report.END_MARKER}\n\nend\n")
    return path


def test_retention_command_writes_traceable_outputs(small_data_dir, tmp_path):
    out, readme = tmp_path / "outputs", _readme(tmp_path / "README.md")
    code = main(["retention", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(readme), "--no-generate"])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data"] == {"seed": 11, "n_prospects": 4000}
    assert metrics["leakage_audit"]["passed"]
    assert {m["model"] for m in metrics["models"]} == {*BASELINES, *LEARNED}
    for name in TABLES:
        assert (out / f"{name}.csv").stat().st_size > 0, name
    for name in FIGURES:
        assert (out / "figures" / f"{name}.png").stat().st_size > 0, name
    block = extract_generated_block(readme, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(metrics)
    assert "stale" not in readme.read_text() and readme.read_text().rstrip().endswith("end")
    # The simulation reports its assumptions alongside the results, labelled as assumed.
    assert metrics["simulation"]["assumptions"] == RetentionAssumptions().as_dict()
    assert "(assumed)" in block and "not observed facts" in block
    policies = {r["policy"] for r in metrics["simulation"]["policies"]}
    assert policies == {"random", *BASELINES, "risk_ranked", "value_ranked"}


def test_retention_command_refuses_to_write_when_leakage_audit_fails(
        small_data_dir, tmp_path, monkeypatch):
    def failing_audit(*args, **kwargs):
        return {"passed": False, "checks": {"labels_only_from_prediction_window": False},
                "details": {}}

    monkeypatch.setattr(report, "leakage_audit", failing_audit)
    out = tmp_path / "outputs"
    code = main(["retention", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(tmp_path / "missing.md"), "--no-generate"])
    assert code == 1
    assert not out.exists()


def test_retention_command_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="northstar generate-data"):
        main(["retention", "--data-dir", str(tmp_path / "none"), "--no-generate"])


# ---------------------------------------------------------------- committed results
@pytest.fixture(scope="module")
def committed():
    return json.loads(METRICS.read_text())


def test_readme_results_block_matches_committed_metrics(committed):
    block = extract_generated_block(README, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(committed)


def test_committed_run_uses_the_default_data_design_and_assumptions(committed):
    assert committed["data"] == {"seed": p.DEFAULT_SEED, "n_prospects": p.DEFAULT_N_PROSPECTS}
    assert committed["config"]["holdout_runs"][0] == "2025-07-01"
    assert committed["config"]["horizon_days"] == 90
    assert committed["config"]["active_days"] == 180
    assert committed["leakage_audit"]["passed"]
    assert committed["simulation"]["assumptions"] == RetentionAssumptions().as_dict()


def test_readme_distinguishes_prediction_from_causal_claims():
    text = README.read_text()
    for phrase in ("Prediction ≠ causation", "associations, not effects",
                   "assumptions, not observed facts", "not measured returns", "(assumed)"):
        assert phrase in text, phrase


def test_prose_claims_in_the_readme_hold_for_the_committed_run(committed):
    """The narrative sections of the README make these claims; keep them true."""
    models = {m["model"]: m for m in committed["models"]}
    champion = committed["champion"]
    assert champion in LEARNED
    # Champion beats both rule baselines with intervals excluding zero.
    assert set(committed["champion_vs_baselines"]) == {f"{champion}_minus_{b}" for b in BASELINES}
    for delta in committed["champion_vs_baselines"].values():
        assert delta["roc_auc_ci"][0] > 0 and delta["average_precision_ci"][0] > 0
    # ... and at every targeting depth.
    for depth in committed["config"]["depths"]:
        prec = {r["model"]: r["precision"] for r in committed["targeting_depths"]
                if r["depth"] == depth}
        assert all(prec[champion] > prec[b] for b in ("random", *BASELINES)), depth
    # Learned models are well calibrated; per-run gaps are small and largest (over) at the end.
    for n in LEARNED:
        assert models[n]["holdout_ece"] < 0.03
        assert 0.9 < models[n]["holdout_calibration_slope"] < 1.1
    gaps = [r["mean_predicted"] - r["observed_churn_rate"] for r in committed["champion_per_run"]]
    assert all(abs(g) < 0.04 for g in gaps)
    assert gaps[-1] > 0 and abs(gaps[-1]) == max(abs(g) for g in gaps)
    # Risk is concentrated.
    assert committed["deciles"][0]["churn_rate"] > 0.8
    assert committed["deciles"][-1]["churn_rate"] < 0.15
    # Drivers: recency and 90-day browsing are top-3 for both learned models.
    for n in LEARNED:
        ranked = [r["feature"] for r in sorted(committed["feature_importance"],
                                               key=lambda r: -r[f"shap_{n}"])]
        assert {"days_since_last_order", "browse_sessions_90d"} <= set(ranked[:3]), n
    seg = {(r["segment"], r["level"]): r for r in committed["segment_drivers"]}
    plus, non = seg[("plus_membership", "active member")], seg[("plus_membership", "non-member")]
    assert plus["churn_rate"] < 0.5 * non["churn_rate"]
    low, rest = seg[("low_csat_contact_180d", "yes")], seg[("low_csat_contact_180d", "no")]
    assert low["churn_rate"] < rest["churn_rate"]  # confounded raw association
    assert low["churn_rate"] > low["mean_predicted"]  # but more churn than the model expects
    assert low["share_of_base"] < 0.03
    # Simulation: churners carry well under the margin of customers who keep buying.
    s = committed["simulation"]
    assert s["holdout_mean_value_churners"] < 0.6 * s["holdout_mean_value_non_churners"]
    by_depth: dict[float, dict[str, dict]] = {}
    for r in s["policies"]:
        by_depth.setdefault(r["depth"], {})[r["policy"]] = r
    for depth, pol in by_depth.items():
        net = {k: v["net_value_per_run"] for k, v in pol.items()}
        assert net["value_ranked"] == max(net.values()), depth
        assert net["risk_ranked"] > net["random"], depth
        breakeven = {k: v["break_even_save_rate"] for k, v in pol.items()}
        assert breakeven["value_ranked"] == min(breakeven.values()), depth
    # Value-ranked stays profitable across the grid; risk-ranked loses money at 5% / $20.
    sens = {(r["save_rate"], r["incentive_cost"], r["policy"]): r["net_value_per_run"]
            for r in s["sensitivity"]}
    assert all(v > 0 for (_, _, pol), v in sens.items() if pol == "value_ranked")
    assert sens[(0.05, 20.0, "risk_ranked")] < 0


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
