"""Section 01 end to end: CLI interface, leakage gate and traceability of documented results."""

from __future__ import annotations

import json
import math

import pytest

from northstar.acquisition import report
from northstar.acquisition.models import BASELINES, LEARNED
from northstar.cli import main
from northstar.paths import ACQUISITION_DIR
from northstar.profile import extract_generated_block
from northstar.synthetic import generate
from northstar.synthetic import params as p

METRICS = ACQUISITION_DIR / "outputs" / "metrics.json"
README = ACQUISITION_DIR / "README.md"
FIGURES = ("gains_curve", "decile_lift", "calibration", "budget_policies", "shap_importance")
TABLES = ("model_comparison", "calibration", "decile_lift", "budget_simulation", "gains_curve",
          "feature_importance", "logistic_coefficients")


@pytest.fixture(scope="module")
def small_data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("acq") / "raw"
    assert main(["generate-data", "--seed", "11", "--n-prospects", "4000", "--out", str(out),
                 "--skip-profile"]) == 0
    return out


def _readme(path):
    path.write_text(f"# Test\n\n{report.BEGIN_MARKER}\nstale\n{report.END_MARKER}\n\nend\n")
    return path


def test_acquisition_command_writes_traceable_outputs(small_data_dir, tmp_path):
    out, readme = tmp_path / "outputs", _readme(tmp_path / "README.md")
    code = main(["acquisition", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(readme), "--no-generate"])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data"] == {"seed": 11, "n_prospects": 4000}
    assert metrics["leakage_audit"]["passed"]
    assert {m["model"] for m in metrics["models"]} == {*BASELINES, *LEARNED}
    for name in TABLES:
        assert (out / f"{name}.csv").stat().st_size > 0
    for name in FIGURES:
        assert (out / "figures" / f"{name}.png").stat().st_size > 0
    # The README block is exactly the rendering of the JSON that was written.
    block = extract_generated_block(readme, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(metrics)
    assert "stale" not in readme.read_text() and readme.read_text().rstrip().endswith("end")
    # Budget simulation covers at least two targeting policies plus random at every capacity.
    policies = {r["policy"] for r in metrics["budget_simulation"]}
    assert {"random", "channel_rate", metrics["champion"]} <= policies


def test_acquisition_command_refuses_to_write_when_leakage_audit_fails(
        small_data_dir, tmp_path, monkeypatch):
    def failing_audit(*args, **kwargs):
        return {"passed": False, "checks": {"train_labels_end_before_holdout_starts": False},
                "details": {}}

    monkeypatch.setattr(report, "leakage_audit", failing_audit)
    out = tmp_path / "outputs"
    code = main(["acquisition", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(tmp_path / "missing.md"), "--no-generate"])
    assert code == 1
    assert not out.exists()


def test_acquisition_command_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="northstar generate-data"):
        main(["acquisition", "--data-dir", str(tmp_path / "none"), "--no-generate"])


# ---------------------------------------------------------------- committed results
@pytest.fixture(scope="module")
def committed():
    return json.loads(METRICS.read_text())


def test_readme_results_block_matches_committed_metrics(committed):
    block = extract_generated_block(README, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(committed)


def test_committed_run_uses_the_default_data_and_design(committed):
    assert committed["data"] == {"seed": p.DEFAULT_SEED, "n_prospects": p.DEFAULT_N_PROSPECTS}
    assert committed["config"]["holdout_runs"][0] == "2025-07-01"
    assert committed["leakage_audit"]["passed"]


def test_prose_claims_in_the_readme_hold_for_the_committed_run(committed):
    """The narrative sections of the README make these claims; keep them true."""
    models = {m["model"]: m for m in committed["models"]}
    champion = committed["champion"]
    assert champion in LEARNED
    # Champion beats both rule baselines on the same holdout, with intervals excluding zero.
    for delta in committed["champion_vs_baselines"].values():
        assert delta["roc_auc_ci"][0] > 0 and delta["average_precision_ci"][0] > 0
    assert set(committed["champion_vs_baselines"]) == {f"{champion}_minus_{b}" for b in BASELINES}
    # Learned models are statistically tied: their AP intervals overlap.
    a, b = (models[n] for n in LEARNED)
    assert a["holdout_ap_ci_low"] < b["holdout_ap_ci_high"]
    assert b["holdout_ap_ci_low"] < a["holdout_ap_ci_high"]
    # Learned models are well calibrated out of time.
    for n in LEARNED:
        assert models[n]["holdout_ece"] < 0.01
        assert 0.9 < models[n]["holdout_calibration_slope"] < 1.1
    # Targeting value: top decile converts at >3x the average; the champion reaches more
    # conversions than every baseline at every simulated capacity.
    assert committed["deciles"][0]["lift"] > 3
    by_cap: dict[float, dict[str, float]] = {}
    for r in committed["budget_simulation"]:
        by_cap.setdefault(r["capacity_share"], {})[r["policy"]] = r["conversions_reached_per_run"]
    for cap, reached in by_cap.items():
        assert all(reached[champion] > reached[b] for b in ("random", *BASELINES)), cap
    # Most first orders in a month come from leads created during that month.
    assert committed["pipeline_coverage"]["share_leads_created_in_window"] > 0.5
    # Recency of the last visit is among the top drivers for both learned models.
    for n in LEARNED:
        ranked = sorted(committed["feature_importance"], key=lambda r: -r[f"shap_{n}"])
        assert "days_since_last_session" in [r["feature"] for r in ranked[:5]]


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
