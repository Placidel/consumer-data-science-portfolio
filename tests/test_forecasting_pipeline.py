"""Section 05 end to end: CLI interface, leakage gate and traceability of documented results."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from northstar.cli import main
from northstar.forecasting import report
from northstar.forecasting.evaluation import BacktestPlan
from northstar.forecasting.models import BASELINES, CHAMPION, MODEL_NAMES, HarmonicConfig
from northstar.paths import FORECAST_DIR
from northstar.profile import extract_generated_block
from northstar.synthetic import generate
from northstar.synthetic import params as p

OUT = FORECAST_DIR / "outputs"
METRICS = OUT / "metrics.json"
README = FORECAST_DIR / "README.md"
FIGURES = ("forecast_fan", "backtest_tracks", "error_by_horizon", "quarter_totals",
           "residual_diagnostics")
TABLES = ("weekly_revenue", "backtest_forecasts", "backtest_quarter_totals",
          "accuracy_by_horizon", "accuracy_by_lead_week", "model_comparison_tests",
          "interval_coverage", "forecast")


@pytest.fixture(scope="module")
def small_data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fc") / "raw"
    assert main(["generate-data", "--seed", "11", "--n-prospects", "4000", "--out", str(out),
                 "--skip-profile"]) == 0
    return out


def _readme(path):
    path.write_text(f"# Test\n\n{report.BEGIN_MARKER}\nstale\n{report.END_MARKER}\n\nend\n")
    return path


def test_forecast_command_writes_traceable_outputs(small_data_dir, tmp_path):
    out, readme = tmp_path / "outputs", _readme(tmp_path / "README.md")
    code = main(["forecast", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(readme), "--no-generate"])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data"] == {"seed": 11, "n_prospects": 4000}
    assert metrics["leakage_audit"]["passed"]
    assert {r["model"] for r in metrics["evaluation"]["overall"]} == set(MODEL_NAMES)
    for name in TABLES:
        assert (out / f"{name}.csv").stat().st_size > 0, name
    for name in FIGURES:
        assert (out / "figures" / f"{name}.png").stat().st_size > 0, name
    block = extract_generated_block(readme, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(metrics)
    assert "stale" not in readme.read_text() and readme.read_text().rstrip().endswith("end")
    assert "not guaranteed bounds" in block and "**assumed**" in block

    # Every backtest row is a complete, observed week and every forecast is defined when the
    # model is; the published forecast covers exactly the next 13 weeks.
    bt = pd.read_csv(out / "backtest_forecasts.csv", parse_dates=["origin", "week_start"])
    assert len(bt) == 66 * len(MODEL_NAMES) * 13
    assert bt.loc[bt["model"] != "seasonal_naive_yoy", "forecast"].notna().all()
    assert bt.loc[bt["role"] == "evaluation", "forecast"].notna().all()
    assert (bt["week_start"] >= bt["origin"]).all()
    champ = bt.loc[(bt["model"] == CHAMPION) & (bt["role"] == "evaluation")]
    assert champ[["lower_80", "upper_80"]].notna().all().all()
    assert (champ["lower_80"] <= champ["lower_50"]).all()
    assert (champ["upper_50"] <= champ["upper_80"]).all()
    assert bt.loc[bt["model"] != CHAMPION, "lower_80"].isna().all()
    fc = pd.read_csv(out / "forecast.csv", parse_dates=["week_start"])
    assert len(fc) == 13 and fc["week_start"].iloc[0] == pd.Timestamp(
        metrics["forecast"]["origin"])
    assert (fc["week_start"].dt.dayofweek == 0).all()
    assert metrics["forecast"]["total"]["forecast"] == pytest.approx(fc[CHAMPION].sum(),
                                                                     rel=1e-6)


def test_forecast_command_refuses_to_write_when_leakage_audit_fails(
        small_data_dir, tmp_path, monkeypatch):
    def failing_audit(*args, **kwargs):
        return {"passed": False,
                "checks": {"forecasts_identical_when_rebuilt_from_orders_before_origin": False},
                "details": {}}

    monkeypatch.setattr(report, "leakage_audit", failing_audit)
    out = tmp_path / "outputs"
    code = main(["forecast", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(tmp_path / "missing.md"), "--no-generate"])
    assert code == 1
    assert not out.exists()


def test_leakage_audit_catches_a_forecast_that_uses_the_whole_series(tables, monkeypatch):
    """A classic leak: scaling by a statistic of the full series (here its mean), which the
    history-slicing contract cannot see. Rebuilding from truncated orders exposes it."""
    from northstar.forecasting import evaluation as ev

    real = ev.weekly_forecast

    def leaky(name, daily, promo, origin, horizon_weeks, config=None):
        return real(name, daily, promo, origin, horizon_weeks, config) * daily.mean() / 1e4

    monkeypatch.setattr(ev, "weekly_forecast", leaky)
    config = report.ForecastConfig(plan=BacktestPlan(last_origin="2025-03-10"))
    with pytest.raises(report.LeakageError, match="rebuilt_from_orders_before_origin"):
        report.run_analysis(tables, config)


def test_forecast_command_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="northstar generate-data"):
        main(["forecast", "--data-dir", str(tmp_path / "none"), "--no-generate"])


# ---------------------------------------------------------------- committed results
@pytest.fixture(scope="module")
def committed():
    return json.loads(METRICS.read_text())


def test_readme_results_block_matches_committed_metrics(committed):
    block = extract_generated_block(README, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(committed)


def test_committed_run_uses_the_default_design(committed):
    assert committed["data"] == {"seed": p.DEFAULT_SEED, "n_prospects": p.DEFAULT_N_PROSPECTS}
    assert committed["config"]["plan"] == BacktestPlan().as_dict()
    assert committed["config"]["harmonic"] == HarmonicConfig().as_dict()
    assert committed["config"]["horizon_days"] == 91
    assert committed["config"]["champion"] == CHAMPION
    assert committed["leakage_audit"]["passed"]
    assert committed["leakage_audit"]["details"]["origins_rebuilt_from_truncated_orders"] == 33


def test_readme_documents_horizon_uncertainty_and_assumptions():
    text = README.read_text()
    for phrase in ("13 weeks (91 days): one fiscal quarter", "not guaranteed bounds",
                   "a range, not just a point", "Known-in-advance inputs",
                   "assumes 2025's promotions repeat", "not statistically significant",
                   "not perfectly pristine", "Rolling-origin (expanding-window) backtest",
                   "never a random split", "treat them as a minimum"):
        assert phrase in text, phrase


def test_prose_claims_in_the_readme_hold_for_the_committed_run(committed):
    """The narrative sections of the README make these claims; keep them true."""
    e = committed["evaluation"]
    overall = {r["model"]: r for r in e["overall"]}
    bucket = {(r["model"], r["bucket"]): r for r in e["by_bucket"]}
    totals = {r["model"]: r for r in e["totals"]}
    dm = {(r["reference"], r["target"]): r for r in e["comparisons"]}
    champ = overall[CHAMPION]

    # Lowest weekly WAPE and 13-week-total error; small bias; baselines biased as described.
    assert champ["wape"] == min(r["wape"] for r in overall.values())
    assert totals[CHAMPION]["mape"] == min(r["mape"] for r in totals.values())
    assert abs(champ["bias"]) < 0.05
    assert -0.15 < overall["naive_4wk"]["bias"] < -0.08
    assert 0.15 < overall["seasonal_naive_yoy"]["bias"] < 0.25
    for ref in BASELINES:
        assert champ["wape"] < overall[ref]["wape"]

    # Advantage at 5-13 weeks, not next week.
    assert bucket[("naive_4wk", "weeks 1-4")]["wape"] <= bucket[(CHAMPION, "weeks 1-4")]["wape"]
    assert bucket[(CHAMPION, "weeks 9-13")]["skill_vs_naive"] > 0.4
    for target in ("week 1", "week 4"):
        assert dm[("naive_4wk", target)]["p_value"] > 0.05
    for target in ("week 8", "week 13", "13-week total"):
        assert dm[("seasonal_naive_yoy", target)]["p_value"] < 0.05
        assert dm[("seasonal_naive_yoy", target)]["mean_loss_diff"] < 0
    for target in ("week 13", "13-week total"):
        assert dm[("naive_4wk", target)]["mean_loss_diff"] < 0
        assert dm[("naive_4wk", target)]["p_value"] > 0.05

    # The promotion calendar adds a little, never significantly.
    assert champ["wape"] < overall["harmonic_no_promo"]["wape"] < champ["wape"] + 0.01
    assert all(r["p_value"] > 0.05 for (ref, _), r in dm.items() if ref == "harmonic_no_promo")

    # Design period: the settings were chosen where the model beat the run rate.
    design = {r["model"]: r for r in committed["design_period"]}
    assert design[CHAMPION]["wape"] < design["naive_4wk"]["wape"]

    # Biggest misses: Feb-Mar origins over-forecast by over a fifth; August under-forecasts.
    q = pd.read_csv(OUT / "backtest_quarter_totals.csv", parse_dates=["origin"])
    q = q.loc[(q["model"] == CHAMPION) & (q["role"] == "evaluation")]
    worst = q.loc[q["pct_error"].abs().idxmax()]
    assert worst["origin"].month in (2, 3) and worst["pct_error"] > 0.2
    assert (q.loc[q["origin"].dt.month.isin([2, 3]), "pct_error"] > 0.2).mean() > 0.5
    aug = q.loc[q["origin"].dt.month == 8, "pct_error"]
    assert aug.between(-0.15, -0.05).all()
    assert totals[CHAMPION]["max_ape"] == pytest.approx(worst["pct_error"], abs=1e-3)

    # Ranges too narrow, especially far out and at the holiday peak; misses on the upside.
    iv = committed["intervals"]
    over80 = next(r for r in iv["overall"] if r["level"] == 0.8)
    assert 0.6 < over80["coverage"] < 0.75 and over80["above"] > over80["below"]
    tot80 = next(r for r in iv["total"] if r["level"] == 0.8)
    assert 0.45 < tot80["coverage"] < 0.6
    b80 = {r["bucket"]: r["coverage"] for r in iv["by_bucket"] if r["level"] == 0.8}
    assert b80["weeks 1-4"] >= 0.75 and b80["weeks 1-4"] > b80["weeks 5-8"] > b80["weeks 9-13"]
    s80 = {r["season"]: r for r in iv["by_season"] if r["level"] == 0.8}
    assert 0.4 < s80["Nov-Dec peak weeks"]["coverage"] < 0.6
    assert s80["Nov-Dec peak weeks"]["coverage"] < s80["other weeks"]["coverage"]
    assert s80["Nov-Dec peak weeks"]["below"] == 0

    # Forward outlook: January dip, recovery through March, slower growth than last quarter,
    # run rate far above, last year x growth above.
    f = committed["forecast"]
    weeks = pd.DataFrame(f["weeks"]).assign(week_start=lambda d: pd.to_datetime(d["week_start"]))
    low = weeks.loc[weeks[CHAMPION].idxmin()]
    assert low["week_start"].month == 1
    assert weeks.loc[weeks["week_start"].dt.month == 3, CHAMPION].min() > low[CHAMPION]
    t, s = f["total"], committed["series"]
    growth = t["forecast"] / t["same_weeks_last_year"]
    assert 1.3 < growth < s["last_13_weeks_revenue"] / s["same_13_weeks_prior_year_revenue"]
    assert t["naive_4wk_total"] > 1.3 * t["forecast"]
    assert t["seasonal_naive_yoy_total"] > t["forecast"]
    assert any(p_["assumed"] for p_ in f["promotions_in_horizon"])


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
