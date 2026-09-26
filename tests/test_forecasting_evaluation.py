"""Section 05 evaluation: backtest design, leakage by poisoning the future, metrics by hand,
empirical interval coverage and the Diebold-Mariano test's size and power."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from northstar.forecasting import evaluation as ev
from northstar.forecasting import models as fm
from northstar.forecasting.series import build_series, promo_calendar


@pytest.fixture(scope="module")
def series(tables):
    daily, promos = build_series(tables)
    return daily, promo_calendar(promos)


# ---------------------------------------------------------------- plan
def test_default_plan_covers_the_documented_origins(series):
    daily, _ = series
    plan = ev.BacktestPlan()
    origins = plan.origins(daily)
    assert len(origins) == 66
    assert origins[0] == pd.Timestamp("2024-07-01") and origins[-1] == pd.Timestamp("2025-09-29")
    # The last origin's 13-week target ends with the last complete week of the data.
    assert origins[-1] + pd.Timedelta(days=plan.horizon_days) == pd.Timestamp("2025-12-29")
    roles = pd.Series([plan.role(o) for o in origins]).value_counts().to_dict()
    assert roles == {"design": 21, "calibration": 12, "evaluation": 33}
    # Every baseline is defined from the first evaluation origin (392 days of history).
    first_eval = pd.Timestamp(plan.first_evaluation_origin)
    assert (first_eval - daily.index[0]).days >= fm.YEAR_DAYS + fm.LEVEL_DAYS


def test_plan_rejects_designs_that_leak_or_misalign():
    daily = pd.Series(1.0, index=pd.date_range("2024-01-01", "2025-12-31"))
    with pytest.raises(ValueError, match="overlap"):
        ev.BacktestPlan(design_last_origin="2025-01-06").origins(daily)
    with pytest.raises(ValueError, match="Mondays"):
        ev.BacktestPlan(first_origin="2024-07-02").origins(daily)
    with pytest.raises(ValueError, match="no evaluation origins"):
        ev.BacktestPlan(last_origin="2024-12-30").origins(daily)


# ---------------------------------------------------------------- backtest and leakage
def test_backtest_rows_pair_each_forecast_with_the_realized_week(series):
    daily, promo = series
    plan = ev.BacktestPlan(first_origin="2025-06-02", design_last_origin="2025-06-02",
                           first_evaluation_origin="2025-09-01")
    bt = ev.run_backtest(daily, promo, plan, models=("naive_4wk", "harmonic"))
    origins = plan.origins(daily)
    assert len(bt) == len(origins) * 2 * plan.horizon_weeks
    row = bt.loc[(bt["origin"] == origins[3]) & (bt["horizon_week"] == 5)].iloc[0]
    week = daily.loc[row["week_start"]: row["week_start"] + pd.Timedelta(days=6)]
    assert row["week_start"] == origins[3] + pd.Timedelta(weeks=4)
    assert row["actual"] == pytest.approx(week.sum())
    assert bt["forecast"].notna().all() and (bt["forecast"] > 0).all()


@pytest.mark.parametrize("name", fm.MODEL_NAMES)
def test_poisoning_every_day_on_or_after_the_origin_changes_no_forecast(series, name):
    daily, promo = series
    origin = pd.Timestamp("2025-03-03")
    poisoned = daily.copy()
    rng = np.random.default_rng(0)
    future = poisoned.index >= origin
    poisoned[future] = rng.uniform(0, 1e7, future.sum())
    clean = ev.weekly_forecast(name, daily, promo, origin, 13)
    dirty = ev.weekly_forecast(name, poisoned, promo, origin, 13)
    np.testing.assert_array_equal(clean, dirty)
    # ... whereas changing the past does change the forecast.
    past = daily.copy()
    past[past.index < origin] *= 1.1
    assert not np.allclose(ev.weekly_forecast(name, past, promo, origin, 13), clean)


def test_quarter_totals_sum_the_weekly_rows(series):
    daily, promo = series
    plan = ev.BacktestPlan(first_origin="2025-06-02", design_last_origin="2025-06-02",
                           first_evaluation_origin="2025-09-01")
    bt = ev.run_backtest(daily, promo, plan, models=("naive_4wk",))
    totals = ev.quarter_totals(bt)
    first = bt.loc[bt["origin"] == bt["origin"].min()]
    t = totals.iloc[0]
    assert t["actual"] == pytest.approx(first["actual"].sum())
    assert t["forecast"] == pytest.approx(first["forecast"].sum())
    assert t["pct_error"] == pytest.approx(t["forecast"] / t["actual"] - 1)


# ---------------------------------------------------------------- metrics
def test_point_metrics_match_hand_calculations():
    m = ev.point_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert m["n"] == 2
    assert m["mae"] == pytest.approx(15.0)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["wape"] == pytest.approx(30 / 300)
    assert m["smape"] == pytest.approx((20 / 210 + 40 / 380) / 2)
    assert m["bias"] == pytest.approx(290 / 300 - 1)
    perfect = ev.point_metrics(np.array([0.0, 5.0]), np.array([0.0, 5.0]))
    assert perfect["mae"] == 0 and perfect["smape"] == 0 and perfect["bias"] == 0


def test_point_metrics_refuse_incomplete_or_mismatched_inputs():
    with pytest.raises(ValueError, match="complete"):
        ev.point_metrics(np.array([1.0, 2.0]), np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="same shape"):
        ev.point_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_wape_weights_weeks_by_revenue_unlike_smape():
    # The same 10% error on a small and a large week: WAPE stays 10%, sMAPE treats them alike.
    m = ev.point_metrics(np.array([10.0, 1000.0]), np.array([11.0, 900.0]))
    assert m["wape"] == pytest.approx(101 / 1010)
    assert ev.metrics_table(pd.DataFrame({"g": ["a", "a"], "actual": [10.0, 1000.0],
                                          "forecast": [11.0, 900.0]}), ["g"]).iloc[0]["mae"] \
        == pytest.approx(50.5)


def test_bucket_labels_cover_the_horizon():
    assert [ev.bucket_label(k) for k in (1, 4, 5, 8, 9, 13)] == [
        "weeks 1-4", "weeks 1-4", "weeks 5-8", "weeks 5-8", "weeks 9-13", "weeks 9-13"]
    with pytest.raises(ValueError):
        ev.bucket_label(14)


# ---------------------------------------------------------------- intervals
def _synthetic_backtest(n_origins: int, weeks: int, sd: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2020-01-06", periods=n_origins, freq="7D")
    rows = []
    for o in origins:
        for k in range(1, weeks + 1):
            rows.append({"origin": o, "role": "evaluation", "model": "m", "horizon_week": k,
                         "week_start": o + pd.Timedelta(weeks=k - 1), "forecast": 100.0,
                         "actual": 100.0 * np.exp(rng.normal(0, sd))})
    return pd.DataFrame(rows)


def test_empirical_intervals_reach_nominal_coverage_on_exchangeable_errors():
    plan = replace(ev.BacktestPlan(), horizon_weeks=3, calibration_origins=52)
    bt = _synthetic_backtest(600, 3, sd=0.1)
    iv = ev.add_weekly_intervals(bt, "m", plan).dropna(subset=["lower_80"])
    cov = ev.coverage_table(iv, plan.levels).set_index("level")
    assert cov.loc[0.8, "coverage"] == pytest.approx(0.8, abs=0.04)
    assert cov.loc[0.5, "coverage"] == pytest.approx(0.5, abs=0.04)
    # Width is multiplicative: about 2 * 1.2816 * sd of the forecast for the 80% range.
    assert cov.loc[0.8, "mean_relative_width"] == pytest.approx(2 * 1.2816 * 0.1, rel=0.15)


def test_intervals_only_use_errors_already_observed_at_the_origin():
    plan = replace(ev.BacktestPlan(), horizon_weeks=4, min_calibration_origins=5)
    bt = _synthetic_backtest(40, 4, sd=0.1, seed=1)
    iv = ev.add_weekly_intervals(bt, "m", plan)
    assert (iv["calibration_last_observed"].dropna() <= iv["origin"][
        iv["calibration_last_observed"].notna()]).all()
    # Lead k's first interval needs 5 errors, each observed k weeks after its origin.
    for k in range(1, 5):
        first = iv.loc[(iv["horizon_week"] == k) & iv["lower_80"].notna(), "origin"].min()
        assert first == bt["origin"].min() + pd.Timedelta(weeks=5 + k - 1)
    # A huge error cannot affect any interval issued before that week was observed.
    shocked = bt.copy()
    target = (shocked["origin"] == shocked["origin"].iloc[0] + pd.Timedelta(weeks=20)) & (
        shocked["horizon_week"] == 4)
    shocked.loc[target, "actual"] = 1e6
    seen = shocked.loc[target, "week_start"].iloc[0] + pd.Timedelta(days=7)
    a = ev.add_weekly_intervals(bt, "m", plan)
    b = ev.add_weekly_intervals(shocked, "m", plan)
    before = a["origin"] < seen
    pd.testing.assert_frame_equal(a.loc[before, ["lower_80", "upper_80"]],
                                  b.loc[before, ["lower_80", "upper_80"]])
    after = (b["origin"] >= seen) & (b["horizon_week"] == 4)
    assert (b.loc[after, "upper_80"] >= a.loc[after, "upper_80"]).all()


def test_forward_intervals_match_the_backtest_rule():
    plan = replace(ev.BacktestPlan(), horizon_weeks=3, min_calibration_origins=5)
    bt = _synthetic_backtest(60, 3, sd=0.2, seed=2)
    totals = ev.quarter_totals(bt)
    origin = bt["origin"].max() + pd.Timedelta(weeks=1)
    weekly, total = ev.forward_intervals(np.array([50.0, 60.0, 70.0]), origin, bt, totals, plan)
    for k, point in zip((1, 2, 3), (50.0, 60.0, 70.0), strict=True):
        past = bt.loc[(bt["horizon_week"] == k)
                      & (bt["week_start"] + pd.Timedelta(days=7) <= origin)].sort_values("origin")
        ratios = (past["actual"] / past["forecast"]).to_numpy()[-plan.calibration_origins:]
        row = weekly.loc[weekly["horizon_week"] == k].iloc[0]
        assert row["lower_80"] == pytest.approx(point * np.quantile(ratios, 0.1))
        assert row["upper_50"] == pytest.approx(point * np.quantile(ratios, 0.75))
    assert total["forecast"].iloc[0] == pytest.approx(180.0)
    assert total["calibration_n"].iloc[0] == plan.calibration_origins


def test_coverage_table_counts_misses_on_each_side():
    rows = pd.DataFrame({"actual": [1.0, 5.0, 9.0, 5.0], "forecast": [5.0] * 4,
                         "lower_80": [2.0] * 4, "upper_80": [8.0] * 4,
                         "lower_50": [4.0] * 4, "upper_50": [6.0] * 4, "g": list("aabb")})
    cov = ev.coverage_table(rows, (0.8,)).iloc[0]
    assert cov["coverage"] == 0.5 and cov["below"] == 0.25 and cov["above"] == 0.25
    assert cov["mean_relative_width"] == pytest.approx(6 / 5)
    by = ev.coverage_table(rows, (0.5,), by="g").set_index("g")
    assert by.loc["a", "coverage"] == 0.5 and by.loc["b", "above"] == 0.5


# ---------------------------------------------------------------- Diebold-Mariano
def test_diebold_mariano_has_the_right_size_under_the_null():
    rng = np.random.default_rng(3)
    rejections = [ev.diebold_mariano(rng.normal(size=40), rng.normal(size=40), 1)["p_value"]
                  < 0.05 for _ in range(1000)]
    assert 0.03 <= np.mean(rejections) <= 0.08


def test_diebold_mariano_size_with_overlapping_horizons():
    # MA(3) errors, as produced by 4-step-ahead forecasts from consecutive origins.
    rng = np.random.default_rng(4)

    def ma(n):
        e = rng.normal(size=n + 3)
        return e[3:] + e[2:-1] + e[1:-2] + e[:-3]

    p = [ev.diebold_mariano(ma(60), ma(60), 4)["p_value"] for _ in range(1000)]
    assert np.mean(np.array(p) < 0.05) <= 0.10


def test_diebold_mariano_detects_a_clearly_better_forecast():
    rng = np.random.default_rng(5)
    results = [ev.diebold_mariano(rng.normal(0, 1, 40), rng.normal(0, 2, 40), 1)
               for _ in range(200)]
    assert np.mean([r["p_value"] < 0.05 for r in results]) > 0.9
    assert all(r["mean_loss_diff"] < 0 for r in results if r["p_value"] < 0.05)


def test_diebold_mariano_input_checks_and_degenerate_case():
    with pytest.raises(ValueError):
        ev.diebold_mariano(np.ones(2), np.ones(2), 1)
    same = ev.diebold_mariano(np.ones(10), np.ones(10), 1)
    assert same["mean_loss_diff"] == 0 and same["p_value"] is None
