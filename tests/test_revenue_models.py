"""Section 04 models and metrics: hand-checked ranking metrics, baselines and the learned model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.revenue import evaluation as ev
from northstar.revenue.dataset import FEATURES, TARGET, build_run
from northstar.revenue.models import (
    MODEL_NAMES,
    RFMCellBaseline,
    fit_model,
    make_model,
    predict,
)


# ---------------------------------------------------------------- metrics
def test_lorenz_gini_and_capture_match_hand_calculation():
    y = np.array([10.0, 0.0, 5.0, 0.0])
    score = np.array([4.0, 3.0, 2.0, 1.0])
    cum_n, cum_y = ev.lorenz(y, score)
    assert cum_n == pytest.approx([0, 0.25, 0.5, 0.75, 1])
    assert cum_y == pytest.approx([0, 2 / 3, 2 / 3, 1, 1])
    # Trapezoid areas: model 0.70833 -> Gini 0.41667; oracle 0.79167 -> Gini 0.58333.
    assert ev.gini(y, score) == pytest.approx(5 / 12)
    assert ev.gini(y, y) == pytest.approx(7 / 12)
    assert ev.normalized_gini(y, score) == pytest.approx(5 / 7)
    assert ev.capture(y, score, [0.25, 0.5, 0.625]) == pytest.approx([2 / 3, 2 / 3, 5 / 6])


def test_ties_are_resolved_by_expectation_and_row_order_does_not_matter():
    y = np.array([10.0, 0.0, 0.0, 0.0])
    # The top-scored pair shares all revenue in one member: taking one of two yields half of it.
    assert ev.capture(y, np.array([1.0, 1.0, 0.0, 0.0]), [0.25])[0] == pytest.approx(0.5)
    # A constant score is exactly random targeting.
    flat = np.zeros(4)
    assert ev.gini(y, flat) == pytest.approx(0.0)
    assert ev.capture(y, flat, [0.3, 0.8]) == pytest.approx([0.3, 0.8])

    rng = np.random.default_rng(0)
    y = rng.gamma(0.5, 100, 500) * (rng.random(500) < 0.4)
    s = np.round(y + rng.normal(0, 50, 500), -1)  # rounded: many ties
    perm = rng.permutation(500)
    assert ev.score_metrics(y, s) == pytest.approx(ev.score_metrics(y[perm], s[perm]))


def test_score_metrics_perfect_constant_and_biased_predictions():
    rng = np.random.default_rng(1)
    y = rng.gamma(0.6, 150, 1000) * (rng.random(1000) < 0.5)
    perfect = ev.score_metrics(y, y)
    assert (perfect["rmse"], perfect["mae"], perfect["bias"]) == pytest.approx((0, 0, 0))
    assert perfect["r2"] == pytest.approx(1) and perfect["normalized_gini"] == pytest.approx(1)
    constant = ev.score_metrics(y, np.full_like(y, y.mean()))
    assert constant["r2"] == pytest.approx(0, abs=1e-12)
    assert constant["normalized_gini"] == pytest.approx(0)
    assert constant["capture_top10"] == pytest.approx(0.1) and constant["lift_top10"] == 1
    assert np.isnan(constant["spearman"])
    doubled = ev.score_metrics(y, 2 * y)
    assert doubled["bias"] == pytest.approx(1.0)
    assert doubled["normalized_gini"] == pytest.approx(1.0)  # ranking is scale-free
    # MAE rewards predicting zero on a zero-inflated target; RMSE does not.
    zero = ev.score_metrics(y, np.zeros_like(y))
    assert zero["mae"] < constant["mae"] and zero["rmse"] > constant["rmse"]


def test_decile_gains_and_capture_tables_are_consistent():
    rng = np.random.default_rng(2)
    y = rng.gamma(0.6, 150, 1000) * (rng.random(1000) < 0.5)
    pred = y + rng.normal(0, 80, 1000).clip(-y)
    deciles = ev.decile_table(y, pred)
    assert deciles["customers"].sum() == 1000 and (deciles["customers"] == 100).all()
    assert deciles["share_of_revenue"].sum() == pytest.approx(1)
    assert deciles["cumulative_capture"].iloc[-1] == pytest.approx(1)
    assert deciles["mean_predicted"].is_monotonic_decreasing

    gains = ev.gains_table(y, {"m": pred}, np.linspace(0, 1, 11))
    assert gains.iloc[0][["random", "oracle", "m"]].tolist() == [0, 0, 0]
    assert gains.iloc[-1][["random", "oracle", "m"]].tolist() == pytest.approx([1, 1, 1])
    assert (gains["oracle"] >= gains["m"] - 1e-12).all()

    captured = ev.capture_table(y, {"m": pred}, [0.1, 0.2])
    row = captured.set_index(["share_targeted", "policy"])
    assert row.loc[(0.1, "random"), "lift"] == pytest.approx(1)
    assert row.loc[(0.1, "m"), "share_of_revenue"] == pytest.approx(ev.capture(y, pred, [0.1])[0])
    assert row.loc[(0.2, "oracle"), "lift"] >= row.loc[(0.2, "m"), "lift"]


def test_cluster_bootstrap_intervals_cover_estimates_and_pair_differences():
    rng = np.random.default_rng(4)
    n = 800
    y = rng.gamma(0.6, 150, n) * (rng.random(n) < 0.5)
    data = pd.DataFrame({"customer_id": np.arange(n), TARGET: y,
                         "good": y + rng.normal(0, 30, n), "noise": rng.normal(0, 1, n)})
    ci = ev.cluster_bootstrap(data, TARGET, ["good", "noise"], references=["noise"], n_boot=100,
                              seed=0)
    low, high = ci["good.normalized_gini"]
    assert low < ev.normalized_gini(y, data["good"]) < high
    assert ci["good.normalized_gini_minus_noise"][0] > 0
    assert "noise.rmse_minus_noise" not in ci
    assert ci == ev.cluster_bootstrap(data, TARGET, ["good", "noise"], references=["noise"],
                                      n_boot=100, seed=0)


def test_active_subset_metrics_restrict_to_recent_buyers():
    data = pd.DataFrame({"orders_180d": [0, 1, 2, 3, 0], TARGET: [0.0, 10.0, 0.0, 30.0, 5.0],
                         "s": [9.0, 1.0, 2.0, 3.0, 0.0]})
    out = ev.active_subset_metrics(data, TARGET, ["s"])
    assert out["customers"] == 3
    assert out["share_of_future_revenue"] == pytest.approx(40 / 45)
    assert out["s"]["bias"] == pytest.approx(6 / 40 - 1)


# ---------------------------------------------------------------- models
def test_rfm_cell_baseline_is_a_smoothed_cell_mean():
    X = pd.DataFrame({"days_since_last_order": [10, 10, 10, 400],
                      "orders_total": [2, 3, 2, 1],
                      "avg_order_value": [50.0, 50.0, 50.0, 50.0]})
    y = np.array([100.0, 200.0, 300.0, 0.0])
    model = RFMCellBaseline(prior_weight=2.0).fit(X, y)
    cells = model.cells(X)
    assert cells.iloc[0] == cells.iloc[1] == cells.iloc[2] != cells.iloc[3]
    global_mean = y.mean()
    assert model.predict(X.head(1))[0] == pytest.approx((600 + 2 * global_mean) / (3 + 2))
    assert model.predict(X.tail(1))[0] == pytest.approx((0 + 2 * global_mean) / (1 + 2))
    unseen = pd.DataFrame({"days_since_last_order": [100], "orders_total": [20],
                           "avg_order_value": [500.0]})
    assert model.predict(unseen)[0] == pytest.approx(global_mean)


@pytest.fixture(scope="module")
def train_and_test(tables):
    train, _ = build_run(tables, "2025-01-01")
    test, _ = build_run(tables, "2025-07-01")
    return train, test


def test_every_model_shares_one_interface_and_predicts_non_negative_revenue(train_and_test):
    train, test = train_and_test
    for name in MODEL_NAMES:
        pred = predict(fit_model(name, train), test)
        assert pred.shape == (len(test),) and np.isfinite(pred).all() and (pred >= 0).all(), name
    assert predict(make_model("run_rate").fit(None, None), test) == pytest.approx(
        test["revenue_180d"].to_numpy())
    assert predict(make_model("bgnbd_gamma_gamma").fit(None, None), test) == pytest.approx(
        test["clv_expected_revenue"].to_numpy())


def test_learned_and_probabilistic_models_rank_better_than_random_and_are_deterministic(
        train_and_test):
    train, test = train_and_test
    y = test[TARGET].to_numpy()
    for name in ("bgnbd_gamma_gamma", "gradient_boosting"):
        assert ev.normalized_gini(y, predict(fit_model(name, train), test)) > 0.5, name
    a = predict(fit_model("gradient_boosting", train), test)
    b = predict(fit_model("gradient_boosting", train.sample(frac=1, random_state=3)), test)
    assert a == pytest.approx(b, rel=1e-6)
    # The learned model reads only the declared features.
    extra = test.assign(**{TARGET: 0.0, "leak": 1.0})
    model = fit_model("gradient_boosting", train)
    assert predict(model, extra) == pytest.approx(predict(model, test[list(FEATURES)]))
