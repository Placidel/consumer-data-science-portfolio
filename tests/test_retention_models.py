"""Section 02 models, evaluation metrics and the retention ROI simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from northstar.acquisition.evaluation import capture_by_share
from northstar.retention import evaluation as ev
from northstar.retention import simulation as sim
from northstar.retention.dataset import (
    FEATURES,
    HORIZON_DAYS,
    TARGET,
    VALUE,
    SplitPlan,
    build_dataset,
    monthly_runs,
)
from northstar.retention.models import (
    MODEL_NAMES,
    PROBABILISTIC,
    RecencyBaseline,
    RFMCellBaseline,
    fit_model,
    logistic_coefficients,
    predict,
    reason_codes,
    rfm_cell,
    shap_importance,
)

PLAN = SplitPlan(fit=monthly_runs("2024-07-01", "2024-12-01"),
                 validation=(pd.Timestamp("2025-03-01"),),
                 holdout=monthly_runs("2025-07-01", "2025-09-01"), horizon_days=HORIZON_DAYS)


@pytest.fixture(scope="module")
def split(tables):
    data = build_dataset(tables, PLAN.fit + PLAN.validation + PLAN.holdout)
    train = data.loc[data["run_cutoff"].isin(PLAN.train)]
    test = data.loc[data["run_cutoff"].isin(PLAN.holdout)].copy()
    return train, test


@pytest.fixture(scope="module")
def scored(split):
    train, test = split
    models = {name: fit_model(name, train) for name in MODEL_NAMES}
    for name, model in models.items():
        test[name] = predict(model, test)
    return models, train, test


# ---------------------------------------------------------------- models
def test_rfm_cells_follow_documented_bucket_edges():
    X = pd.DataFrame({"days_since_last_order": [0, 29.9, 30, 95, 179],
                      "orders_total": [1, 2, 3, 4, 8]})
    assert list(rfm_cell(X)) == ["R0F0", "R0F1", "R1F1", "R3F2", "R4F3"]


def test_rfm_baseline_is_a_smoothed_cell_rate():
    X = pd.DataFrame({"days_since_last_order": [5.0] * 4 + [150.0] * 4,
                      "orders_total": [1.0] * 8})
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    model = RFMCellBaseline(prior_weight=4.0).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p[0] == pytest.approx((1 + 4 * 0.5) / (4 + 4))
    assert p[-1] == pytest.approx((3 + 4 * 0.5) / (4 + 4))
    unseen = pd.DataFrame({"days_since_last_order": [60.0], "orders_total": [20.0]})
    assert model.predict_proba(unseen)[0, 1] == pytest.approx(0.5)


def test_recency_baseline_ranks_by_days_since_last_order():
    X = pd.DataFrame({"days_since_last_order": [1.0, 100.0, 10.0]})
    p = RecencyBaseline().fit(X, np.zeros(3)).predict_proba(X)[:, 1]
    assert list(np.argsort(-p)) == [1, 2, 0]
    assert ((p >= 0) & (p < 1)).all()


def test_models_return_valid_probabilities_and_learned_models_beat_recency(scored):
    _, _, test = scored
    auc = {name: roc_auc_score(test[TARGET], test[name]) for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        assert test[name].between(0, 1).all(), name
        assert auc[name] > 0.6, (name, auc[name])
    assert auc["logistic_regression"] > auc["recency_rule"]
    assert auc["gradient_boosting"] > auc["recency_rule"]


def test_models_ignore_non_feature_columns(scored, split):
    models, _, test = scored
    shuffled = test.copy()
    shuffled[TARGET] = 1 - shuffled[TARGET]
    shuffled[VALUE] = 0.0
    for name in ("logistic_regression", "gradient_boosting"):
        np.testing.assert_allclose(predict(models[name], shuffled), test[name])


def test_explanations_cover_every_feature_and_are_additive(scored):
    models, train, test = scored
    summary, per_feature = shap_importance(models["gradient_boosting"], test, train,
                                           sample_size=150)
    assert set(summary["feature"]) == set(FEATURES)
    # TreeSHAP values plus the expected value reproduce the model's log-odds.
    clf = models["gradient_boosting"].named_steps["model"]
    x = test.loc[per_feature.index, list(FEATURES)]
    raw = clf.decision_function(models["gradient_boosting"].named_steps["pre"].transform(x))
    base = raw - per_feature.sum(axis=1).to_numpy()
    assert np.ptp(base) < 1e-6
    coef = logistic_coefficients(models["logistic_regression"])
    assert set(coef["feature"]) == set(FEATURES)
    codes = reason_codes(per_feature, top=2)
    for idx, text in codes.items():
        names = [n for n in text.split("; ") if n]
        assert len(names) <= 2
        assert all(per_feature.loc[idx, n] > 0 for n in names)


# ---------------------------------------------------------------- evaluation
def _toy_runs() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frames = []
    for run in pd.date_range("2025-01-01", periods=3, freq="MS"):
        n = 200
        y = rng.integers(0, 2, n)
        frames.append(pd.DataFrame({"run_cutoff": run, TARGET: y,
                                    "score": y + rng.normal(0, 0.8, n),
                                    "noise": rng.normal(size=n)}))
    return pd.concat(frames, ignore_index=True)


def test_depth_metrics_match_a_manual_within_run_ranking():
    data = _toy_runs()
    m = ev.score_metrics(data, "score", probabilistic=False, depths=(0.1,))
    hits = sum(run.nlargest(20, "score")[TARGET].sum() for _, run in data.groupby("run_cutoff"))
    assert m["precision_top10"] == pytest.approx(hits / 60)
    assert m["recall_top10"] == pytest.approx(hits / data[TARGET].sum())
    table = ev.depth_table(data, {"score": "score", "noise": "noise"}, depths=(0.1, 0.5))
    rnd = table.loc[table["model"] == "random"]
    assert list(rnd["precision"]) == pytest.approx([data[TARGET].mean()] * 2)
    assert list(rnd["recall"]) == [0.1, 0.5]
    good = table.loc[(table["model"] == "score") & (table["depth"] == 0.1)].iloc[0]
    assert good["precision"] == pytest.approx(m["precision_top10"])
    assert good["customers_per_run"] == pytest.approx(20)


def test_decile_table_partitions_each_run():
    data = _toy_runs()
    table = ev.decile_table(data, "score")
    assert table["customers"].sum() == len(data)
    assert table["churners"].sum() == data[TARGET].sum()
    assert table["cumulative_capture"].iloc[-1] == pytest.approx(1.0)
    assert table["churn_rate"].iloc[0] > table["churn_rate"].iloc[-1]


def test_segment_drivers_cover_every_row_once_per_segment(scored):
    _, _, test = scored
    table = ev.segment_drivers(test, "gradient_boosting")
    for _, seg in table.groupby("segment"):
        assert seg["customer_runs"].sum() == len(test)
        assert seg["share_of_base"].sum() == pytest.approx(1.0)
    assert set(table["segment"]) == set(ev.segments(test))


def test_learned_models_are_reasonably_calibrated_out_of_time(scored):
    _, _, test = scored
    for name in PROBABILISTIC:
        m = ev.score_metrics(test, name, probabilistic=True)
        assert m["ece"] < 0.06, (name, m["ece"])
        assert abs(m["mean_predicted"] - m["base_rate"]) < 0.06


# ---------------------------------------------------------------- simulation
A = sim.RetentionAssumptions(save_rate=0.2, contact_cost=1.0, incentive_cost=10.0,
                             nonchurner_redemption=0.5, value_multiplier=1.0)


def _sim_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "run_cutoff": pd.Timestamp("2025-07-01"),
        TARGET: [1, 1, 0, 0],
        VALUE: [200.0, 20.0, 500.0, 50.0],
        "p": [0.9, 0.8, 0.3, 0.1],
    })


def test_assumptions_are_validated():
    with pytest.raises(ValueError):
        sim.RetentionAssumptions(save_rate=1.5)
    with pytest.raises(ValueError):
        sim.RetentionAssumptions(contact_cost=-1)
    assert set(sim.DESCRIPTIONS) == set(A.as_dict())


def test_outcome_accounting_matches_a_hand_calculation():
    data = _sim_frame()
    r = sim.evaluate_selection(data, np.array([True, False, True, False]), A)
    # Customer 0 churns (value 200): 0.2 saves, 0.2 * 200 = 40 margin, 0.2 * $10 incentive.
    # Customer 2 would buy anyway: 0.5 * $10 subsidy. Two contacts at $1.
    assert r["expected_saves"] == pytest.approx(0.2)
    assert r["incremental_margin"] == pytest.approx(40.0)
    assert r["program_cost"] == pytest.approx(2.0 + 2.0 + 5.0)
    assert r["net_value"] == pytest.approx(40.0 - 9.0)
    # At the break-even save rate the net value is zero.
    at_break_even = sim.evaluate_selection(
        data, np.array([True, False, True, False]),
        sim.RetentionAssumptions(**{**A.as_dict(), "save_rate": r["break_even_save_rate"]}))
    assert at_break_even["net_value"] == pytest.approx(0.0, abs=1e-9)


def test_expected_net_equals_realized_net_when_outcomes_are_certain():
    data = _sim_frame()
    realized = [sim.evaluate_selection(data, np.eye(4, dtype=bool)[i], A)["net_value"]
                for i in range(4)]
    planned = sim.expected_net(data[TARGET].to_numpy(float), data[VALUE].to_numpy(), A)
    np.testing.assert_allclose(planned, realized)


def test_ranked_depth_uses_within_run_top_k_and_random_is_proportional():
    data = _sim_frame()
    top_half = sim.evaluate_depth(data, "p", 0.5, A)
    assert top_half == pytest.approx(
        sim.evaluate_selection(data, np.array([True, True, False, False]), A))
    rnd = sim.evaluate_depth(data, None, 0.5, A)
    everyone = sim.evaluate_selection(data, np.ones(4, bool), A)
    assert rnd["net_value"] == pytest.approx(0.5 * everyone["net_value"])


def test_vectorized_topk_matches_the_reference_tie_aware_ranking():
    rng = np.random.default_rng(5)
    data = pd.DataFrame({"run_cutoff": rng.integers(0, 3, 300),
                         "s": rng.integers(0, 8, 300).astype(float),  # many ties
                         "y": rng.integers(0, 2, 300).astype(float),
                         "v": rng.gamma(2.0, 40.0, 300)})
    depths = [0.0, 0.07, 0.25, 0.5, 1.0]
    got = sim.topk_sums(data, "s", ["y", "v"], depths)
    expected = [[capture_by_share(data, "s", d, c)[1] for c in ("y", "v")] for d in depths]
    np.testing.assert_allclose(got, expected)


def test_policies_and_threshold_rule_on_holdout(scored):
    _, _, test = scored
    test = test.copy()
    test["value_ranked"] = sim.value_score(test, "gradient_boosting", A)
    test["oracle"] = test[TARGET] * test[VALUE]
    table = sim.simulate_policies(
        test, {"random": None, "risk": "gradient_boosting", "value": "value_ranked",
               "oracle": "oracle"}, [0.1, 0.3], A)
    for _, g in table.groupby("depth"):
        net = g.set_index("policy")["net_value_per_run"]
        assert net["oracle"] >= max(net["risk"], net["value"], net["random"])
        assert net["risk"] > net["random"]
        assert net["value"] > net["random"]
        contacts = g["customers_targeted_per_run"]
        assert np.ptp(contacts) == pytest.approx(0.0)  # equal budget for every policy
    thr = sim.threshold_policy(test, "gradient_boosting", A)
    selected = sim.expected_net(test["gradient_boosting"], test[VALUE].to_numpy(), A) > 0
    assert thr["depth"] == pytest.approx(selected.mean())


def test_sensitivity_moves_in_the_expected_direction(scored):
    _, _, test = scored
    table = sim.sensitivity(test, "gradient_boosting", 0.1, A, [0.05, 0.25], [5.0, 20.0])
    for policy, g in table.groupby("policy"):
        net = g.set_index(["save_rate", "incentive_cost"])["net_value_per_run"]
        assert net[(0.25, 5.0)] > net[(0.05, 5.0)], policy
        assert net[(0.25, 5.0)] > net[(0.25, 20.0)], policy
