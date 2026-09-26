"""Section 01 model pipeline: baselines, learned models and explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from northstar.acquisition.dataset import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
    build_dataset,
    monthly_runs,
)
from northstar.acquisition.models import (
    LEARNED,
    MODEL_NAMES,
    ChannelRateBaseline,
    fit_model,
    logistic_coefficients,
    predict,
    shap_importance,
)


@pytest.fixture(scope="module")
def split_data(tables):
    train = build_dataset(tables, monthly_runs("2024-06-01", "2025-05-01"))
    test = build_dataset(tables, monthly_runs("2025-07-01", "2025-09-01"))
    return train, test


@pytest.fixture(scope="module")
def fitted(split_data):
    train, _ = split_data
    return {name: fit_model(name, train) for name in MODEL_NAMES}


def test_channel_baseline_is_a_smoothed_channel_rate():
    X = pd.DataFrame({"acquisition_channel": ["a"] * 10 + ["b"] * 30})
    y = np.r_[np.ones(5), np.zeros(5), np.ones(3), np.zeros(27)]
    model = ChannelRateBaseline(prior_weight=10).fit(X, y)
    overall = 8 / 40
    assert model.rates_["a"] == pytest.approx((5 + 10 * overall) / 20)
    assert model.rates_["b"] == pytest.approx((3 + 10 * overall) / 40)
    unseen = model.predict_proba(pd.DataFrame({"acquisition_channel": ["zzz"]}))[0, 1]
    assert unseen == pytest.approx(overall)


def test_every_model_outputs_valid_scores(fitted, split_data):
    _, test = split_data
    for name, model in fitted.items():
        p = predict(model, test)
        assert p.shape == (len(test),)
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), name


def test_learned_models_beat_the_channel_baseline_out_of_time(fitted, split_data):
    _, test = split_data
    auc = {name: roc_auc_score(test[TARGET], predict(m, test)) for name, m in fitted.items()}
    assert 0.5 < auc["channel_rate"] < 0.75
    for name in LEARNED:
        assert auc[name] > auc["channel_rate"] + 0.08, auc
        assert auc[name] < 0.97, "suspiciously perfect - check for leakage"


def test_training_is_deterministic(split_data):
    train, test = split_data
    for name in LEARNED:
        a = predict(fit_model(name, train), test)
        b = predict(fit_model(name, train), test)
        np.testing.assert_array_equal(a, b)


def test_models_select_columns_by_name_and_tolerate_unseen_levels(fitted, split_data):
    _, test = split_data
    shuffled = test[list(reversed(test.columns))]
    novel = test.copy()
    novel["region"] = "atlantis"
    for name in LEARNED:
        np.testing.assert_allclose(predict(fitted[name], shuffled), predict(fitted[name], test))
        assert np.isfinite(predict(fitted[name], novel)).all()


def test_logistic_coefficients_are_centered_within_categorical_features(fitted):
    coef = logistic_coefficients(fitted["logistic_regression"])
    sums = coef.loc[coef["feature"].isin(CATEGORICAL_FEATURES)].groupby("feature")[
        "coefficient"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 0.0, atol=1e-10)
    assert set(coef["feature"]) == set(FEATURES)
    assert np.allclose(coef["odds_ratio"], np.exp(coef["coefficient"]))


@pytest.mark.parametrize("name", LEARNED)
def test_shap_values_are_additive_per_raw_feature(fitted, split_data, name):
    train, test = split_data
    model = fitted[name]
    summary, per_feature = shap_importance(model, test, train, sample_size=200)
    assert set(per_feature.columns) == set(FEATURES)
    assert set(summary["feature"]) == set(FEATURES)
    # Differences in summed SHAP between two leads equal differences in the model's log-odds.
    sample = test.loc[per_feature.index]
    x = model.named_steps["pre"].transform(sample[list(FEATURES)])
    logit = model.named_steps["model"].decision_function(x)
    total = per_feature.sum(axis=1).to_numpy()
    np.testing.assert_allclose(total - total[0], logit - logit[0], atol=1e-6)
    assert summary["mean_abs_shap"].is_monotonic_decreasing
