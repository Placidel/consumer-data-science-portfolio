"""Section 08: drift statistics, performance checks and the batch scorer."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from northstar.serving import profiles as pr
from northstar.serving.batch import BatchValidationError, score_frame, score_run
from northstar.serving.monitoring import (
    SEVERITY,
    MonitoringConfig,
    auc_standard_error,
    monitor_model,
)
from northstar.serving.registry import load_artifact
from northstar.serving.scoring import Scorer
from northstar.serving.specs import ACQUISITION, CHURN, SPECS
from northstar.serving.training import predict


@pytest.fixture(scope="module")
def scorers(model_registry):
    return {name: Scorer(load_artifact(spec, model_registry)) for name, spec in SPECS.items()}


@pytest.fixture(scope="module")
def reports(scorers, tables):
    return {name: monitor_model(s, tables) for name, s in scorers.items()}


# ---------------------------------------------------------------- PSI primitives
def test_bin_shares_follow_right_closed_bins():
    shares = pr.bin_shares(np.array([-5, 0, 0.5, 1, 1.5, 9]), [0.0, 1.0])
    np.testing.assert_allclose(shares, [2 / 6, 2 / 6, 2 / 6])  # (-inf,0], (0,1], (1,inf)


def test_quantile_edges_collapse_for_discrete_features():
    values = np.r_[np.zeros(75), np.ones(15), np.full(10, 2.0)]
    assert pr.quantile_edges(values) == [0.0, 1.0]  # bins: 0, 1, 2+
    assert len(pr.quantile_edges(np.arange(1000.0))) == 9


def test_psi_is_zero_for_identical_and_matches_hand_computation():
    assert pr.psi([0.5, 0.5], [0.5, 0.5]) == 0
    expected = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
    assert pr.psi([0.5, 0.5], [0.7, 0.3]) == pytest.approx(expected)
    assert np.isfinite(pr.psi([1.0, 0.0], [0.0, 1.0]))  # empty bins are floored


def test_reference_percentile_is_the_share_at_or_below():
    q = np.quantile(np.arange(101.0) / 100, np.linspace(0, 1, 1001))
    pct = pr.reference_percentile(np.array([-1, 0.5, 2]), q)
    assert pct[0] == 0 and pct[2] == 100 and pct[1] == pytest.approx(50, abs=0.2)


def test_feature_drift_flags_only_the_shifted_features(scorers, tables):
    """Scoring the training population itself shows no drift; shifting one input is caught."""
    spec = ACQUISITION
    runs = spec.split.fit + spec.split.validation
    train = pd.concat([spec.build_features(tables, c) for c in runs], ignore_index=True)
    profile = scorers[spec.name].artifact.profile
    same = pr.feature_drift(profile, train).set_index("feature")
    assert same["psi"].max() == pytest.approx(0, abs=1e-12)

    shifted = train.assign(lead_age_days=train["lead_age_days"] * 0.5,
                           acquisition_channel="paid_social")
    drift = pr.feature_drift(profile, shifted).set_index("feature")
    assert drift.loc["lead_age_days", "psi"] > 0.25
    assert drift.loc["acquisition_channel", "psi"] > 0.25
    assert drift.loc["acquisition_channel", "most_shifted_level"] == "paid_social"
    assert drift.drop(["lead_age_days", "acquisition_channel"])["psi"].max() < 1e-12


def test_auc_standard_error_matches_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    p = y * 0.8 + rng.normal(0, 0.6, 600)
    auc = roc_auc_score(y, p)
    boot = []
    for _ in range(400):
        i = rng.integers(0, 600, 600)
        boot.append(roc_auc_score(y[i], p[i]))
    assert auc_standard_error(auc, int(y.sum()), int(600 - y.sum())) == pytest.approx(
        np.std(boot), rel=0.2)


# ---------------------------------------------------------------- monitoring reports
def test_monitoring_scores_every_current_run_and_waits_for_labels(reports):
    for name, (report, frames) in reports.items():
        spec = SPECS[name]
        runs = [r["run"] for r in report["current"]["runs"]]
        assert runs == [str(d.date()) for d in pd.date_range("2025-07-01", "2025-12-01",
                                                              freq="MS")]
        pending = report["performance"]["pending_runs"]
        expected_pending = [] if spec is ACQUISITION else ["2025-11-01", "2025-12-01"]
        assert pending == expected_pending
        assert [r["run"] for r in report["performance"]["by_run"]] == [
            r for r in runs if r not in pending]
        assert set(frames["drift_by_run"]["run"]) == set(runs)
        assert report["status"] in SEVERITY
        assert report["status"] == max([f["severity"] for f in report["findings"]] or ["ok"],
                                       key=SEVERITY.index)


def test_monitored_performance_reproduces_the_training_holdout_evaluation(reports, scorers):
    """Batch scoring through the validation path gives the model card's holdout numbers."""
    for name, (report, _) in reports.items():
        spec = SPECS[name]
        holdout_runs = {str(c.date()) for c in spec.split.holdout}
        matured = {r["run"] for r in report["performance"]["by_run"]}
        assert holdout_runs == matured
        card = scorers[name].artifact.metadata["holdout_metrics"]
        assert report["performance"]["pooled"]["roc_auc"] == pytest.approx(card["roc_auc"],
                                                                           abs=1e-12)
        assert report["performance"]["pooled"]["rows"] == card["rows"]


class _Degraded:
    """Wraps a model: ``shuffle`` keeps the score distribution but destroys the ranking;
    ``underpredict`` keeps the ranking but scales every probability down."""

    def __init__(self, model, mode):
        self.model, self.mode = model, mode

    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, 1]
        p = np.random.default_rng(0).permutation(p) if self.mode == "shuffle" else p * 0.4
        return np.column_stack([1 - p, p])


def test_degraded_models_raise_retrain_and_recalibrate(scorers, tables):
    base = scorers[CHURN.name]
    for mode, check in (("shuffle", "discrimination"), ("underpredict", "calibration")):
        artifact = dataclasses.replace(base.artifact, model=_Degraded(base.artifact.model, mode))
        report, _ = monitor_model(Scorer(artifact), tables)
        checks = {f["check"] for f in report["findings"]}
        assert checks - {"feature_drift"} >= {check}, (mode, report["findings"])
        # Shuffling keeps calibration-in-the-large, scaling keeps the ranking: no cross-talk.
        other = ({"discrimination", "calibration"} - {check}).pop()
        assert other not in checks, (mode, report["findings"])
        assert report["status"] == ("retrain" if mode == "shuffle" else "recalibrate")


def test_thresholds_control_the_findings(scorers, tables):
    scorer = scorers[CHURN.name]
    strict, _ = monitor_model(scorer, tables, MonitoringConfig(psi_moderate=0.0, psi_major=0.0))
    assert all(r["status"] == "investigate" for r in strict["feature_drift"])
    lax, _ = monitor_model(scorer, tables, MonitoringConfig(psi_moderate=9, psi_major=10))
    assert not [f for f in lax["findings"] if f["check"].endswith("drift")]


# ---------------------------------------------------------------- batch scoring
def test_batch_run_scores_the_whole_population_as_a_ranked_list(scorers, tables):
    scorer = scorers[CHURN.name]
    features = CHURN.build_features(tables, "2025-08-01")
    result = score_run(scorer, tables, "2025-08-01")
    scores = result.scores
    assert len(scores) == len(features) and not result.rejected
    assert set(scores["customer_id"]) == set(features["customer_id"])
    assert sorted(scores["rank"]) == list(range(1, len(scores) + 1))
    by_rank = scores.sort_values("rank")
    assert by_rank["churn_probability"].is_monotonic_decreasing
    merged = features.merge(scores, on="customer_id")
    np.testing.assert_allclose(merged["churn_probability"],
                               predict(CHURN, scorer.artifact.model, merged), atol=1e-12)
    assert (scores["model_version"] == scorer.artifact.version).all()


def test_batch_refuses_dates_that_would_reuse_training_labels(scorers, tables):
    with pytest.raises(BatchValidationError, match="reuse training labels"):
        score_run(scorers[ACQUISITION.name], tables, "2025-06-01")
    with pytest.raises(BatchValidationError, match="after the end of the data"):
        score_run(scorers[ACQUISITION.name], tables, "2026-02-01")


def test_batch_invalid_rows_abort_or_are_quarantined(scorers, tables):
    scorer = scorers[ACQUISITION.name]
    frame = ACQUISITION.build_features(tables, "2025-10-01")
    bad = frame.copy()
    bad.loc[bad.index[:2], "email_open_rate"] = 1.5
    with pytest.raises(BatchValidationError, match="2 of"):
        score_frame(scorer, bad)
    result = score_frame(scorer, bad, max_invalid_share=0.01)
    assert len(result.scores) == len(frame) - 2
    assert [r.row for r in result.rejected] == [0, 1]
    assert "email_open_rate" in result.rejected[0].errors[0]
    assert set(result.scores["prospect_id"]) == set(frame["prospect_id"].iloc[2:])
    with pytest.raises(BatchValidationError, match="no records"):
        score_frame(scorer, frame.iloc[:0])


@pytest.mark.parametrize("spec", [ACQUISITION, CHURN], ids=lambda s: s.name)
def test_batch_refuses_outcome_and_unknown_columns(spec, scorers, tables):
    scorer = scorers[spec.name]
    # Mature labels: build_run carries the outcome column next to the features.
    labelled = spec.build_run(tables, pd.Timestamp("2025-08-01"))
    with pytest.raises(BatchValidationError, match=f"'{spec.target}' is the outcome"):
        score_frame(scorer, labelled)
    typo = labelled.drop(columns=spec.target).rename(columns={spec.features[-1]: "misspelt"})
    with pytest.raises(BatchValidationError, match="misspelt"):
        score_frame(scorer, typo)
    # The builder's own output (id, features, run_cutoff, context columns) is accepted.
    clean = labelled.drop(columns=spec.target)
    assert len(score_frame(scorer, clean).scores) == len(clean)


def test_batch_file_needs_an_in_scope_scoring_date(scorers, tables):
    scorer = scorers[ACQUISITION.name]
    frame = ACQUISITION.build_features(tables, "2025-08-01")
    undated = frame.drop(columns="run_cutoff")
    with pytest.raises(BatchValidationError, match="no scoring date"):
        score_frame(scorer, undated)
    with pytest.raises(BatchValidationError, match="reuse training labels"):
        score_frame(scorer, undated, cutoff="2025-06-01")
    dated = score_frame(scorer, undated, cutoff="2025-08-01").scores
    assert (dated["run_cutoff"] == "2025-08-01").all()
    pd.testing.assert_frame_equal(dated, score_frame(scorer, frame).scores)

    # A run_cutoff inside the training period is refused even for a single row.
    early = frame.copy()
    early.loc[early.index[0], "run_cutoff"] = pd.Timestamp("2025-03-01")
    with pytest.raises(BatchValidationError, match="reuse training labels"):
        score_frame(scorer, early)
    with pytest.raises(BatchValidationError, match="disagree"):
        score_frame(scorer, frame, cutoff="2025-09-01")
    garbled = frame.astype({"run_cutoff": object})
    garbled.loc[garbled.index[0], "run_cutoff"] = "not a date"
    with pytest.raises(BatchValidationError, match="must be a date"):
        score_frame(scorer, garbled)
