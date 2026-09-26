"""Section 01 evaluation metrics and budget simulation against hand-computable answers."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from northstar.acquisition import evaluation as ev


def _brute_force_hits(score, y, k):
    """Average hits over every ordering consistent with the scores (ties permuted)."""
    idx = range(len(score))
    orders = [o for o in itertools.permutations(idx)
              if all(score[o[i]] >= score[o[i + 1]] for i in range(len(o) - 1))]
    return np.mean([sum(y[j] for j in o[:k]) for o in orders])


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 6])
def test_expected_hits_resolves_ties_by_expectation(k):
    score = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.1])
    y = np.array([0, 1, 0, 1, 1, 0])
    assert ev.expected_hits_at_k(score, y, k) == pytest.approx(_brute_force_hits(score, y, k))


def test_expected_hits_handles_fractional_and_oversized_k():
    score, y = np.array([3.0, 2.0, 1.0]), np.array([1, 0, 1])
    assert ev.expected_hits_at_k(score, y, 1.5) == pytest.approx(1.0)
    assert ev.expected_hits_at_k(score, y, 2.5) == pytest.approx(1.5)
    assert ev.expected_hits_at_k(score, y, 10) == pytest.approx(2.0)


def _runs(scores: list[np.ndarray], labels: list[np.ndarray]) -> pd.DataFrame:
    frames = [pd.DataFrame({"run_cutoff": pd.Timestamp("2025-01-01") + pd.DateOffset(months=i),
                            "prospect_id": [f"P{i}_{j}" for j in range(len(s))],
                            "score": s, "converted": y})
              for i, (s, y) in enumerate(zip(scores, labels, strict=True))]
    return pd.concat(frames, ignore_index=True)


def test_capacity_is_allocated_within_each_run():
    # Run A has much higher scores; a pooled ranking would spend all capacity on it.
    data = _runs([np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.2, 0.1, 0.05, 0.01])],
                 [np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])])
    contacts, hits = ev.capture_by_share(data, "score", 0.25)
    assert contacts == 2 and hits == 2


def test_score_metrics_are_internally_consistent():
    rng = np.random.default_rng(1)
    s = rng.random(2000)
    y = (rng.random(2000) < 0.05 + 0.2 * s).astype(int)
    data = _runs([s[:1000], s[1000:]], [y[:1000], y[1000:]])
    m = ev.score_metrics(data, "score", probabilistic=True)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["lift_top10"] == pytest.approx(m["precision_top10"] / m["base_rate"])
    contacts, hits = ev.capture_by_share(data, "score", 0.1)
    assert m["recall_top10"] == pytest.approx(hits / y.sum())
    assert m["precision_top10"] == pytest.approx(hits / contacts)
    assert m["recall_top20"] >= m["recall_top10"]
    assert m["brier_skill"] == pytest.approx(
        1 - m["brier"] / (m["base_rate"] * (1 - m["base_rate"])))


def test_rank_only_scores_skip_probability_metrics():
    data = _runs([np.array([5.0, 1.0, 3.0, 2.0])], [np.array([1, 0, 1, 0])])
    m = ev.score_metrics(data, "score", probabilistic=False)
    assert m["roc_auc"] == 1.0
    assert "brier" not in m and "ece" not in m


def test_ece_is_zero_for_calibrated_bins_and_positive_when_overconfident():
    p = np.repeat(np.arange(1, 11) / 10, 10)
    y = np.concatenate([np.r_[np.ones(i), np.zeros(10 - i)] for i in range(1, 11)])
    assert ev.expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)
    assert ev.expected_calibration_error(y, np.clip(p * 1.5, 0, 1)) > 0.1


def test_calibration_slope_detects_overconfidence():
    rng = np.random.default_rng(3)
    logit = rng.normal(-2.5, 1.2, 60_000)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(len(p)) < p).astype(int)
    assert ev.calibration_slope(y, p) == pytest.approx(1.0, abs=0.05)
    overconfident = 1 / (1 + np.exp(-2 * logit))
    assert ev.calibration_slope(y, overconfident) == pytest.approx(0.5, abs=0.05)


def test_decile_table_conserves_leads_and_orders_an_oracle():
    rng = np.random.default_rng(5)
    y = [(rng.random(500) < 0.06).astype(int) for _ in range(3)]
    data = _runs([yy + rng.random(500) * 0.5 for yy in y], y)
    table = ev.decile_table(data, "score")
    assert table["leads"].sum() == len(data)
    assert table["conversions"].sum() == data["converted"].sum()
    assert table["cumulative_capture"].iloc[-1] == pytest.approx(1.0)
    # Every positive outranks every negative and fits in the top decile of its run.
    assert table.loc[table["decile"] == 1, "cumulative_capture"].item() == pytest.approx(1.0)
    assert table["conversion_rate"].is_monotonic_decreasing


def test_budget_simulation_random_and_oracle_bounds():
    rng = np.random.default_rng(7)
    y = [(rng.random(400) < 0.1).astype(int) for _ in range(4)]
    data = _runs([rng.random(400) for _ in range(4)], y)
    data["oracle"] = data["converted"] + 0.0
    sim = ev.budget_simulation(data, {"noise": "score", "oracle": "oracle"}, [0.05, 0.2])
    random = sim.loc[sim["policy"] == "random"].set_index("capacity_share")
    np.testing.assert_allclose(random["precision"], data["converted"].mean())
    assert (random["lift_vs_random"] == 1).all()
    assert random.loc[0.2, "contacts_per_run"] == pytest.approx(0.2 * 400)
    oracle = sim.loc[sim["policy"] == "oracle"].set_index("capacity_share")
    per_run_positives = np.array([yy.sum() for yy in y])
    expected = np.minimum(per_run_positives, 0.05 * 400).sum()
    assert oracle.loc[0.05, "conversions_reached_total"] == pytest.approx(expected)
    assert (oracle["share_of_conversions_captured"] <= 1 + 1e-12).all()
    assert (oracle["conversions_reached_total"] >= random["conversions_reached_total"]).all()


def test_gains_curve_is_monotone_and_ends_at_one():
    rng = np.random.default_rng(9)
    y = [(rng.random(300) < 0.1).astype(int) for _ in range(2)]
    data = _runs([rng.random(300) + yy for yy in y], y)
    shares = np.linspace(0, 1, 11)
    gains = ev.gains_curve(data, "score", shares)
    assert gains[0] == 0 and gains[-1] == pytest.approx(1.0)
    assert np.all(np.diff(gains) >= -1e-12)
    assert np.all(gains >= shares - 1e-9)  # an informative score beats random everywhere


def test_cluster_bootstrap_brackets_point_estimates_and_is_seeded():
    rng = np.random.default_rng(11)
    n = 1500
    y = (rng.random(n) < 0.1).astype(int)
    data = pd.DataFrame({"prospect_id": [f"P{i // 2}" for i in range(n)], "converted": y,
                         "good": y + rng.normal(0, 0.7, n), "weak": y + rng.normal(0, 2.0, n)})
    ci = ev.cluster_bootstrap(data, ["good", "weak"], ["weak"], n_boot=150, seed=1)
    auc_good = roc_auc_score(y, data["good"])
    assert ci["good.roc_auc"][0] < auc_good < ci["good.roc_auc"][1]
    assert ci["good.roc_auc_minus_weak"][0] > 0
    assert "weak.roc_auc_minus_weak" not in ci
    assert ci == ev.cluster_bootstrap(data, ["good", "weak"], ["weak"], n_boot=150, seed=1)
