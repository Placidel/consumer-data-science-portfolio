"""Business invariants and 'learnable but not trivial' signal checks on generated data.

Signal tests follow the leak-free convention later sections use: features come from
``snapshot(tables, cutoff)`` and labels from the full tables after the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.profile import conversion_within
from northstar.schema import FUNNEL_STAGES
from northstar.synthetic import experiment_variant
from northstar.timeline import DEFAULT_CUTOFF, PredictionWindow, snapshot


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U)."""
    ranks = pd.Series(score).rank().to_numpy()
    label = np.asarray(label, dtype=bool)
    n_pos, n_neg = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def test_auc_helper():
    assert auc(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([1, 1, 1, 1]), np.array([0, 1, 0, 1])) == 0.5


def test_lead_conversion_rate_is_plausible_and_varies_by_channel(tables):
    conv = conversion_within(tables["prospects"], tables["customers"], 60)
    assert 0.15 < conv["converted"].mean() < 0.5
    by_channel = conv.groupby("acquisition_channel")["converted"].mean()
    assert by_channel["referral"] > by_channel["display"] + 0.1
    assert by_channel["email"] > by_channel["paid_social"]


def test_order_economics_are_plausible(tables):
    orders = tables["orders"]
    assert 40 < orders["net_amount"].mean() < 200
    discounted = orders["discount_amount"] > 0
    assert 0.05 < discounted.mean() < 0.4
    assert set(orders["order_channel"]) == {"web", "app", "store"}
    lines = tables["order_lines"].merge(tables["products"], on="product_id")
    assert (lines["unit_price"] == lines["list_price"]).all()
    assert (tables["products"]["unit_cost"] < tables["products"]["list_price"]).all()


def test_holiday_season_lifts_daily_demand(tables):
    orders = tables["orders"]
    orders = orders.loc[orders["order_ts"].dt.year == 2025]
    per_day = orders.groupby(orders["order_ts"].dt.date).size()
    months = pd.to_datetime(pd.Series(per_day.index)).dt.month.to_numpy()
    assert per_day[months >= 11].mean() > 1.2 * per_day[(months >= 8) & (months <= 10)].mean()


def test_funnel_counts_never_increase_downstream(tables):
    counts = tables["funnel_events"]["event_type"].value_counts().reindex(FUNNEL_STAGES)
    assert counts.is_monotonic_decreasing
    assert counts.iloc[-1] > 0


def test_experiment_assignment_is_stable_and_balanced(tables):
    a = tables["experiment_assignments"]
    recomputed = [experiment_variant("EXP001", pid, 0.5) for pid in a["prospect_id"]]
    assert (a["variant"] == recomputed).all()
    n = len(a)
    share = (a["variant"] == "treatment").mean()
    assert abs(share - 0.5) < 4 * np.sqrt(0.25 / n)  # sample-ratio mismatch guard
    # Assignment does not depend on anything but the id: re-hashing is idempotent.
    assert experiment_variant("EXP001", "P000123", 0.5) == experiment_variant("EXP001",
                                                                              "P000123", 0.5)


def test_membership_and_support_activity_exist_at_realistic_rates(tables):
    subs = tables["subscription_events"]
    members = subs.loc[subs["event_type"] == "subscribe", "customer_id"].nunique()
    assert 0.05 < members / len(tables["customers"]) < 0.4
    contacts = tables["support_contacts"]
    assert 0.03 < len(contacts) / len(tables["orders"]) < 0.2
    assert 3.0 < contacts["csat_score"].mean() < 4.5


@pytest.fixture(scope="module")
def cutoff_view(tables):
    return snapshot(tables, DEFAULT_CUTOFF)


def test_churn_is_learnable_from_pre_cutoff_recency_but_not_trivially(tables, cutoff_view):
    window = PredictionWindow(DEFAULT_CUTOFF, horizon_days=90)
    last = cutoff_view["orders"].groupby("customer_id")["order_ts"].max()
    active = last[last >= DEFAULT_CUTOFF - pd.Timedelta(days=180)]
    future = tables["orders"].loc[window.label_mask(tables["orders"]["order_ts"])]
    churned = ~active.index.isin(future["customer_id"])
    assert 0.2 < churned.mean() < 0.8
    recency_days = (DEFAULT_CUTOFF - active).dt.days.to_numpy()
    assert 0.6 < auc(recency_days, churned) < 0.95


def test_acquisition_is_learnable_from_pre_cutoff_behaviour(tables, cutoff_view):
    window = PredictionWindow(DEFAULT_CUTOFF, horizon_days=60)
    leads = cutoff_view["prospects"]
    leads = leads.loc[(leads["created_at"] >= DEFAULT_CUTOFF - pd.Timedelta(days=120))
                      & ~leads["prospect_id"].isin(cutoff_view["customers"]["prospect_id"])]
    customers = tables["customers"]
    converted = leads["prospect_id"].isin(
        customers.loc[window.label_mask(customers["customer_since"]), "prospect_id"])
    assert 0.01 < converted.mean() < 0.3
    lead_age = (DEFAULT_CUTOFF - leads["created_at"]).dt.days.to_numpy()
    assert 0.6 < auc(-lead_age, converted) < 0.97
    by_channel = leads.assign(y=converted.to_numpy()).groupby("acquisition_channel")["y"].mean()
    assert by_channel.max() > 2 * by_channel.min()


def test_future_value_correlates_with_past_value(tables, cutoff_view):
    window = PredictionWindow(DEFAULT_CUTOFF, horizon_days=180)
    past = cutoff_view["orders"].loc[
        cutoff_view["orders"]["order_ts"] >= DEFAULT_CUTOFF - pd.Timedelta(days=180)]
    future = tables["orders"].loc[window.label_mask(tables["orders"]["order_ts"])]
    ids = cutoff_view["customers"]["customer_id"]
    x = past.groupby("customer_id")["net_amount"].sum().reindex(ids).fillna(0)
    y = future.groupby("customer_id")["net_amount"].sum().reindex(ids).fillna(0)
    rho = x.rank().corr(y.rank())  # Spearman correlation
    assert 0.3 < rho < 0.9


def test_members_churn_less_descriptively(tables, cutoff_view):
    subs = cutoff_view["subscription_events"].sort_values("event_ts")
    last_event = subs.groupby("customer_id")["event_type"].last()
    members = last_event[last_event != "cancel"].index
    last = cutoff_view["orders"].groupby("customer_id")["order_ts"].max()
    active = last[last >= DEFAULT_CUTOFF - pd.Timedelta(days=180)].index
    future = tables["orders"].loc[
        PredictionWindow(DEFAULT_CUTOFF, 90).label_mask(tables["orders"]["order_ts"]),
        "customer_id"]
    churned = pd.Series(~active.isin(future), index=active)
    assert churned[active.isin(members)].mean() < churned[~active.isin(members)].mean()
