"""Transition counts, matrices, rule-derived structural zeros and decision-point arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.lifecycle import transitions as tr
from northstar.lifecycle.states import (
    NOT_YET,
    STATES,
    LifecyclePanel,
    LifecycleRules,
    build_panel,
)

S = {s: i for i, s in enumerate(STATES)}


def hand_panel(states: list[list[str | None]], revenue: list[list[float]] | None = None,
               channels: list[str] | None = None) -> LifecyclePanel:
    """A panel built directly from state names (rows = people, columns = months)."""
    n, T = len(states), len(states[0])
    codes = np.array([[NOT_YET if s is None else S[s] for s in row] for row in states],
                     dtype=np.int8)
    rev = np.zeros((n, T)) if revenue is None else np.array(revenue, dtype=float)
    nan = np.full((n, T), np.nan)
    people = pd.DataFrame({"prospect_id": [f"P{i}" for i in range(n)],
                           "acquisition_channel": channels or ["email"] * n})
    return LifecyclePanel(people=people, periods=pd.date_range("2024-01-01", periods=T,
                                                              freq="MS"),
                          rules=LifecycleRules(), state=codes, tenure_days=nan,
                          idle_days=nan.copy(), orders_window=nan.copy(),
                          lead_age_days=nan.copy(), orders=(rev > 0).astype(int), revenue=rev)


def test_transition_counts_match_a_hand_count():
    panel = hand_panel([
        ["prospect", "new", "new", "active"],
        ["new", "at_risk", "churned", "churned"],
        [None, "prospect", "prospect", "new"],
        ["loyal", "loyal", "at_risk", "active"],
    ])
    counts = tr.transition_counts(panel)
    expected = {("prospect", "new"): 2, ("new", "new"): 1, ("new", "active"): 1,
                ("new", "at_risk"): 1, ("at_risk", "churned"): 1, ("churned", "churned"): 1,
                ("prospect", "prospect"): 1, ("loyal", "loyal"): 1, ("loyal", "at_risk"): 1,
                ("at_risk", "active"): 1}
    for (a, b), v in expected.items():
        assert counts.loc[a, b] == v, (a, b)
    assert counts.to_numpy().sum() == sum(expected.values()) == 11
    assert tr.transition_counts(panel, [2]).to_numpy().sum() == 4
    # Row sums equal the population of the origin state; the newcomer adds no row at t = 0.
    assert tr.transition_counts(panel, [0]).sum(axis=1).loc["new"] == 1
    matrix = tr.transition_matrix(counts)
    assert matrix.loc["new"].sum() == pytest.approx(1.0)
    assert matrix.loc["new", "active"] == pytest.approx(1 / 3)
    assert matrix.loc["active"].isna().all()  # nobody was active at an origin month


def test_transition_counts_reject_people_who_disappear():
    panel = hand_panel([["new", None]])
    with pytest.raises(ValueError, match="missing"):
        tr.transition_counts(panel)


def test_origin_windows_are_validated():
    panel = hand_panel([["new", "new", "active"]])
    assert tr.recent_origins(panel, 2) == [0, 1]
    with pytest.raises(ValueError):
        tr.transition_counts(panel, [2])
    with pytest.raises(ValueError):
        tr.recent_origins(panel, 3)


def test_toy_history_transitions_and_entries(toy_lifecycle):
    tables, periods = toy_lifecycle
    panel = build_panel(tables, periods=periods)
    counts = tr.transition_counts(panel)
    # P4: new -> at_risk -> churned -> active (a reactivation, never back to `new`) -> at_risk.
    assert counts.loc["new", "at_risk"] == 1
    assert counts.loc["churned", "active"] == 1
    assert counts.loc["active", "loyal"] == 1          # P3
    assert counts.loc["at_risk", "churned"] == 2       # P1 and P4
    entries = tr.entries(panel)
    assert entries.set_index(["period", "state"])["people"].to_dict() == {
        (pd.Timestamp("2024-01-01"), "new"): 3, (pd.Timestamp("2024-03-01"), "prospect"): 1}


def test_impossible_transitions_follow_the_rules():
    default = tr.impossible_transitions(LifecycleRules())
    assert {("active", "churned"), ("loyal", "churned"), ("new", "churned"),
            ("churned", "at_risk"), ("new", "loyal"), ("prospect", "at_risk"),
            ("churned", "new"), ("loyal", "prospect")} <= default
    for allowed in [("at_risk", "churned"), ("churned", "active"), ("at_risk", "loyal"),
                    ("new", "at_risk"), ("prospect", "new")]:
        assert allowed not in default
    # With a churn threshold only 20 days after at-risk, a month can skip at_risk entirely.
    tight = tr.impossible_transitions(LifecycleRules(at_risk_days=90, churn_days=110))
    assert ("active", "churned") not in tight


@pytest.mark.parametrize("rules", [
    LifecycleRules(),
    LifecycleRules(new_days=60, at_risk_days=60, churn_days=120),
    LifecycleRules(loyal_min_orders=3, loyal_min_tenure_days=120, churn_days=365),
])
def test_generated_data_never_makes_a_forbidden_transition(tables, rules):
    panel = build_panel(tables, rules)
    counts = tr.transition_counts(panel)
    for a, b in tr.impossible_transitions(rules):
        assert counts.loc[a, b] == 0, (a, b)


def test_state_populations_are_conserved(tables):
    """people in state s at t+1 = transitions into s from t + leads entering in s at t+1."""
    panel = build_panel(tables)
    counts = panel.state_counts()
    entries = tr.entries(panel).pivot_table(index="period", columns="state", values="people",
                                            aggfunc="sum").reindex(
        index=panel.periods, columns=list(STATES)).fillna(0)
    by_month = tr.transitions_by_period(panel)
    for t in range(panel.n_periods - 1):
        inflow = tr.transition_counts(panel, [t]).sum(axis=0)
        assert (inflow + entries.iloc[t + 1] == counts.iloc[t + 1]).all(), t
        month = by_month.loc[by_month["from_period"] == panel.periods[t], "people"].sum()
        assert month == counts.iloc[t].sum()
    pooled = tr.transition_counts(panel)
    assert pooled.to_numpy().sum() == by_month["people"].sum()


def test_wilson_interval_known_values():
    lo, hi = tr.wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = tr.wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    assert all(np.isnan(tr.wilson_interval(0, 0)))


def test_decision_points_are_well_formed():
    with pytest.raises(ValueError):
        tr.DecisionPoint("x", "x", "at_risk", ("active",), ("active",))
    with pytest.raises(ValueError):
        tr.DecisionPoint("x", "x", "churned", ("active",), ("churned",))
    for point in tr.DECISION_POINTS:
        assert point.origin not in point.unfavourable


def test_decision_point_rates_and_revenue_gap_by_hand():
    at_risk_recovery = tr.DecisionPoint("r", "r", "at_risk", ("active", "loyal"), ("churned",))
    panel = hand_panel(
        [["at_risk", "active", "active", "active"],     # recovers at t=0 -> 1
         ["at_risk", "at_risk", "churned", "churned"],  # pending at t=0, lost at t=1 -> 2
         ["at_risk", "churned", "churned", "churned"],  # lost at t=0 -> 1
         ["new", "new", "active", "active"]],           # never at risk
        revenue=[[0, 100, 50, 10], [0, 0, 0, 0], [0, 0, 0, 5], [20, 0, 30, 0]],
        channels=["email", "email", "referral", "email"])
    rows = tr.decision_rows(panel, at_risk_recovery, None)
    assert len(rows) == 3 and rows["favourable"].sum() == 1
    table = tr.decision_point_table(panel, [0, 1], follow_up_months=2,
                                    points=[at_risk_recovery]).iloc[0]
    assert table["origin_customer_months"] == 4          # 3 at t=0, 1 still at risk at t=1
    assert table["resolved"] == 3 and table["favourable_n"] == 1
    assert table["favourable_rate"] == pytest.approx(1 / 3)
    assert table["unfavourable_per_month"] == pytest.approx(1.0)
    # Follow-up of two months starting with the destination month: t=0 and t=1 origins qualify.
    assert table["follow_up_resolved"] == 3
    assert table["revenue_after_favourable"] == pytest.approx(150.0)
    assert table["revenue_after_unfavourable"] == pytest.approx((0.0 + 0.0) / 2)
    assert table["revenue_gap_per_year"] == pytest.approx(1.0 * 12 * 150.0)
    by = tr.decision_points_by_group(panel, [0, 1], "acquisition_channel",
                                     points=[at_risk_recovery]).set_index("acquisition_channel")
    assert by.loc["email", "resolved"] == 2 and by.loc["email", "favourable_rate"] == 0.5
    assert by.loc["referral", "favourable_rate"] == 0.0


def test_recency_and_lead_age_curves_partition_their_populations(tables):
    panel = build_panel(tables)
    window = tr.recent_origins(panel, 12)
    rec = tr.repurchase_by_recency(panel, window)
    customer_months = sum(int((panel.state[:, t] >= S["new"]).sum()) for t in window)
    assert rec["customer_months"].sum() == customer_months
    assert rec["repurchase_rate"].between(0, 1).all()
    # Recently active customers are far likelier to order next month than long-lapsed ones.
    assert rec["repurchase_rate"].iloc[0] > 3 * rec["repurchase_rate"].iloc[-1]
    leads = tr.conversion_by_lead_age(panel, window)
    assert leads["prospect_months"].sum() == sum(int((panel.state[:, t] == S["prospect"]).sum())
                                                 for t in window)
    value = tr.state_value(panel, window)
    assert value["share_of_customer_months"].sum() == pytest.approx(1.0)
    assert value.set_index("state").loc["loyal", "next_month_purchase_rate"] > value.set_index(
        "state").loc["churned", "next_month_purchase_rate"]
