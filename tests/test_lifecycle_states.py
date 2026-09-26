"""Lifecycle state definitions: precedence, boundaries, point-in-time assignment and the panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.lifecycle.states import (
    CUSTOMER_STATES,
    NOT_YET,
    PRECEDENCE,
    STATES,
    LifecycleRules,
    assign_states,
    build_panel,
    classify,
    people_frame,
)

EXPECTED = {  # state at the end of each month, Jan .. Dec 2024, for the `toy_lifecycle` people
    "P1": ["new"] * 3 + ["active"] + ["at_risk"] * 3 + ["churned"] * 5,
    "P2": [None, None] + ["prospect"] * 10,
    "P3": ["new"] * 3 + ["active"] * 2 + ["loyal"] * 7,
    "P4": ["new"] * 3 + ["at_risk"] * 3 + ["churned"] * 2 + ["active"] * 3 + ["at_risk"],
}


# ---------------------------------------------------------------- rules
def test_rules_validate_thresholds():
    with pytest.raises(ValueError):
        LifecycleRules(at_risk_days=200, churn_days=180)
    with pytest.raises(ValueError):
        LifecycleRules(new_days=120, at_risk_days=90)
    with pytest.raises(ValueError):
        LifecycleRules(loyal_min_tenure_days=60)
    with pytest.raises(ValueError):
        LifecycleRules(loyal_min_orders=1)
    assert set(LifecycleRules().definitions()) == set(STATES)


def test_precedence_lists_every_state_once():
    assert sorted(PRECEDENCE) == sorted(STATES)
    assert PRECEDENCE[0] == "prospect" and PRECEDENCE[-1] == "active"


@pytest.mark.parametrize(("tenure", "idle", "n_window", "expected"), [
    (np.nan, np.nan, 0, "prospect"),
    (400, 181, 0, "churned"),
    (400, 180, 0, "at_risk"),          # boundary: idle exactly 180 days is not churned
    (400, 91, 9, "at_risk"),           # heavy past buyer, but idle: recency wins
    (400, 90, 9, "loyal"),             # boundary: idle exactly 90 days is still engaged
    (90, 10, 7, "new"),                # boundary: tenure exactly 90 days is still new
    (91, 10, 2, "active"),
    (181, 5, 6, "loyal"),
    (180, 5, 6, "active"),             # tenure must exceed 180 days for loyal
    (300, 5, 5, "active"),             # five orders are not enough
])
def test_classify_boundaries_and_precedence(tenure, idle, n_window, expected):
    assert classify([tenure], [idle], [n_window])[0] == expected


def test_classify_is_exhaustive_and_exclusive_on_a_grid():
    rules = LifecycleRules()
    tenure, idle, n = np.meshgrid(np.arange(0, 500, 7.5), np.arange(0, 500, 7.5), np.arange(12))
    ok = idle <= tenure
    tenure, idle, n = tenure[ok], idle[ok], n[ok]
    states = classify(tenure, idle, n, rules)
    assert set(np.unique(states)) == set(CUSTOMER_STATES)
    # Recompute each definition independently (no precedence) and check exactly one holds.
    engaged = idle <= rules.at_risk_days
    new = tenure <= rules.new_days
    loyal = engaged & ~new & (n >= rules.loyal_min_orders) & (
        tenure > rules.loyal_min_tenure_days)
    definitions = {
        "new": new, "loyal": loyal, "active": engaged & ~new & ~loyal,
        "at_risk": (idle > rules.at_risk_days) & (idle <= rules.churn_days),
        "churned": idle > rules.churn_days,
    }
    held = np.stack(list(definitions.values()))
    assert (held.sum(axis=0) == 1).all()
    names = np.array(list(definitions))[held.argmax(axis=0)]
    assert (names == states).all()


def test_classify_rejects_idle_longer_than_tenure():
    with pytest.raises(ValueError):
        classify([10], [20], [1])


# ---------------------------------------------------------------- assignment
def test_assign_states_matches_hand_worked_example(toy_lifecycle):
    tables, periods = toy_lifecycle
    people = people_frame(tables)
    for t, period in enumerate(periods):
        as_of = period + pd.offsets.MonthBegin(1)
        got = assign_states(people, tables["orders"], as_of)["state"].tolist()
        assert got == [EXPECTED[p][t] for p in people["prospect_id"]], period


def test_assign_states_ignores_orders_on_or_after_as_of(toy_lifecycle):
    tables, _ = toy_lifecycle
    people = people_frame(tables)
    as_of = pd.Timestamp("2024-09-10")  # C4's return order is exactly at the as-of instant
    full = assign_states(people, tables["orders"], as_of)
    past = assign_states(people, tables["orders"].loc[tables["orders"]["order_ts"] < as_of],
                         as_of)
    pd.testing.assert_frame_equal(full, past)
    assert full.loc[people["prospect_id"] == "P4", "state"].item() == "churned"


def test_panel_matches_hand_worked_example_and_monthly_spend(toy_lifecycle):
    tables, periods = toy_lifecycle
    panel = build_panel(tables, periods=periods)
    for i, pid in enumerate(panel.people["prospect_id"]):
        assert [panel.labels(t)[i] for t in range(12)] == EXPECTED[pid], pid
    p1 = panel.people.index[panel.people["prospect_id"] == "P1"][0]
    assert panel.revenue[p1, :3].tolist() == [50.0, 30.0, 0.0]
    assert panel.orders[p1].sum() == 2
    assert panel.revenue.sum() == pytest.approx(tables["orders"]["net_amount"].sum())
    assert panel.state[panel.people["prospect_id"] == "P2", :2].tolist() == [[NOT_YET, NOT_YET]]
    assert panel.state_counts().sum(axis=1).tolist() == [3, 3] + [4] * 10


def test_panel_on_generated_data_is_exclusive_exhaustive_and_deterministic(tables):
    panel = build_panel(tables)
    created = (panel.people["created_at"].to_numpy()[:, None]
               < panel.as_of.to_numpy()[None, :])
    present = panel.state != NOT_YET
    assert (present == created).all()
    assert ((panel.state[present] >= 0) & (panel.state[present] < len(STATES))).all()
    # A customer is never a prospect after their first order, and vice versa.
    first_order = panel.people["customer_since"].to_numpy()[:, None]
    ordered = first_order < panel.as_of.to_numpy()[None, :]
    is_prospect = panel.state == STATES.index("prospect")
    assert not (is_prospect & ordered).any()
    assert not (present & ~is_prospect & ~ordered).any()

    again = build_panel(tables)
    for name in ("state", "idle_days", "tenure_days", "orders", "revenue"):
        np.testing.assert_array_equal(getattr(panel, name), getattr(again, name))


def test_panel_states_do_not_depend_on_future_orders(tables):
    cutoff = pd.Timestamp("2025-04-01")
    full = build_panel(tables)
    truncated = dict(tables, orders=tables["orders"].loc[tables["orders"]["order_ts"] < cutoff])
    part = build_panel(truncated)
    t = int(np.flatnonzero(full.as_of == cutoff)[0])
    np.testing.assert_array_equal(full.state[:, :t + 1], part.state[:, :t + 1])
