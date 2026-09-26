"""RFM scoring, the segment grid and point-in-time customer profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.lifecycle import segments as sg
from northstar.lifecycle.states import CUSTOMER_STATES
from northstar.timeline import snapshot


def test_quintile_scores_are_1_to_5_and_ties_share_a_score():
    assert sg.quintile_score(pd.Series(range(1, 11))).tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert sg.quintile_score(pd.Series(range(1, 11)), higher_is_better=False).tolist() == [
        5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    tied = sg.quintile_score(pd.Series([0, 0, 0, 0, 5]))
    assert tied.tolist() == [3, 3, 3, 3, 5]


def test_segment_grid_assigns_every_score_pair_to_exactly_the_documented_segment():
    expected = {}
    for r in range(1, 6):
        for fm in range(1, 6):
            if r >= 4:
                expected[r, fm] = "champions" if fm >= 4 else "loyalists" if fm == 3 else \
                    "promising"
            elif r == 3:
                expected[r, fm] = "loyalists" if fm >= 3 else "needs_attention"
            else:
                expected[r, fm] = "cannot_lose" if fm >= 4 else "slipping" if fm == 3 else \
                    "hibernating"
    r, fm = np.array(list(expected)).T
    assert sg.segment_rule(r, fm).tolist() == list(expected.values())
    assert set(expected.values()) == set(sg.RFM_SEGMENTS) == set(sg.RFM_RULES)
    with pytest.raises(ValueError):
        sg.segment_rule([0], [3])


def test_rfm_scores_combine_frequency_and_spend():
    profile = pd.DataFrame({"idle_days": [5.0, 50.0, 100.0, 200.0, 300.0],
                            "orders_365d": [9, 1, 2, 0, 4],
                            "revenue_365d": [900.0, 40.0, 150.0, 0.0, 100.0]})
    scores = sg.rfm_scores(profile)
    assert scores["r_score"].tolist() == [5, 4, 3, 2, 1]
    assert scores["f_score"].tolist() == [5, 2, 3, 1, 4]
    assert scores["m_score"].tolist() == [5, 2, 4, 1, 3]
    assert scores["fm_score"].tolist() == [5, 2, 4, 1, 4]   # ceil((F + M) / 2)
    assert scores["rfm_segment"].tolist() == ["champions", "promising", "loyalists",
                                              "hibernating", "cannot_lose"]


@pytest.fixture(scope="module")
def profile(tables):
    return sg.customer_profile(tables, "2025-07-01")


def test_profile_covers_exactly_customers_acquired_before_the_as_of_date(tables, profile):
    as_of = pd.Timestamp("2025-07-01")
    acquired = tables["customers"].loc[tables["customers"]["customer_since"] < as_of,
                                       "customer_id"]
    assert sorted(profile["customer_id"]) == sorted(acquired)
    assert profile["customer_id"].is_unique
    assert set(profile["state"]) <= set(CUSTOMER_STATES)
    assert profile[["r_score", "f_score", "m_score", "fm_score"]].isin(range(1, 6)).all().all()
    # Churned customers are never scored as recent; loyal ones never as lapsed.
    assert (profile.loc[profile["state"] == "churned", "r_score"] <= 2).all()
    assert (profile.loc[profile["state"] == "loyal", "r_score"] >= 3).all()


def test_scores_use_only_history_and_outcomes_only_the_window(tables, profile):
    as_of = pd.Timestamp("2025-07-01")
    past_only = sg.customer_profile(snapshot(tables, as_of), as_of)
    score_cols = ["customer_id", "state", "idle_days", "orders_365d", "revenue_365d",
                  "browse_sessions_90d", "email_open_rate_90d", "plus_member", "rfm_segment"]
    pd.testing.assert_frame_equal(past_only[score_cols], profile[score_cols])
    assert (past_only["next_orders"] == 0).all()

    orders = tables["orders"]
    window = orders.loc[(orders["order_ts"] >= as_of)
                        & (orders["order_ts"] < as_of + pd.Timedelta(days=90))]
    expected = window.groupby("customer_id")["net_amount"].sum()
    got = profile.set_index("customer_id")["next_revenue"]
    assert got.sum() == pytest.approx(expected.reindex(got.index).fillna(0).sum())
    assert got.loc[expected.index.intersection(got.index)].to_numpy() == pytest.approx(
        expected.loc[expected.index.intersection(got.index)].to_numpy())


def test_segment_summaries_partition_the_base(profile):
    rfm = sg.segment_summary(profile, "rfm_segment", sg.RFM_SEGMENTS)
    assert rfm["customers"].sum() == len(profile)
    for col in ("share_of_customers", "share_of_revenue_365d", "share_of_next_revenue"):
        assert rfm[col].sum() == pytest.approx(1.0)
    assert list(rfm["rfm_segment"]) == [s for s in sg.RFM_SEGMENTS
                                        if s in set(profile["rfm_segment"])]
    by = rfm.set_index("rfm_segment")
    assert by.loc["champions", "next_purchase_rate"] > by.loc["hibernating", "next_purchase_rate"]
