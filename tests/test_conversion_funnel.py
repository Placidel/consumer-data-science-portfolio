"""Section 03 funnel: ordered-stage definitions, monotonicity and drop-off accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.conversion import funnel as fn
from northstar.schema import FUNNEL_STAGES

T0 = pd.Timestamp("2025-01-01 10:00")


def _events(spec: dict[str, list[int]], out_of_order: str | None = None) -> pd.DataFrame:
    rows = []
    for sid, stages in spec.items():
        for k in stages:
            ts = T0 + pd.Timedelta(minutes=k)
            if sid == out_of_order and k == max(stages):
                ts = T0 - pd.Timedelta(minutes=5)
            rows.append({"event_id": f"{sid}-{k}", "session_id": sid, "event_ts": ts,
                         "event_type": FUNNEL_STAGES[k - 1], "stage_number": k})
    return pd.DataFrame(rows)


def test_depth_is_the_longest_contiguous_prefix_of_stages():
    events = _events({"full": [1, 2, 3, 4, 5], "cart": [1, 2, 3], "skip": [1, 2, 4, 5],
                      "no_start": [2, 3], "bounce": [1]})
    depth = fn.session_depth(events)
    assert depth.to_dict() == {"bounce": 1, "cart": 3, "full": 5, "no_start": 0, "skip": 2}


def test_integrity_report_flags_every_defect_type():
    events = _events({"ok": [1, 2, 3], "skip": [1, 3], "late": [1, 2], "no_start": [2]},
                     out_of_order="late")
    events = pd.concat([events, events.iloc[[0]].assign(event_id="dup"),
                        events.iloc[[0]].assign(event_id="orphan", session_id="ghost")])
    sessions = pd.DataFrame({"session_id": ["ok", "skip", "late", "no_start", "silent"]})
    report = fn.funnel_integrity(sessions, events)
    assert report["sessions_with_skipped_stages"] == 2  # "skip" and "no_start"
    assert report["sessions_missing_session_start"] == 1
    assert report["out_of_order_events"] == 1
    assert report["duplicate_stage_events"] == 1
    assert report["orphan_events"] == 1
    assert report["sessions_without_events"] == 1


def test_stage_table_accounting():
    table = fn.stage_table(np.array([1, 1, 2, 3, 3, 5, 5, 5, 4, 2]))
    assert table["reached"].tolist() == [10, 8, 6, 4, 3]
    assert table["step_conversion"].iloc[1:].tolist() == pytest.approx([0.8, 0.75, 4 / 6, 0.75])
    assert table["lost_at_step"].tolist() == [0, 2, 2, 2, 1]
    assert table["share_of_all_losses"].sum() == pytest.approx(1.0)
    assert table["share_of_start"].iloc[-1] == pytest.approx(0.3)


def test_downstream_counts_can_never_increase():
    fn.assert_monotone([10, 8, 8, 3, 0])
    with pytest.raises(fn.FunnelIntegrityError):
        fn.assert_monotone([10, 8, 9, 3, 1])


def test_skipped_stage_events_do_not_inflate_downstream_counts():
    """A purchase logged without checkout_start must not count as a purchase-stage session."""
    events = _events({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3], "c": [1, 2, 5]})
    naive = events["event_type"].value_counts().reindex(FUNNEL_STAGES).fillna(0)
    assert naive["purchase"] > naive["checkout_start"]  # raw counts would increase
    table = fn.stage_table(fn.session_depth(events))
    assert table["reached"].tolist() == [3, 3, 2, 1, 1]


def test_session_funnel_matches_raw_event_counts_on_generated_data(tables):
    depth = fn.session_depth(tables["funnel_events"])
    table = fn.stage_table(depth)
    raw = tables["funnel_events"]["event_type"].value_counts().reindex(FUNNEL_STAGES)
    assert table["reached"].tolist() == raw.tolist()  # the generator never skips stages
    report = fn.funnel_integrity(tables["sessions"], tables["funnel_events"])
    assert all(v == 0 for k, v in report.items() if k not in ("sessions", "events"))


def test_session_frame_splits_prospects_from_customers_and_counts_visits(tables):
    start, end = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-03-01")
    frame = fn.session_frame(tables, start, end)
    assert frame["session_start"].between(start, end, inclusive="left").all()
    prospects = frame["visitor_type"] == "prospect"
    assert (frame.loc[prospects, "customer_id"].isna()).all()
    assert frame.loc[~prospects, "visit_number"].isna().all()
    assert set(frame.loc[prospects, "visit_number"]) == set(fn.VISIT_LEVELS)
    # A person's first prospect session in all history is their "1st visit".
    s = tables["sessions"]
    firsts = s.loc[s["customer_id"].isna()].sort_values("session_start").drop_duplicates(
        "prospect_id")["session_id"]
    first_in_frame = frame.loc[frame["visit_number"] == "1st visit", "session_id"]
    assert set(first_in_frame) <= set(firsts)
    # Customers convert far better per session than prospects.
    purchase = frame["depth"] == 5
    assert purchase[~prospects].mean() > 2 * purchase[prospects].mean()


def test_segment_funnel_is_monotone_and_partitions_sessions(tables):
    frame = fn.session_frame(tables, pd.Timestamp("2024-03-01"), pd.Timestamp("2025-03-01"))
    prospect = frame.loc[frame["visitor_type"] == "prospect"]
    seg = fn.segment_funnel(prospect)
    for dim in fn.SEGMENT_LEVELS:
        part = seg.loc[seg["dimension"] == dim]
        assert part["sessions"].sum() == len(prospect), dim
        assert part["reached_purchase"].sum() == int((prospect["depth"] == 5).sum()), dim
    counts = seg[[f"reached_{s}" for s in FUNNEL_STAGES]].to_numpy()
    assert (np.diff(counts, axis=1) <= 0).all()


def test_dropoff_opportunity_by_hand():
    seg = pd.DataFrame([
        {"dimension": "device_type", "level": "desktop", "reached_session_start": 100,
         "reached_product_view": 80, "reached_add_to_cart": 40, "reached_checkout_start": 30,
         "reached_purchase": 24},
        {"dimension": "device_type", "level": "mobile", "reached_session_start": 200,
         "reached_product_view": 160, "reached_add_to_cart": 80, "reached_checkout_start": 40,
         "reached_purchase": 28},
    ])
    for i, step in enumerate(fn.STEPS):
        prev, cur = f"reached_{FUNNEL_STAGES[i]}", f"reached_{FUNNEL_STAGES[i + 1]}"
        seg[f"rate_{step}"] = seg[cur] / seg[prev]
    opp = fn.dropoff_opportunity(seg, "device_type", "desktop").set_index("step")
    checkout = opp.loc["add_to_cart->checkout_start"]
    # Mobile: 80 carts at 50% vs desktop 75% -> +20 checkouts, x 70% downstream = +14 purchases.
    assert checkout["extra_step_completions"] == pytest.approx(20)
    assert checkout["purchase_equivalents"] == pytest.approx(14)
    purchase = opp.loc["checkout_start->purchase"]
    assert purchase["extra_step_completions"] == pytest.approx(40 * (0.8 - 0.7))
    assert opp.loc["session_start->product_view", "extra_step_completions"] == 0


def test_prospect_funnel_uses_a_fixed_window_after_lead_creation(tables):
    start, end = pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-01")
    leads = fn.prospect_funnel(tables, start, end, window_days=30)
    p = tables["prospects"]
    assert len(leads) == int(p["created_at"].between(start, end, inclusive="left").sum())
    assert (leads["depth"] >= 1).all()  # every lead's first session is at creation
    # Purchase within the window matches first-order timing from the customers table.
    c = leads.merge(tables["customers"][["prospect_id", "customer_since"]], on="prospect_id",
                    how="left")
    within = (c["customer_since"] - c["created_at"]) < pd.Timedelta(days=30)
    assert ((c["depth"] == 5) == within.fillna(False)).mean() > 0.99
    longer = fn.prospect_funnel(tables, start, end, window_days=90)
    assert (longer["depth"].to_numpy() >= leads["depth"].to_numpy()).all()
