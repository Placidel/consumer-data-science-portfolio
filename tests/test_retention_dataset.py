"""Section 02 dataset: churn definition, time windows, point-in-time features and leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.retention.dataset import (
    ACTIVE_DAYS,
    DEFAULT_SPLIT,
    FEATURES,
    FORBIDDEN_FEATURES,
    HORIZON_DAYS,
    KEY_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    VALUE,
    SplitPlan,
    active_customers,
    build_dataset,
    build_features,
    build_run,
    leakage_audit,
    monthly_runs,
)
from northstar.timeline import DATA_END, DEFAULT_CUTOFF, snapshot

C = pd.Timestamp("2025-07-01")
D = pd.Timedelta(days=1)
H = pd.Timedelta(hours=1)


def _customer(cid: str, since: pd.Timestamp) -> dict:
    return {"customer_id": cid, "prospect_id": "P" + cid[1:], "customer_since": since,
            "acquisition_channel": "email", "region": "west", "age_band": "25-34",
            "income_band": "middle", "device_type": "mobile", "email_opt_in": True}


@pytest.fixture
def toy_tables() -> dict[str, pd.DataFrame]:
    """Hand-built customers whose active status, labels and features are known exactly."""
    customers = pd.DataFrame([
        _customer("C1", C - 100 * D),   # active; orders again at C + 5d        -> label 0
        _customer("C2", C - 300 * D),   # last order C - 200d                   -> not active
        _customer("C3", C - ACTIVE_DAYS * D),  # only order exactly at window start -> active, 1
        _customer("C4", C - 1 * D),     # next order exactly at C + 90d         -> label 1
        _customer("C5", C - 30 * D),    # next order exactly at the cutoff      -> label 0
        _customer("C6", C + 3 * D),     # first order after the cutoff          -> not scored
    ])
    orders = pd.DataFrame([
        # order_id, customer, ts, channel, session, gross, discount
        ("O1", "C1", C - 100 * D, "web", "S1", 100.0, 10.0),
        ("O2", "C1", C - 10 * D, "store", None, 50.0, 0.0),
        ("O3", "C1", C + 5 * D, "web", "S9", 80.0, 0.0),
        ("O4", "C2", C - 300 * D, "web", None, 40.0, 0.0),
        ("O5", "C2", C - 200 * D, "web", None, 40.0, 0.0),
        ("O6", "C3", C - ACTIVE_DAYS * D, "app", None, 60.0, 0.0),
        ("O7", "C4", C - 1 * D, "web", None, 30.0, 0.0),
        ("O8", "C4", C + HORIZON_DAYS * D, "web", None, 30.0, 0.0),
        ("O9", "C5", C - 30 * D, "web", None, 20.0, 0.0),
        ("O10", "C5", C, "web", None, 20.0, 0.0),
        ("O11", "C6", C + 3 * D, "web", None, 20.0, 0.0),
    ], columns=["order_id", "customer_id", "order_ts", "order_channel", "session_id",
                "gross_amount", "discount_amount"])
    orders["net_amount"] = orders["gross_amount"] - orders["discount_amount"]
    orders["campaign_id"] = None
    orders["item_count"] = 1
    products = pd.DataFrame({"product_id": ["A", "B"], "category": ["apparel", "home"],
                             "list_price": [100.0, 50.0], "unit_cost": [40.0, 20.0]})
    lines = pd.DataFrame({
        "order_line_id": [f"L{i}" for i in range(len(orders))],
        "order_id": orders["order_id"],
        "product_id": ["A", "B", "A", "A", "A", "B", "A", "A", "A", "A", "A"],
        "quantity": 1,
        "unit_price": orders["gross_amount"],
        "discount_amount": orders["discount_amount"],
        "net_amount": orders["net_amount"],
    })
    sessions = pd.DataFrame([
        ("S1", "C1", C - 100 * D),        # purchase session of O1 (not browsing)
        ("S2", "C1", C - 20 * D),         # browse, within 30d
        ("S3", "C1", C - 60 * D),         # browse, within 90d only
        ("S4", "C1", C - H),              # browse, latest visit
        ("S9", "C1", C + 5 * D),          # future purchase session
        ("S5", None, C - 40 * D),         # pre-conversion session of someone else
    ], columns=["session_id", "customer_id", "session_start"])
    touches = pd.DataFrame([
        ("C1", "email", C - 7 * D, True, True),
        ("C1", "email", C - 14 * D, True, False),
        ("C1", "email", C - 21 * D, False, False),
        ("C1", "email", C - 120 * D, True, True),    # older than 90 days
        ("C1", "email", C + 1 * D, True, True),      # future
        ("C1", "ad_click", C - 3 * D, True, True),   # not an email
    ], columns=["customer_id", "touch_type", "touch_at", "opened", "clicked"])
    subs = pd.DataFrame([
        ("C1", C - 50 * D, "subscribe", "monthly"),
        ("C1", C - 20 * D, "renew", "monthly"),      # covers the cutoff
        ("C1", C - 5 * D, "cancel", "monthly"),      # pending cancellation, visible
        ("C4", C - 1 * D + H, "subscribe", "annual"),
        ("C3", C - 170 * D, "subscribe", "monthly"),  # lapsed long ago
    ], columns=["customer_id", "event_ts", "event_type", "plan"])
    support = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C4"],
        "contact_ts": [C - 30 * D, C - 10 * D, C - 200 * D, C - 2 * H],
        # resolved slowly with low CSAT; resolved quickly; old; resolved after the cutoff
        "resolved_at": [C - 28 * D, C - 10 * D + H, C - 199 * D, C + D],
        "csat_score": [1.0, 5.0, 1.0, 1.0],
    })
    return {"customers": customers, "orders": orders, "order_lines": lines,
            "products": products, "sessions": sessions, "marketing_touches": touches,
            "subscription_events": subs, "support_contacts": support}


def test_active_base_uses_half_open_observation_window(toy_tables):
    ids = active_customers(toy_tables["orders"], C)
    assert list(ids) == ["C1", "C3", "C4", "C5"]
    # One day later C3's only order falls out of the 180-day window.
    assert "C3" not in active_customers(toy_tables["orders"], C + D)


def test_churn_label_uses_only_the_prediction_window(toy_tables):
    run = build_run(toy_tables, C).set_index("customer_id")
    assert run[TARGET].to_dict() == {"C1": 0, "C3": 1, "C4": 1, "C5": 0}
    # An order exactly at cutoff + horizon is outside the window; one more day captures it.
    longer = build_run(toy_tables, C, horizon_days=HORIZON_DAYS + 1).set_index("customer_id")
    assert longer.loc["C4", TARGET] == 0


def test_features_only_see_pre_cutoff_events(toy_tables):
    f = build_features(toy_tables, C).set_index("customer_id")
    c1 = f.loc["C1"]
    assert c1["tenure_days"] == pytest.approx(100)
    assert c1["days_since_last_order"] == pytest.approx(10)
    assert (c1["orders_total"], c1["orders_90d"], c1["orders_prev_90d"]) == (2, 1, 1)
    assert c1["net_revenue_180d"] == pytest.approx(90 + 50)
    assert c1["avg_order_value"] == pytest.approx(70)
    assert c1["discount_share"] == pytest.approx(10 / 150)
    assert c1["first_order_discounted"] == 1
    assert c1["category_count"] == 2
    assert c1["store_order_share"] == pytest.approx(0.5)
    # Browsing excludes the purchase session S1 and the future session S9.
    assert (c1["browse_sessions_30d"], c1["browse_sessions_90d"]) == (2, 3)
    assert c1["days_since_last_session"] == pytest.approx(1 / 24)
    # Three emails in the last 90 days (the ad click, old and future emails are excluded).
    assert c1["emails_received_90d"] == 3
    assert c1["email_open_rate_90d"] == pytest.approx(2 / 3)
    assert c1["email_clicks_90d"] == 1
    assert (c1["plus_member"], c1["plus_cancelled_180d"]) == (1, 1)
    assert c1["support_contacts_180d"] == 2
    assert (c1["low_csat_contacts_180d"], c1["slow_resolution_contacts_180d"]) == (1, 1)
    assert c1["open_contacts"] == 0
    # Gross margin of pre-cutoff orders in the last 180 days: (90 - 40) + (50 - 20).
    assert c1[VALUE] == pytest.approx(80)

    # C5's order at exactly the cutoff belongs to the outcome window, not the features.
    assert f.loc["C5", "orders_total"] == 1
    assert f.loc["C5", "days_since_last_order"] == pytest.approx(30)


def test_outcomes_not_known_at_cutoff_are_masked(toy_tables):
    c4 = build_features(toy_tables, C).set_index("customer_id").loc["C4"]
    # Contact resolved after the cutoff: still open, and its (future) CSAT of 1 is invisible.
    assert c4["open_contacts"] == 1
    assert c4["low_csat_contacts_180d"] == 0
    assert c4["plus_member"] == 1
    # C3's monthly membership lapsed long before the cutoff.
    c3 = build_features(toy_tables, C).set_index("customer_id").loc["C3"]
    assert c3["plus_member"] == 0
    # Store-only or session-less customers fall back to their tenure, never NaN.
    assert c3["days_since_last_session"] == pytest.approx(c3["tenure_days"])


def test_churn_label_cannot_leak_into_features(tables):
    """Rewriting the entire future changes labels but leaves every feature untouched."""
    cutoff = DEFAULT_CUTOFF
    run = build_run(tables, cutoff)
    future_free = dict(tables)
    future_free["orders"] = tables["orders"].loc[tables["orders"]["order_ts"] < cutoff]
    rerun = build_run(future_free, cutoff)
    cols = [*KEY_COLUMNS, *FEATURES, VALUE]
    pd.testing.assert_frame_equal(run[cols], rerun[cols])
    assert rerun[TARGET].all()  # with no future orders everyone "churns"
    assert 0.2 < run[TARGET].mean() < 0.8
    # The same holds for every table: features equal those built from the snapshot alone.
    truncated = snapshot(tables, cutoff)
    pd.testing.assert_frame_equal(build_features(tables, cutoff)[cols],
                                  build_features(truncated, cutoff)[cols])


def test_feature_schema_has_no_outcome_columns_and_no_missing_values(tables):
    assert not set(FEATURES) & FORBIDDEN_FEATURES
    assert TARGET not in FEATURES and VALUE not in FEATURES
    run = build_run(tables, DEFAULT_CUTOFF)
    assert not run[list(FEATURES)].isna().any().any()
    assert run["customer_id"].is_unique
    assert (run[list(NUMERIC_FEATURES)] >= 0).all().all()


def test_runs_need_full_observation_history_and_complete_labels(tables):
    with pytest.raises(ValueError, match="history"):
        build_run(tables, pd.Timestamp("2024-06-01"))
    with pytest.raises(ValueError, match="right-censored"):
        build_run(tables, DATA_END - pd.Timedelta(days=HORIZON_DAYS - 1))


def test_default_split_windows_are_time_ordered_and_documented():
    plan = DEFAULT_SPLIT
    horizon = pd.Timedelta(days=plan.horizon_days)
    assert plan.horizon_days == HORIZON_DAYS == 90
    assert max(plan.fit) + horizon <= min(plan.validation)
    assert max(plan.train) + horizon <= min(plan.holdout)
    assert max(plan.holdout) + horizon <= DATA_END
    assert min(plan.holdout) == DEFAULT_CUTOFF
    assert min(plan.fit) - pd.Timedelta(days=ACTIVE_DAYS) >= pd.Timestamp("2024-01-01")
    # A split whose training labels overlap the holdout is rejected.
    with pytest.raises(ValueError, match="leak"):
        SplitPlan(fit=monthly_runs("2024-07-01", "2024-12-01"),
                  validation=monthly_runs("2025-03-01", "2025-05-01"),
                  holdout=monthly_runs("2025-07-01", "2025-10-01"), horizon_days=HORIZON_DAYS)


SMALL_PLAN = SplitPlan(fit=monthly_runs("2024-07-01", "2024-09-01"),
                       validation=(pd.Timestamp("2025-01-01"),),
                       holdout=(pd.Timestamp("2025-07-01"),), horizon_days=HORIZON_DAYS)


@pytest.fixture(scope="module")
def small_dataset(tables):
    return build_dataset(tables, SMALL_PLAN.fit + SMALL_PLAN.validation + SMALL_PLAN.holdout)


def test_leakage_audit_passes_on_clean_data(tables, small_dataset):
    audit = leakage_audit(tables, small_dataset, SMALL_PLAN)
    assert audit["passed"], audit
    assert audit["details"]["min_gap_between_last_event_and_cutoff_seconds"] > 0


def test_leakage_audit_catches_a_label_derived_feature(tables, small_dataset):
    leaky = small_dataset.copy()
    # Simulate a bug: "orders in the next 90 days" smuggled in under a legitimate feature name.
    leaky["orders_90d"] = np.where(leaky[TARGET] == 1, 0.0, 1.0 + leaky["orders_90d"])
    audit = leakage_audit(tables, leaky, SMALL_PLAN)
    assert not audit["passed"]
    assert not audit["checks"]["features_unchanged_when_future_rows_removed"]
    assert not audit["checks"]["no_single_feature_suspiciously_predictive"]
    assert audit["details"]["most_predictive_single_feature"] == "orders_90d"


def test_leakage_audit_catches_wrong_labels_and_inactive_customers(tables, small_dataset):
    broken = small_dataset.copy()
    broken.loc[broken.index[:5], TARGET] = 1 - broken.loc[broken.index[:5], TARGET]
    audit = leakage_audit(tables, broken, SMALL_PLAN)
    assert not audit["checks"]["labels_only_from_prediction_window"]
    assert audit["details"]["label_mismatches"] == 5

    lapsed = tables["orders"].groupby("customer_id")["order_ts"].max()
    lapsed = lapsed[lapsed < SMALL_PLAN.holdout[0] - pd.Timedelta(days=ACTIVE_DAYS)].index[0]
    extra = small_dataset.loc[small_dataset["run_cutoff"] == SMALL_PLAN.holdout[0]].head(1).copy()
    extra["customer_id"] = lapsed
    audit = leakage_audit(tables, pd.concat([small_dataset, extra], ignore_index=True),
                          SMALL_PLAN)
    assert not audit["checks"]["every_scored_customer_active_at_cutoff"]
