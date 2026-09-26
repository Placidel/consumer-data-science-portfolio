"""Section 04 dataset: target windows, point-in-time features, spend separation, leakage guards."""

from __future__ import annotations

import pandas as pd
import pytest

from northstar.acquisition.dataset import SplitPlan, monthly_runs
from northstar.revenue.dataset import (
    DEFAULT_SPLIT,
    FEATURES,
    FORBIDDEN_FEATURES,
    HISTORICAL_SPEND,
    HORIZON_DAYS,
    KEY_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    build_dataset,
    build_features,
    build_run,
    customer_base,
    leakage_audit,
)
from northstar.timeline import DATA_END, DATA_START, DEFAULT_CUTOFF, snapshot

C = pd.Timestamp("2025-07-01")
D = pd.Timedelta(days=1)
H = pd.Timedelta(hours=1)
N_FILLER = 20


def _customer(cid: str, since: pd.Timestamp) -> dict:
    return {"customer_id": cid, "prospect_id": "P" + cid[1:], "customer_since": since,
            "acquisition_channel": "email", "region": "west", "age_band": "25-34",
            "income_band": "middle", "device_type": "mobile", "email_opt_in": True}


@pytest.fixture
def toy_tables() -> dict[str, pd.DataFrame]:
    """Hand-built customers with known windows, plus repeat-buying fillers for the CLV fits."""
    customers = [
        _customer("C1", C - 100 * D),   # two orders before, two inside the window
        _customer("C2", C - 450 * D),   # lapsed: nothing in the last 365 days, stays in the base
        _customer("C3", C - 50 * D),    # next order exactly at C + 180d -> outside the window
        _customer("C4", C - 30 * D),    # next order exactly at the cutoff -> inside the window
        _customer("C5", C),             # first order exactly at the cutoff -> not in the base
        _customer("C6", C + 3 * D),     # acquired after the cutoff -> not in the base
    ]
    orders = [
        # order_id, customer, ts, channel, session, gross, discount, items
        ("O1", "C1", C - 100 * D, "web", "S1", 100.0, 10.0, 2),
        ("O2", "C1", C - 10 * D, "store", None, 50.0, 0.0, 1),
        ("O3", "C1", C + 5 * D, "web", "S9", 80.0, 0.0, 1),
        ("O4", "C1", C + (HORIZON_DAYS - 1) * D, "app", None, 40.0, 0.0, 1),
        ("O5", "C2", C - 450 * D, "web", None, 60.0, 0.0, 1),
        ("O6", "C2", C - 380 * D, "web", None, 30.0, 0.0, 1),
        ("O7", "C3", C - 50 * D, "app", None, 45.0, 5.0, 1),
        ("O8", "C3", C + HORIZON_DAYS * D, "web", None, 70.0, 0.0, 1),
        ("O9", "C4", C - 30 * D, "web", None, 25.0, 0.0, 1),
        ("O10", "C4", C, "web", None, 35.0, 0.0, 1),
        ("O11", "C5", C, "web", None, 20.0, 0.0, 1),
        ("O12", "C6", C + 3 * D, "web", None, 20.0, 0.0, 1),
    ]
    for i in range(N_FILLER):
        cid = f"C{100 + i}"
        since = C - (200 + 5 * i) * D
        customers.append(_customer(cid, since))
        for k in range(1 + i % 4):
            orders.append((f"OF{i}_{k}", cid, since + 40 * k * D, "web", None,
                           30.0 + 7 * i + 3 * k, 0.0, 1))
    customers = pd.DataFrame(customers)
    orders = pd.DataFrame(orders, columns=["order_id", "customer_id", "order_ts", "order_channel",
                                           "session_id", "gross_amount", "discount_amount",
                                           "item_count"])
    orders["net_amount"] = orders["gross_amount"] - orders["discount_amount"]
    orders["campaign_id"] = None
    products = pd.DataFrame({"product_id": ["A", "B"], "category": ["apparel", "home"],
                             "list_price": [100.0, 50.0], "unit_cost": [40.0, 20.0]})
    lines = pd.DataFrame({
        "order_line_id": [f"L{i}" for i in range(len(orders))],
        "order_id": orders["order_id"],
        "product_id": ["A" if i != 1 else "B" for i in range(len(orders))],
        "quantity": 1,
        "unit_price": orders["gross_amount"],
        "discount_amount": orders["discount_amount"],
        "net_amount": orders["net_amount"],
    })
    sessions = pd.DataFrame([
        ("S1", "C1", C - 100 * D),
        ("S2", "C1", C - 20 * D),
        ("S9", "C1", C + 5 * D),
    ], columns=["session_id", "customer_id", "session_start"])
    touches = pd.DataFrame([
        ("C1", "email", C - 7 * D, True, True),
        ("C1", "email", C + 1 * D, True, True),
    ], columns=["customer_id", "touch_type", "touch_at", "opened", "clicked"])
    subs = pd.DataFrame([("C1", C - 50 * D, "subscribe", "annual")],
                        columns=["customer_id", "event_ts", "event_type", "plan"])
    support = pd.DataFrame({"customer_id": ["C4"], "contact_ts": [C - 2 * H],
                            "resolved_at": [C + D], "csat_score": [1.0]})
    return {"customers": customers, "orders": orders, "order_lines": lines,
            "products": products, "sessions": sessions, "marketing_touches": touches,
            "subscription_events": subs, "support_contacts": support}


def test_customer_base_is_everyone_acquired_strictly_before_the_cutoff(toy_tables):
    ids = customer_base(toy_tables["customers"], C)
    assert {"C1", "C2", "C3", "C4"} <= set(ids)
    assert not {"C5", "C6"} & set(ids)
    assert len(ids) == 4 + N_FILLER
    assert ids.is_monotonic_increasing


def test_target_is_net_revenue_in_the_half_open_outcome_window(toy_tables):
    run = build_run(toy_tables, C)[0].set_index("customer_id")
    assert run.loc["C1", TARGET] == pytest.approx(80 + 40)  # both window orders, incl. day 179
    assert run.loc["C2", TARGET] == 0.0  # lapsed customers stay in the base with zero value
    assert run.loc["C3", TARGET] == 0.0  # the order exactly at C + 180 days is outside
    assert run.loc["C4", TARGET] == pytest.approx(35)  # the order exactly at C is inside
    longer = build_run(toy_tables, C, horizon_days=HORIZON_DAYS + 1)[0].set_index("customer_id")
    assert longer.loc["C3", TARGET] == pytest.approx(70)


def test_historical_spend_aggregates_exclude_the_target_period(toy_tables):
    run = build_run(toy_tables, C)[0].set_index("customer_id")
    c1 = run.loc["C1"]
    assert c1["revenue_total"] == pytest.approx(90 + 50)
    assert (c1["revenue_90d"], c1["revenue_180d"], c1["revenue_365d"]) == pytest.approx(
        (50, 140, 140))
    assert (c1["orders_total"], c1["orders_90d"], c1["orders_180d"]) == (2, 1, 2)
    # The cutoff-day order of C4 is target, not history.
    c4 = run.loc["C4"]
    assert (c4["orders_total"], c4["revenue_total"]) == (1, pytest.approx(25))
    # C2 is lapsed: lifetime spend but nothing in any trailing window.
    c2 = run.loc["C2"]
    assert c2["revenue_total"] == pytest.approx(90)
    assert c2["revenue_365d"] == 0 and c2["orders_365d"] == 0
    # History + target reconcile to the ledger before the label end, for every customer.
    orders = toy_tables["orders"]
    ledger = orders.loc[orders["order_ts"] < C + HORIZON_DAYS * D].groupby("customer_id")[
        "net_amount"].sum()
    pd.testing.assert_series_equal(run["revenue_total"] + run[TARGET],
                                   ledger.reindex(run.index), check_names=False)


def test_rfm_and_engagement_features_use_only_pre_cutoff_rows(toy_tables):
    f = build_features(toy_tables, C)[0].set_index("customer_id")
    c1 = f.loc["C1"]
    assert c1["tenure_days"] == pytest.approx(100)
    assert c1["days_since_last_order"] == pytest.approx(10)
    assert c1["mean_days_between_orders"] == pytest.approx(90)
    assert (c1["avg_order_value"], c1["max_order_value"]) == pytest.approx((70, 90))
    assert c1["items_per_order"] == pytest.approx(1.5)
    assert c1["discount_share"] == pytest.approx(10 / 150)
    assert c1["category_count"] == 2
    assert c1["store_order_share"] == pytest.approx(0.5)
    assert c1["browse_sessions_30d"] == 1  # S2 only: S1 is a purchase session, S9 is future
    assert c1["emails_received_90d"] == 1  # the future email is invisible
    assert c1["plus_member"] == 1
    # One-order customers: the gap falls back to their order age; no missing values anywhere.
    assert f.loc["C3", "mean_days_between_orders"] == pytest.approx(50)
    assert not f[list(FEATURES)].isna().any().any()
    # The support contact resolved after the cutoff does not reveal its low CSAT.
    assert f.loc["C4", "low_csat_contacts_180d"] == 0


def test_clv_features_are_consistent_probabilities_and_expectations(toy_tables):
    f = build_features(toy_tables, C)[0].set_index("customer_id")
    assert f["bgnbd_p_alive"].between(0, 1).all()
    assert (f["bgnbd_expected_orders"] > 0).all() and (f["gg_expected_order_value"] > 0).all()
    assert f["clv_expected_revenue"].to_numpy() == pytest.approx(
        (f["bgnbd_expected_orders"] * f["gg_expected_order_value"]).to_numpy())
    # No repeat purchase yet (x = 0): the BG/NBD model cannot have observed a dropout.
    assert f.loc["C4", "bgnbd_p_alive"] == pytest.approx(1.0)
    # A repeat buyer idle for 380 days is less likely alive than a recent repeat buyer.
    assert f.loc["C2", "bgnbd_p_alive"] < f.loc["C1", "bgnbd_p_alive"]


def test_target_cannot_leak_into_features(tables):
    """Rewriting the entire future changes the target but leaves every feature untouched."""
    run, params = build_run(tables, DEFAULT_CUTOFF)
    future_free = dict(tables)
    future_free["orders"] = tables["orders"].loc[tables["orders"]["order_ts"] < DEFAULT_CUTOFF]
    rerun, params_free = build_run(future_free, DEFAULT_CUTOFF)
    cols = [*KEY_COLUMNS, *FEATURES]
    pd.testing.assert_frame_equal(run[cols], rerun[cols])
    assert params == params_free  # the CLV models were fit on identical history
    assert (rerun[TARGET] == 0).all()
    assert 0.2 < (run[TARGET] > 0).mean() < 0.8
    truncated = snapshot(tables, DEFAULT_CUTOFF)
    pd.testing.assert_frame_equal(build_features(tables, DEFAULT_CUTOFF)[0][cols],
                                  build_features(truncated, DEFAULT_CUTOFF)[0][cols])


def test_feature_schema_has_no_outcome_columns_and_no_missing_values(tables):
    assert not set(FEATURES) & FORBIDDEN_FEATURES
    assert TARGET not in FEATURES
    assert set(HISTORICAL_SPEND) <= set(NUMERIC_FEATURES)
    run, _ = build_run(tables, DEFAULT_CUTOFF)
    assert not run[list(FEATURES)].isna().any().any()
    assert run["customer_id"].is_unique
    assert (run[list(NUMERIC_FEATURES)] >= 0).all().all()
    assert (run[TARGET] >= 0).all()


def test_runs_need_history_and_complete_labels(tables):
    with pytest.raises(ValueError, match="history"):
        build_run(tables, DATA_START + pd.Timedelta(days=60))
    with pytest.raises(ValueError, match="right-censored"):
        build_run(tables, DATA_END - pd.Timedelta(days=HORIZON_DAYS - 1))


def test_default_split_is_purged_and_ends_on_the_default_cutoff():
    plan = DEFAULT_SPLIT
    horizon = pd.Timedelta(days=plan.horizon_days)
    assert plan.horizon_days == HORIZON_DAYS == 180
    assert max(plan.fit) + horizon <= min(plan.validation)
    assert max(plan.train) + horizon <= min(plan.holdout)
    assert max(plan.holdout) + horizon <= DATA_END
    assert plan.holdout == (DEFAULT_CUTOFF,)
    # Any later validation run would let 180-day training labels overlap the holdout.
    with pytest.raises(ValueError, match="leak"):
        SplitPlan(fit=plan.fit, validation=monthly_runs("2025-02-01", "2025-02-01"),
                  holdout=plan.holdout, horizon_days=HORIZON_DAYS)


SMALL_PLAN = SplitPlan(fit=monthly_runs("2024-06-01", "2024-07-01"),
                       validation=(pd.Timestamp("2025-01-01"),),
                       holdout=(DEFAULT_CUTOFF,), horizon_days=HORIZON_DAYS)


@pytest.fixture(scope="module")
def small_dataset(tables):
    data, _ = build_dataset(tables, SMALL_PLAN.fit + SMALL_PLAN.validation + SMALL_PLAN.holdout)
    return data


def test_leakage_audit_passes_on_clean_data(tables, small_dataset):
    audit = leakage_audit(tables, small_dataset, SMALL_PLAN)
    assert audit["passed"], audit
    assert audit["details"]["min_gap_between_last_event_and_cutoff_seconds"] > 0
    assert audit["details"]["spend_reconciliation_mismatches"] == 0


def test_leakage_audit_catches_a_target_derived_feature(tables, small_dataset):
    leaky = small_dataset.copy()
    # Simulate a bug: next-period revenue smuggled in under a legitimate feature name.
    leaky["revenue_90d"] = 0.9 * leaky[TARGET]
    audit = leakage_audit(tables, leaky, SMALL_PLAN)
    assert not audit["passed"]
    assert not audit["checks"]["features_unchanged_when_future_rows_removed"]
    assert not audit["checks"]["no_single_feature_suspiciously_predictive"]
    assert audit["details"]["most_predictive_single_feature"] == "revenue_90d"


def test_leakage_audit_catches_spend_aggregates_that_include_the_target_period(
        tables, small_dataset):
    contaminated = small_dataset.copy()
    buyers = contaminated[TARGET] > 0
    # Simulate a window bug: lifetime spend computed up to the label end instead of the cutoff.
    contaminated.loc[buyers, "revenue_total"] += contaminated.loc[buyers, TARGET]
    audit = leakage_audit(tables, contaminated, SMALL_PLAN)
    assert not audit["checks"]["historical_spend_excludes_target_period"]
    assert audit["checks"]["labels_only_from_prediction_window"]
    assert audit["details"]["spend_reconciliation_mismatches"] == int(buyers.sum())


def test_leakage_audit_catches_wrong_labels_and_future_customers(tables, small_dataset):
    broken = small_dataset.copy()
    broken.loc[broken.index[:5], TARGET] += 10.0
    audit = leakage_audit(tables, broken, SMALL_PLAN)
    assert not audit["checks"]["labels_only_from_prediction_window"]
    assert audit["details"]["label_mismatches"] == 5

    customers = tables["customers"]
    late = customers.loc[customers["customer_since"] >= DEFAULT_CUTOFF, "customer_id"].iloc[0]
    extra = small_dataset.loc[small_dataset["run_cutoff"] == DEFAULT_CUTOFF].head(1).copy()
    extra["customer_id"] = late
    audit = leakage_audit(tables, pd.concat([small_dataset, extra], ignore_index=True),
                          SMALL_PLAN)
    assert not audit["checks"]["every_scored_customer_acquired_before_cutoff"]
