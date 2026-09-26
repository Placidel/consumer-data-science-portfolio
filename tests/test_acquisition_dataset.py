"""Section 01 dataset: target construction, feature cutoff logic and leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.acquisition.dataset import (
    DEFAULT_SPLIT,
    FEATURES,
    FORBIDDEN_FEATURES,
    KEY_COLUMNS,
    TARGET,
    SplitPlan,
    build_dataset,
    build_features,
    build_run,
    leakage_audit,
    monthly_runs,
    open_leads,
)
from northstar.timeline import DEFAULT_CUTOFF, snapshot

C = pd.Timestamp("2025-07-01")
D = pd.Timedelta(days=1)
M = pd.Timedelta(minutes=1)


def _prospect(pid: str, created: pd.Timestamp, channel: str = "paid_search") -> dict:
    return {"prospect_id": pid, "created_at": created, "acquisition_channel": channel,
            "campaign_id": "CMP001", "region": "west", "age_band": "25-34",
            "income_band": "middle", "device_type": "mobile", "email_opt_in": True}


@pytest.fixture
def toy_tables() -> dict[str, pd.DataFrame]:
    """Hand-built leads whose correct pipeline membership, labels and features are known."""
    prospects = pd.DataFrame([
        _prospect("P1", C - 10 * D),       # open; converts C + 5d        -> label 1
        _prospect("P2", C - 10 * D),       # converted C - 1d             -> not in pipeline
        _prospect("P3", C - 100 * D),      # older than the pipeline window
        _prospect("P4", C + D),            # created after the cutoff
        _prospect("P5", C - 5 * D),        # converts exactly at cutoff   -> label 1
        _prospect("P6", C - 5 * D),        # converts exactly at C + 30d  -> label 0
        _prospect("P7", C - 90 * D),       # created exactly at window start; never converts
        _prospect("P8", C),                # created exactly at the cutoff -> not yet a lead
    ])
    customers = pd.DataFrame({
        "customer_id": ["C1", "C2", "C5", "C6"],
        "prospect_id": ["P1", "P2", "P5", "P6"],
        "customer_since": [C + 5 * D, C - D, C, C + 30 * D],
    })
    sessions = pd.DataFrame([
        # session_id, prospect, customer, start, platform, device, pages
        ("S1", "P1", None, C - 10 * D, "web", "mobile", 3),       # creation visit
        ("S2", "P1", None, C - 3 * D, "app", "mobile", 6),        # reached add_to_cart
        ("S3", "P1", None, C - M, "web", "desktop", 4),           # straddles the cutoff
        ("S4", "P1", "C1", C + 5 * D, "web", "mobile", 9),        # purchase session (future)
        ("S5", "P2", None, C - 10 * D, "web", "mobile", 2),
        ("S6", "P2", "C2", C - 2 * D, "web", "mobile", 8),        # P2's post-conversion visit
    ], columns=["session_id", "prospect_id", "customer_id", "session_start", "platform",
                "device_type", "pages_viewed"])
    events = [
        ("S1", C - 10 * D, 1), ("S1", C - 10 * D + M, 2),
        ("S2", C - 3 * D, 1), ("S2", C - 3 * D + M, 2), ("S2", C - 3 * D + 2 * M, 3),
        ("S3", C - M, 1), ("S3", C + M, 2), ("S3", C + 2 * M, 3), ("S3", C + 3 * M, 4),
        ("S4", C + 5 * D, 1), ("S4", C + 5 * D + M, 5),
        ("S5", C - 10 * D, 1), ("S6", C - 2 * D, 1),
    ]
    funnel = pd.DataFrame(events, columns=["session_id", "event_ts", "stage_number"])
    funnel.insert(0, "event_id", [f"E{i}" for i in range(len(funnel))])
    touches = pd.DataFrame([
        # prospect, customer, type, at, opened, clicked
        ("P1", None, "ad_click", C - 10 * D, True, True),          # sourcing touch (ignored)
        ("P1", None, "email", C - 2 * D, True, True),              # nurture, engaged
        ("P1", None, "email", C - 9 * D, False, False),            # nurture, ignored
        ("P1", None, "ad_click", C - D, True, True),               # retargeting click
        ("P1", None, "email", C + D, True, True),                  # future: must be ignored
        ("P5", None, "email", C - 4 * D, True, False),
    ], columns=["prospect_id", "customer_id", "touch_type", "touch_at", "opened", "clicked"])
    return {"prospects": prospects, "customers": customers, "sessions": sessions,
            "funnel_events": funnel, "marketing_touches": touches}


def test_open_pipeline_membership_follows_cutoff_rules(toy_tables):
    leads = open_leads(toy_tables["prospects"], toy_tables["customers"], C)
    assert sorted(leads["prospect_id"]) == ["P1", "P5", "P6", "P7"]


def test_target_uses_only_the_future_window_with_half_open_bounds(toy_tables):
    run = build_run(toy_tables, C).set_index("prospect_id")
    assert run[TARGET].to_dict() == {"P1": 1, "P5": 1, "P6": 0, "P7": 0}
    # A later horizon captures P6 (converts at exactly C + 30d) as well.
    longer = build_run(toy_tables, C, horizon_days=31).set_index("prospect_id")
    assert longer.loc["P6", TARGET] == 1


def test_features_only_see_pre_cutoff_events(toy_tables):
    f = build_features(toy_tables, C).set_index("prospect_id").loc["P1"]
    assert f["lead_age_days"] == pytest.approx(10)
    assert f["sessions_total"] == 3  # S1, S2, S3 (S4 is after the cutoff)
    assert f["sessions_7d"] == 2
    assert f["days_since_last_session"] == pytest.approx(1 / 1440)
    # S3 started before the cutoff, but its later stages happened after it: only stage 1 counts.
    assert f["max_stage_reached"] == 3  # from S2
    assert f["cart_sessions"] == 1
    assert f["checkout_sessions"] == 0
    assert f["pages_viewed"] == 3 + 6 + 4
    assert f["app_session_share"] == pytest.approx(1 / 3)
    assert f["mobile_session_share"] == pytest.approx(2 / 3)
    # Sourcing touch and the future email are excluded.
    assert (f["emails_received"], f["emails_opened"], f["emails_clicked"]) == (2, 1, 1)
    assert f["email_open_rate"] == pytest.approx(0.5)
    assert f["retarget_clicks"] == 1


def test_leads_without_activity_get_zero_counts_not_nan(toy_tables):
    f = build_features(toy_tables, C).set_index("prospect_id")
    assert not f[list(FEATURES)].isna().any().any()
    p7 = f.loc["P7"]
    assert p7["sessions_total"] == 0 and p7["emails_received"] == 0
    assert p7["days_since_last_session"] == pytest.approx(p7["lead_age_days"])


def test_feature_schema_has_no_outcome_columns():
    assert not set(FEATURES) & FORBIDDEN_FEATURES
    assert TARGET not in FEATURES


def test_features_are_invariant_to_rows_after_the_cutoff(tables):
    full = build_features(tables, DEFAULT_CUTOFF)
    truncated = build_features(snapshot(tables, DEFAULT_CUTOFF), DEFAULT_CUTOFF)
    pd.testing.assert_frame_equal(full, truncated)

    # Scrambling future behavior (and future conversions) must not move any feature.
    mutated = {name: df.copy() for name, df in tables.items()}
    future = mutated["sessions"]["session_start"] >= DEFAULT_CUTOFF
    mutated["sessions"].loc[future, "pages_viewed"] *= 10
    mutated["customers"] = mutated["customers"].loc[
        mutated["customers"]["customer_since"] < DEFAULT_CUTOFF]
    mutated["marketing_touches"] = mutated["marketing_touches"].loc[
        mutated["marketing_touches"]["touch_at"] < DEFAULT_CUTOFF]
    pd.testing.assert_frame_equal(full, build_features(mutated, DEFAULT_CUTOFF))


def test_pipeline_leads_carry_no_post_conversion_information(tables):
    run = build_run(tables, DEFAULT_CUTOFF)
    customers = tables["customers"].set_index("prospect_id")["customer_since"]
    since = run["prospect_id"].map(customers)
    assert not (since < DEFAULT_CUTOFF).any()
    view = snapshot(tables, DEFAULT_CUTOFF)
    for name in ("sessions", "marketing_touches"):
        rows = view[name].loc[view[name]["prospect_id"].isin(run["prospect_id"])]
        assert rows["customer_id"].isna().all()
    # Labels are consistent with the recorded first order.
    positive = run.loc[run[TARGET] == 1, "prospect_id"].map(customers)
    assert ((positive >= DEFAULT_CUTOFF)
            & (positive < DEFAULT_CUTOFF + pd.Timedelta(days=30))).all()
    assert 0.01 < run[TARGET].mean() < 0.3


def test_dataset_stacks_runs_with_unique_keys(tables):
    cutoffs = monthly_runs("2025-05-01", "2025-07-01")
    data = build_dataset(tables, cutoffs)
    assert list(data.columns) == [*KEY_COLUMNS, *FEATURES, TARGET]
    assert not data.duplicated(list(KEY_COLUMNS)).any()
    assert set(data["run_cutoff"]) == set(cutoffs)


def test_default_split_is_time_ordered_without_label_overlap():
    plan = DEFAULT_SPLIT
    horizon = pd.Timedelta(days=plan.horizon_days)
    assert max(plan.fit) + horizon <= min(plan.validation)
    assert max(plan.train) + horizon <= min(plan.holdout)
    assert min(plan.holdout) == DEFAULT_CUTOFF


def test_split_plan_rejects_overlapping_label_windows():
    with pytest.raises(ValueError, match="leak"):
        SplitPlan(fit=monthly_runs("2024-06-01", "2024-09-01"),
                  validation=(pd.Timestamp("2024-09-15"),),
                  holdout=(pd.Timestamp("2024-12-01"),))
    with pytest.raises(ValueError, match="past the end"):
        SplitPlan(fit=monthly_runs("2024-06-01", "2024-09-01"),
                  validation=(pd.Timestamp("2024-11-01"),),
                  holdout=(pd.Timestamp("2025-12-15"),))


SMALL_PLAN = SplitPlan(fit=monthly_runs("2025-01-01", "2025-03-01"),
                       validation=(pd.Timestamp("2025-04-01"),),
                       holdout=monthly_runs("2025-07-01", "2025-08-01"))


@pytest.fixture(scope="module")
def small_dataset(tables):
    return build_dataset(tables, SMALL_PLAN.fit + SMALL_PLAN.validation + SMALL_PLAN.holdout)


def test_leakage_audit_passes_on_clean_data(tables, small_dataset):
    audit = leakage_audit(tables, small_dataset, SMALL_PLAN)
    assert audit["passed"], audit
    assert audit["details"]["customer_linked_rows"] == 0


def test_leakage_audit_flags_a_label_proxy_feature(tables, small_dataset):
    leaky = small_dataset.copy()
    leaky["cart_sessions_7d"] = leaky[TARGET] * 3 + np.random.default_rng(0).random(len(leaky))
    audit = leakage_audit(tables, leaky, SMALL_PLAN)
    assert not audit["passed"]
    assert not audit["checks"]["no_single_feature_suspiciously_predictive"]
    assert audit["details"]["most_predictive_single_feature"] == "cart_sessions_7d"


def test_leakage_audit_flags_already_converted_leads(tables, small_dataset):
    customers = tables["customers"]
    early = customers.loc[customers["customer_since"] < SMALL_PLAN.holdout[0], "prospect_id"]
    tampered = small_dataset.copy()
    row = tampered.loc[tampered["run_cutoff"] == SMALL_PLAN.holdout[0]].index[0]
    tampered.loc[row, "prospect_id"] = early.iloc[-1]
    audit = leakage_audit(tables, tampered, SMALL_PLAN)
    assert not audit["checks"]["no_pipeline_lead_converted_before_cutoff"]
    assert not audit["checks"]["features_unchanged_when_future_rows_removed"]
    assert not audit["passed"]
