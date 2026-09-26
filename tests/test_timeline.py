from __future__ import annotations

import pandas as pd
import pytest

from northstar.timeline import (
    DATA_END,
    DATA_START,
    DEFAULT_CUTOFF,
    TIME_COLUMNS,
    PredictionWindow,
    events_before,
    events_between,
    rolling_cutoffs,
    snapshot,
)
from northstar.validation import validate_foreign_keys


def test_default_cutoff_leaves_room_for_history_and_outcomes():
    assert (DEFAULT_CUTOFF - DATA_START).days >= 365
    PredictionWindow(DEFAULT_CUTOFF, horizon_days=180)  # a 180-day label fits


def test_prediction_window_boundaries_do_not_overlap():
    w = PredictionWindow("2025-04-01", horizon_days=90, lookback_days=30)
    assert w.feature_start == pd.Timestamp("2025-03-02")
    assert w.feature_end == w.label_start == pd.Timestamp("2025-04-01")
    assert w.label_end == pd.Timestamp("2025-06-30")
    ts = pd.Series(pd.to_datetime(["2025-03-01 00:00:00", "2025-03-31 23:59:59",
                                   "2025-04-01 00:00:00", "2025-06-29 00:00:00",
                                   "2025-06-30 00:00:00"]))
    assert w.feature_mask(ts).tolist() == [False, True, False, False, False]
    assert w.label_mask(ts).tolist() == [False, False, True, True, False]
    assert not (w.feature_mask(ts) & w.label_mask(ts)).any()


@pytest.mark.parametrize(
    ("cutoff", "horizon"),
    [("2023-06-01", 30), ("2026-03-01", 30), ("2025-12-01", 90), ("2025-01-01", 0)],
)
def test_prediction_window_rejects_invalid_or_censored_windows(cutoff, horizon):
    with pytest.raises(ValueError):
        PredictionWindow(cutoff, horizon_days=horizon)


def test_lookback_is_clipped_to_data_start():
    w = PredictionWindow("2024-02-01", horizon_days=30, lookback_days=365)
    assert w.feature_start == DATA_START


def test_events_before_is_strict_and_between_is_half_open():
    df = pd.DataFrame({"ts": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])})
    assert events_before(df, "ts", "2025-01-02")["ts"].tolist() == [pd.Timestamp("2025-01-01")]
    got = events_between(df, "ts", "2025-01-02", "2025-01-03")["ts"].tolist()
    assert got == [pd.Timestamp("2025-01-02")]


def test_rolling_cutoffs():
    cuts = rolling_cutoffs("2025-01-01", "2025-03-01", 28)
    assert cuts == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-29"),
                    pd.Timestamp("2025-02-26")]
    with pytest.raises(ValueError):
        rolling_cutoffs("2025-03-01", "2025-01-01", 7)


def test_snapshot_contains_only_pre_cutoff_information(tables):
    cutoff = pd.Timestamp("2025-03-15")
    snap = snapshot(tables, cutoff)
    for name, col in TIME_COLUMNS.items():
        assert (snap[name][col] < cutoff).all(), name
    assert set(snap["order_lines"]["order_id"]) <= set(snap["orders"]["order_id"])
    resolved = snap["support_contacts"]["resolved_at"].dropna()
    assert (resolved < cutoff).all()
    # A snapshot is itself referentially complete: nothing points at a future entity.
    assert validate_foreign_keys(snap) == []
    assert snap["products"].equals(tables["products"])


def test_snapshot_masks_resolution_not_yet_known(tables):
    sc = tables["support_contacts"]
    pending = sc.loc[sc["resolved_at"].notna() & sc["csat_score"].notna()
                     & (sc["resolved_at"] - sc["contact_ts"] > pd.Timedelta(hours=2))].iloc[0]
    cutoff = pending["contact_ts"] + (pending["resolved_at"] - pending["contact_ts"]) / 2
    row = snapshot(tables, cutoff)["support_contacts"].set_index("contact_id").loc[
        pending["contact_id"]]
    assert pd.isna(row["resolved_at"]) and pd.isna(row["csat_score"])
    # The source tables are untouched.
    assert tables["support_contacts"]["csat_score"].notna().sum() == sc["csat_score"].notna().sum()


def test_snapshot_does_not_reveal_future_conversions(tables):
    """Pre-cutoff rows of prospects who convert later must not carry a customer_id."""
    cutoff = DEFAULT_CUTOFF
    snap = snapshot(tables, cutoff)
    future = tables["customers"].loc[tables["customers"]["customer_since"] >= cutoff]
    for name in ("sessions", "marketing_touches"):
        rows = snap[name].loc[snap[name]["prospect_id"].isin(future["prospect_id"])]
        assert len(rows) > 0
        assert rows["customer_id"].isna().all(), name
    assert snap["customers"]["customer_since"].max() < cutoff
    assert cutoff < DATA_END
