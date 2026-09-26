"""Section 05 series construction: aggregation, week boundaries and the promotion calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.forecasting.series import (
    build_series,
    complete_weeks_end,
    daily_revenue,
    history_before,
    planned_promotions,
    promo_calendar,
    repeat_promotions,
    to_weeks,
    weekly_revenue,
)
from northstar.timeline import DATA_END, DATA_START


def _orders(rows):
    return pd.DataFrame(rows, columns=["order_ts", "net_amount"]).assign(
        order_ts=lambda d: pd.to_datetime(d["order_ts"]))


def test_daily_revenue_reconciles_with_the_order_log(tables):
    daily, _ = build_series(tables)
    assert len(daily) == (DATA_END - DATA_START).days
    assert daily.index.is_monotonic_increasing and daily.index.is_unique
    assert daily.sum() == pytest.approx(tables["orders"]["net_amount"].sum(), rel=1e-12)
    by_day = tables["orders"].groupby(tables["orders"]["order_ts"].dt.normalize())["net_amount"]
    assert daily.loc[by_day.sum().index].to_numpy() == pytest.approx(by_day.sum().to_numpy())


def test_days_without_orders_are_zero_and_out_of_range_orders_are_ignored():
    orders = _orders([("2023-12-31 23:00", 999.0), ("2024-01-01 00:00", 10.0),
                      ("2024-01-03 12:00", 5.0), ("2024-01-03 13:00", 7.0),
                      ("2024-01-05 00:00", 1000.0)])
    daily = daily_revenue(orders, start=pd.Timestamp("2024-01-01"),
                          end=pd.Timestamp("2024-01-05"))
    assert daily.tolist() == [10.0, 0.0, 12.0, 0.0]


def test_sunday_midnight_boundary_and_monday_weeks():
    # 2024-03-10 is a Sunday: 23:59:59 belongs to the week of Monday 2024-03-04.
    orders = _orders([("2024-03-04 00:00:00", 1.0), ("2024-03-10 23:59:59", 2.0),
                      ("2024-03-11 00:00:00", 4.0)])
    daily = daily_revenue(orders, start=pd.Timestamp("2024-03-04"),
                          end=pd.Timestamp("2024-03-18"))
    weekly = weekly_revenue(daily)
    assert weekly["week_start"].tolist() == [pd.Timestamp("2024-03-04"),
                                             pd.Timestamp("2024-03-11")]
    assert weekly["net_revenue"].tolist() == [3.0, 4.0]
    assert (weekly["days"] == 7).all()


def test_partial_weeks_are_dropped_at_both_ends():
    # Starts on a Wednesday and ends on a Wednesday: only the full week in between survives.
    days = pd.date_range("2024-01-03", "2024-01-17", freq="D")  # Wed .. Wed
    daily = pd.Series(1.0, index=days)
    weekly = weekly_revenue(daily)
    assert weekly["week_start"].tolist() == [pd.Timestamp("2024-01-08")]
    assert weekly["net_revenue"].tolist() == [7.0]
    assert complete_weeks_end(daily) == pd.Timestamp("2024-01-15")


def test_full_history_has_104_complete_weeks(tables):
    daily, _ = build_series(tables)
    weekly = weekly_revenue(daily)
    assert len(weekly) == 104 and (weekly["days"] == 7).all()
    assert weekly["week_start"].iloc[0] == DATA_START  # 2024-01-01 is a Monday
    assert complete_weeks_end(daily) == pd.Timestamp("2025-12-29")
    in_weeks = daily.loc[daily.index < complete_weeks_end(daily)].sum()
    assert weekly["net_revenue"].sum() == pytest.approx(in_weeks)


def test_to_weeks_sums_seven_day_blocks_and_checks_length():
    assert to_weeks(np.arange(14.0), 2).tolist() == [21.0, 70.0]
    with pytest.raises(ValueError, match="expected 14"):
        to_weeks(np.arange(13.0), 2)


def test_history_before_excludes_the_origin_day():
    daily = pd.Series(1.0, index=pd.date_range("2024-01-01", periods=10))
    hist = history_before(daily, pd.Timestamp("2024-01-05"))
    assert hist.index.max() == pd.Timestamp("2024-01-04") and len(hist) == 4


def test_planned_promotions_come_only_from_promotion_campaigns(tables):
    promos = planned_promotions(tables["campaigns"])
    campaigns = tables["campaigns"].set_index("campaign_id")
    assert len(promos) == 8
    assert (campaigns.loc[promos["campaign_id"], "objective"] == "promotion").all()
    assert set(promos.columns) == {"campaign_id", "campaign_name", "start_date", "end_date"}


def test_promo_calendar_is_inclusive_of_both_ends():
    promos = pd.DataFrame({"start_date": [pd.Timestamp("2024-03-14")],
                           "end_date": [pd.Timestamp("2024-03-24")]})
    flag = promo_calendar(promos, start=pd.Timestamp("2024-03-01"),
                          end=pd.Timestamp("2024-04-01"))
    assert flag.sum() == 11
    assert flag["2024-03-14"] and flag["2024-03-24"]
    assert not flag["2024-03-13"] and not flag["2024-03-25"]


def test_repeated_promotions_keep_weekdays_and_are_flagged_as_assumed(tables):
    promos = planned_promotions(tables["campaigns"])
    until = pd.Timestamp("2026-03-30")
    out = repeat_promotions(promos, until=until)
    assumed = out.loc[out["assumed"]]
    assert len(out.loc[~out["assumed"]]) == len(promos)
    assert len(assumed) == 1  # only the spring sale falls before the end of March 2026
    row = assumed.iloc[0]
    source = promos.loc[promos["campaign_name"] == "Spring Refresh Sale 2025"].iloc[0]
    assert row["start_date"] == source["start_date"] + pd.Timedelta(days=364)
    assert row["start_date"].dayofweek == source["start_date"].dayofweek
    assert row["start_date"] > promos["end_date"].max() and row["start_date"] < until
    assert "assumed" in row["campaign_name"] and pd.isna(row["campaign_id"])
