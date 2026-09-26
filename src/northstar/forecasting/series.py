"""Revenue time series and the planned promotion calendar, built from the shared tables.

The forecast target is **net order revenue** (``orders.net_amount``, net of discounts, before cost
of goods) summed by calendar day. Forecasts are made daily and reported by Monday-to-Sunday week,
because weekly totals are what staffing, replenishment and revenue plans are built on.

Two kinds of inputs exist, and the distinction is what keeps the forecast leak-free:

* **Observed history** - revenue on days strictly before the forecast origin. Every model
  receives only ``history_before(daily, origin)``.
* **Known-in-advance calendar** - day of week, day of year and the *planned* promotion calendar
  (start and end dates of ``campaigns`` with objective ``promotion``). Retail promotion calendars
  are set months ahead, so their dates are treated as known at the origin. Their *outcomes*
  (orders, revenue) are never used. The section README reports the model with and without this
  assumption.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from northstar.timeline import DATA_END, DATA_START

WEEK_DAYS = 7
PROMOTION_OBJECTIVE = "promotion"


def daily_revenue(orders: pd.DataFrame, start: pd.Timestamp = DATA_START,
                  end: pd.Timestamp = DATA_END) -> pd.Series:
    """Net revenue per calendar day in ``[start, end)``; days without orders are 0."""
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    in_range = orders.loc[(orders["order_ts"] >= start) & (orders["order_ts"] < end)]
    revenue = in_range.groupby(in_range["order_ts"].dt.normalize())["net_amount"].sum()
    return revenue.reindex(days, fill_value=0.0).astype(float).rename("net_revenue")


def history_before(daily: pd.Series, origin: pd.Timestamp) -> pd.Series:
    """The only slice of the series a forecast issued at ``origin`` may see (``day < origin``)."""
    return daily.loc[daily.index < pd.Timestamp(origin)]


def complete_weeks_end(daily: pd.Series) -> pd.Timestamp:
    """Exclusive end of the last complete Monday-Sunday week covered by ``daily``."""
    end = daily.index[-1] + pd.Timedelta(days=1)
    return end - pd.Timedelta(days=end.dayofweek)


def weekly_revenue(daily: pd.Series) -> pd.DataFrame:
    """Monday-start weekly totals over complete weeks only (a partial last week is dropped)."""
    first = daily.index[0] + pd.Timedelta(days=(7 - daily.index[0].dayofweek) % 7)
    d = daily.loc[(daily.index >= first) & (daily.index < complete_weeks_end(daily))]
    weeks = d.groupby(d.index - pd.to_timedelta(d.index.dayofweek, unit="D")).agg(["sum", "size"])
    return pd.DataFrame({"week_start": weeks.index, "net_revenue": weeks["sum"].to_numpy(),
                         "days": weeks["size"].to_numpy()})


def to_weeks(values: np.ndarray, weeks: int) -> np.ndarray:
    """Sum a daily path that starts on a Monday into ``weeks`` weekly totals."""
    values = np.asarray(values, float)
    if len(values) != weeks * WEEK_DAYS:
        raise ValueError(f"expected {weeks * WEEK_DAYS} daily values, got {len(values)}")
    return values.reshape(weeks, WEEK_DAYS).sum(axis=1)


def planned_promotions(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Planned promotion windows (inclusive start and end dates) from the campaign table."""
    promos = campaigns.loc[campaigns["objective"] == PROMOTION_OBJECTIVE,
                           ["campaign_id", "campaign_name", "start_date", "end_date"]]
    return promos.sort_values("start_date").reset_index(drop=True)


def repeat_promotions(promos: pd.DataFrame, until: pd.Timestamp, weeks: int = 52) -> pd.DataFrame:
    """Assumed future promotions: each planned window repeated ``weeks`` later (same weekdays).

    Used only for the forward-looking forecast, whose period has no plan in the campaign table
    yet. The rows are flagged ``assumed=True`` and reported in the README as an assumption.
    """
    shift = pd.Timedelta(weeks=weeks)
    last_planned = promos["end_date"].max()
    rows = promos.assign(start_date=promos["start_date"] + shift,
                         end_date=promos["end_date"] + shift)
    rows = rows.loc[(rows["start_date"] > last_planned) & (rows["start_date"] < until)]
    rows = rows.assign(campaign_id=pd.NA,
                       campaign_name=rows["campaign_name"].str.replace(
                           r"\d{4}$", "(assumed repeat)", regex=True))
    out = pd.concat([promos.assign(assumed=False), rows.assign(assumed=True)], ignore_index=True)
    return out.sort_values("start_date").reset_index(drop=True)


def promo_calendar(promos: pd.DataFrame, start: pd.Timestamp = DATA_START,
                   end: pd.Timestamp | None = None) -> pd.Series:
    """Boolean flag per day: is a (planned) promotion running on that day?"""
    end = end if end is not None else DATA_END + pd.Timedelta(days=366)
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    flag = pd.Series(False, index=days, name="promotion")
    for first, last in zip(promos["start_date"], promos["end_date"], strict=True):
        flag.loc[(days >= first) & (days <= last)] = True
    return flag


def build_series(tables: Mapping[str, pd.DataFrame]) -> tuple[pd.Series, pd.DataFrame]:
    """Daily revenue over the full history and the planned promotion windows."""
    return daily_revenue(tables["orders"]), planned_promotions(tables["campaigns"])
