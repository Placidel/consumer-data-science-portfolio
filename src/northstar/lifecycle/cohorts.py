"""Acquisition-cohort retention with fixed denominators and explicit right-censoring.

* A **cohort** is the calendar month of a customer's first order (``customers.customer_since``).
* ``month k`` is the k-th calendar month after the cohort month (``k = 0`` is the acquisition
  month, where retention is 100% by definition).
* **Retention in month k** = customers of the cohort with at least one order in month k,
  divided by the **cohort size at acquisition**. The denominator never changes with k: customers
  who lapse stay in it, and nobody joins a cohort later.
* Months after the end of the data are **unobserved**: their retention is missing, never zero.
* Curves pooled across cohorts use a **fixed set of mature cohorts** (every cohort observed for
  at least ``follow_up_months``), so the pooled denominator is also identical at every k rather
  than shrinking as young cohorts run out of history.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from northstar.timeline import DATA_END, DATA_START


def _month_index(ts: pd.Series) -> pd.Series:
    return ts.dt.year * 12 + ts.dt.month - 1


def _index_to_month(index: pd.Series) -> pd.Series:
    return pd.to_datetime({"year": index // 12, "month": index % 12 + 1, "day": 1})


def cohort_retention(customers: pd.DataFrame, orders: pd.DataFrame,
                     end: pd.Timestamp = DATA_END, by: Sequence[str] = ()) -> pd.DataFrame:
    """Cohort x month-since-acquisition table, optionally split by customer attributes.

    Columns: ``cohort`` (month start), ``months_since_acquisition``, ``observed``,
    ``cohort_size``, ``buyers``, ``retention``, ``revenue``, ``revenue_per_customer`` and
    ``cumulative_revenue_per_customer``. Unobserved cells have NaN outcomes.
    """
    by = list(by)
    last = int(_month_index(pd.Series([pd.Timestamp(end) - pd.Timedelta(days=1)])).iloc[0])
    first = int(_month_index(pd.Series([DATA_START])).iloc[0])
    c = customers[["customer_id", *by]].assign(
        cohort_idx=_month_index(customers["customer_since"]))
    if (c["cohort_idx"] > last).any():
        raise ValueError("customers acquired after the end of the observation period")
    sizes = c.groupby([*by, "cohort_idx"]).size().rename("cohort_size").reset_index()

    o = orders.loc[orders["order_ts"] < end, ["customer_id", "order_ts", "net_amount"]].merge(
        c, on="customer_id", how="inner", validate="many_to_one")
    o["months_since_acquisition"] = _month_index(o["order_ts"]) - o["cohort_idx"]
    if (o["months_since_acquisition"] < 0).any():
        raise ValueError("orders before the customer's first order")
    activity = o.groupby([*by, "cohort_idx", "months_since_acquisition"]).agg(
        buyers=("customer_id", "nunique"), revenue=("net_amount", "sum")).reset_index()

    grid = sizes.merge(pd.DataFrame({"months_since_acquisition": np.arange(last - first + 1)}),
                       how="cross")
    grid["observed"] = grid["cohort_idx"] + grid["months_since_acquisition"] <= last
    out = grid.merge(activity, on=[*by, "cohort_idx", "months_since_acquisition"], how="left")
    for col in ("buyers", "revenue"):
        out[col] = out[col].fillna(0.0).where(out["observed"])
    out["retention"] = out["buyers"] / out["cohort_size"]
    out["revenue_per_customer"] = out["revenue"] / out["cohort_size"]
    out = out.sort_values([*by, "cohort_idx", "months_since_acquisition"], ignore_index=True)
    out["cumulative_revenue_per_customer"] = out.groupby([*by, "cohort_idx"])[
        "revenue_per_customer"].cumsum().where(out["observed"])
    out.insert(len(by), "cohort", _index_to_month(out["cohort_idx"]))
    return out.drop(columns="cohort_idx")


def pooled_retention(table: pd.DataFrame, follow_up_months: int, by: Sequence[str] = ()
                     ) -> pd.DataFrame:
    """Retention curve over the fixed set of cohorts with at least ``follow_up_months`` observed.

    ``customers`` (the pooled denominator) is the same at every ``months_since_acquisition``.
    """
    by = list(by)
    observed_months = table["months_since_acquisition"].where(table["observed"])
    follow_up = observed_months.groupby([table[k] for k in [*by, "cohort"]]).transform("max")
    t = table.assign(_follow=follow_up)
    t = t.loc[(t["_follow"] >= follow_up_months)
              & (t["months_since_acquisition"] <= follow_up_months)]
    if t.empty:
        raise ValueError(f"no cohort has {follow_up_months} months of follow-up")
    out = t.groupby([*by, "months_since_acquisition"]).agg(
        cohorts=("cohort", "nunique"), first_cohort=("cohort", "min"),
        last_cohort=("cohort", "max"), customers=("cohort_size", "sum"),
        buyers=("buyers", "sum"), revenue=("revenue", "sum")).reset_index()
    out["retention"] = out["buyers"] / out["customers"]
    out["revenue_per_customer"] = out["revenue"] / out["customers"]
    per_customer = out.groupby(by)["revenue_per_customer"] if by else out["revenue_per_customer"]
    out["cumulative_revenue_per_customer"] = per_customer.cumsum()
    return out
