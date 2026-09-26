"""RFM segmentation that complements lifecycle states, plus segment-level summaries.

Lifecycle states say *where* a customer is in the relationship (mostly by recency and tenure).
RFM adds *how much* the relationship is worth: within the same state, a customer who bought
eight times for $900 last year and one who bought once for $40 need different treatment.

* **R** (recency) scores days since the last order; **F** (frequency) scores orders and **M**
  (monetary) net revenue in the trailing 365 days. Each is scored 1-5 by quintile of the
  customer base at the as-of date (average ranks, so tied customers share a score). Higher is
  better for all three.
* **FM** = ``ceil((F + M) / 2)`` combines frequency and spend, which are strongly correlated.
* Seven named segments come from an explicit R x FM grid (``segment_rule``); every (R, FM)
  pair maps to exactly one segment.

Profiles are built from ``timeline.snapshot`` at the as-of date. Outcomes (orders and revenue in
the following ``horizon_days``) are attached **after** segmentation, only to describe what each
segment went on to do; they never feed a score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from northstar.lifecycle.states import CUSTOMER_STATES, LifecycleRules, assign_states, people_frame
from northstar.retention.dataset import (
    _email_features,
    _membership_features,
    _session_features,
)
from northstar.timeline import PredictionWindow, snapshot

RFM_WINDOW_DAYS = 365
TABLES_USED = ("prospects", "customers", "orders", "sessions", "marketing_touches",
               "subscription_events")
RFM_SEGMENTS = ("champions", "loyalists", "promising", "needs_attention", "cannot_lose",
                "slipping", "hibernating")
RFM_LABELS = {
    "champions": "Champions",
    "loyalists": "Loyalists",
    "promising": "Promising",
    "needs_attention": "Needs attention",
    "cannot_lose": "Can't lose",
    "slipping": "Slipping",
    "hibernating": "Hibernating",
}
RFM_RULES = {
    "champions": "R 4-5 and FM 4-5",
    "loyalists": "R 4-5 and FM 3, or R 3 and FM 3-5",
    "promising": "R 4-5 and FM 1-2",
    "needs_attention": "R 3 and FM 1-2",
    "cannot_lose": "R 1-2 and FM 4-5",
    "slipping": "R 1-2 and FM 3",
    "hibernating": "R 1-2 and FM 1-2",
}


def quintile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Score 1-5 by quintile of the average rank; ties always share a score."""
    signed = values if higher_is_better else -values
    pct = signed.rank(method="average", pct=True)
    return np.ceil(pct * 5).clip(1, 5).astype(int)


def segment_rule(r, fm) -> np.ndarray:
    """Named RFM segment for arrays of R and FM scores (1-5)."""
    r, fm = np.asarray(r), np.asarray(fm)
    if np.any((r < 1) | (r > 5) | (fm < 1) | (fm > 5)):
        raise ValueError("scores must be in 1..5")
    recent, mid, lapsed = r >= 4, r == 3, r <= 2
    conditions = [recent & (fm >= 4), (recent & (fm == 3)) | (mid & (fm >= 3)), recent,
                  mid, lapsed & (fm >= 4), lapsed & (fm == 3)]
    return np.select(conditions, list(RFM_SEGMENTS[:-1]), RFM_SEGMENTS[-1])


def rfm_scores(profile: pd.DataFrame) -> pd.DataFrame:
    """R, F, M, FM scores and the segment for a customer profile frame."""
    r = quintile_score(profile["idle_days"], higher_is_better=False)
    f = quintile_score(profile["orders_365d"])
    m = quintile_score(profile["revenue_365d"])
    fm = np.ceil((f + m) / 2).astype(int)
    return pd.DataFrame({"r_score": r, "f_score": f, "m_score": m, "fm_score": fm,
                         "rfm_segment": segment_rule(r, fm)}, index=profile.index)


def customer_profile(tables: Mapping[str, pd.DataFrame], as_of: str | pd.Timestamp,
                     rules: LifecycleRules | None = None, horizon_days: int = 90
                     ) -> pd.DataFrame:
    """One row per customer with a first order before ``as_of``: state, RFM, engagement and
    (separately) outcomes in ``[as_of, as_of + horizon_days)``."""
    rules = rules or LifecycleRules()
    window = PredictionWindow(pd.Timestamp(as_of), horizon_days=horizon_days)
    as_of = window.cutoff
    view = snapshot({name: tables[name] for name in TABLES_USED}, as_of)
    people = people_frame(view)
    states = assign_states(people, view["orders"], as_of, rules)
    keep = states["state"].isin(CUSTOMER_STATES)
    base = pd.concat([people.loc[keep, ["customer_id", "acquisition_channel"]],
                      states.loc[keep]], axis=1).set_index("customer_id").sort_index()
    ids = base.index

    orders = view["orders"]
    recent = orders.loc[orders["order_ts"] >= as_of - pd.Timedelta(days=RFM_WINDOW_DAYS)]
    lifetime = orders.groupby("customer_id")["net_amount"].agg(["size", "sum"])
    base["orders_365d"] = recent.groupby("customer_id").size().reindex(ids, fill_value=0)
    base["revenue_365d"] = recent.groupby("customer_id")["net_amount"].sum().reindex(
        ids, fill_value=0.0)
    base["orders_total"] = lifetime["size"].reindex(ids)
    base["avg_order_value"] = (lifetime["sum"] / lifetime["size"]).reindex(ids)
    for part in (_session_features(view, ids, as_of), _email_features(view, ids, as_of),
                 _membership_features(view, ids, as_of)):
        base = base.join(part[[c for c in part.columns if c != "days_since_last_session"]])
    base[["browse_sessions_30d", "browse_sessions_90d"]] = base[
        ["browse_sessions_30d", "browse_sessions_90d"]].fillna(0.0)
    base = base.join(rfm_scores(base))

    # Outcomes: read from the full tables, inside the outcome window only.
    future = tables["orders"].loc[window.label_mask(tables["orders"]["order_ts"])]
    base["next_orders"] = future.groupby("customer_id").size().reindex(ids, fill_value=0)
    base["next_revenue"] = future.groupby("customer_id")["net_amount"].sum().reindex(
        ids, fill_value=0.0)
    return base.reset_index()


def segment_summary(profile: pd.DataFrame, by: str, order: Sequence[str]) -> pd.DataFrame:
    """Size, value, engagement and next-period outcomes per group."""
    p = profile.assign(_buyer=(profile["next_orders"] > 0).astype(float),
                       _email_rate=profile["email_open_rate_90d"].where(
                           profile["emails_received_90d"] > 0))
    out = p.groupby(by).agg(
        customers=("customer_id", "size"), revenue_365d=("revenue_365d", "sum"),
        orders_365d=("orders_365d", "mean"), median_days_since_order=("idle_days", "median"),
        avg_order_value=("avg_order_value", "mean"),
        browse_sessions_90d=("browse_sessions_90d", "mean"),
        email_open_rate_90d=("_email_rate", "mean"), plus_member_rate=("plus_member", "mean"),
        next_purchase_rate=("_buyer", "mean"), next_revenue=("next_revenue", "sum"),
    ).reindex([g for g in order if g in set(p[by])])
    out.insert(1, "share_of_customers", out["customers"] / len(p))
    out.insert(3, "share_of_revenue_365d", out["revenue_365d"] / p["revenue_365d"].sum())
    out["revenue_365d_per_customer"] = out["revenue_365d"] / out["customers"]
    out["next_revenue_per_customer"] = out["next_revenue"] / out["customers"]
    out["share_of_next_revenue"] = out["next_revenue"] / p["next_revenue"].sum()
    return out.rename_axis(by).reset_index()
