"""Leak-free churn dataset: active-customer base, point-in-time features and churn labels.

Framing. On the first day of each month (a *scoring run*, the cutoff ``c``) the CRM team scores
the **active base** and decides who receives a retention offer.

* **Active at c:** a customer with at least one order in the observation window
  ``[c - ACTIVE_DAYS, c)``. Customers idle for longer are already lapsed; they belong to the
  win-back program, not to churn prevention.
* **Churned (label = 1):** no order in the prediction window ``[c, c + HORIZON_DAYS)``. Northstar
  is a non-contractual retailer, so churn is only observable as the absence of purchases. The
  90-day horizon matches the business's own lapse rule (win-back emails start after 90 idle
  days).

Leakage rules, enforced here and re-checked by :func:`leakage_audit` and the tests:

* Features are computed only from ``timeline.snapshot(tables, cutoff)``: every event used has a
  timestamp strictly before the cutoff, and support resolutions/CSAT not yet known at the cutoff
  are masked.
* The label reads only ``orders.order_ts`` inside ``[cutoff, cutoff + horizon)``; no feature is
  derived from that window.
* The split purges runs whose label windows would overlap a later split (see ``SplitPlan``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from northstar.acquisition.dataset import SplitPlan, monthly_runs
from northstar.synthetic.params import MEMBERSHIP_PERIOD_DAYS
from northstar.timeline import DATA_START, TIME_COLUMNS, PredictionWindow, snapshot

HORIZON_DAYS = 90
ACTIVE_DAYS = 180
TABLES_USED = ("customers", "orders", "order_lines", "products", "sessions", "marketing_touches",
               "subscription_events", "support_contacts")

CATEGORICAL_FEATURES = ("acquisition_channel", "region", "age_band", "income_band", "device_type")
NUMERIC_FEATURES = (
    "email_opt_in",
    # purchase behavior (recency, frequency, monetary, trend, mix)
    "tenure_days",
    "days_since_last_order",
    "orders_total",
    "orders_90d",
    "orders_prev_90d",
    "net_revenue_180d",
    "avg_order_value",
    "discount_share",
    "first_order_discounted",
    "category_count",
    "store_order_share",
    "app_order_share",
    # digital engagement
    "browse_sessions_30d",
    "browse_sessions_90d",
    "days_since_last_session",
    "emails_received_90d",
    "email_open_rate_90d",
    "email_clicks_90d",
    # membership
    "plus_member",
    "plus_cancelled_180d",
    # service experience
    "support_contacts_180d",
    "low_csat_contacts_180d",
    "slow_resolution_contacts_180d",
    "open_contacts",
)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
KEY_COLUMNS = ("run_cutoff", "customer_id")
TARGET = "churned"
# Pre-cutoff gross margin; used to value a retained customer in the simulation, not a feature.
VALUE = "margin_180d"

# Columns that would describe the outcome window. None may be a feature.
FORBIDDEN_FEATURES = frozenset(
    {"churned", "orders_next_90d", "next_order_ts", "days_to_next_order", "net_revenue_next_90d",
     "order_ts", "order_id", "customer_id", "run_cutoff"}
)

SLOW_RESOLUTION_HOURS = 24.0
LOW_CSAT_MAX = 2


def _days(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 86_400


def active_customers(orders: pd.DataFrame, cutoff: pd.Timestamp,
                     active_days: int = ACTIVE_DAYS) -> pd.Index:
    """Customers with at least one order in ``[cutoff - active_days, cutoff)``, sorted."""
    cutoff = pd.Timestamp(cutoff)
    ts = orders["order_ts"]
    recent = (ts < cutoff) & (ts >= cutoff - pd.Timedelta(days=active_days))
    return pd.Index(np.sort(orders.loc[recent, "customer_id"].unique()), name="customer_id")


def churn_labels(customer_ids: pd.Series, orders: pd.DataFrame,
                 window: PredictionWindow) -> np.ndarray:
    """1 if the customer places **no** order in ``[cutoff, cutoff + horizon)``."""
    buyers = orders.loc[window.label_mask(orders["order_ts"]), "customer_id"]
    return (~customer_ids.isin(buyers)).to_numpy(dtype=int)


def _order_features(view: Mapping[str, pd.DataFrame], ids: pd.Index,
                    cutoff: pd.Timestamp) -> pd.DataFrame:
    orders = view["orders"]
    o = orders.loc[orders["customer_id"].isin(ids)].sort_values(["customer_id", "order_ts"])
    age = _days(cutoff - o["order_ts"])
    o = o.assign(age=age, in90=age < 90, prev90=(age >= 90) & (age < 180), in180=age < 180,
                 store=o["order_channel"] == "store", app=o["order_channel"] == "app")
    o["net180"] = o["net_amount"].where(o["in180"], 0.0)
    g = o.groupby("customer_id")
    first = g.head(1).set_index("customer_id")
    out = pd.DataFrame({
        "days_since_last_order": g["age"].min(),
        "orders_total": g.size(),
        "orders_90d": g["in90"].sum(),
        "orders_prev_90d": g["prev90"].sum(),
        "net_revenue_180d": g["net180"].sum(),
        "avg_order_value": g["net_amount"].mean(),
        "gross": g["gross_amount"].sum(),
        "discount": g["discount_amount"].sum(),
        "first_order_discounted": (first["discount_amount"] > 0).astype(int),
        "store_order_share": g["store"].mean(),
        "app_order_share": g["app"].mean(),
    }).reindex(ids)
    out["discount_share"] = (out["discount"] / out["gross"]).where(out["gross"] > 0, 0.0)

    # Category breadth and trailing gross margin from order lines of pre-cutoff orders.
    lines = view["order_lines"]
    lines = lines.loc[lines["order_id"].isin(o["order_id"])]
    products = view["products"].set_index("product_id")
    order_info = o.set_index("order_id")
    li = pd.DataFrame({
        "customer_id": lines["order_id"].map(order_info["customer_id"]).to_numpy(),
        "in180": lines["order_id"].map(order_info["in180"]).to_numpy(dtype=bool),
        "category": lines["product_id"].map(products["category"]).to_numpy(),
        "margin": (lines["net_amount"]
                   - lines["quantity"] * lines["product_id"].map(products["unit_cost"])).to_numpy(),
    })
    out["category_count"] = li.groupby("customer_id")["category"].nunique().reindex(ids)
    out[VALUE] = (li.loc[li["in180"]].groupby("customer_id")["margin"].sum().reindex(ids)
                  .fillna(0.0).clip(lower=0.0))
    return out.drop(columns=["gross", "discount"])


def _session_features(view: Mapping[str, pd.DataFrame], ids: pd.Index,
                      cutoff: pd.Timestamp) -> pd.DataFrame:
    sessions = view["sessions"]
    s = sessions.loc[sessions["customer_id"].isin(ids)]
    # Browsing = post-conversion sessions that did not end in an order before the cutoff.
    purchase_sessions = view["orders"]["session_id"].dropna()
    age = _days(cutoff - s["session_start"])
    browse = ~s["session_id"].isin(purchase_sessions)
    s = s.assign(age=age, b30=browse & (age < 30), b90=browse & (age < 90))
    g = s.groupby("customer_id")
    return pd.DataFrame({
        "browse_sessions_30d": g["b30"].sum(),
        "browse_sessions_90d": g["b90"].sum(),
        "days_since_last_session": g["age"].min(),
    }).reindex(ids)


def _email_features(view: Mapping[str, pd.DataFrame], ids: pd.Index,
                    cutoff: pd.Timestamp) -> pd.DataFrame:
    touches = view["marketing_touches"]
    t = touches.loc[touches["customer_id"].isin(ids) & (touches["touch_type"] == "email")]
    t = t.loc[t["touch_at"] >= cutoff - pd.Timedelta(days=90)]
    g = t.groupby("customer_id")
    out = pd.DataFrame({
        "emails_received_90d": g.size(),
        "emails_opened": g["opened"].sum(),
        "email_clicks_90d": g["clicked"].sum(),
    }).reindex(ids).fillna(0.0)
    out["email_open_rate_90d"] = (out["emails_opened"] / out["emails_received_90d"]).where(
        out["emails_received_90d"] > 0, 0.0)
    return out.drop(columns="emails_opened")


def _membership_features(view: Mapping[str, pd.DataFrame], ids: pd.Index,
                         cutoff: pd.Timestamp) -> pd.DataFrame:
    """Northstar Plus status as known at the cutoff.

    A member is someone whose latest subscribe/renew period still covers the cutoff. A cancel
    request is visible as soon as it is logged, even though benefits run to the end of the paid
    period, so a pending cancellation is a legitimate (and realistic) leading signal.
    """
    subs = view["subscription_events"]
    subs = subs.loc[subs["customer_id"].isin(ids)].sort_values(["customer_id", "event_ts"])
    paid = subs.loc[subs["event_type"] != "cancel"]
    last_paid = paid.groupby("customer_id").tail(1).set_index("customer_id")
    period = pd.to_timedelta(last_paid["plan"].map(MEMBERSHIP_PERIOD_DAYS), unit="D")
    covered = (last_paid["event_ts"] + period) > cutoff
    cancels = subs.loc[subs["event_type"] == "cancel"]
    recent_cancel = cancels.loc[cancels["event_ts"] >= cutoff - pd.Timedelta(days=180),
                                "customer_id"]
    return pd.DataFrame({
        "plus_member": covered.astype(int).reindex(ids).fillna(0),
        "plus_cancelled_180d": pd.Series(ids.isin(recent_cancel).astype(int), index=ids),
    })


def _support_features(view: Mapping[str, pd.DataFrame], ids: pd.Index,
                      cutoff: pd.Timestamp) -> pd.DataFrame:
    contacts = view["support_contacts"]
    c = contacts.loc[contacts["customer_id"].isin(ids)]
    recent = c["contact_ts"] >= cutoff - pd.Timedelta(days=180)
    hours = (c["resolved_at"] - c["contact_ts"]).dt.total_seconds() / 3600
    c = c.assign(
        recent=recent,
        low_csat=recent & (c["csat_score"] <= LOW_CSAT_MAX).fillna(False).astype(bool),
        slow=recent & (hours > SLOW_RESOLUTION_HOURS).fillna(False).astype(bool),
        # snapshot() masks resolutions after the cutoff, so these are genuinely open at cutoff.
        open=c["resolved_at"].isna(),
    )
    g = c.groupby("customer_id")
    return pd.DataFrame({
        "support_contacts_180d": g["recent"].sum(),
        "low_csat_contacts_180d": g["low_csat"].sum(),
        "slow_resolution_contacts_180d": g["slow"].sum(),
        "open_contacts": g["open"].sum(),
    }).reindex(ids).fillna(0.0)


def build_features(tables: Mapping[str, pd.DataFrame], cutoff: str | pd.Timestamp,
                   active_days: int = ACTIVE_DAYS) -> pd.DataFrame:
    """Point-in-time features for every active customer at ``cutoff`` (one row per customer).

    Only ``snapshot(tables, cutoff)`` is read, so the result is identical whether or not the
    input contains rows at or after the cutoff. Also returns the non-feature ``VALUE`` column.
    """
    cutoff = pd.Timestamp(cutoff)
    view = snapshot({name: tables[name] for name in TABLES_USED}, cutoff)
    ids = active_customers(view["orders"], cutoff, active_days)
    customers = view["customers"].set_index("customer_id").reindex(ids)

    tenure = _days(cutoff - customers["customer_since"])
    out = pd.DataFrame({
        "run_cutoff": cutoff,
        "customer_id": ids.to_numpy(),
        **{c: customers[c].astype(str).to_numpy() for c in CATEGORICAL_FEATURES},
        "email_opt_in": customers["email_opt_in"].astype(int).to_numpy(),
        "tenure_days": tenure.to_numpy(),
    })
    parts = [_order_features(view, ids, cutoff), _session_features(view, ids, cutoff),
             _email_features(view, ids, cutoff), _membership_features(view, ids, cutoff),
             _support_features(view, ids, cutoff)]
    for part in parts:
        for col in part.columns:
            out[col] = part[col].to_numpy(dtype=float)
    # Store-only customers may have no post-conversion session: treat the last visit as no more
    # recent than becoming a customer.
    out["days_since_last_session"] = out["days_since_last_session"].fillna(out["tenure_days"])
    out[["browse_sessions_30d", "browse_sessions_90d"]] = out[
        ["browse_sessions_30d", "browse_sessions_90d"]].fillna(0.0)
    return out[[*KEY_COLUMNS, *FEATURES, VALUE]]


def build_run(tables: Mapping[str, pd.DataFrame], cutoff: str | pd.Timestamp,
              horizon_days: int = HORIZON_DAYS, active_days: int = ACTIVE_DAYS) -> pd.DataFrame:
    """Features plus the ``churned`` label for one scoring run."""
    window = PredictionWindow(pd.Timestamp(cutoff), horizon_days=horizon_days)
    if window.cutoff - pd.Timedelta(days=active_days) < DATA_START:
        raise ValueError(f"run {window.cutoff.date()} has less than {active_days} days of history;"
                         " the active base would be incomplete")
    features = build_features(tables, window.cutoff, active_days)
    features[TARGET] = churn_labels(features["customer_id"], tables["orders"], window)
    return features


def build_dataset(tables: Mapping[str, pd.DataFrame], cutoffs: Iterable[pd.Timestamp],
                  horizon_days: int = HORIZON_DAYS, active_days: int = ACTIVE_DAYS
                  ) -> pd.DataFrame:
    return pd.concat([build_run(tables, c, horizon_days, active_days) for c in cutoffs],
                     ignore_index=True)


# Default design. Six fitting runs (Jul - Dec 2024) and two validation runs (Mar - Apr 2025) choose
# the champion; four out-of-time holdout runs (Jul - Oct 2025) are scored by a model frozen at the
# 2025-07-01 cutoff. Jan - Feb and May - Jun 2025 are purge gaps: runs there would have 90-day label
# windows reaching into the next split. The first run needs 180 days of order history.
DEFAULT_SPLIT = SplitPlan(
    fit=monthly_runs("2024-07-01", "2024-12-01"),
    validation=monthly_runs("2025-03-01", "2025-04-01"),
    holdout=monthly_runs("2025-07-01", "2025-10-01"),
    horizon_days=HORIZON_DAYS,
)


def leakage_audit(tables: Mapping[str, pd.DataFrame], dataset: pd.DataFrame, plan: SplitPlan,
                  single_feature_auc_limit: float = 0.9) -> dict:
    """Runtime leakage checks on the assembled dataset; ``passed`` is False if any check fails."""
    from sklearn.metrics import roc_auc_score

    # Every scored customer existed and was active (ordered in the observation window).
    orders = tables["orders"]
    since = dataset["customer_id"].map(tables["customers"].set_index("customer_id")[
        "customer_since"])
    not_yet_customer = int((since >= dataset["run_cutoff"]).sum())
    inactive = 0
    latest_event_gap_days = np.inf
    for cutoff, run in dataset.groupby("run_cutoff"):
        view = snapshot({n: tables[n] for n in TABLES_USED}, cutoff)
        inactive += int((~run["customer_id"].isin(
            active_customers(view["orders"], cutoff, ACTIVE_DAYS))).sum())
        for name in ("orders", "sessions", "marketing_touches", "subscription_events",
                     "support_contacts"):
            rows = view[name].loc[view[name]["customer_id"].isin(run["customer_id"])]
            ts = rows[TIME_COLUMNS[name]]
            if len(ts):
                latest_event_gap_days = min(latest_event_gap_days,
                                            (cutoff - ts.max()).total_seconds() / 86_400)

    # Recompute one holdout run from data truncated at its cutoff: features must not change.
    probe = plan.holdout[0]
    truncated = snapshot({n: tables[n] for n in TABLES_USED}, probe)
    cols = [*KEY_COLUMNS, *FEATURES, VALUE]
    full_run = dataset.loc[dataset["run_cutoff"] == probe, cols].reset_index(drop=True)
    trunc_run = build_features(truncated, probe).reset_index(drop=True)
    invariant = full_run.equals(trunc_run)

    # Labels must be exactly "no order in [cutoff, cutoff + horizon)".
    label_mismatch = 0
    for cutoff, run in dataset.groupby("run_cutoff"):
        window = PredictionWindow(cutoff, horizon_days=plan.horizon_days)
        label_mismatch += int((churn_labels(run["customer_id"], orders, window)
                               != run[TARGET].to_numpy()).sum())

    holdout = dataset.loc[dataset["run_cutoff"].isin(plan.holdout)]
    single_auc = {}
    for col in NUMERIC_FEATURES:
        auc = roc_auc_score(holdout[TARGET], holdout[col])
        single_auc[col] = float(max(auc, 1 - auc))
    top_feature = max(single_auc, key=single_auc.get)

    train_label_end = max(plan.train) + pd.Timedelta(days=plan.horizon_days)
    checks = {
        "features_only_from_pre_cutoff_rows": bool(latest_event_gap_days > 0),
        "features_unchanged_when_future_rows_removed": bool(invariant),
        "labels_only_from_prediction_window": label_mismatch == 0,
        "every_scored_customer_active_at_cutoff": inactive == 0 and not_yet_customer == 0,
        "no_outcome_columns_used_as_features": not (set(FEATURES) & FORBIDDEN_FEATURES),
        "train_labels_end_before_holdout_starts": bool(train_label_end <= min(plan.holdout)),
        "no_single_feature_suspiciously_predictive":
            single_auc[top_feature] < single_feature_auc_limit,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "customers_not_yet_acquired": not_yet_customer,
            "customers_inactive_at_cutoff": inactive,
            "label_mismatches": label_mismatch,
            "min_gap_between_last_event_and_cutoff_seconds": round(latest_event_gap_days * 86_400,
                                                                   1),
            "invariance_probe_run": str(probe.date()),
            "last_train_label_end": str(train_label_end.date()),
            "first_holdout_run": str(min(plan.holdout).date()),
            "most_predictive_single_feature": top_feature,
            "most_predictive_single_feature_auc": round(single_auc[top_feature], 4),
            "single_feature_auc_limit": single_feature_auc_limit,
        },
    }
