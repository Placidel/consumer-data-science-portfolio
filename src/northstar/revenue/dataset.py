"""Leak-free customer value dataset: customer base, point-in-time features and future revenue.

Framing. At a *scoring run* (cutoff ``c``, the first day of a month) growth marketing scores the
**whole customer base**, every customer whose first order was before ``c``, including lapsed
ones, and ranks it by expected **net revenue in the next ``HORIZON_DAYS`` days**.

* **Target:** ``future_revenue`` = sum of ``orders.net_amount`` with ``c <= order_ts < c + 180d``
  (0 for customers who do not buy). Net of discounts, before cost of goods. New customers acquired
  during the window are out of scope (section 01 covers acquisition).
* **Historical spend aggregates** (``revenue_total``, ``revenue_90d`` ... ``revenue_365d``) sum
  orders strictly *before* ``c``, so the target period never enters a feature.

Leakage rules, enforced here and re-checked by :func:`leakage_audit` and the tests:

* Features are computed only from ``timeline.snapshot(tables, cutoff)``. The BG/NBD and
  Gamma-Gamma models behind the ``clv_*`` / ``bgnbd_*`` features are refit on that snapshot at
  every run; they use no labels.
* The label reads only ``orders`` inside ``[cutoff, cutoff + horizon)``, and for every customer
  ``revenue_total + future_revenue`` must reconcile to the order log up to ``cutoff + horizon``.
* The split purges runs whose 180-day label windows would overlap a later split (``SplitPlan``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from northstar.acquisition.dataset import SplitPlan, monthly_runs
from northstar.retention.dataset import (
    _email_features,
    _membership_features,
    _session_features,
    _support_features,
)
from northstar.revenue.clv import clv_features
from northstar.timeline import DATA_START, TIME_COLUMNS, PredictionWindow, snapshot

HORIZON_DAYS = 180
MIN_HISTORY_DAYS = 90  # a run needs at least a quarter of order history for its windows
TABLES_USED = ("customers", "orders", "order_lines", "products", "sessions", "marketing_touches",
               "subscription_events", "support_contacts")

CATEGORICAL_FEATURES = ("acquisition_channel", "region", "age_band", "income_band", "device_type")
NUMERIC_FEATURES = (
    "email_opt_in",
    # recency, frequency, monetary (all strictly before the cutoff)
    "tenure_days",
    "days_since_last_order",
    "mean_days_between_orders",
    "orders_total",
    "orders_90d",
    "orders_180d",
    "orders_365d",
    "revenue_total",
    "revenue_90d",
    "revenue_180d",
    "revenue_365d",
    "avg_order_value",
    "max_order_value",
    "items_per_order",
    "discount_share",
    "category_count",
    "store_order_share",
    "app_order_share",
    # digital engagement (shared definitions with section 02)
    "browse_sessions_30d",
    "browse_sessions_90d",
    "days_since_last_session",
    "emails_received_90d",
    "email_open_rate_90d",
    "email_clicks_90d",
    # membership and service
    "plus_member",
    "plus_cancelled_180d",
    "support_contacts_180d",
    "low_csat_contacts_180d",
    # probabilistic CLV, refit on pre-cutoff history at every run
    "bgnbd_p_alive",
    "bgnbd_expected_orders",
    "gg_expected_order_value",
    "clv_expected_revenue",
)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
KEY_COLUMNS = ("run_cutoff", "customer_id")
TARGET = "future_revenue"

# Columns that would describe the outcome window. None may be a feature.
FORBIDDEN_FEATURES = frozenset(
    {"future_revenue", "future_orders", "future_margin", "next_order_ts", "churned", "order_ts",
     "order_id", "customer_id", "run_cutoff"}
)
HISTORICAL_SPEND = ("revenue_total", "revenue_90d", "revenue_180d", "revenue_365d")


def _days(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 86_400


def customer_base(customers: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Index:
    """Customers whose first order was strictly before ``cutoff``, sorted."""
    since = customers["customer_since"]
    return pd.Index(np.sort(customers.loc[since < pd.Timestamp(cutoff), "customer_id"].unique()),
                    name="customer_id")


def future_revenue(customer_ids: pd.Series, orders: pd.DataFrame, window: PredictionWindow
                   ) -> np.ndarray:
    """Net revenue per customer from orders in ``[cutoff, cutoff + horizon)``; 0 if none."""
    in_window = orders.loc[window.label_mask(orders["order_ts"])]
    spend = in_window.groupby("customer_id")["net_amount"].sum()
    return customer_ids.map(spend).fillna(0.0).to_numpy(float)


def _order_features(view: Mapping[str, pd.DataFrame], ids: pd.Index,
                    cutoff: pd.Timestamp) -> pd.DataFrame:
    orders = view["orders"]
    o = orders.loc[orders["customer_id"].isin(ids)].sort_values(["customer_id", "order_ts"])
    age = _days(cutoff - o["order_ts"])
    o = o.assign(age=age, store=o["order_channel"] == "store", app=o["order_channel"] == "app")
    windows = {}
    for days in (90, 180, 365):
        recent = age < days
        windows[f"orders_{days}d"] = recent
        windows[f"revenue_{days}d"] = o["net_amount"].where(recent, 0.0)
    o = o.assign(**windows)
    g = o.groupby("customer_id")
    span = _days(g["order_ts"].max() - g["order_ts"].min())
    n = g.size()
    out = pd.DataFrame({
        "days_since_last_order": g["age"].min(),
        "first_order_age": g["age"].max(),
        "orders_total": n,
        **{f"orders_{d}d": g[f"orders_{d}d"].sum() for d in (90, 180, 365)},
        "revenue_total": g["net_amount"].sum(),
        **{f"revenue_{d}d": g[f"revenue_{d}d"].sum() for d in (90, 180, 365)},
        "avg_order_value": g["net_amount"].mean(),
        "max_order_value": g["net_amount"].max(),
        "items_per_order": g["item_count"].mean(),
        "gross": g["gross_amount"].sum(),
        "discount": g["discount_amount"].sum(),
        "store_order_share": g["store"].mean(),
        "app_order_share": g["app"].mean(),
        "mean_days_between_orders": (span / (n - 1)).where(n > 1),
    }).reindex(ids)
    out["discount_share"] = (out["discount"] / out["gross"]).where(out["gross"] > 0, 0.0)
    # One-order customers have no observed gap yet: use their age as a lower bound.
    out["mean_days_between_orders"] = out["mean_days_between_orders"].fillna(out["first_order_age"])

    lines = view["order_lines"]
    lines = lines.loc[lines["order_id"].isin(o["order_id"])]
    category = lines["product_id"].map(view["products"].set_index("product_id")["category"])
    owner = lines["order_id"].map(o.set_index("order_id")["customer_id"])
    out["category_count"] = category.groupby(owner.to_numpy()).nunique().reindex(ids)
    return out.drop(columns=["gross", "discount", "first_order_age"])


def build_features(tables: Mapping[str, pd.DataFrame], cutoff: str | pd.Timestamp,
                   horizon_days: int = HORIZON_DAYS) -> tuple[pd.DataFrame, dict]:
    """Point-in-time features for the whole customer base at ``cutoff`` (one row per customer).

    Only ``snapshot(tables, cutoff)`` is read, so the result is identical whether or not the
    input contains rows at or after the cutoff. Also returns the fitted CLV parameters.
    """
    cutoff = pd.Timestamp(cutoff)
    view = snapshot({name: tables[name] for name in TABLES_USED}, cutoff)
    ids = customer_base(view["customers"], cutoff)
    customers = view["customers"].set_index("customer_id").reindex(ids)

    out = pd.DataFrame({
        "run_cutoff": cutoff,
        "customer_id": ids.to_numpy(),
        **{c: customers[c].astype(str).to_numpy() for c in CATEGORICAL_FEATURES},
        "email_opt_in": customers["email_opt_in"].astype(int).to_numpy(),
        "tenure_days": _days(cutoff - customers["customer_since"]).to_numpy(),
    })
    clv, clv_params = clv_features(view["orders"], cutoff, ids, horizon_days)
    parts = [_order_features(view, ids, cutoff), _session_features(view, ids, cutoff),
             _email_features(view, ids, cutoff), _membership_features(view, ids, cutoff),
             _support_features(view, ids, cutoff), clv]
    for part in parts:
        for col in part.columns:
            if col in NUMERIC_FEATURES:
                out[col] = part[col].to_numpy(dtype=float)
    # Store-only customers may have no post-conversion session: no more recent than conversion.
    out["days_since_last_session"] = out["days_since_last_session"].fillna(out["tenure_days"])
    out[["browse_sessions_30d", "browse_sessions_90d"]] = out[
        ["browse_sessions_30d", "browse_sessions_90d"]].fillna(0.0)
    return out[[*KEY_COLUMNS, *FEATURES]], clv_params


def build_run(tables: Mapping[str, pd.DataFrame], cutoff: str | pd.Timestamp,
              horizon_days: int = HORIZON_DAYS) -> tuple[pd.DataFrame, dict]:
    """Features plus the ``future_revenue`` label for one scoring run."""
    window = PredictionWindow(pd.Timestamp(cutoff), horizon_days=horizon_days)
    if window.cutoff - pd.Timedelta(days=MIN_HISTORY_DAYS) < DATA_START:
        raise ValueError(f"run {window.cutoff.date()} has less than {MIN_HISTORY_DAYS} days of "
                         "order history")
    features, clv_params = build_features(tables, window.cutoff, horizon_days)
    features[TARGET] = future_revenue(features["customer_id"], tables["orders"], window)
    return features, clv_params


def build_dataset(tables: Mapping[str, pd.DataFrame], cutoffs: Iterable[pd.Timestamp],
                  horizon_days: int = HORIZON_DAYS) -> tuple[pd.DataFrame, dict[str, dict]]:
    """All runs stacked, plus the CLV parameters fitted at each run (keyed by run date)."""
    frames, params = [], {}
    for cutoff in cutoffs:
        run, clv_params = build_run(tables, cutoff, horizon_days)
        frames.append(run)
        params[str(pd.Timestamp(cutoff).date())] = clv_params
    return pd.concat(frames, ignore_index=True), params


# Default design. With a 180-day horizon and 24 months of data, only one out-of-time holdout run
# fits: 2025-07-01 (label window runs to 2025-12-28, exclusive). Training labels must end by
# then, so validation is the 2025-01-01 run and the four fitting runs (Apr - Jul 2024) end their
# label windows by 2024-12-28. Earlier runs would have less than a quarter of history.
DEFAULT_SPLIT = SplitPlan(
    fit=monthly_runs("2024-04-01", "2024-07-01"),
    validation=monthly_runs("2025-01-01", "2025-01-01"),
    holdout=monthly_runs("2025-07-01", "2025-07-01"),
    horizon_days=HORIZON_DAYS,
)


def leakage_audit(tables: Mapping[str, pd.DataFrame], dataset: pd.DataFrame, plan: SplitPlan,
                  single_feature_rank_corr_limit: float = 0.9) -> dict:
    """Runtime leakage checks on the assembled dataset; ``passed`` is False if any check fails."""
    orders = tables["orders"]
    since = dataset["customer_id"].map(
        tables["customers"].set_index("customer_id")["customer_since"])
    not_yet_customer = int((since >= dataset["run_cutoff"]).sum())

    latest_event_gap_days = np.inf
    label_mismatch = spend_mismatch = 0
    for cutoff, run in dataset.groupby("run_cutoff"):
        view = snapshot({n: tables[n] for n in TABLES_USED}, cutoff)
        for name in ("orders", "sessions", "marketing_touches", "subscription_events",
                     "support_contacts"):
            rows = view[name].loc[view[name]["customer_id"].isin(run["customer_id"])]
            ts = rows[TIME_COLUMNS[name]]
            if len(ts):
                latest_event_gap_days = min(latest_event_gap_days,
                                            (cutoff - ts.max()).total_seconds() / 86_400)
        window = PredictionWindow(cutoff, horizon_days=plan.horizon_days)
        label_mismatch += int((~np.isclose(future_revenue(run["customer_id"], orders, window),
                                           run[TARGET].to_numpy(), atol=0.005)).sum())
        # Historical spend + target must reconcile to the order log up to the label end: the
        # target period is counted exactly once, and only in the label.
        through_label_end = orders.loc[orders["order_ts"] < window.label_end]
        ledger = run["customer_id"].map(
            through_label_end.groupby("customer_id")["net_amount"].sum()).fillna(0.0)
        spend_mismatch += int((~np.isclose(run["revenue_total"] + run[TARGET], ledger,
                                           atol=0.01)).sum())

    # Recompute one holdout run from data truncated at its cutoff: features must not change.
    probe = plan.holdout[0]
    truncated = snapshot({n: tables[n] for n in TABLES_USED}, probe)
    cols = [*KEY_COLUMNS, *FEATURES]
    full_run = dataset.loc[dataset["run_cutoff"] == probe, cols].reset_index(drop=True)
    trunc_run = build_features(truncated, probe, plan.horizon_days)[0].reset_index(drop=True)
    invariant = full_run.equals(trunc_run)

    holdout = dataset.loc[dataset["run_cutoff"].isin(plan.holdout)]
    target_rank = holdout[TARGET].rank()
    rank_corr = {col: float(abs(holdout[col].rank().corr(target_rank)))
                 for col in NUMERIC_FEATURES}
    top_feature = max(rank_corr, key=rank_corr.get)

    train_label_end = max(plan.train) + pd.Timedelta(days=plan.horizon_days)
    checks = {
        "features_only_from_pre_cutoff_rows": bool(latest_event_gap_days > 0),
        "features_unchanged_when_future_rows_removed": bool(invariant),
        "labels_only_from_prediction_window": label_mismatch == 0,
        "historical_spend_excludes_target_period": spend_mismatch == 0,
        "every_scored_customer_acquired_before_cutoff": not_yet_customer == 0,
        "no_outcome_columns_used_as_features": not (set(FEATURES) & FORBIDDEN_FEATURES),
        "train_labels_end_before_holdout_starts": bool(train_label_end <= min(plan.holdout)),
        "no_single_feature_suspiciously_predictive":
            rank_corr[top_feature] < single_feature_rank_corr_limit,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "customers_not_yet_acquired": not_yet_customer,
            "label_mismatches": label_mismatch,
            "spend_reconciliation_mismatches": spend_mismatch,
            "min_gap_between_last_event_and_cutoff_seconds": round(latest_event_gap_days * 86_400,
                                                                   1),
            "invariance_probe_run": str(probe.date()),
            "last_train_label_end": str(train_label_end.date()),
            "first_holdout_run": str(min(plan.holdout).date()),
            "most_predictive_single_feature": top_feature,
            "most_predictive_single_feature_rank_corr": round(rank_corr[top_feature], 4),
            "single_feature_rank_corr_limit": single_feature_rank_corr_limit,
        },
    }
