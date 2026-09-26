"""Leak-free lead scoring dataset: open-lead pipeline, point-in-time features and labels.

Framing. On the first day of each month (a *scoring run*, the cutoff) marketing and sales look at
the **open pipeline**: leads created in the previous ``PIPELINE_DAYS`` days that have not yet
placed a first order. Each open lead is scored for the probability that it converts (places its
first order) within the next ``HORIZON_DAYS`` days, and outreach capacity goes to the top of the
list.

Leakage rules, enforced here and re-checked by :func:`leakage_audit` and the tests:

* Features are computed only from ``timeline.snapshot(tables, cutoff)``: every event used has a
  timestamp strictly before the cutoff. Funnel stages reached after the cutoff inside a session
  that started before it are therefore invisible, too.
* Leads that converted before the cutoff are not in the pipeline, so no feature can describe
  post-conversion behavior (and pre-cutoff rows of open leads never carry a ``customer_id``).
* The label uses only ``customers.customer_since`` in ``[cutoff, cutoff + horizon)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from northstar.timeline import DATA_END, TIME_COLUMNS, PredictionWindow, snapshot

HORIZON_DAYS = 30
PIPELINE_DAYS = 90
TABLES_USED = ("prospects", "customers", "sessions", "funnel_events", "marketing_touches")

CATEGORICAL_FEATURES = ("acquisition_channel", "region", "age_band", "income_band", "device_type")
NUMERIC_FEATURES = (
    "email_opt_in",
    "lead_age_days",
    "sessions_total",
    "sessions_7d",
    "sessions_30d",
    "days_since_last_session",
    "max_stage_reached",
    "cart_sessions",
    "cart_sessions_7d",
    "checkout_sessions",
    "pages_viewed",
    "app_session_share",
    "mobile_session_share",
    "emails_received",
    "emails_opened",
    "emails_clicked",
    "email_open_rate",
    "retarget_clicks",
)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
KEY_COLUMNS = ("run_cutoff", "prospect_id")
TARGET = "converted"

# Columns that would describe the outcome or post-conversion behavior. None may be a feature.
FORBIDDEN_FEATURES = frozenset(
    {"customer_id", "customer_since", "converted", "order_id", "order_ts", "net_amount",
     "gross_amount", "item_count", "purchase"}
)


def _days(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 86_400


def open_leads(prospects: pd.DataFrame, customers: pd.DataFrame, cutoff: pd.Timestamp,
               pipeline_days: int = PIPELINE_DAYS) -> pd.DataFrame:
    """Leads created in ``[cutoff - pipeline_days, cutoff)`` with no first order before cutoff."""
    cutoff = pd.Timestamp(cutoff)
    created = prospects["created_at"]
    recent = (created < cutoff) & (created >= cutoff - pd.Timedelta(days=pipeline_days))
    converted_before = prospects["prospect_id"].isin(
        customers.loc[customers["customer_since"] < cutoff, "prospect_id"])
    return prospects.loc[recent & ~converted_before].reset_index(drop=True)


def conversion_labels(lead_ids: pd.Series, customers: pd.DataFrame,
                      window: PredictionWindow) -> np.ndarray:
    """1 if the lead's first order falls in the outcome window ``[cutoff, cutoff + horizon)``."""
    in_window = customers.loc[window.label_mask(customers["customer_since"]), "prospect_id"]
    return lead_ids.isin(in_window).to_numpy(dtype=int)


def _session_features(view: Mapping[str, pd.DataFrame], ids: pd.Index,
                      cutoff: pd.Timestamp) -> pd.DataFrame:
    sessions = view["sessions"]
    sessions = sessions.loc[sessions["prospect_id"].isin(ids)]
    depth = view["funnel_events"].groupby("session_id")["stage_number"].max()
    s = pd.DataFrame({
        "prospect_id": sessions["prospect_id"].to_numpy(),
        "age": _days(cutoff - sessions["session_start"]).to_numpy(),
        "depth": sessions["session_id"].map(depth).fillna(1).to_numpy(),
        "pages": sessions["pages_viewed"].to_numpy(),
        "app": (sessions["platform"] == "app").to_numpy(),
        "mobile": (sessions["device_type"] == "mobile").to_numpy(),
    })
    s["recent7"] = s["age"] < 7
    s["recent30"] = s["age"] < 30
    s["cart"] = s["depth"] >= 3
    s["cart7"] = s["cart"] & s["recent7"]
    s["checkout"] = s["depth"] >= 4
    g = s.groupby("prospect_id")
    return pd.DataFrame({
        "sessions_total": g.size(),
        "sessions_7d": g["recent7"].sum(),
        "sessions_30d": g["recent30"].sum(),
        "days_since_last_session": g["age"].min(),
        "max_stage_reached": g["depth"].max(),
        "cart_sessions": g["cart"].sum(),
        "cart_sessions_7d": g["cart7"].sum(),
        "checkout_sessions": g["checkout"].sum(),
        "pages_viewed": g["pages"].sum(),
        "app_session_share": g["app"].mean(),
        "mobile_session_share": g["mobile"].mean(),
    }).reindex(ids)


def _touch_features(view: Mapping[str, pd.DataFrame], leads: pd.DataFrame) -> pd.DataFrame:
    touches = view["marketing_touches"]
    t = touches.loc[touches["prospect_id"].isin(leads["prospect_id"])]
    # The sourcing touch is logged at the lead's creation instant; it is described by the
    # acquisition channel already, so only follow-up touches count as engagement.
    created = t["prospect_id"].map(leads.set_index("prospect_id")["created_at"])
    t = t.loc[t["touch_at"] > created]
    email = t["touch_type"] == "email"
    t = t.assign(email=email, opened_email=email & t["opened"], clicked_email=email & t["clicked"],
                 retarget=t["touch_type"] == "ad_click")
    g = t.groupby("prospect_id")
    out = pd.DataFrame({
        "emails_received": g["email"].sum(),
        "emails_opened": g["opened_email"].sum(),
        "emails_clicked": g["clicked_email"].sum(),
        "retarget_clicks": g["retarget"].sum(),
    }).reindex(leads["prospect_id"]).fillna(0)
    out["email_open_rate"] = (out["emails_opened"] / out["emails_received"]).where(
        out["emails_received"] > 0, 0.0)
    return out


def build_features(tables: Mapping[str, pd.DataFrame], cutoff: str | pd.Timestamp,
                   pipeline_days: int = PIPELINE_DAYS) -> pd.DataFrame:
    """Point-in-time features for every open lead at ``cutoff`` (one row per lead).

    Only ``snapshot(tables, cutoff)`` is read, so the result is identical whether or not the
    input contains rows at or after the cutoff.
    """
    cutoff = pd.Timestamp(cutoff)
    view = snapshot({name: tables[name] for name in TABLES_USED}, cutoff)
    leads = open_leads(view["prospects"], view["customers"], cutoff, pipeline_days)
    ids = pd.Index(leads["prospect_id"])

    sess = _session_features(view, ids, cutoff)
    touch = _touch_features(view, leads)
    lead_age = _days(cutoff - leads["created_at"]).to_numpy()
    out = pd.DataFrame({
        "run_cutoff": cutoff,
        "prospect_id": leads["prospect_id"].to_numpy(),
        **{c: leads[c].astype(str).to_numpy() for c in CATEGORICAL_FEATURES},
        "email_opt_in": leads["email_opt_in"].astype(int).to_numpy(),
        "lead_age_days": lead_age,
    })
    for col in sess.columns:
        out[col] = sess[col].to_numpy(dtype=float)
    for col in touch.columns:
        out[col] = touch[col].to_numpy(dtype=float)
    # Every lead has its creation session; guard anyway so the schema never contains NaN.
    counts = [c for c in sess.columns if c not in ("days_since_last_session", "max_stage_reached")]
    out[counts] = out[counts].fillna(0.0)
    out["days_since_last_session"] = out["days_since_last_session"].fillna(out["lead_age_days"])
    out["max_stage_reached"] = out["max_stage_reached"].fillna(0.0)
    return out[[*KEY_COLUMNS, *FEATURES]]


def build_run(tables: Mapping[str, pd.DataFrame], cutoff: str | pd.Timestamp,
              horizon_days: int = HORIZON_DAYS, pipeline_days: int = PIPELINE_DAYS
              ) -> pd.DataFrame:
    """Features plus the ``converted`` label for one scoring run."""
    window = PredictionWindow(pd.Timestamp(cutoff), horizon_days=horizon_days)
    features = build_features(tables, window.cutoff, pipeline_days)
    features[TARGET] = conversion_labels(features["prospect_id"], tables["customers"], window)
    return features


def build_dataset(tables: Mapping[str, pd.DataFrame], cutoffs: Iterable[pd.Timestamp],
                  horizon_days: int = HORIZON_DAYS, pipeline_days: int = PIPELINE_DAYS
                  ) -> pd.DataFrame:
    return pd.concat([build_run(tables, c, horizon_days, pipeline_days) for c in cutoffs],
                     ignore_index=True)


@dataclass(frozen=True)
class SplitPlan:
    """Monthly scoring runs, split in time.

    * ``fit`` runs train candidate models; ``validation`` runs pick the champion.
    * The champion is then refit on ``fit + validation`` (all *train* runs) and evaluated once on
      the ``holdout`` runs, whose cutoffs are all on/after the last training label window ends.
    """

    fit: tuple[pd.Timestamp, ...]
    validation: tuple[pd.Timestamp, ...]
    holdout: tuple[pd.Timestamp, ...]
    horizon_days: int = HORIZON_DAYS

    def __post_init__(self) -> None:
        if not (self.fit and self.validation and self.holdout):
            raise ValueError("fit, validation and holdout runs must all be non-empty")
        self._check_ordered(self.fit, self.validation)
        self._check_ordered(self.train, self.holdout)
        last_label_end = max(self.holdout) + pd.Timedelta(days=self.horizon_days)
        if last_label_end > DATA_END:
            raise ValueError("holdout outcome window extends past the end of the data")

    def _check_ordered(self, earlier: tuple[pd.Timestamp, ...],
                       later: tuple[pd.Timestamp, ...]) -> None:
        label_end = max(earlier) + pd.Timedelta(days=self.horizon_days)
        if label_end > min(later):
            raise ValueError(
                f"outcome window of run {max(earlier).date()} ends {label_end.date()}, after the "
                f"first later run {min(later).date()}: labels would leak across the split")

    @property
    def train(self) -> tuple[pd.Timestamp, ...]:
        return self.fit + self.validation

    def role(self, cutoff: pd.Timestamp) -> str:
        if cutoff in self.fit:
            return "fit"
        if cutoff in self.validation:
            return "validation"
        if cutoff in self.holdout:
            return "holdout"
        raise KeyError(cutoff)

    def as_dict(self) -> dict:
        def dates(runs: tuple[pd.Timestamp, ...]) -> list[str]:
            return [str(c.date()) for c in runs]

        return {"horizon_days": self.horizon_days, "fit_runs": dates(self.fit),
                "validation_runs": dates(self.validation), "holdout_runs": dates(self.holdout)}


def monthly_runs(first: str, last: str) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range(first, last, freq="MS"))


# Default design: 12 fitting runs (Apr 2024 - Mar 2025), 3 validation runs (Apr - Jun 2025) and six
# out-of-time holdout runs (Jul - Dec 2025) scored by a model frozen at the 2025-07-01 cutoff.
DEFAULT_SPLIT = SplitPlan(
    fit=monthly_runs("2024-04-01", "2025-03-01"),
    validation=monthly_runs("2025-04-01", "2025-06-01"),
    holdout=monthly_runs("2025-07-01", "2025-12-01"),
)


def leakage_audit(tables: Mapping[str, pd.DataFrame], dataset: pd.DataFrame, plan: SplitPlan,
                  single_feature_auc_limit: float = 0.9) -> dict:
    """Runtime leakage checks on the assembled dataset; ``passed`` is False if any check fails."""
    from sklearn.metrics import roc_auc_score

    customers = tables["customers"]
    since = dataset["prospect_id"].map(customers.set_index("prospect_id")["customer_since"])
    converted_before = int((since < dataset["run_cutoff"]).sum())

    # Pre-cutoff rows of pipeline leads must not be linked to a customer record.
    linked = 0
    latest_event_gap_days = np.inf
    for cutoff, run in dataset.groupby("run_cutoff"):
        view = snapshot({n: tables[n] for n in TABLES_USED}, cutoff)
        for name in ("sessions", "marketing_touches"):
            rows = view[name].loc[view[name]["prospect_id"].isin(run["prospect_id"])]
            linked += int(rows["customer_id"].notna().sum())
            ts = rows[TIME_COLUMNS[name]]
            if len(ts):
                latest_event_gap_days = min(latest_event_gap_days,
                                            (cutoff - ts.max()).total_seconds() / 86_400)

    # Recompute one holdout run from data truncated at its cutoff: features must not change.
    probe = plan.holdout[0]
    truncated = snapshot({n: tables[n] for n in TABLES_USED}, probe)
    full_run = dataset.loc[dataset["run_cutoff"] == probe, [*KEY_COLUMNS, *FEATURES]]
    trunc_run = build_features(truncated, probe)
    invariant = full_run.reset_index(drop=True).equals(trunc_run.reset_index(drop=True))

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
        "no_pipeline_lead_converted_before_cutoff": converted_before == 0,
        "no_customer_linked_rows_in_feature_inputs": linked == 0,
        "no_outcome_columns_used_as_features": not (set(FEATURES) & FORBIDDEN_FEATURES),
        "train_labels_end_before_holdout_starts": bool(train_label_end <= min(plan.holdout)),
        "no_single_feature_suspiciously_predictive":
            single_auc[top_feature] < single_feature_auc_limit,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "pipeline_leads_converted_before_cutoff": converted_before,
            "customer_linked_rows": linked,
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

