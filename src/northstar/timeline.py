"""Shared time concepts for leak-free, time-aware analysis.

Every analysis in the portfolio follows one convention:

* The simulated business history covers ``[DATA_START, DATA_END)`` (24 months).
* A **cutoff** is the instant at which a model or analysis is "run". Features may only use
  information with a timestamp strictly before the cutoff (``ts < cutoff``).
* Outcomes/labels are measured in a window that starts at the cutoff:
  ``cutoff <= ts < cutoff + horizon``.

``PredictionWindow`` encodes that convention and ``snapshot`` returns a point-in-time view of the
shared tables, so later sections can build features from ``snapshot(tables, cutoff)`` and labels
from the full tables without re-implementing the filtering rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

DATA_START = pd.Timestamp("2024-01-01")
DATA_END = pd.Timestamp("2026-01-01")  # exclusive

# Default "as-of" date for modelling sections: 18 months of history before it and six months
# after it, enough for 90/180-day outcome windows plus a later out-of-time evaluation period.
DEFAULT_CUTOFF = pd.Timestamp("2025-07-01")

# Time column for each table. ``order_lines`` inherits its time from ``orders``; ``products`` is a
# static catalog with no time dimension.
TIME_COLUMNS: dict[str, str] = {
    "campaigns": "start_date",
    "experiments": "start_date",
    "prospects": "created_at",
    "customers": "customer_since",
    "marketing_touches": "touch_at",
    "sessions": "session_start",
    "funnel_events": "event_ts",
    "orders": "order_ts",
    "subscription_events": "event_ts",
    "support_contacts": "contact_ts",
    "experiment_assignments": "assigned_at",
}


def _ts(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value)


@dataclass(frozen=True)
class PredictionWindow:
    """Feature (lookback) and outcome (horizon) windows around a cutoff.

    ``feature_start <= ts < cutoff`` is observable history; ``cutoff <= ts < label_end`` is the
    outcome period. The two windows never overlap by construction.
    """

    cutoff: pd.Timestamp
    horizon_days: int
    lookback_days: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff", _ts(self.cutoff))
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.lookback_days is not None and self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive when provided")
        if not DATA_START < self.cutoff < DATA_END:
            raise ValueError(f"cutoff {self.cutoff} is outside the data range")
        if self.label_end > DATA_END:
            raise ValueError(
                f"outcome window ends {self.label_end.date()}, after the data ends "
                f"({DATA_END.date()}); labels would be right-censored"
            )

    @property
    def feature_start(self) -> pd.Timestamp:
        if self.lookback_days is None:
            return DATA_START
        return max(DATA_START, self.cutoff - pd.Timedelta(days=self.lookback_days))

    @property
    def feature_end(self) -> pd.Timestamp:
        return self.cutoff

    @property
    def label_start(self) -> pd.Timestamp:
        return self.cutoff

    @property
    def label_end(self) -> pd.Timestamp:
        return self.cutoff + pd.Timedelta(days=self.horizon_days)

    def feature_mask(self, ts: pd.Series) -> pd.Series:
        return (ts >= self.feature_start) & (ts < self.feature_end)

    def label_mask(self, ts: pd.Series) -> pd.Series:
        return (ts >= self.label_start) & (ts < self.label_end)


def events_before(df: pd.DataFrame, ts_col: str, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Rows strictly before ``cutoff``: the only rows a feature built at ``cutoff`` may see."""
    return df.loc[df[ts_col] < _ts(cutoff)]


def events_between(
    df: pd.DataFrame, ts_col: str, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DataFrame:
    """Rows in the half-open interval ``[start, end)``."""
    ts = df[ts_col]
    return df.loc[(ts >= _ts(start)) & (ts < _ts(end))]


def rolling_cutoffs(
    first: str | pd.Timestamp, last: str | pd.Timestamp, step_days: int
) -> list[pd.Timestamp]:
    """Evenly spaced cutoffs from ``first`` to ``last`` inclusive (for backtests)."""
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    first_ts, last_ts = _ts(first), _ts(last)
    if last_ts < first_ts:
        raise ValueError("last must not precede first")
    cutoffs = []
    current = first_ts
    while current <= last_ts:
        cutoffs.append(current)
        current += pd.Timedelta(days=step_days)
    return cutoffs


def snapshot(
    tables: Mapping[str, pd.DataFrame], cutoff: str | pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Point-in-time view of the shared tables as they would have looked at ``cutoff``.

    * Event and entity tables keep rows whose time column is strictly before the cutoff.
    * ``order_lines`` keeps lines of the surviving orders.
    * Support contacts opened before the cutoff but resolved on/after it have their resolution
      time and CSAT masked, because neither was known yet.
    * Foreign keys that point at entities not yet created (``customer_id`` on rows logged after
      conversion) cannot occur because such rows are themselves after the cutoff.
    """
    cutoff_ts = _ts(cutoff)
    out: dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        if name in TIME_COLUMNS:
            out[name] = events_before(df, TIME_COLUMNS[name], cutoff_ts).copy()
        else:
            out[name] = df.copy()

    if "order_lines" in out and "orders" in out:
        lines = out["order_lines"]
        out["order_lines"] = lines.loc[lines["order_id"].isin(out["orders"]["order_id"])].copy()

    if "support_contacts" in out:
        contacts = out["support_contacts"]
        unresolved_at_cutoff = contacts["resolved_at"] >= cutoff_ts
        contacts.loc[unresolved_at_cutoff, "resolved_at"] = pd.NaT
        contacts.loc[unresolved_at_cutoff, "csat_score"] = pd.NA

    return out
