"""Churn model evaluation: discrimination, calibration, targeting depth and segment drivers.

The retention budget is allocated *within* each monthly run, so every depth metric ranks
customers inside a run and then pools the runs. The tie-aware ranking primitives are shared with
section 01, so tied scores (the RFM baseline gives a whole cell one score) are resolved by their
expectation under random tie-breaking and no metric depends on row order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from northstar.acquisition.evaluation import (
    calibration_slope,
    calibration_table,
    capture_by_share,
    expected_calibration_error,
)
from northstar.retention.dataset import TARGET

RUN = "run_cutoff"
DEPTHS = (0.05, 0.1, 0.2, 0.3)


def depth_tag(depth: float) -> str:
    return f"top{round(depth * 100)}"


def score_metrics(data: pd.DataFrame, score: str, probabilistic: bool,
                  depths: Sequence[float] = DEPTHS) -> dict:
    y, p = data[TARGET].to_numpy(), data[score].to_numpy()
    base = float(y.mean())
    out = {
        "rows": len(data),
        "churners": int(y.sum()),
        "base_rate": base,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }
    for depth in depths:
        contacts, hits = capture_by_share(data, score, depth, TARGET)
        tag = depth_tag(depth)
        out[f"precision_{tag}"] = hits / contacts
        out[f"recall_{tag}"] = hits / y.sum()
        out[f"lift_{tag}"] = hits / contacts / base
    if probabilistic:
        out.update({
            "brier": float(brier_score_loss(y, p)),
            "brier_skill": float(1 - brier_score_loss(y, p) / (base * (1 - base))),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "mean_predicted": float(p.mean()),
            "ece": expected_calibration_error(y, p),
            "calibration_slope": calibration_slope(y, p),
        })
    return out


def depth_table(data: pd.DataFrame, scores: Mapping[str, str],
                depths: Sequence[float] = DEPTHS) -> pd.DataFrame:
    """Precision / recall / lift when a fixed share of every run is targeted, per score."""
    runs = data[RUN].nunique()
    total = data[TARGET].sum()
    base = data[TARGET].mean()
    rows = []
    for depth in depths:
        rows.append({"depth": depth, "model": "random",
                     "customers_per_run": depth * len(data) / runs,
                     "churners_reached_per_run": depth * total / runs, "precision": base,
                     "recall": depth, "lift": 1.0})
        for name, col in scores.items():
            contacts, hits = capture_by_share(data, col, depth, TARGET)
            rows.append({"depth": depth, "model": name, "customers_per_run": contacts / runs,
                         "churners_reached_per_run": hits / runs, "precision": hits / contacts,
                         "recall": hits / total, "lift": hits / contacts / base})
    return pd.DataFrame(rows)


def decile_table(data: pd.DataFrame, score: str) -> pd.DataFrame:
    """Customers ranked within each run into ten equal groups (1 = highest risk), pooled."""
    decile = data.groupby(RUN)[score].transform(
        lambda s: pd.qcut(s.rank(method="first", ascending=False), 10, labels=False) + 1)
    frame = data.assign(decile=decile.astype(int))
    out = frame.groupby("decile").agg(customers=(TARGET, "size"), churners=(TARGET, "sum"),
                                      mean_predicted=(score, "mean"))
    out["churn_rate"] = out["churners"] / out["customers"]
    out["lift"] = out["churn_rate"] / frame[TARGET].mean()
    out["cumulative_capture"] = out["churners"].cumsum() / out["churners"].sum()
    return out.reset_index()


def _bands(values: pd.Series, edges: Sequence[float], labels: Sequence[str]) -> pd.Series:
    return pd.cut(values, bins=list(edges), labels=list(labels), right=False).astype(str)


def segments(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Business segments used for the driver table (all known at the cutoff)."""
    return {
        "lifetime_orders": _bands(data["orders_total"], [1, 2, 4, 8, np.inf],
                                  ["1", "2-3", "4-7", "8+"]),
        "days_since_last_order": _bands(data["days_since_last_order"], [0, 30, 60, 90, 181],
                                        ["0-29", "30-59", "60-89", "90-180"]),
        "browse_sessions_90d": _bands(data["browse_sessions_90d"], [0, 1, 3, np.inf],
                                      ["0", "1-2", "3+"]),
        "plus_membership": pd.Series(
            np.select([data["plus_cancelled_180d"] > 0, data["plus_member"] > 0],
                      ["cancelled in last 180d", "active member"], "non-member"),
            index=data.index),
        "low_csat_contact_180d": (data["low_csat_contacts_180d"] > 0).map(
            {True: "yes", False: "no"}),
        "first_order_discounted": (data["first_order_discounted"] > 0).map(
            {True: "yes", False: "no"}),
        "acquisition_channel": data["acquisition_channel"].astype(str),
    }


def segment_drivers(data: pd.DataFrame, score: str) -> pd.DataFrame:
    """Observed churn rate vs. mean predicted risk by segment (descriptive, not causal).

    ``relative_risk`` compares a segment's observed churn rate with the whole population's, and
    the gap between observed and predicted doubles as a per-segment calibration check.
    """
    base = data[TARGET].mean()
    rows = []
    for name, labels in segments(data).items():
        grouped = data.groupby(labels.to_numpy(), sort=True)
        for level, g in grouped:
            rows.append({"segment": name, "level": level, "customer_runs": len(g),
                         "share_of_base": len(g) / len(data), "churn_rate": g[TARGET].mean(),
                         "mean_predicted": g[score].mean(),
                         "relative_risk": g[TARGET].mean() / base})
    return pd.DataFrame(rows)


def reliability(data: pd.DataFrame, scores: Sequence[str], bins: int = 10) -> pd.DataFrame:
    return pd.concat([calibration_table(data[TARGET], data[s], bins).assign(model=s)
                      for s in scores], ignore_index=True)
