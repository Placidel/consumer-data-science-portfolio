"""Evaluation for imbalanced lead targeting: ranking, calibration, lift and budget policies.

Outreach capacity is allocated *within* each monthly scoring run, so every top-k metric ranks leads
inside a run and then pools the runs. Ties (the channel baseline gives every lead in a channel the
same score) are resolved by their expectation under random tie-breaking, so no metric depends on
row order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

RUN = "run_cutoff"
TARGET = "converted"


def expected_hits_at_k(score: np.ndarray, y: np.ndarray, k: float) -> float:
    """Expected positives among the top ``k`` scores, ties broken uniformly at random.

    ``k`` may be fractional (a share of a run that is not a whole number of leads); the partial
    lead contributes pro rata.
    """
    score, y = np.asarray(score, float), np.asarray(y, float)
    k = float(np.clip(k, 0, len(score)))
    if k == 0:
        return 0.0
    frame = pd.DataFrame({"s": score, "y": y}).groupby("s")["y"].agg(["sum", "count"])
    frame = frame.sort_index(ascending=False)
    taken_before = frame["count"].cumsum() - frame["count"]
    take = np.clip(k - taken_before, 0, frame["count"])
    return float((take / frame["count"] * frame["sum"]).sum())


def capture_by_share(data: pd.DataFrame, score: str, share: float, target: str = TARGET
                     ) -> tuple[float, float]:
    """(contacts, expected positives reached) when contacting ``share`` of every run.

    ``target`` may be any non-negative per-row quantity (a 0/1 label or, say, label x value).
    """
    contacts = hits = 0.0
    for _, run in data.groupby(RUN):
        k = share * len(run)
        contacts += k
        hits += expected_hits_at_k(run[score].to_numpy(), run[target].to_numpy(), k)
    return contacts, hits


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    table = calibration_table(y, p, bins)
    return float((table["n"] * (table["observed_rate"] - table["mean_predicted"]).abs()).sum()
                 / table["n"].sum())


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Equal-frequency bins of predicted probability with observed conversion rates."""
    frame = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False) + 1
    out = frame.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                   observed_rate=("y", "mean"))
    return out.reset_index()


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Slope of a logistic recalibration of y on logit(p); 1 is ideal, <1 means overconfident."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(C=np.inf).fit(logit, y).coef_[0, 0])  # unpenalized


def score_metrics(data: pd.DataFrame, score: str, probabilistic: bool,
                  top_shares: Sequence[float] = (0.1, 0.2)) -> dict:
    y, p = data[TARGET].to_numpy(), data[score].to_numpy()
    base = float(y.mean())
    out = {
        "rows": len(data),
        "conversions": int(y.sum()),
        "base_rate": base,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }
    for share in top_shares:
        contacts, hits = capture_by_share(data, score, share)
        tag = f"top{round(share * 100)}"
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


def per_run_auc(data: pd.DataFrame, score: str, target: str = TARGET) -> dict[str, float]:
    return {str(cutoff.date()): float(roc_auc_score(run[target], run[score]))
            for cutoff, run in data.groupby(RUN)}


def decile_table(data: pd.DataFrame, score: str) -> pd.DataFrame:
    """Leads ranked within each run into ten equal groups (1 = highest score), pooled over runs."""
    decile = data.groupby(RUN)[score].transform(
        lambda s: pd.qcut(s.rank(method="first", ascending=False), 10, labels=False) + 1)
    frame = data.assign(decile=decile.astype(int))
    out = frame.groupby("decile").agg(leads=(TARGET, "size"), conversions=(TARGET, "sum"),
                                      mean_score=(score, "mean"))
    out["conversion_rate"] = out["conversions"] / out["leads"]
    out["lift"] = out["conversion_rate"] / frame[TARGET].mean()
    out["cumulative_capture"] = out["conversions"].cumsum() / out["conversions"].sum()
    return out.reset_index()


def gains_curve(data: pd.DataFrame, score: str, shares: np.ndarray, target: str = TARGET
                ) -> np.ndarray:
    total = data[target].sum()
    return np.array([capture_by_share(data, score, s, target)[1] / total for s in shares])


def budget_simulation(data: pd.DataFrame, policies: Mapping[str, str],
                      capacities: Sequence[float]) -> pd.DataFrame:
    """Conversions reached when a fixed share of each monthly pipeline is contacted.

    ``policies`` maps a policy name to a score column; ``random`` (no targeting) is added and
    evaluated analytically as ``share x conversions``.
    """
    runs = data[RUN].nunique()
    total = data[TARGET].sum()
    rows = []
    for share in capacities:
        results = {"random": (share * len(data), share * total)}
        results.update({name: capture_by_share(data, col, share) for name, col in policies.items()})
        random_hits = results["random"][1]
        for name, (contacts, hits) in results.items():
            rows.append({
                "capacity_share": share,
                "policy": name,
                "contacts_per_run": contacts / runs,
                "conversions_reached_per_run": hits / runs,
                "conversions_reached_total": hits,
                "precision": hits / contacts,
                "share_of_conversions_captured": hits / total,
                "lift_vs_random": hits / random_hits,
                "contacts_per_conversion": contacts / hits if hits else float("nan"),
            })
    return pd.DataFrame(rows)


def cluster_bootstrap(data: pd.DataFrame, scores: Sequence[str], references: Sequence[str],
                      n_boot: int = 200, seed: int = 0, cluster: str = "prospect_id",
                      target: str = TARGET) -> dict:
    """Percentile 95% CIs for AUC / AP and paired differences vs. each of ``references``.

    A lead (or customer) can appear in several monthly runs, so ``cluster`` units, not rows, are
    resampled.
    """
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(data[cluster])
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    sizes = np.diff(np.append(starts, len(codes)))
    y = data[target].to_numpy()
    values = {s: data[s].to_numpy() for s in scores}
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        picked = rng.integers(0, len(starts), len(starts))
        n_rows = sizes[picked]
        offsets = np.arange(n_rows.sum()) - np.repeat(np.cumsum(n_rows) - n_rows, n_rows)
        idx = order[np.repeat(starts[picked], n_rows) + offsets]
        yb = y[idx]
        stats = {}
        for s in scores:
            stats[f"{s}.roc_auc"] = roc_auc_score(yb, values[s][idx])
            stats[f"{s}.average_precision"] = average_precision_score(yb, values[s][idx])
        for s in scores:
            for ref in references:
                if s == ref:
                    continue
                for m in ("roc_auc", "average_precision"):
                    stats[f"{s}.{m}_minus_{ref}"] = stats[f"{s}.{m}"] - stats[f"{ref}.{m}"]
        for key, v in stats.items():
            draws.setdefault(key, []).append(float(v))
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for key, v in draws.items()}
