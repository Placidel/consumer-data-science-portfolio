"""Customer value evaluation: dollar error, calibration-in-the-large, ranking and revenue capture.

Future revenue is zero for most customers and heavily right-skewed, so no single metric is enough:

* **RMSE** is minimized by the conditional mean, the quantity that adds up to a revenue forecast,
  so it selects the champion. **MAE** is reported too, but it is minimized by the conditional
  *median* (zero for most customers), so it rewards under-prediction here.
* **Bias** (total predicted / total actual - 1) is calibration-in-the-large: can the predictions
  be summed into a plan?
* **Normalized Gini** (Gini of the revenue Lorenz curve when customers are ordered by the
  prediction, divided by the oracle's) and **revenue capture** of the top ``k%`` measure what the
  business uses the score for: ranking.

Ranking metrics order the pooled evaluation set. Ties are resolved by their expectation under
random tie-breaking (the Lorenz curve is linear inside a tied group), so no metric depends on row
order. The default design evaluates single scoring runs, so pooling never mixes runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

TOP_SHARES = (0.01, 0.05, 0.1, 0.2)


def share_tag(share: float) -> str:
    return f"top{round(share * 100)}"


def lorenz(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of customers and of revenue when ranking by ``score`` (descending).

    Points sit at tied-group boundaries starting from (0, 0); interpolating linearly between them
    gives the expected curve under random tie-breaking.
    """
    y, score = np.asarray(y, float), np.asarray(score, float)
    if len(y) == 0 or y.sum() <= 0:
        raise ValueError("need at least one customer with positive revenue")
    groups = (pd.DataFrame({"s": score, "y": y}).groupby("s")["y"].agg(["sum", "count"])
              .sort_index(ascending=False))
    cum_n = np.concatenate([[0.0], groups["count"].cumsum().to_numpy(float) / len(y)])
    cum_y = np.concatenate([[0.0], groups["sum"].cumsum().to_numpy(float) / y.sum()])
    return cum_n, cum_y


def capture(y: np.ndarray, score: np.ndarray, shares: Sequence[float]) -> np.ndarray:
    """Share of total revenue held by the top ``share`` of customers ranked by ``score``."""
    cum_n, cum_y = lorenz(y, score)
    return np.interp(np.asarray(shares, float), cum_n, cum_y)


def gini(y: np.ndarray, score: np.ndarray) -> float:
    """Gini of the revenue Lorenz curve ordered by ``score`` (0 = random, oracle = maximum)."""
    cum_n, cum_y = lorenz(y, score)
    area = np.sum(np.diff(cum_n) * (cum_y[1:] + cum_y[:-1]) / 2)  # trapezoid rule
    return float(2 * area - 1)


def normalized_gini(y: np.ndarray, score: np.ndarray) -> float:
    """Gini relative to ranking by the realized revenue itself (1 = perfect ranking)."""
    return gini(y, score) / gini(y, y)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pred, float))))


def score_metrics(y: np.ndarray, pred: np.ndarray, shares: Sequence[float] = TOP_SHARES) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    sst = float(np.sum((y - y.mean()) ** 2))
    out = {
        "rows": len(y),
        "buyer_rate": float((y > 0).mean()),
        "actual_mean": float(y.mean()),
        "predicted_mean": float(pred.mean()),
        "bias": float(pred.sum() / y.sum() - 1),
        "mae": mae(y, pred),
        "rmse": rmse(y, pred),
        "r2": float(1 - np.sum((y - pred) ** 2) / sst),
        # Undefined for a constant prediction (no ranking at all).
        "spearman": (float(pd.Series(pred).rank().corr(pd.Series(y).rank()))
                     if np.ptp(pred) > 0 else float("nan")),
        "normalized_gini": normalized_gini(y, pred),
    }
    for share, captured in zip(shares, capture(y, pred, shares), strict=True):
        out[f"capture_{share_tag(share)}"] = float(captured)
        out[f"lift_{share_tag(share)}"] = float(captured / share)
    return out


def decile_table(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Customers in ten equal groups by predicted value (1 = highest), predicted vs actual."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rank = pd.Series(pred).rank(method="first", ascending=False)
    frame = pd.DataFrame({"decile": pd.qcut(rank, 10, labels=False).to_numpy() + 1, "y": y,
                          "pred": pred})
    out = frame.groupby("decile").agg(customers=("y", "size"), mean_predicted=("pred", "mean"),
                                      mean_actual=("y", "mean"),
                                      buyer_rate=("y", lambda s: (s > 0).mean()),
                                      actual_revenue=("y", "sum"))
    out["share_of_revenue"] = out["actual_revenue"] / y.sum()
    out["cumulative_capture"] = out["share_of_revenue"].cumsum()
    out["predicted_to_actual"] = out["mean_predicted"] / out["mean_actual"]
    return out.reset_index()


def gains_table(y: np.ndarray, scores: Mapping[str, np.ndarray], shares: Sequence[float]
                ) -> pd.DataFrame:
    """Revenue capture curves: random, perfect foresight (oracle) and each score."""
    shares = np.asarray(shares, float)
    return pd.DataFrame({"share_targeted": shares, "random": shares,
                         "oracle": capture(y, y, shares),
                         **{name: capture(y, s, shares) for name, s in scores.items()}})


def capture_table(y: np.ndarray, scores: Mapping[str, np.ndarray], shares: Sequence[float]
                  ) -> pd.DataFrame:
    """Revenue captured by the top ``share`` of customers, per policy, with lift over random."""
    y = np.asarray(y, float)
    total = y.sum()
    policies = {"random": None, **scores, "oracle": y}
    rows = []
    for share in shares:
        k = share * len(y)
        for name, s in policies.items():
            captured = share if s is None else float(capture(y, s, [share])[0])
            rows.append({"share_targeted": share, "policy": name, "customers": k,
                         "revenue_captured": captured * total, "share_of_revenue": captured,
                         "revenue_per_customer": captured * total / k, "lift": captured / share})
    return pd.DataFrame(rows)


def active_subset_metrics(data: pd.DataFrame, target: str, scores: Sequence[str],
                          active_col: str = "orders_180d") -> dict:
    """Headline metrics restricted to customers with an order in the trailing 180 days.

    A robustness check: separating lapsed customers (who mostly spend nothing) from active ones is
    easy, so a model should also rank well *within* the active base.
    """
    active = data.loc[data[active_col] > 0]
    y = active[target].to_numpy(float)
    return {"customers": len(active), "share_of_base": len(active) / len(data),
            "share_of_future_revenue": y.sum() / data[target].sum(),
            **{s: {"rmse": rmse(y, active[s]), "normalized_gini": normalized_gini(y, active[s]),
                   "capture_top10": float(capture(y, active[s], [0.1])[0]),
                   "bias": float(active[s].sum() / y.sum() - 1)} for s in scores}}


BOOTSTRAP_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": rmse,
    "mae": mae,
    "normalized_gini": normalized_gini,
    "capture_top10": lambda y, s: float(capture(y, s, [0.1])[0]),
}


def cluster_bootstrap(data: pd.DataFrame, target: str, scores: Sequence[str],
                      references: Sequence[str], n_boot: int = 200, seed: int = 0,
                      cluster: str = "customer_id") -> dict[str, list[float]]:
    """Percentile 95% CIs for each metric and paired differences vs. each of ``references``.

    Whole ``cluster`` units are resampled, so a customer scored in several runs is never split.
    """
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(data[cluster])
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    sizes = np.diff(np.append(starts, len(codes)))
    y = data[target].to_numpy(float)
    values = {s: data[s].to_numpy(float) for s in scores}
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        picked = rng.integers(0, len(starts), len(starts))
        n_rows = sizes[picked]
        offsets = np.arange(n_rows.sum()) - np.repeat(np.cumsum(n_rows) - n_rows, n_rows)
        idx = order[np.repeat(starts[picked], n_rows) + offsets]
        stats = {f"{s}.{m}": fn(y[idx], values[s][idx])
                 for s in scores for m, fn in BOOTSTRAP_METRICS.items()}
        for s in scores:
            for ref in references:
                if s != ref:
                    for m in BOOTSTRAP_METRICS:
                        stats[f"{s}.{m}_minus_{ref}"] = stats[f"{s}.{m}"] - stats[f"{ref}.{m}"]
        for key, v in stats.items():
            draws.setdefault(key, []).append(float(v))
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for key, v in draws.items()}
