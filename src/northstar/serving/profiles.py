"""Reference distributions stored with each model, and the drift statistics computed against them.

At training time the serving layer saves a compact *reference profile* of the population the model
was fit on: decile bin edges and shares for every numeric feature, level shares for every
categorical feature, and the score distribution. Monitoring later bins a current batch with the
same edges and compares shares with the Population Stability Index (PSI), so drift can be checked
without keeping training rows next to the model.

PSI = sum over bins of (current - reference) * ln(current / reference). Common industry reading:
below 0.1 stable, 0.1 to 0.25 a moderate shift worth watching, above 0.25 a major shift.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

PSI_BINS = 10
SCORE_QUANTILES = 1001  # percentile lookup resolution: 0.1 percentage points
PSI_FLOOR = 1e-4  # share assigned to empty bins so PSI stays finite


def quantile_edges(values: np.ndarray, bins: int = PSI_BINS) -> list[float]:
    """Interior quantile edges at observed values; repeats collapse, so a discrete feature gets
    one bin per common value (e.g. 0 / 1 / 2+) instead of edges between its values."""
    qs = np.quantile(np.asarray(values, float), np.linspace(0, 1, bins + 1)[1:-1],
                     method="inverted_cdf")
    return [float(q) for q in np.unique(qs)]


def bin_shares(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Shares of ``values`` in the bins (-inf, e0], (e0, e1], ..., (e_last, inf)."""
    idx = np.searchsorted(np.asarray(edges, float), np.asarray(values, float), side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def level_shares(values: pd.Series, levels: Sequence[str]) -> np.ndarray:
    counts = pd.Series(values).astype(str).value_counts().reindex(list(levels), fill_value=0)
    return counts.to_numpy(float) / max(len(values), 1)


def psi(reference: Sequence[float], current: Sequence[float], floor: float = PSI_FLOOR) -> float:
    r = np.clip(np.asarray(reference, float), floor, None)
    c = np.clip(np.asarray(current, float), floor, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _numeric_profile(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    edges = quantile_edges(values)
    return {"edges": edges, "shares": bin_shares(values, edges).tolist(),
            "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def build_profile(frame: pd.DataFrame, scores: np.ndarray, numeric: Sequence[str],
                  categorical: Mapping[str, Sequence[str]], description: str) -> dict:
    """Reference profile of ``frame`` (features) and ``scores`` (model output on those rows)."""
    return {
        "description": description,
        "rows": len(frame),
        "numeric": {f: _numeric_profile(frame[f].to_numpy(float)) for f in numeric},
        "categorical": {f: {"levels": list(levels),
                            "shares": level_shares(frame[f], levels).tolist()}
                        for f, levels in categorical.items()},
        "score": {**_numeric_profile(scores),
                  "quantiles": np.quantile(np.asarray(scores, float),
                                           np.linspace(0, 1, SCORE_QUANTILES)).tolist()},
    }


def reference_percentile(scores: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """Percent of the reference population scoring at or below each score (0-100)."""
    q = np.asarray(quantiles, float)
    return 100.0 * np.searchsorted(q, np.asarray(scores, float), side="right") / len(q)


def feature_drift(profile: Mapping, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: PSI against the reference and a mean (or top-level) comparison."""
    rows = []
    for f, ref in profile["numeric"].items():
        values = frame[f].to_numpy(float)
        rows.append({
            "feature": f, "kind": "numeric",
            "psi": psi(ref["shares"], bin_shares(values, ref["edges"])),
            "reference_mean": ref["mean"], "current_mean": float(values.mean()),
            "outside_reference_range": float(np.mean((values < ref["min"])
                                                     | (values > ref["max"]))),
        })
    for f, ref in profile["categorical"].items():
        cur = level_shares(frame[f], ref["levels"])
        top = int(np.argmax(np.abs(cur - np.asarray(ref["shares"]))))
        rows.append({
            "feature": f, "kind": "categorical",
            "psi": psi(ref["shares"], cur),
            # For categoricals the "means" are the share of the level whose share moved most.
            "reference_mean": float(ref["shares"][top]), "current_mean": float(cur[top]),
            "outside_reference_range": float(np.mean(~frame[f].astype(str).isin(
                [lv for lv, s in zip(ref["levels"], ref["shares"], strict=True) if s > 0]))),
            "most_shifted_level": ref["levels"][top],
        })
    return pd.DataFrame(rows)


def score_drift(profile: Mapping, scores: np.ndarray) -> dict:
    ref = profile["score"]
    return {"psi": psi(ref["shares"], bin_shares(scores, ref["edges"])),
            "reference_mean": ref["mean"], "current_mean": float(np.mean(scores))}
