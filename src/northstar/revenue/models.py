"""Customer value models: two RFM-style baselines, a probabilistic CLV model and a learned model.

* ``run_rate`` - "the next 180 days look like the last 180": predicted revenue = net revenue in the
  trailing window of the same length. The implicit rule behind most "top spenders" lists.
* ``rfm_cell_mean`` - classic recency x frequency x monetary segmentation: each customer gets the
  smoothed training mean of future revenue in their RFM cell.
* ``bgnbd_gamma_gamma`` - the probabilistic CLV model of :mod:`northstar.revenue.clv`. It is fit on
  each run's own pre-cutoff history (no labels), so its "training" here is a no-op; its prediction
  is the ``clv_expected_revenue`` column built with the features.
* ``gradient_boosting`` - histogram gradient-boosted trees with a Poisson loss, which models the
  conditional *mean* of a non-negative, zero-inflated, right-skewed target and cannot predict
  negative revenue. The CLV outputs are among its inputs.

All four expose ``fit(X, y)`` and ``predict(X)`` so they are evaluated identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from northstar.acquisition.models import SEED
from northstar.revenue.dataset import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

MODEL_NAMES = ("run_rate", "rfm_cell_mean", "bgnbd_gamma_gamma", "gradient_boosting")
BASELINES = ("run_rate", "rfm_cell_mean")
MODEL_LABELS = {
    "run_rate": "Trailing 180-day revenue (baseline)",
    "rfm_cell_mean": "RFM cell mean (baseline)",
    "bgnbd_gamma_gamma": "BG/NBD + Gamma-Gamma CLV",
    "gradient_boosting": "Gradient boosting (Poisson)",
}

RECENCY_EDGES = (0, 30, 90, 180, 365, np.inf)  # days since last order
FREQUENCY_EDGES = (0, 1, 3, 7, np.inf)  # lifetime orders: 1, 2-3, 4-7, 8+
MONETARY_BANDS = 3  # terciles of average order value, cut on the training data


class RunRateBaseline(RegressorMixin, BaseEstimator):
    """Predicted revenue = net revenue in the trailing 180 days."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RunRateBaseline:
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["revenue_180d"].to_numpy(float)


class RFMCellBaseline(RegressorMixin, BaseEstimator):
    """Smoothed training mean of future revenue per recency x frequency x monetary cell."""

    def __init__(self, prior_weight: float = 20.0):
        self.prior_weight = prior_weight

    def cells(self, X: pd.DataFrame) -> pd.Series:
        r = np.digitize(X["days_since_last_order"].to_numpy(float), RECENCY_EDGES[1:-1])
        f = np.digitize(X["orders_total"].to_numpy(float), FREQUENCY_EDGES[1:-1], right=True)
        m = np.digitize(X["avg_order_value"].to_numpy(float), self.monetary_edges_)
        return pd.Series([f"R{a}F{b}M{c}" for a, b, c in zip(r, f, m, strict=True)],
                         index=X.index)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RFMCellBaseline:
        y = np.asarray(y, dtype=float)
        qs = np.linspace(0, 1, MONETARY_BANDS + 1)[1:-1]
        self.monetary_edges_ = np.quantile(X["avg_order_value"].to_numpy(float), qs)
        self.global_mean_ = float(y.mean())
        stats = pd.DataFrame({"cell": self.cells(X).to_numpy(), "y": y}).groupby("cell")["y"].agg(
            ["sum", "count"])
        self.means_ = ((stats["sum"] + self.prior_weight * self.global_mean_)
                       / (stats["count"] + self.prior_weight)).to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.cells(X).map(self.means_).fillna(self.global_mean_).to_numpy(float)


class ProbabilisticCLV(RegressorMixin, BaseEstimator):
    """BG/NBD expected purchases x Gamma-Gamma expected spend, refit per run on history only."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ProbabilisticCLV:
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["clv_expected_revenue"].to_numpy(float)


def make_gradient_boosting() -> Pipeline:
    pre = ColumnTransformer(
        [("num", "passthrough", list(NUMERIC_FEATURES)),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
          list(CATEGORICAL_FEATURES))],
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=60,
        l2_regularization=1.0, early_stopping=False, random_state=SEED)
    return Pipeline([("pre", pre), ("model", model)])


def make_model(name: str) -> BaseEstimator:
    factories = {
        "run_rate": RunRateBaseline,
        "rfm_cell_mean": RFMCellBaseline,
        "bgnbd_gamma_gamma": ProbabilisticCLV,
        "gradient_boosting": make_gradient_boosting,
    }
    return factories[name]()


def fit_model(name: str, data: pd.DataFrame) -> BaseEstimator:
    return make_model(name).fit(data[list(FEATURES)], data[TARGET].to_numpy(float))


def predict(model: BaseEstimator, data: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(data[list(FEATURES)]), 0.0, None)
