"""Churn models: two CRM-rule baselines and two learned models.

* ``recency_rule`` - today's practice: the longer since the last order, the higher the risk. It
  ranks but does not produce probabilities, so calibration metrics do not apply to it.
* ``rfm_cell_rate`` - a classic RFM segmentation: each customer gets the smoothed training churn
  rate of their recency x frequency cell. Probabilistic, and a strong, explainable benchmark.
* ``logistic_regression`` - regularized log-odds model on log-scaled features and one-hot
  attributes.
* ``gradient_boosting`` - histogram gradient-boosted trees for interactions such as "recent
  buyer who stopped browsing".

All four expose ``fit(X, y)`` and ``predict_proba(X)`` so they are evaluated identically.
Explainability helpers are shared with section 01.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from northstar.acquisition import models as shared
from northstar.retention.dataset import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

SEED = shared.SEED
MODEL_NAMES = ("recency_rule", "rfm_cell_rate", "logistic_regression", "gradient_boosting")
BASELINES = ("recency_rule", "rfm_cell_rate")
LEARNED = ("logistic_regression", "gradient_boosting")
PROBABILISTIC = ("rfm_cell_rate", "logistic_regression", "gradient_boosting")
MODEL_LABELS = {
    "recency_rule": "Days since last order (baseline)",
    "rfm_cell_rate": "RFM cell churn rate (baseline)",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}

RECENCY_EDGES = (0, 30, 60, 90, 120, np.inf)  # days since last order
FREQUENCY_EDGES = (0, 1, 3, 7, np.inf)  # lifetime orders: 1, 2-3, 4-7, 8+


class RecencyBaseline(ClassifierMixin, BaseEstimator):
    """Rank-only rule: more days since the last order = higher churn risk."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RecencyBaseline:
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # A monotone transform into [0, 1); a ranking score, not a calibrated probability.
        days = X["days_since_last_order"].to_numpy(float)
        p = days / (1.0 + days)
        return np.column_stack([1 - p, p])


def rfm_cell(X: pd.DataFrame) -> pd.Series:
    """Recency x frequency cell label such as ``R2F1`` (bucket indices from 0)."""
    r = np.digitize(X["days_since_last_order"].to_numpy(float), RECENCY_EDGES[1:-1], right=False)
    f = np.digitize(X["orders_total"].to_numpy(float), FREQUENCY_EDGES[1:-1], right=True)
    return pd.Series([f"R{a}F{b}" for a, b in zip(r, f, strict=True)], index=X.index)


class RFMCellBaseline(ClassifierMixin, BaseEstimator):
    """Score = training churn rate of the customer's recency x frequency cell, smoothed."""

    def __init__(self, prior_weight: float = 20.0):
        self.prior_weight = prior_weight

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RFMCellBaseline:
        y = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        self.global_rate_ = float(y.mean())
        stats = pd.DataFrame({"cell": rfm_cell(X).to_numpy(), "y": y}).groupby("cell")["y"].agg(
            ["sum", "count"])
        self.rates_ = ((stats["sum"] + self.prior_weight * self.global_rate_)
                       / (stats["count"] + self.prior_weight)).to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = rfm_cell(X).map(self.rates_).fillna(self.global_rate_).to_numpy(float)
        return np.column_stack([1 - p, p])


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make_logistic() -> Pipeline:
    numeric = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer(
        [("num", numeric, list(NUMERIC_FEATURES)), ("cat", _one_hot(), list(CATEGORICAL_FEATURES))],
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=0.5, max_iter=5000))])


def make_gradient_boosting() -> Pipeline:
    # One-hot rather than native categorical splits so TreeSHAP stays exact (additive).
    pre = ColumnTransformer(
        [("num", "passthrough", list(NUMERIC_FEATURES)),
         ("cat", _one_hot(), list(CATEGORICAL_FEATURES))],
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=80,
        l2_regularization=1.0, early_stopping=False, random_state=SEED)
    return Pipeline([("pre", pre), ("model", model)])


def make_model(name: str) -> BaseEstimator:
    factories = {
        "recency_rule": RecencyBaseline,
        "rfm_cell_rate": RFMCellBaseline,
        "logistic_regression": make_logistic,
        "gradient_boosting": make_gradient_boosting,
    }
    return factories[name]()


def fit_model(name: str, data: pd.DataFrame) -> BaseEstimator:
    return make_model(name).fit(data[list(FEATURES)], data[TARGET].to_numpy())


def predict(model: BaseEstimator, data: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(data[list(FEATURES)])[:, 1]


# ---------------------------------------------------------------- explainability
def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    """Centered log-odds coefficients, largest magnitude first (see section 01's helper)."""
    return shared.logistic_coefficients(model, categorical=CATEGORICAL_FEATURES)


def shap_importance(model: Pipeline, data: pd.DataFrame, background: pd.DataFrame,
                    sample_size: int = 2000, seed: int = SEED
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact SHAP (log-odds of churn), one-hot columns summed per raw feature."""
    return shared.shap_importance(model, data, background, sample_size, seed,
                                  features=FEATURES, categorical=CATEGORICAL_FEATURES)


def reason_codes(per_feature: pd.DataFrame, top: int = 3) -> pd.Series:
    """The ``top`` features pushing each customer's churn risk *up* (largest positive SHAP)."""
    def codes(row: pd.Series) -> str:
        pos = row[row > 0].sort_values(ascending=False).head(top)
        return "; ".join(pos.index)

    return per_feature.apply(codes, axis=1)
