"""Lead scoring models: two business-rule baselines and two learned models.

* ``channel_rate`` - today's practice: prioritize leads from the channels that historically
  convert best (training-period 30-day conversion rate by acquisition channel).
* ``recent_activity`` - a sales heuristic: call the most recently active leads first. It ranks but
  does not produce probabilities, so calibration metrics do not apply to it.
* ``logistic_regression`` - regularized, interpretable log-odds model on log-scaled counts and
  one-hot attributes.
* ``gradient_boosting`` - histogram gradient-boosted trees that capture interactions such as
  "recent cart activity on a young lead".

All four expose ``fit(X, y)`` and ``predict_proba(X)`` so they are evaluated identically.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from northstar.acquisition.dataset import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

SEED = 20240101
MODEL_NAMES = ("channel_rate", "recent_activity", "logistic_regression", "gradient_boosting")
BASELINES = ("channel_rate", "recent_activity")
LEARNED = ("logistic_regression", "gradient_boosting")
PROBABILISTIC = ("channel_rate", "logistic_regression", "gradient_boosting")
MODEL_LABELS = {
    "channel_rate": "Channel conversion rate (baseline)",
    "recent_activity": "Most recently active first (baseline)",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}


class ChannelRateBaseline(ClassifierMixin, BaseEstimator):
    """Score = smoothed training conversion rate of the lead's acquisition channel."""

    def __init__(self, prior_weight: float = 20.0):
        self.prior_weight = prior_weight

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ChannelRateBaseline:
        y = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        self.global_rate_ = float(y.mean())
        stats = pd.DataFrame({"channel": X["acquisition_channel"].to_numpy(), "y": y}).groupby(
            "channel")["y"].agg(["sum", "count"])
        self.rates_ = ((stats["sum"] + self.prior_weight * self.global_rate_)
                       / (stats["count"] + self.prior_weight)).to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["acquisition_channel"].map(self.rates_).fillna(self.global_rate_).to_numpy(float)
        return np.column_stack([1 - p, p])


class RecentActivityBaseline(ClassifierMixin, BaseEstimator):
    """Rank-only heuristic: fewer days since the last session = higher priority."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RecentActivityBaseline:
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # A monotone transform into (0, 1]; a ranking score, not a calibrated probability.
        p = 1.0 / (1.0 + X["days_since_last_session"].to_numpy(float))
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
        "channel_rate": ChannelRateBaseline,
        "recent_activity": RecentActivityBaseline,
        "logistic_regression": make_logistic,
        "gradient_boosting": make_gradient_boosting,
    }
    return factories[name]()


def fit_model(name: str, data: pd.DataFrame, target: str = "converted") -> BaseEstimator:
    return make_model(name).fit(data[list(FEATURES)], data[target].to_numpy())


def predict(model: BaseEstimator, data: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(data[list(FEATURES)])[:, 1]


# ---------------------------------------------------------------- explainability
def _source_feature(column: str, categorical: Sequence[str] = CATEGORICAL_FEATURES) -> str:
    """Map a transformed column (e.g. ``acquisition_channel_email``) back to its raw feature."""
    for cat in categorical:
        if column.startswith(cat + "_"):
            return cat
    return column


def logistic_coefficients(model: Pipeline, categorical: Sequence[str] = CATEGORICAL_FEATURES
                          ) -> pd.DataFrame:
    """Coefficients (log-odds) of the fitted logistic model, largest magnitude first.

    Numeric terms are per standard deviation of log1p(feature). One-hot terms are centered within
    their feature (exactly one level is active per lead, so this leaves predictions unchanged)
    and read as "versus the average level" of that attribute.
    """
    names = model.named_steps["pre"].get_feature_names_out()
    out = pd.DataFrame({"term": names, "feature": [_source_feature(n, categorical) for n in names],
                        "coefficient": model.named_steps["model"].coef_[0]})
    is_cat = out["feature"].isin(categorical)
    group_mean = out.groupby("feature")["coefficient"].transform("mean")
    out.loc[is_cat, "coefficient"] -= group_mean[is_cat]
    out["odds_ratio"] = np.exp(out["coefficient"])
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(
        drop=True)


def shap_importance(model: Pipeline, data: pd.DataFrame, background: pd.DataFrame,
                    sample_size: int = 2000, seed: int = SEED,
                    features: Sequence[str] = FEATURES,
                    categorical: Sequence[str] = CATEGORICAL_FEATURES,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact SHAP values in log-odds, summed over each feature's one-hot columns.

    Uses ``LinearExplainer`` for the logistic model and ``TreeExplainer`` for boosted trees, both
    interventional against ``background`` (a training sample), so values are exact and
    additive. Returns (per-feature summary, per-row SHAP matrix for the explained sample).
    ``features``/``categorical`` default to this section's schema; other sections pass theirs.
    """
    import shap

    pre, clf = model.named_steps["pre"], model.named_steps["model"]
    sample = data.sample(n=min(sample_size, len(data)), random_state=seed)
    x = pre.transform(sample[list(features)])
    bg = pre.transform(background.sample(n=min(sample_size, len(background)),
                                         random_state=seed)[list(features)])
    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, shap.maskers.Independent(bg, max_samples=len(bg)))
    else:
        explainer = shap.TreeExplainer(clf)
    values = np.asarray(explainer.shap_values(x))
    names = pre.get_feature_names_out()
    per_column = pd.DataFrame(values, columns=names, index=sample.index)
    per_feature = per_column.T.groupby([_source_feature(n, categorical) for n in names]).sum().T
    # Direction: sign of the correlation between a numeric feature and its SHAP contribution.
    direction = {}
    for f in per_feature.columns:
        if f in categorical:
            direction[f] = "categorical"
        elif sample[f].std() > 0 and per_feature[f].std() > 0:
            corr = np.corrcoef(sample[f].to_numpy(float), per_feature[f].to_numpy())[0, 1]
            direction[f] = "higher -> more likely" if corr > 0 else "higher -> less likely"
        else:
            direction[f] = "no effect"
    summary = pd.DataFrame({
        "feature": per_feature.columns,
        "mean_abs_shap": per_feature.abs().mean().to_numpy(),
        "direction": [direction[f] for f in per_feature.columns],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return summary, per_feature
