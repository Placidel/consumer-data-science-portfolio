"""The two production models and everything the serving layer needs to know about them.

Each spec binds a served model to the section that designed it: the population it scores, the
point-in-time feature builder, the out-of-time split, the candidate models and the selection rule.
Training, the API, batch scoring and monitoring all read the same spec, so they cannot disagree on
feature order, horizon or split.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from northstar.acquisition import dataset as acq_data
from northstar.acquisition import evaluation as acq_ev
from northstar.acquisition import models as acq_models
from northstar.acquisition.dataset import SplitPlan
from northstar.retention import dataset as ret_data
from northstar.retention import evaluation as ret_ev
from northstar.retention import models as ret_models
from northstar.schema import AGE_BANDS, CHANNELS, DEVICE_TYPES, INCOME_BANDS, REGIONS

# Allowed levels of the categorical attributes both models use (the shared schema's domains).
CATEGORY_LEVELS: dict[str, tuple[str, ...]] = {
    "acquisition_channel": CHANNELS,
    "region": REGIONS,
    "age_band": AGE_BANDS,
    "income_band": INCOME_BANDS,
    "device_type": DEVICE_TYPES,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    section: str
    route: str  # API path segment: /v1/<route>/score
    entity: str
    target: str
    score_field: str
    horizon_days: int
    population: str
    intended_use: str
    not_for: str
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    tables_used: tuple[str, ...]
    split: SplitPlan
    candidates: tuple[str, ...]
    model_labels: Mapping[str, str]
    selection_metric: str
    maximize: bool
    # Point-in-time columns the feature builder emits besides the id and features (e.g. margin
    # for section 02's targeting). Batch files may carry them; the model never reads them.
    context_columns: tuple[str, ...]
    build_features: Callable[[Mapping[str, pd.DataFrame], pd.Timestamp], pd.DataFrame]
    build_run: Callable[[Mapping[str, pd.DataFrame], pd.Timestamp], pd.DataFrame]
    fit_model: Callable[[str, pd.DataFrame], object]
    leakage_audit: Callable[[Mapping[str, pd.DataFrame], pd.DataFrame, SplitPlan], dict]
    score_metrics: Callable[[pd.DataFrame, str], dict]

    @property
    def features(self) -> tuple[str, ...]:
        return self.categorical + self.numeric

    @property
    def section_command(self) -> str:
        return {"01_acquisition": "northstar acquisition",
                "02_retention": "northstar retention"}[self.section]

    def select_champion(self, validation: Mapping[str, Mapping[str, float]]) -> str:
        """The section's selection rule, applied to validation-run metrics of each candidate."""
        def key(name: str) -> float:
            return validation[name][self.selection_metric]

        return max(self.candidates, key=key) if self.maximize else min(self.candidates, key=key)


ACQUISITION = ModelSpec(
    name="acquisition_lead_score",
    title="Lead conversion score",
    section="01_acquisition",
    route="acquisition",
    entity="prospect_id",
    target=acq_data.TARGET,
    score_field="conversion_probability",
    horizon_days=acq_data.HORIZON_DAYS,
    population=(f"open leads: created in the {acq_data.PIPELINE_DAYS} days before the scoring "
                "date and without a first order yet"),
    intended_use=("Rank the open pipeline each month so outreach capacity goes to the leads most "
                  f"likely to place a first order in the next {acq_data.HORIZON_DAYS} days."),
    not_for=("Existing customers, leads older than the pipeline window, pricing or credit "
             "decisions, or estimating the causal effect of outreach."),
    categorical=acq_data.CATEGORICAL_FEATURES,
    numeric=acq_data.NUMERIC_FEATURES,
    tables_used=acq_data.TABLES_USED,
    split=acq_data.DEFAULT_SPLIT,
    # The API returns probabilities, so the rank-only recency heuristic is not a candidate.
    candidates=acq_models.PROBABILISTIC,
    model_labels=acq_models.MODEL_LABELS,
    selection_metric="average_precision",  # section 01's rule
    maximize=True,
    context_columns=(),
    build_features=acq_data.build_features,
    build_run=acq_data.build_run,
    fit_model=acq_models.fit_model,
    leakage_audit=acq_data.leakage_audit,
    score_metrics=lambda data, score: acq_ev.score_metrics(data, score, True),
)

CHURN = ModelSpec(
    name="churn_risk",
    title="90-day churn risk",
    section="02_retention",
    route="churn",
    entity="customer_id",
    target=ret_data.TARGET,
    score_field="churn_probability",
    horizon_days=ret_data.HORIZON_DAYS,
    population=(f"active customers: at least one order in the {ret_data.ACTIVE_DAYS} days before "
                "the scoring date"),
    intended_use=("Prioritize active customers for a retention offer, combined with customer "
                  "margin as in section 02's expected-value targeting."),
    not_for=("Lapsed customers (win-back), prospects, individual pricing or service-level "
             "decisions, or judging whether an offer caused a customer to stay."),
    categorical=ret_data.CATEGORICAL_FEATURES,
    numeric=ret_data.NUMERIC_FEATURES,
    tables_used=ret_data.TABLES_USED,
    split=ret_data.DEFAULT_SPLIT,
    candidates=ret_models.PROBABILISTIC,
    model_labels=ret_models.MODEL_LABELS,
    selection_metric="log_loss",  # section 02's rule: the simulation needs calibrated risk
    maximize=False,
    context_columns=("margin_180d",),
    build_features=ret_data.build_features,
    build_run=ret_data.build_run,
    fit_model=ret_models.fit_model,
    leakage_audit=ret_data.leakage_audit,
    score_metrics=lambda data, score: ret_ev.score_metrics(data, score, True),
)

SPECS: dict[str, ModelSpec] = {spec.name: spec for spec in (ACQUISITION, CHURN)}


def get_spec(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(SPECS)}") from None
