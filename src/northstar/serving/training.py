"""Train the production models once, offline, and register them as versioned artifacts.

Training follows the section designs exactly: the section's point-in-time dataset and out-of-time
split, the section's leakage audit as a hard gate, candidates fit on the *fit* runs and compared
on the *validation* runs with the section's selection metric, and the champion refit on all
training runs. The holdout runs are scored once to record out-of-time performance in the model
card; they are never used to fit or select. The API and batch scorer only load the result.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from northstar import __version__
from northstar.acquisition.report import LeakageError
from northstar.io import content_hash
from northstar.serving.profiles import build_profile
from northstar.serving.registry import SCHEMA_VERSION, save_artifact
from northstar.serving.specs import CATEGORY_LEVELS, ModelSpec

__all__ = ["LeakageError", "artifact_version", "predict", "train", "train_and_register"]

REPORTED_METRICS = ("rows", "base_rate", "roc_auc", "average_precision", "log_loss", "brier",
                    "ece", "calibration_slope", "mean_predicted")


def predict(spec: ModelSpec, model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[list(spec.features)])[:, 1]


def _dates(runs) -> list[str]:
    return [str(pd.Timestamp(c).date()) for c in runs]


def _metrics(metrics: Mapping) -> dict:
    return {k: metrics[k] for k in REPORTED_METRICS if k in metrics}


def artifact_version(spec: ModelSpec, tables: Mapping[str, pd.DataFrame], champion: str
                     ) -> tuple[str, str]:
    """Content-addressed version: same data, design, algorithm and libraries -> same version."""
    fingerprint = hashlib.sha256(json.dumps({
        "model": spec.name, "algorithm": champion, "features": list(spec.features),
        "split": spec.split.as_dict(),
        "tables": {name: content_hash(tables[name]) for name in spec.tables_used},
        "scikit_learn": sklearn.__version__, "northstar": __version__,
    }, sort_keys=True).encode()).hexdigest()
    return f"v{min(spec.split.holdout):%Y%m%d}-{fingerprint[:10]}", fingerprint


def train(spec: ModelSpec, tables: Mapping[str, pd.DataFrame],
          data_info: Mapping | None = None) -> tuple[object, dict, dict]:
    """Fit the champion for ``spec``; returns (model, metadata, reference profile)."""
    plan = spec.split
    data = pd.concat([spec.build_run(tables, c) for c in plan.fit + plan.validation + plan.holdout],
                     ignore_index=True)
    data["split"] = data["run_cutoff"].map(plan.role)
    audit = spec.leakage_audit(tables, data, plan)
    if not audit["passed"]:
        failed = [k for k, ok in audit["checks"].items() if not ok]
        raise LeakageError(f"{spec.name}: leakage audit failed {failed}; nothing registered")

    fit = data.loc[data["split"] == "fit"]
    val = data.loc[data["split"] == "validation"]
    train_rows = data.loc[data["split"].isin(["fit", "validation"])]
    hold = data.loc[data["split"] == "holdout"]

    validation = {}
    for name in spec.candidates:
        scores = predict(spec, spec.fit_model(name, fit), val)
        validation[name] = spec.score_metrics(val.assign(_score=scores), "_score")
    champion = spec.select_champion(validation)

    model = spec.fit_model(champion, train_rows)
    train_scores = predict(spec, model, train_rows)
    holdout = spec.score_metrics(hold.assign(_score=predict(spec, model, hold)), "_score")

    deployable_from = min(plan.holdout)
    version, fingerprint = artifact_version(spec, tables, champion)
    profile = build_profile(
        train_rows, train_scores, spec.numeric,
        {f: CATEGORY_LEVELS[f] for f in spec.categorical},
        description=(f"training runs {_dates(plan.train)[0]} to {_dates(plan.train)[-1]} "
                     f"({len(train_rows):,} {spec.entity} x run rows); in-sample scores"))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "name": spec.name,
        "version": version,
        "title": spec.title,
        "section": spec.section,
        "algorithm": champion,
        "algorithm_label": spec.model_labels[champion],
        "entity": spec.entity,
        "target": spec.target,
        "score_field": spec.score_field,
        "horizon_days": spec.horizon_days,
        "population": spec.population,
        "intended_use": spec.intended_use,
        "not_for": spec.not_for,
        "features": list(spec.features),
        "categorical_features": list(spec.categorical),
        "numeric_features": list(spec.numeric),
        "training": {
            **plan.as_dict(),
            "deployable_from": str(deployable_from.date()),
            "train_rows": len(train_rows),
            "train_positives": int(train_rows[spec.target].sum()),
            "fingerprint": fingerprint,
        },
        "selection": {
            "metric": spec.selection_metric,
            "rule": "highest" if spec.maximize else "lowest",
            "candidates": {name: _metrics(m) for name, m in validation.items()},
            "champion": champion,
        },
        "validation_metrics": _metrics(validation[champion]),
        "holdout_metrics": _metrics(holdout),
        "leakage_audit": {"passed": audit["passed"], "checks": audit["checks"]},
        "data": dict(data_info or {}),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "northstar": __version__},
    }
    return model, metadata, profile


def train_and_register(spec: ModelSpec, tables: Mapping[str, pd.DataFrame],
                       root: Path | None = None, data_info: Mapping | None = None,
                       set_latest: bool = True) -> tuple[Path, dict]:
    model, metadata, profile = train(spec, tables, data_info)
    path = save_artifact(model, metadata, profile, root, set_latest=set_latest)
    return path, metadata
