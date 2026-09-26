"""Monitoring a deployed model: input drift, score drift and (once labels mature) performance.

Reference vs current. The reference is the model's own training population, stored with the
artifact (:mod:`northstar.serving.profiles`). The current batches are the monthly scoring runs
from the model's ``deployable_from`` date to the end of the data, scored through the batch
scorer exactly as production would. Both sides are fixed by the data and the split, so the
report is reproducible.

Labels arrive late: a churn score made on 1 November can only be judged after 90 days. Runs
whose outcome window has not closed by the end of the data are monitored for drift only and
listed as pending, which is how the report would look in a live system.

Checks and their severities (thresholds live in :class:`MonitoringConfig` and are reported):

* feature PSI >= moderate -> ``watch``; >= major -> ``investigate`` (inputs moved; performance may
  or may not follow);
* score PSI, same thresholds (the ranked population moved, so capacity plans based on score
  cut-offs may need re-deriving);
* ROC AUC of a matured run more than ``auc_drop`` below the validation estimate, with the run's
  95% interval (Hanley-McNeil) entirely below that estimate -> ``retrain``;
* observed / predicted outcome rate of a matured run outside the calibration band, with observed
  vs expected positives significant at 5% (binomial z) -> ``recalibrate`` (probabilities feed
  the section 02 expected-value rule, so calibration matters).

Performance checks need both conditions, material and beyond sampling noise, so a small month
does not trigger a retrain by chance. PSI has no such test; its thresholds are applied to the
pooled current population, and the worst single run is reported for context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from northstar.serving.batch import score_frame
from northstar.serving.profiles import feature_drift, score_drift
from northstar.serving.scoring import Scorer
from northstar.timeline import DATA_END

Z95 = 1.959964
SEVERITY = ("ok", "watch", "investigate", "recalibrate", "retrain")
LAST_RUN = pd.Timestamp("2025-12-01")  # last monthly scoring run inside the data
RECOMMENDATIONS = {
    "ok": "No action: inputs, scores and matured outcomes are in line with the reference.",
    "watch": "Keep scoring; review the moved inputs at the next monthly check.",
    "investigate": ("Keep scoring, but confirm with the data owner whether the shift is a real "
                    "change in the population or an upstream pipeline change before the next "
                    "run."),
    "recalibrate": ("Rankings still hold, but probabilities are off: refit the calibration (or "
                    "retrain) before probabilities are used in value calculations."),
    "retrain": "Ranking quality dropped below tolerance: retrain on recent runs and re-validate.",
}


@dataclass(frozen=True)
class MonitoringConfig:
    psi_moderate: float = 0.10
    psi_major: float = 0.25
    auc_drop: float = 0.05
    calibration_low: float = 0.80
    calibration_high: float = 1.25
    last_run: pd.Timestamp = LAST_RUN

    def as_dict(self) -> dict:
        return {k: (str(v.date()) if isinstance(v, pd.Timestamp) else v)
                for k, v in asdict(self).items()}


def current_runs(scorer: Scorer, config: MonitoringConfig) -> tuple[pd.Timestamp, ...]:
    first = pd.Timestamp(scorer.artifact.metadata["training"]["deployable_from"])
    return tuple(pd.date_range(first, config.last_run, freq="MS"))


def labels_mature(cutoff: pd.Timestamp, horizon_days: int) -> bool:
    return cutoff + pd.Timedelta(days=horizon_days) <= DATA_END


def _psi_status(value: float, config: MonitoringConfig) -> str:
    if value >= config.psi_major:
        return "investigate"
    return "watch" if value >= config.psi_moderate else "ok"


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of a ROC AUC."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (
        n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _performance(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    n_pos = int(y.sum())
    auc = float(roc_auc_score(y, p))
    se = auc_standard_error(auc, n_pos, len(y) - n_pos)
    expected = float(p.sum())
    return {"rows": len(y), "observed_rate": float(y.mean()), "mean_predicted": float(p.mean()),
            "observed_to_predicted": n_pos / expected,
            # Calibration-in-the-large: observed vs expected positives, binomial z-score.
            "calibration_z": (n_pos - expected) / float(np.sqrt(np.sum(p * (1 - p)))),
            "roc_auc": auc, "roc_auc_ci_low": auc - Z95 * se, "roc_auc_ci_high": auc + Z95 * se,
            "average_precision": float(average_precision_score(y, p))}


def monitor_model(scorer: Scorer, tables: Mapping[str, pd.DataFrame],
                  config: MonitoringConfig | None = None,
                  runs: Sequence[pd.Timestamp] | None = None
                  ) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Score every current run, then compare with the reference; returns (report, tables)."""
    config = config or MonitoringConfig()
    spec, artifact = scorer.spec, scorer.artifact
    profile = artifact.profile
    runs = tuple(runs) if runs is not None else current_runs(scorer, config)

    batches, labels, run_rows = [], {}, []
    for cutoff in runs:
        mature = labels_mature(cutoff, spec.horizon_days)
        frame = spec.build_run(tables, cutoff) if mature else spec.build_features(tables, cutoff)
        # Strict: any invalid record stops the report. The label is kept out of the scorer's
        # input (which refuses outcome columns) and only joined back for performance checks.
        result = score_frame(scorer, frame.drop(columns=spec.target, errors="ignore"),
                             cutoff=cutoff)
        result.scores["run_cutoff"] = str(cutoff.date())
        batches.append((cutoff, result))
        if mature:
            labels[cutoff] = frame.loc[:, spec.target].to_numpy()
        run_rows.append({"run": str(cutoff.date()), "records": len(frame), "labels_mature": mature,
                         "labels_complete_on": str((cutoff + pd.Timedelta(
                             days=spec.horizon_days)).date())})

    current = pd.concat([r.features for _, r in batches], ignore_index=True)
    scores = np.concatenate([r.scores[spec.score_field].to_numpy() for _, r in batches])

    # Feature drift: pooled over all current runs, plus the worst single run.
    pooled = feature_drift(profile, current)
    by_run = pd.concat([feature_drift(profile, r.features).assign(run=str(c.date()))
                        for c, r in batches], ignore_index=True)
    worst = by_run.loc[by_run.groupby("feature")["psi"].idxmax(), ["feature", "psi", "run"]]
    pooled = pooled.merge(worst.rename(columns={"psi": "max_run_psi", "run": "max_run"}),
                          on="feature")
    pooled["status"] = [_psi_status(v, config) for v in pooled["psi"]]
    pooled = pooled.sort_values(["psi", "feature"], ascending=[False, True]).reset_index(drop=True)

    score_pooled = score_drift(profile, scores)
    score_by_run = [{"run": str(c.date()), **score_drift(
        profile, r.scores[spec.score_field].to_numpy())} for c, r in batches]
    score_pooled["max_run_psi"] = max(r["psi"] for r in score_by_run)
    score_pooled["status"] = _psi_status(score_pooled["psi"], config)

    # Performance on matured runs, against the champion's validation-period estimate.
    reference = artifact.metadata["validation_metrics"]
    perf_rows = []
    for cutoff, result in batches:
        if cutoff in labels:
            perf_rows.append({"run": str(cutoff.date()), **_performance(
                labels[cutoff], result.scores[spec.score_field].to_numpy())})
    matured = [c for c, _ in batches if c in labels]
    pooled_perf = (_performance(np.concatenate([labels[c] for c in matured]), np.concatenate(
        [r.scores[spec.score_field].to_numpy() for c, r in batches if c in labels]))
        if matured else None)

    findings = []
    for row in pooled.itertuples():
        if row.status != "ok":
            findings.append({"check": "feature_drift", "severity": row.status,
                             "subject": row.feature,
                             "message": f"PSI {row.psi:.3f} vs reference (worst run "
                                        f"{row.max_run}: {row.max_run_psi:.3f})"})
    if score_pooled["status"] != "ok":
        findings.append({"check": "score_drift", "severity": score_pooled["status"],
                         "subject": spec.score_field,
                         "message": f"PSI {score_pooled['psi']:.3f}; mean score "
                                    f"{score_pooled['reference_mean']:.3f} -> "
                                    f"{score_pooled['current_mean']:.3f}"})
    # A run is flagged only when the change is material *and* larger than its sampling noise,
    # so a small month does not trigger a retrain on chance alone.
    for row in perf_rows:
        if (row["roc_auc"] < reference["roc_auc"] - config.auc_drop
                and row["roc_auc_ci_high"] < reference["roc_auc"]):
            findings.append({"check": "discrimination", "severity": "retrain",
                             "subject": row["run"],
                             "message": f"ROC AUC {row['roc_auc']:.3f} (95% CI up to "
                                        f"{row['roc_auc_ci_high']:.3f}) vs validation "
                                        f"{reference['roc_auc']:.3f}"})
        if (not config.calibration_low <= row["observed_to_predicted"] <= config.calibration_high
                and abs(row["calibration_z"]) > Z95):
            findings.append({"check": "calibration", "severity": "recalibrate",
                             "subject": row["run"],
                             "message": f"observed {row['observed_rate']:.3f} vs predicted "
                                        f"{row['mean_predicted']:.3f} (ratio "
                                        f"{row['observed_to_predicted']:.2f}, z = "
                                        f"{row['calibration_z']:.1f})"})
    status = max((f["severity"] for f in findings), key=SEVERITY.index, default="ok")

    report = {
        "model": {"name": spec.name, "version": artifact.version,
                  "algorithm": artifact.metadata["algorithm"],
                  "deployable_from": artifact.metadata["training"]["deployable_from"]},
        "reference": {"description": profile["description"], "rows": profile["rows"]},
        "current": {"first_run": str(runs[0].date()), "last_run": str(runs[-1].date()),
                    "runs": run_rows, "records": len(current)},
        "score_drift": {**score_pooled, "by_run": score_by_run},
        "feature_drift": pooled.to_dict(orient="records"),
        "performance": {
            "reference": {"source": "champion on validation runs (fit runs only)",
                          **{k: reference[k] for k in ("roc_auc", "average_precision",
                                                       "base_rate", "mean_predicted")}},
            "by_run": perf_rows,
            "pooled": pooled_perf,
            "pending_runs": [r["run"] for r in run_rows if not r["labels_mature"]],
        },
        "findings": sorted(findings, key=lambda f: (-SEVERITY.index(f["severity"]), f["check"],
                                                    f["subject"])),
        "status": status,
        "recommendation": RECOMMENDATIONS[status],
    }
    frames = {
        "drift_features": pooled.assign(model=spec.name),
        "drift_by_run": by_run[["run", "feature", "kind", "psi", "reference_mean",
                                "current_mean"]].assign(model=spec.name),
        "performance_by_run": pd.DataFrame(perf_rows).assign(model=spec.name),
    }
    return report, frames
