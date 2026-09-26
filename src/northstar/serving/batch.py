"""Batch scoring: a whole scoring run (or a feature file) through the same path as the API.

Two inputs are supported:

* **A scoring date.** Features are built from the shared tables with the section's point-in-time
  builder, which is how a monthly CRM or sales list would be produced.
* **A feature file** (CSV or Parquet) produced upstream, one row per lead or customer.

Before any row is looked at, the file as a whole must be in scope:

* its columns are the id, the features, ``run_cutoff`` and the builder's context columns only.
  Anything else, such as an outcome column (``converted``/``churned``) or a misspelt feature, is
  refused, exactly as the API refuses unknown fields;
* every row carries a scoring date (``run_cutoff``, or one date given for the whole file), and
  every date is on or after the model's first usable date, so a file cannot score the training
  period with labels the model has already seen.

Every row is then validated with the API's record schema. By default any invalid row aborts the
batch (a partial list silently drops people); ``max_invalid_share`` lets an operator accept a
small share, in which case the rejected rows and their reasons are returned for a quarantine
file. Scores are ranked within each scoring run (1 = highest) so the output is directly a call
list.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from northstar.serving.scoring import RejectedRecord, Scorer, rejection_summary, validate_frame
from northstar.serving.specs import ModelSpec
from northstar.timeline import DATA_END


class BatchValidationError(ValueError):
    """The batch has more invalid records than allowed, or is out of the model's scope."""


@dataclass(frozen=True)
class BatchResult:
    scores: pd.DataFrame
    rejected: list[RejectedRecord]
    features: pd.DataFrame  # the validated feature rows that were scored (same order as scores)


def check_scoring_date(scorer: Scorer, cutoff: pd.Timestamp) -> None:
    """Refuse dates the model cannot score honestly (in-sample, or outside the data)."""
    deployable = pd.Timestamp(scorer.artifact.metadata["training"]["deployable_from"])
    if cutoff < deployable:
        raise BatchValidationError(
            f"{scorer.spec.name} {scorer.artifact.version} was trained on outcomes observed up to "
            f"{deployable.date()}; scoring {cutoff.date()} would reuse training labels. Use a "
            "date on or after it, or a model trained on an earlier split.")
    if cutoff >= DATA_END:
        raise BatchValidationError(f"{cutoff.date()} is after the end of the data "
                                   f"({DATA_END.date()}).")


def check_columns(spec: ModelSpec, frame: pd.DataFrame) -> None:
    """Refuse columns that are not part of a scoring record (outcomes, typos, anything else)."""
    allowed = {spec.entity, *spec.features, "run_cutoff", *spec.context_columns}
    unknown = [c for c in frame.columns if c not in allowed]
    if unknown:
        hint = (f" {spec.target!r} is the outcome the model predicts; a scoring file must not "
                "contain it." if spec.target in unknown else "")
        raise BatchValidationError(
            f"unexpected column(s) for {spec.name}: {unknown}.{hint} Allowed: the id "
            f"{spec.entity!r}, the model features, 'run_cutoff'"
            + (f" and {list(spec.context_columns)}" if spec.context_columns else "") + ".")


def scoring_dates(scorer: Scorer, frame: pd.DataFrame, cutoff: str | pd.Timestamp | None = None
                  ) -> pd.Series:
    """Each row's scoring date, checked against the model's scope; returns normalized dates.

    ``cutoff`` supplies the date for a file without a ``run_cutoff`` column (and must agree with
    one if the file has it). A file with neither is refused: without a date nothing stops it
    from re-scoring the training period.
    """
    if "run_cutoff" in frame.columns:
        dates = pd.to_datetime(frame["run_cutoff"], errors="coerce", format="ISO8601")
        if dates.isna().any():
            bad = frame.loc[dates.isna(), "run_cutoff"].head(3).tolist()
            raise BatchValidationError(f"run_cutoff must be a date on every row; got {bad}")
        if cutoff is not None and (dates != pd.Timestamp(cutoff)).any():
            raise BatchValidationError(f"the file's run_cutoff values disagree with the "
                                       f"scoring date {pd.Timestamp(cutoff).date()}")
    elif cutoff is not None:
        dates = pd.Series(pd.Timestamp(cutoff), index=frame.index)
    else:
        raise BatchValidationError(
            "the batch has no scoring date: add a run_cutoff column (the feature builders emit "
            "one) or pass the date the features were built for (--cutoff). The date is needed "
            "to refuse scoring periods the model was trained on.")
    dates = dates.dt.normalize()
    for date in dates.unique():
        check_scoring_date(scorer, pd.Timestamp(date))
    return dates


def score_frame(scorer: Scorer, frame: pd.DataFrame, max_invalid_share: float = 0.0,
                cutoff: str | pd.Timestamp | None = None) -> BatchResult:
    spec = scorer.spec
    if frame.empty:
        raise BatchValidationError("the batch contains no records")
    check_columns(spec, frame)
    dates = scoring_dates(scorer, frame, cutoff)
    valid, rejected = validate_frame(spec, frame)
    if len(rejected) > max_invalid_share * len(frame):
        raise BatchValidationError(
            f"{len(rejected)} of {len(frame)} records failed validation "
            f"(allowed share {max_invalid_share:.1%}):\n{rejection_summary(rejected)}")
    scores = scorer.score(valid)
    scores.insert(0, "run_cutoff", dates.loc[valid.index].dt.date.astype(str))
    # Deterministic rank within each run: highest score first, ties broken by id.
    order = scores.sort_values(["run_cutoff", spec.score_field, spec.entity],
                               ascending=[True, False, True])
    scores["rank"] = order.groupby("run_cutoff").cumcount().add(1).reindex(scores.index)
    scores["model_name"] = spec.name
    scores["model_version"] = scorer.artifact.version
    return BatchResult(scores.reset_index(drop=True), rejected, valid.reset_index(drop=True))


def score_run(scorer: Scorer, tables: Mapping[str, pd.DataFrame], cutoff: str | pd.Timestamp
              ) -> BatchResult:
    """Build point-in-time features for the scoring date and score them."""
    cutoff = pd.Timestamp(cutoff)
    check_scoring_date(scorer, cutoff)
    return score_frame(scorer, scorer.spec.build_features(tables, cutoff), cutoff=cutoff)


def read_features(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"prospect_id": str, "customer_id": str})
    raise ValueError(f"Unsupported feature file {path.name}: use .csv or .parquet")


def write_result(result: BatchResult, out: Path) -> list[Path]:
    """Write scores (and a quarantine file of rejected rows, if any); returns written paths."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.scores.to_csv(out, index=False, float_format="%.6f")
    written = [out]
    if result.rejected:
        rejected = out.with_name(out.stem + ".rejected.csv")
        pd.DataFrame([{"row": r.row, "id": r.entity_id, "errors": " | ".join(r.errors)}
                      for r in result.rejected]).to_csv(rejected, index=False)
        written.append(rejected)
    return written
