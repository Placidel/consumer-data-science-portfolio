"""Scoring shared by the API and the batch scorer: one code path from validated records to scores.

A :class:`Scorer` wraps a loaded artifact. It never fits anything; the estimator is loaded once
and reused for every call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from northstar.serving.profiles import reference_percentile
from northstar.serving.registry import ModelArtifact
from northstar.serving.schemas import RECORD_MODELS, ModelInfo, records_frame
from northstar.serving.specs import ModelSpec


@dataclass(frozen=True)
class Scorer:
    artifact: ModelArtifact

    @property
    def spec(self):
        return self.artifact.spec

    @property
    def info(self) -> ModelInfo:
        md = self.artifact.metadata
        return ModelInfo(name=md["name"], version=md["version"], algorithm=md["algorithm"],
                         horizon_days=md["horizon_days"],
                         deployable_from=md["training"]["deployable_from"])

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` holds the entity id and every feature; returns id, probability, percentile."""
        spec = self.spec
        proba = self.artifact.model.predict_proba(frame[list(spec.features)])[:, 1]
        return pd.DataFrame({
            spec.entity: frame[spec.entity].to_numpy(),
            spec.score_field: proba,
            "reference_percentile": reference_percentile(
                proba, self.artifact.profile["score"]["quantiles"]),
        }, index=frame.index)


@dataclass(frozen=True)
class RejectedRecord:
    row: int  # 0-based position in the input
    entity_id: str | None
    errors: tuple[str, ...]


def validate_frame(spec: ModelSpec, frame: pd.DataFrame
                   ) -> tuple[pd.DataFrame, list[RejectedRecord]]:
    """Validate every row with the API's record schema; returns (valid rows, rejections).

    Only the record's fields (id and features) are validated; missing ones are reported per row.
    Other columns are not looked at here: which extra columns a file may carry is a file-level
    decision made by the caller (see ``batch.check_columns``). Valid rows keep their index.
    """
    model = RECORD_MODELS[spec.name]
    fields = list(model.model_fields)
    present = [c for c in fields if c in frame.columns]
    records = frame[present].to_dict(orient="records")
    adapter = TypeAdapter(model)
    valid, index, rejected = [], [], []
    entity = spec.entity
    for pos, (idx, row) in enumerate(zip(frame.index, records, strict=True)):
        try:
            valid.append(adapter.validate_python(row))
            index.append(idx)
        except ValidationError as exc:
            messages = tuple(f"{'.'.join(map(str, e['loc'])) or 'record'}: {e['msg']}"
                             for e in exc.errors())
            rejected.append(RejectedRecord(pos, row.get(entity), messages))
    out = records_frame(valid) if valid else pd.DataFrame(columns=fields)
    out.index = pd.Index(index)
    return out, rejected


def rejection_summary(rejected: Sequence[RejectedRecord], limit: int = 5) -> str:
    lines = [f"row {r.row} ({r.entity_id}): {'; '.join(r.errors)}" for r in rejected[:limit]]
    more = len(rejected) - limit
    return "\n".join(lines + ([f"... and {more} more"] if more > 0 else []))
