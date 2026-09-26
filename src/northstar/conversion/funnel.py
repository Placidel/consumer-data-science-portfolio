"""Ordered conversion funnel built from ``funnel_events``.

Definitions (these are what make the funnel monotone):

- A session's *depth* is the longest contiguous prefix of stages it logged: a session counts
  as reaching stage k only if it also logged stages 1..k-1. Events that skip a stage are
  reported by :func:`funnel_integrity` and never inflate downstream counts.
- The session funnel counts sessions with depth >= k, so every stage is a subset of the one
  before it.
- The prospect funnel counts people whose deepest session within a fixed window after lead
  creation reached stage k. "Ever reached k" implies "ever reached k-1", so it is monotone too.

:func:`assert_monotone` enforces this on every table the pipeline writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd

from northstar.schema import DEVICE_TYPES, FUNNEL_STAGES, PLATFORMS, TRAFFIC_SOURCES

STAGE_LABELS = {"session_start": "Session start", "product_view": "Product view",
                "add_to_cart": "Add to cart", "checkout_start": "Checkout start",
                "purchase": "Purchase"}
STEPS = tuple(f"{a}->{b}" for a, b in pairwise(FUNNEL_STAGES))
VISIT_LEVELS = ("1st visit", "2nd visit", "3rd+ visit")
SEGMENT_LEVELS = {
    "device_type": DEVICE_TYPES,
    "platform": PLATFORMS,
    "traffic_source": TRAFFIC_SOURCES,
    "visit_number": VISIT_LEVELS,
}


class FunnelIntegrityError(ValueError):
    """Raised when a funnel table would show more units downstream than upstream."""


def assert_monotone(counts: Sequence[float], label: str = "funnel") -> None:
    values = np.asarray(counts, dtype=float)
    if np.any(np.diff(values) > 0):
        raise FunnelIntegrityError(f"{label}: counts increase downstream {values.tolist()}")


def _presence(events: pd.DataFrame) -> pd.DataFrame:
    """Session x stage boolean matrix of logged stages (columns 1..5)."""
    codes, sessions = pd.factorize(events["session_id"], sort=True)
    stage = events["stage_number"].to_numpy(dtype=int)
    n_stages = len(FUNNEL_STAGES)
    matrix = np.zeros((len(sessions), n_stages), dtype=bool)
    valid = (stage >= 1) & (stage <= n_stages)
    matrix[codes[valid], stage[valid] - 1] = True
    return pd.DataFrame(matrix, index=pd.Index(sessions, name="session_id"),
                        columns=range(1, n_stages + 1))


def session_depth(events: pd.DataFrame) -> pd.Series:
    """Deepest *contiguous* stage per session (0 if the session_start event is missing)."""
    present = _presence(events)
    depth = np.cumprod(present.to_numpy(), axis=1).sum(axis=1)
    return pd.Series(depth, index=present.index, name="depth")


def funnel_integrity(sessions: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Counts of event-log defects the funnel definition has to guard against."""
    present = _presence(events)
    contiguous = np.cumprod(present.to_numpy(), axis=1).sum(axis=1)
    deepest = np.where(present.to_numpy(), np.arange(1, present.shape[1] + 1), 0).max(axis=1)
    ordered = events.sort_values(["session_id", "stage_number", "event_ts"])
    ts_back = ordered.groupby("session_id")["event_ts"].diff() < pd.Timedelta(0)
    return {
        "sessions": len(sessions),
        "events": len(events),
        "orphan_events": int((~events["session_id"].isin(sessions["session_id"])).sum()),
        "sessions_without_events": int((~sessions["session_id"].isin(present.index)).sum()),
        "sessions_missing_session_start": int((contiguous == 0).sum()),
        "sessions_with_skipped_stages": int((deepest > contiguous).sum()),
        "duplicate_stage_events": int(events.duplicated(["session_id", "stage_number"]).sum()),
        "out_of_order_events": int(ts_back.sum()),
    }


def session_frame(tables: Mapping[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp
                  ) -> pd.DataFrame:
    """Sessions starting in ``[start, end)`` with depth, visitor type and prospect visit number.

    The visit number counts the person's *prospect* sessions over the whole history up to and
    including this one, so it only uses information available when the session starts.
    """
    sessions = tables["sessions"]
    depth = session_depth(tables["funnel_events"])
    frame = sessions.assign(depth=sessions["session_id"].map(depth).fillna(0).astype(int))
    frame["visitor_type"] = np.where(frame["customer_id"].isna(), "prospect", "customer")
    prospect = frame.loc[frame["visitor_type"] == "prospect"].sort_values(
        ["prospect_id", "session_start", "session_id"])
    ordinal = prospect.groupby("prospect_id").cumcount() + 1
    labels = pd.Series(np.select([ordinal == 1, ordinal == 2], VISIT_LEVELS[:2], VISIT_LEVELS[2]),
                       index=prospect.index)
    frame["visit_number"] = labels.reindex(frame.index)
    in_period = (frame["session_start"] >= start) & (frame["session_start"] < end)
    return frame.loc[in_period & (frame["depth"] >= 1)].reset_index(drop=True)


def stage_table(depth: pd.Series | np.ndarray, label: str = "funnel") -> pd.DataFrame:
    """Units reaching each stage, step conversion and where the losses happen."""
    d = np.asarray(depth)
    reached = np.array([(d >= k).sum() for k in range(1, len(FUNNEL_STAGES) + 1)])
    assert_monotone(reached, label)
    prev = np.concatenate([[reached[0]], reached[:-1]])
    lost = prev - reached
    total_lost = lost.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame({
            "stage_number": np.arange(1, len(FUNNEL_STAGES) + 1),
            "stage": FUNNEL_STAGES,
            "reached": reached,
            "share_of_start": reached / reached[0] if reached[0] else np.nan,
            "step_conversion": np.where(prev > 0, reached / prev, np.nan),
            "lost_at_step": lost,
            "share_of_all_losses": lost / total_lost if total_lost else 0.0,
        })


def segment_funnel(frame: pd.DataFrame, dimensions: Mapping[str, Sequence[str]] | None = None
                   ) -> pd.DataFrame:
    """Stage counts and step conversion for every level of each segment dimension (long form)."""
    dimensions = dimensions or SEGMENT_LEVELS
    rows = []
    for dim, levels in dimensions.items():
        for level in levels:
            sub = frame.loc[frame[dim] == level, "depth"]
            if sub.empty:
                continue
            table = stage_table(sub, f"{dim}={level}")
            row = {"dimension": dim, "level": level, "sessions": len(sub)}
            for _, r in table.iterrows():
                row[f"reached_{r['stage']}"] = int(r["reached"])
            for step, rate in zip(STEPS, table["step_conversion"].iloc[1:], strict=True):
                row[f"rate_{step}"] = rate
            row["session_to_purchase"] = table["share_of_start"].iloc[-1]
            rows.append(row)
    return pd.DataFrame(rows)


def dropoff_opportunity(segments: pd.DataFrame, dimension: str, reference: str) -> pd.DataFrame:
    """Descriptive sizing: extra step completions (and the purchases they would carry through at
    the segment's own downstream rates) if each level converted at the reference level's rate.
    A gap-to-benchmark, not a causal estimate."""
    seg = segments.loc[segments["dimension"] == dimension].set_index("level")
    ref = seg.loc[reference]
    rows = []
    for level, r in seg.drop(index=reference).iterrows():
        for i, step in enumerate(STEPS):
            entering = r[f"reached_{FUNNEL_STAGES[i]}"]
            gap = ref[f"rate_{step}"] - r[f"rate_{step}"]
            downstream = np.prod([r[f"rate_{s}"] for s in STEPS[i + 1:]]) if i + 1 < len(STEPS) \
                else 1.0
            rows.append({"dimension": dimension, "level": level, "reference": reference,
                         "step": step, "entering": int(entering),
                         "level_rate": r[f"rate_{step}"], "reference_rate": ref[f"rate_{step}"],
                         "rate_gap": gap, "extra_step_completions": entering * gap,
                         "purchase_equivalents": entering * gap * downstream})
    return pd.DataFrame(rows)


def prospect_funnel(tables: Mapping[str, pd.DataFrame], cohort_start: pd.Timestamp,
                    cohort_end: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """Per-prospect deepest stage within ``window_days`` of lead creation, for leads created in
    ``[cohort_start, cohort_end)``. Every lead in the cohort has a complete window as long as
    ``cohort_end + window_days`` is within the data."""
    prospects = tables["prospects"]
    cohort = prospects.loc[(prospects["created_at"] >= cohort_start)
                           & (prospects["created_at"] < cohort_end),
                           ["prospect_id", "created_at", "device_type", "acquisition_channel"]]
    depth = session_depth(tables["funnel_events"])
    s = tables["sessions"][["session_id", "prospect_id", "session_start"]].merge(
        cohort[["prospect_id", "created_at"]], on="prospect_id")
    s = s.loc[(s["session_start"] >= s["created_at"])
              & (s["session_start"] < s["created_at"] + pd.Timedelta(days=window_days))]
    deepest = s.assign(depth=s["session_id"].map(depth).fillna(0)).groupby("prospect_id")[
        "depth"].max()
    cohort = cohort.assign(depth=cohort["prospect_id"].map(deepest).fillna(0).astype(int))
    return cohort.reset_index(drop=True)
