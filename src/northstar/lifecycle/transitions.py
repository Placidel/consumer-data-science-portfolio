"""Month-to-month lifecycle transitions and the descriptive evidence behind decision points.

A **transition** is the pair (state at the end of month ``t``, state at the end of month
``t + 1``) for one person. People enter the panel when their lead is created and never leave it,
so the transitions out of a state in month ``t`` always sum to the number of people in that state
at ``t``: every row of the transition matrix has a fixed, known denominator.

Everything here is **descriptive**. A higher revenue after a favourable transition shows what
customers who took that path went on to spend; it is not the revenue an intervention would
cause, because the customers who take each path differ in ways the panel does not capture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy import stats

from northstar.lifecycle.states import (
    CUSTOMER_STATES,
    MAX_STEP_DAYS,
    NOT_YET,
    STATES,
    LifecyclePanel,
    LifecycleRules,
)

CODE = {s: i for i, s in enumerate(STATES)}


def _origins(panel: LifecyclePanel, origins: Sequence[int] | None) -> list[int]:
    """Origin month indices ``t`` (transition ``t -> t + 1``); default: every month pair."""
    last = panel.n_periods - 1
    out = list(range(last)) if origins is None else [int(t) for t in origins]
    if any(not 0 <= t < last for t in out):
        raise ValueError(f"origin months must be in [0, {last})")
    return out


def recent_origins(panel: LifecyclePanel, months: int) -> list[int]:
    """The last ``months`` month-to-month transitions."""
    if not 0 < months < panel.n_periods:
        raise ValueError(f"months must be in [1, {panel.n_periods - 1}]")
    return list(range(panel.n_periods - 1 - months, panel.n_periods - 1))


def transition_counts(panel: LifecyclePanel, origins: Sequence[int] | None = None
                      ) -> pd.DataFrame:
    """People moving from each state (rows) to each state (columns), pooled over origins."""
    counts = np.zeros((len(STATES), len(STATES)), dtype=np.int64)
    for t in _origins(panel, origins):
        a, b = panel.state[:, t], panel.state[:, t + 1]
        present = a != NOT_YET
        if np.any(b[present] == NOT_YET):
            raise ValueError(f"people present in month {t} are missing in month {t + 1}")
        np.add.at(counts, (a[present], b[present]), 1)
    return pd.DataFrame(counts, index=pd.Index(STATES, name="from_state"),
                        columns=pd.Index(STATES, name="to_state"))


def transition_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalised transition probabilities (NaN for states with no people)."""
    totals = counts.sum(axis=1)
    return counts.div(totals.where(totals > 0), axis=0)


def transitions_by_period(panel: LifecyclePanel) -> pd.DataFrame:
    """Long table of non-zero transition counts per origin month."""
    frames = []
    for t in _origins(panel, None):
        c = transition_counts(panel, [t]).reset_index().melt(
            id_vars="from_state", var_name="to_state", value_name="people")
        frames.append(c.loc[c["people"] > 0].assign(
            from_period=panel.periods[t], to_period=panel.periods[t + 1],
            _order=lambda d: d["from_state"].map(CODE) * len(STATES) + d["to_state"].map(CODE)))
    out = pd.concat(frames).sort_values(["from_period", "_order"], ignore_index=True)
    return out[["from_period", "to_period", "from_state", "to_state", "people"]]


def entries(panel: LifecyclePanel) -> pd.DataFrame:
    """First observed state of people whose lead was created during each month."""
    first = np.argmax(panel.state != NOT_YET, axis=1)
    codes = panel.state[np.arange(len(first)), first]
    frame = pd.DataFrame({"period": panel.periods[first], "state": np.array(STATES)[codes]})
    return (frame.groupby(["period", "state"]).size().rename("people").reset_index())


def impossible_transitions(rules: LifecycleRules, max_step_days: int = MAX_STEP_DAYS
                           ) -> frozenset[tuple[str, str]]:
    """Transitions the state rules forbid within one monthly step.

    Derived from the thresholds, not observed: a first order is never undone, tenure and idle
    time grow by at most one step without an order, and an order resets idle time below one
    step. The pipeline requires these cells of the observed matrix to be zero.
    """
    out = {(s, "prospect") for s in CUSTOMER_STATES}
    # Non-new customers already have tenure beyond new_days, and tenure only grows.
    out |= {(s, "new") for s in ("active", "loyal", "at_risk", "churned")}
    if max_step_days <= rules.new_days:  # a first order lands in `new`
        out |= {("prospect", s) for s in ("active", "loyal", "at_risk", "churned")}
    if rules.at_risk_days + max_step_days <= rules.churn_days:  # at_risk cannot be skipped
        out |= {(s, "churned") for s in ("new", "active", "loyal")}
    if max_step_days <= rules.at_risk_days:  # a returning churned customer is engaged
        out.add(("churned", "at_risk"))
    if rules.new_days + max_step_days <= rules.loyal_min_tenure_days:
        out.add(("new", "loyal"))
    return frozenset(out)


def wilson_interval(successes: float, n: float, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (NaN when ``n == 0``)."""
    if n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(0.5 + level / 2)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (float(centre - half), float(centre + half))


# ---------------------------------------------------------------- decision points
@dataclass(frozen=True)
class DecisionPoint:
    """A lifecycle moment with a favourable and an unfavourable next state.

    Month ``t -> t + 1`` transitions out of ``origin`` count as *favourable* or *unfavourable*;
    any other destination (for example staying ``new`` for another month) is *pending* and not
    part of the rate. The unfavourable outcome is always a change of state, so each customer
    episode counts at most once on that side.
    """

    key: str
    label: str
    origin: str
    favourable: tuple[str, ...]
    unfavourable: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.origin not in STATES or not set(self.favourable) | set(self.unfavourable) <= set(
                STATES):
            raise ValueError(f"unknown state in decision point {self.key}")
        if set(self.favourable) & set(self.unfavourable):
            raise ValueError(f"favourable and unfavourable overlap in {self.key}")
        if self.origin in self.unfavourable:
            raise ValueError(f"staying in {self.origin} cannot be the unfavourable outcome")

    def as_dict(self) -> dict:
        return asdict(self)


DECISION_POINTS = (
    DecisionPoint("second_purchase", "New customer buys again within the new window", "new",
                  ("active", "loyal"), ("at_risk",)),
    DecisionPoint("active_kept", "Active customer stays engaged (vs. slips to at risk)",
                  "active", ("active", "loyal"), ("at_risk",)),
    DecisionPoint("loyal_kept", "Loyal customer stays engaged (vs. slips to at risk)", "loyal",
                  ("loyal", "active"), ("at_risk",)),
    DecisionPoint("at_risk_recovery", "At-risk customer returns before churning", "at_risk",
                  ("active", "loyal"), ("churned",)),
)


def decision_rows(panel: LifecyclePanel, point: DecisionPoint, origins: Sequence[int] | None,
                  follow_up_months: int | None = None) -> pd.DataFrame:
    """One row per resolved transition out of ``point.origin``.

    With ``follow_up_months = k`` the row also carries the revenue in the ``k`` months starting
    with the destination month, and origins without ``k`` observed months are dropped.
    """
    fav = [CODE[s] for s in point.favourable]
    unfav = [CODE[s] for s in point.unfavourable]
    frames = []
    for t in _origins(panel, origins):
        if follow_up_months is not None and t + follow_up_months >= panel.n_periods:
            continue
        a, b = panel.state[:, t], panel.state[:, t + 1]
        idx = np.flatnonzero((a == CODE[point.origin]) & np.isin(b, fav + unfav))
        frame = pd.DataFrame({"person": idx, "origin": t, "favourable": np.isin(b[idx], fav)})
        if follow_up_months is not None:
            frame["revenue_after"] = panel.revenue[idx, t + 1:t + 1 + follow_up_months].sum(axis=1)
        frames.append(frame)
    if not frames:
        raise ValueError(f"no origin month has {follow_up_months} months of follow-up")
    return pd.concat(frames, ignore_index=True)


def decision_point_table(panel: LifecyclePanel, rate_origins: Sequence[int],
                         follow_up_months: int, points: Sequence[DecisionPoint] = DECISION_POINTS
                         ) -> pd.DataFrame:
    """Rates, volumes and the descriptive revenue gap at each decision point.

    * Rate and monthly volumes use ``rate_origins`` (the recent reporting window).
    * The revenue gap compares the ``follow_up_months`` revenue of favourable and unfavourable
      transitions over every origin month with complete follow-up.
    * ``revenue_gap_per_year`` = unfavourable transitions per month x 12 x revenue gap: the
      annual revenue difference between customers who took the unfavourable path and those who
      took the favourable one. It sizes the gap for planning conversations. It is an upper
      bound, not an estimate of what any program would recover, because the two groups differ
      beyond the transition itself.
    """
    rows = []
    months = len(_origins(panel, rate_origins))
    for point in points:
        r = decision_rows(panel, point, rate_origins)
        n, k = len(r), int(r["favourable"].sum())
        lo, hi = wilson_interval(k, n)
        f = decision_rows(panel, point, None, follow_up_months)
        good = f.loc[f["favourable"], "revenue_after"]
        bad = f.loc[~f["favourable"], "revenue_after"]
        gap = float(good.mean() - bad.mean())
        in_origin = sum(int((panel.state[:, t] == CODE[point.origin]).sum())
                        for t in _origins(panel, rate_origins))
        rows.append({
            "decision_point": point.key, "label": point.label, "origin": point.origin,
            "favourable": "/".join(point.favourable),
            "unfavourable": "/".join(point.unfavourable),
            "origin_customer_months": in_origin, "resolved": n, "favourable_n": k,
            "unfavourable_n": n - k, "favourable_rate": k / n if n else np.nan,
            "rate_ci_low": lo, "rate_ci_high": hi,
            "unfavourable_per_month": (n - k) / months,
            "follow_up_resolved": len(f),
            "revenue_after_favourable": float(good.mean()),
            "revenue_after_unfavourable": float(bad.mean()),
            "revenue_gap": gap,
            "revenue_gap_per_year": (n - k) / months * 12 * gap,
        })
    return pd.DataFrame(rows)


def decision_points_by_group(panel: LifecyclePanel, origins: Sequence[int], group: str,
                             points: Sequence[DecisionPoint] = DECISION_POINTS) -> pd.DataFrame:
    """Favourable rate at each decision point by a person attribute (e.g. acquisition channel)."""
    rows = []
    for point in points:
        r = decision_rows(panel, point, origins)
        r[group] = panel.people[group].to_numpy()[r["person"].to_numpy(dtype=int)]
        for value, g in r.groupby(group, sort=True):
            k, n = int(g["favourable"].sum()), len(g)
            lo, hi = wilson_interval(k, n)
            rows.append({"decision_point": point.key, group: value, "resolved": n,
                         "favourable_n": k, "favourable_rate": k / n, "rate_ci_low": lo,
                         "rate_ci_high": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- descriptive curves
RECENCY_BINS = (*range(0, 361, 30), np.inf)
LEAD_AGE_BINS = (0, 30, 60, 90, 180, 365, np.inf)


def _bin_labels(bins: Sequence[float]) -> list[str]:
    return [f"{int(a)}+" if np.isinf(b) else f"{int(a)}-{int(b)}" for a, b in pairwise(bins)]


def repurchase_by_recency(panel: LifecyclePanel, origins: Sequence[int],
                          bins: Sequence[float] = RECENCY_BINS) -> pd.DataFrame:
    """Share of customers who order during the next month, by days since their last order."""
    idle, bought, spent = [], [], []
    for t in _origins(panel, origins):
        mask = panel.state[:, t] >= CODE["new"]
        idle.append(panel.idle_days[mask, t])
        bought.append(panel.orders[mask, t + 1] > 0)
        spent.append(panel.revenue[mask, t + 1])
    frame = pd.DataFrame({"idle_days": np.concatenate(idle), "bought": np.concatenate(bought),
                          "revenue": np.concatenate(spent)})
    frame["bucket"] = pd.cut(frame["idle_days"], list(bins), right=False,
                             labels=_bin_labels(bins))
    out = frame.groupby("bucket", observed=False).agg(
        customer_months=("bought", "size"), buyers=("bought", "sum"),
        next_month_revenue=("revenue", "mean"))
    out.insert(2, "repurchase_rate", out["buyers"] / out["customer_months"])
    return out.rename_axis("days_since_last_order").reset_index()


def conversion_by_lead_age(panel: LifecyclePanel, origins: Sequence[int],
                           bins: Sequence[float] = LEAD_AGE_BINS) -> pd.DataFrame:
    """Share of not-yet-converted leads who place a first order during the next month."""
    age, converted = [], []
    for t in _origins(panel, origins):
        mask = panel.state[:, t] == CODE["prospect"]
        age.append(panel.lead_age_days[mask, t])
        converted.append(panel.state[mask, t + 1] == CODE["new"])
    frame = pd.DataFrame({"lead_age_days": np.concatenate(age),
                          "converted": np.concatenate(converted)})
    frame["bucket"] = pd.cut(frame["lead_age_days"], list(bins), right=False,
                             labels=_bin_labels(bins))
    out = frame.groupby("bucket", observed=False).agg(
        prospect_months=("converted", "size"), conversions=("converted", "sum"))
    out["conversion_rate"] = out["conversions"] / out["prospect_months"]
    return out.rename_axis("lead_age_days").reset_index()


def state_value(panel: LifecyclePanel, origins: Sequence[int]) -> pd.DataFrame:
    """Next-month purchase rate and revenue by the state a customer is in at month end."""
    rows = []
    for s in CUSTOMER_STATES:
        n = buyers = revenue = 0.0
        for t in _origins(panel, origins):
            mask = panel.state[:, t] == CODE[s]
            n += mask.sum()
            buyers += (panel.orders[mask, t + 1] > 0).sum()
            revenue += panel.revenue[mask, t + 1].sum()
        rows.append({"state": s, "customer_months": int(n), "next_month_purchase_rate":
                     buyers / n if n else np.nan, "next_month_revenue_per_customer":
                     revenue / n if n else np.nan, "next_month_revenue": revenue})
    out = pd.DataFrame(rows)
    out["share_of_customer_months"] = out["customer_months"] / out["customer_months"].sum()
    out["share_of_next_month_revenue"] = out["next_month_revenue"] / out[
        "next_month_revenue"].sum()
    return out
