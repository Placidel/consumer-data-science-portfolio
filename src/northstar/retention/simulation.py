"""Retention offer simulation: what a fixed budget buys under explicit, adjustable assumptions.

What is **observed** (holdout data): who was in the active base, who actually churned in the next
90 days, and each customer's trailing 180-day gross margin.

What is **assumed** (``RetentionAssumptions``; none of it is in the data): how many would-be
churners an offer saves, what a save is worth, and what outreach and incentives cost. No
retention offer was ever randomized in this business, so these parameters cannot be estimated
here. Every result is therefore "expected value *if* the assumptions hold", and the break-even
save rate and sensitivity grid show how conclusions move when they do not.

Per targeted customer ``i`` with realized label ``y`` (1 = churned) and value ``v``:

* a would-be churner is saved with probability ``save_rate``; a save adds ``v`` of margin and
  costs one redeemed incentive;
* a customer who would have bought anyway redeems the incentive with probability
  ``nonchurner_redemption`` (pure subsidy, no incremental margin);
* every targeted customer costs ``contact_cost``.

``net_i = save_rate * y * (v - incentive) - contact_cost - incentive * redemption * (1 - y)``

The same formula with the model's churn probability ``p`` in place of ``y`` gives the *ex-ante*
expected net value used by the value-based policy and its "target while expected net > 0" rule.
The model predicts churn risk; the save rate is a separate (assumed) causal effect, and treating
it as constant across risk levels is itself an assumption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from northstar.retention.dataset import TARGET, VALUE

RUN = "run_cutoff"


@dataclass(frozen=True)
class RetentionAssumptions:
    """Program economics used for planning. **Assumed, not observed.**"""

    save_rate: float = 0.15
    """Share of targeted would-be churners who keep buying because of the offer (the uplift)."""
    contact_cost: float = 1.00
    """Outreach cost per targeted customer (USD): message production, delivery, agent time."""
    incentive_cost: float = 10.00
    """Voucher value per redemption (USD)."""
    nonchurner_redemption: float = 0.50
    """Share of targeted customers who would have bought anyway and redeem the voucher."""
    value_multiplier: float = 1.00
    """Value of a save = multiplier x the customer's trailing 180-day gross margin."""

    def __post_init__(self) -> None:
        for name in ("save_rate", "nonchurner_redemption"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a probability")
        for name in ("contact_cost", "incentive_cost", "value_multiplier"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict:
        return asdict(self)


DESCRIPTIONS = {
    "save_rate": "Share of targeted would-be churners retained by the offer (causal uplift)",
    "contact_cost": "Outreach cost per targeted customer (USD)",
    "incentive_cost": "Voucher value per redemption (USD)",
    "nonchurner_redemption": "Share of targeted non-churners who redeem anyway (subsidy)",
    "value_multiplier": "Value of a save, as a multiple of trailing 180-day gross margin",
}


def save_value(data: pd.DataFrame, a: RetentionAssumptions) -> np.ndarray:
    return a.value_multiplier * data[VALUE].to_numpy(float)


def expected_net(p: np.ndarray, value: np.ndarray, a: RetentionAssumptions) -> np.ndarray:
    """Ex-ante expected net value (USD) of targeting a customer with churn probability ``p``."""
    p = np.asarray(p, float)
    return (a.save_rate * p * (value - a.incentive_cost) - a.contact_cost
            - a.incentive_cost * a.nonchurner_redemption * (1 - p))


def _components(data: pd.DataFrame, a: RetentionAssumptions) -> pd.DataFrame:
    """Per-row realized quantities; top-k sums of these give a policy's outcome."""
    y = data[TARGET].to_numpy(float)
    v = save_value(data, a)
    return pd.DataFrame({
        RUN: data[RUN].to_numpy(),
        "_churner": y,
        "_nonchurner": 1 - y,
        "_churner_value": y * v,
    }, index=data.index)


def _outcome(churners: float, nonchurners: float, churner_value: float, contacts: float,
             a: RetentionAssumptions) -> dict:
    saves = a.save_rate * churners
    incremental_margin = a.save_rate * churner_value
    incentive_spend = a.incentive_cost * (saves + a.nonchurner_redemption * nonchurners)
    cost = a.contact_cost * contacts + incentive_spend
    net = incremental_margin - cost
    # Save rate at which net = 0 (other assumptions held fixed); inf if no save rate breaks even.
    denominator = churner_value - a.incentive_cost * churners
    fixed = a.contact_cost * contacts + a.incentive_cost * a.nonchurner_redemption * nonchurners
    break_even = fixed / denominator if denominator > 0 else float("inf")
    return {"customers_targeted": contacts, "churners_targeted": churners,
            "expected_saves": saves, "incremental_margin": incremental_margin,
            "program_cost": cost, "net_value": net,
            "roi": net / cost if cost > 0 else float("nan"),
            "break_even_save_rate": break_even}


def topk_sums(data: pd.DataFrame, score: str, columns: Sequence[str],
              depths: Sequence[float]) -> np.ndarray:
    """Sum of each column over the top ``depth`` share of every run by ``score``, pooled over runs.

    Returns an array of shape ``(len(depths), len(columns))``. Ties are resolved by their
    expectation under random tie-breaking (the same rule as ``expected_hits_at_k``), computed for
    all depths at once from per-run cumulative sums over distinct score values.
    """
    depths = np.asarray(depths, float)
    total = np.zeros((len(depths), len(columns)))
    for _, run in data.groupby(RUN):
        groups = run.groupby(score)[list(columns)].agg(["sum", "count"]).sort_index(
            ascending=False)
        sums = groups.xs("sum", axis=1, level=1).to_numpy(float)
        counts = groups.xs("count", axis=1, level=1).iloc[:, 0].to_numpy(float)
        ends = np.cumsum(counts)
        cum = np.vstack([np.zeros(len(columns)), np.cumsum(sums, axis=0)])
        k = np.clip(depths * len(run), 0, len(run))
        g = np.minimum(np.searchsorted(ends, k, side="left"), len(counts) - 1)
        taken_before = ends[g] - counts[g]
        frac = np.clip((k - taken_before) / counts[g], 0, 1)[:, None]
        total += cum[g] + frac * sums[g]
    return total


def evaluate_depths(data: pd.DataFrame, score: str | None, depths: Sequence[float],
                    a: RetentionAssumptions) -> list[dict]:
    """Outcomes of targeting the top ``depth`` of every run by ``score`` (None = at random)."""
    comp = _components(data, a)
    cols = ["_churner", "_nonchurner", "_churner_value"]
    if score is None:
        sums = np.outer(depths, comp[cols].sum().to_numpy(float))
    else:
        sums = topk_sums(comp.assign(_score=data[score].to_numpy()), "_score", cols, depths)
    return [_outcome(c, n, v, c + n, a) for c, n, v in sums]


def evaluate_depth(data: pd.DataFrame, score: str | None, depth: float,
                   a: RetentionAssumptions) -> dict:
    """Outcome of targeting the top ``depth`` of every run by ``score`` (None = at random)."""
    return evaluate_depths(data, score, [depth], a)[0]


def evaluate_selection(data: pd.DataFrame, selected: np.ndarray, a: RetentionAssumptions) -> dict:
    """Outcome of targeting an explicit boolean selection of rows."""
    comp = _components(data, a).loc[np.asarray(selected, bool)]
    return _outcome(comp["_churner"].sum(), comp["_nonchurner"].sum(),
                    comp["_churner_value"].sum(), float(len(comp)), a)


def _per_run(result: dict, runs: int) -> dict:
    out = dict(result)
    for key in ("customers_targeted", "churners_targeted", "expected_saves", "incremental_margin",
                "program_cost", "net_value"):
        out[f"{key}_per_run"] = out.pop(key) / runs
    return out


def simulate_policies(data: pd.DataFrame, policies: Mapping[str, str | None],
                      depths: Sequence[float], a: RetentionAssumptions) -> pd.DataFrame:
    """Per-run averages for each (policy, depth); ``None`` as a score column means random."""
    runs = data[RUN].nunique()
    results = {name: evaluate_depths(data, score, depths, a) for name, score in policies.items()}
    rows = [{"depth": depth, "policy": name, **_per_run(results[name][i], runs)}
            for i, depth in enumerate(depths) for name in policies]
    return pd.DataFrame(rows)


def threshold_policy(data: pd.DataFrame, probability: str, a: RetentionAssumptions) -> dict:
    """Target every customer whose ex-ante expected net value is positive (no hindsight)."""
    runs = data[RUN].nunique()
    net = expected_net(data[probability].to_numpy(), save_value(data, a), a)
    selected = net > 0
    return {"policy": "expected_net_positive", "depth": float(selected.mean()),
            **_per_run(evaluate_selection(data, selected, a), runs),
            "planned_net_value_per_run": float(net[selected].sum() / runs)}


def value_score(data: pd.DataFrame, probability: str, a: RetentionAssumptions) -> np.ndarray:
    return expected_net(data[probability].to_numpy(), save_value(data, a), a)


def sensitivity(data: pd.DataFrame, probability: str, depth: float, base: RetentionAssumptions,
                save_rates: Sequence[float], incentives: Sequence[float]) -> pd.DataFrame:
    """Net value per run at a fixed depth when the save rate and incentive differ from plan.

    For each scenario the value-ranked list is re-planned with that scenario's economics; the
    risk-ranked list does not depend on them.
    """
    runs = data[RUN].nunique()
    rows = []
    for s in save_rates:
        for incentive in incentives:
            a = replace(base, save_rate=s, incentive_cost=incentive)
            frame = data.assign(_value_score=value_score(data, probability, a))
            for policy, col in (("risk_ranked", probability), ("value_ranked", "_value_score")):
                r = evaluate_depth(frame, col, depth, a)
                rows.append({"save_rate": s, "incentive_cost": incentive, "policy": policy,
                             "depth": depth, "net_value_per_run": r["net_value"] / runs,
                             "roi": r["roi"]})
    return pd.DataFrame(rows)
