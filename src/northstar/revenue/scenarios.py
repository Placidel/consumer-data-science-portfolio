"""Growth program scenarios: what targeting by predicted value is worth under explicit assumptions.

The program: a growth treatment (for example a Plus invitation or VIP perk) for the top of a
ranked customer list, costing a contact per customer plus a perk for those who redeem it.

What is **observed** (holdout data): each customer's realized net revenue in the next 180 days
and the business's gross margin rate on net revenue before the cutoff.

What is **assumed** (``GrowthAssumptions``; none of it is in the data): how much the treatment
raises a targeted customer's revenue, what contact and perk cost, and how many redeem. No growth
treatment was ever randomized in this business, so the uplift cannot be estimated here. Every
dollar figure is "*if* the assumptions hold", and the break-even uplift and sensitivity grid show
how conclusions move when they do not.

Per targeted customer ``i`` with realized future revenue ``y_i`` and margin rate ``m``:

``net_i = uplift * y_i * m - contact_cost - perk_cost * redemption_rate``

The uplift is *proportional* to what the customer would have spent anyway: the treatment amplifies
existing demand and does not create it for customers who were not going to buy. That is the
assumption that makes ranking by predicted value pay off; under a flat per-customer effect the
ranking would not matter. It is the first thing a randomized test must check.

The same formula with the model's predicted revenue in place of ``y`` gives the *ex-ante* plan,
so calibration errors show up as a gap between planned and realized value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from northstar.revenue.evaluation import capture
from northstar.timeline import snapshot


@dataclass(frozen=True)
class GrowthAssumptions:
    """Program economics used for planning. **Assumed, not observed.**"""

    uplift: float = 0.05
    """Relative increase in a targeted customer's 180-day net revenue caused by the program."""
    contact_cost: float = 2.00
    """Outreach cost per targeted customer (USD)."""
    perk_cost: float = 15.00
    """Cost of the perk (for example a Plus trial month or a gift) per redemption (USD)."""
    redemption_rate: float = 0.40
    """Share of targeted customers who redeem the perk."""

    def __post_init__(self) -> None:
        for name in ("uplift", "redemption_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("contact_cost", "perk_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def cost_per_customer(self) -> float:
        return self.contact_cost + self.perk_cost * self.redemption_rate

    def as_dict(self) -> dict:
        return asdict(self)


DESCRIPTIONS = {
    "uplift": "Relative lift in a targeted customer's 180-day net revenue (causal effect)",
    "contact_cost": "Outreach cost per targeted customer (USD)",
    "perk_cost": "Perk cost per redemption (USD)",
    "redemption_rate": "Share of targeted customers who redeem the perk",
}


def observed_margin_rate(tables: Mapping[str, pd.DataFrame], cutoff: pd.Timestamp) -> float:
    """Gross margin / net revenue over all order lines before ``cutoff`` (observed, not assumed)."""
    view = snapshot({n: tables[n] for n in ("orders", "order_lines", "products")}, cutoff)
    lines = view["order_lines"]
    cost = lines["quantity"] * lines["product_id"].map(
        view["products"].set_index("product_id")["unit_cost"])
    return float((lines["net_amount"].sum() - cost.sum()) / lines["net_amount"].sum())


def _outcome(customers: float, revenue: float, margin_rate: float, a: GrowthAssumptions
             ) -> dict:
    incremental_revenue = a.uplift * revenue
    incremental_margin = incremental_revenue * margin_rate
    cost = a.cost_per_customer * customers
    net = incremental_margin - cost
    return {"customers_targeted": customers, "baseline_revenue": revenue,
            "incremental_revenue": incremental_revenue, "incremental_margin": incremental_margin,
            "program_cost": cost, "net_value": net,
            "roi": net / cost if cost > 0 else float("nan"),
            # Uplift at which net = 0, other assumptions fixed.
            "break_even_uplift": cost / (margin_rate * revenue) if revenue > 0 else float("inf")}


def targeted_revenue(y: np.ndarray, score: np.ndarray | None, depths: Sequence[float]
                     ) -> np.ndarray:
    """Realized revenue of the top ``depth`` share by ``score`` (None = random, in expectation)."""
    y = np.asarray(y, float)
    depths = np.asarray(depths, float)
    if score is None:
        return depths * y.sum()
    return capture(y, score, depths) * y.sum()


def simulate_policies(y: np.ndarray, scores: Mapping[str, np.ndarray | None],
                      depths: Sequence[float], margin_rate: float, a: GrowthAssumptions
                      ) -> pd.DataFrame:
    """Outcome of targeting the top ``depth`` of the base by each score (None = random)."""
    n = len(y)
    rows = []
    for name, score in scores.items():
        revenue = targeted_revenue(y, score, depths)
        for depth, rev in zip(depths, revenue, strict=True):
            rows.append({"depth": depth, "policy": name,
                         **_outcome(depth * n, float(rev), margin_rate, a)})
    return pd.DataFrame(rows)


def planned_vs_realized(y: np.ndarray, pred: np.ndarray, depths: Sequence[float],
                        margin_rate: float, a: GrowthAssumptions) -> pd.DataFrame:
    """Ex-ante plan (predicted revenue of the list) against the realized-under-assumptions value."""
    n = len(y)
    planned = targeted_revenue(pred, pred, depths)
    realized = targeted_revenue(y, pred, depths)
    rows = []
    for depth, p, r in zip(depths, planned, realized, strict=True):
        plan, real = (_outcome(depth * n, float(v), margin_rate, a) for v in (p, r))
        rows.append({"depth": depth, "planned_baseline_revenue": p,
                     "realized_baseline_revenue": r, "planned_net_value": plan["net_value"],
                     "realized_net_value": real["net_value"]})
    return pd.DataFrame(rows)


def threshold_policy(y: np.ndarray, pred: np.ndarray, margin_rate: float, a: GrowthAssumptions
                     ) -> dict:
    """Target every customer whose *predicted* incremental margin exceeds their cost (ex ante)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    selected = a.uplift * pred * margin_rate > a.cost_per_customer
    realized = _outcome(float(selected.sum()), float(y[selected].sum()), margin_rate, a)
    planned = _outcome(float(selected.sum()), float(pred[selected].sum()), margin_rate, a)
    return {"policy": "expected_net_positive", "depth": float(selected.mean()),
            "min_predicted_revenue": a.cost_per_customer / (a.uplift * margin_rate),
            **realized, "planned_net_value": planned["net_value"]}


def sensitivity(y: np.ndarray, scores: Mapping[str, np.ndarray | None], depth: float,
                margin_rate: float, base: GrowthAssumptions, uplifts: Sequence[float],
                perk_costs: Sequence[float]) -> pd.DataFrame:
    """Net value at a fixed depth when the uplift and perk cost differ from plan."""
    rows = []
    for uplift in uplifts:
        for perk in perk_costs:
            a = replace(base, uplift=uplift, perk_cost=perk)
            for name, score in scores.items():
                rev = float(targeted_revenue(y, score, [depth])[0])
                r = _outcome(depth * len(y), rev, margin_rate, a)
                rows.append({"uplift": uplift, "perk_cost": perk, "policy": name, "depth": depth,
                             "net_value": r["net_value"], "roi": r["roi"]})
    return pd.DataFrame(rows)
