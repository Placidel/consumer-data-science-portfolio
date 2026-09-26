"""Value segmentation and a transparent, rule-based next best action (NBA).

Everything here is **prioritization**: it decides who gets which treatment first, using
predictions and pre-cutoff history. None of it measures what a treatment *causes*; realized
revenue is shown next to each group only to check that the groups are ordered as predicted.

* **Value tiers** cut the base by predicted future revenue (top 5%, next 15%, next 30%, bottom
  50%).
* **Value migration** crosses past value (trailing 180-day revenue) with predicted value to find
  customers a "top spenders" list misses (*rising*) and ones it over-rates (*fading*).
* **Next best action** assigns each customer exactly one action from ordered, auditable rules
  (``ActionRules``), plus a *featured category* for the message and, for cross-sell, a suggested
  new category. Both category rules are scored against the categories customers actually bought
  in the outcome window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from northstar.timeline import PredictionWindow, snapshot

VALUE_TIERS = (("Top 5%", 0.05), ("Next 15%", 0.20), ("Next 30%", 0.50), ("Bottom 50%", 1.0))
MIGRATION_GROUPS = ("core", "rising", "fading", "base")
MIGRATION_LABELS = {
    "core": "Core: high past and high predicted value",
    "rising": "Rising: high predicted, not a top past spender",
    "fading": "Fading: top past spender, not high predicted",
    "base": "Base: neither",
}

ACTIONS = ("retention_save", "vip_care", "plus_invite", "cross_sell", "personalized_grow",
           "low_touch")
ACTION_LABELS = {
    "retention_save": "Retention outreach (hand to the section 02 program)",
    "vip_care": "VIP care: early access and service, no discount",
    "plus_invite": "Invite to Northstar Plus",
    "cross_sell": "Cross-sell a new category",
    "personalized_grow": "Personalized content in the featured category",
    "low_touch": "Low-cost newsletter only",
}


@dataclass(frozen=True)
class ActionRules:
    """Thresholds of the NBA rules (business policy choices, not estimates)."""

    vip_share: float = 0.05
    """Top share of the base by predicted value treated as VIP."""
    growth_share: float = 0.20
    """Top share by predicted value eligible for the Plus invitation."""
    engaged_share: float = 0.50
    """Top share by predicted value that receives any paid-for treatment."""
    at_risk_past_share: float = 0.20
    """Top share by trailing 365-day revenue checked for retention risk."""
    at_risk_p_alive: float = 0.50
    """BG/NBD probability-alive below which a past top spender is 'at risk'."""
    cross_sell_max_categories: int = 2
    """Customers who have bought in at most this many categories are cross-sell candidates."""

    def __post_init__(self) -> None:
        if not 0 < self.vip_share <= self.growth_share <= self.engaged_share <= 1:
            raise ValueError("need 0 < vip_share <= growth_share <= engaged_share <= 1")
        if not 0 < self.at_risk_past_share <= 1 or not 0 <= self.at_risk_p_alive <= 1:
            raise ValueError("at-risk thresholds must be shares/probabilities")

    def as_dict(self) -> dict:
        return asdict(self)


def top_share(values: pd.Series, share: float) -> pd.Series:
    """True for the top ``share`` of rows by ``values`` (ties by row order; rows are sorted ids)."""
    rank = values.rank(method="first", ascending=False)
    return rank <= share * len(values)


def value_tier(pred: pd.Series) -> pd.Series:
    rank_share = pred.rank(method="first", ascending=False) / len(pred)
    labels = np.select([rank_share <= s for _, s in VALUE_TIERS[:-1]],
                       [t for t, _ in VALUE_TIERS[:-1]], VALUE_TIERS[-1][0])
    return pd.Series(labels, index=pred.index)


def _group_summary(data: pd.DataFrame, groups: pd.Series, pred: str, target: str,
                   order: tuple[str, ...]) -> pd.DataFrame:
    frame = data.assign(_group=groups.to_numpy(), _pred=data[pred], _y=data[target],
                        _active=(data["orders_180d"] > 0).astype(float),
                        _buyer=(data[target] > 0).astype(float))
    out = frame.groupby("_group").agg(
        customers=("_y", "size"), predicted_revenue=("_pred", "sum"),
        actual_revenue=("_y", "sum"), mean_predicted=("_pred", "mean"),
        mean_actual=("_y", "mean"), buyer_rate=("_buyer", "mean"),
        mean_revenue_180d=("revenue_180d", "mean"), active_share=("_active", "mean"),
        plus_member_rate=("plus_member", "mean"), mean_orders_total=("orders_total", "mean"),
        mean_category_count=("category_count", "mean"), mean_p_alive=("bgnbd_p_alive", "mean"),
    ).reindex([g for g in order if g in set(groups)])
    out.insert(1, "share_of_customers", out["customers"] / len(data))
    out.insert(3, "share_of_predicted", out["predicted_revenue"] / data[pred].sum())
    out.insert(5, "share_of_actual", out["actual_revenue"] / data[target].sum())
    return out.rename_axis("group").reset_index()


def tier_table(data: pd.DataFrame, pred: str, target: str) -> pd.DataFrame:
    return _group_summary(data, value_tier(data[pred]), pred, target,
                          tuple(t for t, _ in VALUE_TIERS))


def migration_group(data: pd.DataFrame, pred: str, share: float = 0.2) -> pd.Series:
    past = top_share(data["revenue_180d"], share) & (data["revenue_180d"] > 0)
    future = top_share(data[pred], share)
    return pd.Series(np.select([past & future, future, past], ["core", "rising", "fading"],
                               "base"), index=data.index)


def migration_table(data: pd.DataFrame, pred: str, target: str, share: float = 0.2
                    ) -> pd.DataFrame:
    return _group_summary(data, migration_group(data, pred, share), pred, target,
                          MIGRATION_GROUPS)


# ---------------------------------------------------------------- categories
def _line_categories(tables: Mapping[str, pd.DataFrame], orders: pd.DataFrame) -> pd.DataFrame:
    lines = tables["order_lines"]
    lines = lines.loc[lines["order_id"].isin(orders["order_id"])]
    return pd.DataFrame({
        "customer_id": lines["order_id"].map(orders.set_index("order_id")["customer_id"]),
        "category": lines["product_id"].map(
            tables["products"].set_index("product_id")["category"]),
        "quantity": lines["quantity"],
    })


def category_profile(tables: Mapping[str, pd.DataFrame], cutoff: pd.Timestamp,
                     ids: pd.Index) -> pd.DataFrame:
    """Pre-cutoff category rules per customer.

    * ``featured_category``: the customer's most-bought category by units (ties alphabetical).
    * ``suggested_new_category``: the most widely bought category (by distinct pre-cutoff buyers
      across the base) that the customer has not bought yet; empty if they own them all.
    """
    view = snapshot({n: tables[n] for n in ("orders", "order_lines", "products")}, cutoff)
    li = _line_categories(view, view["orders"])
    units = li.groupby(["customer_id", "category"])["quantity"].sum().reset_index()
    featured = (units.sort_values(["customer_id", "quantity", "category"],
                                  ascending=[True, False, True])
                .groupby("customer_id").head(1).set_index("customer_id")["category"])
    popularity = units.groupby("category")["customer_id"].nunique().sort_values(
        ascending=False, kind="stable")
    owned = units.groupby("customer_id")["category"].agg(frozenset)

    def suggest(cats: frozenset) -> str:
        return next((c for c in popularity.index if c not in cats), "")

    owned = owned.reindex(ids)
    return pd.DataFrame({
        "featured_category": featured.reindex(ids),
        "owned_categories": owned,
        "suggested_new_category": owned.map(suggest),
        "best_seller_category": popularity.index[0],
        "catalog_categories": view["products"]["category"].nunique(),
    }, index=ids)


def future_categories(tables: Mapping[str, pd.DataFrame], window: PredictionWindow,
                      ids: pd.Index) -> pd.Series:
    """Categories each customer bought in the outcome window (empty set if none). Outcome only."""
    orders = tables["orders"]
    orders = orders.loc[window.label_mask(orders["order_ts"]) & orders["customer_id"].isin(ids)]
    li = _line_categories(tables, orders)
    bought = li.groupby("customer_id")["category"].agg(frozenset)
    return bought.reindex(ids).map(lambda v: v if isinstance(v, frozenset) else frozenset())


def category_rule_evaluation(profile: pd.DataFrame, bought: pd.Series,
                             cross_sell: pd.Series) -> dict:
    """How often the category rules name a category the customer actually bought next.

    * Featured category, among customers who bought anything in the window, vs. featuring the
      global best seller to everyone.
    * Suggested new category, among cross-sell customers who bought at least one *new* category,
      vs. the expected hit rate of suggesting a uniformly random unowned category.
    """
    buyers = bought.map(len) > 0
    featured_hit = [f in b for f, b in zip(profile.loc[buyers, "featured_category"],
                                           bought[buyers], strict=True)]
    best_hit = [profile["best_seller_category"].iloc[0] in b for b in bought[buyers]]

    new = pd.Series([b - o for b, o in zip(bought, profile["owned_categories"], strict=True)],
                    index=bought.index)
    eligible = cross_sell & (new.map(len) > 0)
    suggested_hit = [s in n for s, n in zip(profile.loc[eligible, "suggested_new_category"],
                                            new[eligible], strict=True)]
    n_unowned = (profile.loc[eligible, "catalog_categories"]
                 - profile.loc[eligible, "owned_categories"].map(len))
    random_hit = new[eligible].map(len) / n_unowned
    return {
        "window_buyers": int(buyers.sum()),
        "featured_category_hit_rate": float(np.mean(featured_hit)),
        "best_seller_hit_rate": float(np.mean(best_hit)),
        "best_seller_category": profile["best_seller_category"].iloc[0],
        "cross_sell_new_category_buyers": int(eligible.sum()),
        "suggested_new_category_hit_rate": float(np.mean(suggested_hit)) if suggested_hit
        else None,
        "random_new_category_hit_rate": float(random_hit.mean()) if len(random_hit) else None,
    }


# ---------------------------------------------------------------- next best action
def assign_actions(data: pd.DataFrame, pred: str, rules: ActionRules) -> pd.Series:
    """One action per customer; the first matching rule wins (order = ``ACTIONS``).

    1. ``retention_save``: a top past spender (trailing 365 days) the BG/NBD model thinks has
       probably lapsed. Protecting existing value comes before growing it.
    2. ``vip_care``: top ``vip_share`` by predicted value.
    3. ``plus_invite``: top ``growth_share`` by predicted value, not a Plus member and no Plus
       cancellation in the last 180 days.
    4. ``cross_sell``: top ``engaged_share`` and at most ``cross_sell_max_categories`` categories.
    5. ``personalized_grow``: the rest of the top ``engaged_share``.
    6. ``low_touch``: everyone else.
    """
    past_top = top_share(data["revenue_365d"], rules.at_risk_past_share) & (
        data["revenue_365d"] > 0)
    at_risk = past_top & (data["bgnbd_p_alive"] < rules.at_risk_p_alive)
    vip = top_share(data[pred], rules.vip_share)
    growth = top_share(data[pred], rules.growth_share)
    engaged = top_share(data[pred], rules.engaged_share)
    plus_ok = (data["plus_member"] == 0) & (data["plus_cancelled_180d"] == 0)
    narrow = data["category_count"] <= rules.cross_sell_max_categories
    conditions = [at_risk, vip, growth & plus_ok, engaged & narrow, engaged]
    return pd.Series(np.select(conditions, list(ACTIONS[:-1]), ACTIONS[-1]), index=data.index)


def action_table(data: pd.DataFrame, actions: pd.Series, pred: str, target: str
                 ) -> pd.DataFrame:
    return _group_summary(data, actions, pred, target, ACTIONS)


def priority_list(data: pd.DataFrame, actions: pd.Series, profile: pd.DataFrame, pred: str,
                  target: str, per_action: int = 10) -> pd.DataFrame:
    """Top ``per_action`` customers of each action by predicted value (the call list order)."""
    frame = data.assign(action=actions.to_numpy(),
                        featured_category=profile["featured_category"].to_numpy(),
                        suggested_new_category=np.where(
                            actions.to_numpy() == "cross_sell",
                            profile["suggested_new_category"].to_numpy(), ""))
    frame = frame.sort_values([pred, "customer_id"], ascending=[False, True])
    top = frame.groupby("action", sort=False).head(per_action)
    top = top.assign(_order=top["action"].map({a: i for i, a in enumerate(ACTIONS)}))
    top = top.sort_values(["_order", pred, "customer_id"], ascending=[True, False, True])
    return pd.DataFrame({
        "run_cutoff": top["run_cutoff"].dt.date.astype(str),
        "customer_id": top["customer_id"],
        "action": top["action"],
        "predicted_revenue_180d": top[pred],
        "bgnbd_p_alive": top["bgnbd_p_alive"],
        "revenue_180d": top["revenue_180d"],
        "revenue_365d": top["revenue_365d"],
        "orders_total": top["orders_total"],
        "plus_member": top["plus_member"],
        "featured_category": top["featured_category"],
        "suggested_new_category": top["suggested_new_category"],
        "realized_revenue_next_180d": top[target],
    }).reset_index(drop=True)
