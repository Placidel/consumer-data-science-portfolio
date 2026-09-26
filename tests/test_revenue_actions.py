"""Section 04 segmentation, next-best-action rules, category rules and growth scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.revenue import actions as act
from northstar.revenue import scenarios as sc
from northstar.timeline import PredictionWindow

C = pd.Timestamp("2025-07-01")
D = pd.Timedelta(days=1)


def _base(n: int, **overrides) -> pd.DataFrame:
    frame = pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(n)],
        "run_cutoff": C,
        "pred": np.arange(n, 0, -1, dtype=float),  # C000 has the highest prediction
        "future_revenue": np.arange(n, 0, -1, dtype=float),
        "revenue_180d": 0.0,
        "revenue_365d": 0.0,
        "orders_180d": 1.0,
        "orders_total": 3.0,
        "bgnbd_p_alive": 0.9,
        "plus_member": 0.0,
        "plus_cancelled_180d": 0.0,
        "category_count": 4.0,
    })
    for col, values in overrides.items():
        frame[col] = values
    return frame


# ---------------------------------------------------------------- segmentation
def test_value_tiers_have_the_documented_sizes():
    tiers = act.value_tier(pd.Series(np.random.default_rng(0).random(200)))
    assert tiers.value_counts().to_dict() == {"Top 5%": 10, "Next 15%": 30, "Next 30%": 60,
                                              "Bottom 50%": 100}
    table = act.tier_table(_base(200), "pred", "future_revenue")
    assert table["group"].tolist() == [t for t, _ in act.VALUE_TIERS]
    assert table["share_of_customers"].sum() == pytest.approx(1)
    assert table["share_of_actual"].sum() == pytest.approx(1)
    assert table["mean_predicted"].is_monotonic_decreasing


def test_value_migration_separates_rising_and_fading_customers():
    past = np.zeros(10)
    past[[0, 5]] = [100.0, 90.0]  # C000: top past and top predicted; C005: top past only
    groups = act.migration_group(_base(10, revenue_180d=past), "pred", share=0.2)
    assert groups.tolist()[:6] == ["core", "rising", "base", "base", "base", "fading"]
    # Customers with no recent spend are never "top past" even if the top share is not full.
    none = act.migration_group(_base(10), "pred", share=0.2)
    assert set(none) == {"rising", "base"}


# ---------------------------------------------------------------- next best action
def test_actions_follow_the_documented_rule_order():
    n = 20
    rules = act.ActionRules(vip_share=0.1, growth_share=0.3, engaged_share=0.6,
                            at_risk_past_share=0.1, at_risk_p_alive=0.5,
                            cross_sell_max_categories=2)
    rev365 = np.zeros(n)
    rev365[[0, 10]] = [500.0, 400.0]  # the two top past spenders
    p_alive = np.full(n, 0.9)
    p_alive[[0, 10]] = [0.3, 0.4]  # both look lapsed
    plus = np.zeros(n)
    plus[3] = 1.0  # a member inside the growth band
    cats = np.full(n, 4.0)
    cats[[4, 7, 8, 15]] = 1.0
    data = _base(n, revenue_365d=rev365, bgnbd_p_alive=p_alive, plus_member=plus,
                 category_count=cats)
    actions = act.assign_actions(data, "pred", rules).tolist()
    assert actions[0] == "retention_save"  # at-risk beats VIP
    assert actions[1] == "vip_care"
    assert actions[2] == "plus_invite" and actions[4] == "plus_invite"  # growth before cross-sell
    assert actions[3] == "personalized_grow"  # member with 4 categories
    assert actions[7] == "cross_sell" and actions[8] == "cross_sell"
    assert actions[9] == "personalized_grow"
    assert actions[10] == "retention_save"  # lapsed top past spender outside the value bands
    assert actions[15] == "low_touch"  # narrow but below the engaged band
    assert set(actions[12:]) == {"low_touch"}
    table = act.action_table(data, pd.Series(actions), "pred", "future_revenue")
    assert table["customers"].sum() == n
    assert table["group"].tolist() == [a for a in act.ACTIONS if a in set(actions)]


def test_action_rules_are_validated():
    with pytest.raises(ValueError):
        act.ActionRules(vip_share=0.3, growth_share=0.2)
    with pytest.raises(ValueError):
        act.ActionRules(at_risk_p_alive=1.5)


@pytest.fixture
def category_tables() -> dict[str, pd.DataFrame]:
    products = pd.DataFrame({"product_id": ["A", "H", "F", "B"],
                             "category": ["apparel", "home", "footwear", "beauty"],
                             "list_price": 10.0, "unit_cost": [4.0, 5.0, 6.0, 3.0]})
    # order, customer, ts, [(product, quantity)]
    spec = [
        ("O1", "X", C - 30 * D, [("A", 1), ("H", 2)]),
        ("O2", "Y", C - 20 * D, [("A", 1)]),
        ("O3", "Z", C - 10 * D, [("A", 1), ("F", 1)]),
        ("O4", "W", C - 5 * D, [("H", 1), ("F", 1)]),
        ("O5", "Y", C + 5 * D, [("H", 1), ("F", 1)]),  # outcome window
        ("O6", "X", C + 9 * D, [("H", 1)]),
    ]
    orders = pd.DataFrame([(o, c, ts) for o, c, ts, _ in spec],
                          columns=["order_id", "customer_id", "order_ts"])
    lines = pd.DataFrame([(o, p, q, 10.0 * q) for o, _, _, items in spec for p, q in items],
                         columns=["order_id", "product_id", "quantity", "net_amount"])
    return {"orders": orders, "order_lines": lines, "products": products}


def test_category_rules_use_history_and_are_scored_on_the_window(category_tables):
    ids = pd.Index(["W", "X", "Y", "Z"], name="customer_id")
    profile = act.category_profile(category_tables, C, ids)
    # Featured = most units (X bought 2 home, 1 apparel); ties broken alphabetically (W, Z).
    assert profile["featured_category"].to_dict() == {"W": "footwear", "X": "home",
                                                      "Y": "apparel", "Z": "apparel"}
    # Popularity by distinct buyers before C: apparel 3, footwear 2, home 2 (alphabetical tie).
    assert profile["best_seller_category"].iloc[0] == "apparel"
    assert profile["suggested_new_category"].to_dict() == {"W": "apparel", "X": "footwear",
                                                           "Y": "footwear", "Z": "home"}
    assert (profile["catalog_categories"] == 4).all()

    bought = act.future_categories(category_tables, PredictionWindow(C, horizon_days=180), ids)
    assert bought.to_dict() == {"W": frozenset(), "X": frozenset({"home"}),
                                "Y": frozenset({"home", "footwear"}), "Z": frozenset()}
    cross_sell = pd.Series([True, True, True, True], index=ids)
    out = act.category_rule_evaluation(profile, bought, cross_sell)
    assert out["window_buyers"] == 2
    assert out["featured_category_hit_rate"] == pytest.approx(0.5)  # X (home) hit, Y missed
    assert out["best_seller_hit_rate"] == pytest.approx(0.0)  # neither bought apparel
    # Only Y bought a new category ({home, footwear}; suggested footwear): 1 hit of 1.
    assert out["cross_sell_new_category_buyers"] == 1
    assert out["suggested_new_category_hit_rate"] == 1.0
    assert out["random_new_category_hit_rate"] == pytest.approx(2 / 3)  # 2 of 3 unowned


def test_priority_list_orders_each_action_by_predicted_value():
    data = _base(30)
    actions = pd.Series(np.where(np.arange(30) % 2 == 0, "plus_invite", "low_touch"))
    profile = pd.DataFrame({"featured_category": "home", "suggested_new_category": "beauty"},
                           index=data.index)
    top = act.priority_list(data, actions, profile, "pred", "future_revenue", per_action=3)
    assert top["action"].tolist() == ["plus_invite"] * 3 + ["low_touch"] * 3
    assert top["customer_id"].tolist() == ["C000", "C002", "C004", "C001", "C003", "C005"]
    assert (top["suggested_new_category"] == "").all()  # only cross-sell rows name one


# ---------------------------------------------------------------- scenarios
def test_scenario_accounting_and_break_even_uplift():
    a = sc.GrowthAssumptions(uplift=0.1, contact_cost=2.0, perk_cost=10.0, redemption_rate=0.5)
    assert a.cost_per_customer == pytest.approx(7.0)
    y = np.array([500.0, 0.0, 100.0, 0.0])
    pred = np.array([400.0, 50.0, 300.0, 10.0])
    table = sc.simulate_policies(y, {"random": None, "model": pred}, [0.5], 0.5, a)
    model = table.set_index("policy").loc["model"]
    assert model["customers_targeted"] == 2 and model["baseline_revenue"] == 600
    assert model["incremental_margin"] == pytest.approx(0.1 * 600 * 0.5)
    assert model["program_cost"] == pytest.approx(14.0)
    assert model["net_value"] == pytest.approx(30 - 14)
    assert model["break_even_uplift"] == pytest.approx(14 / (0.5 * 600))
    rand = table.set_index("policy").loc["random"]
    assert rand["baseline_revenue"] == pytest.approx(0.5 * 600)
    # At the break-even uplift the program exactly pays for itself.
    even = sc.GrowthAssumptions(uplift=model["break_even_uplift"], contact_cost=2.0,
                                perk_cost=10.0, redemption_rate=0.5)
    assert sc.simulate_policies(y, {"model": pred}, [0.5], 0.5, even)["net_value"].iloc[
        0] == pytest.approx(0, abs=1e-9)


def test_threshold_plan_and_sensitivity_behave_as_documented():
    rng = np.random.default_rng(7)
    y = rng.gamma(0.6, 300, 2000) * (rng.random(2000) < 0.5)
    pred = np.clip(y + rng.normal(0, 100, 2000), 0, None)
    a = sc.GrowthAssumptions()
    t = sc.threshold_policy(y, pred, 0.5, a)
    assert t["min_predicted_revenue"] == pytest.approx(a.cost_per_customer / (a.uplift * 0.5))
    assert t["depth"] == pytest.approx((pred > t["min_predicted_revenue"]).mean())
    assert t["planned_net_value"] > 0  # every selected customer is planned to pay back

    plan = sc.planned_vs_realized(y, pred, [0.1, 0.2], 0.5, a)
    assert plan["planned_baseline_revenue"].iloc[0] == pytest.approx(
        np.sort(pred)[::-1][:200].sum())

    grid = sc.sensitivity(y, {"random": None, "model": pred}, 0.1, 0.5, a,
                          uplifts=[0.02, 0.05, 0.1], perk_costs=[5.0, 30.0])
    model = grid.loc[grid["policy"] == "model"].set_index(["uplift", "perk_cost"])["net_value"]
    assert model[(0.1, 5.0)] > model[(0.05, 5.0)] > model[(0.02, 5.0)]
    assert model[(0.05, 5.0)] > model[(0.05, 30.0)]
    rand = grid.loc[grid["policy"] == "random"].set_index(["uplift", "perk_cost"])["net_value"]
    assert (model > rand).all()


def test_growth_assumptions_are_validated_and_margin_rate_is_observed(category_tables):
    with pytest.raises(ValueError):
        sc.GrowthAssumptions(uplift=1.5)
    with pytest.raises(ValueError):
        sc.GrowthAssumptions(perk_cost=-1)
    # Pre-cutoff lines: 8 units of revenue 10 each; costs A 4 (3 units), H 5 (3), F 6 (2).
    rate = sc.observed_margin_rate(category_tables, C)
    assert rate == pytest.approx((80 - (3 * 4 + 3 * 5 + 2 * 6)) / 80)
