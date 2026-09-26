"""Schema, key-integrity and business-rule validation for the shared tables.

Every check returns human-readable issue strings; an empty list means the data is valid. The CLI
refuses to write data that fails validation, and the test suite asserts both that generated data
passes and that deliberately corrupted data is caught.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from northstar.schema import TABLES
from northstar.synthetic.simulate import experiment_variant
from northstar.timeline import DATA_END, DATA_START

MONEY_TOL = 0.011

_KIND_CHECKS = {
    "string": pd.api.types.is_string_dtype,
    "timestamp": pd.api.types.is_datetime64_any_dtype,
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "bool": pd.api.types.is_bool_dtype,
}


def _count(mask: pd.Series | np.ndarray) -> int:
    return int(np.asarray(mask).sum())


def validate_schema(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []
    missing = set(TABLES) - set(tables)
    if missing:
        return [f"missing tables: {sorted(missing)}"]
    for name, spec in TABLES.items():
        df = tables[name]
        if list(df.columns) != spec.column_names:
            issues.append(f"{name}: columns {list(df.columns)} != {spec.column_names}")
            continue
        if df.empty:
            issues.append(f"{name}: table is empty")
        for col in spec.columns:
            s = df[col.name]
            if not _KIND_CHECKS[col.kind](s):
                issues.append(f"{name}.{col.name}: dtype {s.dtype} is not {col.kind}")
            nulls = _count(s.isna())
            if nulls and not col.nullable:
                issues.append(f"{name}.{col.name}: {nulls} nulls in non-nullable column")
            if col.allowed is not None:
                bad = sorted(set(s.dropna().unique()) - set(col.allowed))
                if bad:
                    issues.append(f"{name}.{col.name}: unexpected values {bad[:5]}")
        pk = list(spec.primary_key)
        if df[pk].isna().any().any():
            issues.append(f"{name}: null primary key")
        dupes = _count(df.duplicated(pk))
        if dupes:
            issues.append(f"{name}: {dupes} duplicate primary keys {pk}")
        if spec.time_column:
            ts = df[spec.time_column]
            out = _count((ts < DATA_START) | (ts >= DATA_END))
            if out:
                issues.append(f"{name}.{spec.time_column}: {out} values outside the data range")
    return issues


def validate_foreign_keys(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []
    for name, spec in TABLES.items():
        for fk in spec.foreign_keys:
            values = tables[name][fk.column].dropna()
            orphans = _count(~values.isin(tables[fk.ref_table][fk.ref_column]))
            if orphans:
                issues.append(f"{name}.{fk.column}: {orphans} values missing from "
                              f"{fk.ref_table}.{fk.ref_column}")
    return issues


def _customer_lookup(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return tables["customers"][["customer_id", "prospect_id", "customer_since"]]


def _check_people(t: Mapping[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []
    prospects, customers = t["prospects"], t["customers"]
    if customers["prospect_id"].duplicated().any():
        issues.append("customers: a prospect converted into more than one customer")
    merged = customers.merge(prospects, on="prospect_id", suffixes=("", "_p"))
    for col in ("acquisition_channel", "region", "age_band", "income_band", "device_type",
                "email_opt_in"):
        n = _count(merged[col] != merged[f"{col}_p"])
        if n:
            issues.append(f"customers.{col}: {n} rows differ from the prospect record")
    n = _count(merged["customer_since"] < merged["created_at"])
    if n:
        issues.append(f"customers: {n} converted before the lead was created")
    if not customers["customer_since"].is_monotonic_increasing:
        issues.append("customers: ids are not issued in order of conversion")

    # Sourcing campaign matches channel and is live on the creation date.
    camp = prospects.merge(t["campaigns"], on="campaign_id", how="left")
    organic = camp["acquisition_channel"] == "organic_search"
    if _count(organic != camp["campaign_id"].isna()):
        issues.append("prospects: campaign_id must be null exactly for organic leads")
    paid = ~organic
    if _count(paid & (camp["channel"] != camp["acquisition_channel"])):
        issues.append("prospects: sourcing campaign channel differs from acquisition channel")
    day = camp["created_at"].dt.normalize()
    if _count(paid & ((day < camp["start_date"]) | (day > camp["end_date"]))):
        issues.append("prospects: sourcing campaign not active on the creation date")
    return issues


def _check_customer_id_timing(df: pd.DataFrame, ts_col: str, name: str,
                              customers: pd.DataFrame) -> list[str]:
    """customer_id must be set exactly when the row happens at/after the person's conversion."""
    issues: list[str] = []
    m = df[["prospect_id", "customer_id", ts_col]].merge(
        customers.rename(columns={"customer_id": "expected_customer"}), on="prospect_id",
        how="left",
    )
    after = m["customer_since"].notna() & (m[ts_col] >= m["customer_since"])
    has_id = m["customer_id"].notna()
    n = _count(after != has_id)
    if n:
        issues.append(f"{name}: {n} rows where customer_id presence disagrees with conversion time")
    n = _count(has_id & (m["customer_id"] != m["expected_customer"]))
    if n:
        issues.append(f"{name}: {n} rows whose customer_id belongs to another person")
    return issues


def _check_funnel(t: Mapping[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []
    ev = t["funnel_events"].sort_values(["session_id", "stage_number"])
    g = ev.groupby("session_id")["stage_number"]
    stats = pd.DataFrame({"n": g.size(), "max": g.max(), "nunique": g.nunique(), "min": g.min()})
    bad = (stats["n"] != stats["max"]) | (stats["nunique"] != stats["n"]) | (stats["min"] != 1)
    if bad.any():
        issues.append(f"funnel_events: {_count(bad)} sessions skip or repeat funnel stages")
    backwards = ev.groupby("session_id")["event_ts"].diff() < pd.Timedelta(0)
    if backwards.any():
        issues.append(f"funnel_events: {_count(backwards)} events earlier than the previous stage")
    sessions = t["sessions"].set_index("session_id")
    first = ev.loc[ev["stage_number"] == 1].set_index("session_id")["event_ts"]
    if len(first) != len(sessions):
        issues.append("funnel_events: every session needs exactly one session_start event")
    elif _count(first.reindex(sessions.index) != sessions["session_start"]):
        issues.append("funnel_events: session_start event time differs from session start")
    return issues


def _check_orders(t: Mapping[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []
    orders, lines, customers = t["orders"], t["order_lines"], t["customers"]
    if _count((orders["net_amount"] - (orders["gross_amount"] - orders["discount_amount"])).abs()
              > MONEY_TOL):
        issues.append("orders: net_amount != gross_amount - discount_amount")
    if _count((orders[["gross_amount", "discount_amount"]] < 0).any(axis=1)
              | (orders["net_amount"] <= 0) | (orders["discount_amount"] > orders["gross_amount"])):
        issues.append("orders: negative amounts or discount above gross")
    line_net = lines["quantity"] * lines["unit_price"] - lines["discount_amount"]
    if _count((line_net - lines["net_amount"]).abs() > MONEY_TOL):
        issues.append("order_lines: net_amount != quantity * unit_price - discount_amount")
    if _count(lines["quantity"] <= 0):
        issues.append("order_lines: non-positive quantity")
    sums = lines.groupby("order_id")[["quantity", "discount_amount", "net_amount"]].sum()
    o = orders.set_index("order_id").join(sums, rsuffix="_lines")
    if o["quantity"].isna().any():
        issues.append("orders: order without lines")
    else:
        if _count(o["quantity"] != o["item_count"]):
            issues.append("orders: item_count != sum of line quantities")
        for col in ("discount_amount", "net_amount"):
            if _count((o[col] - o[f"{col}_lines"]).abs() > MONEY_TOL * 10):
                issues.append(f"orders: {col} != sum of line {col}")

    first = orders.groupby("customer_id")["order_ts"].min()
    c = customers.set_index("customer_id")
    if len(first) != len(c):
        issues.append("customers: every customer must have at least one order")
    elif _count(first.reindex(c.index) != c["customer_since"]):
        issues.append("customers: customer_since differs from the first order time")

    store = orders["order_channel"] == "store"
    if _count(store == orders["session_id"].notna()):
        issues.append("orders: session_id must be null exactly for store orders")

    # Digital orders <-> purchase events.
    purchases = t["funnel_events"].loc[t["funnel_events"]["event_type"] == "purchase"]
    digital = orders.loc[~store, ["order_id", "customer_id", "order_ts", "order_channel",
                                  "session_id"]]
    if digital["session_id"].duplicated().any():
        issues.append("orders: a purchase session is linked to more than one order")
    if set(purchases["session_id"]) != set(digital["session_id"]):
        issues.append("orders: purchase events and digital orders do not correspond one-to-one")
    m = digital.merge(purchases[["session_id", "event_ts"]], on="session_id", how="left")
    m = m.merge(t["sessions"][["session_id", "prospect_id", "platform"]], on="session_id",
                how="left")
    m = m.merge(customers[["customer_id", "prospect_id"]], on="customer_id",
                suffixes=("", "_customer"))
    if _count(m["event_ts"] != m["order_ts"]):
        issues.append("orders: order_ts differs from the purchase event time")
    if _count(m["platform"] != m["order_channel"]):
        issues.append("orders: order_channel differs from the purchase session platform")
    if _count(m["prospect_id"] != m["prospect_id_customer"]):
        issues.append("orders: purchase session belongs to another person")

    # Discount campaigns.
    oc = orders.loc[orders["campaign_id"].notna()].merge(t["campaigns"], on="campaign_id")
    expected = (oc["gross_amount"] * oc["discount_pct"]).round(2)
    if _count((oc["discount_amount"] - expected).abs() > 0.01 * oc["item_count"] + MONEY_TOL):
        issues.append("orders: discount does not match the campaign discount")
    promo = oc["objective"] == "promotion"
    day = oc["order_ts"].dt.normalize()
    if _count(promo & ((day < oc["start_date"]) | (day > oc["end_date"]))):
        issues.append("orders: promotion applied outside its dates")
    if _count(orders["campaign_id"].isna() & (orders["discount_amount"] > 0)):
        issues.append("orders: discount without a campaign")
    return issues


def _check_subscriptions(t: Mapping[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []
    ev = t["subscription_events"].sort_values(["customer_id", "event_ts",
                                               "subscription_event_id"])
    # State machine per customer: no plan --subscribe--> plan --renew--> plan --cancel--> no plan.
    current_plan: dict[str, str | None] = {}
    bad_customers = set()
    for cust, event, plan in zip(ev["customer_id"], ev["event_type"], ev["plan"], strict=True):
        state = current_plan.get(cust)
        if event == "subscribe":
            valid = state is None
            current_plan[cust] = plan
        else:
            valid = state == plan
            if event == "cancel":
                current_plan[cust] = None
        if not valid:
            bad_customers.add(cust)
    if bad_customers:
        issues.append(f"subscription_events: {len(bad_customers)} customers with invalid "
                      "subscribe/renew/cancel sequences")
    m = ev.merge(_customer_lookup(t), on="customer_id")
    if _count(m["event_ts"] < m["customer_since"]):
        issues.append("subscription_events: membership event before the first order")
    return issues


def _check_support(t: Mapping[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []
    sc = t["support_contacts"].merge(_customer_lookup(t), on="customer_id")
    if _count(sc["contact_ts"] < sc["customer_since"]):
        issues.append("support_contacts: contact before the customer existed")
    if _count(sc["resolved_at"] < sc["contact_ts"]):
        issues.append("support_contacts: resolved before being opened")
    if _count(sc["resolved_at"].isna() & sc["csat_score"].notna()):
        issues.append("support_contacts: CSAT recorded for an unresolved contact")
    csat = sc["csat_score"].dropna()
    if _count((csat < 1) | (csat > 5)):
        issues.append("support_contacts: CSAT outside 1-5")
    linked = sc.loc[sc["order_id"].notna()].merge(
        t["orders"][["order_id", "customer_id", "order_ts"]], on="order_id",
        suffixes=("", "_order"))
    if _count(linked["customer_id"] != linked["customer_id_order"]):
        issues.append("support_contacts: order belongs to another customer")
    if _count(linked["contact_ts"] < linked["order_ts"]):
        issues.append("support_contacts: contact about an order before the order was placed")
    return issues


def _check_experiments(t: Mapping[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []
    a = t["experiment_assignments"].merge(t["experiments"], on="experiment_id")
    expected = [experiment_variant(e, pid, share) for e, pid, share
                in zip(a["experiment_id"], a["prospect_id"], a["treatment_share"], strict=True)]
    if _count(a["variant"].to_numpy() != np.array(expected, dtype=object)):
        issues.append("experiment_assignments: variant differs from the deterministic hash rule")
    day = a["assigned_at"].dt.normalize()
    if _count((day < a["start_date"]) | (day > a["end_date"])):
        issues.append("experiment_assignments: assignment outside the experiment window")
    m = a.merge(t["customers"][["prospect_id", "customer_since"]], on="prospect_id", how="left")
    if _count(m["customer_since"].notna() & (m["assigned_at"] >= m["customer_since"])):
        issues.append("experiment_assignments: prospect assigned after already converting")
    # Assignment is logged at the first session inside the window.
    s = t["sessions"].merge(t["experiments"][["experiment_id", "start_date", "end_date"]],
                            how="cross")
    s = s.loc[(s["session_start"] >= s["start_date"])
              & (s["session_start"] < s["end_date"] + pd.Timedelta(days=1))
              & s["customer_id"].isna()]
    first = s.groupby(["experiment_id", "prospect_id"])["session_start"].min().rename("first")
    j = a.set_index(["experiment_id", "prospect_id"]).join(first, how="outer")
    if _count(j["first"] != j["assigned_at"]):
        issues.append("experiment_assignments: not every eligible prospect is assigned at "
                      "their first in-window session")
    return issues


def _check_touches(t: Mapping[str, pd.DataFrame]) -> list[str]:
    issues: list[str] = []
    tc = t["marketing_touches"]
    if _count(tc["clicked"] & ~tc["opened"]):
        issues.append("marketing_touches: clicked without being opened")
    if _count(tc["cost"] < 0):
        issues.append("marketing_touches: negative cost")
    return issues


def validate_business_rules(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    customers = _customer_lookup(tables)
    return [
        *_check_people(tables),
        *_check_customer_id_timing(tables["sessions"], "session_start", "sessions", customers),
        *_check_customer_id_timing(tables["marketing_touches"], "touch_at",
                                   "marketing_touches", customers),
        *_check_funnel(tables),
        *_check_orders(tables),
        *_check_subscriptions(tables),
        *_check_support(tables),
        *_check_experiments(tables),
        *_check_touches(tables),
    ]


def validate_all(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    """All checks. Business rules run only if schema and keys are sound."""
    issues = validate_schema(tables)
    if issues:
        return issues
    issues = validate_foreign_keys(tables)
    if issues:
        return issues
    return validate_business_rules(tables)
