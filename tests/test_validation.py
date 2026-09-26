"""The validator accepts generated data and rejects each class of corruption."""

from __future__ import annotations

import re
from collections.abc import Callable

import pandas as pd
import pytest

from northstar.paths import DOCS_DIR
from northstar.schema import TABLES, render_data_dictionary
from northstar.validation import (
    validate_all,
    validate_business_rules,
    validate_foreign_keys,
    validate_schema,
)


def test_generated_data_passes_every_check(tables):
    assert validate_schema(tables) == []
    assert validate_foreign_keys(tables) == []
    assert validate_business_rules(tables) == []


def test_primary_keys_are_unique_and_foreign_keys_resolve(tables):
    for name, spec in TABLES.items():
        df = tables[name]
        assert not df.duplicated(list(spec.primary_key)).any(), name
        for fk in spec.foreign_keys:
            values = df[fk.column].dropna()
            assert values.isin(tables[fk.ref_table][fk.ref_column]).all(), (name, fk.column)


def _dup_order(t):
    t["orders"] = pd.concat([t["orders"], t["orders"].iloc[[0]]], ignore_index=True)


def _orphan_line(t):
    t["order_lines"].loc[0, "product_id"] = "SKU9999"


def _null_required(t):
    t["prospects"].loc[3, "region"] = pd.NA


def _bad_category(t):
    t["prospects"].loc[3, "acquisition_channel"] = "tv"


def _out_of_range(t):
    t["sessions"].loc[0, "session_start"] = pd.Timestamp("2026-02-01")


def _net_mismatch(t):
    t["orders"].loc[0, "net_amount"] += 5.0


def _line_sum_mismatch(t):
    t["order_lines"].loc[0, "quantity"] += 1
    t["order_lines"].loc[0, "net_amount"] += t["order_lines"].loc[0, "unit_price"]


def _customer_since_shift(t):
    t["customers"].loc[5, "customer_since"] += pd.Timedelta(days=2)


def _skip_funnel_stage(t):
    ev = t["funnel_events"]
    victim = ev.loc[ev["stage_number"] == 4, "session_id"].iloc[0]
    t["funnel_events"] = ev.loc[~((ev["session_id"] == victim) & (ev["stage_number"] == 2))]


def _leak_customer_id(t):
    s = t["sessions"]
    pre = s.index[s["customer_id"].isna() & s["prospect_id"].isin(t["customers"]["prospect_id"])]
    row = pre[0]
    pid = s.loc[row, "prospect_id"]
    cust = t["customers"].loc[t["customers"]["prospect_id"] == pid, "customer_id"].iloc[0]
    s.loc[row, "customer_id"] = cust


def _flip_variant(t):
    a = t["experiment_assignments"]
    a.loc[0, "variant"] = "control" if a.loc[0, "variant"] == "treatment" else "treatment"


def _drop_subscribe(t):
    ev = t["subscription_events"]
    t["subscription_events"] = ev.drop(ev.index[ev["event_type"] == "subscribe"][0])


def _contact_before_order(t):
    sc = t["support_contacts"]
    row = sc.index[sc["order_id"].notna()][0]
    order_ts = t["orders"].set_index("order_id").loc[sc.loc[row, "order_id"], "order_ts"]
    sc.loc[row, "contact_ts"] = order_ts - pd.Timedelta(hours=1)


def _click_without_open(t):
    tc = t["marketing_touches"]
    row = tc.index[~tc["opened"]][0]
    tc.loc[row, "clicked"] = True


def _store_order_with_session(t):
    o = t["orders"]
    store = o.index[o["order_channel"] == "store"][0]
    o.loc[store, "session_id"] = o.loc[o["session_id"].notna(), "session_id"].iloc[0]


CORRUPTIONS: dict[str, tuple[Callable, str]] = {
    "duplicate primary key": (_dup_order, "duplicate primary keys"),
    "orphan foreign key": (_orphan_line, "order_lines.product_id"),
    "null in required column": (_null_required, "prospects.region"),
    "unknown category": (_bad_category, "unexpected values"),
    "timestamp outside history": (_out_of_range, "outside the data range"),
    "order arithmetic": (_net_mismatch, "net_amount != gross_amount - discount_amount"),
    "order vs lines": (_line_sum_mismatch, "sum of line"),
    "first order timing": (_customer_since_shift, "customer_since"),
    "funnel stage skipped": (_skip_funnel_stage, "skip or repeat funnel stages"),
    "customer id before conversion": (_leak_customer_id, "customer_id presence"),
    "experiment variant": (_flip_variant, "deterministic hash rule"),
    "membership sequence": (_drop_subscribe, "invalid subscribe/renew/cancel"),
    "support timing": (_contact_before_order, "before the order was placed"),
    "click without open": (_click_without_open, "clicked without being opened"),
    "store order session": (_store_order_with_session, "session_id must be null"),
}


@pytest.mark.parametrize("case", list(CORRUPTIONS))
def test_validator_detects_corruption(mutable_tables, case):
    corrupt, expected = CORRUPTIONS[case]
    corrupt(mutable_tables)
    issues = validate_all(mutable_tables)
    assert any(expected in issue for issue in issues), issues


def test_data_dictionary_is_generated_from_schema():
    committed = (DOCS_DIR / "data_dictionary.md").read_text()
    assert committed == render_data_dictionary(), (
        "docs/data_dictionary.md is stale; run `northstar data-dictionary`"
    )


PII_TOKENS = ("first_name", "last_name", "full_name", "email_address", "phone", "street",
              "address", "zip", "postal", "birth", "dob", "ssn", "ip_", "latitude", "longitude")


def test_schema_contains_no_personal_data_fields():
    for name, spec in TABLES.items():
        for col in spec.column_names:
            assert not any(tok in col for tok in PII_TOKENS), f"{name}.{col}"


def test_identifiers_are_synthetic_codes(tables):
    patterns = {("prospects", "prospect_id"): r"P\d{6}", ("customers", "customer_id"): r"C\d{6}",
                ("orders", "order_id"): r"O\d{7}", ("sessions", "session_id"): r"S\d{7}"}
    for (table, col), pattern in patterns.items():
        assert tables[table][col].str.fullmatch(pattern).all(), (table, col)
    assert re.fullmatch(r"[A-Z][a-z]+ item \d{2}", tables["products"]["product_name"].iloc[0])


OUTCOME_TOKENS = ("convert", "churn", "ltv", "lifetime", "total", "future", "label", "target",
                  "last_order", "n_orders", "score")


@pytest.mark.parametrize("table", ["prospects", "customers"])
def test_entity_tables_store_no_outcomes_or_aggregates(table):
    """Entity tables hold only attributes known at creation; outcomes live in event logs."""
    for col in TABLES[table].column_names:
        assert not any(tok in col for tok in OUTCOME_TOKENS), f"{table}.{col}"
