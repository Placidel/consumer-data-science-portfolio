"""Cohort retention: fixed denominators, explicit censoring and pooled curves over fixed cohorts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.lifecycle.cohorts import cohort_retention, pooled_retention

END = pd.Timestamp("2024-04-01")  # observation ends after March 2024


@pytest.fixture
def toy(toy_lifecycle):
    tables, _ = toy_lifecycle
    # Add a February customer who never orders again.
    customers = pd.concat([tables["customers"], pd.DataFrame({
        "customer_id": ["C5"], "prospect_id": ["P5"],
        "customer_since": [pd.Timestamp("2024-02-05")]})], ignore_index=True)
    orders = pd.concat([tables["orders"], pd.DataFrame({
        "customer_id": ["C5"], "order_ts": [pd.Timestamp("2024-02-05")],
        "net_amount": [25.0]})], ignore_index=True)
    customers["acquisition_channel"] = ["email", "referral", "email", "referral"]
    return customers, orders


def _cell(table, cohort, k, col="retention"):
    row = table.loc[(table["cohort"] == pd.Timestamp(cohort))
                    & (table["months_since_acquisition"] == k)]
    return row[col].item()


def test_cohort_table_matches_a_hand_count(toy):
    customers, orders = toy
    table = cohort_retention(customers, orders, end=END)
    # January cohort: C1, C3, C4. February: C5.
    assert _cell(table, "2024-01-01", 0, "cohort_size") == 3
    assert _cell(table, "2024-01-01", 0) == 1.0
    assert _cell(table, "2024-01-01", 1) == pytest.approx(2 / 3)   # C1 (Feb 15), C3 (Feb 2)
    assert _cell(table, "2024-01-01", 2) == pytest.approx(1 / 3)   # C3 only
    assert _cell(table, "2024-02-01", 0) == 1.0
    assert _cell(table, "2024-02-01", 1) == 0.0                    # observed, nobody ordered
    assert _cell(table, "2024-01-01", 0, "revenue_per_customer") == pytest.approx(
        (50 + 20 + 40) / 3)
    assert _cell(table, "2024-01-01", 2, "cumulative_revenue_per_customer") == pytest.approx(
        (50 + 20 + 40 + 30 + 20 + 20) / 3)


def test_denominators_never_change_and_unobserved_months_are_missing(toy):
    customers, orders = toy
    table = cohort_retention(customers, orders, end=END)
    sizes = table.groupby("cohort")["cohort_size"].agg(["nunique", "first"])
    assert (sizes["nunique"] == 1).all()
    assert sizes["first"].to_dict() == {pd.Timestamp("2024-01-01"): 3,
                                        pd.Timestamp("2024-02-01"): 1}
    feb2 = table.loc[(table["cohort"] == pd.Timestamp("2024-02-01"))
                     & (table["months_since_acquisition"] == 2)]
    assert not feb2["observed"].item() and np.isnan(feb2["retention"].item())
    assert (table["retention"].isna() == ~table["observed"]).all()
    # Later orders (after the end of observation) never leak into the table.
    assert table["buyers"].max() <= 3


def test_pooled_curve_uses_a_fixed_set_of_cohorts(toy):
    customers, orders = toy
    table = cohort_retention(customers, orders, end=END)
    one = pooled_retention(table, 1)
    assert one["customers"].tolist() == [4, 4]
    assert one["retention"].tolist() == pytest.approx([1.0, 2 / 4])
    two = pooled_retention(table, 2)
    assert two["customers"].tolist() == [3, 3, 3]      # February has only one month observed
    assert two["cohorts"].tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="follow-up"):
        pooled_retention(table, 3)


def test_split_by_attribute_partitions_the_cohorts(toy):
    customers, orders = toy
    table = cohort_retention(customers, orders, end=END, by=["acquisition_channel"])
    total = cohort_retention(customers, orders, end=END)
    k0 = table.loc[table["months_since_acquisition"] == 0].groupby("cohort")["cohort_size"].sum()
    assert k0.to_dict() == total.loc[total["months_since_acquisition"] == 0].set_index(
        "cohort")["cohort_size"].to_dict()
    pooled = pooled_retention(table, 1, by=["acquisition_channel"])
    assert set(pooled["acquisition_channel"]) == {"email", "referral"}
    assert pooled.groupby("acquisition_channel")["customers"].nunique().eq(1).all()


def test_orders_before_first_order_are_rejected(toy):
    customers, orders = toy
    bad = pd.concat([orders, pd.DataFrame({"customer_id": ["C5"],
                                           "order_ts": [pd.Timestamp("2024-01-20")],
                                           "net_amount": [1.0]})], ignore_index=True)
    with pytest.raises(ValueError, match="before"):
        cohort_retention(customers, bad, end=END)


def test_generated_data_cohorts(tables):
    table = cohort_retention(tables["customers"], tables["orders"])
    sizes = table.groupby("cohort")["cohort_size"].agg(["nunique", "first"])
    assert (sizes["nunique"] == 1).all()
    assert sizes["first"].sum() == len(tables["customers"])
    assert (table.loc[table["months_since_acquisition"] == 0, "retention"] == 1.0).all()
    observed = table.loc[table["observed"]]
    assert observed["retention"].between(0, 1).all()
    assert observed.groupby("cohort")["cumulative_revenue_per_customer"].apply(
        lambda s: s.is_monotonic_increasing).all()
    total = observed["revenue"].sum()
    assert total == pytest.approx(tables["orders"]["net_amount"].sum())
    pooled = pooled_retention(table, 12)
    assert (pooled["customers"] == pooled["customers"].iloc[0]).all() and len(pooled) == 13
