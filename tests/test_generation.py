from __future__ import annotations

import pandas as pd
import pytest

from northstar.io import load_tables, write_tables
from northstar.schema import TABLE_NAMES
from northstar.synthetic import generate
from northstar.timeline import DATA_END, DATA_START

SMALL = 1_500


def test_same_seed_reproduces_identical_tables():
    first = generate(seed=11, n_prospects=SMALL)
    second = generate(seed=11, n_prospects=SMALL)
    assert list(first) == list(second) == list(TABLE_NAMES)
    for name in TABLE_NAMES:
        pd.testing.assert_frame_equal(first[name], second[name], check_exact=True)


def test_same_seed_reproduces_byte_identical_files(tmp_path):
    m1 = write_tables(generate(seed=11, n_prospects=SMALL), tmp_path / "a", seed=11,
                      n_prospects=SMALL)
    m2 = write_tables(generate(seed=11, n_prospects=SMALL), tmp_path / "b", seed=11,
                      n_prospects=SMALL)
    assert m1 == m2
    for name in TABLE_NAMES:
        a = (tmp_path / "a" / f"{name}.parquet").read_bytes()
        b = (tmp_path / "b" / f"{name}.parquet").read_bytes()
        assert a == b, name


def test_different_seed_changes_behavioral_tables():
    a = generate(seed=1, n_prospects=SMALL)
    b = generate(seed=2, n_prospects=SMALL)
    assert not a["orders"]["net_amount"].equals(b["orders"]["net_amount"])
    assert not a["prospects"]["created_at"].equals(b["prospects"]["created_at"])
    # Reference tables that do not depend on the random draws stay fixed.
    pd.testing.assert_frame_equal(a["campaigns"], b["campaigns"])


def test_parquet_round_trip_preserves_values_and_dtypes(tmp_path):
    tables = generate(seed=3, n_prospects=SMALL)
    write_tables(tables, tmp_path, seed=3, n_prospects=SMALL)
    loaded = load_tables(tmp_path)
    for name in TABLE_NAMES:
        pd.testing.assert_frame_equal(loaded[name], tables[name], check_exact=True)


def test_n_prospects_controls_scale_and_rejects_tiny_runs():
    tables = generate(seed=5, n_prospects=500)
    assert len(tables["prospects"]) == 500
    with pytest.raises(ValueError):
        generate(seed=5, n_prospects=10)


def test_history_spans_at_least_18_months(tables):
    months = tables["orders"]["order_ts"].dt.to_period("M")
    assert months.nunique() >= 18
    assert tables["prospects"]["created_at"].min() < DATA_START + pd.Timedelta(days=7)
    assert tables["orders"]["order_ts"].max() > DATA_END - pd.Timedelta(days=7)
    assert (DATA_END - DATA_START).days >= 18 * 30


def test_every_entity_required_by_the_spec_is_populated(tables):
    required = ["prospects", "customers", "marketing_touches", "campaigns", "sessions",
                "funnel_events", "orders", "order_lines", "subscription_events",
                "support_contacts"]
    for name in required:
        assert len(tables[name]) > 0, name
