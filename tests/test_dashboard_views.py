"""Filters behind the dashboard controls slice saved outputs correctly."""

from __future__ import annotations

import pandas as pd
import pytest

from northstar.dashboard import views
from northstar.dashboard.artifacts import load_all
from northstar.paths import PROJECTS_DIR


@pytest.fixture(scope="module")
def art():
    return load_all(PROJECTS_DIR)


def test_month_window_is_inclusive_and_order_insensitive(art):
    monthly = art["foundation"].tables["monthly_kpis"]
    window = views.month_window(monthly, "2025-01", "2025-06")
    assert window["month"].tolist() == [f"2025-0{i}" for i in range(1, 7)]
    pd.testing.assert_frame_equal(window, views.month_window(monthly, "2025-06", "2025-01"))
    totals = views.window_totals(window)
    assert totals["months"] == 6
    assert totals["net_revenue"] == pytest.approx(window["net_revenue"].sum())
    assert totals["average_order_value"] == pytest.approx(totals["net_revenue"]
                                                         / totals["orders"])


def test_full_window_reconciles_with_the_foundation_profile(art):
    """Monthly rows sum to the section 00 headline, so window totals share its definitions."""
    monthly = art["foundation"].tables["monthly_kpis"]
    totals = views.window_totals(monthly)
    headline = art["foundation"].metrics["headline"]
    assert totals["net_revenue"] == pytest.approx(headline["net_revenue"], abs=0.05)
    assert totals["orders"] == headline["orders"]
    assert totals["new_customers"] == headline["customers"]


def test_filter_channels(art):
    channels = art["foundation"].tables["channel_summary"]
    shown = views.filter_channels(channels, ["email", "referral"])
    assert sorted(shown["acquisition_channel"]) == ["email", "referral"]
    assert views.filter_channels(channels, []).empty


def test_budget_at_capacity(art):
    budget = art["acquisition"].tables["budget_simulation"]
    rows = views.budget_at_capacity(budget, 0.2)
    assert (rows["capacity_share"] == 0.2).all()
    assert rows["conversions_reached_per_run"].is_monotonic_decreasing
    assert set(rows["policy"]) == set(budget["policy"])
    with pytest.raises(ValueError, match="capacity"):
        views.budget_at_capacity(budget, 0.77)


def test_curve_at_depth_snaps_to_a_saved_depth(art):
    curve = art["retention"].tables["retention_value_curve"]
    at = views.curve_at_depth(curve, 0.205, ["risk_ranked", "random"])
    assert set(at["policy"]) == {"risk_ranked", "random"}
    assert at["depth"].tolist() == pytest.approx([0.2, 0.2])
    assert 0 not in views.depths(curve)
    best = views.best_depth(curve, "value_ranked", "net_value_per_run")
    assert best["net_value_per_run"] == curve.loc[curve["policy"] == "value_ranked",
                                                  "net_value_per_run"].max()


def test_segment_funnel(art):
    seg = views.segment_funnel(art["conversion"].tables["funnel_segments"], "device_type")
    assert (seg["dimension"] == "device_type").all()
    assert seg["sessions"].is_monotonic_decreasing
    with pytest.raises(ValueError):
        views.segment_funnel(art["conversion"].tables["funnel_segments"], "weather")


@pytest.mark.parametrize("level", [50, 80])
def test_forecast_view_uses_the_chosen_saved_interval(art, level):
    fc = art["forecast"]
    champion = fc.metrics["config"]["champion"]
    history, ahead = views.forecast_view(fc.tables["weekly_revenue"], fc.tables["forecast"],
                                         champion, level, 26)
    assert len(history) == 26
    assert history["week_start"].iloc[-1] == fc.tables["weekly_revenue"]["week_start"].max()
    assert ahead["lower"].tolist() == fc.tables["forecast"][f"lower_{level}"].tolist()
    assert ahead["upper"].tolist() == fc.tables["forecast"][f"upper_{level}"].tolist()
    # Empirical intervals come from past (skewed) errors, so the point forecast need not be
    # centred in them; the band itself must be ordered.
    assert (ahead["lower"] <= ahead["upper"]).all()
    assert ahead["week_start"].min() > history["week_start"].max()


def test_forecast_view_rejects_unsaved_levels(art):
    fc = art["forecast"]
    with pytest.raises(ValueError):
        views.forecast_view(fc.tables["weekly_revenue"], fc.tables["forecast"], "harmonic", 95,
                            13)


@pytest.mark.parametrize("as_share", [True, False])
def test_state_mix(art, as_share):
    counts = art["lifecycle"].tables["state_counts_by_month"]
    mix = views.state_mix(counts, as_share)
    assert set(mix["state"]) == set(views.CUSTOMER_STATES)
    by_period = mix.groupby("period")
    if as_share:
        sums = by_period["value"].sum()
        assert sums[counts.set_index("period")["customers"] > 0].round(6).eq(1).all()
    else:
        pd.testing.assert_series_equal(by_period["value"].sum(),
                                       counts.set_index("period")["customers"],
                                       check_names=False, check_dtype=False)


def test_largest_channels(art):
    by_channel = art["lifecycle"].tables["cohort_retention_by_channel"]
    top = views.largest_channels(by_channel, 3)
    first = by_channel.loc[by_channel["months_since_acquisition"] == 0]
    assert top == first.sort_values("customers", ascending=False)[
        "acquisition_channel"].head(3).tolist()


def test_labels():
    assert views.label("bgnbd_gamma_gamma") == "BG/NBD + Gamma-Gamma"
    assert views.label("organic_search") == "Organic search"
    assert views.label("2-3") == "2-3"
