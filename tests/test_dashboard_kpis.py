"""Dashboard KPIs must be read from saved section outputs and trace back to a file and field."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from northstar.cli import main
from northstar.dashboard import kpis
from northstar.dashboard.artifacts import SECTIONS, load_all, provenance
from northstar.dashboard.kpis import (
    KPIS,
    OVERVIEW,
    THEMES,
    FieldError,
    evaluate_all,
    fmt_value,
    kpi_catalog,
    render_markdown,
    render_template,
    resolve,
)
from northstar.paths import DASHBOARD_DIR, PROJECTS_DIR
from northstar.profile import BEGIN_MARKER, END_MARKER, extract_generated_block

README = DASHBOARD_DIR / "README.md"
CATALOG = DASHBOARD_DIR / "outputs" / "kpi_catalog.csv"

DOC = {
    "config": {"champion": "b", "runs": ["2025-01", "2025-02", "2025-03"]},
    "models": [{"model": "a", "champion": False, "auc": 0.6},
               {"model": "b", "champion": True, "auc": 0.8}],
    "ci": [0.1, 0.3],
}


@pytest.fixture(scope="module")
def committed():
    return load_all(PROJECTS_DIR)


# --- field paths and formatting ----------------------------------------------------------------

def test_resolve_follows_keys_selectors_references_and_indices():
    assert resolve(DOC, "models[champion=True].auc") == 0.8
    assert resolve(DOC, "models[model=a].auc") == 0.6
    assert resolve(DOC, "models[model=@config.champion].auc") == 0.8
    assert resolve(DOC, "ci.1") == 0.3
    assert resolve(DOC, "config.runs.-1") == "2025-03"


@pytest.mark.parametrize("path", ["models[model=z].auc", "config.missing", "ci.5",
                                  "models[champion=True].nope", "config..runs"])
def test_resolve_fails_loudly_instead_of_guessing(path):
    with pytest.raises(FieldError):
        resolve(DOC, path)


@pytest.mark.parametrize(("value", "fmt", "expected"), [
    (5402375.92, "usd", "$5.40M"), (14764.14, "usd", "$14,764"), (84.126, "usd", "$84.13"),
    (-3179.63, "usd", "-$3,180"), (0.3001, "pct", "30.0%"), (0.028927, "pp", "+2.9 pp"),
    (-0.0104, "pp", "-1.0 pp"), (3.8607, "x", "3.86×"), (40000, "int", "40,000"),
    (0.81049, "auc", "0.810"), ("product_view->add_to_cart", "text",
                                "product view → add to cart"),
    (None, "pct", "n/a"), (float("nan"), "usd", "n/a"),
])
def test_fmt_value(value, fmt, expected):
    assert fmt_value(value, fmt) == expected


def test_render_template_fills_fields_with_optional_format():
    text = render_template("champion {config.champion}, AUC {models[champion=True].auc:auc}, "
                           "runs {config.runs.0} to {config.runs.-1}", DOC)
    assert text == "champion b, AUC 0.800, runs 2025-01 to 2025-03"


# --- the registry ------------------------------------------------------------------------------

def test_registry_is_consistent():
    keys = [k.key for k in KPIS]
    assert len(keys) == len(set(keys))
    assert {k.theme for k in KPIS} == set(THEMES)
    assert all(k.section in SECTIONS for k in KPIS)
    assert set(OVERVIEW) == set(THEMES)
    # Executive overview covers acquisition, conversion, retention, revenue and lifecycle.
    for theme, pair in OVERVIEW.items():
        assert all(kpis.KPI_BY_KEY[key].theme == theme for key in pair)


def test_every_kpi_resolves_from_committed_outputs(committed):
    values = evaluate_all(committed)
    broken = {k: v.error for k, v in values.items() if not v.available}
    assert not broken
    for v in values.values():
        assert "{" not in v.definition, f"unfilled placeholder in {v.kpi.key}"
        assert v.source.startswith(f"projects/{SECTIONS[v.kpi.section].slug}/outputs/")
        assert v.kpi.field in v.tooltip and v.source in v.tooltip


def test_kpi_values_equal_the_saved_fields_they_cite(committed):
    """Independent lookups (no resolver) of a sample of KPIs from each section's files."""
    def load(slug: str, name: str = "metrics.json") -> dict:
        return json.loads((PROJECTS_DIR / slug / "outputs" / name).read_text())

    v = {k: x.value for k, x in evaluate_all(committed).items()}
    profile = load("00_foundation", "data_profile.json")
    assert v["net_revenue"] == profile["headline"]["net_revenue"]
    assert v["lead_conversion_60d"] == profile["headline"]["lead_conversion_rate_60d"]

    acq = load("01_acquisition")
    champ = next(m for m in acq["models"] if m["model"] == acq["champion"])
    assert v["lead_score_lift_top10"] == champ["holdout_lift_top10"]

    ret = load("02_retention")
    champ = next(m for m in ret["models"] if m["model"] == ret["champion"])
    assert v["churn_rate_90d"] == champ["holdout_base_rate"]
    assert (v["retention_net_value"]
            == ret["simulation"]["expected_net_positive"]["net_value_per_run"])

    conv = load("03_conversion")
    assert v["checkout_lift"] == conv["experiment"]["primary"]["diff"]
    funnel = pd.read_csv(PROJECTS_DIR / "03_conversion/outputs/funnel_prospect_cohort.csv")
    purchase = funnel.loc[funnel["stage"] == "purchase", "share_of_start"].item()
    assert v["lead_purchase_rate_30d"] == pytest.approx(purchase, abs=1e-6)

    rev = load("04_revenue_growth")
    champ = next(m for m in rev["models"] if m["model"] == rev["champion"])
    assert v["revenue_capture_top10"] == champ["holdout_capture_top10"]

    fc = load("05_predictive_analytics")
    assert v["forecast_13w"] == fc["forecast"]["total"]["forecast"]
    wape = {m["model"]: m["wape"] for m in fc["evaluation"]["overall"]}
    assert v["forecast_wape"] == wape[fc["config"]["champion"]]

    dp = pd.read_csv(PROJECTS_DIR / "06_lifecycle/outputs/decision_points.csv")
    second = dp.loc[dp["decision_point"] == "second_purchase", "favourable_rate"].item()
    assert v["second_purchase_rate"] == pytest.approx(second, abs=1e-6)


def test_comparisons_and_intervals_are_computed_from_saved_fields(committed):
    values = evaluate_all(committed)
    lc = committed["lifecycle"].metrics["states"]
    loyal = values["loyal_share"]
    assert loyal.change == pytest.approx(lc["current"]["customer_shares"]["loyal"]
                                         - lc["year_ago"]["customer_shares"]["loyal"])
    assert loyal.comparison.endswith(f"vs {lc['year_ago']['period']}")
    total = committed["forecast"].metrics["forecast"]["total"]
    fc = values["forecast_13w"]
    assert fc.change == pytest.approx(total["forecast"] / total["same_weeks_last_year"] - 1)
    assert (fc.lower, fc.upper) == (total["lower_80"], total["upper_80"])
    lift = values["checkout_lift"]
    assert lift.lower < lift.value < lift.upper
    assert lift.interval.startswith("95% CI")


# --- provenance and missing outputs ------------------------------------------------------------

def test_missing_section_is_reported_not_raised(dashboard_projects):
    (dashboard_projects / "02_retention/outputs/metrics.json").unlink()
    artifacts = load_all(dashboard_projects)
    assert not artifacts["retention"].available
    assert artifacts["retention"].missing == ["metrics.json"]
    values = evaluate_all(artifacts)
    assert not values["churn_rate_90d"].available
    assert "northstar retention" in values["churn_rate_90d"].error
    assert values["net_revenue"].available
    prov = provenance(artifacts).set_index("key")
    assert not prov.loc["retention", "available"]
    assert prov.loc["retention", "regenerate_with"] == "northstar retention"


def test_provenance_flags_sections_built_from_different_data(dashboard_projects):
    path = dashboard_projects / "04_revenue_growth/outputs/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["data"]["seed"] += 1
    path.write_text(json.dumps(metrics))
    prov = provenance(load_all(dashboard_projects)).set_index("key")
    assert not prov.loc["revenue", "consistent"]
    assert prov.drop(index="revenue")["consistent"].all()


# --- the generated results block ---------------------------------------------------------------

def test_readme_results_block_matches_committed_outputs(committed):
    expected = render_markdown(kpi_catalog(committed), provenance(committed))
    block = extract_generated_block(README, kpis.BEGIN_MARKER, kpis.END_MARKER)
    assert block == expected


def test_committed_kpi_catalog_matches_committed_outputs(committed):
    saved = pd.read_csv(CATALOG, keep_default_na=False)
    fresh = kpi_catalog(committed)
    assert saved["key"].tolist() == fresh["key"].tolist()
    assert saved["display"].tolist() == fresh["display"].tolist()
    assert saved["field"].tolist() == fresh["field"].tolist()


def test_dashboard_kpis_command_writes_catalog_and_readme(dashboard_projects, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n\n{kpis.BEGIN_MARKER}\nstale\n{kpis.END_MARKER}\n")
    out = tmp_path / "outputs"
    args = ["dashboard-kpis", "--projects-dir", str(dashboard_projects), "--out-dir", str(out),
            "--readme", str(readme)]
    assert main(args) == 0
    catalog = pd.read_csv(out / "kpi_catalog.csv")
    assert len(catalog) == len(KPIS) and catalog["value"].notna().all()
    assert (out / "provenance.csv").exists()
    block = extract_generated_block(readme, kpis.BEGIN_MARKER, kpis.END_MARKER)
    assert "stale" not in block and "| Revenue | Net revenue, 24 months ★ |" in block
    # The section 00 README markers are distinct, so the two blocks never overwrite each other.
    assert kpis.BEGIN_MARKER != BEGIN_MARKER and kpis.END_MARKER != END_MARKER

    (dashboard_projects / "05_predictive_analytics/outputs/metrics.json").unlink()
    assert main(args) == 0
    assert main([*args, "--strict"]) == 1
    assert "| 05 Revenue forecast | no |" in extract_generated_block(
        readme, kpis.BEGIN_MARKER, kpis.END_MARKER)
