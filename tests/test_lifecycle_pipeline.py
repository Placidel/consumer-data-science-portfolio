"""Section 06 end to end: CLI interface, integrity gate and traceability of documented results."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from northstar.cli import main
from northstar.lifecycle import cohorts as co
from northstar.lifecycle import report
from northstar.lifecycle import states as st
from northstar.lifecycle.states import CUSTOMER_STATES, STATES, LifecycleRules
from northstar.paths import LIFECYCLE_DIR
from northstar.profile import extract_generated_block
from northstar.synthetic import generate
from northstar.synthetic import params as p

OUT = LIFECYCLE_DIR / "outputs"
METRICS = OUT / "metrics.json"
README = LIFECYCLE_DIR / "README.md"
FIGURES = ("state_mix", "transition_matrix", "cohort_retention", "retention_curves",
           "repurchase_by_recency", "decision_points", "rfm_lifecycle")
TABLES = ("state_counts_by_month", "transition_counts", "transition_matrix",
          "transitions_by_month", "entries_by_month", "state_value", "decision_points",
          "decision_points_by_channel", "repurchase_by_recency", "conversion_by_lead_age",
          "cohort_retention", "cohort_retention_pooled", "cohort_retention_by_channel",
          "rfm_segments", "lifecycle_state_summary", "rfm_by_lifecycle")


@pytest.fixture(scope="module")
def small_data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("lc") / "raw"
    assert main(["generate-data", "--seed", "11", "--n-prospects", "4000", "--out", str(out),
                 "--skip-profile"]) == 0
    return out


def _readme(path):
    path.write_text(f"# Test\n\n{report.BEGIN_MARKER}\nstale\n{report.END_MARKER}\n\nend\n")
    return path


def test_lifecycle_command_writes_traceable_outputs(small_data_dir, tmp_path):
    out, readme = tmp_path / "outputs", _readme(tmp_path / "README.md")
    code = main(["lifecycle", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(readme), "--no-generate"])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data"] == {"seed": 11, "n_prospects": 4000}
    assert metrics["audit"]["passed"]
    for name in TABLES:
        assert (out / f"{name}.csv").stat().st_size > 0, name
    for name in FIGURES:
        assert (out / "figures" / f"{name}.png").stat().st_size > 0, name
    block = extract_generated_block(readme, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(metrics)
    assert "stale" not in readme.read_text() and readme.read_text().rstrip().endswith("end")

    # The written tables agree with each other and with the metrics.
    matrix = pd.read_csv(out / "transition_matrix.csv").set_index("from_state")
    counts = pd.read_csv(out / "transition_counts.csv").set_index("from_state")
    assert list(matrix.columns) == list(STATES)
    assert matrix.loc[list(STATES)].sum(axis=1).round(3).eq(1.0).all()
    assert counts.to_numpy().sum() == sum(sum(r.values()) for r in
                                          metrics["transitions"]["counts"].values())
    monthly = pd.read_csv(out / "state_counts_by_month.csv")
    assert (monthly[list(CUSTOMER_STATES)].sum(axis=1) == monthly["customers"]).all()
    assert monthly["customers"].iloc[-1] == metrics["population"]["customers"]
    pooled = pd.read_csv(out / "cohort_retention_pooled.csv")
    assert (pooled["customers"] == pooled["customers"].iloc[0]).all()


def test_lifecycle_command_refuses_to_write_when_audit_fails(small_data_dir, tmp_path,
                                                             monkeypatch):
    def failing_audit(*args, **kwargs):
        return {"passed": False, "checks": {"no_transition_the_state_rules_forbid": False},
                "details": {}}

    monkeypatch.setattr(report, "integrity_audit", failing_audit)
    out = tmp_path / "outputs"
    code = main(["lifecycle", "--data-dir", str(small_data_dir), "--out-dir", str(out),
                 "--readme", str(tmp_path / "missing.md"), "--no-generate"])
    assert code == 1
    assert not out.exists()


def test_audit_catches_states_that_look_one_month_ahead(tables, monkeypatch):
    """A classic look-ahead bug: labelling each month with the next month's state."""
    real = st.build_panel

    def look_ahead(*args, **kwargs):
        panel = real(*args, **kwargs)
        panel.state = np.concatenate([panel.state[:, 1:], panel.state[:, -1:]], axis=1)
        return panel

    monkeypatch.setattr(report, "build_panel", look_ahead)
    with pytest.raises(report.IntegrityError, match="states_unchanged_when_later_orders"):
        report.run_analysis(tables)


def test_audit_catches_a_pooled_curve_whose_denominator_shrinks(tables, monkeypatch):
    """The naive pooled curve averages whatever cohorts are observed at each month."""

    def naive(table, follow_up_months, by=()):
        t = table.loc[table["observed"]
                      & (table["months_since_acquisition"] <= follow_up_months)]
        out = t.groupby([*by, "months_since_acquisition"]).agg(
            cohorts=("cohort", "nunique"), first_cohort=("cohort", "min"),
            last_cohort=("cohort", "max"), customers=("cohort_size", "sum"),
            buyers=("buyers", "sum"), revenue=("revenue", "sum")).reset_index()
        out["retention"] = out["buyers"] / out["customers"]
        out["revenue_per_customer"] = out["revenue"] / out["customers"]
        out["cumulative_revenue_per_customer"] = out["revenue_per_customer"].cumsum()
        return out

    monkeypatch.setattr(co, "pooled_retention", naive)
    with pytest.raises(report.IntegrityError, match="pooled_curve_uses_one_fixed_cohort_set"):
        report.run_analysis(tables)


def test_lifecycle_command_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="northstar generate-data"):
        main(["lifecycle", "--data-dir", str(tmp_path / "none"), "--no-generate"])


# ---------------------------------------------------------------- committed results
@pytest.fixture(scope="module")
def committed():
    return json.loads(METRICS.read_text())


def test_readme_results_block_matches_committed_metrics(committed):
    block = extract_generated_block(README, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(committed)


def test_committed_run_uses_the_default_design(committed):
    assert committed["data"] == {"seed": p.DEFAULT_SEED, "n_prospects": p.DEFAULT_N_PROSPECTS}
    assert committed["config"]["rules"] == LifecycleRules().as_dict()
    assert committed["config"] == {**committed["config"], **report.LifecycleConfig().as_dict()}
    assert committed["audit"]["passed"]
    assert committed["transitions"]["first_origin"] == "2024-12"
    assert committed["transitions"]["last_destination"] == "2025-12"
    assert committed["population"]["periods"] == 24


def test_readme_documents_definitions_and_the_descriptive_causal_distinction():
    text = README.read_text()
    for phrase in ("first match wins", "Descriptive patterns, not causal explanations",
                   "upper bounds", "must not be added up", "each is a hypothesis to test",
                   "denominator is fixed", "missing, not zero", "one fixed", "policy choices",
                   "randomised holdout", "Precedence"):
        assert phrase in text, phrase


def test_prose_claims_in_the_readme_hold_for_the_committed_run(committed):
    """The narrative sections of the README make these claims; keep them true."""
    cur, prev = committed["states"]["current"], committed["states"]["year_ago"]
    engaged = ("new", "active", "loyal")
    # Ageing base: churned share roughly doubled to about two in five; engaged share fell.
    assert 1.8 < cur["customer_shares"]["churned"] / prev["customer_shares"]["churned"] < 2.2
    assert 0.35 < cur["customer_shares"]["churned"] < 0.45
    assert sum(cur["customer_shares"][s] for s in engaged) < sum(
        prev["customer_shares"][s] for s in engaged)

    # Loyal: ~a seventh of customer-months, over 40% of next-month revenue, majority buy.
    value = {r["state"]: r for r in committed["state_value"]}
    assert 0.12 < value["loyal"]["share_of_customer_months"] < 0.16
    assert value["loyal"]["share_of_next_month_revenue"] > 0.4
    assert value["loyal"]["next_month_purchase_rate"] > 0.5
    assert value["loyal"]["next_month_purchase_rate"] == max(
        r["next_month_purchase_rate"] for r in value.values())
    assert 0.015 < value["churned"]["next_month_purchase_rate"] < 0.03

    # Decision points.
    dp = {r["decision_point"]: r for r in committed["decision_points"]}
    second, recovery = dp["second_purchase"], dp["at_risk_recovery"]
    assert second["rate_ci_high"] < 0.5
    assert second["revenue_after_favourable"] > 3 * second["revenue_after_unfavourable"]
    assert 0.28 < recovery["favourable_rate"] < 0.38
    assert dp["loyal_kept"]["favourable_rate"] > 0.9
    gaps = sorted(dp, key=lambda k: -dp[k]["revenue_gap_per_year"])
    assert set(gaps[:2]) == {"second_purchase", "at_risk_recovery"}
    for k in gaps[:2]:
        assert dp[k]["revenue_gap_per_year"] > 2 * dp["loyal_kept"]["revenue_gap_per_year"]
    ch = {(r["decision_point"], r["acquisition_channel"]): r["favourable_rate"]
          for r in committed["decision_points_by_channel"]}
    for good in ("organic_search", "referral"):
        for bad in ("display", "paid_social"):
            assert ch[("second_purchase", good)] > ch[("second_purchase", bad)]
            assert ch[("at_risk_recovery", good)] > ch[("at_risk_recovery", bad)]

    # Recency curve: 90-120 almost as good as 60-90, 150-180 below half of it, 360+ below 1%;
    # the chance roughly halves from 60-90 to 120-150 and is below 3% after 240 days.
    rec = {r["days_since_last_order"]: r["repurchase_rate"]
           for r in committed["repurchase_by_recency"]}
    assert rec["90-120"] > 0.85 * rec["60-90"]
    assert rec["150-180"] < 0.5 * rec["90-120"]
    assert 0.4 < rec["120-150"] / rec["60-90"] < 0.65
    assert all(rec[b] < 0.03 for b in ("240-270", "270-300", "300-330", "330-360", "360+"))
    assert rec["360+"] < 0.01

    # Cohorts: ~30% in month 1, falling over two months, ~20% by month 12; linear-ish revenue.
    pooled = {r["months_since_acquisition"]: r for r in committed["cohorts"]["pooled"]}
    assert 0.27 < pooled[1]["retention"] < 0.33
    assert pooled[3]["retention"] < pooled[2]["retention"] < pooled[1]["retention"]
    assert 0.18 < pooled[12]["retention"] < 0.22
    assert (pooled[1]["retention"] - pooled[3]["retention"]) > (
        pooled[3]["retention"] - pooled[6]["retention"])
    increments = np.diff([pooled[k]["cumulative_revenue_per_customer"] for k in range(1, 13)])
    assert increments.min() > 0.6 * increments.mean()
    by = {(r["acquisition_channel"], r["months_since_acquisition"]): r
          for r in committed["cohorts"]["by_channel"]}
    m12 = {c: r["retention"] for (c, k), r in by.items() if k == 12}
    rev = {c: r["cumulative_revenue_per_customer"] for (c, k), r in by.items() if k == 12}
    assert max(m12, key=m12.get) == "referral" == max(rev, key=rev.get)
    assert min(m12, key=m12.get) == "display" == min(rev, key=rev.get)
    assert m12["display"] < 0.5 * m12["referral"]

    # Leads: first 30 days several times the 30-60 rate, nil after six months, ~a fifth
    # convert in their creation month.
    leads = {r["lead_age_days"]: r["conversion_rate"] for r in committed["conversion_by_lead_age"]}
    assert leads["0-30"] > 3 * leads["30-60"]
    assert leads["180-365"] < 0.001 and leads["365+"] < 0.001
    t = committed["transitions"]
    assert 0.15 < t["new_leads_entering_as_new"] / t["new_leads"] < 0.25
    reactivation = t["probabilities"]["churned"]["active"] + t["probabilities"]["churned"][
        "loyal"]
    assert 0.015 < reactivation < 0.03

    # RFM: Champions ~a quarter of customers, >60% of trailing revenue, ~70% of next revenue.
    seg = committed["segments"]
    rfm = {r["rfm_segment"]: r for r in seg["rfm"]}
    assert 0.22 < rfm["champions"]["share_of_customers"] < 0.3
    assert rfm["champions"]["share_of_revenue_365d"] > 0.6
    assert 0.65 < rfm["champions"]["share_of_next_revenue"] < 0.75
    at_risk = seg["crosstab"]["at_risk"]
    assert max(at_risk, key=at_risk.get) == "loyalists"
    assert 0.2 < at_risk["needs_attention"] / sum(at_risk.values()) < 0.3
    assert seg["crosstab"]["churned"]["cannot_lose"] > 0
    assert seg["crosstab"]["churned"]["champions"] == 0


def _assert_close(fresh, committed, path="metrics"):
    if isinstance(committed, dict):
        assert set(fresh) == set(committed), path
        for k in committed:
            _assert_close(fresh[k], committed[k], f"{path}.{k}")
    elif isinstance(committed, list):
        assert len(fresh) == len(committed), path
        for i, (x, y) in enumerate(zip(fresh, committed, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(committed, float) and not isinstance(committed, bool):
        assert math.isclose(fresh, committed, rel_tol=1e-6, abs_tol=1e-4), path
    else:
        assert fresh == committed, path


@pytest.mark.slow
def test_committed_metrics_are_reproduced_from_default_generation(committed):
    tables = generate(seed=p.DEFAULT_SEED, n_prospects=p.DEFAULT_N_PROSPECTS)
    fresh, _ = report.run_analysis(tables)
    expected = {k: v for k, v in committed.items() if k != "data"}
    _assert_close(json.loads(json.dumps(fresh)), expected)
