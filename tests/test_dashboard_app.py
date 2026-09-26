"""Headless tests of the Streamlit dashboard (Streamlit AppTest plus a real server launch)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from northstar.cli import dashboard_command
from northstar.dashboard import APP_PATH
from northstar.dashboard.artifacts import PROJECTS_ENV, load_all
from northstar.dashboard.kpis import OVERVIEW, evaluate_all, fmt_value
from northstar.paths import PROJECTS_DIR

PAGES = ("Executive overview", "Acquisition", "Conversion", "Retention", "Revenue growth",
         "Forecast", "Lifecycle", "Definitions & sources")
TIMEOUT = 120


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv(PROJECTS_ENV, str(PROJECTS_DIR))
    at = AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT)
    return at.run()


def _goto(at: AppTest, page: str) -> AppTest:
    return at.radio(key="page").set_value(page).run()


def _metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def _no_errors(at: AppTest) -> None:
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_every_page_renders_without_exceptions(app):
    assert tuple(app.radio(key="page").options) == PAGES
    for page in PAGES:
        _goto(app, page)
        _no_errors(app)
        assert app.title[0].value  # every page has a heading


def test_overview_shows_every_headline_kpi_from_saved_outputs(app):
    values = evaluate_all(load_all(PROJECTS_DIR))
    shown = _metrics(app)
    for pair in OVERVIEW.values():
        for key in pair:
            v = values[key]
            assert shown[v.kpi.label] == v.display
    # Tooltips carry the definition and the source field.
    helps = {m.label: m.help for m in app.metric}
    net = values["net_revenue"]
    assert net.kpi.field in helps[net.kpi.label] and "data_profile.json" in helps[net.kpi.label]


def test_month_filter_recomputes_window_totals_from_monthly_kpis(app):
    monthly = pd.read_csv(PROJECTS_DIR / "00_foundation/outputs/monthly_kpis.csv")
    assert _metrics(app)["Net revenue in window"] == fmt_value(monthly["net_revenue"].sum(),
                                                               "usd")
    app.select_slider(key="overview_months").set_range("2025-01", "2025-06").run()
    _no_errors(app)
    window = monthly.loc[monthly["month"].between("2025-01", "2025-06")]
    shown = _metrics(app)
    assert shown["Net revenue in window"] == fmt_value(window["net_revenue"].sum(), "usd")
    assert shown["Orders in window"] == fmt_value(window["orders"].sum(), "int")
    assert shown["New customers in window"] == fmt_value(window["new_customers"].sum(), "int")


def test_retention_depth_slider_reads_the_saved_value_curve(app):
    _goto(app, "Retention")
    curve = pd.read_csv(PROJECTS_DIR / "02_retention/outputs/retention_value_curve.csv")
    target = sorted(d for d in curve["depth"].unique() if d > 0)[-1]
    app.select_slider(key="ret_depth").set_value(target).run()
    _no_errors(app)
    table = app.dataframe[0].value
    expected = curve.loc[(curve["depth"] == target) & (curve["policy"] == "value_ranked")]
    row = table.loc[table["Policy"] == "Highest expected value first"].iloc[0]
    assert row["Net value / run"] == fmt_value(expected["net_value_per_run"].item(), "usd")

    app.multiselect(key="ret_policies").set_value(["random"]).run()
    _no_errors(app)
    assert app.dataframe[0].value["Policy"].tolist() == ["Random"]


def test_acquisition_filters(app):
    _goto(app, "Acquisition")
    app.multiselect(key="acq_channels").set_value(["email", "referral"]).run()
    _no_errors(app)
    channel_table = app.dataframe[0].value
    assert channel_table["Channel"].tolist() == ["Email", "Referral"]

    budget = pd.read_csv(PROJECTS_DIR / "01_acquisition/outputs/budget_simulation.csv")
    cap = budget["capacity_share"].max()
    app.select_slider(key="acq_capacity").set_value(cap).run()
    _no_errors(app)
    row = budget.loc[(budget["capacity_share"] == cap)
                     & (budget["policy"] == "logistic_regression")].iloc[0]
    text = " ".join(m.value for m in app.markdown)
    assert f"**{cap:.0%}** capacity" in text
    assert f"**{row['conversions_reached_per_run']:.0f}** eventual buyers" in text


@pytest.mark.parametrize(("page", "key", "value"), [
    ("Conversion", "conv_dimension", "traffic_source"),
    ("Revenue growth", "rev_depth", 0.3),
    ("Lifecycle", "lc_metric", "cumulative_revenue_per_customer"),
    ("Lifecycle", "lc_share", False),
    ("Forecast", "fc_level", 50),
    ("Forecast", "fc_baselines", ["naive_4wk", "seasonal_naive_yoy"]),
    ("Definitions & sources", "defs_themes", ["Retention"]),
])
def test_other_controls_rerender_cleanly(app, page, key, value):
    _goto(app, page)
    app.get_by_key(key).set_value(value).run()
    _no_errors(app)


def test_forecast_interval_level_switches_saved_total_interval(app):
    _goto(app, "Forecast")
    total = json.loads((PROJECTS_DIR / "05_predictive_analytics/outputs/metrics.json")
                       .read_text())["forecast"]["total"]
    app.segmented_control(key="fc_level").set_value(50).run()
    text = " ".join(m.value for m in app.markdown)
    assert fmt_value(total["lower_50"], "usd").replace("$", "\\$") in text


def test_dashboard_displays_whatever_the_saved_outputs_say(dashboard_projects, monkeypatch):
    """Editing a saved output changes the dashboard: nothing is hard-coded."""
    profile = dashboard_projects / "00_foundation/outputs/data_profile.json"
    data = json.loads(profile.read_text())
    data["headline"]["net_revenue"] = 1234567.0
    profile.write_text(json.dumps(data))
    retention = dashboard_projects / "02_retention/outputs/metrics.json"
    data = json.loads(retention.read_text())
    for model in data["models"]:
        if model["champion"]:
            model["holdout_base_rate"] = 0.123
    retention.write_text(json.dumps(data))

    monkeypatch.setenv(PROJECTS_ENV, str(dashboard_projects))
    at = AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT).run()
    _no_errors(at)
    shown = _metrics(at)
    assert shown["Net revenue, 24 months"] == "$1.23M"
    assert shown["90-day churn rate"] == "12.3%"


def test_missing_section_outputs_degrade_gracefully(dashboard_projects, monkeypatch):
    for f in (dashboard_projects / "02_retention/outputs").iterdir():
        f.unlink()
    monkeypatch.setenv(PROJECTS_ENV, str(dashboard_projects))
    at = AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT).run()
    _no_errors(at)
    assert _metrics(at)["90-day churn rate"] == "n/a"
    assert any("missing" in w.value for w in at.sidebar.warning)
    _goto(at, "Retention")
    _no_errors(at)
    assert any("northstar retention" in w.value for w in at.warning)
    for page in PAGES:  # other pages keep working
        _goto(at, page)
        _no_errors(at)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_dashboard_command_line():
    cmd = dashboard_command(8123, headless=True)
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert cmd[4] == str(APP_PATH)
    assert cmd[cmd.index("--server.port") + 1] == "8123"
    assert cmd[cmd.index("--server.headless") + 1] == "true"


def test_dashboard_launches_headlessly_as_a_server():
    """`northstar dashboard --headless` starts a real Streamlit server that answers requests."""
    port = _free_port()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "northstar", "dashboard", "--headless",
                             "--port", str(port)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        deadline, health = time.time() + 60, None
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health",
                                            timeout=2) as r:
                    health = r.read().decode()
                    break
            except OSError:
                time.sleep(0.5)
        assert health == "ok", proc.stdout.read() if proc.poll() is not None else "timeout"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert r.status == 200
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
