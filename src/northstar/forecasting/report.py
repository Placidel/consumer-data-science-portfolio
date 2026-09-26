"""Section 05 pipeline: aggregate revenue, backtest the forecasters on rolling origins, audit for
leakage, calibrate empirical prediction intervals, issue the forward forecast, and write tables,
figures and the README results block.

Everything reported in ``projects/05_predictive_analytics/README.md`` between the generated-block
markers is rendered from ``outputs/metrics.json`` by :func:`render_markdown`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from northstar.acquisition.report import LeakageError
from northstar.forecasting import evaluation as ev
from northstar.forecasting import models as fm
from northstar.forecasting.series import (
    WEEK_DAYS,
    build_series,
    complete_weeks_end,
    daily_revenue,
    history_before,
    promo_calendar,
    repeat_promotions,
    to_weeks,
    weekly_revenue,
)
from northstar.profile import (
    GRID,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _style,
    update_generated_block,
)

__all__ = ["ForecastConfig", "LeakageError", "render_markdown", "run_analysis", "write_outputs"]

BEGIN_MARKER = "<!-- BEGIN GENERATED: forecast-results -->"
END_MARKER = "<!-- END GENERATED: forecast-results -->"
PEAK_MONTHS = (11, 12)  # target weeks starting in Nov-Dec: the holiday peak
DM_LEADS = (1, 4, 8, 13)

# Reference categorical palette in fixed slot order; realized values in primary ink.
COLORS = {"harmonic": "#2a78d6", "naive_4wk": "#eb6834", "seasonal_naive_yoy": "#1baf7a",
          "harmonic_no_promo": "#eda100", "actual": TEXT_PRIMARY}
BUCKET_COLORS = {"weeks 1-4": "#9ec5f4", "weeks 5-8": "#5a9be8", "weeks 9-13": "#1c5aa8"}


@dataclass(frozen=True)
class ForecastConfig:
    plan: ev.BacktestPlan = field(default_factory=ev.BacktestPlan)
    harmonic: fm.HarmonicConfig = field(default_factory=fm.HarmonicConfig)
    history_weeks_in_plot: int = 52


# ---------------------------------------------------------------- helpers
def _clean(obj):
    """JSON-safe copy with floats rounded (stable diffs; NaN -> None)."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer | bool | np.bool_):
        return obj.item() if isinstance(obj, np.generic) else obj
    if isinstance(obj, float | np.floating):
        return None if np.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, pd.Timestamp | np.datetime64):
        return str(pd.Timestamp(obj).date())
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


# ---------------------------------------------------------------- leakage audit
def leakage_audit(tables: Mapping[str, pd.DataFrame], daily: pd.Series, promo: pd.Series,
                  backtest: pd.DataFrame, intervals: list[pd.DataFrame], plan: ev.BacktestPlan,
                  config: fm.HarmonicConfig) -> dict:
    """Runtime checks that no forecast, feature or interval uses information from its future.

    The central check rebuilds the revenue series from **orders truncated at each evaluation
    origin** and requires every model to reproduce its backtest forecast exactly.
    """
    orders = tables["orders"]
    evaluation = backtest.loc[backtest["role"] == "evaluation"]
    mismatches, last_train_gap = [], []
    for origin, rows in evaluation.groupby("origin", sort=True):
        truncated = daily_revenue(orders.loc[orders["order_ts"] < origin], end=origin)
        for name, g in rows.groupby("model", sort=False):
            fc = ev.weekly_forecast(name, truncated, promo, origin, plan.horizon_weeks, config)
            if not np.allclose(fc, g.sort_values("horizon_week")["forecast"], rtol=1e-9,
                               equal_nan=True):
                mismatches.append(f"{name}@{origin.date()}")
        fit = fm.fit_harmonic(history_before(daily, origin), promo, config)
        last_train_gap.append((origin - fit.last_train_day).days)

    # The model interface itself must refuse a history that reaches the origin.
    probe = evaluation["origin"].min()
    leaky = daily.loc[daily.index <= probe]
    refused = []
    for name in fm.MODEL_NAMES:
        try:
            fm.forecast(name, leaky, probe, plan.horizon_days, promo, config)
            refused.append(False)
        except fm.HistoryError:
            refused.append(True)

    interval_late = sum(int((iv["calibration_last_observed"] > iv["origin"]).sum())
                        for iv in intervals)
    target_end = backtest["week_start"].max() + pd.Timedelta(days=WEEK_DAYS)
    design_end = pd.Timestamp(plan.design_last_origin) + pd.Timedelta(days=plan.horizon_days)
    checks = {
        "forecasts_identical_when_rebuilt_from_orders_before_origin": not mismatches,
        "models_refuse_history_on_or_after_origin": all(refused),
        "harmonic_fit_only_on_days_before_origin": min(last_train_gap) >= 1,
        "seasonal_naive_looks_back_a_full_season": plan.horizon_days <= fm.YEAR_DAYS,
        "interval_errors_observed_before_origin": interval_late == 0,
        "design_targets_end_before_evaluation": design_end <= pd.Timestamp(
            plan.first_evaluation_origin),
        "targets_are_complete_observed_weeks": bool(target_end <= complete_weeks_end(daily)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "origins_rebuilt_from_truncated_orders": int(evaluation["origin"].nunique()),
            "forecast_mismatches": mismatches[:10],
            "min_days_between_last_training_day_and_origin": int(min(last_train_gap)),
            "interval_rows_using_unobserved_errors": interval_late,
            "last_design_target_day": str((design_end - pd.Timedelta(days=1)).date()),
            "first_evaluation_origin": plan.first_evaluation_origin,
        },
    }


# ---------------------------------------------------------------- analysis
def _series_summary(daily: pd.Series, weekly: pd.DataFrame) -> dict:
    end = complete_weeks_end(daily)
    last13 = weekly.tail(13)["net_revenue"]
    prior = weekly.loc[weekly["week_start"] < weekly["week_start"].iloc[-1] - pd.Timedelta(
        weeks=51)].tail(13)["net_revenue"]
    return {
        "first_day": daily.index[0], "last_day": daily.index[-1],
        "days": len(daily), "complete_weeks": len(weekly),
        "first_week": weekly["week_start"].iloc[0], "last_week": weekly["week_start"].iloc[-1],
        "partial_days_dropped": int((daily.index >= end).sum()),
        "total_revenue": float(daily.sum()),
        "last_13_weeks_revenue": float(last13.sum()),
        "same_13_weeks_prior_year_revenue": float(prior.sum()),
        "mean_daily_revenue_by_weekday": {
            d: float(v) for d, v in zip(
                ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
                daily.loc[daily.index >= daily.index[-1] - pd.Timedelta(days=363)]
                .groupby(lambda x: x.dayofweek).mean(), strict=True)},
    }


def _with_skill(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Add MAE skill relative to the naive run rate (1 - MAE / naive MAE)."""
    naive = table.loc[table["model"] == "naive_4wk", [*keys, "mae"]].rename(
        columns={"mae": "_naive"})
    out = (table.merge(naive, on=keys, how="left") if keys
           else table.assign(_naive=naive["_naive"].iloc[0]))
    return out.assign(skill_vs_naive=1 - out["mae"] / out["_naive"]).drop(columns="_naive")


def _model_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_o=df["model"].map(fm.MODEL_NAMES.index)).sort_values(
        ["_o", *[c for c in ("bucket", "horizon_week") if c in df]]).drop(columns="_o")


def _dm_table(backtest: pd.DataFrame, totals: pd.DataFrame, plan: ev.BacktestPlan) -> pd.DataFrame:
    ev_rows = backtest.loc[backtest["role"] == "evaluation"]
    ev_tot = totals.loc[totals["role"] == "evaluation"]
    rows = []
    for ref in [m for m in fm.MODEL_NAMES if m != fm.CHAMPION]:
        targets = [(f"week {k}", k, ev_rows.loc[ev_rows["horizon_week"] == k]) for k in DM_LEADS]
        targets.append((f"{plan.horizon_weeks}-week total", plan.horizon_weeks, ev_tot))
        for label, lead, g in targets:
            a = g.loc[g["model"] == fm.CHAMPION].sort_values("origin")
            b = g.loc[g["model"] == ref].sort_values("origin")
            res = ev.diebold_mariano((a["forecast"] - a["actual"]).to_numpy(),
                                     (b["forecast"] - b["actual"]).to_numpy(), lead)
            rows.append({"reference": ref, "target": label, **res,
                         "champion_mae": float((a["forecast"] - a["actual"]).abs().mean()),
                         "reference_mae": float((b["forecast"] - b["actual"]).abs().mean()),
                         "champion_wins_share": float(
                             ((a["forecast"] - a["actual"]).abs().to_numpy()
                              < (b["forecast"] - b["actual"]).abs().to_numpy()).mean())})
    return pd.DataFrame(rows)


def run_analysis(tables: Mapping[str, pd.DataFrame], config: ForecastConfig | None = None
                 ) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Run the full section 05 analysis; returns (metrics dict, output tables)."""
    config = config or ForecastConfig()
    plan, hcfg = config.plan, config.harmonic
    daily, promos = build_series(tables)
    weekly = weekly_revenue(daily)
    promo = promo_calendar(promos)

    backtest = ev.run_backtest(daily, promo, plan, config=hcfg)
    totals = ev.quarter_totals(backtest)
    weekly_iv = ev.add_weekly_intervals(backtest, fm.CHAMPION, plan)
    total_iv = ev.add_total_intervals(totals, fm.CHAMPION, plan)

    audit = leakage_audit(tables, daily, promo, backtest, [weekly_iv, total_iv], plan, hcfg)
    if not audit["passed"]:
        failed = [k for k, ok in audit["checks"].items() if not ok]
        raise LeakageError(f"Leakage audit failed: {failed}; details: {audit['details']}")

    # ---- accuracy on the evaluation origins (scored once) and the design origins
    evaluation = backtest.loc[backtest["role"] == "evaluation"].assign(
        bucket=lambda d: d["horizon_week"].map(ev.bucket_label))
    overall = _with_skill(ev.metrics_table(evaluation, ["model"]), [])
    by_bucket = _with_skill(ev.metrics_table(evaluation, ["model", "bucket"]), ["bucket"])
    by_lead = _with_skill(ev.metrics_table(evaluation, ["model", "horizon_week"]),
                          ["horizon_week"])
    design = backtest.loc[backtest["role"] == "design"].dropna(subset=["forecast"])
    design_overall = ev.metrics_table(design, ["model"])
    ev_totals = totals.loc[totals["role"] == "evaluation"]
    total_summary = pd.DataFrame([
        {"model": m, "origins": len(g), "mape": float(g["pct_error"].abs().mean()),
         "median_ape": float(g["pct_error"].abs().median()),
         "max_ape": float(g["pct_error"].abs().max()),
         "bias": float(g["forecast"].sum() / g["actual"].sum() - 1)}
        for m, g in ev_totals.groupby("model", sort=False)])
    comparisons = _dm_table(backtest, totals, plan)

    # ---- uncertainty: coverage of the empirical intervals on the evaluation origins
    ev_iv = weekly_iv.loc[weekly_iv["role"] == "evaluation"].assign(
        season=lambda d: np.where(d["week_start"].dt.month.isin(PEAK_MONTHS),
                                  "Nov-Dec peak weeks", "other weeks"),
        bucket=lambda d: d["horizon_week"].map(ev.bucket_label))
    cov_all = ev.coverage_table(ev_iv, plan.levels)
    cov_bucket = ev.coverage_table(ev_iv, plan.levels, by="bucket")
    cov_season = ev.coverage_table(ev_iv, plan.levels, by="season")
    cov_lead = ev.coverage_table(ev_iv, plan.levels, by="horizon_week")
    cov_total = ev.coverage_table(total_iv.loc[total_iv["role"] == "evaluation"], plan.levels)

    # ---- forward forecast: issued on the Monday after the last complete week
    origin = complete_weeks_end(daily)
    horizon_end = origin + pd.Timedelta(days=plan.horizon_days)
    live_promos = repeat_promotions(promos, until=horizon_end)
    live_promo = promo_calendar(live_promos, end=horizon_end)
    k = np.arange(1, plan.horizon_weeks + 1)
    live = pd.DataFrame({"week_start": origin + pd.to_timedelta(7 * (k - 1), unit="D"),
                         "horizon_week": k})
    for name in fm.MODEL_NAMES:
        live[name] = ev.weekly_forecast(name, daily, live_promo, origin, plan.horizon_weeks, hcfg)
    past = backtest.loc[backtest["model"] == fm.CHAMPION]
    live_iv, live_total_iv = ev.forward_intervals(
        live[fm.CHAMPION].to_numpy(), origin, past, totals.loc[totals["model"] == fm.CHAMPION],
        plan)
    bounds = [c for c in live_iv.columns if c.startswith(("lower_", "upper_"))]
    live = pd.concat([live, live_iv[[*bounds, "calibration_n"]]], axis=1)
    last_year = daily.loc[origin - pd.Timedelta(days=fm.YEAR_DAYS):
                          horizon_end - pd.Timedelta(days=fm.YEAR_DAYS + 1)]
    live["same_week_last_year"] = to_weeks(last_year.to_numpy(), plan.horizon_weeks)
    live_fit = fm.fit_harmonic(history_before(daily, origin), live_promo, hcfg)
    upcoming = live_promos.loc[(live_promos["end_date"] >= origin)
                               & (live_promos["start_date"] < horizon_end)]

    metrics = {
        "config": {"plan": plan.as_dict(), "harmonic": hcfg.as_dict(),
                   "models": list(fm.MODEL_NAMES), "baselines": list(fm.BASELINES),
                   "champion": fm.CHAMPION, "horizon_days": plan.horizon_days,
                   "peak_months": list(PEAK_MONTHS)},
        "series": _series_summary(daily, weekly),
        "promotions": _records(live_promos.assign(
            days=(live_promos["end_date"] - live_promos["start_date"]).dt.days + 1)
            [["campaign_name", "start_date", "end_date", "days", "assumed"]]),
        "backtest": {
            "origins": int(backtest["origin"].nunique()),
            "roles": [{"role": r, "origins": int(g["origin"].nunique()),
                       "first_origin": g["origin"].min(), "last_origin": g["origin"].max(),
                       "last_target_day": g["week_start"].max() + pd.Timedelta(days=6)}
                      for r, g in backtest.groupby("role", sort=False)],
        },
        "leakage_audit": audit,
        "design_period": _records(_model_order(design_overall)),
        "evaluation": {
            "overall": _records(_model_order(overall)),
            "by_bucket": _records(_model_order(by_bucket)),
            "totals": _records(_model_order(total_summary)),
            "comparisons": _records(comparisons),
        },
        "intervals": {
            "method": "empirical quantiles of actual/forecast over the most recent "
                      f"{plan.calibration_origins} past origins at the same lead whose outcome "
                      "was observed before the origin",
            "overall": _records(cov_all), "by_bucket": _records(cov_bucket),
            "by_season": _records(cov_season), "total": _records(cov_total),
        },
        "forecast": {
            "origin": origin, "last_history_day": origin - pd.Timedelta(days=1),
            "horizon_end": horizon_end - pd.Timedelta(days=1),
            "weeks": _records(live),
            "total": {"forecast": float(live[fm.CHAMPION].sum()),
                      **{c: float(live_total_iv[c].iloc[0]) for c in bounds},
                      "calibration_n": int(live_total_iv["calibration_n"].iloc[0]),
                      "same_weeks_last_year": float(live["same_week_last_year"].sum()),
                      **{f"{m}_total": float(live[m].sum()) for m in fm.MODEL_NAMES}},
            "promotions_in_horizon": _records(upcoming[["campaign_name", "start_date",
                                                        "end_date", "assumed"]]),
            "harmonic_fit": {"first_train_day": live_fit.first_train_day,
                             "last_train_day": live_fit.last_train_day,
                             "annual_terms": live_fit.annual,
                             "trend_growth_per_year_at_origin": float(
                                 np.expm1(live_fit.slope_per_year)),
                             "level_correction": live_fit.level_correction,
                             "promotion_effect": float(np.expm1(live_fit.coef[
                                 2 + len(live_fit.knots) + 6]))},
        },
    }
    out = {
        "weekly_revenue": weekly,
        # Interval columns are filled for the champion only.
        "backtest_forecasts": backtest.merge(
            weekly_iv[["origin", "model", "horizon_week", *bounds, "calibration_n"]],
            on=["origin", "model", "horizon_week"], how="left"),
        "backtest_quarter_totals": totals.merge(
            total_iv[["origin", "model", *bounds, "calibration_n"]], on=["origin", "model"],
            how="left"),
        "accuracy_by_horizon": _model_order(pd.concat([
            overall.assign(bucket="all weeks"), by_bucket])),
        "accuracy_by_lead_week": _model_order(by_lead),
        "model_comparison_tests": comparisons,
        "interval_coverage": pd.concat([
            cov_all.assign(group="all lead weeks"),
            cov_bucket.rename(columns={"bucket": "group"}),
            cov_season.rename(columns={"season": "group"}),
            cov_lead.assign(group=lambda d: "week " + d["horizon_week"].astype(str)).drop(
                columns="horizon_week"),
            cov_total.assign(group=f"{plan.horizon_weeks}-week total")], ignore_index=True),
        "forecast": live,
    }
    return _clean(metrics), out


# ---------------------------------------------------------------- figures
def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def _axes(ax: plt.Axes, x: str, y: str) -> None:
    ax.set_xlabel(x, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(y, color=TEXT_SECONDARY, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(ax: plt.Axes, **kw) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, **kw)


def save_figures(metrics: Mapping, out: Mapping[str, pd.DataFrame], fig_dir: Path,
                 history_weeks: int = 52) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    k_max = metrics["config"]["plan"]["horizon_weeks"]
    lv = [round(v * 100) for v in metrics["config"]["plan"]["levels"]]

    # 1. Forward forecast fan chart.
    weekly, live = out["weekly_revenue"].tail(history_weeks), out["forecast"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=120)
    ax.plot(weekly["week_start"], weekly["net_revenue"] / 1000, color=COLORS["actual"],
            linewidth=2, label="Actual weekly net revenue")
    for level, alpha in zip(sorted(lv, reverse=True), (0.16, 0.32), strict=False):
        ax.fill_between(live["week_start"], live[f"lower_{level}"] / 1000,
                        live[f"upper_{level}"] / 1000, color=COLORS["harmonic"], alpha=alpha,
                        linewidth=0, label=f"{level}% range from past forecast errors")
    ax.plot(live["week_start"], live["harmonic"] / 1000, color=COLORS["harmonic"], linewidth=2,
            marker="o", markersize=4, label=fm.SHORT_LABELS["harmonic"])
    ax.plot(live["week_start"], live["naive_4wk"] / 1000, color=COLORS["naive_4wk"],
            linewidth=1.6, linestyle="--", label=fm.SHORT_LABELS["naive_4wk"] + " (baseline)")
    ax.plot(live["week_start"], live["same_week_last_year"] / 1000, color=TEXT_SECONDARY,
            linewidth=1.2, linestyle=":", label="Same weeks last year (actual)")
    ax.axvline(pd.Timestamp(metrics["forecast"]["origin"]), color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_ylim(bottom=0)
    _axes(ax, "Week starting (Monday)", "Weekly net revenue ($K)")
    _legend(ax, loc="upper left")
    _style(ax, f"Next {k_max} weeks: weekly net revenue forecast",
           f"Issued {metrics['forecast']['origin']} from data before that day; shaded ranges are "
           "empirical, not guaranteed bounds")
    paths.append(_finish(fig, fig_dir / "forecast_fan.png"))

    # 2. Backtest tracks at a short and the longest lead.
    bt = out["backtest_forecasts"]
    bt = bt.loc[bt["role"] == "evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=120)
    top = bt.loc[bt["horizon_week"].isin([1, k_max]), ["actual", "forecast"]].max().max()
    for ax, lead in zip(axes, (1, k_max), strict=True):
        g = bt.loc[bt["horizon_week"] == lead]
        actual = g.loc[g["model"] == fm.CHAMPION].sort_values("week_start")
        ax.plot(actual["week_start"], actual["actual"] / 1000, color=COLORS["actual"],
                linewidth=2.2, label="Actual")
        for name in fm.MODEL_NAMES:
            s = g.loc[g["model"] == name].sort_values("week_start")
            ax.plot(s["week_start"], s["forecast"] / 1000, color=COLORS[name],
                    linewidth=2 if name == fm.CHAMPION else 1.4, label=fm.SHORT_LABELS[name])
        ax.set_ylim(0, top / 1000 * 1.05)  # same scale in both panels
        _axes(ax, "Target week", "Weekly net revenue ($K)" if lead == 1 else "")
        ax.tick_params(axis="x", labelrotation=30)
        _style(ax, f"Forecast made {lead} week{'s' if lead > 1 else ''} ahead",
               "Evaluation origins; each point is a different forecast")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=8, labelcolor=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(fig_dir / "backtest_tracks.png", metadata={"Software": None})
    plt.close(fig)
    paths.append(fig_dir / "backtest_tracks.png")

    # 3. Error growth with lead time.
    lead = out["accuracy_by_lead_week"]
    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=120)
    for name in fm.MODEL_NAMES:
        s = lead.loc[lead["model"] == name]
        ax.plot(s["horizon_week"], s["wape"] * 100, color=COLORS[name], marker="o",
                markersize=4, linewidth=2.2 if name == fm.CHAMPION else 1.6,
                label=fm.MODEL_LABELS[name])
    ax.set_xticks(range(1, k_max + 1))
    ax.set_ylim(bottom=0)
    _axes(ax, "Lead time (weeks ahead)", "WAPE (%)")
    _legend(ax, loc="upper left")
    _style(ax, "Forecast error by lead time",
           "Weighted absolute percentage error of weekly revenue, evaluation origins")
    paths.append(_finish(fig, fig_dir / "error_by_horizon.png"))

    # 4. Quarter (13-week) totals by origin.
    tot = out["backtest_quarter_totals"]
    tot = tot.loc[tot["role"] == "evaluation"]
    champ = tot.loc[tot["model"] == fm.CHAMPION].sort_values("origin")
    wide = max(lv)
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=120)
    ax.fill_between(champ["origin"], champ[f"lower_{wide}"] / 1e6, champ[f"upper_{wide}"] / 1e6,
                    color=COLORS["harmonic"], alpha=0.16, linewidth=0,
                    label=f"{wide}% empirical range (harmonic)")
    ax.plot(champ["origin"], champ["actual"] / 1e6, color=COLORS["actual"], linewidth=2.2,
            label=f"Actual {k_max}-week revenue")
    for name in fm.MODEL_NAMES:
        s = tot.loc[tot["model"] == name].sort_values("origin")
        ax.plot(s["origin"], s["forecast"] / 1e6, color=COLORS[name],
                linewidth=2 if name == fm.CHAMPION else 1.4, label=fm.SHORT_LABELS[name])
    ax.set_ylim(bottom=0)
    _axes(ax, "Forecast origin (Monday)", f"Revenue over the next {k_max} weeks ($M)")
    _legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _style(ax, f"{k_max}-week revenue: forecast vs. actual",
           "One forecast per weekly origin; evaluation period")
    paths.append(_finish(fig, fig_dir / "quarter_totals.png"))

    # 5. Diagnostics: signed error over time by lead bucket; interval coverage by lead week.
    bt_c = bt.loc[bt["model"] == fm.CHAMPION].assign(
        pct=lambda d: (d["forecast"] / d["actual"] - 1) * 100,
        bucket=lambda d: d["horizon_week"].map(ev.bucket_label))
    cov = out["interval_coverage"]
    cov = cov.loc[cov["group"].str.match(r"^week \d+$")].assign(
        lead=lambda d: d["group"].str.slice(5).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for bucket, color in BUCKET_COLORS.items():
        s = bt_c.loc[bt_c["bucket"] == bucket]
        ax.scatter(s["week_start"], s["pct"], s=14, color=color, label=f"Lead {bucket}",
                   edgecolors=SURFACE, linewidths=0.5)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    _axes(ax, "Target week", "Forecast error (% of actual)")
    ax.tick_params(axis="x", labelrotation=30)
    _legend(ax, loc="lower left")
    _style(ax, "Harmonic regression: signed error by target week",
           "Positive = over-forecast; evaluation origins")
    ax = axes[1]
    for level, style in zip(sorted(lv), ("--", "-"), strict=False):
        s = cov.loc[np.isclose(cov["level"] * 100, level)].sort_values("lead")
        ax.plot(s["lead"], s["coverage"] * 100, color=COLORS["harmonic"], linestyle=style,
                marker="o", markersize=4, label=f"{level}% range: observed coverage")
        ax.axhline(level, color=TEXT_SECONDARY, linestyle=style, linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, k_max + 1, 2))
    _axes(ax, "Lead time (weeks ahead)", "Actuals inside the range (%)")
    _legend(ax, loc="lower left")
    _style(ax, "Interval coverage vs. nominal", "Gray lines: nominal level")
    paths.append(_finish(fig, fig_dir / "residual_diagnostics.png"))
    return paths


# ---------------------------------------------------------------- README rendering
def _pct(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def _spct(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v * 100:+.{digits}f}%"


def _usd(v: float | None) -> str:
    if v is None:
        return "-"
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def _p(v: float | None) -> str:
    if v is None:
        return "-"
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def render_markdown(metrics: Mapping) -> str:
    cfg = metrics["config"]
    plan = cfg["plan"]
    kmax = plan["horizon_weeks"]
    levels = [round(v * 100) for v in plan["levels"]]
    s = metrics["series"]
    data = metrics.get("data")
    source = f"Data seed `{data['seed']}`, {data['n_prospects']:,} prospects. " if data else ""
    lines = [
        f"_{source}Rendered from `outputs/metrics.json`. Target = weekly net order revenue "
        f"(Monday-Sunday), forecast {kmax} weeks ahead from every Monday origin._",
        "",
        f"**Series.** {s['days']:,} days ({s['first_day']} to {s['last_day']}), "
        f"{s['complete_weeks']} complete weeks ({s['first_week']} to {s['last_week']}); "
        f"{s['partial_days_dropped']} days of the final partial week are excluded from weekly "
        f"totals. Total net revenue {_usd(s['total_revenue'])}. Last {kmax} complete weeks: "
        f"{_usd(s['last_13_weeks_revenue'])} vs. {_usd(s['same_13_weeks_prior_year_revenue'])} "
        "in the same weeks a year earlier.",
        "",
        "**Rolling-origin backtest** (every model refit on an expanding window at every origin)",
        "",
        "| Role | Origins | First origin | Last origin | Last target day | Used for |",
        "|---|---:|---|---|---|---|",
    ]
    used = {"design": "choosing the harmonic model's settings",
            "calibration": "interval track record only (not scored)",
            "evaluation": "**scored once**: all accuracy and coverage below"}
    for r in metrics["backtest"]["roles"]:
        lines.append(f"| {r['role']} | {r['origins']} | {r['first_origin']} | {r['last_origin']} "
                     f"| {r['last_target_day']} | {used[r['role']]} |")
    lines += [
        "",
        "**Leakage audit** (the pipeline refuses to write results if any check fails)",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    audit = metrics["leakage_audit"]
    for name, ok in audit["checks"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'pass' if ok else 'FAIL'} |")
    d = audit["details"]
    lines += [
        "",
        f"{d['origins_rebuilt_from_truncated_orders']} evaluation origins were re-forecast from "
        "orders truncated at the origin; forecast mismatches: "
        f"{len(d['forecast_mismatches'])}. Last design target day {d['last_design_target_day']};"
        f" first evaluation origin {d['first_evaluation_origin']}.",
        "",
        "**Accuracy on the evaluation origins** (weekly revenue; skill = 1 - MAE / naive MAE)",
        "",
        "| Model | Weekly MAE | RMSE | WAPE | sMAPE | Bias (total) | Skill vs. naive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["evaluation"]["overall"]:
        label = fm.MODEL_LABELS[r["model"]]
        label = f"**{label}**" if r["model"] == cfg["champion"] else label
        lines.append(f"| {label} | {_usd(r['mae'])} | {_usd(r['rmse'])} | {_pct(r['wape'])} | "
                     f"{_pct(r['smape'])} | {_spct(r['bias'])} | {_spct(r['skill_vs_naive'])} |")
    buckets = list(dict.fromkeys(r["bucket"] for r in metrics["evaluation"]["by_bucket"]))
    lines += [
        "",
        "WAPE by lead time (MAE skill vs. naive in parentheses):",
        "",
        "| Model | " + " | ".join(f"Lead {b}" for b in buckets) + " |",
        "|---|" + "---:|" * len(buckets),
    ]
    by_bucket = {(r["model"], r["bucket"]): r for r in metrics["evaluation"]["by_bucket"]}
    for name in cfg["models"]:
        cells = " | ".join(f"{_pct(by_bucket[(name, b)]['wape'])} "
                           f"({_spct(by_bucket[(name, b)]['skill_vs_naive'], 0)})"
                           for b in buckets)
        lines.append(f"| {fm.MODEL_LABELS[name]} | {cells} |")
    lines += [
        "",
        f"{kmax}-week (quarter) total, one forecast per origin:",
        "",
        "| Model | Origins | Mean abs. % error | Median | Worst | Bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["evaluation"]["totals"]:
        lines.append(f"| {fm.MODEL_LABELS[r['model']]} | {r['origins']} | {_pct(r['mape'])} | "
                     f"{_pct(r['median_ape'])} | {_pct(r['max_ape'])} | {_spct(r['bias'])} |")
    lines += [
        "",
        f"Development-period accuracy ({metrics['backtest']['roles'][0]['origins']} design "
        "origins, used to fix the harmonic settings; the seasonal naive is not yet defined):",
        "",
        "| Model | Weekly MAE | WAPE | Bias (total) |",
        "|---|---:|---:|---:|",
        *[f"| {fm.MODEL_LABELS[r['model']]} | {_usd(r['mae'])} | {_pct(r['wape'])} | "
          f"{_spct(r['bias'])} |" for r in metrics["design_period"]],
        "",
        f"**Is the difference real?** Diebold-Mariano tests of equal mean absolute error, "
        f"{fm.MODEL_LABELS[cfg['champion']]} vs. each alternative (Newey-West variance for "
        "overlapping horizons, Harvey-Leybourne-Newbold correction; negative difference = "
        "harmonic more accurate):",
        "",
        "| Alternative | Target | Origins | MAE harmonic | MAE alternative | Mean difference | "
        "Harmonic wins | DM statistic | p-value |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["evaluation"]["comparisons"]:
        stat = "-" if r["dm_stat"] is None else f"{r['dm_stat']:+.2f}"
        lines.append(f"| {fm.SHORT_LABELS[r['reference']]} | {r['target']} | {r['n']} | "
                     f"{_usd(r['champion_mae'])} | {_usd(r['reference_mae'])} | "
                     f"{_usd(r['mean_loss_diff'])} | {_pct(r['champion_wins_share'], 0)} | "
                     f"{stat} | {_p(r['p_value'])} |")
    iv = metrics["intervals"]
    lines += [
        "",
        f"**Uncertainty.** Ranges are {iv['method']}. They describe how wrong past forecasts "
        "were; they are **not guaranteed bounds**. Observed coverage on the evaluation origins:",
        "",
        "| Group | Nominal | Forecasts | Observed coverage | Actual below range | Actual above "
        "range | Mean width (% of forecast) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = ([("all lead weeks", r) for r in iv["overall"]]
              + [(f"lead {r['bucket']}", r) for r in iv["by_bucket"]]
              + [(r["season"], r) for r in iv["by_season"]]
              + [(f"{kmax}-week total", r) for r in iv["total"]])
    for label, r in groups:
        lines.append(f"| {label} | {r['level']:.0%} | {r['n']} | {_pct(r['coverage'])} | "
                     f"{_pct(r['below'])} | {_pct(r['above'])} | "
                     f"{_pct(r['mean_relative_width'])} |")

    f = metrics["forecast"]
    t = f["total"]
    lo, hi = min(levels), max(levels)
    fit = f["harmonic_fit"]
    lines += [
        "",
        f"**Forward forecast**, issued Monday {f['origin']} from data through "
        f"{f['last_history_day']}, for {f['origin']} to {f['horizon_end']}:",
        "",
        "| Week starting | Harmonic forecast | "
        + " | ".join(f"{lv}% range" for lv in levels)
        + " | Naive run rate | Last year × growth | Same week last year (actual) |",
        "|---|---:|" + "---:|" * len(levels) + "---:|---:|---:|",
    ]
    for w in f["weeks"]:
        ranges = " | ".join(f"{_usd(w[f'lower_{lv}'])} - {_usd(w[f'upper_{lv}'])}"
                            for lv in levels)
        lines.append(f"| {w['week_start']} | {_usd(w['harmonic'])} | {ranges} | "
                     f"{_usd(w['naive_4wk'])} | {_usd(w['seasonal_naive_yoy'])} | "
                     f"{_usd(w['same_week_last_year'])} |")
    ranges = " | ".join(f"{_usd(t[f'lower_{lv}'])} - {_usd(t[f'upper_{lv}'])}" for lv in levels)
    lines.append(f"| **{kmax}-week total** | **{_usd(t['forecast'])}** | {ranges} | "
                 f"{_usd(t['naive_4wk_total'])} | {_usd(t['seasonal_naive_yoy_total'])} | "
                 f"{_usd(t['same_weeks_last_year'])} |")
    promo_text = "; ".join(
        f"{p['campaign_name']} {p['start_date']} to {p['end_date']}"
        f"{' (**assumed**: repeats last year 52 weeks later)' if p['assumed'] else ''}"
        for p in f["promotions_in_horizon"]) or "none"
    lines += [
        "",
        f"- Harmonic total vs. the same {kmax} weeks last year: "
        f"{_spct(t['forecast'] / t['same_weeks_last_year'] - 1)}. The {lo}% range for the total "
        f"is {_usd(t[f'lower_{lo}'])} to {_usd(t[f'upper_{lo}'])} and the {hi}% range "
        f"{_usd(t[f'lower_{hi}'])} to {_usd(t[f'upper_{hi}'])} (from "
        f"{t['calibration_n']} past quarter forecasts).",
        f"- Promotions in the horizon: {promo_text}.",
        f"- Fitted model at the origin: trained on {fit['first_train_day']} to "
        f"{fit['last_train_day']}; annual terms {'on' if fit['annual_terms'] else 'off'}; "
        f"underlying trend growth {_spct(fit['trend_growth_per_year_at_origin'])} a year before "
        f"damping; estimated promotion-day effect {_spct(fit['promotion_effect'])}; level "
        f"correction {fit['level_correction']:+.3f} (log scale).",
    ]
    return "\n".join(lines)


def write_outputs(tables: Mapping[str, pd.DataFrame], out_dir: Path, readme: Path | None = None,
                  config: ForecastConfig | None = None, data_manifest: Mapping | None = None
                  ) -> dict:
    """Run the analysis and write metrics JSON, CSV tables, figures and the README block."""
    config = config or ForecastConfig()
    metrics, out = run_analysis(tables, config)
    if data_manifest is not None:
        metrics = {"data": {"seed": data_manifest["seed"],
                            "n_prospects": data_manifest["n_prospects"]}, **metrics}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name, df in out.items():
        numeric = df.select_dtypes("number").columns
        df.assign(**df[numeric].round(4)).to_csv(out_dir / f"{name}.csv", index=False,
                                                 date_format="%Y-%m-%d")
    save_figures(metrics, out, out_dir / "figures", config.history_weeks_in_plot)
    if readme is not None and readme.exists():
        update_generated_block(readme, render_markdown(metrics), BEGIN_MARKER, END_MARKER)
    return metrics

