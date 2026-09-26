"""Rolling-origin backtest, forecast accuracy metrics, empirical prediction intervals and the
Diebold-Mariano comparison.

Backtest design: a forecast is issued every Monday (the *origin*) for the next ``horizon_weeks``
Monday-Sunday weeks, using only days before the origin. Every model is refit at every origin on
an expanding window. Origins play one of three roles:

* ``design`` - early origins used to choose the harmonic model's settings. Their targets all end
  before the first evaluation origin.
* ``calibration`` - origins between design and evaluation. They are not scored and exist only so
  the interval method has a track record of errors.
* ``evaluation`` - scored once. The first evaluation origin is also the first date at which
  every baseline is defined.

Prediction intervals are **empirical**: at each origin, the interval for lead week ``k`` is the
point forecast times quantiles of ``actual / forecast`` over the most recent past forecasts at the
same lead whose outcome was already observed at the origin. They are a statement about past
forecast errors, not guaranteed bounds. Their coverage is checked on the evaluation origins.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from northstar.forecasting import models as fm
from northstar.forecasting.series import WEEK_DAYS, complete_weeks_end, history_before, to_weeks

HORIZON_BUCKETS = ((1, 4), (5, 8), (9, 13))


@dataclass(frozen=True)
class BacktestPlan:
    horizon_weeks: int = 13  # one fiscal quarter of weekly forecasts
    first_origin: str = "2024-07-01"
    design_last_origin: str = "2024-11-18"
    first_evaluation_origin: str = "2025-02-17"
    last_origin: str | None = None  # default: last origin whose full horizon is observed
    step_weeks: int = 1
    calibration_origins: int = 26  # rolling window of past forecasts used for the intervals
    min_calibration_origins: int = 12
    levels: tuple[float, ...] = (0.5, 0.8)

    @property
    def horizon_days(self) -> int:
        return self.horizon_weeks * WEEK_DAYS

    def origins(self, daily: pd.Series) -> list[pd.Timestamp]:
        last = (pd.Timestamp(self.last_origin) if self.last_origin is not None
                else complete_weeks_end(daily) - pd.Timedelta(days=self.horizon_days))
        origins = list(pd.date_range(self.first_origin, last, freq=f"{7 * self.step_weeks}D"))
        self.validate(origins)
        return origins

    def validate(self, origins: Sequence[pd.Timestamp]) -> None:
        if any(o.dayofweek != 0 for o in origins):
            raise ValueError("forecast origins must be Mondays")
        design_end = pd.Timestamp(self.design_last_origin) + pd.Timedelta(days=self.horizon_days)
        if design_end > pd.Timestamp(self.first_evaluation_origin):
            raise ValueError("design-period targets overlap the evaluation period")
        if not any(o >= pd.Timestamp(self.first_evaluation_origin) for o in origins):
            raise ValueError("no evaluation origins")

    def role(self, origin: pd.Timestamp) -> str:
        if origin <= pd.Timestamp(self.design_last_origin):
            return "design"
        if origin >= pd.Timestamp(self.first_evaluation_origin):
            return "evaluation"
        return "calibration"

    def as_dict(self) -> dict:
        return {**asdict(self), "levels": list(self.levels)}


# ---------------------------------------------------------------- backtest
def weekly_forecast(name: str, daily: pd.Series, promo: pd.Series, origin: pd.Timestamp,
                    horizon_weeks: int, config: fm.HarmonicConfig | None = None) -> np.ndarray:
    """Weekly totals for the ``horizon_weeks`` weeks starting at ``origin`` (a Monday)."""
    hist = history_before(daily, origin)
    path = fm.forecast(name, hist, origin, horizon_weeks * WEEK_DAYS, promo, config)
    return to_weeks(path, horizon_weeks)


def run_backtest(daily: pd.Series, promo: pd.Series, plan: BacktestPlan,
                 models: Iterable[str] = fm.MODEL_NAMES,
                 config: fm.HarmonicConfig | None = None) -> pd.DataFrame:
    """One row per (origin, model, lead week) with the forecast and the realized weekly total."""
    rows = []
    k = np.arange(1, plan.horizon_weeks + 1)
    for origin in plan.origins(daily):
        target = daily.loc[origin: origin + pd.Timedelta(days=plan.horizon_days - 1)]
        actual = to_weeks(target.to_numpy(), plan.horizon_weeks)
        week_start = origin + pd.to_timedelta(7 * (k - 1), unit="D")
        for name in models:
            fc = weekly_forecast(name, daily, promo, origin, plan.horizon_weeks, config)
            rows.append(pd.DataFrame({"origin": origin, "role": plan.role(origin), "model": name,
                                      "horizon_week": k, "week_start": week_start,
                                      "actual": actual, "forecast": fc}))
    return pd.concat(rows, ignore_index=True)


def quarter_totals(backtest: pd.DataFrame) -> pd.DataFrame:
    """Sum of the forecast and actual over the full horizon, per origin and model."""
    out = (backtest.groupby(["origin", "role", "model"], sort=False)[["actual", "forecast"]]
           .sum(min_count=1).reset_index())
    out["pct_error"] = out["forecast"] / out["actual"] - 1
    return out


# ---------------------------------------------------------------- point metrics
def point_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    """MAE, RMSE, WAPE, sMAPE and bias (total forecast / total actual - 1)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    if a.shape != f.shape or len(a) == 0:
        raise ValueError("actual and forecast must be non-empty and the same shape")
    if np.isnan(f).any() or np.isnan(a).any():
        raise ValueError("metrics require complete forecasts; drop undefined ones first")
    err = f - a
    denom = np.abs(a) + np.abs(f)
    smape = np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0.0)
    return {"n": len(a), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "wape": float(np.abs(err).sum() / np.abs(a).sum()),
            "smape": float(np.mean(smape)), "bias": float(f.sum() / a.sum() - 1)}


def metrics_table(backtest: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key, g in backtest.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key, strict=True)),
                     **point_metrics(g["actual"].to_numpy(), g["forecast"].to_numpy())})
    return pd.DataFrame(rows)


def bucket_label(horizon_week: int) -> str:
    for lo, hi in HORIZON_BUCKETS:
        if lo <= horizon_week <= hi:
            return f"weeks {lo}-{hi}"
    raise ValueError(f"lead week {horizon_week} outside {HORIZON_BUCKETS}")


# ---------------------------------------------------------------- intervals
def _quantile_names(levels: Iterable[float]) -> list[tuple[float, str, str]]:
    return [(lv, f"lower_{round(lv * 100)}", f"upper_{round(lv * 100)}") for lv in levels]


def _interval_frame(keys: pd.DataFrame, pool_origin: np.ndarray, pool_ready: np.ndarray,
                    pool_ratio: np.ndarray, plan: BacktestPlan) -> pd.DataFrame:
    """For each row in ``keys`` (origin, forecast), empirical ratio quantiles of the pool.

    ``pool_ready`` is the date by which each past error was observed; a row may only use errors
    with ``pool_ready <= origin``, and of those the ``plan.calibration_origins`` most recent.
    """
    order = np.argsort(pool_origin, kind="stable")
    pool_origin, pool_ready, pool_ratio = pool_origin[order], pool_ready[order], pool_ratio[order]
    out = {name: [] for _, lo, hi in _quantile_names(plan.levels) for name in (lo, hi)}
    out["calibration_n"], out["calibration_last_observed"] = [], []
    for origin, point in zip(keys["origin"], keys["forecast"], strict=True):
        usable = pool_ready <= origin
        ratios = pool_ratio[usable][-plan.calibration_origins:]
        ready = pool_ready[usable][-plan.calibration_origins:]
        ok = len(ratios) >= plan.min_calibration_origins
        for lv, lo, hi in _quantile_names(plan.levels):
            q = np.quantile(ratios, [(1 - lv) / 2, (1 + lv) / 2]) if ok else [np.nan, np.nan]
            out[lo].append(point * q[0])
            out[hi].append(point * q[1])
        out["calibration_n"].append(len(ratios))
        out["calibration_last_observed"].append(ready.max() if len(ready) else pd.NaT)
    return pd.DataFrame(out, index=keys.index)


def add_weekly_intervals(backtest: pd.DataFrame, model: str, plan: BacktestPlan) -> pd.DataFrame:
    """Rows of ``model`` with empirical intervals per lead week (see module docstring)."""
    rows = backtest.loc[backtest["model"] == model].copy()
    rows["observed_at"] = rows["week_start"] + pd.Timedelta(days=WEEK_DAYS)
    parts = []
    for _, g in rows.groupby("horizon_week", sort=True):
        parts.append(pd.concat([g, _interval_frame(
            g, g["origin"].to_numpy(), g["observed_at"].to_numpy(),
            (g["actual"] / g["forecast"]).to_numpy(), plan)], axis=1))
    return pd.concat(parts).sort_index()


def add_total_intervals(totals: pd.DataFrame, model: str, plan: BacktestPlan) -> pd.DataFrame:
    rows = totals.loc[totals["model"] == model].copy()
    rows["observed_at"] = rows["origin"] + pd.Timedelta(days=plan.horizon_days)
    return pd.concat([rows, _interval_frame(
        rows, rows["origin"].to_numpy(), rows["observed_at"].to_numpy(),
        (rows["actual"] / rows["forecast"]).to_numpy(), plan)], axis=1)


def forward_intervals(point: np.ndarray, origin: pd.Timestamp, past: pd.DataFrame,
                      past_totals: pd.DataFrame, plan: BacktestPlan
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intervals for a new forecast issued at ``origin`` from the backtest's error record.

    Returns (weekly intervals, interval for the horizon total).
    """
    k = np.arange(1, plan.horizon_weeks + 1)
    keys = pd.DataFrame({"origin": origin, "horizon_week": k, "forecast": point})
    parts = []
    for week in k:
        g = past.loc[past["horizon_week"] == week]
        parts.append(_interval_frame(
            keys.loc[keys["horizon_week"] == week], g["origin"].to_numpy(),
            (g["week_start"] + pd.Timedelta(days=WEEK_DAYS)).to_numpy(),
            (g["actual"] / g["forecast"]).to_numpy(), plan))
    weekly = pd.concat([keys, pd.concat(parts)], axis=1)
    total_key = pd.DataFrame({"origin": [origin], "forecast": [point.sum()]})
    total = pd.concat([total_key, _interval_frame(
        total_key, past_totals["origin"].to_numpy(),
        (past_totals["origin"] + pd.Timedelta(days=plan.horizon_days)).to_numpy(),
        (past_totals["actual"] / past_totals["forecast"]).to_numpy(), plan)], axis=1)
    return weekly, total


def coverage_table(rows: pd.DataFrame, levels: Iterable[float], by: str | None = None
                   ) -> pd.DataFrame:
    """Share of outcomes inside each interval and its mean width relative to the forecast."""
    groups = [(None, rows)] if by is None else list(rows.groupby(by, sort=True))
    out = []
    for key, g in groups:
        for lv, lo, hi in _quantile_names(levels):
            inside = (g["actual"] >= g[lo]) & (g["actual"] <= g[hi])
            out.append({**({by: key} if by else {}), "level": lv, "n": len(g),
                        "coverage": float(inside.mean()),
                        "below": float((g["actual"] < g[lo]).mean()),
                        "above": float((g["actual"] > g[hi]).mean()),
                        "mean_relative_width": float(((g[hi] - g[lo]) / g["forecast"]).mean())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- forecast comparison
def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, lead: int) -> dict:
    """Diebold-Mariano test of equal mean absolute error, with the Harvey-Leybourne-Newbold
    small-sample correction.

    ``errors_*`` are forecast errors from consecutive, equally spaced origins at the same
    ``lead`` (in origin steps). Overlapping horizons make the loss differential MA(lead - 1),
    so its long-run variance uses Newey-West (Bartlett) weights up to lag ``lead - 1``, which
    keeps the estimate positive. Negative ``mean_loss_diff`` means ``a`` is more accurate.
    """
    d = np.abs(np.asarray(errors_a, float)) - np.abs(np.asarray(errors_b, float))
    n = len(d)
    if n < 3 or lead < 1:
        raise ValueError("need at least three paired errors and a positive lead")
    dc = d - d.mean()
    var = dc @ dc / n
    for lag in range(1, min(lead, n)):
        var += 2 * (1 - lag / lead) * (dc[lag:] @ dc[:-lag]) / n
    if var <= 0:
        return {"n": n, "lead": lead, "mean_loss_diff": float(d.mean()), "dm_stat": None,
                "p_value": None}
    dm = d.mean() / np.sqrt(var / n)
    hln = np.sqrt(max((n + 1 - 2 * lead + lead * (lead - 1) / n) / n, 0.0))
    stat = dm * hln
    return {"n": n, "lead": lead, "mean_loss_diff": float(d.mean()), "dm_stat": float(stat),
            "p_value": float(2 * stats.t.sf(abs(stat), df=n - 1))}
