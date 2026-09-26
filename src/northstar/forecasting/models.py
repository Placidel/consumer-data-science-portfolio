"""Daily revenue forecasters behind one interface.

Every model is called as ``forecast(name, history, origin, horizon_days, promo)`` where
``history`` is daily revenue on days strictly before ``origin`` and ``promo`` is the planned
promotion calendar. :func:`check_history` enforces this contract, so no model can be handed a
day on or after the origin.

* ``naive_4wk`` (baseline): every future day equals the mean of the last 28 days. This is the
  "run-rate" forecast a planner would pencil in.
* ``seasonal_naive_yoy`` (baseline): the same day 52 weeks earlier, scaled by year-over-year
  growth of the last 28 days. This is the classic "last year plus growth" plan. It needs
  392 days of history, so it is undefined (NaN) before then.
* ``harmonic`` (the forecasting model): a regression on log daily revenue with a penalized
  piecewise-linear trend, annual Fourier terms, day-of-week effects and the planned promotion
  flag. The trend is extrapolated with damping, and the forecast is anchored to the level of the
  last four weeks (see :class:`HarmonicConfig`).
* ``harmonic_no_promo`` (ablation): the same model without the promotion calendar. It shows how
  much of the result rests on treating promotion dates as known in advance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from northstar.timeline import DATA_START

MODEL_NAMES = ("naive_4wk", "seasonal_naive_yoy", "harmonic_no_promo", "harmonic")
BASELINES = ("naive_4wk", "seasonal_naive_yoy")
CHAMPION = "harmonic"  # pre-registered: the model whose forecasts and intervals are published
MODEL_LABELS = {
    "naive_4wk": "Naive: trailing 4-week run rate (baseline)",
    "seasonal_naive_yoy": "Seasonal naive: last year × YoY growth (baseline)",
    "harmonic_no_promo": "Harmonic regression without promo calendar (ablation)",
    "harmonic": "Harmonic regression: damped trend + calendar",
}
SHORT_LABELS = {"naive_4wk": "Naive run rate", "seasonal_naive_yoy": "Last year × growth",
                "harmonic_no_promo": "Harmonic, no promos", "harmonic": "Harmonic regression"}
LEVEL_DAYS = 28
YEAR_DAYS = 364  # 52 weeks: keeps the weekday aligned for the seasonal naive


class HistoryError(ValueError):
    """Raised when a model is handed data it could not have had at the forecast origin."""


def check_history(history: pd.Series, origin: pd.Timestamp) -> None:
    """History must be contiguous daily data ending the day before ``origin``."""
    origin = pd.Timestamp(origin)
    if len(history) == 0:
        raise HistoryError("empty history")
    if history.index.max() >= origin:
        raise HistoryError(f"history contains {history.index.max().date()}, on or after the "
                           f"forecast origin {origin.date()}")
    if history.index[-1] != origin - pd.Timedelta(days=1):
        raise HistoryError(f"history ends {history.index[-1].date()}, not the day before the "
                           f"origin {origin.date()}")
    if len(history) != (history.index[-1] - history.index[0]).days + 1:
        raise HistoryError("history is not a contiguous daily series")


# ---------------------------------------------------------------- baselines
def naive_4wk(history: pd.Series, horizon_days: int) -> np.ndarray:
    return np.full(horizon_days, history.iloc[-LEVEL_DAYS:].mean())


def seasonal_naive_yoy(history: pd.Series, horizon_days: int) -> np.ndarray:
    """``y[d - 364] * mean(last 28 days) / mean(same 28 days a year earlier)``."""
    if horizon_days > YEAR_DAYS:
        raise ValueError("the seasonal naive cannot look further ahead than one season")
    values = history.to_numpy(float)
    if len(values) < YEAR_DAYS + LEVEL_DAYS:
        return np.full(horizon_days, np.nan)
    n = len(values)
    growth = values[-LEVEL_DAYS:].mean() / values[n - YEAR_DAYS - LEVEL_DAYS: n - YEAR_DAYS].mean()
    # Day origin + j maps to history position n + j - 364, always strictly before the origin.
    return values[n - YEAR_DAYS: n - YEAR_DAYS + horizon_days] * growth


# ---------------------------------------------------------------- harmonic regression
@dataclass(frozen=True)
class HarmonicConfig:
    """Design choices, fixed on the development origins before the evaluation was scored.

    Penalties are on the mean-squared-error scale of log revenue; time is measured in years.
    """

    burn_in_days: int = 28  # skip the launch month, when the customer base was a handful
    knot_spacing_days: int = 28  # the trend may change slope every four weeks ...
    min_knot_gap_days: int = 56  # ... but the final slope is estimated from >= 8 weeks
    trend_penalty: float = 0.01  # ridge penalty on slope changes (smooths the trend)
    fourier_order: int = 6  # annual harmonics: resolves features about two months wide
    fourier_penalty: float = 0.001
    min_annual_history_days: int = 365  # annual terms only once a full year is observed
    damping: float = 0.97  # per-day damping of the trend slope (half-life ~23 days)
    level_window_days: int = 28  # anchor: mean residual of the last four weeks
    smearing_window_days: int = 182  # retransformation (log -> dollars) variance window
    use_promotions: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HarmonicFit:
    config: HarmonicConfig
    coef: np.ndarray
    knots: np.ndarray  # in years since DATA_START
    annual: bool
    promotions: bool  # False when disabled or when no promotion day falls in the training data
    first_train_day: pd.Timestamp
    last_train_day: pd.Timestamp
    level_correction: float
    smearing: float
    slope_per_year: float
    trend_at_origin: float


def _years(days: pd.DatetimeIndex) -> np.ndarray:
    return (days - DATA_START).days.to_numpy() / 365.0


def _promo_flags(days: pd.DatetimeIndex, promo: pd.Series) -> np.ndarray:
    flag = promo.reindex(days)
    if flag.isna().any():
        raise ValueError("promotion calendar does not cover every day it is needed for")
    return flag.to_numpy(float)


def _calendar(days: pd.DatetimeIndex, promo: pd.Series, config: HarmonicConfig, annual: bool,
              promotions: bool) -> tuple[np.ndarray, np.ndarray]:
    """Known-in-advance features and their penalties: day of week, promotion, annual terms."""
    parts = [np.eye(7)[days.dayofweek.to_numpy()][:, 1:]]
    pen = [0.0] * 6
    if promotions:
        parts.append(_promo_flags(days, promo)[:, None])
        pen.append(0.0)
    if annual:
        doy = days.dayofyear.to_numpy() / 365.25
        parts.append(np.column_stack([f(2 * np.pi * k * doy)
                                      for k in range(1, config.fourier_order + 1)
                                      for f in (np.sin, np.cos)]))
        pen += [config.fourier_penalty] * (2 * config.fourier_order)
    return np.column_stack(parts), np.array(pen)


def fit_harmonic(history: pd.Series, promo: pd.Series, config: HarmonicConfig | None = None
                 ) -> HarmonicFit:
    config = config or HarmonicConfig()
    origin = history.index[-1] + pd.Timedelta(days=1)
    train = history.loc[history.index >= DATA_START + pd.Timedelta(days=config.burn_in_days)]
    if len(train) < 2 * LEVEL_DAYS:
        raise HistoryError("too little history to fit the harmonic regression")
    days = train.index
    z = np.log1p(train.to_numpy(float))
    t = _years(days)
    t_origin = (origin - DATA_START).days / 365.0
    first_knot = (train.index[0] - DATA_START).days + config.knot_spacing_days
    last_knot = (origin - DATA_START).days - config.min_knot_gap_days
    knots = np.arange(first_knot, last_knot + 1, config.knot_spacing_days) / 365.0
    annual = len(train) >= config.min_annual_history_days
    promotions = config.use_promotions and bool(_promo_flags(days, promo).any())
    hinges = np.maximum(t[:, None] - knots[None, :], 0.0)
    cal, cal_pen = _calendar(days, promo, config, annual, promotions)
    x = np.column_stack([np.ones(len(z)), t, hinges, cal])
    pen = np.concatenate([[0.0, 0.0], np.full(len(knots), config.trend_penalty), cal_pen])
    coef = np.linalg.solve(x.T @ x + len(z) * np.diag(pen), x.T @ z)
    resid = z - x @ coef
    slope = coef[1] + coef[2: 2 + len(knots)].sum()
    trend = coef[0] + coef[1] * t_origin + coef[2: 2 + len(knots)] @ np.maximum(t_origin - knots, 0)
    recent = resid[-config.smearing_window_days:]
    return HarmonicFit(
        config=config, coef=coef, knots=knots, annual=annual, promotions=promotions,
        first_train_day=days[0], last_train_day=days[-1],
        level_correction=float(resid[-config.level_window_days:].mean()),
        smearing=float(np.mean(np.exp(recent - recent.mean()))),
        slope_per_year=float(slope), trend_at_origin=float(trend))


def predict_harmonic(fit: HarmonicFit, origin: pd.Timestamp, horizon_days: int,
                     promo: pd.Series) -> np.ndarray:
    """Damped-trend extrapolation plus calendar effects, back-transformed to dollars."""
    days = pd.date_range(origin, periods=horizon_days, freq="D")
    phi = fit.config.damping
    # Day h after the origin moves the trend by slope * (phi + phi^2 + ... + phi^h) days.
    h = np.arange(horizon_days, dtype=float)
    damped_days = phi * (1 - phi ** h) / (1 - phi) if phi < 1 else h
    trend = fit.trend_at_origin + fit.slope_per_year * damped_days / 365.0
    cal, _ = _calendar(days, promo, fit.config, fit.annual, fit.promotions)
    log_level = trend + cal @ fit.coef[2 + len(fit.knots):] + fit.level_correction
    return np.maximum(np.exp(log_level) * fit.smearing - 1.0, 0.0)


# ---------------------------------------------------------------- dispatcher
def model_config(name: str, base: HarmonicConfig | None = None) -> HarmonicConfig:
    base = base or HarmonicConfig()
    return replace(base, use_promotions=False) if name == "harmonic_no_promo" else base


def forecast(name: str, history: pd.Series, origin: pd.Timestamp, horizon_days: int,
             promo: pd.Series, config: HarmonicConfig | None = None) -> np.ndarray:
    """Daily forecast for ``[origin, origin + horizon_days)`` from history before ``origin``."""
    origin = pd.Timestamp(origin)
    check_history(history, origin)
    if name == "naive_4wk":
        return naive_4wk(history, horizon_days)
    if name == "seasonal_naive_yoy":
        return seasonal_naive_yoy(history, horizon_days)
    if name in ("harmonic", "harmonic_no_promo"):
        fit = fit_harmonic(history, promo, model_config(name, config))
        return predict_harmonic(fit, origin, horizon_days, promo)
    raise KeyError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
