"""Section 05 forecasters: the no-future-data contract, baselines by hand and the harmonic
regression's recovery of known structure."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from northstar.forecasting import models as fm
from northstar.timeline import DATA_START

NO_PROMOS = pd.Series(False, index=pd.date_range(DATA_START, "2027-01-01"))


def _series(values, start=DATA_START):
    return pd.Series(np.asarray(values, float),
                     index=pd.date_range(start, periods=len(values), freq="D"))


def _simulated(days: int, *, growth=0.3, saturday=0.15, promo_effect=0.3, noise=0.05, seed=0):
    """log revenue = log 1000 + growth * years + Saturday effect + promotion effect + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(DATA_START, periods=days, freq="D")
    calendar = pd.date_range(DATA_START, "2027-01-01")
    promo = pd.Series((calendar.dayofyear % 60) < 7, index=calendar)
    t = (idx - DATA_START).days.to_numpy() / 365.0
    log_y = (np.log(1000) + growth * t + saturday * (idx.dayofweek == 5)
             + promo_effect * promo.reindex(idx).to_numpy() + rng.normal(0, noise, days))
    return pd.Series(np.expm1(log_y), index=idx), promo


# ---------------------------------------------------------------- the history contract
@pytest.mark.parametrize("name", fm.MODEL_NAMES)
def test_every_model_refuses_history_that_reaches_the_origin(name):
    y = _series(np.full(500, 100.0))
    origin = y.index[-1]  # the last history day *is* the origin
    with pytest.raises(fm.HistoryError, match="on or after"):
        fm.forecast(name, y, origin, 14, NO_PROMOS)


def test_history_must_end_the_day_before_the_origin_and_be_contiguous():
    y = _series(np.full(100, 1.0))
    with pytest.raises(fm.HistoryError, match="not the day before"):
        fm.check_history(y, y.index[-1] + pd.Timedelta(days=3))
    with pytest.raises(fm.HistoryError, match="contiguous"):
        fm.check_history(y.drop(y.index[50]), y.index[-1] + pd.Timedelta(days=1))
    with pytest.raises(fm.HistoryError, match="empty"):
        fm.check_history(y.iloc[:0], y.index[0])
    fm.check_history(y, y.index[-1] + pd.Timedelta(days=1))


def test_unknown_model_is_rejected():
    y = _series(np.full(100, 1.0))
    with pytest.raises(KeyError, match="unknown model"):
        fm.forecast("prophet", y, y.index[-1] + pd.Timedelta(days=1), 7, NO_PROMOS)


# ---------------------------------------------------------------- baselines
def test_naive_run_rate_is_the_mean_of_the_last_28_days():
    y = _series(np.r_[np.full(100, 5.0), np.arange(28.0)])
    fc = fm.forecast("naive_4wk", y, y.index[-1] + pd.Timedelta(days=1), 10, NO_PROMOS)
    assert fc.tolist() == [13.5] * 10


def test_seasonal_naive_matches_a_hand_calculation():
    n = 400
    y = _series(np.arange(1.0, n + 1))  # y[i] = i + 1
    origin = y.index[-1] + pd.Timedelta(days=1)
    fc = fm.forecast("seasonal_naive_yoy", y, origin, 14, NO_PROMOS)
    growth = np.mean(np.arange(n - 27, n + 1)) / np.mean(np.arange(n - 27 - 364, n + 1 - 364))
    expected = (np.arange(n - 364, n - 364 + 14) + 1) * growth
    assert fc == pytest.approx(expected)
    # Each forecast day references the day 364 days earlier, which is before the origin.
    assert (origin + pd.Timedelta(days=13)) - pd.Timedelta(days=364) < origin


def test_seasonal_naive_is_undefined_without_a_year_and_four_weeks_of_history():
    y = _series(np.full(364 + 27, 1.0))
    fc = fm.forecast("seasonal_naive_yoy", y, y.index[-1] + pd.Timedelta(days=1), 7, NO_PROMOS)
    assert np.isnan(fc).all()
    with pytest.raises(ValueError, match="one season"):
        fm.seasonal_naive_yoy(_series(np.ones(800)), 365)


# ---------------------------------------------------------------- harmonic regression
def test_harmonic_regression_recovers_weekday_promotion_and_growth_effects():
    y, promo = _simulated(500)
    fit = fm.fit_harmonic(y, promo, fm.HarmonicConfig())
    nk = len(fit.knots)
    dow = fit.coef[2 + nk: 2 + nk + 6]  # Tue..Sun relative to Monday
    assert dow[4] == pytest.approx(0.15, abs=0.03)  # Saturday
    assert np.abs(np.delete(dow, 4)).max() < 0.03
    assert fit.coef[2 + nk + 6] == pytest.approx(0.3, abs=0.03)  # promotion day
    assert fit.slope_per_year == pytest.approx(0.3, abs=0.12)
    assert fit.annual and fit.last_train_day == y.index[-1]
    assert fit.first_train_day == DATA_START + pd.Timedelta(days=28)


def test_undamped_forecast_of_a_clean_series_extrapolates_the_trend():
    y, promo = _simulated(500, noise=0.0)
    origin = y.index[-1] + pd.Timedelta(days=1)
    config = replace(fm.HarmonicConfig(), damping=1.0)
    fc = fm.forecast("harmonic", y, origin, 91, promo, config)
    truth, _ = _simulated(591, noise=0.0)
    assert fc == pytest.approx(truth.iloc[500:].to_numpy(), rel=0.02)


def test_damping_and_the_promotion_switch_behave_as_documented():
    y, promo = _simulated(500)
    origin = y.index[-1] + pd.Timedelta(days=1)
    base = fm.HarmonicConfig()
    damped = fm.forecast("harmonic", y, origin, 91, promo, base)
    undamped = fm.forecast("harmonic", y, origin, 91, promo, replace(base, damping=1.0))
    ratio = damped / undamped
    assert ratio[0] == pytest.approx(1.0, rel=1e-9)  # no extrapolation on the origin day
    assert np.all(np.diff(ratio) < 0)  # growth tapers off with lead time
    no_promo = fm.forecast("harmonic_no_promo", y, origin, 91, promo, base)
    promo_days = promo.reindex(pd.date_range(origin, periods=91)).to_numpy()
    assert promo_days.any()
    assert np.mean(damped[promo_days] / no_promo[promo_days]) > 1.2
    assert fm.model_config("harmonic_no_promo").use_promotions is False
    assert fm.model_config("harmonic").use_promotions is True


def test_annual_terms_need_a_full_year_and_forecasts_are_non_negative():
    short, promo = _simulated(300)
    assert not fm.fit_harmonic(short, promo).annual
    long, _ = _simulated(28 + 365)
    assert fm.fit_harmonic(long, promo).annual
    no_promo_fit = fm.fit_harmonic(long, NO_PROMOS)
    assert not no_promo_fit.promotions  # nothing to learn from: the flag is dropped
    collapsing = _series(np.r_[np.full(300, 1000.0), np.geomspace(1000, 0.01, 100)])
    fc = fm.forecast("harmonic", collapsing, collapsing.index[-1] + pd.Timedelta(days=1), 91,
                     NO_PROMOS)
    assert np.all(fc >= 0) and np.all(np.isfinite(fc))


def test_the_promotion_calendar_must_cover_the_forecast_period():
    y, promo = _simulated(400)
    short_calendar = promo.loc[: y.index[-1] + pd.Timedelta(days=10)]
    with pytest.raises(ValueError, match="does not cover"):
        fm.forecast("harmonic", y, y.index[-1] + pd.Timedelta(days=1), 30, short_calendar)
