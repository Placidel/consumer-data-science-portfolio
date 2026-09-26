"""Section 05 - predictive analytics: weekly revenue forecasting with rolling-origin backtests."""

from northstar.forecasting.models import (
    BASELINES,
    CHAMPION,
    MODEL_NAMES,
    HarmonicConfig,
    HistoryError,
    forecast,
)
from northstar.forecasting.series import (
    build_series,
    daily_revenue,
    history_before,
    promo_calendar,
    weekly_revenue,
)

__all__ = [
    "BASELINES",
    "CHAMPION",
    "MODEL_NAMES",
    "HarmonicConfig",
    "HistoryError",
    "build_series",
    "daily_revenue",
    "forecast",
    "history_before",
    "promo_calendar",
    "weekly_revenue",
]
